#!/usr/bin/env python3
"""Collect high-weight exploratory points for conditional tail-NF training.

Sampling uses the exact full-NF/broadened-copula mixture.  For every retained
point the file stores the condition, log target, component log densities,
mixture log density, and both original-NF and exploratory importance weights.
No model is trained by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_adaptive_integrand_structure_search as adaptive  # noqa: E402
import run_final_checkpoint_integration_comparison as common  # noqa: E402
from diagnose_nf_weight_tails_32d import (  # noqa: E402
    COND_DIM,
    DIM,
    broadened_copula_log_prob_u,
    sample_from_broadened_copula,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "results/adaptive_integrand_all_weak/"
            "sparse_wave_32_final_model.pt"
        ),
    )
    parser.add_argument(
        "--conditions-file", type=Path,
        default=Path("results/nf_vs_vegas_160/conditions.csv"),
    )
    parser.add_argument("--condition-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--evals-per-condition", type=int, default=10_000_000)
    parser.add_argument("--keep-per-condition", type=int, default=20_000)
    parser.add_argument("--broad-mixture-fraction", type=float, default=0.1)
    parser.add_argument("--copula-broad-fraction", type=float, default=0.03)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/nf_tail_training_32d_c0"),
    )
    return parser.parse_args()


def load_conditions(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    if "condition_index" not in frame:
        frame.insert(0, "condition_index", np.arange(len(frame), dtype=int))
    result = {}
    for index in indices:
        selected = frame[frame["condition_index"].astype(int) == index]
        if len(selected) != 1:
            raise ValueError(f"condition index {index} occurs {len(selected)} times")
        result[index] = selected[["c0", "c1", "c2"]].to_numpy(
            dtype=np.float32
        )[0]
    return result


def merge_top(existing: dict | None, incoming: dict, keep: int) -> dict:
    if existing is None:
        combined = incoming
    else:
        combined = {
            key: torch.cat([existing[key], incoming[key]], dim=0)
            for key in incoming
        }
    count = min(keep, len(combined["logw_nf"]))
    indices = torch.topk(combined["logw_nf"], count, sorted=False).indices
    return {key: value[indices] for key, value in combined.items()}


@torch.inference_mode()
def collect_condition(model, condition, index: int, args, device) -> dict:
    seed = args.seed + 1_000_003 * index
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    retained = None
    processed = 0
    while processed < args.evals_per_condition:
        count = min(args.gpu_chunk, args.evals_per_condition - processed)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)

        use_broad = torch.rand(count, device=device) < args.broad_mixture_fraction
        x = adaptive.sample_from_model(model, count, cond)
        broad_count = int(use_broad.sum().item())
        if broad_count:
            x[use_broad] = sample_from_broadened_copula(
                model, broad_count, cond[use_broad],
                args.copula_broad_fraction,
            )

        logq_nf = model.log_prob(x, cond).reshape(-1)
        u, log_marginal = model.x_to_u_and_log_marginal(x, cond)
        logq_broad = log_marginal + broadened_copula_log_prob_u(
            model, u, cond, args.copula_broad_fraction
        )
        logq_explorer = torch.logaddexp(
            logq_nf + math.log1p(-args.broad_mixture_fraction),
            logq_broad + math.log(args.broad_mixture_fraction),
        )
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)

        local_keep = min(args.keep_per_condition, count)
        local_indices = torch.topk(logf - logq_nf, local_keep).indices
        incoming = {
            "x": x[local_indices].detach().cpu(),
            "condition": cond[local_indices].detach().cpu(),
            "condition_index": torch.full(
                (local_keep,), index, dtype=torch.int64
            ),
            "logf": logf[local_indices].detach().cpu(),
            "logq_nf": logq_nf[local_indices].detach().cpu(),
            "logq_broad": logq_broad[local_indices].detach().cpu(),
            "logq_explorer": logq_explorer[local_indices].detach().cpu(),
            "logw_nf": (logf - logq_nf)[local_indices].detach().cpu(),
            "logw_explorer": (
                logf - logq_explorer
            )[local_indices].detach().cpu(),
            "sampled_from_broad": use_broad[local_indices].detach().cpu(),
        }
        retained = merge_top(retained, incoming, args.keep_per_condition)
        processed += count
        if processed % max(1_000_000, args.gpu_chunk) == 0 or processed == args.evals_per_condition:
            print(
                f"condition {index}: {processed:,}/{args.evals_per_condition:,}; "
                f"retained max logW_nf={float(retained['logw_nf'].max()):.4f}; "
                f"cut={float(retained['logw_nf'].min()):.4f}",
                flush=True,
            )
    return retained


def main() -> None:
    args = parse_args()
    if not 0.0 < args.broad_mixture_fraction < 1.0:
        raise ValueError("broad mixture fraction must lie in (0,1)")
    if not 0.0 < args.copula_broad_fraction < 1.0:
        raise ValueError("copula broad fraction must lie in (0,1)")
    if args.keep_per_condition <= 0:
        raise ValueError("keep-per-condition must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    model, _, checkpoint = common.load_nf(args.checkpoint, device)
    adaptive.DIM = DIM
    conditions = load_conditions(args.conditions_file, args.condition_indices)

    pieces = []
    for index, condition in conditions.items():
        pieces.append(collect_condition(model, condition, index, args, device))
    dataset = {
        key: torch.cat([piece[key] for piece in pieces], dim=0)
        for key in pieces[0]
    }
    torch.save(dataset, args.output / "tail_training_points.pt")

    rows = []
    for index in args.condition_indices:
        mask = dataset["condition_index"] == index
        rows.append({
            "condition_index": index,
            "evaluations": args.evals_per_condition,
            "retained": int(mask.sum()),
            "minimum_retained_logw_nf": float(dataset["logw_nf"][mask].min()),
            "maximum_logw_nf": float(dataset["logw_nf"][mask].max()),
            "maximum_logw_explorer": float(
                dataset["logw_explorer"][mask].max()
            ),
            "fraction_retained_from_broad": float(
                dataset["sampled_from_broad"][mask].float().mean()
            ),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "conditions_file": str(args.conditions_file),
            "condition_indices": args.condition_indices,
            "evals_per_condition": args.evals_per_condition,
            "keep_per_condition": args.keep_per_condition,
            "broad_mixture_fraction": args.broad_mixture_fraction,
            "copula_broad_fraction": args.copula_broad_fraction,
            "seed": args.seed,
            "training_performed": False,
            "checkpoint_benchmark": checkpoint.get("benchmark"),
        }, stream, indent=2)
    print("\nTAIL TRAINING DATA SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
