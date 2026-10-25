#!/usr/bin/env python3
"""Collect second-round residual points against a repaired NF mixture.

Current proposal:
    q1 = (1 - tail_fraction) q0 + tail_fraction qT1

Exploration proposal:
    r = (1 - explore_fraction) q1 + explore_fraction q_broad

The retained ranking is log(f / q1), so round two targets deficiencies that
remain after the first learned tail component.
"""

from __future__ import annotations

import argparse
import copy
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
from collect_nf_tail_training_32d import load_conditions, merge_top  # noqa: E402
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
        "--tail-checkpoint", type=Path,
        default=Path(
            "results/nf_tail_copula_32d_multicondition/tail_model.pt"
        ),
    )
    parser.add_argument(
        "--conditions-file", type=Path,
        default=Path("results/nf_vs_vegas_160/conditions.csv"),
    )
    parser.add_argument("--condition-indices", type=int, nargs="+", required=True)
    parser.add_argument("--evals-per-condition", type=int, default=2_000_000)
    parser.add_argument("--keep-per-condition", type=int, default=5000)
    parser.add_argument("--tail-mixture-fraction", type=float, default=0.3)
    parser.add_argument("--explore-fraction", type=float, default=0.1)
    parser.add_argument("--copula-broad-fraction", type=float, default=0.03)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/nf_tail_round2_training_32d"),
    )
    return parser.parse_args()


def load_models(base_path: Path, tail_path: Path, device):
    base, _, base_payload = common.load_nf(base_path, device)
    tail_payload = torch.load(tail_path, map_location=device)
    tail = copy.deepcopy(base)
    tail.load_state_dict(tail_payload["tail_model_state_dict"], strict=True)
    base.eval()
    tail.eval()
    return base, tail, base_payload, tail_payload


@torch.inference_mode()
def collect_condition(base, tail, condition, index: int, args, device) -> dict:
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

        # First draw from q1, then replace an independent subset with q_broad.
        use_tail = torch.rand(count, device=device) < args.tail_mixture_fraction
        x = adaptive.sample_from_model(base, count, cond)
        tail_count = int(use_tail.sum().item())
        if tail_count:
            x[use_tail] = adaptive.sample_from_model(
                tail, tail_count, cond[use_tail]
            )
        use_broad = torch.rand(count, device=device) < args.explore_fraction
        broad_count = int(use_broad.sum().item())
        if broad_count:
            x[use_broad] = sample_from_broadened_copula(
                base, broad_count, cond[use_broad],
                args.copula_broad_fraction,
            )

        logq_base = base.log_prob(x, cond).reshape(-1)
        logq_tail1 = tail.log_prob(x, cond).reshape(-1)
        logq_current = torch.logaddexp(
            logq_base + math.log1p(-args.tail_mixture_fraction),
            logq_tail1 + math.log(args.tail_mixture_fraction),
        )
        u, log_marginal = base.x_to_u_and_log_marginal(x, cond)
        logq_broad = log_marginal + broadened_copula_log_prob_u(
            base, u, cond, args.copula_broad_fraction
        )
        logq_explorer = torch.logaddexp(
            logq_current + math.log1p(-args.explore_fraction),
            logq_broad + math.log(args.explore_fraction),
        )
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)
        logw_current = logf - logq_current

        local_keep = min(args.keep_per_condition, count)
        local_indices = torch.topk(logw_current, local_keep).indices
        incoming = {
            "x": x[local_indices].detach().cpu(),
            "condition": cond[local_indices].detach().cpu(),
            "condition_index": torch.full(
                (local_keep,), index, dtype=torch.int64
            ),
            "logf": logf[local_indices].detach().cpu(),
            "logq_base": logq_base[local_indices].detach().cpu(),
            "logq_tail1": logq_tail1[local_indices].detach().cpu(),
            "logq_current": logq_current[local_indices].detach().cpu(),
            "logq_broad": logq_broad[local_indices].detach().cpu(),
            "logq_explorer": logq_explorer[local_indices].detach().cpu(),
            "logw_current": logw_current[local_indices].detach().cpu(),
            "logw_explorer": (
                logf - logq_explorer
            )[local_indices].detach().cpu(),
            "sampled_from_broad": use_broad[local_indices].detach().cpu(),
            "sampled_from_tail1_before_broad": use_tail[
                local_indices
            ].detach().cpu(),
        }
        # merge_top expects the ranking key logw_nf; temporarily alias it.
        incoming["logw_nf"] = incoming["logw_current"]
        retained = merge_top(retained, incoming, args.keep_per_condition)
        processed += count
        if processed % max(1_000_000, args.gpu_chunk) == 0 or processed == args.evals_per_condition:
            print(
                f"condition {index}: {processed:,}/{args.evals_per_condition:,}; "
                f"max logW_q1={float(retained['logw_current'].max()):.4f}; "
                f"cut={float(retained['logw_current'].min()):.4f}",
                flush=True,
            )
    # The compatibility alias is intentionally retained for the current trainer.
    return retained


def main() -> None:
    args = parse_args()
    for name in ("tail_mixture_fraction", "explore_fraction", "copula_broad_fraction"):
        value = getattr(args, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0,1)")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    base, tail, base_payload, tail_payload = load_models(
        args.checkpoint, args.tail_checkpoint, device
    )
    adaptive.DIM = DIM
    conditions = load_conditions(args.conditions_file, args.condition_indices)

    pieces = [
        collect_condition(base, tail, condition, index, args, device)
        for index, condition in conditions.items()
    ]
    dataset = {
        key: torch.cat([piece[key] for piece in pieces], dim=0)
        for key in pieces[0]
    }
    torch.save(dataset, args.output / "tail_round2_training_points.pt")

    rows = []
    for index in args.condition_indices:
        mask = dataset["condition_index"] == index
        rows.append({
            "condition_index": index,
            "evaluations": args.evals_per_condition,
            "retained": int(mask.sum()),
            "minimum_retained_logw_current": float(
                dataset["logw_current"][mask].min()
            ),
            "maximum_logw_current": float(
                dataset["logw_current"][mask].max()
            ),
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
            "tail_checkpoint": str(args.tail_checkpoint),
            "conditions_file": str(args.conditions_file),
            "condition_indices": args.condition_indices,
            "evals_per_condition": args.evals_per_condition,
            "keep_per_condition": args.keep_per_condition,
            "tail_mixture_fraction": args.tail_mixture_fraction,
            "explore_fraction": args.explore_fraction,
            "copula_broad_fraction": args.copula_broad_fraction,
            "seed": args.seed,
            "current_proposal": "(1-tail_fraction)*base + tail_fraction*tail1",
            "training_performed": False,
            "base_benchmark": base_payload.get("benchmark"),
            "tail1_training_conditions": tail_payload.get("condition_indices"),
        }, stream, indent=2)
    print("\nROUND-TWO TAIL DATA SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
