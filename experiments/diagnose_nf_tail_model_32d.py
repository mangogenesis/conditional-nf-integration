#!/usr/bin/env python3
"""Diagnose an exact frozen-base/tail-NF mixture on the 32D benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    TAIL_FRACTIONS,
    checkpoints,
    hill_estimate,
    load_condition,
    moment_row,
    regime_label,
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
        default=Path("results/nf_tail_copula_32d_c0/tail_model.pt"),
    )
    parser.add_argument(
        "--conditions-file", type=Path,
        default=Path("results/nf_vs_vegas_160/conditions.csv"),
    )
    parser.add_argument("--condition-index", type=int, default=0)
    parser.add_argument("--tail-mixture-fraction", type=float, default=0.1)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--evals", type=int, default=1_000_000)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/nf_tail_model_32d_c0_diagnostic"),
    )
    return parser.parse_args()


def load_models(base_path: Path, tail_path: Path, device):
    base, _, base_checkpoint = common.load_nf(base_path, device)
    tail_payload = torch.load(tail_path, map_location=device)
    tail = copy.deepcopy(base)
    tail.load_state_dict(tail_payload["tail_model_state_dict"], strict=True)
    base.eval()
    tail.eval()
    return base, tail, base_checkpoint, tail_payload


@torch.inference_mode()
def sample_logweights(
    base, tail, condition, evaluations, chunk, seed, device, tail_fraction,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    parts = []
    remaining = evaluations
    while remaining:
        count = min(remaining, chunk)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)
        use_tail = torch.rand(count, device=device) < tail_fraction
        x = adaptive.sample_from_model(base, count, cond)
        tail_count = int(use_tail.sum().item())
        if tail_count:
            x[use_tail] = adaptive.sample_from_model(
                tail, tail_count, cond[use_tail]
            )

        logq_base = base.log_prob(x, cond).reshape(-1)
        logq_tail = tail.log_prob(x, cond).reshape(-1)
        logq = torch.logaddexp(
            logq_base + math.log1p(-tail_fraction),
            logq_tail + math.log(tail_fraction),
        )
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)
        parts.append((logf - logq).double().cpu().numpy())
        remaining -= count
    return np.concatenate(parts).astype(np.float64, copy=False)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.tail_mixture_fraction < 1.0:
        raise ValueError("tail mixture fraction must lie in (0,1)")
    args.output.mkdir(parents=True, exist_ok=True)
    condition = load_condition(args.conditions_file, args.condition_index)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    base, tail, base_payload, tail_payload = load_models(
        args.checkpoint, args.tail_checkpoint, device
    )
    adaptive.DIM = DIM

    tail_rows, moment_rows = [], []
    for replicate in range(args.replicates):
        seed = args.seed + 1_000_003 * args.condition_index + 10_007 * replicate
        logw = sample_logweights(
            base, tail, condition, args.evals, args.gpu_chunk, seed,
            device, args.tail_mixture_fraction,
        )
        for fraction in TAIL_FRACTIONS:
            k, threshold, xi, se = hill_estimate(logw, fraction)
            tail_rows.append({
                "condition_index": args.condition_index,
                "replicate": replicate,
                "seed": seed,
                "evaluations": args.evals,
                "tail_fraction": fraction,
                "tail_count": k,
                "threshold_log_weight": threshold,
                "hill_xi": xi,
                "hill_standard_error": se,
                "suggested_regime": regime_label(xi),
            })
        for count in checkpoints(args.evals):
            row = moment_row(logw, count, replicate)
            row["condition_index"] = args.condition_index
            moment_rows.append(row)
        final = moment_rows[-1]
        print(
            f"replicate {replicate + 1}/{args.replicates}: "
            f"I={final['first_moment']:.7f} "
            f"M2={final['second_moment']:.6g} "
            f"ESS={final['normalized_ess']:.4g} "
            f"max-share={final['max_weight_share']:.4g}",
            flush=True,
        )

    tails = pd.DataFrame(tail_rows)
    moments = pd.DataFrame(moment_rows)
    tails.to_csv(args.output / "hill_estimates.csv", index=False)
    moments.to_csv(args.output / "moment_convergence.csv", index=False)
    grouped = tails.groupby("tail_fraction", as_index=False).agg(
        replicates=("replicate", "count"),
        mean_hill_xi=("hill_xi", "mean"),
        sd_hill_xi=("hill_xi", "std"),
        minimum_hill_xi=("hill_xi", "min"),
        maximum_hill_xi=("hill_xi", "max"),
    )
    representative = float(grouped[
        grouped["tail_fraction"].isin([0.002, 0.005, 0.01])
    ]["mean_hill_xi"].median())
    final = moments[moments["samples"] == args.evals]
    empirical_sd = float(final["first_moment"].std(ddof=1))
    empirical_sem = empirical_sd / math.sqrt(args.replicates)
    summary = pd.DataFrame([{
        "condition_index": args.condition_index,
        "replicates": args.replicates,
        "evaluations_per_replicate": args.evals,
        "tail_mixture_fraction": args.tail_mixture_fraction,
        "mean_estimate": float(final["first_moment"].mean()),
        "empirical_sd": empirical_sd,
        "empirical_sem": empirical_sem,
        "representative_hill_xi": representative,
        "suggested_regime": regime_label(representative),
        "minimum_final_ess": float(final["normalized_ess"].min()),
        "maximum_final_weight_share": float(final["max_weight_share"].max()),
        "minimum_final_second_moment": float(final["second_moment"].min()),
        "maximum_final_second_moment": float(final["second_moment"].max()),
    }])
    grouped.to_csv(args.output / "hill_summary_by_threshold.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "tail_checkpoint": str(args.tail_checkpoint),
            "condition_index": args.condition_index,
            "condition": condition.tolist(),
            "tail_mixture_fraction": args.tail_mixture_fraction,
            "replicates": args.replicates,
            "evaluations": args.evals,
            "tail_training_weight_ess": tail_payload.get("training_weight_ess"),
            "tail_weight_temperature_beta": tail_payload.get(
                "weight_temperature_beta"
            ),
            "base_benchmark": base_payload.get("benchmark"),
        }, stream, indent=2)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for replicate, group in moments.groupby("replicate"):
        ax.plot(
            group["samples"], group["second_moment"], "o-",
            alpha=0.75, label=f"replicate {replicate}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Samples", ylabel="Empirical second moment of W")
    if args.replicates <= 10:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "second_moment_convergence.png", dpi=180)
    plt.close(fig)

    print("\nTAIL-NF MIXTURE DIAGNOSTIC", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nHILL THRESHOLD SENSITIVITY", flush=True)
    print(grouped.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
