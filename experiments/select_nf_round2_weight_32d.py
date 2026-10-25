#!/usr/bin/env python3
"""Select the round-two boosting step on independent training-condition runs."""

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
from collect_nf_tail_training_32d import load_conditions  # noqa: E402
from diagnose_nf_ensemble_32d import load_models, sample_logweights  # noqa: E402
from diagnose_nf_weight_tails_32d import (  # noqa: E402
    DIM,
    TAIL_FRACTIONS,
    hill_estimate,
    moment_row,
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
    parser.add_argument("--tail1-checkpoint", type=Path, required=True)
    parser.add_argument("--tail2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--conditions-file", type=Path,
        default=Path("results/nf_vs_vegas_160/conditions.csv"),
    )
    parser.add_argument("--condition-indices", type=int, nargs="+", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--evals", type=int, default=250_000)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/nf_tail_round2_weight_selection"),
    )
    return parser.parse_args()


def weights_for_alpha(alpha: float) -> list[float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("every alpha must lie in (0,1)")
    return [0.7 * (1.0 - alpha), 0.3 * (1.0 - alpha), alpha]


def representative_hill(logw: np.ndarray) -> float:
    estimates = {
        fraction: hill_estimate(logw, fraction)[2]
        for fraction in TAIL_FRACTIONS
    }
    return float(np.median([
        estimates[0.002], estimates[0.005], estimates[0.01]
    ]))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    models, _, _ = load_models(
        args.checkpoint,
        [args.tail1_checkpoint, args.tail2_checkpoint],
        device,
    )
    adaptive.DIM = DIM
    conditions = load_conditions(args.conditions_file, args.condition_indices)

    rows = []
    total = len(args.alphas) * len(conditions) * args.replicates
    completed = 0
    for alpha in args.alphas:
        weights = weights_for_alpha(alpha)
        for condition_index, condition in conditions.items():
            for replicate in range(args.replicates):
                # Common seed schedule across alpha candidates reduces comparison noise.
                seed = (
                    args.seed
                    + 1_000_003 * condition_index
                    + 10_007 * replicate
                )
                logw = sample_logweights(
                    models, weights, condition, args.evals, args.gpu_chunk,
                    seed, device,
                )
                moment = moment_row(logw, args.evals, replicate)
                rows.append({
                    "alpha": alpha,
                    "base_weight": weights[0],
                    "tail1_weight": weights[1],
                    "tail2_weight": weights[2],
                    "condition_index": condition_index,
                    "replicate": replicate,
                    "seed": seed,
                    "estimate": moment["first_moment"],
                    "second_moment": moment["second_moment"],
                    "normalized_ess": moment["normalized_ess"],
                    "max_weight_share": moment["max_weight_share"],
                    "representative_hill_xi": representative_hill(logw),
                })
                completed += 1
                print(
                    f"{completed:3d}/{total} alpha={alpha:.3f} "
                    f"c={condition_index:3d} r={replicate:2d} "
                    f"I={moment['first_moment']:.6f} "
                    f"M2={moment['second_moment']:.4g} "
                    f"ESS={moment['normalized_ess']:.4g}",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(args.output / "run_results.csv", index=False)
    summaries = []
    for alpha, group in raw.groupby("alpha", sort=True):
        by_condition = group.groupby("condition_index", as_index=False).agg(
            mean_second_moment=("second_moment", "mean"),
            maximum_second_moment=("second_moment", "max"),
            minimum_ess=("normalized_ess", "min"),
            maximum_weight_share=("max_weight_share", "max"),
            mean_hill_xi=("representative_hill_xi", "mean"),
            mean_estimate=("estimate", "mean"),
        )
        summaries.append({
            "alpha": alpha,
            "base_weight": float(group["base_weight"].iloc[0]),
            "tail1_weight": float(group["tail1_weight"].iloc[0]),
            "tail2_weight": float(group["tail2_weight"].iloc[0]),
            "runs": len(group),
            "mean_second_moment": float(group["second_moment"].mean()),
            "median_second_moment": float(group["second_moment"].median()),
            "p90_second_moment": float(group["second_moment"].quantile(0.9)),
            "maximum_second_moment": float(group["second_moment"].max()),
            "worst_condition_mean_second_moment": float(
                by_condition["mean_second_moment"].max()
            ),
            "minimum_ess": float(group["normalized_ess"].min()),
            "maximum_weight_share": float(group["max_weight_share"].max()),
            "mean_hill_xi": float(group["representative_hill_xi"].mean()),
            "maximum_condition_mean_hill_xi": float(
                by_condition["mean_hill_xi"].max()
            ),
            "mean_estimate": float(group["estimate"].mean()),
            "maximum_condition_absolute_bias": float(
                (by_condition["mean_estimate"] - 1.0).abs().max()
            ),
        })
    summary = pd.DataFrame(summaries).sort_values("alpha")
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "tail1_checkpoint": str(args.tail1_checkpoint),
            "tail2_checkpoint": str(args.tail2_checkpoint),
            "condition_indices": args.condition_indices,
            "alphas": args.alphas,
            "replicates": args.replicates,
            "evaluations": args.evals,
            "seed": args.seed,
            "selection_data_role": "training-condition independent weight selection",
        }, stream, indent=2)
    print("\nROUND-TWO WEIGHT SELECTION", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
