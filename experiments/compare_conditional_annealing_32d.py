#!/usr/bin/env python3
"""Paired frozen-proposal test of local conditional annealing in 32D.

The condition list is taken from the final radius in an annealed continuation
``evaluation.csv``.  The source and annealed proposals are frozen and evaluated
with the same conditions, replicate indices, sample budgets, and random seeds.
No training or interval tuning is performed.
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
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_geometry_ladder_coverage_32d as frozen  # noqa: E402


METHODS = ("source", "annealed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--annealed-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--continuation-evaluation", type=Path, required=True,
        help="evaluation.csv produced by the annealed continuation run",
    )
    parser.add_argument(
        "--radius", type=float,
        help="Radius to extract; defaults to the largest radius in the file.",
    )
    parser.add_argument(
        "--include-anchor", action="store_true",
        help="Include the anchor in addition to held-out local conditions.",
    )
    parser.add_argument(
        "--conditions", type=int, default=8,
        help="Maximum number of non-anchor conditions to compare.",
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--evals", type=int, default=250_000)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=[50_000, 100_000, 250_000]
    )
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument(
        "--proposal", choices=("pure", "mixture"), default="pure"
    )
    parser.add_argument("--uniform-fraction", type=float, default=0.01)
    parser.add_argument(
        "--coverage", type=float, nargs="+", default=(0.68, 0.95)
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--restart", action="store_true",
        help="Discard partial per-condition results and rerun everything.",
    )
    args = parser.parse_args()
    if args.conditions <= 0 or args.replicates <= 1 or args.evals <= 1:
        parser.error("conditions must be positive, replicates/evals must exceed 1")
    if any(value <= 1 or value > args.evals for value in args.milestones):
        parser.error("milestones must lie in [2, --evals]")
    if not all(0.0 < value < 1.0 for value in args.coverage):
        parser.error("coverage levels must lie in (0,1)")
    if args.proposal == "mixture" and not 0.0 < args.uniform_fraction < 1.0:
        parser.error("--uniform-fraction must lie in (0,1)")
    return args


def load_conditions(args: argparse.Namespace) -> tuple[np.ndarray, float]:
    frame = pd.read_csv(args.continuation_evaluation)
    required = {"radius", "is_anchor", "c0", "c1", "c2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{args.continuation_evaluation} lacks columns {sorted(missing)}"
        )
    radius = float(frame["radius"].max() if args.radius is None else args.radius)
    selected = frame[np.isclose(frame["radius"], radius)].copy()
    if not args.include_anchor:
        selected = selected[~selected["is_anchor"].astype(bool)]
    else:
        anchors = selected[selected["is_anchor"].astype(bool)]
        local_rows = selected[~selected["is_anchor"].astype(bool)].head(
            args.conditions
        )
        selected = pd.concat([anchors.head(1), local_rows], ignore_index=True)
    if not args.include_anchor:
        selected = selected.head(args.conditions)
    if selected.empty:
        raise ValueError(f"no evaluation conditions found at radius {radius}")
    values = selected[["c0", "c1", "c2"]].to_numpy(np.float64)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("all conditions must lie in [0,1]^3")
    return values, radius


def final_rows(detail: pd.DataFrame, final_samples: int) -> pd.DataFrame:
    return detail[detail["samples"] == final_samples].copy()


def summarize_conditions(final: pd.DataFrame) -> pd.DataFrame:
    result = final.groupby(
        ["method", "condition_index"], as_index=False
    ).agg(
        runs=("replicate", "size"),
        mean_estimate=("first_moment", "mean"),
        empirical_sd=("first_moment", "std"),
        mean_absolute_error=("first_moment", lambda x: np.mean(np.abs(x - 1.0))),
        maximum_absolute_error=("first_moment", lambda x: np.max(np.abs(x - 1.0))),
        median_m2=("second_moment", "median"),
        q90_m2=("second_moment", lambda x: x.quantile(0.90)),
        maximum_m2=("second_moment", "max"),
        median_normalized_ess=("normalized_ess", "median"),
        minimum_normalized_ess=("normalized_ess", "min"),
        median_max_weight_share=("max_weight_share", "median"),
        maximum_weight_share=("max_weight_share", "max"),
        median_hill_xi_005=("hill_xi_005", "median"),
    )
    conditions = final[["condition_index", "c0", "c1", "c2"]].drop_duplicates()
    return result.merge(conditions, on="condition_index", how="left")


def summarize_methods(final: pd.DataFrame) -> pd.DataFrame:
    return final.groupby("method", as_index=False).agg(
        runs=("replicate", "size"),
        mean_estimate=("first_moment", "mean"),
        mean_absolute_error=("first_moment", lambda x: np.mean(np.abs(x - 1.0))),
        maximum_absolute_error=("first_moment", lambda x: np.max(np.abs(x - 1.0))),
        median_m2=("second_moment", "median"),
        q90_m2=("second_moment", lambda x: x.quantile(0.90)),
        maximum_m2=("second_moment", "max"),
        median_normalized_ess=("normalized_ess", "median"),
        minimum_normalized_ess=("normalized_ess", "min"),
        median_max_weight_share=("max_weight_share", "median"),
        maximum_weight_share=("max_weight_share", "max"),
        median_hill_xi_005=("hill_xi_005", "median"),
    )


def paired_condition_summary(by_condition: pd.DataFrame) -> pd.DataFrame:
    source = by_condition[by_condition["method"] == "source"].drop(
        columns="method"
    )
    annealed = by_condition[by_condition["method"] == "annealed"].drop(
        columns="method"
    )
    merged = source.merge(
        annealed, on=["condition_index", "c0", "c1", "c2"],
        suffixes=("_source", "_annealed"),
    )
    merged["median_m2_reduction"] = (
        merged["median_m2_source"] / merged["median_m2_annealed"]
    )
    merged["median_ess_improvement"] = (
        merged["median_normalized_ess_annealed"]
        / merged["median_normalized_ess_source"]
    )
    merged["hill_xi_reduction"] = (
        merged["median_hill_xi_005_source"]
        - merged["median_hill_xi_005_annealed"]
    )
    merged["absolute_bias_source"] = np.abs(merged["mean_estimate_source"] - 1.0)
    merged["absolute_bias_annealed"] = np.abs(
        merged["mean_estimate_annealed"] - 1.0
    )
    return merged


def coverage_summary(final: pd.DataFrame, levels) -> pd.DataFrame:
    rows = []
    for method, group in final.groupby("method", sort=True):
        estimate = group["first_moment"].to_numpy(float)
        m2 = group["second_moment"].to_numpy(float)
        n = group["samples"].to_numpy(float)
        variance_mean = np.maximum(m2 - estimate * estimate, 0.0) / np.maximum(
            n - 1.0, 1.0
        )
        for level in levels:
            alpha = 1.0 - level
            radii = {
                "normal_sample_variance": (
                    norm.ppf(1.0 - alpha / 2.0) * np.sqrt(variance_mean)
                ),
                "empirical_m2_chebyshev": np.sqrt(m2 / (n * alpha)),
            }
            for interval_method, radius in radii.items():
                covered = np.abs(estimate - 1.0) <= radius
                rows.append({
                    "proposal": method,
                    "interval_method": interval_method,
                    "nominal_coverage": level,
                    "runs": len(group),
                    "covered_runs": int(covered.sum()),
                    "empirical_coverage": float(covered.mean()),
                    "mean_halfwidth": float(radius.mean()),
                    "maximum_halfwidth": float(radius.max()),
                    "maximum_absolute_error": float(
                        np.max(np.abs(estimate - 1.0))
                    ),
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    conditions, radius = load_conditions(args)
    pd.DataFrame(conditions, columns=["c0", "c1", "c2"]).rename_axis(
        "condition_index"
    ).reset_index().to_csv(args.output / "conditions.csv", index=False)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loaded = {}
    for method, checkpoint_path in (
        ("source", args.source_checkpoint),
        ("annealed", args.annealed_checkpoint),
    ):
        loaded[method] = frozen.load_model(checkpoint_path, device)
    if loaded["source"][3] != loaded["annealed"][3]:
        raise ValueError("source and annealed checkpoints use different benchmarks")

    total = len(METHODS) * len(conditions) * args.replicates
    done = 0
    all_rows = []
    print(
        "PAIRED FROZEN CONDITIONAL COMPARISON\n"
        f"radius={radius:.6g} conditions={len(conditions)} "
        f"replicates={args.replicates} evals/run={args.evals:,}\n"
        f"proposal={args.proposal}",
        flush=True,
    )
    for method in METHODS:
        adaptive, model, _, benchmark, checkpoint_fraction = loaded[method]
        mixture_fraction = (
            args.uniform_fraction
            if args.proposal == "mixture" else checkpoint_fraction
        )
        for condition_index, condition in enumerate(conditions):
            condition_dir = args.output / method / f"c{condition_index}"
            condition_dir.mkdir(parents=True, exist_ok=True)
            moment_path = condition_dir / "moment_convergence.csv"
            if moment_path.exists() and not args.restart:
                condition_frame = pd.read_csv(moment_path)
                condition_rows = condition_frame.to_dict("records")
                completed = set(condition_frame.loc[
                    condition_frame["samples"] == args.evals, "replicate"
                ].astype(int))
            else:
                condition_rows = []
                completed = set()
            for row in condition_rows:
                row["method"] = method
            all_rows.extend(condition_rows)
            done += len(completed)

            for replicate in range(args.replicates):
                if replicate in completed:
                    continue
                rows = frozen.run_replicate(
                    adaptive, model, benchmark, condition, condition_index,
                    replicate, mixture_fraction, args, device,
                )
                condition_rows.extend(rows)
                method_rows = [{**row, "method": method} for row in rows]
                all_rows.extend(method_rows)
                pd.DataFrame(condition_rows).to_csv(moment_path, index=False)
                done += 1
                final = rows[-1]
                print(
                    f"{done:4d}/{total} {method:8s} c={condition_index:02d} "
                    f"r={replicate:02d} I={final['first_moment']:.7f} "
                    f"M2={final['second_moment']:.5g} "
                    f"ESS={final['normalized_ess']:.4f} "
                    f"xi={final['hill_xi_005']:.3f}",
                    flush=True,
                )

    detail = pd.DataFrame(all_rows)
    detail.to_csv(args.output / "replicate_results.csv", index=False)
    final = final_rows(detail, args.evals)
    by_condition = summarize_conditions(final)
    methods = summarize_methods(final)
    paired = paired_condition_summary(by_condition)
    coverage = coverage_summary(final, args.coverage)
    by_condition.to_csv(args.output / "condition_summary.csv", index=False)
    methods.to_csv(args.output / "method_summary.csv", index=False)
    paired.to_csv(args.output / "paired_condition_comparison.csv", index=False)
    coverage.to_csv(args.output / "coverage_summary.csv", index=False)

    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "source_checkpoint": str(args.source_checkpoint),
            "annealed_checkpoint": str(args.annealed_checkpoint),
            "continuation_evaluation": str(args.continuation_evaluation),
            "radius": radius,
            "conditions": len(conditions),
            "include_anchor": args.include_anchor,
            "replicates": args.replicates,
            "evaluations_per_run": args.evals,
            "milestones": sorted(set(args.milestones + [args.evals])),
            "proposal": args.proposal,
            "uniform_fraction": args.uniform_fraction,
            "training_performed": False,
            "true_integral_used_for_evaluation_only": 1.0,
            "paired_conditions_and_seeds": True,
        }, stream, indent=2)

    print("\nMETHOD SUMMARY")
    print(methods.to_string(index=False))
    print("\nPAIRED CONDITION IMPROVEMENT")
    columns = [
        "condition_index", "median_m2_reduction", "median_ess_improvement",
        "hill_xi_reduction", "absolute_bias_source", "absolute_bias_annealed",
    ]
    print(paired[columns].to_string(index=False))
    print("\nINTERVAL COVERAGE")
    print(coverage.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
