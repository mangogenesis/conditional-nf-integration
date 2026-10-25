#!/usr/bin/env python3
"""External validation of frozen 32D conditional-moment intervals.

Replicates 0..R-1 estimate the condition-specific first and second moments.
The next test replicates are evaluated using critical multipliers fixed before
this dataset was generated.  Nothing is fitted to the test conditions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/nf_error_external_32d/replicate_results.csv"),
    )
    parser.add_argument("--pilot-replicates", type=int, default=20)
    parser.add_argument("--test-replicates", type=int, default=25)
    parser.add_argument("--q68", type=float, default=0.9229010680751596)
    parser.add_argument("--q95", type=float, default=2.1189533096596533)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nf_error_external_32d/validation"),
    )
    return parser.parse_args()


def pilot_se(pilot: pd.DataFrame) -> tuple[float, float, float]:
    sizes = pilot["evaluations"].unique()
    if len(sizes) != 1:
        raise ValueError("replicates must have one common sample size")
    n = int(sizes[0])
    first = float(pilot["estimate"].mean())
    run_second = (
        n * np.square(pilot["current_se"].to_numpy())
        + np.square(pilot["estimate"].to_numpy())
    )
    second = float(run_second.mean())
    variance = max(0.0, second - first * first)
    return math.sqrt(variance / n), first, second


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    required = {
        "condition_index", "replicate", "evaluations", "estimate",
        "current_se", "error", "current_covered_68", "current_covered_95",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"input lacks columns: {sorted(missing)}")

    test_parts = []
    condition_rows = []
    for condition_index, group in data.groupby("condition_index", sort=True):
        group = group.sort_values("replicate")
        required_count = args.pilot_replicates + args.test_replicates
        if len(group) < required_count:
            raise ValueError(
                f"condition {condition_index} has {len(group)} runs; "
                f"{required_count} required"
            )
        pilot = group.iloc[:args.pilot_replicates]
        test = group.iloc[
            args.pilot_replicates:required_count
        ].copy()
        standard_error, first, second = pilot_se(pilot)
        test["pilot_se"] = standard_error
        test["pilot_first_moment"] = first
        test["pilot_second_moment"] = second
        test["frozen_q68"] = args.q68
        test["frozen_q95"] = args.q95
        test["gaussian_pilot_covered_68"] = (
            np.abs(test["error"]) <= standard_error
        )
        test["gaussian_pilot_covered_95"] = (
            np.abs(test["error"]) <= 1.96 * standard_error
        )
        test["frozen_tail_covered_68"] = (
            np.abs(test["error"]) <= args.q68 * standard_error
        )
        test["frozen_tail_covered_95"] = (
            np.abs(test["error"]) <= args.q95 * standard_error
        )
        test["z_pilot"] = test["error"] / standard_error
        test_parts.append(test)

        condition_rows.append({
            "condition_index": int(condition_index),
            "c0": float(group["c0"].iloc[0]),
            "c1": float(group["c1"].iloc[0]),
            "c2": float(group["c2"].iloc[0]),
            "pilot_first_moment": first,
            "pilot_second_moment": second,
            "pilot_se": standard_error,
            "test_empirical_sd": float(test["estimate"].std(ddof=1)),
            "ordinary_coverage_95": float(test["current_covered_95"].mean()),
            "gaussian_pilot_coverage_95": float(
                test["gaussian_pilot_covered_95"].mean()
            ),
            "frozen_tail_coverage_95": float(
                test["frozen_tail_covered_95"].mean()
            ),
        })

    test = pd.concat(test_parts, ignore_index=True)
    conditions = pd.DataFrame(condition_rows)
    summary = pd.DataFrame([{
        "external_conditions": int(test["condition_index"].nunique()),
        "pilot_replicates_per_condition": args.pilot_replicates,
        "test_replicates_per_condition": args.test_replicates,
        "test_runs": len(test),
        "frozen_q68": args.q68,
        "frozen_q95": args.q95,
        "ordinary_coverage_68": float(test["current_covered_68"].mean()),
        "ordinary_coverage_95": float(test["current_covered_95"].mean()),
        "gaussian_pilot_coverage_68": float(
            test["gaussian_pilot_covered_68"].mean()
        ),
        "gaussian_pilot_coverage_95": float(
            test["gaussian_pilot_covered_95"].mean()
        ),
        "frozen_tail_coverage_68": float(test["frozen_tail_covered_68"].mean()),
        "frozen_tail_coverage_95": float(test["frozen_tail_covered_95"].mean()),
        "ordinary_mean_halfwidth_95": float((1.96 * test["current_se"]).mean()),
        "gaussian_pilot_mean_halfwidth_95": float((1.96 * test["pilot_se"]).mean()),
        "frozen_tail_mean_halfwidth_95": float(
            (args.q95 * test["pilot_se"]).mean()
        ),
        "z_pilot_mean": float(test["z_pilot"].mean()),
        "z_pilot_std": float(test["z_pilot"].std(ddof=1)),
    }])

    test.to_csv(args.output / "external_test_results.csv", index=False)
    conditions.to_csv(args.output / "condition_summary.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "input": str(args.input),
            "pilot_replicates": args.pilot_replicates,
            "test_replicates": args.test_replicates,
            "frozen_q68": args.q68,
            "frozen_q95": args.q95,
            "fitting_performed": False,
        }, stream, indent=2)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(conditions))
    ax.plot(x, conditions["ordinary_coverage_95"], "o-", label="ordinary")
    ax.plot(
        x, conditions["gaussian_pilot_coverage_95"], "s-",
        label="pilot + 1.96",
    )
    ax.plot(
        x, conditions["frozen_tail_coverage_95"], "^-",
        label="pilot + frozen tail",
    )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="External condition index",
        ylabel="95% coverage",
        ylim=(0.65, 1.02),
        xticks=x,
        xticklabels=conditions["condition_index"].astype(str),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "external_coverage_by_condition.png", dpi=180)
    plt.close(fig)

    print("FROZEN EXTERNAL TAIL VALIDATION", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
