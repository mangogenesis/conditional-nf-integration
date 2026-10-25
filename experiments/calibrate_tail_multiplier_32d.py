#!/usr/bin/env python3
"""Cross-condition tail calibration for the 32D importance estimator.

For every randomized split and condition:
  1. estimate the conditional weight second moment from pilot replicates;
  2. scale independent test errors by that pilot standard error;
  3. estimate critical values using test scores from *other* conditions only;
  4. evaluate coverage on the excluded condition.

The known unit integral is used to calibrate and score the synthetic benchmark.
No NF training or new model evaluations are performed.
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
        default=Path("results/nf_error_kl_32d/replicate_results.csv"),
    )
    parser.add_argument("--pilot-replicates", type=int, default=20)
    parser.add_argument("--test-replicates", type=int, default=25)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nf_error_kl_32d/tail_calibration"),
    )
    return parser.parse_args()


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    """Finite-sample upper quantile with the split-conformal rank correction."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = int(math.ceil((len(ordered) + 1) * coverage))
    rank = min(max(rank, 1), len(ordered))
    return float(ordered[rank - 1])


def split_condition(
    group: pd.DataFrame,
    condition_index: int,
    split: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = args.pilot_replicates + args.test_replicates
    if required > len(group):
        raise ValueError(
            f"condition {condition_index} has {len(group)} runs; {required} requested"
        )
    rng = np.random.default_rng(
        args.seed + 10_000_019 * split + 1_000_003 * condition_index
    )
    order = rng.permutation(len(group))
    pilot = group.iloc[order[:args.pilot_replicates]].copy()
    test = group.iloc[order[args.pilot_replicates:required]].copy()
    return pilot, test


def pilot_standard_error(pilot: pd.DataFrame) -> tuple[float, float, float]:
    sample_sizes = pilot["evaluations"].unique()
    if len(sample_sizes) != 1:
        raise ValueError("replicates within a condition must use equal sample sizes")
    n = int(sample_sizes[0])
    first = float(pilot["estimate"].mean())
    run_second = (
        n * np.square(pilot["current_se"].to_numpy())
        + np.square(pilot["estimate"].to_numpy())
    )
    second = float(run_second.mean())
    variance = max(0.0, second - first * first)
    return math.sqrt(variance / n), first, second


def one_split(data: pd.DataFrame, split: int, args: argparse.Namespace):
    test_parts = []
    pilot_rows = []
    for condition_index, group in data.groupby("condition_index", sort=True):
        condition_index = int(condition_index)
        group = group.sort_values("replicate").reset_index(drop=True)
        pilot, test = split_condition(group, condition_index, split, args)
        pilot_se, first, second = pilot_standard_error(pilot)
        test["split"] = split
        test["pilot_se"] = pilot_se
        test["pilot_first_moment"] = first
        test["pilot_second_moment"] = second
        test["absolute_standardized_error"] = (
            np.abs(test["error"].to_numpy()) / pilot_se
        )
        test_parts.append(test)
        pilot_rows.append({
            "split": split,
            "condition_index": condition_index,
            "pilot_se": pilot_se,
            "pilot_first_moment": first,
            "pilot_second_moment": second,
        })

    tests = pd.concat(test_parts, ignore_index=True)
    evaluated_parts = []
    condition_rows = []
    for condition_index, heldout in tests.groupby("condition_index", sort=True):
        calibration = tests[tests["condition_index"] != condition_index]
        q68 = conformal_quantile(
            calibration["absolute_standardized_error"].to_numpy(), 0.68
        )
        q95 = conformal_quantile(
            calibration["absolute_standardized_error"].to_numpy(), 0.95
        )
        heldout = heldout.copy()
        heldout["heldout_q68"] = q68
        heldout["heldout_q95"] = q95
        heldout["gaussian_pilot_covered_68"] = (
            np.abs(heldout["error"]) <= heldout["pilot_se"]
        )
        heldout["gaussian_pilot_covered_95"] = (
            np.abs(heldout["error"]) <= 1.96 * heldout["pilot_se"]
        )
        heldout["tail_calibrated_covered_68"] = (
            np.abs(heldout["error"]) <= q68 * heldout["pilot_se"]
        )
        heldout["tail_calibrated_covered_95"] = (
            np.abs(heldout["error"]) <= q95 * heldout["pilot_se"]
        )
        evaluated_parts.append(heldout)
        condition_rows.append({
            "split": split,
            "condition_index": int(condition_index),
            "heldout_q68": q68,
            "heldout_q95": q95,
            "ordinary_coverage_95": float(heldout["current_covered_95"].mean()),
            "gaussian_pilot_coverage_95": float(
                heldout["gaussian_pilot_covered_95"].mean()
            ),
            "tail_calibrated_coverage_95": float(
                heldout["tail_calibrated_covered_95"].mean()
            ),
        })
    return (
        pd.concat(evaluated_parts, ignore_index=True),
        pd.DataFrame(condition_rows),
        pd.DataFrame(pilot_rows),
    )


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

    all_tests, all_conditions, all_pilots = [], [], []
    split_rows = []
    for split in range(args.splits):
        tests, conditions, pilots = one_split(data, split, args)
        all_tests.append(tests)
        all_conditions.append(conditions)
        all_pilots.append(pilots)
        split_rows.append({
            "split": split,
            "ordinary_coverage_68": float(tests["current_covered_68"].mean()),
            "ordinary_coverage_95": float(tests["current_covered_95"].mean()),
            "gaussian_pilot_coverage_68": float(
                tests["gaussian_pilot_covered_68"].mean()
            ),
            "gaussian_pilot_coverage_95": float(
                tests["gaussian_pilot_covered_95"].mean()
            ),
            "tail_calibrated_coverage_68": float(
                tests["tail_calibrated_covered_68"].mean()
            ),
            "tail_calibrated_coverage_95": float(
                tests["tail_calibrated_covered_95"].mean()
            ),
            "mean_q68": float(tests["heldout_q68"].mean()),
            "mean_q95": float(tests["heldout_q95"].mean()),
            "ordinary_mean_halfwidth_95": float((1.96 * tests["current_se"]).mean()),
            "gaussian_pilot_mean_halfwidth_95": float((1.96 * tests["pilot_se"]).mean()),
            "tail_calibrated_mean_halfwidth_95": float(
                (tests["heldout_q95"] * tests["pilot_se"]).mean()
            ),
        })

    tests = pd.concat(all_tests, ignore_index=True)
    conditions = pd.concat(all_conditions, ignore_index=True)
    pilots = pd.concat(all_pilots, ignore_index=True)
    splits = pd.DataFrame(split_rows)
    summary = pd.DataFrame([{
        "splits": args.splits,
        "conditions": int(data["condition_index"].nunique()),
        "pilot_replicates": args.pilot_replicates,
        "test_replicates": args.test_replicates,
        "mean_ordinary_coverage_68": float(splits["ordinary_coverage_68"].mean()),
        "mean_ordinary_coverage_95": float(splits["ordinary_coverage_95"].mean()),
        "mean_gaussian_pilot_coverage_68": float(splits["gaussian_pilot_coverage_68"].mean()),
        "mean_gaussian_pilot_coverage_95": float(splits["gaussian_pilot_coverage_95"].mean()),
        "mean_tail_calibrated_coverage_68": float(splits["tail_calibrated_coverage_68"].mean()),
        "sd_tail_calibrated_coverage_68": float(splits["tail_calibrated_coverage_68"].std(ddof=1)),
        "mean_tail_calibrated_coverage_95": float(splits["tail_calibrated_coverage_95"].mean()),
        "sd_tail_calibrated_coverage_95": float(splits["tail_calibrated_coverage_95"].std(ddof=1)),
        "minimum_tail_calibrated_coverage_95": float(splits["tail_calibrated_coverage_95"].min()),
        "maximum_tail_calibrated_coverage_95": float(splits["tail_calibrated_coverage_95"].max()),
        "mean_heldout_q68": float(splits["mean_q68"].mean()),
        "mean_heldout_q95": float(splits["mean_q95"].mean()),
        "ordinary_mean_halfwidth_95": float(splits["ordinary_mean_halfwidth_95"].mean()),
        "gaussian_pilot_mean_halfwidth_95": float(splits["gaussian_pilot_mean_halfwidth_95"].mean()),
        "tail_calibrated_mean_halfwidth_95": float(splits["tail_calibrated_mean_halfwidth_95"].mean()),
    }])

    tests.to_csv(args.output / "heldout_test_results.csv", index=False)
    conditions.to_csv(args.output / "condition_results.csv", index=False)
    pilots.to_csv(args.output / "pilot_moments.csv", index=False)
    splits.to_csv(args.output / "split_summary.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "input": str(args.input),
            "pilot_replicates": args.pilot_replicates,
            "test_replicates": args.test_replicates,
            "splits": args.splits,
            "seed": args.seed,
            "calibration": "leave-one-condition-out split-conformal quantile",
        }, stream, indent=2)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    values = [
        splits["ordinary_coverage_95"],
        splits["gaussian_pilot_coverage_95"],
        splits["tail_calibrated_coverage_95"],
    ]
    ax.boxplot(values, tick_labels=["ordinary", "pilot + 1.96", "pilot + held-out q"])
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set(ylabel="95% empirical coverage", ylim=(0.75, 1.01))
    fig.tight_layout()
    fig.savefig(args.output / "coverage_comparison.png", dpi=180)
    plt.close(fig)

    print("HELD-OUT TAIL CALIBRATION", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
