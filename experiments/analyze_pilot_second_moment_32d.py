#!/usr/bin/env python3
"""Calibrate 32D error bars from an independent pilot second moment.

For each condition, early replicates form an independent pilot estimate of
E_q[W] and E_q[W^2].  Later replicates test the resulting fixed conditional
standard error.  No true integral is used to construct the interval; the known
unit integral is used only to score coverage afterward.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRUE_INTEGRAL = 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/nf_error_kl_32d/replicate_results.csv"),
    )
    parser.add_argument("--pilot-replicates", type=int, default=25)
    parser.add_argument(
        "--test-replicates",
        type=int,
        default=25,
        help="Fixed number of disjoint test replicates per condition.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Randomizes the disjoint pilot/test split reproducibly.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nf_error_kl_32d/pilot_second_moment"),
    )
    return parser.parse_args()


def main():
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

    rows = []
    condition_rows = []
    for condition_index, group in data.groupby("condition_index", sort=True):
        group = group.sort_values("replicate")
        required_count = args.pilot_replicates + args.test_replicates
        if required_count > len(group):
            raise ValueError(
                f"condition {condition_index} has {len(group)} replicates, but "
                f"{required_count} are requested"
            )
        rng = np.random.default_rng(
            args.split_seed + 1_000_003 * int(condition_index)
        )
        order = rng.permutation(len(group))
        pilot = group.iloc[order[:args.pilot_replicates]].copy()
        test = group.iloc[
            order[args.pilot_replicates:required_count]
        ].copy()
        if len(pilot) < 2 or len(test) < 2:
            raise ValueError(
                f"condition {condition_index} has {len(pilot)} pilot and "
                f"{len(test)} test replicates"
            )
        unique_n = pilot["evaluations"].unique()
        if len(unique_n) != 1:
            raise ValueError("pilot replicates must have equal sample sizes")
        n = int(unique_n[0])

        # The existing estimator obeys SE^2=(m2-m1^2)/N, hence each run's
        # sample second moment is recoverable without retaining every weight.
        run_second_moment = (
            n * np.square(pilot["current_se"].to_numpy())
            + np.square(pilot["estimate"].to_numpy())
        )
        pilot_first = float(pilot["estimate"].mean())
        pilot_second = float(run_second_moment.mean())
        pilot_weight_variance = max(0.0, pilot_second - pilot_first**2)
        pilot_se = math.sqrt(pilot_weight_variance / n)
        pilot_relative_second_moment = pilot_second / pilot_first**2
        pilot_chi2 = max(0.0, pilot_relative_second_moment - 1.0)

        test["pilot_first_moment"] = pilot_first
        test["pilot_second_moment"] = pilot_second
        test["pilot_weight_variance"] = pilot_weight_variance
        test["pilot_chi2"] = pilot_chi2
        test["pilot_se"] = pilot_se
        test["pilot_covered_68"] = np.abs(test["error"]) <= pilot_se
        test["pilot_covered_95"] = np.abs(test["error"]) <= 1.96 * pilot_se
        test["z_pilot"] = test["error"] / pilot_se
        rows.append(test)

        empirical_sd = float(test["estimate"].std(ddof=1))
        condition_rows.append({
            "condition_index": int(condition_index),
            "pilot_replicates": len(pilot),
            "test_replicates": len(test),
            "pilot_first_moment": pilot_first,
            "pilot_second_moment": pilot_second,
            "pilot_chi2": pilot_chi2,
            "pilot_se": pilot_se,
            "test_empirical_sd": empirical_sd,
            "pilot_se_over_test_empirical_sd": pilot_se / empirical_sd,
            "ordinary_coverage_95": float(test["current_covered_95"].mean()),
            "pilot_coverage_95": float(test["pilot_covered_95"].mean()),
        })

    detail = pd.concat(rows, ignore_index=True)
    conditions = pd.DataFrame(condition_rows)
    summary = pd.DataFrame([{
        "conditions": int(detail["condition_index"].nunique()),
        "pilot_replicates_per_condition": args.pilot_replicates,
        "test_replicates_per_condition": args.test_replicates,
        "split_seed": args.split_seed,
        "test_runs": len(detail),
        "ordinary_coverage_68": float(detail["current_covered_68"].mean()),
        "ordinary_coverage_95": float(detail["current_covered_95"].mean()),
        "pilot_second_moment_coverage_68": float(detail["pilot_covered_68"].mean()),
        "pilot_second_moment_coverage_95": float(detail["pilot_covered_95"].mean()),
        "ordinary_mean_se": float(detail["current_se"].mean()),
        "pilot_mean_se": float(detail["pilot_se"].mean()),
        "median_pilot_se_over_empirical_sd": float(
            conditions["pilot_se_over_test_empirical_sd"].median()
        ),
        "z_pilot_mean": float(detail["z_pilot"].mean()),
        "z_pilot_std": float(detail["z_pilot"].std(ddof=1)),
    }])

    detail.to_csv(args.output / "test_replicate_results.csv", index=False)
    conditions.to_csv(args.output / "condition_summary.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(conditions["test_empirical_sd"], conditions["pilot_se"])
    limit = 1.08 * max(
        conditions["test_empirical_sd"].max(), conditions["pilot_se"].max()
    )
    ax.plot([0, limit], [0, limit], "k--", linewidth=1)
    ax.set(
        xlabel="Empirical SD in independent test replicates",
        ylabel="SE from pilot conditional second moment",
    )
    fig.tight_layout()
    fig.savefig(args.output / "pilot_se_vs_empirical_sd.png", dpi=180)
    plt.close(fig)

    print("PILOT CONDITIONAL SECOND-MOMENT CALIBRATION", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
