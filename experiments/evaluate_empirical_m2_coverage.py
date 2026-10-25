#!/usr/bin/env python3
"""Evaluate external coverage of importance-sampling error intervals.

This script performs no training and does not tune an interval multiplier.  It
is intended for a final, untouched validation set.  Each input directory must
contain ``moment_convergence.csv`` as written by the NF tail diagnostics.

The empirical-M2 Chebyshev interval is a method to be validated here; replacing
the population second moment by its sample estimate is not, by itself, a
distribution-free coverage theorem.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, required=True,
        help="Root containing c*/moment_convergence.csv files.",
    )
    parser.add_argument("--true-integral", type=float, default=1.0)
    parser.add_argument(
        "--coverage", type=float, nargs="+", default=[0.68, 0.95],
    )
    parser.add_argument(
        "--binomial-confidence", type=float, default=0.95,
        help="Confidence level for exact intervals on measured coverage.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
    )
    parser.add_argument(
        "--tail-calibration-runs", type=int, default=0,
        help=(
            "Reserve this many earliest replicates per condition for an "
            "independent tail screen. Coverage is then measured only on the "
            "remaining replicates. Zero disables the adaptive rule."
        ),
    )
    parser.add_argument("--hill-xi-threshold", type=float, default=0.25)
    parser.add_argument(
        "--max-weight-share-threshold", type=float, default=0.01
    )
    parser.add_argument(
        "--m2-q90-median-threshold", type=float, default=2.0
    )
    parser.add_argument("--m2-drift-threshold", type=float, default=1.5)
    return parser.parse_args()


def exact_binomial_interval(successes: int, trials: int, level: float):
    """Two-sided Clopper-Pearson interval."""
    tail = 0.5 * (1.0 - level)
    lower = 0.0 if successes == 0 else beta.ppf(
        tail, successes, trials - successes + 1
    )
    upper = 1.0 if successes == trials else beta.ppf(
        1.0 - tail, successes + 1, trials - successes
    )
    return float(lower), float(upper)


def one_sided_lower(successes: int, trials: int, level: float) -> float:
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - level, successes, trials - successes + 1))


def load_moment_rows(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("c*/moment_convergence.csv"))
    if not paths:
        paths = sorted(root.rglob("moment_convergence.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No moment_convergence.csv files found below {root}"
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"samples", "first_moment", "second_moment"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}")
        if "condition_index" not in frame:
            name = path.parent.name
            if not name.startswith("c") or not name[1:].isdigit():
                raise ValueError(f"Cannot infer condition index from {path}")
            frame["condition_index"] = int(name[1:])
        frame["source"] = str(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if "replicate" not in result:
        result["replicate"] = result.groupby("condition_index").cumcount()
    return result


def tail_screen(
    rows: pd.DataFrame,
    calibration_runs: int,
    hill_threshold: float,
    share_threshold: float,
    m2_spread_threshold: float,
    m2_drift_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify each condition using runs excluded from coverage testing."""
    diagnostics = []
    test_parts = []
    for condition_index, all_condition in rows.groupby("condition_index"):
        final_n = int(all_condition["samples"].max())
        final = all_condition[all_condition["samples"] == final_n].copy()
        replicates = sorted(final["replicate"].astype(int).unique())
        if len(replicates) <= calibration_runs:
            raise ValueError(
                f"condition {condition_index} has {len(replicates)} runs, "
                f"not enough for {calibration_runs} calibration runs plus test"
            )
        calibration_ids = set(replicates[:calibration_runs])
        calibration = final[final["replicate"].isin(calibration_ids)]
        test_parts.append(final[~final["replicate"].isin(calibration_ids)])

        median_m2 = float(calibration["second_moment"].median())
        q90_m2 = float(calibration["second_moment"].quantile(0.90))
        m2_q90_over_median = q90_m2 / max(median_m2, np.finfo(float).tiny)
        q90_share = float(calibration["max_weight_share"].quantile(0.90))
        if "hill_xi_005" not in calibration:
            raise ValueError(
                "tail screening requires hill_xi_005; regenerate coverage "
                "data with the updated frozen-proposal evaluator"
            )
        median_hill = float(calibration["hill_xi_005"].median())

        calibration_all_n = all_condition[
            all_condition["replicate"].isin(calibration_ids)
        ]
        milestone_m2 = calibration_all_n.groupby("samples")[
            "second_moment"
        ].median()
        m2_drift = float(milestone_m2.max() / milestone_m2.min())
        milestone_share = calibration_all_n.groupby("samples")[
            "max_weight_share"
        ].median()
        if len(milestone_share) >= 2:
            first_n, last_n = milestone_share.index[0], milestone_share.index[-1]
            share_decay_exponent = float(
                math.log(milestone_share.iloc[0] / milestone_share.iloc[-1])
                / math.log(last_n / first_n)
            )
        else:
            share_decay_exponent = float("nan")

        failures = []
        if median_hill >= hill_threshold:
            failures.append("hill_xi")
        if q90_share >= share_threshold:
            failures.append("max_weight_share")
        if m2_q90_over_median >= m2_spread_threshold:
            failures.append("m2_run_spread")
        if m2_drift >= m2_drift_threshold:
            failures.append("m2_sample_size_drift")
        heavy = bool(failures)
        diagnostics.append({
            "condition_index": int(condition_index),
            "calibration_runs": calibration_runs,
            "test_runs": len(replicates) - calibration_runs,
            "final_samples": final_n,
            "median_hill_xi_005": median_hill,
            "q90_max_weight_share": q90_share,
            "median_m2": median_m2,
            "q90_m2": q90_m2,
            "m2_q90_over_median": m2_q90_over_median,
            "m2_milestone_drift": m2_drift,
            "max_share_decay_exponent": share_decay_exponent,
            "heavy_tail": heavy,
            "selected_method": (
                "empirical_m2_chebyshev" if heavy
                else "normal_sample_variance"
            ),
            "failed_checks": ";".join(failures),
        })
    return pd.concat(test_parts, ignore_index=True), pd.DataFrame(diagnostics)


def add_adaptive_intervals(
    intervals: pd.DataFrame, diagnostics: pd.DataFrame
) -> pd.DataFrame:
    selected = diagnostics.set_index("condition_index")["selected_method"]
    pieces = [intervals]
    adaptive = intervals[
        intervals.apply(
            lambda row: row["method"] == selected.loc[row["condition_index"]],
            axis=1,
        )
    ].copy()
    adaptive["selected_method"] = adaptive["method"]
    adaptive["method"] = "adaptive_tail_screened"
    pieces.append(adaptive)
    return pd.concat(pieces, ignore_index=True)


def interval_rows(runs: pd.DataFrame, true_integral: float, coverages):
    output = []
    for row in runs.itertuples(index=False):
        n = int(row.samples)
        estimate = float(row.first_moment)
        m2 = float(row.second_moment)
        # This is the unbiased sample variance of W divided by N, expressed
        # through the raw first and second sample moments.
        variance_mean = max(m2 - estimate * estimate, 0.0) / max(n - 1, 1)
        standard_error = math.sqrt(variance_mean)
        for coverage in coverages:
            alpha = 1.0 - coverage
            z = float(norm.ppf(1.0 - alpha / 2.0))
            methods = {
                "normal_sample_variance": z * standard_error,
                # Conservative form does not subtract an unknown I^2.
                "empirical_m2_chebyshev": math.sqrt(m2 / (n * alpha)),
                # Included as a diagnostic; it has the same empirical-M2
                # caveat and additionally estimates the centered variance.
                "empirical_centered_chebyshev": math.sqrt(
                    max(m2 - estimate * estimate, 0.0) / (n * alpha)
                ),
            }
            for method, radius in methods.items():
                output.append({
                    "condition_index": int(row.condition_index),
                    "replicate": int(row.replicate),
                    "samples": n,
                    "estimate": estimate,
                    "second_moment": m2,
                    "nominal_coverage": coverage,
                    "method": method,
                    "radius": radius,
                    "lower": estimate - radius,
                    "upper": estimate + radius,
                    "covered": abs(estimate - true_integral) <= radius,
                    "absolute_error": abs(estimate - true_integral),
                    "source": row.source,
                })
    return pd.DataFrame(output)


def summarize(intervals: pd.DataFrame, binomial_confidence: float):
    rows = []
    keys = ["method", "nominal_coverage"]
    for (method, nominal), group in intervals.groupby(keys, sort=True):
        successes = int(group["covered"].sum())
        trials = len(group)
        lower, upper = exact_binomial_interval(
            successes, trials, binomial_confidence
        )
        rows.append({
            "method": method,
            "nominal_coverage": nominal,
            "runs": trials,
            "covered_runs": successes,
            "empirical_coverage": successes / trials,
            "exact_two_sided_lower": lower,
            "exact_two_sided_upper": upper,
            "exact_one_sided_lower": one_sided_lower(
                successes, trials, binomial_confidence
            ),
            "mean_halfwidth": float(group["radius"].mean()),
            "median_halfwidth": float(group["radius"].median()),
            "maximum_halfwidth": float(group["radius"].max()),
            "maximum_absolute_error": float(group["absolute_error"].max()),
        })
    return pd.DataFrame(rows)


def conditional_summary(intervals: pd.DataFrame):
    return intervals.groupby(
        ["method", "nominal_coverage", "condition_index"], as_index=False
    ).agg(
        runs=("covered", "size"),
        empirical_coverage=("covered", "mean"),
        mean_halfwidth=("radius", "mean"),
        maximum_absolute_error=("absolute_error", "max"),
        maximum_second_moment=("second_moment", "max"),
    )


def sample_size_summary(rows: pd.DataFrame, coverages) -> pd.DataFrame:
    """Describe M2 stability, weight concentration, and width scaling."""
    output = []
    for (condition_index, samples), group in rows.groupby(
        ["condition_index", "samples"], sort=True
    ):
        estimate = group["first_moment"].to_numpy(float)
        m2 = group["second_moment"].to_numpy(float)
        variance_mean = np.maximum(m2 - estimate * estimate, 0.0) / max(
            int(samples) - 1, 1
        )
        for coverage in coverages:
            alpha = 1.0 - coverage
            z = float(norm.ppf(1.0 - alpha / 2.0))
            radii = {
                "normal_sample_variance": z * np.sqrt(variance_mean),
                "empirical_m2_chebyshev": np.sqrt(
                    m2 / (int(samples) * alpha)
                ),
                "empirical_centered_chebyshev": np.sqrt(
                    np.maximum(m2 - estimate * estimate, 0.0)
                    / (int(samples) * alpha)
                ),
            }
            for method, radius in radii.items():
                median_width = float(np.median(radius))
                output.append({
                    "condition_index": int(condition_index),
                    "samples": int(samples),
                    "runs": len(group),
                    "nominal_coverage": coverage,
                    "method": method,
                    "median_m2": float(np.median(m2)),
                    "q90_m2": float(np.quantile(m2, 0.90)),
                    "maximum_m2": float(np.max(m2)),
                    "m2_q90_over_median": float(
                        np.quantile(m2, 0.90) / max(
                            np.median(m2), np.finfo(float).tiny
                        )
                    ),
                    "median_max_weight_share": float(
                        group["max_weight_share"].median()
                    ),
                    "q90_max_weight_share": float(
                        group["max_weight_share"].quantile(0.90)
                    ),
                    "median_hill_xi_005": (
                        float(group["hill_xi_005"].median())
                        if "hill_xi_005" in group else float("nan")
                    ),
                    "median_halfwidth": median_width,
                    "sqrt_n_scaled_median_halfwidth": (
                        math.sqrt(int(samples)) * median_width
                    ),
                })
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    if not all(0.0 < value < 1.0 for value in args.coverage):
        raise ValueError("Every requested coverage must lie in (0,1)")
    if not 0.0 < args.binomial_confidence < 1.0:
        raise ValueError("binomial confidence must lie in (0,1)")
    if args.tail_calibration_runs < 0:
        raise ValueError("tail calibration runs cannot be negative")
    if any(value <= 0.0 for value in (
        args.hill_xi_threshold,
        args.max_weight_share_threshold,
        args.m2_q90_median_threshold,
        args.m2_drift_threshold,
    )):
        raise ValueError("all tail-screen thresholds must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = load_moment_rows(args.results_root)
    final_n = all_rows.groupby("condition_index")["samples"].transform("max")
    runs = all_rows[all_rows["samples"] == final_n].copy()
    diagnostics = None
    if args.tail_calibration_runs:
        runs, diagnostics = tail_screen(
            all_rows,
            args.tail_calibration_runs,
            args.hill_xi_threshold,
            args.max_weight_share_threshold,
            args.m2_q90_median_threshold,
            args.m2_drift_threshold,
        )
    intervals = interval_rows(runs, args.true_integral, args.coverage)
    if diagnostics is not None:
        intervals = add_adaptive_intervals(intervals, diagnostics)
    summary = summarize(intervals, args.binomial_confidence)
    by_condition = conditional_summary(intervals)
    size_summary = sample_size_summary(all_rows, args.coverage)

    intervals.to_csv(args.output / "intervals.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    by_condition.to_csv(args.output / "coverage_by_condition.csv", index=False)
    size_summary.to_csv(
        args.output / "sample_size_diagnostics.csv", index=False
    )
    if diagnostics is not None:
        diagnostics.to_csv(args.output / "tail_diagnostics.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "results_root": str(args.results_root),
            "true_integral": args.true_integral,
            "coverage": args.coverage,
            "binomial_confidence": args.binomial_confidence,
            "runs": len(runs),
            "conditions": int(runs["condition_index"].nunique()),
            "tail_calibration_runs_per_condition": args.tail_calibration_runs,
            "tail_screen_thresholds": {
                "hill_xi_005": args.hill_xi_threshold,
                "q90_max_weight_share": args.max_weight_share_threshold,
                "m2_q90_over_median": args.m2_q90_median_threshold,
                "m2_milestone_drift": args.m2_drift_threshold,
            },
            "training_performed": False,
            "external_validation_only": True,
        }, stream, indent=2)

    print("\nEXTERNAL INTERVAL COVERAGE")
    print(summary.to_string(index=False))
    if diagnostics is not None:
        print("\nINDEPENDENT TAIL SCREEN")
        print(diagnostics.to_string(index=False))
    print("\nSAMPLE-SIZE DIAGNOSTICS")
    print(size_summary.to_string(index=False))
    print("\nWORST CONDITIONAL COVERAGE")
    worst = by_condition.groupby(
        ["method", "nominal_coverage"], as_index=False
    ).agg(
        conditions=("condition_index", "size"),
        minimum_condition_coverage=("empirical_coverage", "min"),
        maximum_condition_m2=("maximum_second_moment", "max"),
    )
    print(worst.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
