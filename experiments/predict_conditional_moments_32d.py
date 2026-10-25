#!/usr/bin/env python3
"""Predict conditional importance-weight moments from the condition alone.

An RBF kernel-ridge model learns log Var_q[W | C=c] from calibration
conditions.  Hyperparameters and tail multipliers are obtained without using
the external conditions.  The frozen model is then evaluated on an external
condition bank.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LENGTH_SCALES = (0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20)
RIDGES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("results/nf_error_kl_32d/replicate_results.csv"),
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("results/nf_error_external_32d/replicate_results.csv"),
    )
    parser.add_argument("--external-pilot-replicates", type=int, default=20)
    parser.add_argument("--external-test-replicates", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nf_error_external_32d/predicted_moments"),
    )
    return parser.parse_args()


def condition_moments(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_index, group in frame.groupby("condition_index", sort=True):
        sizes = group["evaluations"].unique()
        if len(sizes) != 1:
            raise ValueError("each condition must have one sample size")
        n = int(sizes[0])
        first = float(group["estimate"].mean())
        run_second = (
            n * np.square(group["current_se"].to_numpy())
            + np.square(group["estimate"].to_numpy())
        )
        second = float(run_second.mean())
        variance = max(second - first * first, 1e-16)
        rows.append({
            "condition_index": int(condition_index),
            "c0": float(group["c0"].iloc[0]),
            "c1": float(group["c1"].iloc[0]),
            "c2": float(group["c2"].iloc[0]),
            "replicates": len(group),
            "evaluations_per_replicate": n,
            "first_moment": first,
            "second_moment": second,
            "weight_variance": variance,
            "log_weight_variance": math.log(variance),
            "empirical_estimate_sd": float(group["estimate"].std(ddof=1)),
        })
    return pd.DataFrame(rows)


def rbf(x: np.ndarray, y: np.ndarray, length_scale: float) -> np.ndarray:
    squared = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * squared / (length_scale * length_scale))


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    length_scale: float,
    ridge: float,
) -> np.ndarray:
    mean = float(train_y.mean())
    kernel = rbf(train_x, train_x, length_scale)
    alpha = np.linalg.solve(
        kernel + ridge * np.eye(len(train_x)), train_y - mean
    )
    return mean + rbf(test_x, train_x, length_scale) @ alpha


def select_hyperparameters(moments: pd.DataFrame):
    x = moments[["c0", "c1", "c2"]].to_numpy(dtype=np.float64)
    y = moments["log_weight_variance"].to_numpy(dtype=np.float64)
    rows = []
    best = None
    for length_scale in LENGTH_SCALES:
        for ridge in RIDGES:
            predictions = np.empty_like(y)
            for index in range(len(y)):
                keep = np.arange(len(y)) != index
                predictions[index] = fit_predict(
                    x[keep], y[keep], x[index:index + 1], length_scale, ridge
                )[0]
            rmse = float(np.sqrt(np.mean(np.square(predictions - y))))
            mae = float(np.mean(np.abs(predictions - y)))
            row = {
                "length_scale": length_scale,
                "ridge": ridge,
                "loocv_log_variance_rmse": rmse,
                "loocv_log_variance_mae": mae,
            }
            rows.append(row)
            if best is None or (rmse, mae, length_scale, ridge) < (
                best[0], best[1], best[2], best[3]
            ):
                best = (rmse, mae, length_scale, ridge, predictions.copy())
    return best, pd.DataFrame(rows)


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = int(math.ceil((len(ordered) + 1) * coverage))
    rank = min(max(rank, 1), len(ordered))
    return float(ordered[rank - 1])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(args.calibration)
    external = pd.read_csv(args.external)
    required = {
        "condition_index", "c0", "c1", "c2", "replicate", "evaluations",
        "estimate", "current_se", "error", "current_covered_68",
        "current_covered_95",
    }
    for name, frame in (("calibration", calibration), ("external", external)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} input lacks columns: {sorted(missing)}")

    calibration_moments = condition_moments(calibration)
    best, search = select_hyperparameters(calibration_moments)
    _, _, length_scale, ridge, loo_log_variance = best
    calibration_moments["loo_predicted_log_weight_variance"] = loo_log_variance
    calibration_moments["loo_predicted_weight_variance"] = np.exp(
        loo_log_variance
    )

    # Use only cross-condition predictions to calibrate the standardized tails.
    calibration_scored = calibration.merge(
        calibration_moments[[
            "condition_index", "loo_predicted_weight_variance",
        ]], on="condition_index", how="left",
    )
    calibration_scored["predicted_se"] = np.sqrt(
        calibration_scored["loo_predicted_weight_variance"]
        / calibration_scored["evaluations"]
    )
    calibration_scores = (
        np.abs(calibration_scored["error"])
        / calibration_scored["predicted_se"]
    ).to_numpy()
    q68 = conformal_quantile(calibration_scores, 0.68)
    q95 = conformal_quantile(calibration_scores, 0.95)

    # Freeze the moment model, then predict at wholly external conditions.
    train_x = calibration_moments[["c0", "c1", "c2"]].to_numpy()
    train_y = calibration_moments["log_weight_variance"].to_numpy()
    external_all_moments = condition_moments(external)
    external_x = external_all_moments[["c0", "c1", "c2"]].to_numpy()
    predicted_log_variance = fit_predict(
        train_x, train_y, external_x, length_scale, ridge
    )
    predictions = external_all_moments.copy()
    predictions["predicted_log_weight_variance"] = predicted_log_variance
    predictions["predicted_weight_variance"] = np.exp(predicted_log_variance)

    # Preserve the predeclared external test partition used previously.
    test_parts = []
    for condition_index, group in external.groupby("condition_index", sort=True):
        group = group.sort_values("replicate")
        start = args.external_pilot_replicates
        stop = start + args.external_test_replicates
        if len(group) < stop:
            raise ValueError(f"condition {condition_index} has too few replicates")
        test_parts.append(group.iloc[start:stop].copy())
    test = pd.concat(test_parts, ignore_index=True)
    test = test.merge(
        predictions[["condition_index", "predicted_weight_variance"]],
        on="condition_index", how="left",
    )
    test["predicted_se"] = np.sqrt(
        test["predicted_weight_variance"] / test["evaluations"]
    )
    test["predicted_moment_covered_68"] = (
        np.abs(test["error"]) <= q68 * test["predicted_se"]
    )
    test["predicted_moment_covered_95"] = (
        np.abs(test["error"]) <= q95 * test["predicted_se"]
    )
    test["predicted_standardized_error"] = test["error"] / test["predicted_se"]

    condition_results = test.groupby("condition_index", as_index=False).agg(
        predicted_moment_coverage_68=("predicted_moment_covered_68", "mean"),
        predicted_moment_coverage_95=("predicted_moment_covered_95", "mean"),
        external_test_empirical_sd=("estimate", "std"),
    ).merge(predictions, on="condition_index", how="left")

    summary = pd.DataFrame([{
        "calibration_conditions": len(calibration_moments),
        "external_conditions": len(predictions),
        "external_test_runs": len(test),
        "selected_length_scale": length_scale,
        "selected_ridge": ridge,
        "loocv_log_variance_rmse": best[0],
        "frozen_predicted_moment_q68": q68,
        "frozen_predicted_moment_q95": q95,
        "ordinary_external_coverage_68": float(test["current_covered_68"].mean()),
        "ordinary_external_coverage_95": float(test["current_covered_95"].mean()),
        "predicted_moment_external_coverage_68": float(
            test["predicted_moment_covered_68"].mean()
        ),
        "predicted_moment_external_coverage_95": float(
            test["predicted_moment_covered_95"].mean()
        ),
        "ordinary_mean_halfwidth_95": float((1.96 * test["current_se"]).mean()),
        "predicted_moment_mean_halfwidth_95": float(
            (q95 * test["predicted_se"]).mean()
        ),
        "predicted_z_mean": float(test["predicted_standardized_error"].mean()),
        "predicted_z_std": float(test["predicted_standardized_error"].std(ddof=1)),
        "correlation_predicted_vs_external_log_variance": float(
            condition_results["predicted_log_weight_variance"].corr(
                condition_results["log_weight_variance"]
            )
        ),
    }])

    search.to_csv(args.output / "hyperparameter_search.csv", index=False)
    calibration_moments.to_csv(args.output / "calibration_moments.csv", index=False)
    calibration_scored.to_csv(args.output / "calibration_scores.csv", index=False)
    predictions.to_csv(args.output / "external_moment_predictions.csv", index=False)
    test.to_csv(args.output / "external_test_results.csv", index=False)
    condition_results.to_csv(args.output / "condition_results.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "calibration": str(args.calibration),
            "external": str(args.external),
            "features": ["c0", "c1", "c2"],
            "target": "log conditional importance-weight variance",
            "model": "RBF kernel ridge",
            "length_scale": length_scale,
            "ridge": ridge,
            "q68": q68,
            "q95": q95,
            "external_fitting_performed": False,
        }, stream, indent=2)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(
        condition_results["log_weight_variance"],
        condition_results["predicted_log_weight_variance"],
    )
    low = min(
        condition_results["log_weight_variance"].min(),
        condition_results["predicted_log_weight_variance"].min(),
    )
    high = max(
        condition_results["log_weight_variance"].max(),
        condition_results["predicted_log_weight_variance"].max(),
    )
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set(
        xlabel="External empirical log weight variance",
        ylabel="Predicted log weight variance from condition",
    )
    fig.tight_layout()
    fig.savefig(args.output / "predicted_vs_external_log_variance.png", dpi=180)
    plt.close(fig)

    print("CONDITION-ONLY MOMENT PREDICTION", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
