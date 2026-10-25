#!/usr/bin/env python3
"""Test whether observable KL diagnostics predict 32D integration error.

The analysis uses leave-one-condition-out evaluation: no replicate from the
condition being evaluated is used to fit its conditional error model.  This is
a deliberately small, transparent calibration baseline rather than a claim of
a final uncertainty estimator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = [
    "c0",
    "c1",
    "c2",
    "reverse_kl",
    "reverse_kl_se",
    "log_weight_std",
    "normalized_ess",
    "max_weight_share",
    "top10_weight_share",
    "top100_weight_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/nf_error_kl_32d/replicate_results.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/nf_error_kl_32d/analysis")
    )
    parser.add_argument("--ridge", type=float, default=1.0)
    return parser.parse_args()


def design(frame: pd.DataFrame, mean=None, scale=None):
    raw = frame[FEATURES].to_numpy(dtype=np.float64)
    # Positive tail variables are better behaved on a log scale.
    for name in ("reverse_kl_se", "log_weight_std", "max_weight_share",
                 "top10_weight_share", "top100_weight_share"):
        column = FEATURES.index(name)
        raw[:, column] = np.log(np.maximum(raw[:, column], 1e-12))
    # Keep ESS away from singular logit endpoints.
    column = FEATURES.index("normalized_ess")
    clipped = np.clip(raw[:, column], 1e-6, 1.0 - 1e-6)
    raw[:, column] = np.log(clipped / (1.0 - clipped))
    if mean is None:
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0, ddof=1)
        scale[scale == 0] = 1.0
    standardized = (raw - mean) / scale
    return np.column_stack([np.ones(len(frame)), standardized]), mean, scale


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def quantile_higher(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="higher"))


def leave_one_condition_out(frame: pd.DataFrame, ridge: float) -> pd.DataFrame:
    predictions = []
    for condition_index in sorted(frame["condition_index"].unique()):
        train = frame[frame["condition_index"] != condition_index].copy()
        test = frame[frame["condition_index"] == condition_index].copy()
        x_train, mean, scale = design(train)
        x_test, _, _ = design(test, mean, scale)

        # Work in units of the ordinary MC standard error.  The first model
        # predicts systematic displacement; the second predicts local scale.
        z_train = train["error"].to_numpy() / train["current_se"].to_numpy()
        beta_mean = ridge_fit(x_train, z_train, ridge)
        fitted_train = x_train @ beta_mean
        predicted_test = x_test @ beta_mean

        residual_train = z_train - fitted_train
        log_square = np.log(np.square(residual_train) + 1e-4)
        beta_scale = ridge_fit(x_train, log_square, ridge)
        train_scale = np.sqrt(np.exp(x_train @ beta_scale))
        test_scale = np.sqrt(np.exp(x_test @ beta_scale))

        # Calibrate interval multipliers only on other conditions.  This makes
        # the reported coverage genuinely out-of-condition, though marginal
        # rather than a formal condition-wise guarantee.
        scores = np.abs(residual_train) / np.maximum(train_scale, 1e-8)
        q68 = quantile_higher(scores, 0.68)
        q95 = quantile_higher(scores, 0.95)

        result = test.copy()
        result["predicted_z_mean"] = predicted_test
        result["predicted_z_scale"] = test_scale
        result["predicted_error_mean"] = (
            predicted_test * result["current_se"].to_numpy()
        )
        result["conditional_halfwidth_68"] = (
            q68 * test_scale * result["current_se"].to_numpy()
        )
        result["conditional_halfwidth_95"] = (
            q95 * test_scale * result["current_se"].to_numpy()
        )
        centered_error = (
            result["error"].to_numpy()
            - result["predicted_error_mean"].to_numpy()
        )
        result["conditional_covered_68"] = (
            np.abs(centered_error) <= result["conditional_halfwidth_68"]
        )
        result["conditional_covered_95"] = (
            np.abs(centered_error) <= result["conditional_halfwidth_95"]
        )
        predictions.append(result)
    return pd.concat(predictions, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    required = set(FEATURES + [
        "condition_index", "replicate", "estimate", "error", "current_se",
        "current_covered_68", "current_covered_95",
    ])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "input lacks KL diagnostics; rerun the updated calibration script. "
            f"Missing: {sorted(missing)}"
        )
    if frame[FEATURES + ["error", "current_se"]].isna().any().any():
        raise ValueError("input contains missing calibration values")

    predictions = leave_one_condition_out(frame, args.ridge)
    predictions.to_csv(args.output / "heldout_condition_predictions.csv", index=False)

    summary = pd.DataFrame([{
        "conditions": int(frame["condition_index"].nunique()),
        "runs": len(frame),
        "ordinary_coverage_68": float(frame["current_covered_68"].mean()),
        "ordinary_coverage_95": float(frame["current_covered_95"].mean()),
        "conditional_coverage_68": float(predictions["conditional_covered_68"].mean()),
        "conditional_coverage_95": float(predictions["conditional_covered_95"].mean()),
        "ordinary_mean_halfwidth_95": float((1.96 * frame["current_se"]).mean()),
        "conditional_mean_halfwidth_95": float(
            predictions["conditional_halfwidth_95"].mean()
        ),
        "ordinary_rmse": float(np.sqrt(np.mean(np.square(frame["error"])))),
        "conditional_centered_rmse": float(np.sqrt(np.mean(np.square(
            predictions["error"] - predictions["predicted_error_mean"]
        )))),
        "correlation_reverse_kl_abs_error": float(
            frame["reverse_kl"].corr(frame["absolute_error"])
        ),
        "correlation_reverse_kl_squared_error": float(
            frame["reverse_kl"].corr(np.square(frame["error"]))
        ),
    }])
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "input": str(args.input),
            "ridge": args.ridge,
            "features": FEATURES,
            "validation": "leave-one-condition-out",
        }, stream, indent=2)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    scatter = ax.scatter(
        frame["reverse_kl"], frame["absolute_error"],
        c=frame["normalized_ess"], cmap="viridis", alpha=0.75,
    )
    ax.set(xlabel="Held-out reverse KL", ylabel="Absolute integration error")
    fig.colorbar(scatter, ax=ax, label="Normalized ESS")
    fig.tight_layout()
    fig.savefig(args.output / "kl_vs_absolute_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    centered = predictions["error"] - predictions["predicted_error_mean"]
    scaled = centered / predictions["conditional_halfwidth_95"]
    ax.hist(scaled, bins=np.linspace(-2.5, 2.5, 41))
    ax.axvline(-1, color="black", linestyle="--", linewidth=1)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Centered error / predicted 95% half-width", ylabel="Runs")
    fig.tight_layout()
    fig.savefig(args.output / "conditional_interval_calibration.png", dpi=180)
    plt.close(fig)

    print("CONDITIONAL ERROR CALIBRATION", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
