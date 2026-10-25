#!/usr/bin/env python3
"""Calibrate importance-sampling error estimates for the frozen 32D NF.

This script never trains or modifies the model.  It reuses the condition bank
from the 160-condition experiment, selects conditions across the observed ESS
range, and performs independent integration replicates at every condition.
The known sparse_wave_32 integral (one) is used only for calibration metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_adaptive_integrand_structure_search as adaptive  # noqa: E402
import run_final_checkpoint_integration_comparison as common  # noqa: E402


DIM = 32
COND_DIM = 3
TRUE_INTEGRAL = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/adaptive_integrand_all_weak/"
            "sparse_wave_32_final_model.pt"
        ),
    )
    parser.add_argument(
        "--conditions-file",
        type=Path,
        default=Path("results/nf_vs_vegas_160/conditions.csv"),
    )
    parser.add_argument(
        "--prior-results",
        type=Path,
        default=Path("results/nf_vs_vegas_160/condition_results.csv"),
        help="Used only to select conditions across the previous NF ESS range.",
    )
    parser.add_argument(
        "--condition-indices",
        type=int,
        nargs="*",
        help="Explicit condition indices; overrides ESS-stratified selection.",
    )
    parser.add_argument("--n-conditions", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--evals", type=int, default=100_000)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument(
        "--proposal",
        choices=("pure", "mixture"),
        default="pure",
        help="pure reproduces the published 32D run; mixture uses checkpoint epsilon.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nf_error_calibration_32d"),
    )
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("--replicates must be at least 2")
    if args.batches < 2 or args.evals % args.batches:
        parser.error("--batches must be at least 2 and divide --evals exactly")
    if args.gpu_chunk <= 0:
        parser.error("--gpu-chunk must be positive")
    return args


def read_conditions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"c0", "c1", "c2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"conditions file lacks columns {sorted(missing)}")
    if "condition_index" not in frame:
        frame.insert(0, "condition_index", np.arange(len(frame), dtype=int))
    return frame[["condition_index", "c0", "c1", "c2"]].copy()


def choose_indices(args: argparse.Namespace, conditions: pd.DataFrame) -> list[int]:
    available = set(conditions["condition_index"].astype(int))
    if args.condition_indices:
        selected = list(dict.fromkeys(args.condition_indices))
    else:
        prior = pd.read_csv(args.prior_results)
        if "method" in prior:
            prior = prior[prior["method"] == "nf_pure"]
        prior = prior.dropna(subset=["normalized_ess"]).sort_values(
            "normalized_ess"
        )
        if prior.empty:
            raise ValueError("prior results contain no finite NF ESS values")
        positions = np.linspace(0, len(prior) - 1, args.n_conditions)
        selected = (
            prior.iloc[np.rint(positions).astype(int)]["condition_index"]
            .astype(int)
            .tolist()
        )
        selected = list(dict.fromkeys(selected))
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"unknown condition indices: {unknown}")
    return selected


def stable_weight_diagnostics(logw: np.ndarray) -> dict[str, float]:
    log_total = float(logsumexp(logw))
    order = np.sort(logw)

    def top_share(k: int) -> float:
        k = min(k, len(order))
        return float(math.exp(logsumexp(order[-k:]) - log_total))

    return {
        # Because the benchmark has I(c)=1, -E_q[log w] is exactly
        # KL(q || p).  For an unknown integral it is the observable
        # unnormalised reverse-KL objective, shifted by -log I(c).
        "reverse_kl": float(-np.mean(logw)),
        "reverse_kl_se": float(np.std(logw, ddof=1) / math.sqrt(len(logw))),
        "log_weight_std": float(np.std(logw, ddof=1)),
        "log_weight_q01": float(np.quantile(logw, 0.01)),
        "log_weight_q05": float(np.quantile(logw, 0.05)),
        "log_weight_q95": float(np.quantile(logw, 0.95)),
        "log_weight_q99": float(np.quantile(logw, 0.99)),
        "max_weight_share": float(math.exp(float(order[-1]) - log_total)),
        "top10_weight_share": top_share(10),
        "top100_weight_share": top_share(100),
        "log_weight_min": float(order[0]),
        "log_weight_median": float(np.median(order)),
        "log_weight_max": float(order[-1]),
    }


def batch_statistics(logw: np.ndarray, batches: int) -> tuple[float, float, float]:
    batch_size = len(logw) // batches
    matrix = logw.reshape(batches, batch_size)
    estimates = np.exp(logsumexp(matrix, axis=1) - math.log(batch_size))
    mean = float(estimates.mean())
    stderr = float(estimates.std(ddof=1) / math.sqrt(batches))
    critical = float(student_t.ppf(0.975, df=batches - 1))
    return mean, stderr, critical


@torch.inference_mode()
def run_replicate(
    model,
    uniform_fraction: float,
    condition: np.ndarray,
    condition_index: int,
    replicate: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    run_seed = args.seed + 1_000_003 * condition_index + 10_007 * replicate
    torch.manual_seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
        torch.cuda.synchronize()
    started = time.perf_counter()
    pieces: list[np.ndarray] = []
    remaining = args.evals
    while remaining:
        count = min(remaining, args.gpu_chunk)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)
        if args.proposal == "pure":
            x = adaptive.sample_from_model(model, count, cond)
            logq = model.log_prob(x, cond).reshape(-1)
        else:
            x, logq = adaptive.sample_from_proposal(
                proposal_model=model,
                n=count,
                cond=cond,
                device=device,
                uniform_fraction=uniform_fraction,
            )
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)
        pieces.append((logf - logq).double().cpu().numpy())
        remaining -= count
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    logw = np.concatenate(pieces).astype(np.float64, copy=False)
    estimate, current_se, ess = common.summarize_log_weights([logw])
    batch_estimate, batch_se, batch_critical = batch_statistics(
        logw, args.batches
    )
    error = estimate - TRUE_INTEGRAL
    row = {
        "condition_index": condition_index,
        "c0": float(condition[0]),
        "c1": float(condition[1]),
        "c2": float(condition[2]),
        "replicate": replicate,
        "seed": run_seed,
        "proposal": args.proposal,
        "evaluations": args.evals,
        "batches": args.batches,
        "estimate": estimate,
        "error": error,
        "absolute_error": abs(error),
        "current_se": current_se,
        "batch_estimate": batch_estimate,
        "batch_se": batch_se,
        "normalized_ess": ess,
        "current_covered_68": abs(error) <= current_se,
        "current_covered_95": abs(error) <= 1.96 * current_se,
        "batch_covered_68": abs(batch_estimate - 1.0) <= batch_se,
        "batch_covered_95_normal": abs(batch_estimate - 1.0) <= 1.96 * batch_se,
        "batch_covered_95_t": abs(batch_estimate - 1.0)
        <= batch_critical * batch_se,
        "z_current": error / current_se if current_se > 0 else math.nan,
        "z_batch": (batch_estimate - 1.0) / batch_se
        if batch_se > 0
        else math.nan,
        "seconds": seconds,
    }
    row.update(stable_weight_diagnostics(logw))
    return row


def condition_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_index, group in detail.groupby("condition_index", sort=True):
        empirical_sd = float(group["estimate"].std(ddof=1))
        mean_current_se = float(group["current_se"].mean())
        mean_batch_se = float(group["batch_se"].mean())
        rows.append({
            "condition_index": int(condition_index),
            "c0": float(group["c0"].iloc[0]),
            "c1": float(group["c1"].iloc[0]),
            "c2": float(group["c2"].iloc[0]),
            "replicates": len(group),
            "mean_estimate": float(group["estimate"].mean()),
            "bias": float(group["estimate"].mean() - 1.0),
            "empirical_sd": empirical_sd,
            "rmse": float(np.sqrt(np.mean(np.square(group["error"])))),
            "mean_current_se": mean_current_se,
            "current_se_over_empirical_sd": mean_current_se / empirical_sd,
            "mean_batch_se": mean_batch_se,
            "batch_se_over_empirical_sd": mean_batch_se / empirical_sd,
            "current_coverage_68": float(group["current_covered_68"].mean()),
            "current_coverage_95": float(group["current_covered_95"].mean()),
            "batch_coverage_68": float(group["batch_covered_68"].mean()),
            "batch_coverage_95_normal": float(
                group["batch_covered_95_normal"].mean()
            ),
            "batch_coverage_95_t": float(group["batch_covered_95_t"].mean()),
            "mean_normalized_ess": float(group["normalized_ess"].mean()),
            "mean_max_weight_share": float(group["max_weight_share"].mean()),
            "maximum_weight_share": float(group["max_weight_share"].max()),
        })
    return pd.DataFrame(rows)


def overall_summary(detail: pd.DataFrame, by_condition: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "conditions": int(detail["condition_index"].nunique()),
        "replicates_per_condition": int(detail.groupby("condition_index").size().min()),
        "total_runs": len(detail),
        "mean_estimate": float(detail["estimate"].mean()),
        "pooled_bias": float(detail["estimate"].mean() - 1.0),
        "mean_absolute_error": float(detail["absolute_error"].mean()),
        "current_coverage_68": float(detail["current_covered_68"].mean()),
        "current_coverage_95": float(detail["current_covered_95"].mean()),
        "batch_coverage_68": float(detail["batch_covered_68"].mean()),
        "batch_coverage_95_normal": float(detail["batch_covered_95_normal"].mean()),
        "batch_coverage_95_t": float(detail["batch_covered_95_t"].mean()),
        "median_current_se_over_empirical_sd": float(
            by_condition["current_se_over_empirical_sd"].median()
        ),
        "median_batch_se_over_empirical_sd": float(
            by_condition["batch_se_over_empirical_sd"].median()
        ),
        "z_current_mean": float(detail["z_current"].mean()),
        "z_current_std": float(detail["z_current"].std(ddof=1)),
        "z_batch_mean": float(detail["z_batch"].mean()),
        "z_batch_std": float(detail["z_batch"].std(ddof=1)),
        "mean_normalized_ess": float(detail["normalized_ess"].mean()),
        "maximum_weight_share": float(detail["max_weight_share"].max()),
    }])


def make_plots(detail: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(summary["empirical_sd"], summary["mean_current_se"], label="current")
    ax.scatter(summary["empirical_sd"], summary["mean_batch_se"], label="batch")
    limit = 1.08 * max(
        summary["empirical_sd"].max(),
        summary["mean_current_se"].max(),
        summary["mean_batch_se"].max(),
    )
    ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="ideal")
    ax.set(xlabel="Empirical SD across replicates", ylabel="Mean reported SE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "reported_vs_empirical_se.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(summary))
    ax.plot(x, summary["current_coverage_95"], "o-", label="current 95%")
    ax.plot(x, summary["batch_coverage_95_t"], "s-", label="batch t 95%")
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Condition (ordered by selected index)",
        ylabel="Empirical coverage",
        ylim=(0, 1.03),
        xticks=x,
        xticklabels=summary["condition_index"].astype(str),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "coverage_by_condition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bins = np.linspace(-4.5, 4.5, 37)
    ax.hist(detail["z_current"].dropna(), bins=bins, alpha=0.6, label="current")
    ax.hist(detail["z_batch"].dropna(), bins=bins, alpha=0.5, label="batch")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Standardized error z", ylabel="Runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "standardized_errors.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = read_conditions(args.conditions_file)
    selected = choose_indices(args, conditions)
    selected_frame = conditions[
        conditions["condition_index"].astype(int).isin(selected)
    ].copy()
    selected_frame["selection_order"] = selected_frame["condition_index"].map(
        {value: order for order, value in enumerate(selected)}
    )
    selected_frame = selected_frame.sort_values("selection_order")
    selected_frame.to_csv(args.output / "selected_conditions.csv", index=False)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, uniform_fraction, checkpoint = common.load_nf(args.checkpoint, device)
    adaptive.DIM = DIM

    partial_path = args.output / "partial_replicate_results.csv"
    if partial_path.exists() and not args.restart:
        rows = pd.read_csv(partial_path).to_dict("records")
    else:
        rows = []
    completed = {
        (int(row["condition_index"]), int(row["replicate"])) for row in rows
    }

    print(
        f"Frozen 32D NF error calibration | proposal={args.proposal} | "
        f"conditions={len(selected)} | replicates={args.replicates} | "
        f"evals/run={args.evals:,}\n"
        f"checkpoint={args.checkpoint}\n"
        f"checkpoint mixture fraction={uniform_fraction:.6g} | "
        f"edges={sum(len(value) for value in checkpoint['graph'].values())}",
        flush=True,
    )

    total = len(selected) * args.replicates
    done = len(completed)
    for record in selected_frame.to_dict("records"):
        index = int(record["condition_index"])
        condition = np.array([record["c0"], record["c1"], record["c2"]])
        for replicate in range(args.replicates):
            if (index, replicate) in completed:
                continue
            row = run_replicate(
                model, uniform_fraction, condition, index, replicate, args, device
            )
            rows.append(row)
            done += 1
            pd.DataFrame(rows).sort_values(
                ["condition_index", "replicate"]
            ).to_csv(partial_path, index=False)
            print(
                f"{done:4d}/{total} c={index:3d} r={replicate:2d} "
                f"I={row['estimate']:.7f} SE={row['current_se']:.3g} "
                f"ESS={row['normalized_ess']:.4f} "
                f"max-share={row['max_weight_share']:.3g}",
                flush=True,
            )

    detail = pd.DataFrame(rows).sort_values(["condition_index", "replicate"])
    detail.to_csv(args.output / "replicate_results.csv", index=False)
    by_condition = condition_summary(detail)
    by_condition.to_csv(args.output / "condition_summary.csv", index=False)
    overall = overall_summary(detail, by_condition)
    overall.to_csv(args.output / "overall_summary.csv", index=False)
    make_plots(detail, by_condition, args.output)

    config = vars(args).copy()
    config.update({
        "checkpoint": str(args.checkpoint),
        "conditions_file": str(args.conditions_file),
        "prior_results": str(args.prior_results),
        "output": str(args.output),
        "selected_condition_indices": selected,
        "true_integral": TRUE_INTEGRAL,
        "checkpoint_uniform_fraction": uniform_fraction,
        "training_performed": False,
    })
    with (args.output / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2)

    print("\nOVERALL CALIBRATION", flush=True)
    print(overall.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
