#!/usr/bin/env python3
"""Integrate sparse_wave_32 with externally calibrated uncertainty.

The frozen NF is never retrained.  For each condition, independent pilot
batches estimate the conditional first and second moments of the importance
weight.  A separate production sample estimates the integral.  The reported
interval uses a critical multiplier fixed by prior held-out calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_adaptive_integrand_structure_search as adaptive  # noqa: E402
import run_final_checkpoint_integration_comparison as common  # noqa: E402


DIM = 32
COND_DIM = 3
DEFAULT_Q68 = 0.9229010680751596
DEFAULT_Q95 = 2.1189533096596533


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
    parser.add_argument("--condition-indices", type=int, nargs="+", required=True)
    parser.add_argument("--pilot-batches", type=int, default=20)
    parser.add_argument("--pilot-evals", type=int, default=100_000)
    parser.add_argument("--production-evals", type=int, default=100_000)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--q68", type=float, default=DEFAULT_Q68)
    parser.add_argument("--q95", type=float, default=DEFAULT_Q95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("results/calibrated_nf_32d")
    )
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.pilot_batches < 2:
        parser.error("--pilot-batches must be at least 2")
    if min(args.pilot_evals, args.production_evals, args.gpu_chunk) <= 0:
        parser.error("evaluation counts and chunk size must be positive")
    return args


def load_conditions(path: Path, indices: list[int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"c0", "c1", "c2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"conditions file lacks columns: {sorted(missing)}")
    if "condition_index" not in frame:
        frame.insert(0, "condition_index", np.arange(len(frame), dtype=int))
    frame["condition_index"] = frame["condition_index"].astype(int)
    indexed = frame.set_index("condition_index")
    unknown = sorted(set(indices) - set(indexed.index))
    if unknown:
        raise ValueError(f"unknown condition indices: {unknown}")
    return indexed.loc[indices].reset_index()[["condition_index", "c0", "c1", "c2"]]


@torch.inference_mode()
def draw_logweights(
    model,
    condition: np.ndarray,
    evaluations: int,
    chunk: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    parts = []
    remaining = evaluations
    while remaining:
        count = min(remaining, chunk)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)
        x = adaptive.sample_from_model(model, count, cond)
        logq = model.log_prob(x, cond).reshape(-1)
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)
        parts.append((logf - logq).double().cpu().numpy())
        remaining -= count
    return np.concatenate(parts).astype(np.float64, copy=False)


def logweight_summary(logw: np.ndarray) -> dict[str, float]:
    n = len(logw)
    log_sum = float(logsumexp(logw))
    log_sum2 = float(logsumexp(2.0 * logw))
    estimate = math.exp(log_sum - math.log(n))
    second = math.exp(log_sum2 - math.log(n))
    variance = max(0.0, second - estimate * estimate)
    return {
        "estimate": estimate,
        "second_moment": second,
        "weight_variance": variance,
        "same_run_se": math.sqrt(variance / n),
        "normalized_ess": math.exp(2.0 * log_sum - math.log(n) - log_sum2),
        "max_weight_share": math.exp(float(np.max(logw)) - log_sum),
        "reverse_kl": float(-np.mean(logw)),
        "log_weight_std": float(np.std(logw, ddof=1)),
    }


def relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(new), 1e-16)


def integrate_condition(
    model,
    record: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    index = int(record["condition_index"])
    condition = np.array([record["c0"], record["c1"], record["c2"]])
    pilot_rows = []
    started = time.perf_counter()
    for batch in range(args.pilot_batches):
        seed = args.seed + 100_003 * index + 1_009 * batch
        logw = draw_logweights(
            model, condition, args.pilot_evals, args.gpu_chunk, seed, device
        )
        summary = logweight_summary(logw)
        summary.update({
            "condition_index": index,
            "pilot_batch": batch,
            "seed": seed,
            "evaluations": args.pilot_evals,
        })
        pilot_rows.append(summary)

    pilot = pd.DataFrame(pilot_rows)
    pilot_first = float(pilot["estimate"].mean())
    pilot_second = float(pilot["second_moment"].mean())
    pilot_variance = max(0.0, pilot_second - pilot_first * pilot_first)
    production_seed = args.seed + 100_003 * index + 90_000_001
    production_logw = draw_logweights(
        model,
        condition,
        args.production_evals,
        args.gpu_chunk,
        production_seed,
        device,
    )
    production = logweight_summary(production_logw)
    calibrated_se = math.sqrt(pilot_variance / args.production_evals)

    halfway = args.pilot_batches // 2
    early_first = float(pilot.iloc[:halfway]["estimate"].mean())
    early_second = float(pilot.iloc[:halfway]["second_moment"].mean())
    early_variance = max(0.0, early_second - early_first * early_first)
    second_moment_cv = float(
        pilot["second_moment"].std(ddof=1) / pilot["second_moment"].mean()
    )
    late_variance_shift = relative_change(pilot_variance, early_variance)
    maximum_pilot_share = float(pilot["max_weight_share"].max())

    flags = []
    if second_moment_cv > 1.0:
        flags.append("pilot_second_moment_high_cv")
    if late_variance_shift > 0.20:
        flags.append("pilot_variance_not_stable")
    if maximum_pilot_share > 0.01:
        flags.append("pilot_extreme_weight")
    if production["max_weight_share"] > 0.01:
        flags.append("production_extreme_weight")
    if production["normalized_ess"] < 0.10:
        flags.append("production_low_ess")

    estimate = production["estimate"]
    result = {
        "condition_index": index,
        "c0": float(condition[0]),
        "c1": float(condition[1]),
        "c2": float(condition[2]),
        "estimate": estimate,
        "calibrated_se": calibrated_se,
        "lower_68": estimate - args.q68 * calibrated_se,
        "upper_68": estimate + args.q68 * calibrated_se,
        "lower_95": estimate - args.q95 * calibrated_se,
        "upper_95": estimate + args.q95 * calibrated_se,
        "q68": args.q68,
        "q95": args.q95,
        "pilot_batches": args.pilot_batches,
        "pilot_evals_per_batch": args.pilot_evals,
        "production_evals": args.production_evals,
        "pilot_first_moment": pilot_first,
        "pilot_second_moment": pilot_second,
        "pilot_weight_variance": pilot_variance,
        "pilot_second_moment_cv": second_moment_cv,
        "pilot_late_variance_shift": late_variance_shift,
        "pilot_max_weight_share": maximum_pilot_share,
        "same_run_se": production["same_run_se"],
        "production_normalized_ess": production["normalized_ess"],
        "production_max_weight_share": production["max_weight_share"],
        "production_reverse_kl": production["reverse_kl"],
        "production_log_weight_std": production["log_weight_std"],
        "reliability_ok": not flags,
        "reliability_flags": ";".join(flags),
        "seconds": time.perf_counter() - started,
    }
    return result, pilot_rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = load_conditions(args.conditions_file, args.condition_indices)
    conditions.to_csv(args.output / "conditions.csv", index=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, _, checkpoint = common.load_nf(args.checkpoint, device)
    adaptive.DIM = DIM

    results_path = args.output / "results.csv"
    pilots_path = args.output / "pilot_batches.csv"
    if not args.restart and results_path.exists():
        results = pd.read_csv(results_path).to_dict("records")
        pilots = (
            pd.read_csv(pilots_path).to_dict("records")
            if pilots_path.exists() else []
        )
    else:
        results, pilots = [], []
    completed = {int(row["condition_index"]) for row in results}
    for done, record in enumerate(conditions.to_dict("records"), 1):
        if int(record["condition_index"]) in completed:
            print(
                f"{done}/{len(conditions)} c={int(record['condition_index']):3d} "
                "already complete",
                flush=True,
            )
            continue
        result, pilot_rows = integrate_condition(model, record, args, device)
        results.append(result)
        pilots.extend(pilot_rows)
        pd.DataFrame(results).sort_values("condition_index").to_csv(
            results_path, index=False
        )
        pd.DataFrame(pilots).sort_values(
            ["condition_index", "pilot_batch"]
        ).to_csv(pilots_path, index=False)
        print(
            f"{done}/{len(conditions)} c={result['condition_index']:3d} "
            f"I={result['estimate']:.7f} +/- "
            f"{args.q95 * result['calibrated_se']:.3g} (95%) "
            f"ESS={result['production_normalized_ess']:.4f} "
            f"flags={result['reliability_flags'] or 'none'}",
            flush=True,
        )

    config = vars(args).copy()
    for key in ("checkpoint", "conditions_file", "output"):
        config[key] = str(config[key])
    config.update({
        "dimension": DIM,
        "training_performed": False,
        "checkpoint_benchmark": checkpoint.get("benchmark"),
        "interval_method": "independent conditional second moment with frozen tail multiplier",
    })
    with (args.output / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
