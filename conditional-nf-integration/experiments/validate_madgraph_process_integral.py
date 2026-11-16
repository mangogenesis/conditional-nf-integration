#!/usr/bin/env python3
"""Held-out NF and randomized-Sobol validation for a configured process."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from scipy.stats import qmc

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import run_adaptive_integrand_structure_search as adaptive
from madgraph_process_integrand import ConfiguredMadGraphIntegrand, load_process_spec


GEV2_TO_PB = 3.893793721e8
COND_DIM = 3


def condition_features(z: float) -> np.ndarray:
    return np.array([z, z * z, math.log1p(z) / math.log(2.0)], dtype=np.float32)


def stable_mean(log_values: np.ndarray) -> float:
    finite = np.isfinite(log_values)
    if not np.any(finite):
        return 0.0
    return math.exp(float(logsumexp(log_values[finite]) - math.log(len(log_values))))


def weight_diagnostics(log_weights: np.ndarray) -> dict:
    """Summarize an importance-weight sample without dropping -inf values."""
    finite = np.isfinite(log_weights)
    count = int(log_weights.size)
    if not finite.any():
        return {
            "max_weight_over_mean": float("inf"),
            "cut_pass_fraction": 0.0,
            "zero_weight_fraction": 1.0,
            "std_logw_finite": float("nan"),
        }
    finite_log_weights = log_weights[finite]
    log_mean = float(logsumexp(log_weights) - math.log(count))
    max_over_mean = math.exp(
        min(float(finite_log_weights.max()) - log_mean, 700.0)
    )
    pass_fraction = float(finite.mean())
    return {
        "max_weight_over_mean": max_over_mean,
        "cut_pass_fraction": pass_fraction,
        "zero_weight_fraction": 1.0 - pass_fraction,
        "std_logw_finite": (
            float(np.std(finite_log_weights))
            if finite_log_weights.size > 1
            else 0.0
        ),
    }


def load_model(checkpoint_path: Path, dimension: int, device: torch.device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    adaptive.DIM = dimension
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def sobol_replicates(evaluator, energy: float, args):
    estimates = []
    rows = []
    points = 1 << args.sobol_power
    for replicate in range(args.sobol_replicates):
        started = time.perf_counter()
        engine = qmc.Sobol(
            d=evaluator.dimension, scramble=True,
            seed=args.seed + int(round(energy)) * 100 + replicate,
        )
        unit = engine.random_base2(args.sobol_power)
        energies = np.full(points, energy, dtype=np.float64)
        estimate = stable_mean(evaluator.log_integrand(unit, energies))
        estimates.append(estimate)
        rows.append({
            "energy_gev": energy, "method": "sobol_qmc", "replicate": replicate,
            "evaluations": points, "estimate_gev2": estimate,
            "estimate_pb": estimate * GEV2_TO_PB,
            "seconds": time.perf_counter() - started,
        })
    values = np.asarray(estimates)
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / math.sqrt(len(values)))
    return rows, {
        "energy_gev": energy, "method": "sobol_qmc",
        "estimate_gev2": mean, "stderr_gev2": stderr,
        "estimate_pb": mean * GEV2_TO_PB, "stderr_pb": stderr * GEV2_TO_PB,
        "relative_stderr": stderr / abs(mean), "normalized_ess": math.nan,
    }


@torch.no_grad()
def nf_estimate(model, evaluator, energy: float, minimum: float, maximum: float, args, device):
    z = (energy - minimum) / (maximum - minimum)
    feature = torch.as_tensor(condition_features(z), device=device).reshape(1, COND_DIM)
    log_weights = []
    remaining = args.nf_evals
    started = time.perf_counter()
    while remaining:
        count = min(remaining, args.gpu_chunk)
        cond = feature.expand(count, COND_DIM)
        unit, logq = adaptive.sample_from_proposal(
            proposal_model=model, n=count, cond=cond, device=device,
            uniform_fraction=args.uniform_fraction,
        )
        unit_numpy = unit.detach().cpu().double().numpy()
        energies = np.full(count, energy, dtype=np.float64)
        logf = evaluator.log_integrand(unit_numpy, energies)
        log_weights.append(logf - logq.detach().cpu().double().numpy())
        remaining -= count
    logw = np.concatenate(log_weights)
    log_sum = float(logsumexp(logw))
    log_sum2 = float(logsumexp(2.0 * logw))
    count = len(logw)
    estimate = math.exp(log_sum - math.log(count))
    second = math.exp(log_sum2 - math.log(count))
    variance = max(0.0, second - estimate * estimate)
    stderr = math.sqrt(variance / count)
    ess = math.exp(2.0 * log_sum - math.log(count) - log_sum2)
    return {
        "energy_gev": energy, "method": "conditional_nf", "replicate": 0,
        "evaluations": count, "estimate_gev2": estimate,
        "stderr_gev2": stderr, "estimate_pb": estimate * GEV2_TO_PB,
        "stderr_pb": stderr * GEV2_TO_PB,
        "relative_stderr": stderr / abs(estimate), "normalized_ess": ess,
        "seconds": time.perf_counter() - started,
        **weight_diagnostics(logw),
    }


def uniform_estimate(evaluator, energy: float, args) -> dict:
    """Equal-budget uniform Monte Carlo: log q(u) = 0 on the unit hypercube."""
    count = args.nf_evals
    rng = np.random.default_rng(args.seed + int(round(energy)) * 1000)
    started = time.perf_counter()
    unit = rng.random((count, evaluator.dimension))
    energies = np.full(count, energy, dtype=np.float64)
    logf = evaluator.log_integrand(unit, energies)
    elapsed = time.perf_counter() - started

    if np.isfinite(logf).any():
        log_sum = float(logsumexp(logf))
        log_sum2 = float(logsumexp(2.0 * logf))
        estimate = math.exp(log_sum - math.log(count))
        second = math.exp(log_sum2 - math.log(count))
        variance = max(0.0, second - estimate * estimate)
        stderr = math.sqrt(variance / count)
        ess = math.exp(2.0 * log_sum - math.log(count) - log_sum2)
    else:
        estimate = 0.0
        stderr = 0.0
        ess = 0.0

    return {
        "energy_gev": energy, "method": "uniform_mc", "replicate": 0,
        "evaluations": count, "estimate_gev2": estimate,
        "stderr_gev2": stderr, "estimate_pb": estimate * GEV2_TO_PB,
        "stderr_pb": stderr * GEV2_TO_PB,
        "relative_stderr": stderr / abs(estimate) if estimate else float("inf"),
        "normalized_ess": ess, "seconds": elapsed,
        **weight_diagnostics(logf),
    }


def compare_estimates(nf: dict, uniform: dict) -> tuple[float, float, float]:
    """Pull, ESS ratio and variance-reduction factor (NF against uniform)."""
    combined = math.hypot(nf["stderr_pb"], uniform["stderr_pb"])
    pull = (
        (nf["estimate_pb"] - uniform["estimate_pb"]) / combined
        if combined > 0.0 else float("nan")
    )
    ess_ratio = (
        nf["normalized_ess"] / uniform["normalized_ess"]
        if uniform["normalized_ess"] > 0.0 else float("inf")
    )
    variance_factor = (
        (uniform["stderr_pb"] / nf["stderr_pb"]) ** 2
        if nf["stderr_pb"] > 0.0 else float("inf")
    )
    return pull, ess_ratio, variance_factor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--channel",
        default=None,
        help="Channel name from the config's `channels` list (required for multi-subprocess configs).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--energies", type=float, nargs="+", default=None)
    parser.add_argument("--sobol-power", type=int, default=18)
    parser.add_argument("--sobol-replicates", type=int, default=8)
    parser.add_argument("--nf-evals", type=int, default=500_000)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--alpha-s", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-sobol",
        action="store_true",
        help=(
            "Skip the randomized-Sobol reference and evaluate only the "
            "conditional NF. Useful for cut-regulated processes where the "
            "Sobol baseline would spend most of its matrix-element calls on "
            "rejected points."
        ),
    )
    parser.add_argument(
        "--compare-uniform",
        action="store_true",
        help=(
            "Evaluate a uniform Monte-Carlo baseline at the same budget as "
            "the NF (--nf-evals). Intended for --no-sobol runs where a fast, "
            "equal-budget reference is wanted."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config_data.get("channels") and args.channel is None:
        raise ValueError(
            "This config defines multiple channels; pass --channel with one of: "
            + ", ".join(item["name"] for item in config_data["channels"])
        )
    spec = load_process_spec(args.config, channel=args.channel)
    alpha_s = (
        args.alpha_s
        if args.alpha_s is not None
        else float(spec.get("alpha_s", 0.118))
    )
    evaluator = ConfiguredMadGraphIntegrand(spec, PROJECT, alpha_s)
    condition = spec["condition"]
    minimum, maximum = float(condition["minimum"]), float(condition["maximum"])
    energies = args.energies or [float(item) for item in condition["held_out"]]
    if any(item < minimum or item > maximum for item in energies):
        raise ValueError("Every validation energy must lie inside the condition range")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, evaluator.dimension, device)
    expected = f"madgraph_{spec['name']}_{evaluator.dimension}"
    if checkpoint.get("benchmark") != expected:
        raise ValueError(
            f"Checkpoint benchmark {checkpoint.get('benchmark')!r}; expected {expected!r}"
        )

    detail_rows, summary_rows = [], []
    for energy in energies:
        nf = nf_estimate(model, evaluator, energy, minimum, maximum, args, device)
        uniform = (
            uniform_estimate(evaluator, energy, args)
            if args.compare_uniform else None
        )

        if args.no_sobol:
            detail_rows.append(nf)
            summary_rows.append(nf)
            print(
                f"{energy:g} GeV: NF={nf['estimate_pb']:.8g} +/- "
                f"{nf['stderr_pb']:.2g} pb "
                f"(rel={nf['relative_stderr'] * 100:.3f}%) "
                f"ESS={nf['normalized_ess']:.5f} "
                f"time={nf['seconds']:.2f}s",
                flush=True,
            )
            if uniform is not None:
                detail_rows.append(uniform)
                summary_rows.append(uniform)
                pull, ess_ratio, variance_factor = compare_estimates(nf, uniform)
                print(
                    f"{energy:g} GeV: Uniform="
                    f"{uniform['estimate_pb']:.8g} +/- "
                    f"{uniform['stderr_pb']:.2g} pb "
                    f"(rel={uniform['relative_stderr'] * 100:.3f}%) "
                    f"ESS={uniform['normalized_ess']:.5f} "
                    f"time={uniform['seconds']:.2f}s; "
                    f"pull={pull:+.2f}σ; "
                    f"ESS_NF/ESS_uniform={ess_ratio:.2f}x; "
                    f"variance_reduction=(σ_u/σ_NF)^2={variance_factor:.2f}x",
                    flush=True,
                )
            continue

        sobol_rows, sobol_summary = sobol_replicates(evaluator, energy, args)
        detail_rows.extend(sobol_rows)
        detail_rows.append(nf)
        summary_rows.extend((sobol_summary, nf))
        if uniform is not None:
            detail_rows.append(uniform)
            summary_rows.append(uniform)
        combined = math.hypot(sobol_summary["stderr_pb"], nf["stderr_pb"])
        pull = (nf["estimate_pb"] - sobol_summary["estimate_pb"]) / combined
        print(
            f"{energy:g} GeV: NF={nf['estimate_pb']:.8g} +/- {nf['stderr_pb']:.2g} pb "
            f"ESS={nf['normalized_ess']:.5f}; Sobol={sobol_summary['estimate_pb']:.8g} "
            f"+/- {sobol_summary['stderr_pb']:.2g} pb; pull={pull:.2f}", flush=True,
        )
        if uniform is not None:
            pull_u, ess_ratio, variance_factor = compare_estimates(nf, uniform)
            print(
                f"{energy:g} GeV: Uniform={uniform['estimate_pb']:.8g} +/- "
                f"{uniform['stderr_pb']:.2g} pb "
                f"(rel={uniform['relative_stderr'] * 100:.3f}%) "
                f"ESS={uniform['normalized_ess']:.5f} "
                f"time={uniform['seconds']:.2f}s; "
                f"pull={pull_u:+.2f}σ; "
                f"ESS_NF/ESS_uniform={ess_ratio:.2f}x; "
                f"variance_reduction=(σ_u/σ_NF)^2={variance_factor:.2f}x",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(args.output / "replicate_results.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output / "summary.csv", index=False)
    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    config["resolved_energies"] = energies
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
