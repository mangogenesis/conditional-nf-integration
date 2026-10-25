#!/usr/bin/env python3
"""Validate the conditional ttH NF integral with randomized Sobol QMC.

The reference calculation uses the MadGraph matrix element and the existing
resonance-aware phase-space map directly; it does not use the trained NF.
One Sobol coordinate samples sqrt(s_hat) uniformly and the other 20 sample the
phase-space cube, so the averaged result is exactly the quantity reported by
the conditional training script.  A fixed-energy scan is also produced.

This validates the NF importance-sampling estimate, but not a normalization
error shared by the MadGraph wrapper and its phase-space Jacobian.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from madgraph_tth_integrand import MadGraphTTHIntegrand


DIM = 20
COND_DIM = 3
GEV2_TO_FB = 3.893793721e11
_WORKER_EVALUATOR = None


def initialize_me_worker(standalone: str, width_window: float, alpha_s: float):
    """Load an independent f2py matrix-element module in each worker process."""
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = MadGraphTTHIntegrand(
        Path(standalone), width_window=width_window, alpha_s=alpha_s
    )


def worker_log_integrand(x: np.ndarray, energy: np.ndarray) -> np.ndarray:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("Matrix-element worker was not initialized")
    return _WORKER_EVALUATOR.log_integrand(x, energy)


def condition_features(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    return np.column_stack((z, z * z, np.log1p(z) / math.log(2.0)))


def stable_mean_from_logs(log_values: np.ndarray) -> float:
    finite = np.isfinite(log_values)
    if not np.any(finite):
        return 0.0
    return math.exp(float(logsumexp(log_values[finite]) - math.log(log_values.size)))


def summarize_replicates(estimates: list[float]) -> dict[str, float]:
    values = np.asarray(estimates, dtype=np.float64)
    stderr = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else math.nan
    mean = float(values.mean())
    return {
        "estimate_gev2": mean,
        "stderr_gev2": stderr,
        "estimate_fb": mean * GEV2_TO_FB,
        "stderr_fb": stderr * GEV2_TO_FB,
        "relative_stderr": stderr / abs(mean) if mean else math.inf,
    }


def evaluate_log_integrand(evaluator, x, energy, chunk, executor=None):
    if executor is not None:
        futures = []
        for start in range(0, x.shape[0], chunk):
            stop = min(start + chunk, x.shape[0])
            futures.append(
                executor.submit(worker_log_integrand, x[start:stop], energy[start:stop])
            )
        return np.concatenate([future.result() for future in futures])
    pieces = []
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        pieces.append(evaluator.log_integrand(x[start:stop], energy[start:stop]))
    return np.concatenate(pieces)


def sobol_average(evaluator, args, executor=None):
    points = 1 << args.sobol_power
    rows, estimates = [], []
    for replicate in range(args.replicates):
        started = time.perf_counter()
        engine = qmc.Sobol(d=DIM + 1, scramble=True, seed=args.seed + replicate)
        sample = engine.random_base2(args.sobol_power)
        z = sample[:, 0]
        energy = args.sqrt_s_min + (args.sqrt_s_max - args.sqrt_s_min) * z
        logf = evaluate_log_integrand(
            evaluator, sample[:, 1:], energy, args.me_chunk, executor
        )
        estimate = stable_mean_from_logs(logf)
        estimates.append(estimate)
        rows.append({
            "scope": "energy_average", "energy_gev": math.nan,
            "method": "sobol_qmc", "replicate": replicate,
            "evaluations": points, "estimate_gev2": estimate,
            "estimate_fb": estimate * GEV2_TO_FB,
            "seconds": time.perf_counter() - started,
        })
        print(f"Sobol average {replicate + 1}/{args.replicates}: {estimate:.12e} GeV^-2", flush=True)
    return rows, summarize_replicates(estimates)


def sobol_fixed_energy(evaluator, energy_value, args, executor=None):
    points = 1 << args.fixed_power
    rows, estimates = [], []
    for replicate in range(args.fixed_replicates):
        started = time.perf_counter()
        seed = args.seed + 100_000 + int(round(energy_value)) * 100 + replicate
        engine = qmc.Sobol(d=DIM, scramble=True, seed=seed)
        x = engine.random_base2(args.fixed_power)
        energy = np.full(points, energy_value, dtype=np.float64)
        estimate = stable_mean_from_logs(
            evaluate_log_integrand(evaluator, x, energy, args.me_chunk, executor)
        )
        estimates.append(estimate)
        rows.append({
            "scope": "fixed_energy", "energy_gev": energy_value,
            "method": "sobol_qmc", "replicate": replicate,
            "evaluations": points, "estimate_gev2": estimate,
            "estimate_fb": estimate * GEV2_TO_FB,
            "seconds": time.perf_counter() - started,
        })
    return rows, summarize_replicates(estimates)


def load_nf(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("benchmark") != "madgraph_tth_20":
        raise ValueError(f"Checkpoint benchmark is {checkpoint.get('benchmark')!r}, expected 'madgraph_tth_20'")
    adaptive.DIM = DIM
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def nf_fixed_energy(model, evaluator, energy_value, args, device, executor=None):
    z_value = (energy_value - args.sqrt_s_min) / (args.sqrt_s_max - args.sqrt_s_min)
    feature = torch.as_tensor(condition_features([z_value]), device=device, dtype=torch.float32)
    parts = []
    pending = []
    remaining = args.nf_evals
    started = time.perf_counter()
    while remaining:
        count = min(remaining, args.gpu_chunk)
        cond = feature.expand(count, COND_DIM)
        x, logq = adaptive.sample_from_proposal(
            proposal_model=model, n=count, cond=cond, device=device,
            uniform_fraction=args.uniform_fraction,
        )
        x_np = x.detach().cpu().double().numpy()
        energy = np.full(count, energy_value, dtype=np.float64)
        logq_np = logq.detach().cpu().double().numpy()
        if executor is None:
            logf = evaluate_log_integrand(evaluator, x_np, energy, args.me_chunk)
            parts.append(logf - logq_np)
        else:
            pending.append((executor.submit(worker_log_integrand, x_np, energy), logq_np))
        remaining -= count
    for future, logq_np in pending:
        parts.append(future.result() - logq_np)
    logw = np.concatenate(parts)
    log_sum = float(logsumexp(logw))
    log_sum2 = float(logsumexp(2.0 * logw))
    n = logw.size
    estimate = math.exp(log_sum - math.log(n))
    second = math.exp(log_sum2 - math.log(n))
    variance = max(0.0, second - estimate * estimate)
    stderr = math.sqrt(variance / n)
    ess = math.exp(2.0 * log_sum - math.log(n) - log_sum2)
    return {
        "scope": "fixed_energy", "energy_gev": energy_value,
        "method": "nf_mixture", "replicate": 0, "evaluations": n,
        "estimate_gev2": estimate, "stderr_gev2": stderr,
        "estimate_fb": estimate * GEV2_TO_FB,
        "stderr_fb": stderr * GEV2_TO_FB,
        "relative_stderr": stderr / abs(estimate), "normalized_ess": ess,
        "seconds": time.perf_counter() - started,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", type=Path, default=Path("generated/gg_tth_full_standalone"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/madgraph_tth_copula_thorough/madgraph_tth_20_final_model.pt"))
    parser.add_argument("--sqrt-s-min", type=float, default=600.0)
    parser.add_argument("--sqrt-s-max", type=float, default=2000.0)
    parser.add_argument("--energies", type=float, nargs="+", default=(600.0, 800.0, 1000.0, 1400.0, 2000.0))
    parser.add_argument("--width-window", type=float, default=15.0)
    parser.add_argument("--alpha-s", type=float, default=0.118)
    parser.add_argument("--sobol-power", type=int, default=15, help="points per average replicate = 2^power")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--fixed-power", type=int, default=13)
    parser.add_argument("--fixed-replicates", type=int, default=4)
    parser.add_argument("--nf-evals", type=int, default=100_000)
    parser.add_argument("--checkpoint-evals", type=int, default=100_000,
                        help="sample count used for the checkpoint's final reported estimate")
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--me-chunk", type=int, default=512)
    parser.add_argument("--me-workers", type=int, default=1,
                        help="parallel matrix-element worker processes")
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--skip-average", action="store_true",
                        help="skip the 21-D energy-averaged Sobol calculation")
    parser.add_argument("--skip-nf", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/madgraph_tth_validation"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sqrt_s_max <= args.sqrt_s_min:
        raise ValueError("--sqrt-s-max must exceed --sqrt-s-min")
    if any(e < args.sqrt_s_min or e > args.sqrt_s_max for e in args.energies):
        raise ValueError("fixed energies must lie inside the trained energy interval")
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator = MadGraphTTHIntegrand(
        args.standalone, width_window=args.width_window, alpha_s=args.alpha_s
    )
    executor = None
    if args.me_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.me_workers,
            initializer=initialize_me_worker,
            initargs=(str(args.standalone.resolve()), args.width_window, args.alpha_s),
        )
    rows = []
    summaries = []
    average = None
    if not args.skip_average:
        average_rows, average = sobol_average(evaluator, args, executor)
        rows.extend(average_rows)
        summaries.append({"scope": "energy_average", "energy_gev": math.nan,
                          "method": "sobol_qmc", **average})

    model = None
    if not args.skip_nf:
        model, checkpoint = load_nf(args.checkpoint, torch.device(args.device))
        if not args.skip_average:
            final = checkpoint.get("final_metrics", {})
            nf_average = float(final.get("integral_estimate", math.nan))
            nf_average_ess = float(final.get("ess", math.nan))
            nf_average_stderr = (
                abs(nf_average) * math.sqrt((1.0 / nf_average_ess - 1.0) / args.checkpoint_evals)
                if math.isfinite(nf_average) and 0.0 < nf_average_ess <= 1.0 else math.nan
            )
            summaries.append({
                "scope": "energy_average", "energy_gev": math.nan,
                "method": "nf_checkpoint", "estimate_gev2": nf_average,
                "estimate_fb": nf_average * GEV2_TO_FB,
                "stderr_gev2": nf_average_stderr,
                "stderr_fb": nf_average_stderr * GEV2_TO_FB,
                "relative_stderr": nf_average_stderr / abs(nf_average),
                "normalized_ess": nf_average_ess,
            })
        if average is not None:
            combined_stderr = math.hypot(average["stderr_gev2"], nf_average_stderr)
            difference = nf_average - average["estimate_gev2"]
            print(
                f"Average comparison: NF-Sobol={difference:.3e} GeV^-2 "
                f"({difference / combined_stderr:.2f} combined sigma)", flush=True,
            )

    if not args.skip_fixed:
        for energy in args.energies:
            fixed_rows, fixed_summary = sobol_fixed_energy(
                evaluator, energy, args, executor
            )
            rows.extend(fixed_rows)
            summaries.append({"scope": "fixed_energy", "energy_gev": energy,
                              "method": "sobol_qmc", **fixed_summary})
            print(f"Sobol {energy:g} GeV: {fixed_summary['estimate_fb']:.6g} +/- "
                  f"{fixed_summary['stderr_fb']:.2g} fb", flush=True)
            if model is not None:
                nf = nf_fixed_energy(
                    model, evaluator, energy, args, torch.device(args.device), executor
                )
                rows.append(nf)
                summaries.append({key: nf.get(key, math.nan) for key in (
                    "scope", "energy_gev", "method", "estimate_gev2", "stderr_gev2",
                    "estimate_fb", "stderr_fb", "relative_stderr", "normalized_ess")})
                print(f"NF    {energy:g} GeV: {nf['estimate_fb']:.6g} +/- {nf['stderr_fb']:.2g} fb "
                      f"ESS={nf['normalized_ess']:.4f}", flush=True)

    if executor is not None:
        executor.shutdown(wait=True)

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    detail.to_csv(args.output / "replicate_results.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("\nVALIDATION SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
