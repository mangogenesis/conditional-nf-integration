#!/usr/bin/env python3
"""Compare a pure conditional NF with CPU VEGAS on shared conditions.

Both methods integrate sparse_wave_32 on the same saved condition bank.  They
receive only f(x|c); the known integral is used only after each estimate has
been produced.  NF training cost is amortized and is not counted here, so the
CSV reports inference integration evaluations and wall time explicitly.  The
optional CLI method choices retain the additional diagnostic baselines.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from scipy.stats import qmc

import run_adaptive_integrand_structure_search as adaptive
from run_vegas_sparse_wave_32_cpu import SparseWave32, integrate_condition


DIM = 32
COND_DIM = 3
TRUE_INTEGRAL = 1.0


def load_or_create_conditions(args):
    if args.conditions_file is not None:
        frame = pd.read_csv(args.conditions_file)
        missing = {"c0", "c1", "c2"} - set(frame.columns)
        if missing:
            raise ValueError(f"conditions file is missing columns: {sorted(missing)}")
        conditions = frame[["c0", "c1", "c2"]].to_numpy(dtype=np.float64)
    elif args.condition_design == "interstitial":
        if args.n_conditions != 32:
            raise ValueError(
                "the interstitial design contains exactly 32 centers; "
                "use --n-conditions 32 or --condition-design random"
            )
        # Centers of a 4 x 4 x 2 partition of [0,1]^3.  These avoid boundaries,
        # cover the condition space evenly, and act as deterministic unseen
        # interpolation points rather than a randomly clustered test bank.
        conditions = np.asarray([
            ((i + 0.5) / 4.0, (j + 0.5) / 4.0, (k + 0.5) / 2.0)
            for i in range(4) for j in range(4) for k in range(2)
        ], dtype=np.float64)
    else:
        conditions = np.random.default_rng(args.condition_seed).random(
            (args.n_conditions, COND_DIM), dtype=np.float64
        )
    if len(conditions) == 0 or np.any((conditions < 0.0) | (conditions > 1.0)):
        raise ValueError("conditions must be a nonempty array in [0,1]^3")
    return conditions


def load_nf(checkpoint_path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("benchmark") != "sparse_wave_32":
        raise ValueError("checkpoint is not for sparse_wave_32")
    adaptive.DIM = DIM
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    uniform_fraction = float(
        checkpoint.get("final_metrics", {}).get("uniform_fraction", 0.10)
    )
    return model, uniform_fraction, checkpoint


def summarize_log_weights(parts):
    logw = np.concatenate(parts).astype(np.float64, copy=False)
    n = logw.size
    log_sum = logsumexp(logw)
    log_sum2 = logsumexp(2.0 * logw)
    estimate = math.exp(log_sum - math.log(n))
    second = math.exp(log_sum2 - math.log(n))
    variance = max(0.0, second - estimate * estimate)
    stderr = math.sqrt(variance / n)
    ess = math.exp(2.0 * log_sum - math.log(n) - log_sum2)
    return estimate, stderr, ess


@torch.no_grad()
def evaluate_nf(model, uniform_fraction, conditions, args, device, method):
    rows = []
    for index, condition in enumerate(conditions):
        torch.manual_seed(args.seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + index)
        started = time.perf_counter()
        parts = []
        remaining = args.nf_evals
        while remaining:
            count = min(remaining, args.gpu_chunk)
            cond = torch.as_tensor(condition, device=device, dtype=torch.float32)
            cond = cond.expand(count, COND_DIM)
            if uniform_fraction == 0.0:
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
            parts.append((logf - logq).double().cpu().numpy())
            remaining -= count
        estimate, stderr, ess = summarize_log_weights(parts)
        rows.append(make_row(
            index, condition, method, estimate, stderr, ess,
            args.nf_evals, time.perf_counter() - started,
        ))
        print(
            f"{method:10s} {index + 1:3d}/{len(conditions)} I={estimate:.7f} "
            f"ESS={ess:.4f}", flush=True,
        )
    return rows


def evaluate_uniform_condition(job):
    index, condition, seed, evaluations, cpu_chunk = job
    rng = np.random.default_rng(seed + 100_000 + index)
    started = time.perf_counter()
    integrand = SparseWave32(condition)
    parts = []
    remaining = evaluations
    while remaining:
        count = min(remaining, cpu_chunk)
        x = rng.random((count, DIM), dtype=np.float64)
        parts.append(integrand.log_value(x))
        remaining -= count
    estimate, stderr, ess = summarize_log_weights(parts)
    return make_row(
        index, condition, "uniform_mc", estimate, stderr, ess,
        evaluations, time.perf_counter() - started,
    )


def run_parallel_conditions(worker, jobs, workers, label):
    rows = []
    workers = min(workers, len(jobs))
    if workers == 1:
        for completed, job in enumerate(jobs, 1):
            rows.append(worker(job))
            print(f"{label} {completed:3d}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                print(f"{label} {completed:3d}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: row["condition_index"])
    return rows


def evaluate_uniform(conditions, args):
    jobs = [
        (index, condition, args.seed, args.uniform_evals, args.cpu_chunk)
        for index, condition in enumerate(conditions)
    ]
    return run_parallel_conditions(
        evaluate_uniform_condition, jobs, args.workers, "uniform"
    )


def evaluate_sobol_condition(job):
    index, condition, seed, sobol_power, sobol_replicates = job
    points = 1 << sobol_power
    started = time.perf_counter()
    estimates, all_logf = [], []
    integrand = SparseWave32(condition)
    for replicate in range(sobol_replicates):
        engine = qmc.Sobol(
            d=DIM, scramble=True,
            seed=seed + 200_000 + 1000 * index + replicate,
        )
        x = engine.random_base2(sobol_power)
        logf = integrand.log_value(x)
        all_logf.append(logf)
        estimates.append(math.exp(logsumexp(logf) - math.log(points)))
    estimates = np.asarray(estimates)
    _, _, ess = summarize_log_weights(all_logf)
    stderr = (
        float(estimates.std(ddof=1) / math.sqrt(len(estimates)))
        if len(estimates) > 1 else math.nan
    )
    return make_row(
        index, condition, "sobol_qmc", float(estimates.mean()), stderr, ess,
        points * sobol_replicates, time.perf_counter() - started,
    )


def evaluate_sobol(conditions, args):
    jobs = [
        (index, condition, args.seed, args.sobol_power, args.sobol_replicates)
        for index, condition in enumerate(conditions)
    ]
    return run_parallel_conditions(
        evaluate_sobol_condition, jobs, args.workers, "Sobol"
    )


def evaluate_vegas(conditions, args):
    jobs = [(
        index, condition, args.seed + 300_000,
        args.vegas_warmup_nitn, args.vegas_warmup_neval,
        args.vegas_production_nitn, args.vegas_production_neval,
        args.vegas_alpha, args.vegas_min_batch,
    ) for index, condition in enumerate(conditions)]
    results = []
    workers = min(args.workers, len(jobs))
    if workers == 1:
        results = [integrate_condition(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [pool.submit(integrate_condition, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"VEGAS {completed:3d}/{len(jobs)}", flush=True)
    results.sort(key=lambda item: item.index)
    return [make_row(
        result.index,
        np.array([result.c0, result.c1, result.c2]),
        "vegas",
        result.estimate,
        result.sdev,
        math.nan,
        result.total_evaluations,
        result.total_seconds,
        q_value=result.q_value,
    ) for result in results]


def make_row(index, condition, method, estimate, stderr, ess, evaluations, seconds,
             q_value=math.nan):
    error = estimate - TRUE_INTEGRAL
    return {
        "condition_index": index,
        "c0": float(condition[0]), "c1": float(condition[1]),
        "c2": float(condition[2]), "method": method,
        "estimate": float(estimate), "std_error": float(stderr),
        "normalized_ess": float(ess), "error": float(error),
        "absolute_error": float(abs(error)), "evaluations": int(evaluations),
        "covered_95": bool(
            math.isfinite(stderr) and abs(error) <= 1.96 * stderr
        ),
        "seconds": float(seconds), "q_value": float(q_value),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("results/adaptive_integrand_all_weak/sparse_wave_32_final_model.pt"),
    )
    parser.add_argument("--conditions-file", type=Path)
    parser.add_argument("--n-conditions", type=int, default=32)
    parser.add_argument(
        "--condition-design", choices=("interstitial", "random"),
        default="interstitial",
        help="interstitial uses centers of a 4x4x2 partition; ignored when "
             "--conditions-file is supplied",
    )
    parser.add_argument("--condition-seed", type=int, default=20260913)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--methods", nargs="+",
        choices=("nf_pure", "nf_mixture", "vegas", "sobol", "uniform"),
        default=("nf_pure", "vegas"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nf-evals", type=int, default=1_000_000)
    parser.add_argument("--uniform-evals", type=int, default=1_000_000)
    parser.add_argument("--sobol-power", type=int, default=17)
    parser.add_argument("--sobol-replicates", type=int, default=8)
    parser.add_argument("--gpu-chunk", type=int, default=16_384)
    parser.add_argument("--cpu-chunk", type=int, default=131_072)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--vegas-warmup-nitn", type=int, default=5)
    parser.add_argument("--vegas-warmup-neval", type=int, default=100_000)
    parser.add_argument("--vegas-production-nitn", type=int, default=5)
    parser.add_argument("--vegas-production-neval", type=int, default=100_000)
    parser.add_argument("--vegas-alpha", type=float, default=0.30)
    parser.add_argument("--vegas-min-batch", type=int, default=50_000)
    parser.add_argument("--output", type=Path,
                        default=Path("results/final_checkpoint_comparison"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = load_or_create_conditions(args)
    pd.DataFrame(conditions, columns=["c0", "c1", "c2"]).to_csv(
        args.output / "conditions.csv", index_label="condition_index"
    )
    rows = []
    model = None
    if "nf_pure" in args.methods or "nf_mixture" in args.methods:
        model, uniform_fraction, checkpoint = load_nf(
            args.checkpoint, torch.device(args.device)
        )
        print(
            f"Loaded NF: edges={sum(len(v) for v in checkpoint['graph'].values())}, "
            f"uniform={uniform_fraction:.3f}", flush=True,
        )
        if "nf_pure" in args.methods:
            rows.extend(evaluate_nf(
                model, 0.0, conditions, args, torch.device(args.device), "nf_pure"
            ))
        if "nf_mixture" in args.methods:
            rows.extend(evaluate_nf(
                model, uniform_fraction, conditions, args,
                torch.device(args.device), "nf_mixture"
            ))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if "sobol" in args.methods:
        rows.extend(evaluate_sobol(conditions, args))
    if "uniform" in args.methods:
        rows.extend(evaluate_uniform(conditions, args))
    if "vegas" in args.methods:
        rows.extend(evaluate_vegas(conditions, args))

    detail = pd.DataFrame(rows).sort_values(["condition_index", "method"])
    detail.to_csv(args.output / "condition_results.csv", index=False)
    summary = detail.groupby("method", sort=False).agg(
        conditions=("condition_index", "count"),
        mean_estimate=("estimate", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        median_absolute_error=("absolute_error", "median"),
        maximum_absolute_error=("absolute_error", "max"),
        rmse=("error", lambda values: math.sqrt(float(np.mean(values * values)))),
        mean_normalized_ess=("normalized_ess", "mean"),
        median_normalized_ess=("normalized_ess", "median"),
        coverage_95=("covered_95", "mean"),
        total_evaluations=("evaluations", "sum"),
        total_condition_seconds=("seconds", "sum"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    print("\nCOMPARISON SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
