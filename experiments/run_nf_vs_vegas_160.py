#!/usr/bin/env python3
"""High-throughput, resumable NF-versus-VEGAS comparison on 160 conditions.

The two methods always use the same persisted condition bank and the same
integrand.  Only f(x|c) is exposed during integration; the known unit integral
is consulted afterward for evaluation metrics.  NF training time is reported
separately from inference and is not reconstructed by this script.
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
from scipy.stats import qmc

import run_adaptive_integrand_structure_search as adaptive
import run_final_checkpoint_integration_comparison as common
from run_vegas_sparse_wave_32_cpu import integrate_condition


N_CONDITIONS = 160
COND_DIM = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("results/adaptive_integrand_all_weak/sparse_wave_32_final_model.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/nf_vs_vegas_160"))
    parser.add_argument("--condition-seed", type=int, default=20260916)
    parser.add_argument("--seed", type=int, default=1731)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--nf-evals", type=int, default=1_000_000)
    parser.add_argument("--gpu-chunk", type=int, default=65_536)
    # Defaults give each method 1,000,000 requested evaluations/condition.
    parser.add_argument("--vegas-warmup-nitn", type=int, default=10)
    parser.add_argument("--vegas-warmup-neval", type=int, default=50_000)
    parser.add_argument("--vegas-production-nitn", type=int, default=10)
    parser.add_argument("--vegas-production-neval", type=int, default=50_000)
    parser.add_argument("--vegas-alpha", type=float, default=0.30)
    parser.add_argument("--vegas-min-batch", type=int, default=50_000)
    parser.add_argument("--restart", action="store_true",
                        help="ignore a prior partial result CSV but retain its condition bank")
    return parser.parse_args()


def load_or_create_condition_bank(output: Path, seed: int) -> np.ndarray:
    path = output / "conditions.csv"
    if path.exists():
        frame = pd.read_csv(path)
        conditions = frame[["c0", "c1", "c2"]].to_numpy(dtype=np.float64)
        if conditions.shape != (N_CONDITIONS, COND_DIM):
            raise ValueError(f"existing condition bank has shape {conditions.shape}")
        return conditions
    # Generate a reproducible, space-filling bank.  Drawing a power-of-two Sobol
    # net first and retaining 160 points is more even than IID random conditions.
    engine = qmc.Sobol(d=COND_DIM, scramble=True, seed=seed)
    conditions = engine.random_base2(8)[:N_CONDITIONS]
    pd.DataFrame(conditions, columns=["c0", "c1", "c2"]).to_csv(
        path, index_label="condition_index"
    )
    return conditions


def load_partial(path: Path, restart: bool) -> list[dict]:
    if restart or not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def save_partial(rows: list[dict], path: Path):
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(
            subset=["condition_index", "method"], keep="last"
        ).sort_values(["condition_index", "method"])
    frame.to_csv(path, index=False)


@torch.inference_mode()
def evaluate_one_nf(model, condition: np.ndarray, index: int, args, device):
    torch.manual_seed(args.seed + index)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + index)
        torch.cuda.synchronize()
    started = time.perf_counter()
    parts = []
    remaining = args.nf_evals
    while remaining:
        count = min(remaining, args.gpu_chunk)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)
        x = adaptive.sample_from_model(model, count, cond)
        logq = model.log_prob(x, cond).reshape(-1)
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)
        parts.append((logf - logq).double().cpu().numpy())
        remaining -= count
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    estimate, stderr, ess = common.summarize_log_weights(parts)
    return common.make_row(
        index, condition, "nf_pure", estimate, stderr, ess,
        args.nf_evals, elapsed,
    )


def vegas_job(index, condition, args):
    return (
        index, condition, args.seed + 300_000,
        args.vegas_warmup_nitn, args.vegas_warmup_neval,
        args.vegas_production_nitn, args.vegas_production_neval,
        args.vegas_alpha, args.vegas_min_batch,
    )


def summarize(rows: list[dict], output: Path):
    detail = pd.DataFrame(rows).drop_duplicates(
        subset=["condition_index", "method"], keep="last"
    ).sort_values(["condition_index", "method"])
    summary = detail.groupby("method", sort=False).agg(
        conditions=("condition_index", "count"),
        mean_estimate=("estimate", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        median_absolute_error=("absolute_error", "median"),
        maximum_absolute_error=("absolute_error", "max"),
        rmse=("error", lambda value: math.sqrt(float(np.mean(value * value)))),
        mean_normalized_ess=("normalized_ess", "mean"),
        median_normalized_ess=("normalized_ess", "median"),
        coverage_95=("covered_95", "mean"),
        total_evaluations=("evaluations", "sum"),
        total_condition_seconds=("seconds", "sum"),
    ).reset_index()
    detail.to_csv(output / "condition_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    print("\nCOMPARISON SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    partial_path = args.output / "partial_condition_results.csv"
    conditions = load_or_create_condition_bank(args.output, args.condition_seed)
    rows = load_partial(partial_path, args.restart)
    completed = {
        (int(row["condition_index"]), str(row["method"])) for row in rows
    }

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, _, checkpoint = common.load_nf(args.checkpoint, device)
    print(
        f"NF vs VEGAS | conditions={N_CONDITIONS} | "
        f"NF evals/c={args.nf_evals:,} | workers={args.workers}\n"
        f"shared condition bank={args.output / 'conditions.csv'}\n"
        f"NF edges={sum(len(value) for value in checkpoint['graph'].values())}",
        flush=True,
    )

    missing_nf = [i for i in range(N_CONDITIONS) if (i, "nf_pure") not in completed]
    for done, index in enumerate(missing_nf, 1):
        row = evaluate_one_nf(model, conditions[index], index, args, device)
        rows.append(row)
        save_partial(rows, partial_path)
        print(
            f"NF {done:3d}/{len(missing_nf)} condition={index:3d} "
            f"I={row['estimate']:.7f} ESS={row['normalized_ess']:.4f}",
            flush=True,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    missing_vegas = [i for i in range(N_CONDITIONS) if (i, "vegas") not in completed]
    jobs = [vegas_job(i, conditions[i], args) for i in missing_vegas]
    workers = min(args.workers, len(jobs)) if jobs else 0
    if workers == 1:
        iterator = enumerate((integrate_condition(job) for job in jobs), 1)
        for done, result in iterator:
            row = common.make_row(
                result.index, np.array([result.c0, result.c1, result.c2]), "vegas",
                result.estimate, result.sdev, math.nan, result.total_evaluations,
                result.total_seconds, q_value=result.q_value,
            )
            rows.append(row)
            save_partial(rows, partial_path)
            print(f"VEGAS {done:3d}/{len(jobs)} condition={result.index:3d}", flush=True)
    elif workers:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [pool.submit(integrate_condition, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                row = common.make_row(
                    result.index, np.array([result.c0, result.c1, result.c2]), "vegas",
                    result.estimate, result.sdev, math.nan, result.total_evaluations,
                    result.total_seconds, q_value=result.q_value,
                )
                rows.append(row)
                save_partial(rows, partial_path)
                print(f"VEGAS {done:3d}/{len(jobs)} condition={result.index:3d}", flush=True)

    summarize(rows, args.output)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
