#!/usr/bin/env python3
"""CPU VEGAS baseline for the 32D ``sparse_wave_32`` benchmark.

The target sampler is never used.  For every condition c, VEGAS receives only
the callable f(x | c) on [0, 1]^32.  The exact integral (1) is used exclusively
afterwards to report error.

Independent conditions are run in separate processes.  Within a process the
integrand is evaluated in vectorized batches, which is normally faster than
nested multiprocessing inside VEGAS.
"""

from __future__ import annotations

import os

# Prevent NumPy/BLAS from creating another thread pool inside every worker.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import gvar as gv
import numpy as np
import vegas
from scipy.special import ndtr


DIM = 32
CONDITION_DIM = 3
TRUE_INTEGRAL = 1.0
LOG_2PI = math.log(2.0 * math.pi)


class SparseWave32(vegas.LBatchIntegrand):
    """Vectorized implementation matching ``sparse_wave_32`` exactly."""

    def __init__(self, condition: np.ndarray):
        super().__init__()
        condition = np.asarray(condition, dtype=np.float64)
        if condition.shape != (CONDITION_DIM,):
            raise ValueError(f"condition must have shape ({CONDITION_DIM},)")
        self.c0, self.c1, self.c2 = map(float, condition)
        self.evaluation_count = 0

    def log_value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != DIM:
            raise ValueError(f"x must have shape (batch, {DIM}); received {x.shape}")
        log_f = np.zeros(x.shape[0], dtype=np.float64)
        gate = 1.0 / (1.0 + math.exp(-6.0 * (self.c0 - 0.5)))

        # x_0 and x_1 are uniform roots, so their log densities are zero.
        for child in range(2, DIM):
            parent = child - 1
            skip = max(0, child // 2 - 1)
            a = x[:, parent]
            b = x[:, skip]

            phase = 0.031 * child + 0.10 * self.c1
            local_wave = np.sin(2.0 * math.pi * (a + phase))
            skip_wave = np.cos(2.0 * math.pi * (b - 0.07 * self.c0))
            interaction = np.sin(2.0 * math.pi * (a - b))
            signal = (
                gate * local_wave
                + (1.0 - gate) * skip_wave
                + 0.15 * interaction
            )
            mu = 0.50 + 0.25 * np.tanh(1.35 * signal)
            sigma = 0.060 + 0.018 * self.c2 + 0.003 * float(child % 3)

            value = x[:, child]
            z = (value - mu) / sigma
            log_base = -0.5 * z * z - math.log(sigma) - 0.5 * LOG_2PI
            lower = (0.0 - mu) / sigma
            upper = (1.0 - mu) / sigma
            # The 1e-12 floor deliberately matches the NF benchmark.
            normalizer = np.maximum(ndtr(upper) - ndtr(lower), 1.0e-12)
            log_f += log_base - np.log(normalizer)

        return log_f

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self.evaluation_count += x.shape[0]
        # Underflow to zero is harmless and much cheaper than log-space VEGAS.
        with np.errstate(under="ignore"):
            return np.exp(self.log_value(x))


def scalar_log_integrand(x: np.ndarray, condition: np.ndarray) -> float:
    """Independent scalar reference used only to catch implementation errors."""
    c0, c1, c2 = map(float, condition)
    gate = 1.0 / (1.0 + math.exp(-6.0 * (c0 - 0.5)))
    log_f = 0.0
    for child in range(2, DIM):
        parent = child - 1
        skip = max(0, child // 2 - 1)
        a = float(x[parent])
        b = float(x[skip])
        phase = 0.031 * child + 0.10 * c1
        signal = (
            gate * math.sin(2.0 * math.pi * (a + phase))
            + (1.0 - gate) * math.cos(2.0 * math.pi * (b - 0.07 * c0))
            + 0.15 * math.sin(2.0 * math.pi * (a - b))
        )
        mu = 0.50 + 0.25 * math.tanh(1.35 * signal)
        sigma = 0.060 + 0.018 * c2 + 0.003 * float(child % 3)
        z = (float(x[child]) - mu) / sigma
        lower = (0.0 - mu) / sigma
        upper = (1.0 - mu) / sigma
        phi_lower = 0.5 * (1.0 + math.erf(lower / math.sqrt(2.0)))
        phi_upper = 0.5 * (1.0 + math.erf(upper / math.sqrt(2.0)))
        normalizer = max(phi_upper - phi_lower, 1.0e-12)
        log_f += (
            -0.5 * z * z - math.log(sigma) - 0.5 * LOG_2PI
            - math.log(normalizer)
        )
    return log_f


def validate_integrand(conditions: np.ndarray, seed: int, points: int) -> float:
    """Compare batch and independent scalar log-integrands before integration."""
    rng = np.random.default_rng(seed)
    worst_error = 0.0
    for condition in conditions[: min(len(conditions), 4)]:
        x = rng.random((points, DIM), dtype=np.float64)
        batch_values = SparseWave32(condition).log_value(x)
        scalar_values = np.asarray(
            [scalar_log_integrand(point, condition) for point in x]
        )
        worst_error = max(
            worst_error, float(np.max(np.abs(batch_values - scalar_values)))
        )
    if not np.isfinite(worst_error) or worst_error > 1.0e-9:
        raise RuntimeError(
            f"integrand validation failed: maximum log-value error={worst_error:.3e}"
        )
    return worst_error


@dataclass
class ConditionResult:
    index: int
    c0: float
    c1: float
    c2: float
    estimate: float
    sdev: float
    abs_error: float
    rel_error: float
    z_score: float
    q_value: float
    warmup_evaluations: int
    production_evaluations: int
    total_evaluations: int
    warmup_seconds: float
    production_seconds: float
    total_seconds: float


def integrate_condition(job: tuple) -> ConditionResult:
    (
        index,
        condition,
        seed,
        warmup_nitn,
        warmup_neval,
        production_nitn,
        production_neval,
        alpha,
        min_batch,
    ) = job

    # VEGAS uses gvar's generator for its random numbers.
    gv.ranseed(seed + index)
    integrand = SparseWave32(condition)
    integrator = vegas.Integrator(
        [[0.0, 1.0]] * DIM,
        alpha=alpha,
        min_neval_batch=min_batch,
        nproc=1,
    )

    started = time.perf_counter()
    integrator(integrand, nitn=warmup_nitn, neval=warmup_neval)
    warmup_done = time.perf_counter()
    warmup_evaluations = integrand.evaluation_count

    # Freeze the trained grid so the final uncertainty comes from identically
    # distributed production iterations rather than a changing proposal.
    result = integrator(
        integrand,
        nitn=production_nitn,
        neval=production_neval,
        adapt=False,
    )
    finished = time.perf_counter()
    total_evaluations = integrand.evaluation_count
    production_evaluations = total_evaluations - warmup_evaluations

    estimate = float(result.mean)
    sdev = float(result.sdev)
    error = estimate - TRUE_INTEGRAL
    return ConditionResult(
        index=index,
        c0=float(condition[0]),
        c1=float(condition[1]),
        c2=float(condition[2]),
        estimate=estimate,
        sdev=sdev,
        abs_error=abs(error),
        rel_error=abs(error) / TRUE_INTEGRAL,
        z_score=error / sdev if sdev > 0.0 else math.nan,
        q_value=float(getattr(result, "Q", math.nan)),
        warmup_evaluations=warmup_evaluations,
        production_evaluations=production_evaluations,
        total_evaluations=total_evaluations,
        warmup_seconds=warmup_done - started,
        production_seconds=finished - warmup_done,
        total_seconds=finished - started,
    )


def load_conditions(args: argparse.Namespace) -> np.ndarray:
    if args.condition is not None:
        return np.asarray([args.condition], dtype=np.float64)
    if args.conditions_file is None:
        return np.random.default_rng(args.condition_seed).random(
            (args.n_conditions, CONDITION_DIM), dtype=np.float64
        )

    values = []
    with args.conditions_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"c0", "c1", "c2"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("conditions CSV must contain columns c0,c1,c2")
        for row in reader:
            values.append([float(row["c0"]), float(row["c1"]), float(row["c2"])])
    if not values:
        raise ValueError("conditions CSV is empty")
    return np.asarray(values, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-throughput CPU VEGAS baseline for sparse_wave_32."
    )
    conditions = parser.add_mutually_exclusive_group()
    conditions.add_argument(
        "--condition", nargs=3, type=float, metavar=("C0", "C1", "C2")
    )
    conditions.add_argument("--conditions-file", type=Path)
    parser.add_argument("--n-conditions", type=int, default=32)
    parser.add_argument("--condition-seed", type=int, default=20260911)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--warmup-nitn", type=int, default=10)
    parser.add_argument("--warmup-neval", type=int, default=1_000_000)
    parser.add_argument("--production-nitn", type=int, default=10)
    parser.add_argument("--production-neval", type=int, default=1_000_000)
    parser.add_argument(
        "--alpha", type=float, default=0.30,
        help="VEGAS adaptation damping; lower is steadier for sharp targets.",
    )
    parser.add_argument("--min-batch", type=int, default=100_000)
    parser.add_argument(
        "--validation-points", type=int, default=64,
        help="Scalar/batch agreement checks per condition before VEGAS.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/vegas_sparse_wave_32_cpu.csv"),
    )
    args = parser.parse_args()
    if args.n_conditions < 1 or args.workers < 1:
        parser.error("--n-conditions and --workers must be positive")
    for name in ("warmup_nitn", "warmup_neval", "production_nitn", "production_neval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.validation_points < 1:
        parser.error("--validation-points must be positive")
    return args


def main() -> None:
    args = parse_args()
    conditions = load_conditions(args)
    if np.any((conditions < 0.0) | (conditions > 1.0)):
        raise ValueError("all condition coordinates must lie in [0,1]")
    validation_error = validate_integrand(
        conditions, args.seed + 10_000_000, args.validation_points
    )

    worker_count = min(args.workers, len(conditions))
    jobs = [
        (
            i,
            condition,
            args.seed,
            args.warmup_nitn,
            args.warmup_neval,
            args.production_nitn,
            args.production_neval,
            args.alpha,
            args.min_batch,
        )
        for i, condition in enumerate(conditions)
    ]

    print("=" * 88, flush=True)
    print("CPU VEGAS: sparse_wave_32 (f(x|c) only)", flush=True)
    print(f"conditions              = {len(conditions)}", flush=True)
    print(f"parallel worker processes = {worker_count}", flush=True)
    print(f"warmup                  = {args.warmup_nitn} x {args.warmup_neval:,}", flush=True)
    print(
        f"production              = {args.production_nitn} x "
        f"{args.production_neval:,} (grid frozen)", flush=True
    )
    print(f"adaptation alpha         = {args.alpha}", flush=True)
    print(
        f"integrand validation     = PASS (max |delta log f|={validation_error:.3e})",
        flush=True,
    )
    print("true integral            = hidden during integration", flush=True)
    print("=" * 88, flush=True)

    wall_started = time.perf_counter()
    results: list[ConditionResult] = []
    if worker_count == 1:
        for job in jobs:
            result = integrate_condition(job)
            results.append(result)
            print(
                f"[{result.index + 1:3d}/{len(jobs)}] "
                f"I={result.estimate:.8g} +/- {result.sdev:.2g}  "
                f"|err|={result.abs_error:.3g}  Q={result.q_value:.3f}  "
                f"evals={result.total_evaluations:,}  time={result.total_seconds:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(integrate_condition, job): job[0] for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[{len(results):3d}/{len(jobs)}] condition {result.index:3d}  "
                    f"I={result.estimate:.8g} +/- {result.sdev:.2g}  "
                    f"|err|={result.abs_error:.3g}  Q={result.q_value:.3f}  "
                    f"evals={result.total_evaluations:,}  time={result.total_seconds:.1f}s",
                    flush=True,
                )

    wall_seconds = time.perf_counter() - wall_started
    results.sort(key=lambda item: item.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    estimates = np.asarray([result.estimate for result in results])
    sdevs = np.asarray([result.sdev for result in results])
    abs_errors = np.abs(estimates - TRUE_INTEGRAL)
    covered = abs_errors <= 1.96 * sdevs
    requested_evaluations_per_condition = (
        args.warmup_nitn * args.warmup_neval
        + args.production_nitn * args.production_neval
    )
    actual_warmup_evaluations = sum(r.warmup_evaluations for r in results)
    actual_production_evaluations = sum(r.production_evaluations for r in results)
    actual_evaluations = sum(r.total_evaluations for r in results)

    print("\nSUMMARY", flush=True)
    print(f"mean estimate            = {estimates.mean():.10g}", flush=True)
    print(f"mean absolute error      = {abs_errors.mean():.6g}", flush=True)
    print(f"median absolute error    = {np.median(abs_errors):.6g}", flush=True)
    print(f"maximum absolute error   = {abs_errors.max():.6g}", flush=True)
    print(f"RMSE                     = {np.sqrt(np.mean((estimates - 1.0) ** 2)):.6g}", flush=True)
    print(f"mean VEGAS uncertainty   = {sdevs.mean():.6g}", flush=True)
    print(f"95% interval coverage    = {covered.mean():.1%}", flush=True)
    print(f"requested evaluations/c  = {requested_evaluations_per_condition:,}", flush=True)
    print(
        f"requested evaluations    = "
        f"{requested_evaluations_per_condition * len(results):,}", flush=True
    )
    print(f"actual warmup evaluations = {actual_warmup_evaluations:,}", flush=True)
    print(f"actual production evals   = {actual_production_evaluations:,}", flush=True)
    print(f"actual total evaluations  = {actual_evaluations:,}", flush=True)
    print(
        f"actual/requested ratio    = "
        f"{actual_evaluations / (requested_evaluations_per_condition * len(results)):.6f}",
        flush=True,
    )
    print(f"wall time                = {wall_seconds:.1f}s", flush=True)
    print(f"CSV                      = {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
