#!/usr/bin/env python3
"""Validate a conditional NF with a certified second-moment safety mixture.

For the normalized sparse-wave-100 benchmark, let r be the same sparse
conditional construction with every truncated-normal scale multiplied by
``kappa >= 1``.  For each conditional coordinate,

    f_j(x)/r_j(x) <= kappa / Z_min,

where Z_min is a uniform lower bound on the target truncation normalizer.
With 98 conditional coordinates and

    q = (1-epsilon) q_NF + epsilon r,

we have the pointwise domination and second-moment certificate

    f/q <= (kappa/Z_min)^98 / epsilon,
    M2 = integral f^2/q <= (kappa/Z_min)^98 / epsilon.

The certified Chebyshev interval therefore has finite-sample coverage without
assuming Gaussian weights or replacing the population M2 by an estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sparse_wave_100d_base as benchmark100  # noqa: E402

benchmark100.register_benchmark()

import run_adaptive_integrand_structure_search as adaptive  # noqa: E402


DIM = 100
COND_DIM = 3
CONDITIONAL_FACTORS = DIM - 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, choices=(32, 100), default=100)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
    )
    parser.add_argument("--conditions", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--evals", type=int, default=100_000)
    parser.add_argument("--chunk", type=int, default=5_000)
    parser.add_argument("--safety-fraction", type=float, default=0.10)
    parser.add_argument("--kappa", type=float, default=1.01)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume completed condition/replicate pairs from run_results.csv.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/sparse_wave_100_certified_validation"),
    )
    return parser.parse_args()


def normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def normal_icdf(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    value = value.clamp(eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * value - 1.0)


def truncated_normal_logpdf(x, mu, sigma):
    low = normal_cdf(-mu / sigma)
    high = normal_cdf((1.0 - mu) / sigma)
    log_z = torch.log((high - low).clamp_min(torch.finfo(x.dtype).tiny))
    z = (x - mu) / sigma
    return (
        -0.5 * z.square()
        - math.log(math.sqrt(2.0 * math.pi))
        - torch.log(sigma)
        - log_z
    )


def sample_truncated_normal(mu, sigma):
    low = normal_cdf(-mu / sigma)
    high = normal_cdf((1.0 - mu) / sigma)
    u = low + torch.rand_like(mu) * (high - low)
    return (mu + sigma * normal_icdf(u)).clamp(0.0, 1.0)


def benchmark_mu_sigma(child, x, condition):
    if DIM == 100:
        return benchmark100.sparse_wave_100_mu_sigma(child, x, condition)
    if DIM == 32:
        return adaptive.sparse_wave_32_mu_sigma(child, x, condition)
    raise RuntimeError(f"unsupported dimension {DIM}")


def benchmark_log_integrand(x, condition):
    if DIM == 100:
        return benchmark100.sparse_wave_100_log_integrand(x, condition)
    if DIM == 32:
        return adaptive.sparse_wave_32_log_integrand(x, condition)
    raise RuntimeError(f"unsupported dimension {DIM}")


@torch.inference_mode()
def sample_safety(condition: torch.Tensor, kappa: float):
    """Sample from and evaluate the analytically controlled safety density."""
    n = condition.shape[0]
    x = torch.empty(n, DIM, device=condition.device, dtype=condition.dtype)
    x[:, :2] = torch.rand(n, 2, device=x.device, dtype=x.dtype)
    log_r = torch.zeros(n, device=x.device, dtype=x.dtype)
    for child in range(2, DIM):
        mu, sigma = benchmark_mu_sigma(child, x, condition)
        broad_sigma = kappa * sigma
        value = sample_truncated_normal(mu, broad_sigma)
        x[:, child:child + 1] = value
        log_r += truncated_normal_logpdf(value, mu, broad_sigma).reshape(-1)
    return x, log_r


@torch.inference_mode()
def safety_log_prob(x, condition, kappa: float):
    result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for child in range(2, DIM):
        mu, sigma = benchmark_mu_sigma(child, x, condition)
        result += truncated_normal_logpdf(
            x[:, child:child + 1], mu, kappa * sigma
        ).reshape(-1)
    inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def certified_second_moment_bound(kappa: float, epsilon: float):
    # From the benchmark definition: mu in [0.24,0.76] and target sigma in
    # [0.055,0.085].  The smallest truncation mass occurs at an extreme mean
    # and the largest scale.  Safety normalizers are at most one.
    mu_edge = 0.30 if DIM == 100 else 0.25
    sigma_max = 0.166 if DIM == 100 else 0.084
    z_min = norm.cdf((1.0 - mu_edge) / sigma_max) - norm.cdf(
        -mu_edge / sigma_max
    )
    # The safety conditional has the same mean and a broader scale.  Its
    # truncation mass is no larger than the target truncation mass, hence at
    # the common mode f_j/r_j = kappa * Z_r/Z_f <= kappa; away from the mode
    # the Gaussian exponential ratio only decreases.  Thus kappa is a uniform
    # pointwise ratio bound for every conditional factor.
    per_factor_ratio = kappa
    log_domination = (
        CONDITIONAL_FACTORS * math.log(per_factor_ratio) - math.log(epsilon)
    )
    return math.exp(log_domination), z_min, per_factor_ratio


@torch.inference_mode()
def draw_logweights(model, condition, args, seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    pieces = []
    remaining = args.evals
    log_nf_fraction = math.log1p(-args.safety_fraction)
    log_safety_fraction = math.log(args.safety_fraction)
    while remaining:
        count = min(remaining, args.chunk)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)
        use_safety = torch.rand(count, device=device) < args.safety_fraction
        x = adaptive.sample_from_model(model, count, cond)
        safety_count = int(use_safety.sum().item())
        if safety_count:
            safety_x, _ = sample_safety(cond[use_safety], args.kappa)
            x[use_safety] = safety_x
        log_nf = model.log_prob(x, cond).reshape(-1)
        log_safety = safety_log_prob(x, cond, args.kappa)
        log_q = torch.logaddexp(
            log_nf + log_nf_fraction,
            log_safety + log_safety_fraction,
        )
        log_f = benchmark_log_integrand(x, cond)
        pieces.append((log_f - log_q).double().cpu())
        remaining -= count
    return torch.cat(pieces).numpy()


def moments_from_logweights(logw: np.ndarray):
    maximum = float(np.max(logw))
    scaled = np.exp(logw - maximum)
    mean_w = math.exp(maximum) * float(scaled.mean())
    mean_w2 = math.exp(2.0 * maximum) * float(np.square(scaled).mean())
    variance_mean = max(mean_w2 - mean_w * mean_w, 0.0) / len(logw)
    return mean_w, mean_w2, math.sqrt(variance_mean)


def one_sided_binomial_lower(successes: int, trials: int, level=0.95):
    if successes == 0:
        return 0.0
    return float(beta_distribution.ppf(
        1.0 - level, successes, trials - successes + 1
    ))


def load_nf(path: Path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    expected = benchmark100.BENCHMARK if DIM == 100 else "sparse_wave_32"
    if checkpoint.get("benchmark") != expected:
        raise ValueError(
            f"checkpoint benchmark is {checkpoint.get('benchmark')!r}, "
            f"expected {expected!r}"
        )
    adaptive.DIM = DIM
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def main() -> None:
    global DIM, CONDITIONAL_FACTORS
    args = parse_args()
    DIM = args.dimension
    CONDITIONAL_FACTORS = DIM - 2
    if args.checkpoint is None:
        args.checkpoint = (
            Path("results/sparse_wave_100_adaptive/sparse_wave_100_final_model.pt")
            if DIM == 100 else
            Path("results/adaptive_integrand_all_weak/sparse_wave_32_final_model.pt")
        )
    if not 0.0 < args.safety_fraction < 1.0:
        raise ValueError("safety fraction must lie in (0,1)")
    if args.kappa < 1.0:
        raise ValueError("kappa must be at least one for the stated certificate")
    if not 0.0 < args.coverage < 1.0:
        raise ValueError("coverage must lie in (0,1)")

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    adaptive.DIM = DIM
    model, checkpoint = load_nf(args.checkpoint, device)
    model.eval()

    u2, z_min, per_factor = certified_second_moment_bound(
        args.kappa, args.safety_fraction
    )
    alpha = 1.0 - args.coverage
    certified_radius = math.sqrt(u2 / (args.evals * alpha))
    z = float(norm.ppf(1.0 - alpha / 2.0))

    rng = np.random.default_rng(args.seed)
    conditions = rng.random((args.conditions, COND_DIM))
    results_path = args.output / "run_results.csv"
    existing = (
        pd.read_csv(results_path)
        if args.resume and results_path.exists() else pd.DataFrame()
    )
    completed_keys = set()
    if not existing.empty:
        completed_keys = set(zip(
            existing["condition_index"].astype(int),
            existing["replicate"].astype(int),
        ))
        print(f"resuming {len(completed_keys)} completed runs", flush=True)
    elif results_path.exists():
        results_path.unlink()

    total = args.conditions * args.replicates
    completed = len(completed_keys)
    for condition_index, condition in enumerate(conditions):
        for replicate in range(args.replicates):
            if (condition_index, replicate) in completed_keys:
                continue
            seed = args.seed + 1_000_003 * condition_index + 10_007 * replicate
            logw = draw_logweights(model, condition, args, seed, device)
            estimate, empirical_m2, ordinary_se = moments_from_logweights(logw)
            empirical_radius = math.sqrt(
                empirical_m2 / (args.evals * alpha)
            )
            centered_radius = math.sqrt(
                max(empirical_m2 - estimate * estimate, 0.0)
                / (args.evals * alpha)
            )
            normal_radius = z * ordinary_se
            row = {
                "condition_index": condition_index,
                "replicate": replicate,
                "seed": seed,
                "c0": condition[0], "c1": condition[1], "c2": condition[2],
                "evaluations": args.evals,
                "estimate": estimate,
                "empirical_second_moment": empirical_m2,
                "ordinary_standard_error": ordinary_se,
                "certified_u2": u2,
                "certified_radius": certified_radius,
                "empirical_m2_radius": empirical_radius,
                "empirical_centered_radius": centered_radius,
                "normal_radius": normal_radius,
            }
            for name in (
                "certified", "empirical_m2", "empirical_centered", "normal"
            ):
                radius = row[f"{name}_radius"]
                row[f"{name}_covered"] = abs(estimate - 1.0) <= radius
            pd.DataFrame([row]).to_csv(
                results_path,
                mode="a",
                header=not results_path.exists(),
                index=False,
            )
            completed += 1
            print(
                f"{completed}/{total} c={condition_index:3d} r={replicate:3d} "
                f"I={estimate:.7f} M2={empirical_m2:.5g} "
                f"cert95=+/-{certified_radius:.4g}",
                flush=True,
            )

    raw = pd.read_csv(results_path)
    summaries = []
    for name in ("certified", "empirical_m2", "empirical_centered", "normal"):
        covered = raw[f"{name}_covered"]
        successes = int(covered.sum())
        summaries.append({
            "method": name,
            "nominal_coverage": args.coverage,
            "runs": len(raw),
            "covered_runs": successes,
            "empirical_coverage": successes / len(raw),
            "one_sided_95_lower_coverage": one_sided_binomial_lower(
                successes, len(raw)
            ),
            "mean_halfwidth": float(raw[f"{name}_radius"].mean()),
            "maximum_halfwidth": float(raw[f"{name}_radius"].max()),
            "maximum_absolute_error": float((raw["estimate"] - 1.0).abs().max()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "benchmark": checkpoint.get("benchmark"),
            "dimension": DIM,
            "true_integral": 1.0,
            "conditions": args.conditions,
            "replicates": args.replicates,
            "evaluations": args.evals,
            "safety_fraction": args.safety_fraction,
            "kappa": args.kappa,
            "target_truncation_z_min": z_min,
            "per_factor_domination_bound": per_factor,
            "conditional_factor_count": CONDITIONAL_FACTORS,
            "certified_u2": u2,
            "coverage": args.coverage,
            "base_checkpoint_benchmark": checkpoint.get("benchmark"),
        }, stream, indent=2)

    print("\nCERTIFIED 100D COVERAGE")
    print(f"Z_min                 = {z_min:.12g}")
    print(f"per-factor ratio      = {per_factor:.12g}")
    print(f"certified U2          = {u2:.12g}")
    print(f"certified half-width  = {certified_radius:.12g}")
    print(summary.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
