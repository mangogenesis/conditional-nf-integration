#!/usr/bin/env python3
"""Validate exact base/energy-bomb-repair mixtures on fresh 100D samples."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_irregular_mixture_100d as irregular  # noqa: E402
from explore_energy_bombs_irregular_100d import load_proposal  # noqa: E402


adaptive = irregular.adaptive


def load_repair(path, device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["repair_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def sample_exact_mixture(base, repair, cond, alpha):
    n = cond.shape[0]
    if alpha <= 0:
        return base.sample(n, cond)
    if alpha >= 1:
        return repair.sample(n, cond)
    use_repair = torch.rand(n, device=cond.device) < alpha
    x = base.sample(n, cond)
    if use_repair.any():
        x[use_repair] = repair.sample(int(use_repair.sum().item()), cond[use_repair])
    return x


@torch.no_grad()
def mixture_log_prob(base, repair, x, cond, alpha):
    log_base = base.log_prob(x, cond).reshape(-1)
    if alpha <= 0:
        return log_base
    log_repair = repair.log_prob(x, cond).reshape(-1)
    if alpha >= 1:
        return log_repair
    return torch.logaddexp(
        log_base + math.log1p(-alpha), log_repair + math.log(alpha)
    )


@torch.no_grad()
def evaluate_run(base, repair, condition, alpha, evaluations, chunk):
    sum_w = torch.zeros((), dtype=torch.float64, device=condition.device)
    sum_w2 = torch.zeros_like(sum_w)
    maximum = torch.full((), -torch.inf, dtype=torch.float64, device=condition.device)
    count = 0
    while count < evaluations:
        size = min(chunk, evaluations - count)
        cond = condition.expand(size, -1)
        x = sample_exact_mixture(base, repair, cond, alpha)
        log_f = irregular.irregular_mixture_100_log_integrand(x, cond).double()
        log_q = mixture_log_prob(base, repair, x, cond, alpha).double()
        weights = torch.exp(log_f - log_q)
        sum_w += weights.sum()
        sum_w2 += weights.square().sum()
        maximum = torch.maximum(maximum, weights.max())
        count += size
    estimate = sum_w / evaluations
    m2 = sum_w2 / evaluations
    variance = torch.clamp(m2 - estimate.square(), min=0.0)
    stderr = torch.sqrt(variance / evaluations)
    return {
        "estimate": float(estimate.item()),
        "stderr": float(stderr.item()),
        "second_moment": float(m2.item()),
        "normalized_ess": float((estimate.square() / m2.clamp_min(1e-300)).item()),
        "maximum_weight_share": float((maximum / sum_w.clamp_min(1e-300)).item()),
        "covered_95_normal": bool(abs(float(estimate.item()) - 1.0) <= 1.96 * float(stderr.item())),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--repair-checkpoint", type=Path, required=True)
    parser.add_argument("--mixture-fractions", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--conditions", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--evals", type=int, default=100000)
    parser.add_argument("--chunk", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=92831)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("results/irregular_mixture_100_bomb_repair_validation"))
    return parser.parse_args()


def main():
    args = parse_args()
    if any(alpha < 0 or alpha > 1 for alpha in args.mixture_fractions):
        raise ValueError("mixture fractions must lie in [0,1]")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    irregular.register_benchmark()
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)
    base, _ = load_proposal(args.base_checkpoint, device)
    repair, _ = load_repair(args.repair_checkpoint, device)
    conditions = adaptive.sample_conditions(args.conditions, device)
    rows = []
    total = len(args.mixture_fractions) * args.conditions * args.replicates
    finished = 0
    for alpha in args.mixture_fractions:
        for condition_index, condition in enumerate(conditions):
            for replicate in range(args.replicates):
                adaptive.seed_everything(
                    args.seed + 100000 * int(round(alpha * 1000))
                    + 1000 * condition_index + replicate
                )
                result = evaluate_run(
                    base, repair, condition[None, :], alpha, args.evals, args.chunk
                )
                rows.append({
                    "mixture_fraction": alpha,
                    "condition_index": condition_index,
                    "replicate": replicate,
                    "evaluations": args.evals,
                    **result,
                })
                finished += 1
                print(
                    f"{finished}/{total} alpha={alpha:.3f} c={condition_index:02d} "
                    f"r={replicate} I={result['estimate']:.7g} "
                    f"M2={result['second_moment']:.6g} "
                    f"ESS={result['normalized_ess']:.4g}"
                )
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby("mixture_fraction", as_index=False)
        .agg(
            runs=("estimate", "size"),
            mean_estimate=("estimate", "mean"),
            empirical_sd=("estimate", "std"),
            mean_reported_se=("stderr", "mean"),
            coverage_95=("covered_95_normal", "mean"),
            median_second_moment=("second_moment", "median"),
            q95_second_moment=("second_moment", lambda x: x.quantile(0.95)),
            maximum_second_moment=("second_moment", "max"),
            median_normalized_ess=("normalized_ess", "median"),
            minimum_normalized_ess=("normalized_ess", "min"),
            maximum_weight_share=("maximum_weight_share", "max"),
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output / "run_results.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    print("\nENERGY-BOMB REPAIR VALIDATION")
    print(summary.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
