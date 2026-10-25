#!/usr/bin/env python3
"""Mass-aware stochastic energy bombs via annealed SMC in 100 dimensions.

The bridge is

    gamma_beta(x|c) = q0(x|c) ** (1-beta) * f(x|c) ** beta,

so Z_0=1 and Z_1 is the desired integral.  Adaptive incremental weights carry
probability mass between temperatures.  After every resampling step, a
Metropolis-adjusted Langevin kernel in logit coordinates rejuvenates the
particles while preserving the current bridge distribution.
"""

from __future__ import annotations

import argparse
import json
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
from explore_energy_bombs_irregular_100d import load_proposal, logit  # noqa: E402


adaptive = irregular.adaptive


@torch.no_grad()
def log_f_and_q(base, x, cond):
    log_f = irregular.irregular_mixture_100_log_integrand(x, cond).double()
    log_q = base.log_prob(x, cond).reshape(-1).double()
    return log_f, log_q


def normalized_ess_from_log_weights(log_weight):
    normalized = torch.softmax(log_weight.double(), dim=0)
    return float((1.0 / (normalized.numel() * normalized.square().sum())).item())


def choose_next_beta(beta, log_ratio, target_ess, bisection_steps=24):
    """Largest beta whose incremental normalized ESS meets the target."""
    if beta >= 1.0:
        return 1.0
    remaining = 1.0 - beta
    if normalized_ess_from_log_weights(remaining * log_ratio) >= target_ess:
        return 1.0
    low = 0.0
    high = remaining
    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        if normalized_ess_from_log_weights(middle * log_ratio) >= target_ess:
            low = middle
        else:
            high = middle
    return beta + max(low, 1e-8)


@torch.no_grad()
def systematic_resample(log_weight):
    n = log_weight.numel()
    probabilities = torch.softmax(log_weight.double(), dim=0)
    cumulative = probabilities.cumsum(0)
    start = torch.rand((), device=log_weight.device, dtype=torch.float64) / n
    positions = start + torch.arange(n, device=log_weight.device, dtype=torch.float64) / n
    return torch.searchsorted(cumulative, positions).clamp_max(n - 1)


def bridge_log_density_and_gradient(base, y, cond, beta, gradient_clip):
    """Evaluate log gamma_beta in logit coordinates and its y-gradient."""
    y = y.detach().requires_grad_(True)
    x = torch.sigmoid(y)
    log_f = irregular.irregular_mixture_100_log_integrand(x, cond).reshape(-1)
    log_q = base.log_prob(x, cond).reshape(-1)
    log_jacobian = (
        torch.nn.functional.logsigmoid(y)
        + torch.nn.functional.logsigmoid(-y)
    ).sum(dim=1)
    log_pi = (1.0 - beta) * log_q + beta * log_f + log_jacobian
    gradient = torch.autograd.grad(log_pi.sum(), y)[0]
    norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gradient = gradient * torch.clamp(gradient_clip / norm, max=1.0)
    return log_pi.detach().double(), gradient.detach()


def mala_step(base, y, cond, beta, step_size, gradient_clip, chunk):
    """One chunked MALA transition with the exact asymmetric correction."""
    outputs = []
    acceptances = []
    for start in range(0, y.shape[0], chunk):
        current = y[start:start + chunk]
        current_cond = cond[start:start + chunk]
        log_current, gradient_current = bridge_log_density_and_gradient(
            base, current, current_cond, beta, gradient_clip
        )
        mean_forward = current + 0.5 * step_size * gradient_current
        proposed = mean_forward + math.sqrt(step_size) * torch.randn_like(current)
        log_proposed, gradient_proposed = bridge_log_density_and_gradient(
            base, proposed, current_cond, beta, gradient_clip
        )
        mean_reverse = proposed + 0.5 * step_size * gradient_proposed
        log_forward = -(proposed - mean_forward).double().square().sum(dim=1) / (2.0 * step_size)
        log_reverse = -(current - mean_reverse).double().square().sum(dim=1) / (2.0 * step_size)
        log_acceptance = log_proposed - log_current + log_reverse - log_forward
        accepted = torch.log(torch.rand_like(log_acceptance)) < torch.minimum(
            log_acceptance, torch.zeros_like(log_acceptance)
        )
        outputs.append(torch.where(accepted[:, None], proposed, current).detach())
        acceptances.append(accepted.double())
    return torch.cat(outputs), float(torch.cat(acceptances).mean().item())


def run_smc_condition(base, condition, particles, target_ess, mala_steps,
                      mala_step_size, gradient_clip, chunk, max_stages):
    cond = condition.expand(particles, -1)
    with torch.no_grad():
        x = base.sample(particles, cond)
    y = logit(x)
    beta = 0.0
    log_z = 0.0
    stages = []
    current_step_size = mala_step_size

    for stage in range(max_stages):
        with torch.no_grad():
            x = torch.sigmoid(y)
            log_f, log_q = log_f_and_q(base, x, cond)
            log_ratio = log_f - log_q
            next_beta = choose_next_beta(beta, log_ratio, target_ess)
            delta = next_beta - beta
            incremental = delta * log_ratio
            log_increment = torch.logsumexp(incremental, dim=0) - math.log(particles)
            log_z += float(log_increment.item())
            normalized_ess = normalized_ess_from_log_weights(incremental)
            indices = systematic_resample(incremental)
            y = y[indices]
            cond = cond[indices]

        acceptance_values = []
        for _ in range(mala_steps):
            y, acceptance = mala_step(
                base, y, cond, next_beta, current_step_size,
                gradient_clip, chunk,
            )
            acceptance_values.append(acceptance)
        mean_acceptance = (
            sum(acceptance_values) / len(acceptance_values)
            if acceptance_values else float("nan")
        )
        stages.append({
            "stage": stage,
            "beta_previous": beta,
            "beta": next_beta,
            "delta_beta": delta,
            "incremental_normalized_ess": normalized_ess,
            "log_z_increment": float(log_increment.item()),
            "log_z": log_z,
            "mala_acceptance": mean_acceptance,
            "mala_step_size": current_step_size,
            "log_ratio_median": float(log_ratio.median().item()),
            "log_ratio_maximum": float(log_ratio.max().item()),
        })
        print(
            f"  stage {stage:03d}: beta={next_beta:.6f} delta={delta:.3g} "
            f"ESS={normalized_ess:.3f} logZ={log_z:.6f} "
            f"MALA={mean_acceptance:.3f} h={current_step_size:.3g}"
        )

        # Adapt only between invariant MALA kernels, never inside acceptance.
        if acceptance_values:
            if mean_acceptance < 0.45:
                current_step_size *= 0.75
            elif mean_acceptance > 0.75:
                current_step_size *= 1.25
            current_step_size = min(max(current_step_size, 1e-6), 0.05)
        beta = next_beta
        if beta >= 1.0 - 1e-12:
            return log_z, torch.sigmoid(y).detach(), stages
    raise RuntimeError(
        f"SMC did not reach beta=1 after {max_stages} stages; final beta={beta}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditions", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--target-ess", type=float, default=0.70)
    parser.add_argument("--mala-steps", type=int, default=2)
    parser.add_argument("--mala-step-size", type=float, default=5e-4)
    parser.add_argument("--gradient-clip", type=float, default=50.0)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--max-stages", type=int, default=250)
    parser.add_argument("--keep-particles", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("results/irregular_mixture_100_mass_aware_bombs"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.target_ess < 1.0:
        raise ValueError("target ESS must lie in (0,1)")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    irregular.register_benchmark()
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)
    base, checkpoint = load_proposal(args.checkpoint, device)
    conditions = adaptive.sample_conditions(args.conditions, device)
    run_rows = []
    stage_rows = []
    saved_particles = []

    for condition_index, condition in enumerate(conditions):
        for replicate in range(args.replicates):
            seed = args.seed + 1000 * condition_index + replicate
            adaptive.seed_everything(seed)
            print(f"\nCONDITION {condition_index}/{args.conditions - 1} REPLICATE {replicate}")
            log_z, final_particles, stages = run_smc_condition(
                base, condition[None, :], args.particles, args.target_ess,
                args.mala_steps, args.mala_step_size, args.gradient_clip,
                args.chunk, args.max_stages,
            )
            estimate = math.exp(log_z)
            run_rows.append({
                "condition_index": condition_index,
                "replicate": replicate,
                "seed": seed,
                "particles": args.particles,
                "stages": len(stages),
                "log_integral_estimate": log_z,
                "integral_estimate": estimate,
                "absolute_error": abs(estimate - 1.0),
                "minimum_incremental_normalized_ess": min(
                    row["incremental_normalized_ess"] for row in stages
                ),
                "mean_mala_acceptance": sum(
                    row["mala_acceptance"] for row in stages
                ) / len(stages),
            })
            for row in stages:
                stage_rows.append({
                    "condition_index": condition_index,
                    "replicate": replicate,
                    **row,
                })
            keep = min(args.keep_particles, final_particles.shape[0])
            selected = torch.randperm(final_particles.shape[0], device=device)[:keep]
            saved_particles.append({
                "condition": condition.detach().cpu(),
                "condition_index": condition_index,
                "replicate": replicate,
                "x": final_particles[selected].cpu(),
            })
            print(f"  FINAL: I={estimate:.8g} logI={log_z:.6g} stages={len(stages)}")

    runs = pd.DataFrame(run_rows)
    summary = {
        "runs": len(runs),
        "mean_integral_estimate": float(runs["integral_estimate"].mean()),
        "empirical_sd": float(runs["integral_estimate"].std(ddof=1)),
        "mean_absolute_error": float(runs["absolute_error"].mean()),
        "maximum_absolute_error": float(runs["absolute_error"].max()),
        "mean_stages": float(runs["stages"].mean()),
        "minimum_incremental_normalized_ess": float(
            runs["minimum_incremental_normalized_ess"].min()
        ),
        "mean_mala_acceptance": float(runs["mean_mala_acceptance"].mean()),
        "true_integral_used_for_training": False,
        "checkpoint_beta": float(checkpoint.get("beta", float("nan"))),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output / "run_results.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(args.output / "stage_history.csv", index=False)
    torch.save(saved_particles, args.output / "final_weighted_particles.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nMASS-AWARE ENERGY-BOMB SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
