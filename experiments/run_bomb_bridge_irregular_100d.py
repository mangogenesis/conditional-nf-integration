#!/usr/bin/env python3
"""Bomb-rejuvenated annealing with separate marginal and copula fitting.

This runner retains the existing adaptive bridge and model classes, but
replaces each ordinary weighted training cache by a mass-resampled population
of local stochastic walker clouds.  MALA targets f(x|c)^beta in logit
coordinates.  Marginals are fitted first and frozen; the bridge copula is then
fitted independently by the upstream two-pass trainer.

The walker endpoint density is not evaluated.  The cache is therefore a
pseudo-sample from the tempered target, used only for proposal training.  The
final integral must still be measured with fresh samples from the normalized
trained NF.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_irregular_mixture_100d as irregular  # noqa: E402
from explore_energy_bombs_irregular_100d import logit  # noqa: E402


adaptive = irregular.adaptive
BENCHMARK = irregular.BENCHMARK
ORIGINAL_BUILD_WEIGHTED_CACHE = adaptive.build_weighted_cache

BOMB_MALA_STEPS = 6
BOMB_STEP_SIZE = 0.01
BOMB_GRADIENT_CLIP = 50.0
BOMB_CHUNK = 2048
BOMB_MIN_ACCEPTANCE = 0.35
BOMB_MAX_ACCEPTANCE = 0.85


def tempered_log_density_and_gradient(y, cond, beta):
    """Log f(sigmoid(y)|c)^beta plus the logistic volume element."""
    y = y.detach().requires_grad_(True)
    x = torch.sigmoid(y)
    log_f = adaptive.target_log_integrand(BENCHMARK, x, cond).reshape(-1)
    log_jacobian = (
        torch.nn.functional.logsigmoid(y)
        + torch.nn.functional.logsigmoid(-y)
    ).sum(dim=1)
    log_pi = beta * log_f + log_jacobian
    gradient = torch.autograd.grad(log_pi.sum(), y)[0]
    norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gradient = gradient * torch.clamp(BOMB_GRADIENT_CLIP / norm, max=1.0)
    return log_pi.detach().double(), gradient.detach()


def tempered_mala_step(y, cond, beta, step_size):
    outputs = []
    accepted_parts = []
    displacement_parts = []
    for start in range(0, y.shape[0], BOMB_CHUNK):
        current = y[start:start + BOMB_CHUNK]
        current_cond = cond[start:start + BOMB_CHUNK]
        log_current, gradient_current = tempered_log_density_and_gradient(
            current, current_cond, beta
        )
        mean_forward = current + 0.5 * step_size * gradient_current
        proposed = mean_forward + math.sqrt(step_size) * torch.randn_like(current)
        log_proposed, gradient_proposed = tempered_log_density_and_gradient(
            proposed, current_cond, beta
        )
        mean_reverse = proposed + 0.5 * step_size * gradient_proposed
        log_forward = -(proposed - mean_forward).double().square().sum(1) / (2 * step_size)
        log_reverse = -(current - mean_reverse).double().square().sum(1) / (2 * step_size)
        log_accept = log_proposed - log_current + log_reverse - log_forward
        accepted = torch.log(torch.rand_like(log_accept)) < torch.minimum(
            log_accept, torch.zeros_like(log_accept)
        )
        updated = torch.where(accepted[:, None], proposed, current).detach()
        outputs.append(updated)
        accepted_parts.append(accepted.double())
        displacement_parts.append(
            (torch.sigmoid(updated) - torch.sigmoid(current))
            .double().square().sum(1).sqrt()
        )
    return (
        torch.cat(outputs),
        float(torch.cat(accepted_parts).mean().item()),
        float(torch.cat(displacement_parts).mean().item()),
    )


def resample_weighted_cache(cache):
    """Every selected ancestor releases an independently moving walker."""
    conditions, points, dimension = cache["x"].shape
    indices = torch.multinomial(cache["alpha"], points, replacement=True)
    gather_x = indices[..., None].expand(-1, -1, dimension)
    gather_c = indices[..., None].expand(-1, -1, cache["cond"].shape[-1])
    x = torch.gather(cache["x"], 1, gather_x)
    cond = torch.gather(cache["cond"], 1, gather_c)
    unique_fraction = sum(
        torch.unique(indices[row]).numel() / points
        for row in range(conditions)
    ) / max(conditions, 1)
    return x, cond, unique_fraction


def bomb_rejuvenated_cache(
    benchmark, proposal_model, n_conditions, points_per_condition, device,
    seed, label, target_temperature=1.0,
    uniform_fraction=None,
):
    """Build the usual weighted cache, then resample and rejuvenate it."""
    kwargs = dict(
        benchmark=benchmark,
        proposal_model=proposal_model,
        n_conditions=n_conditions,
        points_per_condition=points_per_condition,
        device=device,
        seed=seed,
        label=label + " [pre-bomb]",
        target_temperature=target_temperature,
    )
    if uniform_fraction is not None:
        kwargs["uniform_fraction"] = uniform_fraction
    cache = ORIGINAL_BUILD_WEIGHTED_CACHE(**kwargs)
    x, cond, unique_fraction = resample_weighted_cache(cache)
    flat_cond = cond.reshape(-1, cond.shape[-1])
    y = logit(x.reshape(-1, x.shape[-1]))
    step_size = BOMB_STEP_SIZE
    acceptances = []
    displacements = []

    for step in range(BOMB_MALA_STEPS):
        y, acceptance, displacement = tempered_mala_step(
            y, flat_cond, target_temperature, step_size
        )
        acceptances.append(acceptance)
        displacements.append(displacement)
        if acceptance < BOMB_MIN_ACCEPTANCE:
            step_size *= 0.7
        elif acceptance > BOMB_MAX_ACCEPTANCE:
            step_size *= 1.35
        step_size = min(max(step_size, 1e-5), 0.10)

    cache["x"] = torch.sigmoid(y).reshape(n_conditions, points_per_condition, -1).detach()
    cache["cond"] = cond.detach()
    cache["alpha"] = torch.full(
        (n_conditions, points_per_condition),
        1.0 / points_per_condition,
        device=device,
    )
    cache["bomb_mean_acceptance"] = sum(acceptances) / len(acceptances)
    cache["bomb_mean_displacement"] = sum(displacements) / len(displacements)
    cache["bomb_ancestor_unique_fraction"] = unique_fraction
    print()
    print(f"BOMB-REJUVENATED CACHE: {label}")
    print(f"MALA steps                  = {BOMB_MALA_STEPS}")
    print(f"mean MALA acceptance        = {cache['bomb_mean_acceptance']:.4f}")
    print(f"mean x-space displacement   = {cache['bomb_mean_displacement']:.6f}")
    print(f"unique ancestor fraction    = {unique_fraction:.4f}")
    print(f"final MALA step size        = {step_size:.6g}")
    return cache


def parse_wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bomb-mala-steps", type=int, default=6)
    parser.add_argument("--bomb-step-size", type=float, default=0.01)
    parser.add_argument("--bomb-gradient-clip", type=float, default=50.0)
    parser.add_argument("--bomb-chunk", type=int, default=2048)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main():
    global BOMB_MALA_STEPS, BOMB_STEP_SIZE, BOMB_GRADIENT_CLIP, BOMB_CHUNK
    args = parse_wrapper_args()
    BOMB_MALA_STEPS = args.bomb_mala_steps
    BOMB_STEP_SIZE = args.bomb_step_size
    BOMB_GRADIENT_CLIP = args.bomb_gradient_clip
    BOMB_CHUNK = args.bomb_chunk
    if "--joint-marginal-copula" in sys.argv:
        raise ValueError(
            "This experiment requires independent marginal then copula fitting; "
            "remove --joint-marginal-copula."
        )
    irregular.register_benchmark()
    adaptive.build_weighted_cache = bomb_rejuvenated_cache
    print(
        "Bomb bridge enabled: mass resampling + tempered MALA cache; "
        "marginals trained and frozen before independent copula fitting"
    )
    adaptive.main()


if __name__ == "__main__":
    main()
