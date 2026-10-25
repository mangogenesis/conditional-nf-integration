"""Persistent annealed-SMC discovery plus clustered conditional specialists.

Training and validation populations persist across annealing temperatures.  An
incremental SMC bridge reweights by f(x|c) ** (beta_new-beta_old), resamples
within each condition, and rejuvenates with MALA.  The resulting equally
weighted particles are training data only.  Ordinary importance sampling is
retained for residual diagnostics and final integral estimation.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

import run_adaptive_mixture_irregular_100d as mixture
import run_bomb_bridge_irregular_100d as bomb
import run_clustered_bomb_mixture_irregular_100d as clustered


adaptive = mixture.adaptive
BENCHMARK = mixture.BENCHMARK
ORIGINAL_BUILD = bomb.ORIGINAL_BUILD_WEIGHTED_CACHE
ORIGINAL_SELECT_TEMPERATURE = adaptive.select_next_temperature
POPULATIONS = {}
ARGS = None
INITIAL_BETA = 0.0


def parse_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smc-target-ess", type=float, default=0.70)
    parser.add_argument("--smc-minimum-q05-ess", type=float, default=0.30)
    parser.add_argument("--smc-mala-steps", type=int, default=6)
    parser.add_argument("--smc-step-size", type=float, default=0.01)
    parser.add_argument("--smc-gradient-clip", type=float, default=50.0)
    parser.add_argument("--smc-chunk", type=int, default=1024)
    parser.add_argument("--smc-forced-increment", type=float, default=0.02)
    parser.add_argument("--smc-initial-beta", type=float)
    wrapper, remaining = parser.parse_known_args(argv)
    specialist, remaining = clustered.parse_wrapper_arguments(remaining)
    return wrapper, specialist, remaining


def infer_initial_beta(arguments, explicit):
    if explicit is not None:
        return float(explicit)
    if "--resume" not in arguments:
        return 0.0
    position = arguments.index("--resume")
    if position + 1 >= len(arguments):
        return 0.0
    path = arguments[position + 1]
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        return float(checkpoint["beta"])
    except (OSError, KeyError, RuntimeError):
        return 0.0


def normalized_ess(weights):
    return 1.0 / (
        weights.shape[1] * weights.double().square().sum(dim=1).clamp_min(1e-300)
    )


@torch.no_grad()
def log_integrand(x, cond):
    pieces = []
    flat_x = x.reshape(-1, x.shape[-1])
    flat_cond = cond.reshape(-1, cond.shape[-1])
    for start in range(0, len(flat_x), ARGS.smc_chunk):
        stop = min(start + ARGS.smc_chunk, len(flat_x))
        pieces.append(
            adaptive.target_log_integrand(
                BENCHMARK, flat_x[start:stop], flat_cond[start:stop]
            ).reshape(-1).double()
        )
    return torch.cat(pieces).reshape(x.shape[:2])


def incremental_weights(log_f, delta):
    logw = delta * log_f
    logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
    return torch.exp(logw)


def choose_next_beta(log_f, current, target):
    """Largest bridge step satisfying median and lower-tail ESS criteria."""
    if target <= current + 1e-12:
        return target

    def acceptable(candidate):
        ess = normalized_ess(incremental_weights(log_f, candidate - current))
        median = float(torch.quantile(ess, 0.50).item())
        q05 = float(torch.quantile(ess, 0.05).item())
        return median >= ARGS.smc_target_ess and q05 >= ARGS.smc_minimum_q05_ess

    if acceptable(target):
        return target
    low, high = current, target
    for _ in range(24):
        middle = 0.5 * (low + high)
        if acceptable(middle):
            low = middle
        else:
            high = middle
    if low <= current + 1e-8:
        raise RuntimeError(
            f"SMC could not find a safe positive bridge step at beta={current:.6f}"
        )
    return low


def resample(x, cond, weights):
    conditions, points, dimension = x.shape
    indices = torch.multinomial(weights.float(), points, replacement=True)
    gather_x = indices[..., None].expand(-1, -1, dimension)
    gather_c = indices[..., None].expand(-1, -1, cond.shape[-1])
    unique = sum(
        torch.unique(indices[row]).numel() / points
        for row in range(conditions)
    ) / conditions
    return torch.gather(x, 1, gather_x), torch.gather(cond, 1, gather_c), unique


@torch.no_grad()
def project_onto_positive_support(state, key):
    """Handle the discontinuous beta=0 -> beta=0+ support restriction.

    For targets that are exactly zero on part of the cube, f**beta removes
    that region for every positive beta, however small.  No ESS-controlled
    positive step can avoid this finite initial loss.  Condition the uniform
    beta-zero population on finite log f once, then start ordinary annealing.
    """
    log_f = log_integrand(state["x"], state["cond"])
    finite = torch.isfinite(log_f)
    count = finite.sum(dim=1)
    if bool((count == 0).any()):
        failed = int((count == 0).sum().item())
        raise RuntimeError(
            f"SMC support projection found no positive target points in "
            f"{failed} {key} populations; increase points per population"
        )
    if bool(finite.all()):
        return state
    weights = finite.double()
    weights = weights / weights.sum(dim=1, keepdim=True)
    support_fraction = count.double() / finite.shape[1]
    x, cond, unique = resample(state["x"], state["cond"], weights)
    projected = {"x": x, "cond": cond, "beta": float(state["beta"])}
    print(
        f"SMC SUPPORT PROJECTION {key}: positive-volume fraction "
        f"median={float(support_fraction.median()):.4f} "
        f"q05={float(torch.quantile(support_fraction, .05)):.4f} "
        f"unique={unique:.4f}; ordinary ESS control starts after projection"
    )
    return projected


def rejuvenate(x, cond, beta):
    y = torch.logit(x.clamp(1e-6, 1.0 - 1e-6)).reshape(-1, x.shape[-1])
    flat_cond = cond.reshape(-1, cond.shape[-1])
    step_size = ARGS.smc_step_size
    acceptances = []
    displacements = []
    bomb.BOMB_CHUNK = ARGS.smc_chunk
    bomb.BOMB_GRADIENT_CLIP = ARGS.smc_gradient_clip
    for _ in range(ARGS.smc_mala_steps):
        y, acceptance, displacement = bomb.tempered_mala_step(
            y, flat_cond, beta, step_size
        )
        acceptances.append(acceptance)
        displacements.append(displacement)
        if acceptance < 0.35:
            step_size *= 0.7
        elif acceptance > 0.85:
            step_size *= 1.35
        step_size = min(max(step_size, 1e-5), 0.10)
    return (
        torch.sigmoid(y).reshape_as(x).detach(),
        sum(acceptances) / len(acceptances),
        sum(displacements) / len(displacements),
        step_size,
    )


def bootstrap_population(
    key, proposal_model, n_conditions, points, device, seed,
    uniform_fraction,
):
    # At beta=0 the bridge is exactly uniform on the unit cube.  Building a
    # weighted cache would form ``0 * log_f``; targets with genuine zero-density
    # regions have log_f=-inf there, and IEEE arithmetic turns 0*(-inf) into
    # NaN.  Sample the known beta-zero law directly instead.
    if abs(INITIAL_BETA) <= 1.0e-15:
        adaptive.seed_everything(seed)
        anchors = adaptive.sample_conditions(n_conditions, device)
        cond = anchors[:, None, :].expand(
            n_conditions, points, adaptive.COND_DIM
        ).clone()
        x = torch.rand(
            n_conditions,
            points,
            adaptive.DIM,
            dtype=cond.dtype,
            device=device,
        )
        state = {"x": x, "cond": cond, "beta": INITIAL_BETA}
        POPULATIONS[key] = state
        print(
            f"SMC BOOTSTRAP {key}: beta=0 sampled exact uniform law "
            "unique=1.0000; no reweighting or MALA required"
        )
        return state

    build_kwargs = dict(
        benchmark=BENCHMARK,
        proposal_model=proposal_model,
        n_conditions=n_conditions,
        points_per_condition=points,
        device=device,
        seed=seed,
        label=f"persistent {key} bootstrap beta={INITIAL_BETA:.6f}",
        target_temperature=INITIAL_BETA,
    )
    if uniform_fraction is not None:
        build_kwargs["uniform_fraction"] = uniform_fraction
    cache = ORIGINAL_BUILD(**build_kwargs)
    x, cond, unique = resample(cache["x"], cache["cond"], cache["alpha"])
    x, acceptance, displacement, step_size = rejuvenate(x, cond, INITIAL_BETA)
    state = {"x": x, "cond": cond, "beta": INITIAL_BETA}
    POPULATIONS[key] = state
    print(
        f"SMC BOOTSTRAP {key}: beta={INITIAL_BETA:.6f} "
        f"unique={unique:.4f} MALA={acceptance:.4f} "
        f"displacement={displacement:.6f} h={step_size:.6g}"
    )
    return state


def persistent_smc_cache(
    benchmark, proposal_model, n_conditions, points_per_condition, device,
    seed, label, target_temperature=1.0, uniform_fraction=None,
):
    del benchmark
    key = "train" if label.startswith("mixture train") else "validation"
    state = POPULATIONS.get(key)
    expected = (n_conditions, points_per_condition)
    if state is None or state["x"].shape[:2] != expected:
        if state is not None:
            print(
                f"SMC {key.upper()} SIZE CHANGED: {tuple(state['x'].shape[:2])} "
                f"-> {expected}; rebuilding the persistent population"
            )
        state = bootstrap_population(
            key, proposal_model, n_conditions, points_per_condition, device,
            seed, uniform_fraction,
        )

    current = float(state["beta"])
    target = float(target_temperature)
    if target < current - 1e-10:
        raise ValueError(f"SMC temperature moved backward: {current} -> {target}")
    if current <= 1.0e-15 and target > current + 1.0e-15:
        state = project_onto_positive_support(state, key)
        POPULATIONS[key] = state
    bridge = 0
    while current < target - 1e-10:
        log_f = log_integrand(state["x"], state["cond"])
        next_beta = choose_next_beta(log_f, current, target)
        weights = incremental_weights(log_f, next_beta - current)
        ess = normalized_ess(weights)
        x, cond, unique = resample(state["x"], state["cond"], weights)
        x, acceptance, displacement, step_size = rejuvenate(x, cond, next_beta)
        current = next_beta
        state = {"x": x, "cond": cond, "beta": current}
        POPULATIONS[key] = state
        bridge += 1
        print(
            f"SMC {key} bridge {bridge:02d}: beta={current:.6f} "
            f"ESS median={float(torch.quantile(ess, .5)):.4f} "
            f"q05={float(torch.quantile(ess, .05)):.4f} "
            f"unique={unique:.4f} MALA={acceptance:.4f} "
            f"dx={displacement:.6f} h={step_size:.6g}"
        )

    return {
        "x": state["x"],
        "cond": state["cond"],
        "alpha": torch.full(
            expected,
            1.0 / points_per_condition,
            device=device,
        ),
        "smc_beta": current,
        "smc_bridge_steps": bridge,
    }


def cache_dispatch(*args, **kwargs):
    label = str(kwargs.get("label", ""))
    if not label and len(args) >= 7:
        label = str(args[6])
    if label.startswith("mixture train") or label.startswith("mixture val"):
        return persistent_smc_cache(*args, **kwargs)
    return ORIGINAL_BUILD(*args, **kwargs)


def smc_temperature_controller(benchmark, proposal, current_beta, device, seed):
    """Let ordinary diagnostics choose easy steps and SMC rescue hard ones."""
    chosen, uniform_fraction, passed = ORIGINAL_SELECT_TEMPERATURE(
        benchmark, proposal, current_beta, device, seed
    )
    if passed:
        return chosen, uniform_fraction, True
    forced = min(1.0, current_beta + ARGS.smc_forced_increment)
    if forced <= current_beta + 1e-12:
        return chosen, uniform_fraction, passed
    print(
        f"SMC TEMPERATURE RESCUE: proposal test held beta={current_beta:.6f}; "
        f"persistent SMC will bridge to {forced:.6f} with adaptive substeps"
    )
    return forced, uniform_fraction, True


def main():
    global ARGS, INITIAL_BETA
    ARGS, specialist_args, remaining = parse_arguments(sys.argv[1:])
    INITIAL_BETA = infer_initial_beta(remaining, ARGS.smc_initial_beta)
    clustered.WRAPPER_ARGS = specialist_args
    sys.argv = [sys.argv[0], *remaining]
    adaptive.build_weighted_cache = cache_dispatch
    adaptive.select_next_temperature = smc_temperature_controller
    mixture.train_mixture_on_cache = clustered.train_clustered_two_pass
    print(
        "Persistent SMC clustered mixture enabled: incremental reweighting + "
        "resampling + MALA; marginal -> copula specialists; conditional mass gate"
    )
    print(f"Initial SMC beta={INITIAL_BETA:.6f}")
    mixture.main()


if __name__ == "__main__":
    main()
