#!/usr/bin/env python3
"""Train the conditional marginal+copula NF on a hidden irregular 100D mixture.

The target is a condition-dependent mixture of normalized autoregressive
components.  Components use different random long-range dependencies,
frequencies, widths, and gates.  The resulting joint density has no single
factorization, but its integral is exactly one for every condition.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sparse_wave_100d_base as base  # noqa: E402


adaptive = base.adaptive
DIMENSION = 100
COMPONENTS = 8
BENCHMARK = "irregular_mixture_100"
SEED = 73129


def _make_parameters():
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    parent_a = torch.zeros(COMPONENTS, DIMENSION, dtype=torch.long)
    parent_b = torch.zeros(COMPONENTS, DIMENSION, dtype=torch.long)
    for component in range(COMPONENTS):
        for child in range(1, DIMENSION):
            parent_a[component, child] = child - 1
            parent_b[component, child] = torch.randint(
                0, child, (1,), generator=generator
            )
    amplitude = 0.12 + 0.13 * torch.rand(COMPONENTS, generator=generator)
    interaction = 0.04 + 0.10 * torch.rand(COMPONENTS, generator=generator)
    frequency = torch.randint(1, 7, (COMPONENTS, DIMENSION), generator=generator).float()
    phase = 2 * math.pi * torch.rand(COMPONENTS, DIMENSION, generator=generator)
    condition_phase = 0.3 + 1.4 * torch.rand(COMPONENTS, 3, generator=generator)
    gate_weight = torch.randn(COMPONENTS, 3, generator=generator)
    gate_bias = torch.linspace(1.2, -1.2, COMPONENTS)
    # Include narrow rare modes, moderate ridges, and broad background modes.
    base_sigma = torch.tensor([0.025, 0.035, 0.050, 0.070, 0.090, 0.120, 0.155, 0.200])
    return {
        "parent_a": parent_a,
        "parent_b": parent_b,
        "amplitude": amplitude,
        "interaction": interaction,
        "frequency": frequency,
        "phase": phase,
        "condition_phase": condition_phase,
        "gate_weight": gate_weight,
        "gate_bias": gate_bias,
        "base_sigma": base_sigma,
    }


PARAMETERS = _make_parameters()
_DEVICE_CACHE = {}


def parameters_on(device, dtype):
    key = (str(device), dtype)
    if key not in _DEVICE_CACHE:
        _DEVICE_CACHE[key] = {
            name: value.to(device=device, dtype=dtype)
            if value.is_floating_point() else value.to(device=device)
            for name, value in PARAMETERS.items()
        }
    return _DEVICE_CACHE[key]


def mixture_log_weights(cond, parameters):
    logits = cond @ parameters["gate_weight"].transpose(0, 1)
    logits = logits + parameters["gate_bias"][None, :]
    # Oscillating condition gates create smooth regime changes and rare modes.
    logits = logits + 0.65 * torch.sin(
        2 * math.pi * (
            cond[:, :1] * parameters["condition_phase"][:, 0][None, :]
            + cond[:, 1:2] * parameters["condition_phase"][:, 1][None, :]
        )
    )
    return torch.log_softmax(logits, dim=1)


def irregular_mixture_100_log_integrand(x, cond):
    if x.ndim != 2 or x.shape[1] != DIMENSION:
        raise ValueError(f"{BENCHMARK} requires shape (batch,{DIMENSION})")
    if cond.ndim != 2 or cond.shape[1] != 3:
        raise ValueError("condition must have shape (batch,3)")

    parameters = parameters_on(x.device, x.dtype)
    batch = x.shape[0]
    log_component = torch.zeros(batch, COMPONENTS, device=x.device, dtype=x.dtype)
    component_index = torch.arange(COMPONENTS, device=x.device)

    for child in range(DIMENSION):
        if child == 0:
            a = cond[:, :1].expand(-1, COMPONENTS)
            b = cond[:, 1:2].expand(-1, COMPONENTS)
        else:
            pa = parameters["parent_a"][:, child]
            pb = parameters["parent_b"][:, child]
            a = x[:, pa]
            b = x[:, pb]

        frequency = parameters["frequency"][:, child][None, :]
        phase = parameters["phase"][:, child][None, :]
        c0, c1, c2 = cond[:, :1], cond[:, 1:2], cond[:, 2:3]
        wave = torch.sin(2 * math.pi * frequency * a + phase + 0.8 * c0)
        cross = torch.cos(2 * math.pi * (a - b) + 0.5 * phase - 0.7 * c1)
        switch = torch.tanh(4.0 * (c2 - 0.5) + 1.2 * cross)
        signal = (
            parameters["amplitude"][None, :] * wave
            + parameters["interaction"][None, :] * cross * switch
        )
        # Component- and coordinate-specific drift prevents repeated geometry.
        drift = 0.035 * torch.sin(
            0.17 * child + component_index[None, :] + math.pi * c1
        )
        mu = (0.5 + signal + drift).clamp(0.055, 0.945)
        sigma = parameters["base_sigma"][None, :] * (
            0.78 + 0.38 * c2 + 0.06 * ((child + component_index) % 5)[None, :]
        )
        x_child = x[:, child:child + 1].expand(-1, COMPONENTS)
        log_component = log_component + adaptive.truncated_normal_logpdf(
            x_child, mu, sigma
        )

    log_density = torch.logsumexp(
        mixture_log_weights(cond, parameters) + log_component, dim=1
    )
    inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
    return torch.where(inside, log_density, torch.full_like(log_density, -torch.inf))


def irregular_dependency_superset(dim):
    graph = {child: [] for child in range(dim)}
    for child in range(1, dim):
        parents = {child - 1}
        for component in range(COMPONENTS):
            parents.add(int(PARAMETERS["parent_b"][component, child]))
        graph[child] = sorted(parents)
    return graph


def register_benchmark():
    base.register_benchmark()
    previous_target = adaptive.target_log_integrand
    previous_configure = adaptive.configure_speed_profile

    def target_log_integrand(benchmark, x, cond):
        if benchmark == BENCHMARK:
            return irregular_mixture_100_log_integrand(x, cond)
        return previous_target(benchmark, x, cond)

    adaptive.SUPPORTED_BENCHMARKS.add(BENCHMARK)
    adaptive.BENCHMARK_DIMENSIONS[BENCHMARK] = DIMENSION
    adaptive.ORACLE_INTEGRALS[BENCHMARK] = 1.0
    adaptive.target_log_integrand = target_log_integrand
    adaptive.full_ar_graph = irregular_dependency_superset

    def configure_irregular_profile(profile):
        previous_configure(profile)
        # Spend additional steps only while held-out likelihood improves.
        adaptive.MAX_MARGINAL_STEPS = 1800
        adaptive.MARGINAL_EARLY_STOP_PATIENCE = 2
        adaptive.BRIDGE_COPULA_MAX_STEPS = 1800
        adaptive.BRIDGE_COPULA_PATIENCE = 2
        adaptive.KL_MAX_STEPS = 45000
        adaptive.CHI2_MAX_STEPS = 12000
        # Permit finer annealing moves when the preferred ESS target is missed.
        adaptive.TEMPERATURE_MIN_INCREMENT = 0.0005
        adaptive.TEMPERATURE_MAX_INCREMENT = 0.05

    adaptive.configure_speed_profile = configure_irregular_profile
    print(
        "Irregular 100D mixture: 8 hidden normalized components; exact I(c)=1; "
        "finer ESS-controlled annealing and expanded early-stopped optimization"
    )


def main():
    register_benchmark()
    adaptive.main()


if __name__ == "__main__":
    main()
