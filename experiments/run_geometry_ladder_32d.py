#!/usr/bin/env python3
"""Controlled 32D geometry ladder for the original conditional copula NF.

The core four normalized targets form a 2x2 design:

    factorized / autoregressive  x  broad / sharp.

The marginal stage additionally includes uniform, anisotropic, skewed,
boundary-singular, and logit-heavy-tailed factorized targets.  Only the target
geometry changes.  All benchmarks use the original single
conditional marginal+copula model, the same training code, and exact integral
I(c)=1.  There are no target mixtures or repair specialists.
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

import run_irregular_mixture_32d as irregular32  # noqa: E402


adaptive = irregular32.adaptive
family = irregular32.family
DIMENSION = 32

FACTORIZED_BROAD = "factorized_broad_32"
FACTORIZED_SHARP = "factorized_sharp_32"
UNIFORM = "uniform_32"
FACTORIZED_ANISOTROPIC = "factorized_anisotropic_32"
FACTORIZED_BOUNDARY = "factorized_boundary_32"
FACTORIZED_BOUNDARY_INFINITE_M2 = "factorized_boundary_infinite_m2_32"
FACTORIZED_SKEWED = "factorized_skewed_32"
FACTORIZED_LOGIT_STUDENT = "factorized_logit_student_32"
AUTOREGRESSIVE_BROAD = "autoregressive_broad_32"
AUTOREGRESSIVE_SHARP = "autoregressive_sharp_32"
BENCHMARKS = {
    UNIFORM,
    FACTORIZED_BROAD,
    FACTORIZED_SHARP,
    FACTORIZED_ANISOTROPIC,
    FACTORIZED_BOUNDARY,
    FACTORIZED_BOUNDARY_INFINITE_M2,
    FACTORIZED_SKEWED,
    FACTORIZED_LOGIT_STUDENT,
    AUTOREGRESSIVE_BROAD,
    AUTOREGRESSIVE_SHARP,
}


def factorized_mu(child, cond):
    """Smooth condition-dependent means with no x-coordinate dependence."""
    c0 = cond[:, 0:1]
    c1 = cond[:, 1:2]
    c2 = cond[:, 2:3]
    phase = 0.173 * child
    signal = (
        0.72 * torch.sin(2.0 * math.pi * c0 + phase)
        + 0.28 * torch.cos(2.0 * math.pi * c1 - 0.7 * phase)
        + 0.15 * torch.sin(2.0 * math.pi * c2 + 0.3 * phase)
    )
    return 0.50 + 0.27 * torch.tanh(signal)


def factorized_log_integrand(x, cond, sigma):
    if x.shape[1] != DIMENSION:
        raise ValueError(f"factorized benchmark requires {DIMENSION} dimensions")
    result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    scale = torch.full(
        (x.shape[0], 1), float(sigma), device=x.device, dtype=x.dtype
    )
    for child in range(DIMENSION):
        mu = factorized_mu(child, cond)
        result += adaptive.truncated_normal_logpdf(
            x[:, child:child + 1], mu, scale
        ).reshape(-1)
    inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def anisotropic_log_integrand(x, cond):
    """Independent coordinates spanning broad-to-sharp length scales."""
    result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    # Geometric spacing prevents the broad coordinates from numerically
    # dominating the definition of the scale ladder.
    coordinate_scale = torch.logspace(
        math.log10(0.025), math.log10(0.15), DIMENSION,
        device=x.device, dtype=x.dtype,
    )
    condition_scale = 0.90 + 0.20 * cond[:, 2:3]
    for child in range(DIMENSION):
        mu = factorized_mu(child, cond)
        sigma = coordinate_scale[child] * condition_scale
        result += adaptive.truncated_normal_logpdf(
            x[:, child:child + 1], mu, sigma
        ).reshape(-1)
    inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def beta_logpdf(value, alpha, beta):
    """Stable normalized beta log density on the open unit interval."""
    tiny = torch.finfo(value.dtype).tiny
    value = value.clamp(tiny, 1.0 - torch.finfo(value.dtype).eps)
    log_normalizer = (
        torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    )
    return (
        (alpha - 1.0) * torch.log(value)
        + (beta - 1.0) * torch.log1p(-value)
        - log_normalizer
    )


def boundary_log_integrand(x, cond, infinite_uniform_m2=False):
    """Independent integrable singularities at both cube boundaries.

    When ``infinite_uniform_m2`` is true, beta exponents are below 1/2, so
    importance sampling from a uniform proposal has an infinite second moment.
    The ordinary version keeps exponents above 1/2 and therefore isolates
    boundary geometry without simultaneously imposing that pathology.
    """
    result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for child in range(DIMENSION):
        phase = 0.13 * child
        base = 0.30 if infinite_uniform_m2 else 0.62
        amplitude = 0.14
        alpha = base + amplitude * torch.sigmoid(
            3.0 * (cond[:, 0:1] - 0.5) + phase
        )
        beta = base + amplitude * torch.sigmoid(
            3.0 * (cond[:, 1:2] - 0.5) - phase
        )
        result += beta_logpdf(x[:, child:child + 1], alpha, beta).reshape(-1)
    inside = ((x > 0.0) & (x < 1.0)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def skewed_log_integrand(x, cond):
    """Smooth asymmetric marginals with alternating skew directions."""
    result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for child in range(DIMENSION):
        low = 1.8 + 1.2 * cond[:, 0:1]
        high = 5.0 + 2.0 * cond[:, 1:2] + 0.15 * float(child % 4)
        if child % 2:
            alpha, beta = high, low
        else:
            alpha, beta = low, high
        result += beta_logpdf(x[:, child:child + 1], alpha, beta).reshape(-1)
    inside = ((x > 0.0) & (x < 1.0)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def logit_student_log_integrand(x, cond):
    """Independent Student-t variables mapped exactly onto the unit cube."""
    degrees = 3.0
    log_constant = (
        math.lgamma((degrees + 1.0) / 2.0)
        - math.lgamma(degrees / 2.0)
        - 0.5 * math.log(degrees * math.pi)
    )
    result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    epsilon = torch.finfo(x.dtype).eps
    for child in range(DIMENSION):
        value = x[:, child:child + 1].clamp(epsilon, 1.0 - epsilon)
        y = torch.log(value) - torch.log1p(-value)
        location = 1.1 * torch.tanh(
            1.4 * (
                cond[:, 0:1] - 0.5
                + 0.35 * torch.sin(
                    2.0 * math.pi * cond[:, 1:2] + 0.11 * child
                )
            )
        )
        scale = 0.55 + 0.25 * cond[:, 2:3] + 0.03 * float(child % 3)
        standardized = (y - location) / scale
        log_y = (
            log_constant
            - torch.log(scale)
            - 0.5 * (degrees + 1.0)
            * torch.log1p(standardized.square() / degrees)
        )
        # dy/dx = 1 / (x (1-x)).
        result += (
            log_y - torch.log(value) - torch.log1p(-value)
        ).reshape(-1)
    inside = ((x > 0.0) & (x < 1.0)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def component_mu_sigma(child, x, cond, sigma):
    """One component of the irregular family, without target mixing."""
    parameters = family.parameters_on(x.device, x.dtype)
    component = 0
    if child == 0:
        a = cond[:, 0:1]
        b = cond[:, 1:2]
    else:
        parent_a = int(parameters["parent_a"][component, child])
        parent_b = int(parameters["parent_b"][component, child])
        a = x[:, parent_a:parent_a + 1]
        b = x[:, parent_b:parent_b + 1]

    frequency = parameters["frequency"][component, child]
    phase = parameters["phase"][component, child]
    wave = torch.sin(
        2.0 * math.pi * frequency * a + phase + 0.8 * cond[:, 0:1]
    )
    cross = torch.cos(
        2.0 * math.pi * (a - b) + 0.5 * phase - 0.7 * cond[:, 1:2]
    )
    switch = torch.tanh(4.0 * (cond[:, 2:3] - 0.5) + 1.2 * cross)
    signal = (
        parameters["amplitude"][component] * wave
        + parameters["interaction"][component] * cross * switch
    )
    drift = 0.035 * torch.sin(
        torch.as_tensor(
            0.17 * child + component, device=x.device, dtype=x.dtype
        )
        + math.pi * cond[:, 1:2]
    )
    mu = (0.5 + signal + drift).clamp(0.055, 0.945)
    # Retain mild condition/coordinate heterogeneity while fixing the overall
    # broad-versus-sharp scale.
    scale = float(sigma) * (
        0.88 + 0.20 * cond[:, 2:3] + 0.025 * float(child % 5)
    )
    return mu, scale


def autoregressive_log_integrand(x, cond, sigma):
    if x.shape[1] != DIMENSION:
        raise ValueError(
            f"autoregressive benchmark requires {DIMENSION} dimensions"
        )
    result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for child in range(DIMENSION):
        mu, scale = component_mu_sigma(child, x, cond, sigma)
        result += adaptive.truncated_normal_logpdf(
            x[:, child:child + 1], mu, scale
        ).reshape(-1)
    inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def ladder_graph(dim):
    """Fixed superset containing the single AR target's true dependencies."""
    parameters = family.PARAMETERS
    graph = {child: [] for child in range(dim)}
    for child in range(1, dim):
        graph[child] = sorted({
            int(parameters["parent_a"][0, child]),
            int(parameters["parent_b"][0, child]),
        })
    return graph


def register_benchmarks():
    # This restricts the shared irregular-family parameters to 32D before the
    # autoregressive single-component target is evaluated.
    irregular32.configure_family()
    previous_target = adaptive.target_log_integrand

    def target_log_integrand(benchmark, x, cond):
        if benchmark == UNIFORM:
            inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
            zeros = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            return torch.where(
                inside, zeros, torch.full_like(zeros, -torch.inf)
            )
        if benchmark == FACTORIZED_BROAD:
            return factorized_log_integrand(x, cond, sigma=0.15)
        if benchmark == FACTORIZED_SHARP:
            return factorized_log_integrand(x, cond, sigma=0.025)
        if benchmark == FACTORIZED_ANISOTROPIC:
            return anisotropic_log_integrand(x, cond)
        if benchmark == FACTORIZED_BOUNDARY:
            return boundary_log_integrand(x, cond)
        if benchmark == FACTORIZED_BOUNDARY_INFINITE_M2:
            return boundary_log_integrand(x, cond, infinite_uniform_m2=True)
        if benchmark == FACTORIZED_SKEWED:
            return skewed_log_integrand(x, cond)
        if benchmark == FACTORIZED_LOGIT_STUDENT:
            return logit_student_log_integrand(x, cond)
        if benchmark == AUTOREGRESSIVE_BROAD:
            return autoregressive_log_integrand(x, cond, sigma=0.15)
        if benchmark == AUTOREGRESSIVE_SHARP:
            return autoregressive_log_integrand(x, cond, sigma=0.025)
        return previous_target(benchmark, x, cond)

    adaptive.SUPPORTED_BENCHMARKS.update(BENCHMARKS)
    for benchmark in BENCHMARKS:
        adaptive.BENCHMARK_DIMENSIONS[benchmark] = DIMENSION
        adaptive.ORACLE_INTEGRALS[benchmark] = 1.0
    adaptive.target_log_integrand = target_log_integrand
    adaptive.full_ar_graph = ladder_graph
    print(
        "32D geometry ladder: marginal-shape basis plus "
        "factorized/autoregressive x broad/sharp; single original "
        "conditional marginal+copula NF; exact I(c)=1"
    )


def main():
    register_benchmarks()
    adaptive.main()


if __name__ == "__main__":
    main()
