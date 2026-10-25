#!/usr/bin/env python3
"""Train the existing conditional marginal+copula NF on a normalized 100D target.

This is a thin registration wrapper around the validated adaptive trainer.  It
does not modify spline transforms or their inverse.  The benchmark is a
direct, sparse extension of ``sparse_wave_32`` and is normalized by
construction, so its integral is exactly one at every condition.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

torch.set_float32_matmul_precision("high")


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_adaptive_integrand_structure_search as adaptive  # noqa: E402
from nf.vectorized_cubic import (  # noqa: E402
    vectorized_copula_log_prob,
    vectorized_marginal_inverse,
)


DIMENSION = 100
BENCHMARK = "sparse_wave_100"
VECTORIZED_EVAL_CHUNK = 8192


def chunked_vectorized_log_prob(self, x, cond):
    """Evaluate large cache/proposal batches without a G x N x H allocation."""
    outputs = []
    for start in range(0, x.shape[0], VECTORIZED_EVAL_CHUNK):
        end = min(start + VECTORIZED_EVAL_CHUNK, x.shape[0])
        cond_chunk = cond if cond.shape[0] == 1 else cond[start:end]
        u, log_marginal = vectorized_marginal_inverse(
            self.marginals, x[start:end], cond_chunk
        )
        outputs.append(
            log_marginal + vectorized_copula_log_prob(
                self.copula, u, cond_chunk
            )
        )
    return torch.cat(outputs, dim=0)


def sparse_wave_100_mu_sigma(child, x, cond):
    """Sparse nonlinear parents with condition-dependent geometry."""
    parent = child - 1
    skip = max(0, child // 2 - 1)
    a = x[:, parent:parent + 1]
    b = x[:, skip:skip + 1]
    c0 = cond[:, 0:1]
    c1 = cond[:, 1:2]
    c2 = cond[:, 2:3]

    # Frequencies and phase drift across coordinates, while remaining smooth
    # in c.  The tanh keeps every conditional mean away from the cube boundary.
    frequency = 1.0 + float(child % 4)
    phase = 0.017 * child + 0.12 * c1
    local_wave = torch.sin(2.0 * math.pi * (frequency * a + phase))
    skip_wave = torch.cos(2.0 * math.pi * (b - 0.09 * c0))
    interaction = torch.sin(2.0 * math.pi * (a - b + 0.05 * c2))
    gate = torch.sigmoid(7.0 * (c0 - 0.5) + 0.5 * torch.sin(math.pi * c1))
    signal = gate * local_wave + (1.0 - gate) * skip_wave + 0.18 * interaction
    # A 100D product amplifies each coordinate's concentration.  These scales
    # keep the aggregate bridge difficulty comparable to the validated 32D
    # benchmark instead of making the uniform-to-target path exponentially
    # sharper solely because 68 coordinates were added.
    mu = 0.50 + 0.20 * torch.tanh(1.20 * signal)
    sigma = 0.120 + 0.030 * c2 + 0.004 * float(child % 5)
    return mu, sigma


def sparse_wave_100_log_integrand(x, cond):
    """Normalized 100D density available to training only through log f."""
    if x.ndim != 2 or x.shape[1] != DIMENSION:
        raise ValueError(f"{BENCHMARK} requires shape (batch,{DIMENSION})")
    if cond.ndim != 2 or cond.shape[1] != 3:
        raise ValueError("condition must have shape (batch,3)")
    logf = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
    # x0 and x1 are independent uniforms.  Every later factor is a normalized
    # truncated normal conditional, hence the complete joint integrates to one.
    for child in range(2, DIMENSION):
        mu, sigma = sparse_wave_100_mu_sigma(child, x, cond)
        logf += adaptive.truncated_normal_logpdf(
            x[:, child:child + 1], mu, sigma
        ).reshape(-1)
    return torch.where(inside, logf, torch.full_like(logf, -torch.inf))


def target_log_integrand(benchmark, x, cond):
    if benchmark == BENCHMARK:
        return sparse_wave_100_log_integrand(x, cond)
    return _ORIGINAL_TARGET_LOG_INTEGRAND(benchmark, x, cond)


def sparse_dependency_graph(dim):
    """Graph containing exactly the two structural parents of each target node."""
    graph = {child: [] for child in range(dim)}
    for child in range(2, dim):
        graph[child] = sorted({child - 1, max(0, child // 2 - 1)})
    return graph


_ORIGINAL_TARGET_LOG_INTEGRAND = adaptive.target_log_integrand
_ORIGINAL_CONFIGURE_SPEED_PROFILE = adaptive.configure_speed_profile


def configure_100d_speed_profile(profile: str) -> None:
    """Use thorough budgets with ESS-adaptive annealing for the formal run.

    The upstream trainer selects a fixed temperature schedule when the profile
    string is ``thorough``.  That schedule was validated at lower dimension but
    is too aggressive in 100D.  For this wrapper, ``balanced`` deliberately
    means thorough compute budgets plus a genuinely adaptive schedule.
    """
    if profile == "balanced":
        _ORIGINAL_CONFIGURE_SPEED_PROFILE("thorough")
        adaptive.TEMPERATURE_MIN_INCREMENT = 0.005
        adaptive.TEMPERATURE_MAX_INCREMENT = 0.10
        # At each annealing stage the 100D validation loss commonly reaches
        # its optimum immediately.  Avoid spending 900--1200 additional steps
        # after deterioration has already been detected.
        adaptive.MIN_MARGINAL_STEPS = 300
        adaptive.MAX_MARGINAL_STEPS = 900
        adaptive.MARGINAL_CHECK_INTERVAL = 300
        adaptive.MARGINAL_EARLY_STOP_PATIENCE = 1
        adaptive.BRIDGE_COPULA_MAX_STEPS = 900
        adaptive.BRIDGE_COPULA_CHECK_INTERVAL = 300
        adaptive.BRIDGE_COPULA_PATIENCE = 1
        print(
            "100D formal mode: thorough budgets with ESS-adaptive annealing "
            "(temperature increment 0.005--0.10; early stage stopping)"
        )
    else:
        _ORIGINAL_CONFIGURE_SPEED_PROFILE(profile)


def register_benchmark() -> None:
    adaptive.SUPPORTED_BENCHMARKS.add(BENCHMARK)
    adaptive.BENCHMARK_DIMENSIONS[BENCHMARK] = DIMENSION
    adaptive.ORACLE_INTEGRALS[BENCHMARK] = 1.0
    adaptive.target_log_integrand = target_log_integrand
    adaptive.configure_speed_profile = configure_100d_speed_profile
    # A full 100D autoregressive graph has 4,950 edges and is unnecessary for
    # this deliberately sparse target.  All trainer calls requesting its
    # initial graph receive the exact sparse structural superset instead.
    adaptive.full_ar_graph = sparse_dependency_graph

    # Preserve the independently parameterized flows and their state_dict
    # layout, but batch equal-context cubic inverses during teacher-forced
    # likelihood evaluation. Autoregressive sampling remains sequential.
    fast_components = []
    if hasattr(adaptive, "StructuredCopula"):
        adaptive.StructuredCopula.log_prob_u = vectorized_copula_log_prob
        fast_components.append("copula")
    if hasattr(adaptive, "MarginalCopulaModel"):
        adaptive.MarginalCopulaModel.x_to_u_and_log_marginal = (
            lambda self, x, cond: vectorized_marginal_inverse(
                self.marginals, x, cond
            )
        )
        adaptive.MarginalCopulaModel.log_prob = chunked_vectorized_log_prob
        fast_components.append("marginals")
    if fast_components:
        print("Vectorized cubic inverse enabled for: " + ", ".join(fast_components))
    else:
        print(
            "WARNING: adaptive trainer does not expose StructuredCopula/"
            "MarginalCopulaModel; vectorized fast path was not installed"
        )


def main() -> None:
    register_benchmark()
    adaptive.main()


if __name__ == "__main__":
    main()
