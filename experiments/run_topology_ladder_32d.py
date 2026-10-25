#!/usr/bin/env python3
"""Controlled 32D topology ladder for the original conditional copula NF.

Every branch is an affine copy of the same normalized autoregressive target.
Only the number of disconnected branches and their mixture weights change:

* ``topology_one_branch_32``: one connected branch;
* ``topology_two_equal_32``: two disconnected branches with equal mass;
* ``topology_two_conditional_32``: the same branches with smoothly varying
  condition-dependent masses.

The first coordinate maps the reference target into either [0, 0.4] or
[0.6, 1].  The open gap makes the two-branch support genuinely disconnected,
while the affine Jacobian keeps every component exactly normalized.  Hence
I(c)=1 is known analytically for every benchmark.
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

import run_geometry_ladder_32d as geometry  # noqa: E402


adaptive = geometry.adaptive
DIMENSION = 32
BRANCH_WIDTH = 0.40
REFERENCE_SIGMA = 0.05

ONE_BRANCH = "topology_one_branch_32"
TWO_EQUAL = "topology_two_equal_32"
TWO_CONDITIONAL = "topology_two_conditional_32"
BENCHMARKS = {ONE_BRANCH, TWO_EQUAL, TWO_CONDITIONAL}


def branch_log_integrand(x, cond, branch):
    """Density of one affine image of the common reference geometry."""
    if branch not in (0, 1):
        raise ValueError("branch must be 0 or 1")
    lower = 0.0 if branch == 0 else 1.0 - BRANCH_WIDTH
    upper = lower + BRANCH_WIDTH
    inside = (
        (x[:, 0] >= lower)
        & (x[:, 0] <= upper)
        & ((x[:, 1:] >= 0.0) & (x[:, 1:] <= 1.0)).all(dim=1)
    )
    reference = torch.cat(
        ((x[:, 0:1] - lower) / BRANCH_WIDTH, x[:, 1:]), dim=1
    )
    log_density = geometry.autoregressive_log_integrand(
        reference, cond, sigma=REFERENCE_SIGMA
    ) - math.log(BRANCH_WIDTH)
    return torch.where(
        inside, log_density, torch.full_like(log_density, -torch.inf)
    )


def conditional_branch_one_probability(cond):
    """Smooth, bounded branch mass used by the conditional topology test."""
    logit = (
        2.2 * (cond[:, 0] - 0.5)
        + 1.2 * torch.sin(2.0 * math.pi * cond[:, 1])
        - 0.8 * (cond[:, 2] - 0.5)
    )
    # Keep both branches non-negligible; the first test concerns conditional
    # allocation, not vanishing rare modes.
    return 0.10 + 0.80 * torch.sigmoid(logit)


def branch_probabilities(benchmark, cond):
    if benchmark == ONE_BRANCH:
        p1 = torch.zeros(cond.shape[0], device=cond.device, dtype=cond.dtype)
    elif benchmark == TWO_EQUAL:
        p1 = torch.full(
            (cond.shape[0],), 0.5, device=cond.device, dtype=cond.dtype
        )
    elif benchmark == TWO_CONDITIONAL:
        p1 = conditional_branch_one_probability(cond)
    else:
        raise ValueError(f"unsupported topology benchmark {benchmark!r}")
    return torch.stack((1.0 - p1, p1), dim=1)


def topology_log_integrand(benchmark, x, cond):
    log0 = branch_log_integrand(x, cond, branch=0)
    if benchmark == ONE_BRANCH:
        return log0
    log1 = branch_log_integrand(x, cond, branch=1)
    probabilities = branch_probabilities(benchmark, cond).clamp_min(
        torch.finfo(x.dtype).tiny
    )
    return torch.logsumexp(
        torch.stack(
            (
                log0 + probabilities[:, 0].log(),
                log1 + probabilities[:, 1].log(),
            ),
            dim=1,
        ),
        dim=1,
    )


def register_benchmarks():
    geometry.register_benchmarks()
    previous_target = adaptive.target_log_integrand

    def target_log_integrand(benchmark, x, cond):
        if benchmark in BENCHMARKS:
            return topology_log_integrand(benchmark, x, cond)
        return previous_target(benchmark, x, cond)

    adaptive.SUPPORTED_BENCHMARKS.update(BENCHMARKS)
    for benchmark in BENCHMARKS:
        adaptive.BENCHMARK_DIMENSIONS[benchmark] = DIMENSION
        adaptive.ORACLE_INTEGRALS[benchmark] = 1.0
    adaptive.target_log_integrand = target_log_integrand
    adaptive.full_ar_graph = geometry.ladder_graph
    print(
        "32D topology ladder: identical normalized autoregressive geometry "
        "on one or two disconnected branches; exact I(c)=1"
    )


def main():
    register_benchmarks()
    adaptive.main()


if __name__ == "__main__":
    main()
