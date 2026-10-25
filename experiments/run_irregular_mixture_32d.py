#!/usr/bin/env python3
"""Run the original annealed marginal+copula NF on the 32D irregular family.

This uses the first 32 coordinates of exactly the normalized eight-component
construction from the 100D stress test.  Thus it preserves the same gates,
widths, phases and dependency parameters rather than drawing an easier target.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_irregular_mixture_100d as family  # noqa: E402


adaptive = family.adaptive
DIMENSION = 32
BENCHMARK = "irregular_mixture_32"


def configure_family():
    # Restrict the already-created 100D autoregressive family to its first 32
    # coordinates.  Every retained conditional factor stays normalized, so
    # this is an exact normalized lower-dimensional member of the same family.
    parameters = {}
    for name, value in family.PARAMETERS.items():
        if name in {"parent_a", "parent_b", "frequency", "phase"}:
            parameters[name] = value[:, :DIMENSION].clone()
        else:
            parameters[name] = value.clone()
    family.DIMENSION = DIMENSION
    family.BENCHMARK = BENCHMARK
    family.PARAMETERS = parameters
    family._DEVICE_CACHE.clear()


def register_benchmark():
    configure_family()
    family.base.register_benchmark()
    previous_target = adaptive.target_log_integrand
    previous_configure = adaptive.configure_speed_profile

    def target_log_integrand(benchmark, x, cond):
        if benchmark == BENCHMARK:
            return family.irregular_mixture_100_log_integrand(x, cond)
        return previous_target(benchmark, x, cond)

    def configure_profile(profile):
        previous_configure(profile)
        # Keep the original annealed bridge and two-pass marginal/copula fit.
        # The smaller minimum increment avoids forcing a bridge step when the
        # conditional ESS says the current proposal is not ready.
        adaptive.TEMPERATURE_MIN_INCREMENT = 0.001
        adaptive.TEMPERATURE_MAX_INCREMENT = 0.10

    adaptive.SUPPORTED_BENCHMARKS.add(BENCHMARK)
    adaptive.BENCHMARK_DIMENSIONS[BENCHMARK] = DIMENSION
    adaptive.ORACLE_INTEGRALS[BENCHMARK] = 1.0
    adaptive.target_log_integrand = target_log_integrand
    adaptive.full_ar_graph = family.irregular_dependency_superset
    adaptive.configure_speed_profile = configure_profile
    print(
        "Irregular 32D mixture: same eight normalized conditional components "
        "as the 100D family; exact I(c)=1"
    )


def main():
    register_benchmark()
    adaptive.main()


if __name__ == "__main__":
    main()
