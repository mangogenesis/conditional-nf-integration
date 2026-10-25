#!/usr/bin/env python3
"""Blind multi-structure discovery for the heterogeneous 32D benchmark.

This runner deliberately does not expose the benchmark's four structure labels
to training.  It combines two generic mechanisms already used elsewhere in
the project:

* a persistent annealed-SMC population, which prevents every bridge stage from
  forgetting all but one region; and
* an overcomplete conditional mixture, whose components may specialize and
  whose unused components can subsequently be pruned.

The number of mixture components is a capacity limit, not a statement about
the number of target structures.  Structure labels are used only by the
separate frozen audit script.
"""

from __future__ import annotations

import argparse
import sys

import run_adaptive_mixture_irregular_100d as mixture
import run_bomb_bridge_irregular_100d as bomb
import run_heterogeneous_topology_32d as heterogeneous
import run_smc_clustered_irregular_100d as smc


adaptive = heterogeneous.adaptive


def parse_wrapper_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--discovery-points", type=int, default=4096)
    parser.add_argument("--discovery-train-populations", type=int, default=64)
    parser.add_argument("--discovery-validation-populations", type=int, default=32)
    args, remaining = parser.parse_known_args(argv)
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("fixed condition must lie in [0,1]^3")
    if min(
        args.discovery_points,
        args.discovery_train_populations,
        args.discovery_validation_populations,
    ) <= 0:
        parser.error("discovery budgets must be positive")
    return args, remaining


def main():
    wrapper, remaining = parse_wrapper_arguments(sys.argv[1:])

    # Register the target once, including the fixed-condition sampler.  The
    # reused mixture runner calls register_benchmark again, so replace that
    # second call by a no-op after registration.
    heterogeneous.register_benchmark(tuple(wrapper.fixed_condition))
    mixture.irregular.register_benchmark = lambda: None
    mixture.irregular.DIMENSION = heterogeneous.DIMENSION
    mixture.irregular.irregular_dependency_superset = heterogeneous.dependency_graph

    # Redirect every reused helper to the heterogeneous target.  These module
    # constants are process-local and do not alter any saved source benchmark.
    for module in (mixture, bomb, smc):
        module.BENCHMARK = heterogeneous.BENCHMARK

    # The underlying runner configures its speed profile after this wrapper is
    # entered.  Preserve that configuration, then install discovery budgets.
    original_configure = adaptive.configure_speed_profile

    def configure_with_discovery_budgets(profile):
        original_configure(profile)
        adaptive.UNIFORM_BOOTSTRAP_POINTS = wrapper.discovery_points
        adaptive.MARGINAL_ADAPTIVE_POINTS = wrapper.discovery_points
        adaptive.TEMPERATURE_PROBE_POINTS = wrapper.discovery_points
        adaptive.MARGINAL_TRAIN_CONDITIONS = (
            wrapper.discovery_train_populations
        )
        adaptive.VAL_CONDITIONS = wrapper.discovery_validation_populations

    adaptive.configure_speed_profile = configure_with_discovery_budgets
    adaptive.DIM = heterogeneous.DIMENSION
    # Let the persistent SMC population make arbitrarily small safe progress
    # rather than aborting before its rescue logic can act.
    adaptive.ALLOW_MICRO_ANNEALING = True
    adaptive.TEMPERATURE_MICRO_INCREMENT = 1.0e-4

    print(
        "Blind heterogeneous discovery enabled: persistent annealed SMC + "
        "overcomplete mixture; no structure count or labels are supplied"
    )
    print(
        f"fixed condition={tuple(wrapper.fixed_condition)} "
        f"populations={wrapper.discovery_train_populations}/"
        f"{wrapper.discovery_validation_populations} "
        f"points={wrapper.discovery_points}"
    )

    sys.argv = [sys.argv[0], *remaining]
    smc.main()


if __name__ == "__main__":
    main()
