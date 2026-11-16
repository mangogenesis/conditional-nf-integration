#!/usr/bin/env python3
"""Smoke-test a configured MadGraph matrix element and phase-space map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from madgraph_process_integrand import ConfiguredMadGraphIntegrand, load_process_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--alpha-s", type=float, default=0.118)
    args = parser.parse_args()
    if args.points <= 0:
        raise ValueError("--points must be positive")

    spec = load_process_spec(args.config)
    evaluator = ConfiguredMadGraphIntegrand(spec, PROJECT, args.alpha_s)
    condition = spec["condition"]
    rng = np.random.default_rng(args.seed)
    unit = rng.random((args.points, evaluator.dimension))
    energy = rng.uniform(float(condition["minimum"]), float(condition["maximum"]), args.points)

    checks = evaluator.phase_space.validate(unit, energy)
    started = time.perf_counter()
    log_values = evaluator.log_integrand(unit, energy)
    elapsed = time.perf_counter() - started
    finite = np.isfinite(log_values)

    print("CONFIGURED MADGRAPH INTEGRAND VALIDATION")
    print(f"process                   = {spec['name']}")
    print(f"dimension                 = {evaluator.dimension}")
    for name, value in checks.items():
        print(f"{name:25s} = {value:.12g}")
    print(f"finite log-integrand frac = {finite.mean():.12g}")
    print(f"cut-rejected fraction     = {1.0 - finite.mean():.12g}")
    print(f"positive integrand frac   = {np.mean(finite):.12g}")
    print(f"throughput points/s       = {args.points / max(elapsed, 1e-12):.3f}")
    if checks["finite_fraction"] != 1.0:
        raise SystemExit("Validation failed: non-finite phase-space momenta")
    if checks["minimum_jacobian"] <= 0.0:
        raise SystemExit("Validation failed: non-positive phase-space Jacobian")
    if finite.mean() <= 0.0:
        raise SystemExit(
            "Validation failed: no phase-space points pass the configured cuts"
        )


if __name__ == "__main__":
    main()
