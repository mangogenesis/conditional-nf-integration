#!/usr/bin/env python3
"""Validate the 20-D phase-space map and MadGraph matrix-element call."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from madgraph_tth_integrand import MadGraphTTHIntegrand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--standalone",
        type=Path,
        default=Path("generated/gg_tth_full_standalone"),
    )
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--sqrt-s-min", type=float, default=600.0)
    parser.add_argument("--sqrt-s-max", type=float, default=2000.0)
    parser.add_argument("--width-window", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    evaluator = MadGraphTTHIntegrand(
        args.standalone, width_window=args.width_window
    )
    u = rng.random((args.points, 20))
    sqrt_s = rng.uniform(args.sqrt_s_min, args.sqrt_s_max, args.points)

    checks = evaluator.phase_space.validate(u, sqrt_s)
    start = time.perf_counter()
    logf = evaluator.log_integrand(u, sqrt_s)
    elapsed = time.perf_counter() - start
    finite = np.isfinite(logf)

    print("MADGRAPH ttH INTEGRAND VALIDATION")
    print(f"points                    = {args.points:,}")
    print(f"sqrt(s_hat) range         = {args.sqrt_s_min:g} .. {args.sqrt_s_max:g} GeV")
    print(f"resonance width window    = +/- {args.width_window:g} Gamma")
    print(f"momentum residual         = {checks['max_momentum_residual']:.3e} GeV")
    print(f"mass-shell residual       = {checks['max_mass_shell_residual']:.3e} GeV^2")
    print(f"minimum phase Jacobian    = {checks['minimum_jacobian']:.3e}")
    print(f"finite mapping fraction   = {checks['finite_fraction']:.6f}")
    print(f"finite integrand fraction = {finite.mean():.6f}")
    if finite.any():
        print(f"log f range               = {logf[finite].min():.6g} .. {logf[finite].max():.6g}")
    print(f"matrix-element throughput = {args.points / max(elapsed, 1e-12):.1f} points/s")

    if checks["max_momentum_residual"] > 1e-7:
        raise RuntimeError("Momentum-conservation validation failed")
    if checks["max_mass_shell_residual"] > 1e-6:
        raise RuntimeError("Final-state mass-shell validation failed")
    if finite.mean() < 0.99:
        raise RuntimeError("Too many non-finite MadGraph integrand evaluations")
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
