#!/usr/bin/env python3
"""Fresh heterogeneous training with a nonvanishing uniform safety proposal.

The benchmark and NF architecture are unchanged.  This launcher only keeps the
uniform share of every annealing proposal at ten percent so that an improving
NF cannot remove global exploration before the bridge reaches beta=1.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402


def pop_uniform_fraction():
    option = "--protected-uniform-fraction"
    if option not in sys.argv:
        return 0.10
    index = sys.argv.index(option)
    if len(sys.argv) <= index + 1:
        raise SystemExit(f"{option} requires a value")
    value = float(sys.argv[index + 1])
    del sys.argv[index:index + 2]
    if not 0 < value < 1:
        raise SystemExit(f"{option} must lie in (0,1)")
    return value


def main():
    uniform_fraction = pop_uniform_fraction()
    heterogeneous.adaptive.PROPOSAL_UNIFORM_FRACTION = uniform_fraction
    heterogeneous.adaptive.PROPOSAL_MIN_UNIFORM_FRACTION = uniform_fraction
    print(
        "Protected annealed KL: every proposal retains "
        f"{100 * uniform_fraction:.1f}% uniform mass",
        flush=True,
    )
    heterogeneous.main()


if __name__ == "__main__":
    main()
