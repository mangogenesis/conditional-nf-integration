#!/usr/bin/env python3
"""Run the explicit adaptive mixture-of-copula trainer on irregular 32D."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_irregular_mixture_32d as target32  # noqa: E402
import run_adaptive_mixture_irregular_100d as mixture  # noqa: E402


def main():
    # The trainer only needs this small benchmark interface.  Rebinding it
    # keeps all mixture/repair logic identical to the 100D experiment while
    # changing only the normalized target family member and graph dimension.
    benchmark = SimpleNamespace(
        adaptive=target32.adaptive,
        BENCHMARK=target32.BENCHMARK,
        DIMENSION=target32.DIMENSION,
        register_benchmark=target32.register_benchmark,
        irregular_dependency_superset=(
            target32.family.irregular_dependency_superset
        ),
    )
    mixture.irregular = benchmark
    mixture.adaptive = target32.adaptive
    mixture.BENCHMARK = target32.BENCHMARK
    mixture.main()


if __name__ == "__main__":
    main()
