#!/usr/bin/env python3
"""Train a dense autoregressive NF on the unchanged heterogeneous benchmark.

The target definition remains in ``run_heterogeneous_topology_32d.py``.  This
launcher changes only the model graph: coordinate x_j may depend on every
earlier coordinate x_0,...,x_{j-1}.  In particular, x_0 can communicate which
disconnected branch is active to all later conditional densities.
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


def full_autoregressive_graph(dim):
    """Complete ordered DAG: every variable sees all earlier variables."""
    return {child: list(range(child)) for child in range(dim)}


def main():
    fixed_condition = heterogeneous._pop_fixed_condition()
    heterogeneous.register_benchmark(fixed_condition)
    heterogeneous.adaptive.full_ar_graph = full_autoregressive_graph
    edges = sum(
        len(parents) for parents in full_autoregressive_graph(
            heterogeneous.DIMENSION
        ).values()
    )
    print(
        "Model graph override: full ordered autoregressive graph; "
        f"edges={edges}; benchmark unchanged"
    )
    heterogeneous.adaptive.main()


if __name__ == "__main__":
    main()
