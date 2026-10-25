#!/usr/bin/env python3
"""Frozen structure-resolved audit for a blind heterogeneous mixture run."""

from __future__ import annotations

import sys

import torch

import audit_heterogeneous_topology_32d as audit
import run_adaptive_mixture_irregular_100d as mixture
import run_heterogeneous_topology_32d as heterogeneous
from nf.conditional_mixture import ConditionalMixtureProposal


def load_mixture(path, device):
    heterogeneous.register_benchmark()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("benchmark") != heterogeneous.BENCHMARK:
        raise ValueError(
            f"expected {heterogeneous.BENCHMARK!r}, "
            f"found {checkpoint.get('benchmark')!r}"
        )
    adaptive = heterogeneous.adaptive
    adaptive.DIM = heterogeneous.DIMENSION
    graph = checkpoint["graph"]
    components = [
        adaptive.make_real_model(graph, None, device)
        for _ in range(int(checkpoint["components"]))
    ]
    maximum = int(checkpoint.get("args", {}).get("max_components", len(components)))
    maximum = max(maximum, len(components))
    model = ConditionalMixtureProposal(
        components, cond_dim=adaptive.COND_DIM, max_components=maximum
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return adaptive, model


def main():
    # Reuse the complete independent audit and change only checkpoint loading.
    audit.load_model = load_mixture
    audit.main()


if __name__ == "__main__":
    main()
