"""Discover an effective copula block, then run persistent-SMC training.

The transformation remains a full 100-dimensional normalized density.  A
pilot population ranks coordinates by conditional dependence.  The smallest
set explaining a requested fraction of that dependence receives the expensive
copula graph; remaining coordinates retain cheap conditional residual edges.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import run_irregular_mixture_100d as irregular
import run_smc_clustered_irregular_100d as smc
from explore_energy_bombs_irregular_100d import (
    initialize_bombs,
    load_proposal,
    run_langevin,
)


adaptive = irregular.adaptive


def parse_dimension_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dimension-pilot-conditions", type=int, default=16)
    parser.add_argument("--dimension-pilot-checkpoint", type=Path)
    parser.add_argument("--dimension-pilot-bombs", type=int, default=4)
    parser.add_argument("--dimension-pilot-walkers", type=int, default=64)
    parser.add_argument("--dimension-pilot-steps", type=int, default=100)
    parser.add_argument("--dimension-pilot-step-size", type=float, default=0.002)
    parser.add_argument("--dimension-explained-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-effective-dimension", type=int, default=8)
    parser.add_argument("--maximum-effective-dimension", type=int, default=48)
    parser.add_argument("--residual-copula-parents", type=int, default=1)
    parser.add_argument("--dimension-report", type=Path)
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def argument_value(arguments, name):
    if name not in arguments:
        return None
    position = arguments.index(name)
    if position + 1 >= len(arguments):
        return None
    return arguments[position + 1]


@torch.no_grad()
def dependence_statistics(points, condition_ids):
    """Average within-condition absolute correlation and coordinate scores."""
    dimension = points.shape[1]
    correlation = torch.zeros(
        dimension, dimension, dtype=torch.double, device=points.device
    )
    groups = 0
    for value in torch.unique(condition_ids):
        selected = points[condition_ids == value].double()
        if selected.shape[0] < 3:
            continue
        centered = selected - selected.mean(dim=0, keepdim=True)
        scale = centered.square().mean(dim=0).sqrt().clamp_min(1e-8)
        standardized = centered / scale
        corr = standardized.T @ standardized / selected.shape[0]
        correlation += corr.abs()
        groups += 1
    correlation /= max(groups, 1)
    correlation.fill_diagonal_(0.0)
    score = correlation.sum(dim=1)
    return correlation, score


def choose_active_dimensions(score, explained_fraction, minimum, maximum):
    order = torch.argsort(score, descending=True)
    ordered = score[order].clamp_min(0)
    total = ordered.sum().clamp_min(1e-300)
    cumulative = torch.cumsum(ordered, dim=0) / total
    required = int(torch.searchsorted(
        cumulative,
        torch.tensor(explained_fraction, device=cumulative.device),
    ).item()) + 1
    count = min(max(required, minimum), maximum, len(score))
    return sorted(int(value) for value in order[:count].tolist()), order, cumulative


def make_adaptive_graph_factory(active, correlation, residual_parents):
    active_set = set(active)

    def graph_factory(dimension):
        graph = {child: [] for child in range(dimension)}
        for child in range(1, dimension):
            eligible = [parent for parent in active if parent < child]
            if child in active_set:
                graph[child] = eligible
            elif residual_parents > 0 and eligible:
                ranked = sorted(
                    eligible,
                    key=lambda parent: float(correlation[child, parent]),
                    reverse=True,
                )
                graph[child] = ranked[:residual_parents]
        return graph

    return graph_factory


def discover_dimension(args, remaining):
    checkpoint = args.dimension_pilot_checkpoint
    if checkpoint is None:
        raise ValueError(
            "Adaptive-dimension discovery requires --dimension-pilot-checkpoint "
            "so its pilot walkers start from an existing broad proposal."
        )
    device_name = argument_value(remaining, "--device") or "cuda"
    device = torch.device(device_name)
    adaptive.DIM = irregular.DIMENSION
    adaptive.COND_DIM = 3
    proposal, _ = load_proposal(Path(checkpoint), device)
    conditions = adaptive.sample_conditions(args.dimension_pilot_conditions, device)
    y, repeated_conditions = initialize_bombs(
        proposal,
        conditions,
        args.dimension_pilot_bombs,
        args.dimension_pilot_walkers,
        radius=0.25,
    )
    _, y, _ = run_langevin(
        y,
        repeated_conditions,
        steps=args.dimension_pilot_steps,
        step_size=args.dimension_pilot_step_size,
        start_temperature=1.0,
        end_temperature=0.15,
        gradient_clip=50.0,
        report_interval=max(args.dimension_pilot_steps, 1),
    )
    points = torch.sigmoid(y)
    per_condition = args.dimension_pilot_bombs * args.dimension_pilot_walkers
    condition_ids = torch.arange(
        args.dimension_pilot_conditions, device=device
    ).repeat_interleave(per_condition)
    correlation, score = dependence_statistics(points, condition_ids)
    active, order, cumulative = choose_active_dimensions(
        score,
        args.dimension_explained_fraction,
        args.minimum_effective_dimension,
        args.maximum_effective_dimension,
    )
    graph_factory = make_adaptive_graph_factory(
        active, correlation, args.residual_copula_parents
    )
    graph = graph_factory(irregular.DIMENSION)
    edge_count = sum(len(parents) for parents in graph.values())
    full_edge_count = irregular.DIMENSION * (irregular.DIMENSION - 1) // 2
    report = {
        "ambient_dimension": irregular.DIMENSION,
        "selected_effective_dimension": len(active),
        "active_coordinates": active,
        "explained_dependency_fraction": float(cumulative[len(active) - 1]),
        "selected_graph_edges": edge_count,
        "dense_graph_edges": full_edge_count,
        "residual_parents_per_coordinate": args.residual_copula_parents,
        "coordinate_ranking": [int(value) for value in order.tolist()],
        "coordinate_scores": [float(value) for value in score.tolist()],
    }
    print("\nADAPTIVE EFFECTIVE-DIMENSION PILOT")
    print(json.dumps(report, indent=2))
    if args.dimension_report is not None:
        args.dimension_report.parent.mkdir(parents=True, exist_ok=True)
        args.dimension_report.write_text(json.dumps(report, indent=2) + "\n")
    return graph_factory


def main():
    args, remaining = parse_dimension_arguments(sys.argv[1:])
    graph_factory = discover_dimension(args, remaining)
    irregular.irregular_dependency_superset = graph_factory
    sys.argv = [sys.argv[0], *remaining]
    print(
        "Adaptive complexity allocation enabled: exact 100D density with a "
        "learned expensive copula block and cheap residual coordinates"
    )
    smc.main()


if __name__ == "__main__":
    main()
