#!/usr/bin/env python3
"""Train an all-anchor specialist for condition-localized mixture repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_irregular_mixture_100d as irregular  # noqa: E402
from explore_energy_bombs_irregular_100d import load_proposal  # noqa: E402
from train_belief_copula_repair_100d import (  # noqa: E402
    choose_source, mean_nll, train_stage,
)


adaptive = irregular.adaptive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-data", type=Path, required=True)
    parser.add_argument("--minimum-h", type=float, default=10.0)
    parser.add_argument("--marginal-steps", type=int, default=1200)
    parser.add_argument("--copula-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--anchor-penalty", type=float, default=1e-4)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--validation-trajectory-fraction", type=float, default=0.2)
    parser.add_argument("--maximum-gate-centers", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def trajectory_split(data, eligible, args):
    trajectory = data["trajectory_id"].long()
    unique = torch.unique(trajectory[eligible])
    generator = torch.Generator().manual_seed(args.seed + 1)
    unique = unique[torch.randperm(len(unique), generator=generator)]
    count = max(1, int(round(len(unique) * args.validation_trajectory_fraction)))
    count = min(count, len(unique) - 1)
    validation_ids = unique[:count]
    validation = eligible & torch.isin(trajectory, validation_ids)
    train = eligible & ~validation
    return train, validation, validation_ids


@torch.no_grad()
def farthest_centers(conditions, maximum):
    unique = torch.unique(conditions.double(), dim=0)
    if len(unique) <= maximum:
        return unique.float()
    selected = [0]
    distance = torch.full((len(unique),), torch.inf, dtype=torch.double)
    for _ in range(1, maximum):
        newest = unique[selected[-1]]
        distance = torch.minimum(
            distance, (unique - newest).square().sum(dim=1)
        )
        distance[selected] = -1.0
        selected.append(int(distance.argmax()))
    return unique[selected].float()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)

    base, base_checkpoint = load_proposal(args.base_checkpoint, device)
    data = torch.load(args.continuation_data, map_location="cpu", weights_only=False)
    eligible = data["h"].double() >= args.minimum_h
    train, validation, validation_ids = trajectory_split(data, eligible, args)
    x = data["x"].float().to(device)
    cond = data["cond"].float().to(device)
    train_device = train.to(device)
    validation_device = validation.to(device)

    source, component_scores = choose_source(
        base, x[train_device], cond[train_device]
    )
    repair = copy.deepcopy(base.components[source]).to(device)
    for parameter in repair.parameters():
        parameter.requires_grad_(True)
    anchor_state = {
        name: parameter.detach().clone()
        for name, parameter in repair.named_parameters()
    }
    initial_validation = mean_nll(
        repair, x[validation_device], cond[validation_device]
    )

    bank = adaptive.get_marginal_bank(repair)
    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    for parameter in bank.parameters():
        parameter.requires_grad_(True)
    marginal = train_stage(
        "marginal", repair, bank.parameters(), x, cond,
        train_device, validation_device, args.marginal_steps, args, anchor_state,
    )

    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    for parameter in repair.copula.parameters():
        parameter.requires_grad_(True)
    copula = train_stage(
        "copula", repair, repair.copula.parameters(), x, cond,
        train_device, validation_device, args.copula_steps, args, anchor_state,
    )
    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    final_validation = mean_nll(
        repair, x[validation_device], cond[validation_device]
    )
    gate_centers = farthest_centers(
        data["cond"][eligible], args.maximum_gate_centers
    )

    checkpoint = {
        "benchmark": irregular.BENCHMARK,
        "dimension": irregular.DIMENSION,
        "graph": base_checkpoint["graph"],
        "repair_state_dict": adaptive.cpu_state_dict(repair),
        "source_component": source,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "continuation_data": str(args.continuation_data.resolve()),
        "gate_centers": gate_centers,
        "validation_trajectory_ids": validation_ids,
        "component_scores": component_scores,
    }
    model_path = args.output / "localized_basin_repair_model.pt"
    torch.save(checkpoint, model_path)
    pd.DataFrame(marginal["history"] + copula["history"]).to_csv(
        args.output / "training_history.csv", index=False
    )
    summary = {
        "eligible_points": int(eligible.sum()),
        "train_points": int(train.sum()),
        "validation_points": int(validation.sum()),
        "anchors_used": int(torch.unique(data["anchor_id"][eligible]).numel()),
        "training_trajectories": int(torch.unique(
            data["trajectory_id"][train]
        ).numel()),
        "validation_trajectories": len(validation_ids),
        "gate_centers": len(gate_centers),
        "source_component": source,
        "initial_validation_nll": initial_validation,
        "final_validation_nll": final_validation,
        "trajectory_holdout_improved": final_validation < initial_validation,
        "warning": (
            "This specialist is not safe as a global replacement. It must be "
            "used through the normalized condition-localized mixture gate."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nLOCALIZED BASIN SPECIALIST")
    print(json.dumps(summary, indent=2))
    print(f"\nModel: {model_path.resolve()}")


if __name__ == "__main__":
    main()
