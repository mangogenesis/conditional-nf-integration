#!/usr/bin/env python3
"""Train a conditional repair specialist on continued 100D basin paths.

External evaluation holds out complete anchors (and therefore their source
condition regions).  Checkpoint selection holds out complete trajectories
from the remaining anchors.  No point from a held-out group is used for
optimization.
"""

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
    choose_source,
    mean_nll,
    train_stage,
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
    parser.add_argument("--anchor-penalty", type=float, default=1e-5)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--external-anchor-fraction", type=float, default=0.25)
    parser.add_argument("--validation-trajectory-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def grouped_split(data, eligible, args):
    anchor = data["anchor_id"].long()
    trajectory = data["trajectory_id"].long()
    anchors = torch.unique(anchor[eligible])
    if len(anchors) < 4:
        raise RuntimeError("At least four eligible anchors are required")
    generator = torch.Generator().manual_seed(args.seed + 1)
    anchors = anchors[torch.randperm(len(anchors), generator=generator)]
    external_count = max(1, int(round(
        len(anchors) * args.external_anchor_fraction
    )))
    external_count = min(external_count, len(anchors) - 2)
    external_anchors = anchors[:external_count]
    development_anchors = anchors[external_count:]
    external = eligible & torch.isin(anchor, external_anchors)
    development = eligible & torch.isin(anchor, development_anchors)

    trajectories = torch.unique(trajectory[development])
    trajectories = trajectories[
        torch.randperm(len(trajectories), generator=generator)
    ]
    validation_count = max(1, int(round(
        len(trajectories) * args.validation_trajectory_fraction
    )))
    validation_count = min(validation_count, len(trajectories) - 1)
    validation_trajectories = trajectories[:validation_count]
    validation = development & torch.isin(
        trajectory, validation_trajectories
    )
    train = development & ~validation
    return train, validation, external, external_anchors, validation_trajectories


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)

    base, base_checkpoint = load_proposal(args.base_checkpoint, device)
    data = torch.load(
        args.continuation_data, map_location="cpu", weights_only=False
    )
    eligible = data["h"].double() >= args.minimum_h
    train, validation, external, external_anchors, validation_trajectories = (
        grouped_split(data, eligible, args)
    )
    x = data["x"].float().to(device)
    cond = data["cond"].float().to(device)
    train = train.to(device)
    validation = validation.to(device)
    external = external.to(device)

    source, component_scores = choose_source(base, x[train], cond[train])
    repair = copy.deepcopy(base.components[source]).to(device)
    for parameter in repair.parameters():
        parameter.requires_grad_(True)
    anchor_state = {
        name: parameter.detach().clone()
        for name, parameter in repair.named_parameters()
    }
    initial_validation = mean_nll(repair, x[validation], cond[validation])
    initial_external = mean_nll(repair, x[external], cond[external])

    bank = adaptive.get_marginal_bank(repair)
    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    for parameter in bank.parameters():
        parameter.requires_grad_(True)
    marginal = train_stage(
        "marginal", repair, bank.parameters(), x, cond, train, validation,
        args.marginal_steps, args, anchor_state,
    )

    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    for parameter in repair.copula.parameters():
        parameter.requires_grad_(True)
    copula = train_stage(
        "copula", repair, repair.copula.parameters(), x, cond, train,
        validation, args.copula_steps, args, anchor_state,
    )
    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    final_validation = mean_nll(repair, x[validation], cond[validation])
    final_external = mean_nll(repair, x[external], cond[external])

    accepted = (
        final_validation < initial_validation
        and final_external < initial_external
    )
    checkpoint = {
        "benchmark": irregular.BENCHMARK,
        "dimension": irregular.DIMENSION,
        "graph": base_checkpoint["graph"],
        "repair_state_dict": adaptive.cpu_state_dict(repair),
        "source_component": source,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "continuation_data": str(args.continuation_data.resolve()),
        "accepted_on_grouped_holdout": accepted,
        "external_anchor_ids": external_anchors,
        "validation_trajectory_ids": validation_trajectories,
        "component_scores": component_scores,
    }
    torch.save(checkpoint, args.output / "continued_basin_repair_model.pt")
    history = marginal["history"] + copula["history"]
    pd.DataFrame(history).to_csv(
        args.output / "training_history.csv", index=False
    )
    summary = {
        "eligible_points": int(eligible.sum()),
        "train_points": int(train.sum()),
        "validation_points": int(validation.sum()),
        "external_points": int(external.sum()),
        "training_anchors": int(torch.unique(
            data["anchor_id"][train.cpu()]
        ).numel()),
        "external_anchors": len(external_anchors),
        "validation_trajectories": len(validation_trajectories),
        "source_component": source,
        "initial_validation_nll": initial_validation,
        "final_validation_nll": final_validation,
        "initial_external_nll": initial_external,
        "final_external_nll": final_external,
        "accepted_on_grouped_holdout": accepted,
        "marginal": {k: v for k, v in marginal.items() if k != "history"},
        "copula": {k: v for k, v in copula.items() if k != "history"},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("\nCONTINUED-BASIN REPAIR SUMMARY")
    print(json.dumps(summary, indent=2))
    if not accepted:
        print("\nREJECTED: grouped holdout NLL did not improve on both splits.")
    else:
        print("\nACCEPTED FOR FRESH IMPORTANCE-SAMPLING VALIDATION.")
    print(f"\nModel: {(args.output / 'continued_basin_repair_model.pt').resolve()}")


if __name__ == "__main__":
    main()
