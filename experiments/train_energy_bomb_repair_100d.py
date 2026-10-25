#!/usr/bin/env python3
"""Fit a normalized conditional repair NF to 100D energy-bomb discoveries."""

from __future__ import annotations

import argparse
import copy
import json
import math
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


adaptive = irregular.adaptive


def split_by_condition_and_bomb(data, holdout_condition_fraction=0.25):
    """Prevent correlated walkers from leaking across dataset partitions."""
    cond = data["cond"]
    bomb = data["bomb"].long()
    unique_conditions, condition_inverse = torch.unique(cond, dim=0, return_inverse=True)
    condition_count = unique_conditions.shape[0]
    holdout_count = max(1, int(round(condition_count * holdout_condition_fraction)))
    generator = torch.Generator(device="cpu").manual_seed(88421)
    order = torch.randperm(condition_count, generator=generator)
    external_conditions = order[:holdout_count]
    development_conditions = order[holdout_count:]
    external_mask = torch.isin(condition_inverse, external_conditions)

    validation_bombs = []
    for condition_index in development_conditions.tolist():
        candidates = torch.unique(bomb[condition_inverse == condition_index])
        validation_bombs.append(candidates[-1])
    validation_bombs = torch.stack(validation_bombs)
    validation_mask = (~external_mask) & torch.isin(bomb, validation_bombs)
    train_mask = (~external_mask) & (~validation_mask)
    return train_mask, validation_mask, external_mask, condition_count


@torch.no_grad()
def mean_nll(model, x, cond, chunk=2048):
    total = 0.0
    count = 0
    model.eval()
    for start in range(0, x.shape[0], chunk):
        value = -model.log_prob(x[start:start + chunk], cond[start:start + chunk]).reshape(-1)
        total += float(value.double().sum().item())
        count += value.numel()
    return total / max(count, 1)


@torch.no_grad()
def choose_source_component(base_model, x, cond):
    scores = base_model.component_log_probs(x, cond).mean(dim=0)
    return int(scores.argmax().item()), scores.detach().cpu()


def make_bomb_groups(indices, bomb):
    """Build a padded index table so equal-bomb sampling stays on the GPU."""
    groups = [indices[bomb[indices] == value] for value in torch.unique(bomb[indices])]
    lengths = torch.tensor([group.numel() for group in groups], device=indices.device)
    width = int(lengths.max().item())
    table = torch.full(
        (len(groups), width), -1, dtype=torch.long, device=indices.device
    )
    for row, group in enumerate(groups):
        table[row, :group.numel()] = group
    return table, lengths


def sample_equal_bombs(table, lengths, batch_size):
    """Give each bomb equal expected mass, independent of retained point count."""
    rows = torch.randint(0, table.shape[0], (batch_size,), device=table.device)
    columns = torch.floor(
        torch.rand(batch_size, device=table.device) * lengths[rows]
    ).long()
    return table[rows, columns]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--bomb-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--anchor-penalty", type=float, default=1e-5)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--holdout-condition-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("results/irregular_mixture_100_bomb_repair"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    irregular.register_benchmark()
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)
    base_model, base_checkpoint = load_proposal(args.base_checkpoint, device)
    data = torch.load(args.bomb_data, map_location="cpu", weights_only=False)
    x = data["x"].to(device)
    cond = data["cond"].to(device)
    bomb = data["bomb"].long().to(device)
    train_mask, validation_mask, external_mask, condition_count = (
        split_by_condition_and_bomb(
            {"cond": cond.cpu(), "bomb": bomb.cpu()},
            args.holdout_condition_fraction,
        )
    )
    train_mask = train_mask.to(device)
    validation_mask = validation_mask.to(device)
    external_mask = external_mask.to(device)
    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten()
    bomb_table, bomb_lengths = make_bomb_groups(train_indices, bomb)

    source_index, component_scores = choose_source_component(base_model, x[train_mask], cond[train_mask])
    repair = copy.deepcopy(base_model.components[source_index]).to(device)
    # load_proposal deliberately freezes the complete base proposal.  A deep
    # copy preserves those flags, so explicitly make only the repair clone
    # trainable while leaving the original mixture frozen.
    for parameter in repair.parameters():
        parameter.requires_grad_(True)
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in repair.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        repair.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_validation = mean_nll(repair, x[validation_mask], cond[validation_mask])
    initial_validation = best_validation
    initial_external = mean_nll(repair, x[external_mask], cond[external_mask])
    best_state = adaptive.cpu_state_dict(repair)
    best_step = 0
    patience = 0
    history = []

    for step in range(1, args.steps + 1):
        repair.train()
        index = sample_equal_bombs(bomb_table, bomb_lengths, args.batch_size)
        nll = -repair.log_prob(x[index], cond[index]).reshape(-1).mean()
        anchor_loss = torch.zeros((), device=device)
        for name, parameter in repair.named_parameters():
            anchor_loss = anchor_loss + (parameter - anchor[name]).square().mean()
        loss = nll + args.anchor_penalty * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(repair.parameters(), 10.0)
        optimizer.step()

        if step == 1 or step % args.check_interval == 0 or step == args.steps:
            validation = mean_nll(repair, x[validation_mask], cond[validation_mask])
            external = mean_nll(repair, x[external_mask], cond[external_mask])
            row = {
                "step": step,
                "train_nll": float(nll.detach().item()),
                "validation_nll": validation,
                "external_condition_nll": external,
                "anchor_loss": float(anchor_loss.detach().item()),
            }
            history.append(row)
            print(
                f"repair step {step:4d}/{args.steps}: train={row['train_nll']:.6f} "
                f"bomb-val={validation:.6f} external-cond={external:.6f} "
                f"anchor={row['anchor_loss']:.3e} patience={patience}/{args.patience}"
            )
            if validation < best_validation - 1e-4:
                best_validation = validation
                best_state = adaptive.cpu_state_dict(repair)
                best_step = step
                patience = 0
            else:
                patience += 1
            if patience >= args.patience:
                break

    adaptive.load_cpu_state(repair, best_state)
    final_external = mean_nll(repair, x[external_mask], cond[external_mask])
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "benchmark": irregular.BENCHMARK,
            "dimension": irregular.DIMENSION,
            "graph": base_checkpoint["graph"],
            "repair_state_dict": adaptive.cpu_state_dict(repair),
            "source_component": source_index,
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "bomb_data": str(args.bomb_data.resolve()),
            "best_step": best_step,
            "initial_validation_nll": initial_validation,
            "best_validation_nll": best_validation,
            "initial_external_condition_nll": initial_external,
            "final_external_condition_nll": final_external,
            "component_scores": component_scores,
        },
        args.output / "bomb_repair_model.pt",
    )
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    split_summary = {
        "conditions": condition_count,
        "train_points": int(train_mask.sum().item()),
        "validation_points": int(validation_mask.sum().item()),
        "external_condition_points": int(external_mask.sum().item()),
        "source_component": source_index,
        "best_step": best_step,
        "initial_validation_nll": initial_validation,
        "best_validation_nll": best_validation,
        "initial_external_condition_nll": initial_external,
        "final_external_condition_nll": final_external,
    }
    (args.output / "summary.json").write_text(json.dumps(split_summary, indent=2) + "\n")
    print("\nREPAIR TRAINING SUMMARY")
    print(json.dumps(split_summary, indent=2))
    print(f"\nRepair model: {(args.output / 'bomb_repair_model.pt').resolve()}")


if __name__ == "__main__":
    main()
