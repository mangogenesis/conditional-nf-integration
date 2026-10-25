#!/usr/bin/env python3
"""Train a two-pass marginal/copula specialist on belief-guided discoveries."""

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
from explore_energy_bombs_irregular_100d import load_proposal, logit  # noqa: E402


adaptive = irregular.adaptive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--belief-data", type=Path, required=True)
    parser.add_argument("--minimum-h", type=float, default=20.0)
    parser.add_argument("--maximum-centers", type=int, default=128)
    parser.add_argument("--centers-per-condition", type=int, default=5)
    parser.add_argument("--augment-per-center", type=int, default=32)
    parser.add_argument("--augmentation-radius", type=float, default=0.05)
    parser.add_argument("--marginal-steps", type=int, default=1200)
    parser.add_argument("--copula-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-penalty", type=float, default=1e-6)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--holdout-condition-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=93721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def mean_nll(model, x, cond, chunk=2048):
    total = 0.0
    count = 0
    model.eval()
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        value = -model.log_prob(x[start:stop], cond[start:stop]).reshape(-1)
        total += float(value.double().sum())
        count += value.numel()
    return total / max(count, 1)


@torch.no_grad()
def select_and_expand(data, args, device):
    eligible = (data["round"] > 0) & (data["h"] >= args.minimum_h)
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if not len(indices):
        raise RuntimeError("No belief-discovery points satisfy the selection criteria")
    eligible_conditions = data["cond"][indices]
    _, condition_inverse = torch.unique(
        eligible_conditions, dim=0, return_inverse=True
    )
    selected_groups = []
    for condition_index in range(int(condition_inverse.max()) + 1):
        local = indices[condition_inverse == condition_index]
        local_order = torch.argsort(data["h"][local], descending=True)
        selected_groups.append(local[local_order[: args.centers_per_condition]])
    indices = torch.cat(selected_groups)
    if len(indices) > args.maximum_centers:
        # Round-robin ranks preserve condition coverage before adding second,
        # third, ... centers from the same condition.
        balanced = []
        for rank in range(args.centers_per_condition):
            for group in selected_groups:
                if rank < len(group):
                    balanced.append(group[rank])
                    if len(balanced) >= args.maximum_centers:
                        break
            if len(balanced) >= args.maximum_centers:
                break
        indices = torch.stack(balanced)
    centers = data["x"][indices].to(device)
    conditions = data["cond"][indices].to(device)
    center_h = data["h"][indices].double()
    y = logit(centers)[:, None, :] + args.augmentation_radius * torch.randn(
        len(centers), args.augment_per_center, centers.shape[-1], device=device
    )
    x = torch.sigmoid(y).reshape(-1, centers.shape[-1])
    cond = conditions[:, None, :].expand(
        -1, args.augment_per_center, -1
    ).reshape(-1, conditions.shape[-1])
    center_id = torch.arange(len(centers), device=device).repeat_interleave(
        args.augment_per_center
    )
    selected_condition_count = len(torch.unique(data["cond"][indices], dim=0))
    return x, cond, center_id, center_h, selected_condition_count


def split_conditions(cond, center_id, seed, holdout_fraction):
    unique, inverse = torch.unique(cond.detach().cpu(), dim=0, return_inverse=True)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique), generator=generator)
    holdout_count = max(1, int(round(len(unique) * holdout_fraction)))
    if len(unique) < 3:
        raise RuntimeError("At least three distinct conditions are needed")
    holdout_count = min(holdout_count, len(unique) - 2)
    external_conditions = order[:holdout_count]
    development_conditions = order[holdout_count:]
    external = torch.isin(inverse, external_conditions)
    development = ~external
    development_centers = torch.unique(center_id.detach().cpu()[development])
    center_order = development_centers[
        torch.randperm(len(development_centers), generator=generator)
    ]
    validation_center_count = max(1, len(center_order) // 5)
    validation_centers = center_order[:validation_center_count]
    validation = development & torch.isin(
        center_id.detach().cpu(), validation_centers
    )
    train = development & ~validation
    return train, validation, external, len(unique)


def choose_source(base, x, cond):
    with torch.no_grad():
        scores = base.component_log_probs(x, cond).mean(dim=0)
    return int(scores.argmax()), scores.detach().cpu()


def train_stage(
    label, model, parameters, x, cond, train_mask, validation_mask,
    steps, args, anchor,
):
    parameters = list(parameters)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    best_value = mean_nll(model, x[validation_mask], cond[validation_mask])
    initial_value = best_value
    best_state = adaptive.cpu_state_dict(model)
    best_step = 0
    patience = 0
    history = []
    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten()

    for step in range(1, steps + 1):
        model.train()
        selection = train_indices[
            torch.randint(0, len(train_indices), (args.batch_size,), device=x.device)
        ]
        nll = -model.log_prob(x[selection], cond[selection]).reshape(-1).mean()
        penalty = torch.zeros((), device=x.device)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                penalty = penalty + (parameter - anchor[name]).square().mean()
        loss = nll + args.anchor_penalty * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()

        if step == 1 or step % args.check_interval == 0 or step == steps:
            value = mean_nll(model, x[validation_mask], cond[validation_mask])
            row = {
                "stage": label, "step": step,
                "train_nll": float(nll.detach()),
                "validation_nll": value,
                "best_validation_nll": best_value,
            }
            history.append(row)
            print(
                f"{label} step {step:4d}/{steps}: train={row['train_nll']:.6f} "
                f"val={value:.6f} best={best_value:.6f} "
                f"patience={patience}/{args.patience}"
            )
            if value < best_value - 1e-4:
                best_value = value
                best_state = adaptive.cpu_state_dict(model)
                best_step = step
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break
    adaptive.load_cpu_state(model, best_state)
    return {
        "label": label,
        "initial_validation_nll": initial_value,
        "best_validation_nll": best_value,
        "best_step": best_step,
        "history": history,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)
    base, base_checkpoint = load_proposal(args.base_checkpoint, device)
    data = torch.load(args.belief_data, map_location="cpu", weights_only=False)
    x, cond, center_id, center_h, selected_condition_count = select_and_expand(
        data, args, device
    )
    train, validation, external, condition_count = split_conditions(
        cond, center_id, args.seed + 1, args.holdout_condition_fraction
    )
    train = train.to(device)
    validation = validation.to(device)
    external = external.to(device)
    source, component_scores = choose_source(base, x[train], cond[train])
    repair = copy.deepcopy(base.components[source]).to(device)
    for parameter in repair.parameters():
        parameter.requires_grad_(True)
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in repair.named_parameters()
    }
    initial_external = mean_nll(repair, x[external], cond[external])

    bank = adaptive.get_marginal_bank(repair)
    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    for parameter in bank.parameters():
        parameter.requires_grad_(True)
    marginal = train_stage(
        "marginal", repair, bank.parameters(), x, cond, train, validation,
        args.marginal_steps, args, anchor,
    )

    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    for parameter in repair.copula.parameters():
        parameter.requires_grad_(True)
    copula = train_stage(
        "copula", repair, repair.copula.parameters(), x, cond, train, validation,
        args.copula_steps, args, anchor,
    )
    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    final_external = mean_nll(repair, x[external], cond[external])

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "benchmark": irregular.BENCHMARK,
            "dimension": irregular.DIMENSION,
            "graph": base_checkpoint["graph"],
            "repair_state_dict": adaptive.cpu_state_dict(repair),
            "source_component": source,
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "belief_data": str(args.belief_data.resolve()),
            "selection": {
                "minimum_h": args.minimum_h,
                "centers": int(center_h.numel()),
                "conditions": selected_condition_count,
                "minimum_center_h": float(center_h.min()),
                "maximum_center_h": float(center_h.max()),
                "augmentation_radius": args.augmentation_radius,
            },
            "component_scores": component_scores,
        },
        args.output / "belief_copula_repair_model.pt",
    )
    pd.DataFrame(marginal["history"] + copula["history"]).to_csv(
        args.output / "training_history.csv", index=False
    )
    summary = {
        "conditions": condition_count,
        "centers": int(center_h.numel()),
        "selected_conditions": selected_condition_count,
        "augmented_points": len(x),
        "train_points": int(train.sum()),
        "validation_points": int(validation.sum()),
        "external_points": int(external.sum()),
        "source_component": source,
        "initial_external_nll": initial_external,
        "final_external_nll": final_external,
        "marginal": {key: value for key, value in marginal.items() if key != "history"},
        "copula": {key: value for key, value in copula.items() if key != "history"},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nBELIEF COPULA REPAIR SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nRepair model: {(args.output / 'belief_copula_repair_model.pt').resolve()}")


if __name__ == "__main__":
    main()
