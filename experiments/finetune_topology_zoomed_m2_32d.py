#!/usr/bin/env python3
"""Zoomed within-condition M2 refinement for the 32D topology benchmark.

The frozen input model first discovers high-M2 phase-space points at difficult
conditions.  Independent train and validation caches then use an exact mixture
of the global defensive proposal and small uniform boxes around those points.
The first marginal is frozen, preserving the learned two-branch topology.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import finetune_conditional_renyi_32d as renyi  # noqa: E402
import finetune_topology_within_condition_m2_32d as base  # noqa: E402
import run_topology_ladder_32d as topology  # noqa: E402
import train_local_conditional_continuation_32d as local  # noqa: E402
import train_topology_branch_conditional_32d as branch  # noqa: E402
from nf.regional_m2 import (  # noqa: E402
    box_mixture_log_prob,
    diverse_weighted_centers,
    sample_box_mixture,
)


DIM = topology.DIMENSION
COND_DIM = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--priority-conditions-file", type=Path, required=True)
    parser.add_argument("--priority-count", type=int, default=10)
    parser.add_argument("--anchor", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--condition-radius", type=float, default=0.25)
    parser.add_argument("--priority-condition-fraction", type=float, default=0.50)
    parser.add_argument("--priority-jitter-radius", type=float, default=0.01)

    parser.add_argument("--discovery-evals", type=int, default=500_000)
    parser.add_argument("--discovery-keep", type=int, default=256)
    parser.add_argument("--centers-per-condition", type=int, default=4)
    parser.add_argument("--minimum-center-distance", type=float, default=0.05)
    parser.add_argument("--discovery-chunk", type=int, default=10_000)

    parser.add_argument("--train-conditions", type=int, default=128)
    parser.add_argument("--validation-conditions", type=int, default=64)
    parser.add_argument("--points-per-condition", type=int, default=32768)
    parser.add_argument("--cache-condition-block", type=int, default=4)
    parser.add_argument("--anchor-fraction", type=float, default=0.10)
    parser.add_argument("--uniform-fraction", type=float, default=0.02)
    parser.add_argument("--regional-sample-fraction", type=float, default=0.25)
    parser.add_argument("--regional-box-halfwidth", type=float, default=0.03)

    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--condition-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument("--tail-weight", type=float, default=0.50)
    parser.add_argument("--anchor-penalty", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr-plateau-patience", type=int, default=4)
    parser.add_argument("--lr-reduction-factor", type=float, default=0.3)
    parser.add_argument("--maximum-lr-reductions", type=int, default=3)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    positive = (
        args.priority_count, args.discovery_evals, args.discovery_keep,
        args.centers_per_condition, args.discovery_chunk,
        args.train_conditions, args.validation_conditions,
        args.points_per_condition, args.cache_condition_block, args.steps,
        args.condition_batch, args.check_interval, args.patience,
    )
    if min(positive) <= 0:
        parser.error("all count and budget arguments must be positive")
    for name in (
        "priority_condition_fraction", "uniform_fraction",
        "regional_sample_fraction", "tail_fraction",
    ):
        value = getattr(args, name)
        if not 0.0 < value < 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in (0,1)")
    if args.condition_radius <= 0 or args.priority_jitter_radius < 0:
        parser.error("condition radius must be positive and jitter nonnegative")
    if not 0 < args.regional_box_halfwidth <= 1:
        parser.error("regional box halfwidth must lie in (0,1]")
    if args.minimum_center_distance < 0:
        parser.error("minimum center distance must be nonnegative")
    return args


def load_priority_conditions(path, count, radius):
    frame = pd.read_csv(path)
    missing = {"c0", "c1", "c2"}.difference(frame.columns)
    if missing:
        raise ValueError(f"priority file lacks columns {sorted(missing)}")
    if "radius" in frame.columns:
        selected = frame[np.isclose(frame["radius"], radius)]
        if len(selected):
            frame = selected
    if "normalized_ess" in frame.columns:
        frame = frame.sort_values("normalized_ess", ascending=True)
    frame = frame.head(count).copy()
    if len(frame) == 0:
        raise ValueError("priority-condition selection is empty")
    values = frame[["c0", "c1", "c2"]].to_numpy(np.float32)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("priority conditions must lie in [0,1]^3")
    return frame, values


def defensive_log_prob(proposal, x, cond, uniform_fraction):
    logq = proposal.log_prob(x, cond).reshape(-1)
    return torch.logaddexp(
        logq + math.log1p(-uniform_fraction),
        torch.full_like(logq, math.log(uniform_fraction)),
    ), logq


@torch.no_grad()
def discover_centers(
    adaptive, proposal, benchmark, conditions, args, device
):
    all_centers = []
    records = []
    for condition_index, condition in enumerate(conditions):
        best_x = None
        best_logc = None
        remaining = args.discovery_evals
        condition_tensor = torch.tensor(
            condition, dtype=torch.float32, device=device
        )
        while remaining:
            count = min(remaining, args.discovery_chunk)
            cond = condition_tensor.expand(count, COND_DIM)
            x, logr = adaptive.sample_from_proposal(
                proposal_model=proposal,
                n=count,
                cond=cond,
                device=device,
                uniform_fraction=args.uniform_fraction,
            )
            logq = proposal.log_prob(x, cond).reshape(-1)
            logf = adaptive.target_log_integrand(
                benchmark, x, cond
            ).reshape(-1)
            logc = 2.0 * logf.double() - logq.double() - logr.double()
            take = min(args.discovery_keep, count)
            values, indices = torch.topk(logc, take)
            candidate_x = x[indices]
            if best_x is None:
                best_x, best_logc = candidate_x, values
            else:
                joined_x = torch.cat((best_x, candidate_x))
                joined_logc = torch.cat((best_logc, values))
                keep = torch.topk(
                    joined_logc, min(args.discovery_keep, len(joined_logc))
                ).indices
                best_x = joined_x[keep]
                best_logc = joined_logc[keep]
            remaining -= count
        center_count = min(args.centers_per_condition, len(best_x))
        centers = diverse_weighted_centers(
            best_x, best_logc, center_count,
            minimum_distance=args.minimum_center_distance,
        )
        # Keep centers on observed high-contribution samples so boxes are
        # anchored on genuine target-support points.
        centers = best_x[torch.cdist(centers, best_x).argmin(dim=1)]
        all_centers.append(centers.detach())
        for center_index, center in enumerate(centers):
            record = {
                "priority_condition_index": condition_index,
                "center_index": center_index,
                "c0": float(condition[0]),
                "c1": float(condition[1]),
                "c2": float(condition[2]),
            }
            record.update({f"x{j}": float(center[j]) for j in range(DIM)})
            records.append(record)
        print(
            f"discovery condition={condition_index + 1}/{len(conditions)} "
            f"max-log-contribution={float(best_logc.max()):.5f} "
            f"centers={len(centers)}",
            flush=True,
        )
    return torch.stack(all_centers), pd.DataFrame(records)


def sample_cache_conditions(args, count, priority_conditions, device):
    sampler = local.make_condition_sampler(
        args.anchor, args.condition_radius, args.anchor_fraction
    )
    conditions = sampler(count, device)
    priority_count = min(
        count, max(1, int(round(count * args.priority_condition_fraction)))
    )
    priority = torch.as_tensor(
        priority_conditions, dtype=torch.float32, device=device
    )
    chosen = torch.randint(0, len(priority), (priority_count,), device=device)
    selected = priority[chosen]
    if args.priority_jitter_radius:
        selected = selected + (
            2.0 * torch.rand_like(selected) - 1.0
        ) * args.priority_jitter_radius
    conditions[:priority_count] = selected.clamp(0.0, 1.0)
    return conditions, priority_count


@torch.no_grad()
def build_zoomed_cache(
    adaptive, proposal, benchmark, priority_conditions, region_centers,
    condition_count, args, device, seed, label,
):
    adaptive.seed_everything(seed)
    conditions, priority_count = sample_cache_conditions(
        args, condition_count, priority_conditions, device
    )
    priority_tensor = torch.as_tensor(
        priority_conditions, dtype=torch.float32, device=device
    )
    xs, conds, logfs, logss = [], [], [], []
    regional_groups = 0
    for start in range(0, condition_count, args.cache_condition_block):
        anchors = conditions[start:start + args.cache_condition_block]
        groups = len(anchors)
        group_x, group_cond, group_logf, group_logs = [], [], [], []
        for offset, anchor in enumerate(anchors):
            cond = anchor.expand(args.points_per_condition, COND_DIM)
            x, _ = adaptive.sample_from_proposal(
                proposal_model=proposal,
                n=args.points_per_condition,
                cond=cond,
                device=device,
                uniform_fraction=args.uniform_fraction,
            )
            is_priority = start + offset < priority_count
            if is_priority:
                nearest = int(torch.cdist(
                    anchor[None, :], priority_tensor
                ).argmin())
                centers = region_centers[nearest]
                use_local = (
                    torch.rand(args.points_per_condition, device=device)
                    < args.regional_sample_fraction
                )
                if use_local.any():
                    x[use_local] = sample_box_mixture(
                        centers, int(use_local.sum()),
                        args.regional_box_halfwidth,
                    )
                logr, _ = defensive_log_prob(
                    proposal, x, cond, args.uniform_fraction
                )
                logg = box_mixture_log_prob(
                    x, centers, args.regional_box_halfwidth
                )
                logs = torch.logaddexp(
                    logr + math.log1p(-args.regional_sample_fraction),
                    logg + math.log(args.regional_sample_fraction),
                )
                regional_groups += 1
            else:
                logs, _ = defensive_log_prob(
                    proposal, x, cond, args.uniform_fraction
                )
            logf = adaptive.target_log_integrand(
                benchmark, x, cond
            ).reshape(-1)
            group_x.append(x)
            group_cond.append(cond)
            group_logf.append(logf)
            group_logs.append(logs)
        xs.append(torch.stack(group_x))
        conds.append(torch.stack(group_cond))
        logfs.append(torch.stack(group_logf))
        logss.append(torch.stack(group_logs))

    cache = {
        "x": torch.cat(xs),
        "cond": torch.cat(conds),
        "logf": torch.cat(logfs),
        "logr": torch.cat(logss),
        "label": label,
    }
    alpha = torch.softmax((cache["logf"] - cache["logr"]).double(), dim=1)
    raw_ess = 1.0 / alpha.square().sum(dim=1)
    print(f"\nZOOMED M2 CACHE: {label}", flush=True)
    print(f"conditions             = {condition_count:,}", flush=True)
    print(f"priority groups        = {priority_count:,}", flush=True)
    print(f"regional groups        = {regional_groups:,}", flush=True)
    print(f"points / condition     = {args.points_per_condition:,}", flush=True)
    print(f"regional sample share  = {args.regional_sample_fraction:.3f}", flush=True)
    print(f"regional box halfwidth = {args.regional_box_halfwidth:.4f}", flush=True)
    print(f"median raw ESS         = {float(raw_ess.median()):.2f}", flush=True)
    print(
        f"q05 raw ESS            = {float(torch.quantile(raw_ess, 0.05)):.2f}",
        flush=True,
    )
    print(f"minimum raw ESS        = {float(raw_ess.min()):.2f}", flush=True)
    return cache


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    adaptive, model, source_checkpoint, benchmark = local.load_model(
        args.checkpoint, device
    )
    if benchmark != topology.TWO_CONDITIONAL:
        raise ValueError("this refinement requires topology_two_conditional_32")
    priority_frame, priority_conditions = load_priority_conditions(
        args.priority_conditions_file, args.priority_count,
        args.condition_radius,
    )
    priority_frame.to_csv(args.output / "priority_conditions.csv", index=False)

    model.to(device)
    proposal = renyi.freeze_copy(model, device)
    region_centers, center_frame = discover_centers(
        adaptive, proposal, benchmark, priority_conditions, args, device
    )
    center_frame.to_csv(args.output / "regional_centers.csv", index=False)
    torch.save({
        "priority_conditions": torch.as_tensor(priority_conditions),
        "centers": region_centers.cpu(),
        "box_halfwidth": args.regional_box_halfwidth,
    }, args.output / "zoom_regions.pt")

    train_cache = build_zoomed_cache(
        adaptive, proposal, benchmark, priority_conditions, region_centers,
        args.train_conditions, args, device, args.seed + 1, "train",
    )
    validation_cache = build_zoomed_cache(
        adaptive, proposal, benchmark, priority_conditions, region_centers,
        args.validation_conditions, args, device, args.seed + 2, "validation",
    )

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.marginals[0].parameters():
        parameter.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    reference = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # The topology marginal is frozen, so branch terms are diagnostics rather
    # than optimization terms.
    args.freeze_topology_marginal = True
    args.branch_weight = 0.0
    args.worst_fraction = 0.20
    args.worst_weight = 0.50
    initial = base.validation_metrics(model, validation_cache, args)
    best = initial
    best_state = branch.cpu_state(model)
    best_step = 0
    patience = plateau = reductions = 0
    history = []
    print(
        "\nZOOMED WITHIN-CONDITION M2 REFINEMENT: "
        f"initial R2={initial['mean_r2']:.6f} "
        f"tail-R2={initial['mean_tail_r2']:.6f} "
        f"branch-MAE={initial['branch_mean_absolute_mass_error']:.6f}",
        flush=True,
    )

    for start in range(0, args.steps, args.check_interval):
        block = min(args.check_interval, args.steps - start)
        losses = []
        model.train()
        for _ in range(block):
            indices = torch.randint(
                len(train_cache["x"]),
                (args.condition_batch,), device=device,
            )
            full, tail = base.train_group_losses(
                model, train_cache, indices, args
            )
            loss = full.mean() + args.tail_weight * tail.mean()
            loss = loss + args.anchor_penalty * branch.parameter_anchor(
                parameters, reference
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))

        step = start + block
        current = base.validation_metrics(model, validation_cache, args)
        threshold = args.minimum_improvement * max(abs(best["score"]), 1.0)
        improved = current["score"] < best["score"] - threshold
        if improved:
            best = current
            best_state = branch.cpu_state(model)
            best_step = step
            patience = plateau = 0
        else:
            patience += 1
            plateau += 1

        if (
            plateau >= args.lr_plateau_patience
            and reductions < args.maximum_lr_reductions
        ):
            _, reduced = branch.reduce_learning_rate(
                optimizer, args.lr_reduction_factor,
                args.minimum_learning_rate,
            )
            if reduced:
                reductions += 1
                patience = plateau = 0
                model.load_state_dict(best_state, strict=True)
                optimizer.state.clear()
        lr = min(group["lr"] for group in optimizer.param_groups)
        history.append({
            "step": step,
            "train_loss": float(np.mean(losses)),
            "learning_rate": lr,
            "best_score": best["score"],
            **{f"validation_{key}": value for key, value in current.items()},
        })
        print(
            f"step {step:5d}/{args.steps}: train={np.mean(losses):.6f} "
            f"R2={current['mean_r2']:.5f} "
            f"q90={current['q90_r2']:.5f} "
            f"tail={current['mean_tail_r2']:.5f} "
            f"lr={lr:.1e} best={best['score']:.6f} "
            f"patience={patience}/{args.patience}",
            flush=True,
        )
        exhausted = (
            reductions >= args.maximum_lr_reductions
            or lr <= args.minimum_learning_rate * (1.0 + 1e-12)
        )
        if patience >= args.patience and exhausted:
            break

    model.load_state_dict(best_state, strict=True)
    final = base.validation_metrics(model, validation_cache, args)
    pd.DataFrame(history).to_csv(
        args.output / "training_history.csv", index=False
    )
    checkpoint = {
        **source_checkpoint,
        "model_state_dict": branch.cpu_state(model),
        "topology_zoomed_m2_refinement": {
            "source_checkpoint": str(args.checkpoint),
            "priority_conditions_file": str(args.priority_conditions_file),
            "condition_radius": args.condition_radius,
            "priority_condition_fraction": args.priority_condition_fraction,
            "regional_sample_fraction": args.regional_sample_fraction,
            "regional_box_halfwidth": args.regional_box_halfwidth,
            "topology_marginal_frozen": True,
            "best_step": best_step,
            "initial": initial,
            "final": final,
            "training_access": "f(x|c) only",
            "exact_target_sampler_used": False,
        },
    }
    torch.save(checkpoint, args.output / "zoomed_m2_refined_model.pt")
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }, stream, indent=2)
    print("\nZOOMED M2 SUMMARY")
    print(json.dumps({
        "best_step": best_step, "initial": initial, "final": final
    }, indent=2))
    print(
        f"\nModel: {(args.output / 'zoomed_m2_refined_model.pt').resolve()}"
    )


if __name__ == "__main__":
    main()
