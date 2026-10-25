#!/usr/bin/env python3
"""Tail-sensitive M2 refinement at a fixed conditional-topology radius.

The input model is treated as a frozen proposal used to build independent
defensive caches.  For every condition, training minimizes the full M2 score
and a smooth objective restricted to the largest within-condition M2
contributions.  This is intentionally different from selecting the worst
conditions.  The first marginal can be frozen to preserve the already learned
two-branch topology exactly.
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
import run_topology_ladder_32d as topology  # noqa: E402
import train_local_conditional_continuation_32d as local  # noqa: E402
import train_topology_branch_conditional_32d as branch  # noqa: E402


DIM = topology.DIMENSION
COND_DIM = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--condition-radius", type=float, default=0.15)
    parser.add_argument("--train-conditions", type=int, default=128)
    parser.add_argument("--validation-conditions", type=int, default=64)
    parser.add_argument("--points-per-condition", type=int, default=32768)
    parser.add_argument("--cache-condition-block", type=int, default=4)
    parser.add_argument("--anchor-fraction", type=float, default=0.10)
    parser.add_argument("--uniform-fraction", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--condition-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument("--tail-weight", type=float, default=0.25)
    parser.add_argument("--anchor-penalty", type=float, default=1e-3)
    parser.add_argument("--branch-weight", type=float, default=20.0)
    parser.add_argument("--worst-fraction", type=float, default=0.20)
    parser.add_argument("--worst-weight", type=float, default=0.50)
    parser.add_argument(
        "--freeze-topology-marginal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze x0 marginal so learned branch masses and gap stay fixed.",
    )
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-plateau-patience", type=int, default=4)
    parser.add_argument("--lr-reduction-factor", type=float, default=0.3)
    parser.add_argument("--maximum-lr-reductions", type=int, default=2)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.condition_radius < 0:
        parser.error("--condition-radius must be nonnegative")
    if not 0 < args.anchor_fraction <= 1:
        parser.error("--anchor-fraction must lie in (0,1]")
    if not 0 < args.uniform_fraction < 1:
        parser.error("--uniform-fraction must lie in (0,1)")
    if not 0 < args.tail_fraction <= 1:
        parser.error("--tail-fraction must lie in (0,1]")
    if args.tail_weight < 0 or args.branch_weight < 0:
        parser.error("loss weights must be nonnegative")
    if not 0 < args.worst_fraction <= 1 or args.worst_weight < 0:
        parser.error("invalid branch-validation robust-loss settings")
    return args


def train_group_losses(model, cache, indices, args):
    """Full-M2 plus top-tail score losses, separately for each condition."""
    x = cache["x"][indices]
    cond = cache["cond"][indices]
    logf = cache["logf"][indices]
    logr = cache["logr"][indices]
    groups, points, _ = x.shape
    logq = model.log_prob(
        x.reshape(groups * points, DIM),
        cond.reshape(groups * points, COND_DIM),
    ).reshape(groups, points)

    # x ~ r and M2(q)=int f^2/q, hence the cache contribution is
    # a=f^2/(r q).  q is detached only while constructing score weights;
    # -E_a[log q] has the desired M2 gradient.
    log_contribution = (
        2.0 * logf.double() - logr.double() - logq.detach().double()
    )
    full_weight = torch.softmax(log_contribution, dim=1).to(logq.dtype)
    full_loss = -(full_weight * logq).sum(dim=1)

    tail_count = max(1, int(math.ceil(args.tail_fraction * points)))
    tail_value, tail_index = torch.topk(
        log_contribution, tail_count, dim=1
    )
    tail_logq = torch.gather(logq, 1, tail_index)
    tail_weight = torch.softmax(tail_value, dim=1).to(logq.dtype)
    tail_loss = -(tail_weight * tail_logq).sum(dim=1)
    return full_loss, tail_loss


def target_branch_masses(cache):
    """Estimate branch masses from this script's logf/logr cache format."""
    x0 = cache["x"][:, :, 0]
    alpha = torch.softmax(
        (cache["logf"] - cache["logr"]).double(), dim=1
    )
    p0 = (alpha * (x0 <= topology.BRANCH_WIDTH)).sum(dim=1)
    p1 = (
        alpha * (x0 >= 1.0 - topology.BRANCH_WIDTH)
    ).sum(dim=1)
    total = (p0 + p1).clamp_min(torch.finfo(alpha.dtype).tiny)
    return torch.stack((p0 / total, p1 / total), dim=1).float().detach()


@torch.no_grad()
def validation_metrics(model, cache, args):
    model.eval()
    r2_values = []
    tail_r2_values = []
    points = cache["x"].shape[1]
    log_points = math.log(points)
    tail_count = max(1, int(math.ceil(args.tail_fraction * points)))
    for start in range(0, len(cache["x"]), 2):
        x = cache["x"][start:start + 2]
        cond = cache["cond"][start:start + 2]
        logf = cache["logf"][start:start + 2].double()
        logr = cache["logr"][start:start + 2].double()
        groups = len(x)
        logq = model.log_prob(
            x.reshape(groups * points, DIM),
            cond.reshape(groups * points, COND_DIM),
        ).reshape(groups, points).double()
        log_i = torch.logsumexp(logf - logr, dim=1) - log_points
        log_a = 2.0 * logf - logr - logq
        r2_values.append(
            (torch.logsumexp(log_a, dim=1) - log_points - 2.0 * log_i).cpu()
        )
        top = torch.topk(log_a, tail_count, dim=1).values
        tail_r2_values.append(
            (torch.logsumexp(top, dim=1) - math.log(tail_count) - 2.0 * log_i).cpu()
        )

    r2 = torch.cat(r2_values)
    tail_r2 = torch.cat(tail_r2_values)
    cond = cache["cond"][:, 0, :]
    target = target_branch_masses(cache)
    branch_metrics = branch.branch_validation(model, cond, target, args)
    score = (
        r2.mean()
        + args.tail_weight * tail_r2.mean()
        + args.branch_weight * branch_metrics["mean_kl"]
    )
    return {
        "score": float(score),
        "mean_r2": float(r2.mean()),
        "median_r2": float(r2.median()),
        "q90_r2": float(torch.quantile(r2, 0.90)),
        "mean_tail_r2": float(tail_r2.mean()),
        "q90_tail_r2": float(torch.quantile(tail_r2, 0.90)),
        "median_ess_proxy": float(torch.exp(-r2.median())),
        "q10_ess_proxy": float(torch.exp(-torch.quantile(r2, 0.90))),
        **{f"branch_{key}": value for key, value in branch_metrics.items()},
    }


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
    model.to(device)
    proposal = renyi.freeze_copy(model, device)
    train_cache = renyi.build_cache(
        adaptive, benchmark, proposal, args.train_conditions,
        args.points_per_condition, args, device, args.seed,
        f"within-condition M2 train radius={args.condition_radius}",
    )
    validation_cache = renyi.build_cache(
        adaptive, benchmark, proposal, args.validation_conditions,
        args.points_per_condition, args, device, args.seed + 1,
        f"within-condition M2 validation radius={args.condition_radius}",
    )

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if args.freeze_topology_marginal:
        for parameter in model.marginals[0].parameters():
            parameter.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    reference = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_target = target_branch_masses(train_cache)

    initial = validation_metrics(model, validation_cache, args)
    best = initial
    best_state = branch.cpu_state(model)
    best_step = 0
    patience = 0
    plateau = 0
    reductions = 0
    history = []
    print(
        "\nWITHIN-CONDITION M2 REFINEMENT: "
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
                len(train_cache["x"]), (args.condition_batch,), device=device
            )
            full, tail = train_group_losses(model, train_cache, indices, args)
            loss = full.mean() + args.tail_weight * tail.mean()
            if not args.freeze_topology_marginal and args.branch_weight > 0:
                proposal_mass, _ = branch.branch_probabilities_from_model(
                    model, train_cache["cond"][indices, 0, :]
                )
                branch_loss = branch.branch_kl(
                    train_target[indices], proposal_mass
                ).mean()
                loss = loss + args.branch_weight * branch_loss
            loss = loss + args.anchor_penalty * branch.parameter_anchor(
                parameters, reference
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))

        step = start + block
        current = validation_metrics(model, validation_cache, args)
        threshold = args.minimum_improvement * max(abs(best["score"]), 1.0)
        improved = current["score"] < best["score"] - threshold
        if improved:
            best = current
            best_state = branch.cpu_state(model)
            best_step = step
            patience = 0
            plateau = 0
        else:
            patience += 1
            plateau += 1

        reduced = False
        if plateau >= args.lr_plateau_patience and reductions < args.maximum_lr_reductions:
            lr, reduced = branch.reduce_learning_rate(
                optimizer, args.lr_reduction_factor, args.minimum_learning_rate
            )
            if reduced:
                reductions += 1
                plateau = 0
                patience = 0
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
            f"R2={current['mean_r2']:.5f} q90={current['q90_r2']:.5f} "
            f"tail={current['mean_tail_r2']:.5f} "
            f"branch-MAE={current['branch_mean_absolute_mass_error']:.5f} "
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
    final = validation_metrics(model, validation_cache, args)
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    checkpoint = {
        **source_checkpoint,
        "model_state_dict": branch.cpu_state(model),
        "topology_within_condition_m2_refinement": {
            "source_checkpoint": str(args.checkpoint),
            "condition_radius": args.condition_radius,
            "tail_fraction": args.tail_fraction,
            "tail_weight": args.tail_weight,
            "topology_marginal_frozen": args.freeze_topology_marginal,
            "best_step": best_step,
            "initial": initial,
            "final": final,
            "training_access": "f(x|c) only",
            "exact_target_sampler_used": False,
        },
    }
    torch.save(checkpoint, args.output / "conditional_m2_refined_model.pt")
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }, stream, indent=2)
    print("\nWITHIN-CONDITION M2 SUMMARY")
    print(json.dumps({"best_step": best_step, "initial": initial, "final": final}, indent=2))
    print(f"\nModel: {(args.output / 'conditional_m2_refined_model.pt').resolve()}")


if __name__ == "__main__":
    main()
