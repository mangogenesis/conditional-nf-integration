#!/usr/bin/env python3
"""Branch-aware conditional continuation for the 32D topology benchmark.

For every condition group, target branch masses are estimated from normalized
importance weights.  The first marginal is then calibrated through its CDF:

    q(B0|c) = F_theta(0.4|c),
    q(B1|c) = 1 - F_theta(0.6|c).

Training uses only evaluations of f(x|c).  Analytic benchmark branch weights
and exact target samples are used only by the post-training audit inherited
from ``train_local_conditional_continuation_32d``.
"""

from __future__ import annotations

import argparse
import copy
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

import run_topology_ladder_32d as topology  # noqa: E402
import train_local_conditional_continuation_32d as local  # noqa: E402


DIM = topology.DIMENSION
COND_DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--anchor", type=float, nargs=COND_DIM, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument(
        "--radius-ladder", type=float, nargs="+",
        default=(0.025, 0.05, 0.10, 0.20),
    )
    parser.add_argument("--train-conditions", type=int, default=128)
    parser.add_argument("--validation-conditions", type=int, default=64)
    parser.add_argument("--points-per-condition", type=int, default=8192)
    parser.add_argument("--anchor-fraction", type=float, default=0.25)
    parser.add_argument(
        "--full-cube-radius", type=float, default=1.0,
        help=(
            "At or above this radius, draw non-anchor conditions uniformly "
            "from [0,1]^3 instead of clipping a Euclidean ball."
        ),
    )
    parser.add_argument("--uniform-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-validation-q05-raw-ess", type=float, default=20.0)

    parser.add_argument("--branch-steps", type=int, default=40_000)
    parser.add_argument("--branch-condition-batch", type=int, default=32)
    parser.add_argument("--branch-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--branch-shape-weight", type=float, default=0.1,
        help=(
            "Penalty preserving the normalized density inside each topology "
            "branch while its total mass is recalibrated."
        ),
    )
    parser.add_argument("--branch-shape-grid-points", type=int, default=16)
    parser.add_argument("--joint-steps", type=int, default=20_000)
    parser.add_argument("--joint-condition-batch", type=int, default=4)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-5)
    parser.add_argument("--branch-weight", type=float, default=20.0)
    parser.add_argument(
        "--freeze-topology-marginal-during-joint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the calibrated x0 marginal fixed while the copula and "
            "remaining marginals adapt."
        ),
    )
    parser.add_argument("--worst-fraction", type=float, default=0.20)
    parser.add_argument("--worst-weight", type=float, default=0.50)
    parser.add_argument("--anchor-penalty", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument(
        "--lr-plateau-patience", type=int, default=5,
        help="Validation checks without improvement before reducing the LR.",
    )
    parser.add_argument("--lr-reduction-factor", type=float, default=0.5)
    parser.add_argument("--maximum-lr-reductions", type=int, default=4)
    parser.add_argument("--minimum-branch-learning-rate", type=float, default=1e-7)
    parser.add_argument("--minimum-joint-learning-rate", type=float, default=1e-7)

    parser.add_argument("--evaluation-conditions", type=int, default=16)
    parser.add_argument("--evaluation-samples", type=int, default=200_000)
    parser.add_argument("--evaluation-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(radius <= 0.0 for radius in args.radius_ladder):
        parser.error("all radii must be positive")
    if sorted(args.radius_ladder) != list(args.radius_ladder):
        parser.error("--radius-ladder must be increasing")
    if not 0.0 < args.anchor_fraction < 1.0:
        parser.error("--anchor-fraction must lie in (0,1)")
    if args.full_cube_radius <= 0.0:
        parser.error("--full-cube-radius must be positive")
    if not 0.0 < args.uniform_fraction < 1.0:
        parser.error("--uniform-fraction must lie in (0,1)")
    if args.branch_weight < 0.0:
        parser.error("--branch-weight must be nonnegative")
    if args.branch_shape_weight < 0.0:
        parser.error("--branch-shape-weight must be nonnegative")
    if args.branch_shape_grid_points < 2:
        parser.error("--branch-shape-grid-points must be at least two")
    if args.lr_plateau_patience < 1:
        parser.error("--lr-plateau-patience must be positive")
    if not 0.0 < args.lr_reduction_factor < 1.0:
        parser.error("--lr-reduction-factor must lie in (0,1)")
    if args.maximum_lr_reductions < 0:
        parser.error("--maximum-lr-reductions must be nonnegative")
    return args


def freeze_copy(model, device):
    result = copy.deepcopy(model).to(device).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def make_stage_condition_sampler(args, radius):
    if radius < args.full_cube_radius:
        return local.make_condition_sampler(
            args.anchor, radius, args.anchor_fraction
        )
    anchor = torch.tensor(args.anchor, dtype=torch.float32)

    def sample_conditions(n, device):
        count_anchor = min(n, int(round(args.anchor_fraction * n)))
        count_cube = n - count_anchor
        pieces = []
        if count_anchor:
            pieces.append(anchor.to(device).expand(
                count_anchor, COND_DIM
            ).clone())
        if count_cube:
            pieces.append(torch.rand(count_cube, COND_DIM, device=device))
        result = torch.cat(pieces, dim=0)
        return result[torch.randperm(n, device=device)]

    return sample_conditions


def stage_evaluation_conditions(args, radius, count, seed):
    if radius < args.full_cube_radius:
        return local.evaluation_conditions(
            args.anchor, radius, count, seed
        )
    generator = np.random.default_rng(seed)
    return generator.random((count, COND_DIM))


def cpu_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def target_branch_masses(cache):
    """Estimate target branch masses with grouped normalized IS weights."""
    x0 = cache["x"][:, :, 0]
    alpha = cache["alpha"].double()
    p0 = (alpha * (x0 <= topology.BRANCH_WIDTH)).sum(dim=1)
    p1 = (alpha * (x0 >= 1.0 - topology.BRANCH_WIDTH)).sum(dim=1)
    total = (p0 + p1).clamp_min(torch.finfo(alpha.dtype).tiny)
    return torch.stack((p0 / total, p1 / total), dim=1).float().detach()


def cache_q05_raw_ess(cache):
    raw_ess = 1.0 / cache["alpha"].double().square().sum(dim=1)
    return float(torch.quantile(raw_ess, 0.05))


def branch_probabilities_from_model(model, cond):
    """Differentiable structural masses from the first marginal's CDF."""
    marginal = model.marginals[0]
    lower = torch.full(
        (len(cond), 1), topology.BRANCH_WIDTH,
        device=cond.device, dtype=cond.dtype,
    )
    upper = torch.full(
        (len(cond), 1), 1.0 - topology.BRANCH_WIDTH,
        device=cond.device, dtype=cond.dtype,
    )
    cdf_lower, _ = marginal.inverse(lower, cond)
    cdf_upper, _ = marginal.inverse(upper, cond)
    q0 = cdf_lower.reshape(-1)
    q1 = 1.0 - cdf_upper.reshape(-1)
    gap = (cdf_upper - cdf_lower).reshape(-1)
    return torch.stack((q0, q1), dim=1), gap


def branch_kl(target, proposal):
    epsilon = torch.finfo(proposal.dtype).eps
    target = target.clamp_min(epsilon)
    proposal = proposal.clamp_min(epsilon)
    return (target * (target.log() - proposal.log())).sum(dim=1)


def marginal_branch_probabilities(marginal, cond):
    lower = torch.full(
        (len(cond), 1), topology.BRANCH_WIDTH,
        device=cond.device, dtype=cond.dtype,
    )
    upper = torch.full(
        (len(cond), 1), 1.0 - topology.BRANCH_WIDTH,
        device=cond.device, dtype=cond.dtype,
    )
    cdf_lower, _ = marginal.inverse(lower, cond)
    cdf_upper, _ = marginal.inverse(upper, cond)
    return torch.stack(
        (cdf_lower.reshape(-1), 1.0 - cdf_upper.reshape(-1)), dim=1
    )


def branch_shape_distillation(model, reference_marginal, cond, grid_points):
    """Preserve each branch's normalized shape while allowing mass changes."""
    epsilon = 10.0 * torch.finfo(cond.dtype).eps
    left = torch.linspace(
        epsilon, topology.BRANCH_WIDTH - epsilon, grid_points,
        device=cond.device, dtype=cond.dtype,
    )
    right = torch.linspace(
        1.0 - topology.BRANCH_WIDTH + epsilon, 1.0 - epsilon, grid_points,
        device=cond.device, dtype=cond.dtype,
    )
    grid = torch.stack((left, right), dim=0)
    batch = len(cond)
    x = grid[None, :, :].expand(batch, -1, -1)
    expanded_cond = cond[:, None, None, :].expand(
        -1, 2, grid_points, -1
    )
    flat_x = x.reshape(-1, 1)
    flat_cond = expanded_cond.reshape(-1, cond.shape[-1])

    current_logq = model.marginals[0].log_prob(
        flat_x, flat_cond
    ).reshape(batch, 2, grid_points)
    current_mass = marginal_branch_probabilities(
        model.marginals[0], cond
    ).clamp_min(epsilon)
    current_normalized = current_logq - current_mass.log()[:, :, None]

    with torch.no_grad():
        reference_logq = reference_marginal.log_prob(
            flat_x, flat_cond
        ).reshape(batch, 2, grid_points)
        reference_mass = marginal_branch_probabilities(
            reference_marginal, cond
        ).clamp_min(epsilon)
        reference_normalized = (
            reference_logq - reference_mass.log()[:, :, None]
        )
    return (current_normalized - reference_normalized).square().mean()


def robust_mean(values, args):
    count = max(1, int(math.ceil(args.worst_fraction * len(values))))
    return values.mean() + args.worst_weight * torch.topk(values, count).values.mean()


@torch.no_grad()
def branch_validation(model, cond, target, args):
    model.eval()
    proposal, gap = branch_probabilities_from_model(model, cond)
    losses = branch_kl(target, proposal)
    error = (proposal[:, 1] - target[:, 1]).abs()
    count = max(1, int(math.ceil(args.worst_fraction * len(losses))))
    return {
        "score": float(
            losses.mean()
            + args.worst_weight * torch.topk(losses, count).values.mean()
        ),
        "mean_kl": float(losses.mean()),
        "worst_kl": float(torch.topk(losses, count).values.mean()),
        "mean_absolute_mass_error": float(error.mean()),
        "maximum_absolute_mass_error": float(error.max()),
        "mean_gap_mass": float(gap.mean()),
        "maximum_gap_mass": float(gap.max()),
    }


def parameter_anchor(parameters, initial):
    total = torch.zeros((), device=parameters[0].device)
    count = 0
    for parameter, reference in zip(parameters, initial):
        total = total + (parameter - reference).square().sum()
        count += parameter.numel()
    return total / max(count, 1)


def reduce_learning_rate(optimizer, factor, minimum):
    """Reduce every parameter-group LR and report whether it changed."""
    previous = min(group["lr"] for group in optimizer.param_groups)
    updated = max(minimum, previous * factor)
    if updated >= previous:
        return previous, False
    for group in optimizer.param_groups:
        group["lr"] = max(minimum, group["lr"] * factor)
    return min(group["lr"] for group in optimizer.param_groups), True


def train_branch_stage(model, train_cache, validation_cache, args, label):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.marginals[0].parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    reference_marginal = copy.deepcopy(model.marginals[0]).eval()
    for parameter in reference_marginal.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.branch_learning_rate,
        weight_decay=args.weight_decay,
    )

    train_cond = train_cache["cond"][:, 0, :]
    validation_cond = validation_cache["cond"][:, 0, :]
    train_target = target_branch_masses(train_cache)
    validation_target = target_branch_masses(validation_cache)
    initial = branch_validation(
        model, validation_cond, validation_target, args
    )
    best = initial
    best_state = cpu_state(model)
    best_step = 0
    patience = 0
    lr_reductions = 0
    history = []
    print(
        f"\nBRANCH-ALLOCATION STAGE {label}: "
        f"MAE={initial['mean_absolute_mass_error']:.6f} "
        f"max={initial['maximum_absolute_mass_error']:.6f} "
        f"gap={initial['mean_gap_mass']:.3e}",
        flush=True,
    )

    for start in range(0, args.branch_steps, args.check_interval):
        block = min(args.check_interval, args.branch_steps - start)
        train_losses = []
        train_shape_penalties = []
        model.marginals[0].train()
        for _ in range(block):
            indices = torch.randint(
                len(train_cond),
                (args.branch_condition_batch,),
                device=train_cond.device,
            )
            proposal, _ = branch_probabilities_from_model(
                model, train_cond[indices]
            )
            losses = branch_kl(train_target[indices], proposal)
            loss = robust_mean(losses, args)
            shape_penalty = branch_shape_distillation(
                model,
                reference_marginal,
                train_cond[indices],
                args.branch_shape_grid_points,
            )
            loss = loss + args.branch_shape_weight * shape_penalty
            loss = loss + args.anchor_penalty * parameter_anchor(
                parameters, initial_parameters
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            train_shape_penalties.append(float(shape_penalty.detach()))

        step = start + block
        current = branch_validation(
            model, validation_cond, validation_target, args
        )
        improved = current["score"] < (
            best["score"] - args.minimum_improvement
        )
        if improved:
            best = current
            best_state = cpu_state(model)
            best_step = step
            patience = 0
        else:
            patience += 1
        learning_rate = min(group["lr"] for group in optimizer.param_groups)
        reduced = False
        if (
            not improved
            and patience >= args.lr_plateau_patience
            and lr_reductions < args.maximum_lr_reductions
        ):
            learning_rate, reduced = reduce_learning_rate(
                optimizer,
                args.lr_reduction_factor,
                args.minimum_branch_learning_rate,
            )
            if reduced:
                lr_reductions += 1
                patience = 0
                model.load_state_dict(best_state, strict=True)
                optimizer.state.clear()
        history.append({
            "step": step,
            "train_loss": float(np.mean(train_losses)),
            "train_shape_penalty": float(np.mean(train_shape_penalties)),
            **{f"validation_{key}": value for key, value in current.items()},
            "best_score": best["score"],
            "learning_rate": learning_rate,
            "lr_reductions": lr_reductions,
        })
        print(
            f"branch step {step:5d}/{args.branch_steps}: "
            f"train={np.mean(train_losses):.6g} "
            f"shape={np.mean(train_shape_penalties):.3e} "
            f"MAE={current['mean_absolute_mass_error']:.5f} "
            f"max={current['maximum_absolute_mass_error']:.5f} "
            f"gap={current['mean_gap_mass']:.2e} "
            f"lr={learning_rate:.2e} "
            f"patience={patience}/{args.patience} "
            f"reductions={lr_reductions}/{args.maximum_lr_reductions}",
            flush=True,
        )
        if reduced:
            print(
                f"  BRANCH LR REDUCTION: restored best step {best_step}; "
                f"new lr={learning_rate:.3e}",
                flush=True,
            )
        schedule_exhausted = (
            lr_reductions >= args.maximum_lr_reductions
            or learning_rate <= args.minimum_branch_learning_rate * (1.0 + 1e-12)
        )
        if patience >= args.patience and schedule_exhausted:
            break

    model.load_state_dict(best_state, strict=True)
    return {
        "initial": initial,
        "best": best,
        "best_step": best_step,
        "history": history,
    }


def joint_validation(model, cache, target_branch, args):
    nll = local.validation_score(
        model, cache, args.worst_fraction, args.worst_weight
    )
    branch = branch_validation(
        model, cache["cond"][:, 0, :], target_branch, args
    )
    return {
        "score": nll["score"] + args.branch_weight * branch["score"],
        "nll_score": nll["score"],
        **{f"branch_{key}": value for key, value in branch.items()},
    }


def train_joint_stage(model, train_cache, validation_cache, args, label):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if args.freeze_topology_marginal_during_joint:
        for parameter in model.marginals[0].parameters():
            parameter.requires_grad_(False)
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.joint_learning_rate,
        weight_decay=args.weight_decay,
    )
    train_target = target_branch_masses(train_cache)
    validation_target = target_branch_masses(validation_cache)
    initial = joint_validation(model, validation_cache, validation_target, args)
    best = initial
    best_state = cpu_state(model)
    best_step = 0
    patience = 0
    lr_reductions = 0
    history = []
    print(
        f"\nJOINT KL + BRANCH STAGE {label}: "
        f"score={initial['score']:.6f} "
        f"branch-MAE={initial['branch_mean_absolute_mass_error']:.6f} "
        f"x0-frozen={args.freeze_topology_marginal_during_joint}",
        flush=True,
    )

    for start in range(0, args.joint_steps, args.check_interval):
        block = min(args.check_interval, args.joint_steps - start)
        train_losses = []
        model.train()
        for _ in range(block):
            indices = torch.randint(
                len(train_cache["x"]),
                (args.joint_condition_batch,),
                device=train_cache["x"].device,
            )
            nll = local.grouped_losses(model, train_cache, indices)
            proposal, _ = branch_probabilities_from_model(
                model, train_cache["cond"][indices, 0, :]
            )
            branch = branch_kl(train_target[indices], proposal)
            loss = robust_mean(nll, args)
            loss = loss + args.branch_weight * robust_mean(branch, args)
            loss = loss + args.anchor_penalty * parameter_anchor(
                parameters, initial_parameters
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        step = start + block
        current = joint_validation(
            model, validation_cache, validation_target, args
        )
        improved = current["score"] < (
            best["score"] - args.minimum_improvement
        )
        if improved:
            best = current
            best_state = cpu_state(model)
            best_step = step
            patience = 0
        else:
            patience += 1
        learning_rate = min(group["lr"] for group in optimizer.param_groups)
        reduced = False
        if (
            not improved
            and patience >= args.lr_plateau_patience
            and lr_reductions < args.maximum_lr_reductions
        ):
            learning_rate, reduced = reduce_learning_rate(
                optimizer,
                args.lr_reduction_factor,
                args.minimum_joint_learning_rate,
            )
            if reduced:
                lr_reductions += 1
                patience = 0
                model.load_state_dict(best_state, strict=True)
                optimizer.state.clear()
        history.append({
            "step": step,
            "train_loss": float(np.mean(train_losses)),
            **{f"validation_{key}": value for key, value in current.items()},
            "best_score": best["score"],
            "learning_rate": learning_rate,
            "lr_reductions": lr_reductions,
        })
        print(
            f"joint step {step:5d}/{args.joint_steps}: "
            f"train={np.mean(train_losses):.6f} "
            f"score={current['score']:.6f} "
            f"branch-MAE={current['branch_mean_absolute_mass_error']:.5f} "
            f"lr={learning_rate:.2e} "
            f"patience={patience}/{args.patience} "
            f"reductions={lr_reductions}/{args.maximum_lr_reductions}",
            flush=True,
        )
        if reduced:
            print(
                f"  JOINT LR REDUCTION: restored best step {best_step}; "
                f"new lr={learning_rate:.3e}",
                flush=True,
            )
        schedule_exhausted = (
            lr_reductions >= args.maximum_lr_reductions
            or learning_rate <= args.minimum_joint_learning_rate * (1.0 + 1e-12)
        )
        if patience >= args.patience and schedule_exhausted:
            break

    model.load_state_dict(best_state, strict=True)
    return {
        "initial": initial,
        "best": best,
        "best_step": best_step,
        "history": history,
    }


def main() -> None:
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
        raise ValueError(
            "branch-aware continuation requires topology_two_conditional_32"
        )
    model.to(device)
    stage_rows = []
    evaluation_rows = []

    for radius_index, radius in enumerate(args.radius_ladder):
        adaptive.sample_conditions = make_stage_condition_sampler(args, radius)
        proposal = freeze_copy(model, device)
        train_cache = adaptive.build_weighted_cache(
            benchmark=benchmark,
            proposal_model=proposal,
            n_conditions=args.train_conditions,
            points_per_condition=args.points_per_condition,
            device=device,
            seed=args.seed + 1000 * radius_index,
            label=f"branch-aware train radius={radius:.6g}",
            target_temperature=1.0,
            uniform_fraction=args.uniform_fraction,
        )
        validation_cache = adaptive.build_weighted_cache(
            benchmark=benchmark,
            proposal_model=proposal,
            n_conditions=args.validation_conditions,
            points_per_condition=args.points_per_condition,
            device=device,
            seed=args.seed + 1000 * radius_index + 1,
            label=f"branch-aware validation radius={radius:.6g}",
            target_temperature=1.0,
            uniform_fraction=args.uniform_fraction,
        )
        validation_q05_raw_ess = cache_q05_raw_ess(validation_cache)
        if (
            validation_q05_raw_ess
            < args.minimum_validation_q05_raw_ess
        ):
            print(
                "STOPPING: validation q05 raw ESS is below the branch-mass "
                "estimation floor.",
                flush=True,
            )
            break

        branch_result = train_branch_stage(
            model, train_cache, validation_cache, args,
            label=f"radius={radius:.6g}",
        )
        joint_result = train_joint_stage(
            model, train_cache, validation_cache, args,
            label=f"radius={radius:.6g}",
        )

        radius_dir = args.output / f"radius_{radius:.6g}"
        radius_dir.mkdir(exist_ok=True)
        pd.DataFrame(branch_result["history"]).to_csv(
            radius_dir / "branch_training_history.csv", index=False
        )
        pd.DataFrame(joint_result["history"]).to_csv(
            radius_dir / "joint_training_history.csv", index=False
        )
        checkpoint = {
            **source_checkpoint,
            "model_state_dict": cpu_state(model),
            "topology_branch_conditional_continuation": {
                "anchor": list(args.anchor),
                "radius": radius,
                "target_branch_mass_source": (
                    "grouped normalized importance weights"
                ),
                "analytic_branch_weights_used_for_training": False,
                "exact_target_sampler_used_for_training": False,
                "branch_result": branch_result,
                "joint_result": joint_result,
            },
        }
        torch.save(checkpoint, radius_dir / "conditional_model.pt")

        conditions = np.vstack([
            np.asarray(args.anchor, dtype=float),
            stage_evaluation_conditions(
                args, radius, args.evaluation_conditions,
                args.seed + 10_000 + radius_index,
            ),
        ])
        radius_evaluations = []
        for condition_index, condition in enumerate(conditions):
            metrics = local.evaluate_condition(
                adaptive, model, benchmark, condition, args, device,
                args.seed + 100_000 * radius_index + condition_index,
            )
            row = {
                "radius": radius,
                "condition_index": condition_index,
                "is_anchor": condition_index == 0,
                "c0": condition[0], "c1": condition[1], "c2": condition[2],
                **metrics,
            }
            radius_evaluations.append(row)
            evaluation_rows.append(row)
            print(
                f"radius={radius:.4g} condition={condition_index:02d} "
                f"I={metrics['estimate']:.6f} "
                f"ESS={metrics['normalized_ess']:.4f} "
                f"target-p1={metrics['target_branch_1_mass']:.3f} "
                f"NF-p1={metrics['proposal_branch_1_mass']:.3f} "
                f"gap={metrics['proposal_gap_mass']:.2e}",
                flush=True,
            )
        frame = pd.DataFrame(radius_evaluations)
        stage_rows.append({
            "radius": radius,
            "train_median_raw_ess": train_cache["median_raw_ess"],
            "validation_median_raw_ess": validation_cache["median_raw_ess"],
            "validation_q05_raw_ess": validation_q05_raw_ess,
            "branch_best_step": branch_result["best_step"],
            "joint_best_step": joint_result["best_step"],
            "evaluation_mean_branch_mass_error": float(
                frame["absolute_branch_1_mass_error"].mean()
            ),
            "evaluation_maximum_branch_mass_error": float(
                frame["absolute_branch_1_mass_error"].max()
            ),
            "evaluation_minimum_ess": float(frame["normalized_ess"].min()),
            "evaluation_maximum_gap_mass": float(
                frame["proposal_gap_mass"].max()
            ),
        })
        pd.DataFrame(stage_rows).to_csv(
            args.output / "stage_summary.csv", index=False
        )
        pd.DataFrame(evaluation_rows).to_csv(
            args.output / "evaluation.csv", index=False
        )

    torch.save({
        **source_checkpoint,
        "model_state_dict": cpu_state(model),
        "topology_branch_conditional_continuation": {
            "anchor": list(args.anchor),
            "radius_ladder": list(args.radius_ladder),
            "target_branch_mass_source": "grouped normalized importance weights",
            "analytic_branch_weights_used_for_training": False,
            "exact_target_sampler_used_for_training": False,
        },
    }, args.output / "latest_conditional_model.pt")
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }, stream, indent=2)
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
