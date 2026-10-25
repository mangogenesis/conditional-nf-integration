#!/usr/bin/env python3
"""Track dangerous 100D target basins along paths in condition space.

The belief search finds isolated pairs (x, c) where the current proposal has
poor coverage.  Fitting those pairs directly lets a conditional model memorize
condition slices.  This script instead continues each basin as c changes.

For each high-h anchor it builds several condition-space paths.  At every path
step, walkers start from the basin found at the previous condition and are
updated with Metropolis-adjusted Langevin transitions in logit coordinates,
targeting

    f(sigmoid(y) | c) |dx/dy|.

The output is discovery/training data, not unbiased integration data.  Entire
trajectory IDs must be held out when training a repair proposal.
"""

from __future__ import annotations

import argparse
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
from explore_energy_bombs_irregular_100d import load_proposal, logit  # noqa: E402


adaptive = irregular.adaptive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--belief-data", type=Path, required=True)
    parser.add_argument("--minimum-h", type=float, default=20.0)
    parser.add_argument("--anchors-per-condition", type=int, default=1)
    parser.add_argument("--paths-per-anchor", type=int, default=3)
    parser.add_argument("--condition-steps", type=int, default=8)
    parser.add_argument("--condition-step-size", type=float, default=0.05)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--initial-radius", type=float, default=0.04)
    parser.add_argument("--mala-steps", type=int, default=12)
    parser.add_argument("--mala-step-size", type=float, default=2e-3)
    parser.add_argument("--gradient-clip", type=float, default=50.0)
    parser.add_argument("--keep-per-step", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def log_target(y, cond):
    x = torch.sigmoid(y)
    log_f = irregular.irregular_mixture_100_log_integrand(x, cond).reshape(-1)
    log_jacobian = (
        torch.nn.functional.logsigmoid(y)
        + torch.nn.functional.logsigmoid(-y)
    ).sum(dim=1)
    return log_f + log_jacobian


def value_and_gradient(y, cond, gradient_clip):
    y = y.detach().requires_grad_(True)
    value = log_target(y, cond)
    gradient = torch.autograd.grad(value.sum(), y)[0]
    norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gradient = gradient * torch.clamp(gradient_clip / norm, max=1.0)
    return value.detach(), gradient.detach()


def mala_transition(y, cond, step_size, gradient_clip):
    """One vectorized MALA transition with the clipped drift used both ways."""
    current_value, current_gradient = value_and_gradient(
        y, cond, gradient_clip
    )
    variance = 2.0 * step_size
    current_mean = y + step_size * current_gradient
    proposed = current_mean + math.sqrt(variance) * torch.randn_like(y)
    proposed_value, proposed_gradient = value_and_gradient(
        proposed, cond, gradient_clip
    )
    reverse_mean = proposed + step_size * proposed_gradient
    forward = -(proposed - current_mean).square().sum(dim=1) / (2.0 * variance)
    reverse = -(y - reverse_mean).square().sum(dim=1) / (2.0 * variance)
    log_accept = proposed_value - current_value + reverse - forward
    accepted = torch.log(torch.rand_like(log_accept)) < torch.minimum(
        log_accept, torch.zeros_like(log_accept)
    )
    next_y = torch.where(accepted[:, None], proposed, y)
    return next_y.detach(), accepted


@torch.no_grad()
def select_anchors(data, args):
    eligible = (data["round"] > 0) & (data["h"] >= args.minimum_h)
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if not len(indices):
        raise RuntimeError("No discovered points satisfy --minimum-h")
    conditions = data["cond"][indices]
    _, inverse = torch.unique(conditions, dim=0, return_inverse=True)
    selected = []
    for group in range(int(inverse.max()) + 1):
        local = indices[inverse == group]
        order = torch.argsort(data["h"][local], descending=True)
        selected.append(local[order[: args.anchors_per_condition]])
    return torch.cat(selected)


def reflected_condition_path(origin, direction, steps, step_size):
    path = []
    value = origin.clone()
    velocity = direction / direction.norm().clamp_min(1e-12)
    for _ in range(steps):
        value = value + step_size * velocity
        below = value < 0.0
        above = value > 1.0
        velocity = torch.where(below | above, -velocity, velocity)
        value = torch.where(below, -value, value)
        value = torch.where(above, 2.0 - value, value).clamp(0.0, 1.0)
        path.append(value.clone())
    return path


@torch.no_grad()
def diagnostics(base, x, cond):
    log_f = irregular.irregular_mixture_100_log_integrand(x, cond).reshape(-1)
    log_q = base.log_prob(x, cond).reshape(-1)
    h = 2.0 * log_f - log_q
    return log_f, log_q, h


def main():
    args = parse_args()
    if args.keep_per_step > args.walkers:
        raise ValueError("--keep-per-step cannot exceed --walkers")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)
    base, checkpoint = load_proposal(args.checkpoint, device)
    data = torch.load(args.belief_data, map_location="cpu", weights_only=False)
    anchor_indices = select_anchors(data, args)

    saved = {key: [] for key in (
        "x", "cond", "log_f", "log_q", "h", "trajectory_id",
        "anchor_id", "path_id", "condition_step", "walker_rank",
    )}
    rows = []
    trajectory_id = 0

    for anchor_id, index in enumerate(anchor_indices.tolist()):
        anchor_x = data["x"][index].to(device).float()
        anchor_cond = data["cond"][index].to(device).float()
        anchor_h = float(data["h"][index])
        for path_id in range(args.paths_per_anchor):
            direction = torch.randn_like(anchor_cond)
            conditions = reflected_condition_path(
                anchor_cond, direction, args.condition_steps,
                args.condition_step_size,
            )
            y = logit(anchor_x)[None, :] + args.initial_radius * torch.randn(
                args.walkers, anchor_x.numel(), device=device
            )
            path_acceptance = []
            for condition_step, condition in enumerate(conditions, start=1):
                cond = condition[None, :].expand(args.walkers, -1)
                accepted_count = 0
                for _ in range(args.mala_steps):
                    y, accepted = mala_transition(
                        y, cond, args.mala_step_size, args.gradient_clip
                    )
                    accepted_count += int(accepted.sum())
                path_acceptance.append(
                    accepted_count / (args.mala_steps * args.walkers)
                )
                x = torch.sigmoid(y)
                log_f, log_q, h = diagnostics(base, x, cond)
                keep = torch.topk(h, args.keep_per_step).indices
                for key, value in (
                    ("x", x[keep]), ("cond", cond[keep]),
                    ("log_f", log_f[keep]), ("log_q", log_q[keep]),
                    ("h", h[keep]),
                ):
                    saved[key].append(value.detach().cpu())
                count = len(keep)
                saved["trajectory_id"].append(torch.full((count,), trajectory_id))
                saved["anchor_id"].append(torch.full((count,), anchor_id))
                saved["path_id"].append(torch.full((count,), path_id))
                saved["condition_step"].append(torch.full((count,), condition_step))
                saved["walker_rank"].append(torch.arange(count))

            rows.append({
                "trajectory_id": trajectory_id,
                "anchor_id": anchor_id,
                "path_id": path_id,
                "anchor_h": anchor_h,
                "condition_distance": float((conditions[-1] - anchor_cond).norm()),
                "mean_mala_acceptance": sum(path_acceptance) / len(path_acceptance),
                "minimum_mala_acceptance": min(path_acceptance),
                "maximum_retained_h": float(saved["h"][-1].max()),
                "final_mean_log_f": float(saved["log_f"][-1].mean()),
            })
            print(
                f"trajectory {trajectory_id + 1}: anchor={anchor_id} "
                f"path={path_id} distance={rows[-1]['condition_distance']:.3f} "
                f"accept={rows[-1]['mean_mala_acceptance']:.3f} "
                f"final max h={rows[-1]['maximum_retained_h']:.3f}"
            )
            trajectory_id += 1

    output = {key: torch.cat(value, dim=0) for key, value in saved.items()}
    output.update({
        "base_checkpoint": str(args.checkpoint.resolve()),
        "belief_data": str(args.belief_data.resolve()),
        "checkpoint_beta": float(checkpoint.get("beta", 1.0)),
        "warning": (
            "MALA trajectories are discovery/training data. Hold out complete "
            "trajectory_id groups and use fresh proposal samples for integration."
        ),
    })
    torch.save(output, args.output / "continued_basin_points.pt")
    trajectories = pd.DataFrame(rows)
    trajectories.to_csv(args.output / "trajectory_summary.csv", index=False)
    summary = {
        "anchors": len(anchor_indices),
        "source_conditions": len(torch.unique(data["cond"][anchor_indices], dim=0)),
        "trajectories": trajectory_id,
        "condition_steps": args.condition_steps,
        "retained_points": len(output["x"]),
        "minimum_h": float(output["h"].min()),
        "median_h": float(output["h"].median()),
        "maximum_h": float(output["h"].max()),
        "mean_mala_acceptance": float(trajectories["mean_mala_acceptance"].mean()),
        "mean_final_condition_distance": float(trajectories["condition_distance"].mean()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nCONDITION-SPACE BASIN CONTINUATION")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
