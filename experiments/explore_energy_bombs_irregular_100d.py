#!/usr/bin/env python3
"""Explore a 100D integrand with local stochastic-process "bombs".

A bomb consists of one center and a cloud of nearby walkers.  Walkers evolve
in logit coordinates with annealed Langevin dynamics targeting

    log pi_y(y|c) = log f(sigmoid(y)|c) + log |dx/dy|.

The Jacobian term makes the ideal continuous-time stationary law in y
correspond to f(x|c) dx.  This script is a *discovery* tool: unadjusted finite
steps do not provide a tractable endpoint density, so endpoints are saved as
repair-training data and are never used directly for an unbiased integral.
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
from nf.conditional_mixture import ConditionalMixtureProposal  # noqa: E402


adaptive = irregular.adaptive
BENCHMARK = irregular.BENCHMARK


def logit(x, epsilon=1e-6):
    x = x.clamp(epsilon, 1.0 - epsilon)
    return torch.log(x) - torch.log1p(-x)


def log_target_in_logit(y, cond):
    """Log target with the exact logistic change-of-variables Jacobian."""
    x = torch.sigmoid(y)
    log_f = irregular.irregular_mixture_100_log_integrand(x, cond)
    log_jacobian = (
        torch.nn.functional.logsigmoid(y)
        + torch.nn.functional.logsigmoid(-y)
    ).sum(dim=1)
    return log_f + log_jacobian, log_f


def effective_cloud_dimension(points):
    """Participation-ratio dimension of one local walker cloud."""
    centered = points.double() - points.double().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    eigenvalues = singular_values.square() / max(points.shape[0] - 1, 1)
    denominator = eigenvalues.square().sum().clamp_min(1e-300)
    return float((eigenvalues.sum().square() / denominator).item())


def load_proposal(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    components = [
        adaptive.make_real_model(checkpoint["graph"], None, device)
        for _ in range(int(checkpoint["components"]))
    ]
    maximum = max(
        int(checkpoint["components"]),
        int(checkpoint.get("args", {}).get("max_components", 8)),
    )
    model = ConditionalMixtureProposal(
        components, cond_dim=adaptive.COND_DIM, max_components=maximum
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def initialize_bombs(model, conditions, bombs_per_condition, walkers, radius):
    repeated_condition = conditions.repeat_interleave(bombs_per_condition, dim=0)
    centers = model.sample(repeated_condition.shape[0], repeated_condition)
    center_y = logit(centers)
    y = center_y[:, None, :] + radius * torch.randn(
        center_y.shape[0], walkers, center_y.shape[1],
        device=center_y.device, dtype=center_y.dtype,
    )
    cond = repeated_condition[:, None, :].expand(-1, walkers, -1)
    return y.reshape(-1, y.shape[-1]), cond.reshape(-1, cond.shape[-1])


def run_langevin(y, cond, steps, step_size, start_temperature,
                  end_temperature, gradient_clip, report_interval):
    initial_y = y.detach().clone()
    history = []
    for step in range(steps):
        fraction = step / max(steps - 1, 1)
        # Geometric cooling retains exploration early and settles late.
        temperature = start_temperature * (
            end_temperature / start_temperature
        ) ** fraction
        eta = step_size * (0.35 + 0.65 * temperature / start_temperature)
        y = y.detach().requires_grad_(True)
        log_pi, log_f = log_target_in_logit(y, cond)
        gradient = torch.autograd.grad(log_pi.sum(), y)[0]
        norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gradient = gradient * torch.clamp(gradient_clip / norm, max=1.0)
        with torch.no_grad():
            noise = torch.randn_like(y)
            y = y + eta * gradient + math.sqrt(2.0 * eta * temperature) * noise

        if step == 0 or (step + 1) % report_interval == 0 or step + 1 == steps:
            row = {
                "step": step + 1,
                "temperature": temperature,
                "mean_log_f": float(log_f.mean().item()),
                "q90_log_f": float(torch.quantile(log_f, 0.90).item()),
                "maximum_log_f": float(log_f.max().item()),
                "mean_gradient_norm": float(norm.mean().item()),
            }
            history.append(row)
            print(
                f"Langevin {step + 1:4d}/{steps}: T={temperature:.4g} "
                f"mean logf={row['mean_log_f']:.4f} "
                f"q90={row['q90_log_f']:.4f} max={row['maximum_log_f']:.4f}"
            )
    return initial_y, y.detach(), history


@torch.no_grad()
def summarize_and_select(model, initial_y, final_y, cond, walkers,
                         keep_per_bomb):
    initial_x = torch.sigmoid(initial_y)
    final_x = torch.sigmoid(final_y)
    initial_log_f = irregular.irregular_mixture_100_log_integrand(initial_x, cond)
    final_log_f = irregular.irregular_mixture_100_log_integrand(final_x, cond)
    final_log_q = model.log_prob(final_x, cond).reshape(-1)
    diagnostic_log_ratio = final_log_f - final_log_q
    bombs = final_x.shape[0] // walkers
    rows = []
    selected = []

    for bomb in range(bombs):
        sl = slice(bomb * walkers, (bomb + 1) * walkers)
        score = diagnostic_log_ratio[sl]
        keep = torch.topk(score, min(keep_per_bomb, walkers)).indices + bomb * walkers
        selected.append(keep)
        cloud = final_x[sl]
        spread = cloud.double().std(dim=0, unbiased=False)
        rows.append({
            "bomb": bomb,
            "condition_0": float(cond[bomb * walkers, 0].item()),
            "condition_1": float(cond[bomb * walkers, 1].item()),
            "condition_2": float(cond[bomb * walkers, 2].item()),
            "initial_mean_log_f": float(initial_log_f[sl].mean().item()),
            "final_mean_log_f": float(final_log_f[sl].mean().item()),
            "energy_gain": float((final_log_f[sl] - initial_log_f[sl]).mean().item()),
            "maximum_log_f": float(final_log_f[sl].max().item()),
            "maximum_log_f_minus_log_q": float(score.max().item()),
            "mean_coordinate_spread": float(spread.mean().item()),
            "effective_cloud_dimension": effective_cloud_dimension(cloud),
        })

    selected = torch.cat(selected)
    data = {
        "x": final_x[selected].cpu(),
        "cond": cond[selected].cpu(),
        "log_f": final_log_f[selected].cpu(),
        "log_q_nf": final_log_q[selected].cpu(),
        "diagnostic_log_f_over_q": diagnostic_log_ratio[selected].cpu(),
        "bomb": torch.div(selected, walkers, rounding_mode="floor").cpu(),
        "warning": (
            "Langevin endpoint density is unknown. These points are discovery/"
            "repair-training data, not unbiased importance samples."
        ),
    }
    return pd.DataFrame(rows), data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditions", type=int, default=16)
    parser.add_argument("--bombs-per-condition", type=int, default=8)
    parser.add_argument("--walkers-per-bomb", type=int, default=64)
    parser.add_argument("--initial-radius", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--step-size", type=float, default=2e-3)
    parser.add_argument("--start-temperature", type=float, default=1.0)
    parser.add_argument("--end-temperature", type=float, default=0.08)
    parser.add_argument("--gradient-clip", type=float, default=25.0)
    parser.add_argument("--keep-per-bomb", type=int, default=8)
    parser.add_argument("--report-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("results/irregular_mixture_100_energy_bombs"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start_temperature <= 0 or args.end_temperature <= 0:
        raise ValueError("temperatures must be positive")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    irregular.register_benchmark()
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)
    model, checkpoint = load_proposal(args.checkpoint, device)
    conditions = adaptive.sample_conditions(args.conditions, device)
    y, cond = initialize_bombs(
        model, conditions, args.bombs_per_condition,
        args.walkers_per_bomb, args.initial_radius,
    )
    print(
        f"ENERGY BOMBS: dimension={irregular.DIMENSION} "
        f"bombs={args.conditions * args.bombs_per_condition} "
        f"walkers/bomb={args.walkers_per_bomb} total={y.shape[0]}"
    )
    initial_y, final_y, history = run_langevin(
        y, cond, args.steps, args.step_size, args.start_temperature,
        args.end_temperature, args.gradient_clip, args.report_interval,
    )
    summary, training_data = summarize_and_select(
        model, initial_y, final_y, cond, args.walkers_per_bomb,
        args.keep_per_bomb,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "bomb_summary.csv", index=False)
    pd.DataFrame(history).to_csv(args.output / "trajectory_history.csv", index=False)
    torch.save(training_data, args.output / "discovered_repair_points.pt")
    metadata = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "checkpoint_beta": float(checkpoint.get("beta", float("nan"))),
        "dimension": irregular.DIMENSION,
        "selected_points": int(training_data["x"].shape[0]),
        "endpoint_density_known": False,
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\nBOMB SUMMARY")
    print(
        summary[["energy_gain", "maximum_log_f_minus_log_q",
                 "mean_coordinate_spread", "effective_cloud_dimension"]]
        .describe().to_string()
    )
    print(f"\nResults: {args.output.resolve()}")
    print("IMPORTANT: endpoints are repair-training data, not integration samples.")


if __name__ == "__main__":
    main()
