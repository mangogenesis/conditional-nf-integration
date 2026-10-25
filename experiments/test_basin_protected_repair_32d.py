#!/usr/bin/env python3
"""Fast kill test for basin-protected repair on irregular 32D.

The discovery stage uses only log f and the frozen proposal.  Langevin "bombs"
seek regions that contribute strongly to the second moment, geometrically
diverse endpoints define local logistic-normal safety components, and a small
protected mixture fraction is reserved for those components.

The benchmark's exact sampler and hidden component labels are used only for
evaluation.  They never enter discovery or proposal construction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_irregular_geometry_metrics_32d as metrics  # noqa: E402
import run_irregular_mixture_32d as target32  # noqa: E402
from explore_energy_bombs_irregular_100d import load_proposal  # noqa: E402


family = target32.family
adaptive = target32.adaptive


def csv_floats(value):
    return [float(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditions", type=int, default=1)
    parser.add_argument("--bombs-per-condition", type=int, default=64)
    parser.add_argument("--walkers-per-bomb", type=int, default=16)
    parser.add_argument("--initial-radius", type=float, default=0.50)
    parser.add_argument("--uniform-start-fraction", type=float, default=0.50)
    parser.add_argument(
        "--oracle-start-fraction", type=float, default=0.0,
        help=(
            "Diagnostic control only: replace this fraction of bomb starts "
            "with exact target samples. Never use for a deployable result."
        ),
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--step-size", type=float, default=0.002)
    parser.add_argument("--start-temperature", type=float, default=0.50)
    parser.add_argument("--end-temperature", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=50.0)
    parser.add_argument("--maximum-centers", type=int, default=32)
    parser.add_argument("--candidate-fraction", type=float, default=0.50)
    parser.add_argument("--mixture-fractions", type=csv_floats,
                        default=[0.05, 0.10, 0.20])
    parser.add_argument("--bandwidths", type=csv_floats,
                        default=[0.25, 0.50, 1.00])
    parser.add_argument("--oracle-evals", type=int, default=20000)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def logit(x, epsilon=1e-6):
    x = x.clamp(epsilon, 1.0 - epsilon)
    return torch.log(x) - torch.log1p(-x)


def log_jacobian(y):
    return (F.logsigmoid(y) + F.logsigmoid(-y)).sum(dim=1)


def initialize_walkers(model, condition, bombs, walkers, radius,
                       uniform_fraction, oracle_fraction):
    cond_center = condition[None].expand(bombs, -1)
    centers = model.sample(bombs, cond_center)
    assignment = torch.rand(bombs, device=condition.device)
    use_oracle = assignment < oracle_fraction
    use_uniform = (
        (assignment >= oracle_fraction)
        & (assignment < oracle_fraction + uniform_fraction)
    )
    if use_oracle.any():
        oracle_cond = cond_center[use_oracle]
        oracle_x, _ = metrics.sample_target(oracle_cond)
        centers[use_oracle] = oracle_x
    if use_uniform.any():
        centers[use_uniform] = torch.rand(
            int(use_uniform.sum()), target32.DIMENSION,
            device=condition.device, dtype=condition.dtype,
        )
    center_y = logit(centers)
    y = center_y[:, None, :] + radius * torch.randn(
        bombs, walkers, target32.DIMENSION,
        device=condition.device, dtype=condition.dtype,
    )
    cond = condition[None, None, :].expand(bombs, walkers, -1)
    return y.reshape(-1, target32.DIMENSION), cond.reshape(-1, 3)


def second_moment_energy(model, y, cond):
    x = torch.sigmoid(y)
    log_p = family.irregular_mixture_100_log_integrand(x, cond)
    log_q = model.log_prob(x, cond).reshape(-1)
    return 2.0 * log_p - log_q + log_jacobian(y), log_p - log_q


def run_bombs(model, condition, args):
    y, cond = initialize_walkers(
        model, condition, args.bombs_per_condition, args.walkers_per_bomb,
        args.initial_radius, args.uniform_start_fraction,
        args.oracle_start_fraction,
    )
    for step in range(args.steps):
        fraction = step / max(args.steps - 1, 1)
        temperature = args.start_temperature * (
            args.end_temperature / args.start_temperature
        ) ** fraction
        eta = args.step_size * (0.35 + 0.65 * temperature / args.start_temperature)
        y = y.detach().requires_grad_(True)
        energy, ratio = second_moment_energy(model, y, cond)
        gradient = torch.autograd.grad(energy.sum(), y)[0]
        norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gradient = gradient * torch.clamp(args.gradient_clip / norm, max=1.0)
        y = y + eta * gradient + math.sqrt(2.0 * eta * temperature) * torch.randn_like(y)
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps:
            print(
                f"bomb step {step + 1:3d}/{args.steps}: "
                f"mean R={float(ratio.detach().mean()):.3f} "
                f"q95 R={float(torch.quantile(ratio.detach(), 0.95)):.3f} "
                f"max R={float(ratio.detach().max()):.3f}",
                flush=True,
            )

    with torch.no_grad():
        energy, ratio = second_moment_energy(model, y, cond)
        energy = energy.reshape(args.bombs_per_condition, args.walkers_per_bomb)
        ratio = ratio.reshape(args.bombs_per_condition, args.walkers_per_bomb)
        y = y.reshape(args.bombs_per_condition, args.walkers_per_bomb, -1)
        winner = energy.argmax(dim=1)
        row = torch.arange(args.bombs_per_condition, device=y.device)
        return y[row, winner].detach(), ratio[row, winner].detach()


@torch.no_grad()
def diverse_centers(endpoint_y, endpoint_ratio, maximum, candidate_fraction):
    candidate_count = max(maximum, int(math.ceil(len(endpoint_y) * candidate_fraction)))
    candidate_count = min(len(endpoint_y), candidate_count)
    candidates = torch.topk(endpoint_ratio, candidate_count).indices
    points = endpoint_y[candidates].double()
    selected = [0]
    distance = torch.full((candidate_count,), float("inf"), device=points.device)
    while len(selected) < min(maximum, candidate_count):
        newest = points[selected[-1]]
        distance = torch.minimum(distance, (points - newest).square().sum(dim=1))
        distance[torch.tensor(selected, device=points.device)] = -1.0
        selected.append(int(distance.argmax()))
    return endpoint_y[candidates[torch.tensor(selected, device=points.device)]]


@torch.no_grad()
def local_log_prob(x, centers_y, bandwidth, center_chunk=32):
    y = logit(x).double()
    terms = []
    constant = -0.5 * target32.DIMENSION * math.log(
        2.0 * math.pi * bandwidth ** 2
    )
    for start in range(0, len(centers_y), center_chunk):
        center = centers_y[start:start + center_chunk].double()
        distance2 = (y[:, None, :] - center[None, :, :]).square().sum(dim=2)
        terms.append(constant - 0.5 * distance2 / (bandwidth ** 2))
    log_q_y = torch.logsumexp(torch.cat(terms, dim=1), dim=1) - math.log(len(centers_y))
    return log_q_y - log_jacobian(y)


@torch.no_grad()
def evaluate_setting(model, condition, centers_y, fraction, bandwidth,
                     evaluations, chunk):
    ratios = []
    labels = []
    remaining = evaluations
    while remaining:
        size = min(chunk, remaining)
        cond = condition[None].expand(size, -1)
        x, component = metrics.sample_target(cond)
        log_p = family.irregular_mixture_100_log_integrand(x, cond).double()
        log_base = model.log_prob(x, cond).reshape(-1).double()
        if fraction == 0.0:
            log_q = log_base
        else:
            log_local = local_log_prob(x, centers_y, bandwidth)
            log_q = torch.logaddexp(
                log_base + math.log1p(-fraction),
                log_local + math.log(fraction),
            )
        ratios.append((log_p - log_q).cpu())
        labels.append(component.cpu())
        remaining -= size
    ratio = torch.cat(ratios)
    label = torch.cat(labels)
    row = {
        "mixture_fraction": fraction,
        "bandwidth": bandwidth,
        "forward_kl": float(ratio.mean()),
        "log_m2": metrics.logmeanexp(ratio),
        "ratio_q90": float(torch.quantile(ratio, 0.90)),
        "ratio_q99": float(torch.quantile(ratio, 0.99)),
        "target_mass_r_gt_20": float((ratio > 20.0).double().mean()),
    }
    component_rows = []
    for component in range(family.COMPONENTS):
        selected = label == component
        values = ratio[selected]
        component_rows.append({
            "mixture_fraction": fraction,
            "bandwidth": bandwidth,
            "component": component,
            "samples": int(selected.sum()),
            "mean_log_ratio": float(values.mean()),
            "q99_log_ratio": float(torch.quantile(values, 0.99)),
            "tail_mass_r_gt_20": float((values > 20.0).double().mean()),
        })
    row["worst_component_mean_log_ratio"] = max(
        item["mean_log_ratio"] for item in component_rows
    )
    row["worst_component_tail_mass_r_gt_20"] = max(
        item["tail_mass_r_gt_20"] for item in component_rows
    )
    return row, component_rows


@torch.no_grad()
def endpoint_component_responsibility(endpoint_y, condition):
    """Hidden labels are used here only to audit what discovery found."""
    x = torch.sigmoid(endpoint_y)
    cond = condition[None].expand(len(x), -1)
    parameters = family.parameters_on(x.device, x.dtype)
    log_component = torch.zeros(
        len(x), family.COMPONENTS, device=x.device, dtype=x.dtype
    )
    component_index = torch.arange(family.COMPONENTS, device=x.device)
    for child in range(target32.DIMENSION):
        if child == 0:
            a = cond[:, :1].expand(-1, family.COMPONENTS)
            b = cond[:, 1:2].expand(-1, family.COMPONENTS)
        else:
            a = x[:, parameters["parent_a"][:, child]]
            b = x[:, parameters["parent_b"][:, child]]
        frequency = parameters["frequency"][:, child][None]
        phase = parameters["phase"][:, child][None]
        wave = torch.sin(2 * math.pi * frequency * a + phase + 0.8 * cond[:, :1])
        cross = torch.cos(2 * math.pi * (a - b) + 0.5 * phase - 0.7 * cond[:, 1:2])
        switch = torch.tanh(4.0 * (cond[:, 2:3] - 0.5) + 1.2 * cross)
        signal = (
            parameters["amplitude"][None] * wave
            + parameters["interaction"][None] * cross * switch
        )
        drift = 0.035 * torch.sin(
            0.17 * child + component_index[None] + math.pi * cond[:, 1:2]
        )
        mu = (0.5 + signal + drift).clamp(0.055, 0.945)
        sigma = parameters["base_sigma"][None] * (
            0.78 + 0.38 * cond[:, 2:3]
            + 0.06 * ((child + component_index) % 5)[None]
        )
        log_component += adaptive.truncated_normal_logpdf(
            x[:, child:child + 1].expand(-1, family.COMPONENTS), mu, sigma
        )
    posterior = family.mixture_log_weights(cond, parameters) + log_component
    return posterior.argmax(dim=1)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    target32.configure_family()
    adaptive.DIM = target32.DIMENSION
    if not 0.0 <= args.oracle_start_fraction <= 1.0:
        raise ValueError("oracle-start-fraction must lie in [0,1]")
    if not 0.0 <= args.uniform_start_fraction <= 1.0:
        raise ValueError("uniform-start-fraction must lie in [0,1]")
    if args.oracle_start_fraction + args.uniform_start_fraction > 1.0:
        raise ValueError(
            "oracle-start-fraction + uniform-start-fraction must be <= 1"
        )
    if args.oracle_start_fraction > 0.0:
        print(
            "ORACLE-SEEDED CONTROL: exact target samples are being used for "
            "diagnostic initialization; this is not a deployable method.",
            flush=True,
        )
    model, checkpoint = load_proposal(args.checkpoint, device)
    conditions = adaptive.sample_conditions(args.conditions, device)

    all_summary = []
    all_components = []
    discovery_rows = []
    saved_centers = []
    for condition_index, condition in enumerate(conditions):
        print(f"\nCONDITION {condition_index}", flush=True)
        endpoints, endpoint_ratio = run_bombs(model, condition, args)
        centers = diverse_centers(
            endpoints, endpoint_ratio, args.maximum_centers,
            args.candidate_fraction,
        )
        labels = endpoint_component_responsibility(centers, condition)
        counts = torch.bincount(labels, minlength=family.COMPONENTS).cpu()
        print(f"discovered hidden-component audit counts: {counts.tolist()}")
        for component, count in enumerate(counts.tolist()):
            discovery_rows.append({
                "condition_index": condition_index,
                "component": component,
                "selected_centers": count,
            })
        saved_centers.append({
            "condition": condition.cpu(),
            "centers_y": centers.cpu(),
            "endpoint_ratio": endpoint_ratio.cpu(),
        })

        settings = [(0.0, 1.0)] + [
            (fraction, bandwidth)
            for fraction in args.mixture_fractions
            for bandwidth in args.bandwidths
        ]
        for fraction, bandwidth in settings:
            row, component_rows = evaluate_setting(
                model, condition, centers, fraction, bandwidth,
                args.oracle_evals, args.chunk,
            )
            row["condition_index"] = condition_index
            all_summary.append(row)
            for item in component_rows:
                all_components.append({"condition_index": condition_index, **item})
            print(
                f"fraction={fraction:.3f} bandwidth={bandwidth:.3f} "
                f"KL={row['forward_kl']:.3f} logM2={row['log_m2']:.3f} "
                f"worst-component={row['worst_component_mean_log_ratio']:.3f}",
                flush=True,
            )

    summary = pd.DataFrame(all_summary)
    components = pd.DataFrame(all_components)
    discovery = pd.DataFrame(discovery_rows)
    summary.to_csv(args.output / "setting_metrics.csv", index=False)
    components.to_csv(args.output / "component_metrics.csv", index=False)
    discovery.to_csv(args.output / "discovery_audit.csv", index=False)
    torch.save(saved_centers, args.output / "discovered_centers.pt")

    aggregate = summary.groupby(["mixture_fraction", "bandwidth"], as_index=False).agg(
        mean_forward_kl=("forward_kl", "mean"),
        mean_log_m2=("log_m2", "mean"),
        maximum_log_m2=("log_m2", "max"),
        mean_target_tail_mass=("target_mass_r_gt_20", "mean"),
        worst_component_mean=("worst_component_mean_log_ratio", "max"),
        worst_component_tail_mass=("worst_component_tail_mass_r_gt_20", "max"),
    )
    aggregate.to_csv(args.output / "aggregate.csv", index=False)
    best = aggregate.sort_values(
        ["maximum_log_m2", "worst_component_mean"]
    ).iloc[0].to_dict()
    result = {
        "checkpoint_beta": float(checkpoint.get("beta", float("nan"))),
        "conditions": args.conditions,
        "bombs_per_condition": args.bombs_per_condition,
        "oracle_start_fraction": args.oracle_start_fraction,
        "centers_per_condition": min(args.maximum_centers, args.bombs_per_condition),
        "best_setting_by_worst_log_m2": best,
        "warning": (
            "Hidden component labels and exact target samples were used only "
            "for evaluation, not for discovery or proposal construction."
        ),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    print("\nBASIN-PROTECTED KILL TEST")
    print(aggregate.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
