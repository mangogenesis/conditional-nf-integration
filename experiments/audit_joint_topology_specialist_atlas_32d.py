#!/usr/bin/env python3
"""Frozen proposal-side audit of a jointly trained specialist atlas."""

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

import evaluate_topology_mode_coverage_32d as coverage  # noqa: E402
import train_joint_topology_specialist_atlas_32d as joint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--conditions-per-center", type=int, default=8)
    parser.add_argument("--condition-radius", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.conditions_per_center, args.samples, args.chunk) <= 0:
        parser.error("counts must be positive")
    if args.condition_radius < 0:
        parser.error("condition radius must be nonnegative")
    return args


def load_atlas(path, device):
    try:
        atlas = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        atlas = torch.load(path, map_location=device)
    if atlas.get("format") != "joint_topology_specialist_atlas_v1":
        raise ValueError("unsupported atlas format")
    adaptive, global_model, benchmark = coverage.load_model(
        Path(atlas["global_checkpoint"]), device
    )
    local_adaptive, template, local_benchmark = coverage.load_model(
        Path(atlas["initial_specialist_checkpoint"]), device
    )
    del local_adaptive
    if benchmark != local_benchmark or benchmark != atlas.get("benchmark"):
        raise ValueError("atlas checkpoint benchmark mismatch")
    specialists = []
    for state in atlas["specialist_state_dicts"]:
        model = copy.deepcopy(template).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        specialists.append(model)
    centers = atlas["centers"].to(device=device, dtype=torch.float32)
    gate_args = argparse.Namespace(
        gate_length_scale=float(atlas["gate_length_scale"]),
        global_fraction=float(atlas["global_fraction"]),
    )
    return adaptive, global_model, specialists, centers, gate_args, benchmark


def conditions_around_centers(centers, count, radius, seed):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    owners = []
    for index, center in enumerate(centers.cpu()):
        rows.append(center)
        owners.append(index)
        remaining = count - 1
        if remaining:
            direction = torch.randn(remaining, 3, generator=generator)
            direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
            radial = torch.rand(remaining, 1, generator=generator).pow(1 / 3)
            rows.extend((center + radius * radial * direction).clamp(0, 1))
            owners.extend([index] * remaining)
    return torch.stack(rows), owners


@torch.inference_mode()
def audit_one(
    adaptive, global_model, specialists, centers, gate_args,
    benchmark, condition, args, device, seed,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    logw_sum = None
    logw2_sum = None
    maximum_log_weight = -math.inf
    branch0 = branch1 = gap = 0
    gate = joint.specialist_gate(
        condition.reshape(1, 3), centers,
        gate_args.gate_length_scale, gate_args.global_fraction,
    ).reshape(-1)
    remaining = args.samples
    while remaining:
        count = min(remaining, args.chunk)
        cond = condition.expand(count, 3)
        x, logq = joint.sample_atlas(
            adaptive, global_model, specialists, cond, centers, gate_args
        )
        logf = coverage.topology.topology_log_integrand(
            benchmark, x, cond
        ).reshape(-1)
        logw = (logf - logq).double()
        part1 = torch.logsumexp(logw, dim=0).cpu()
        part2 = torch.logsumexp(2 * logw, dim=0).cpu()
        logw_sum = part1 if logw_sum is None else torch.logaddexp(logw_sum, part1)
        logw2_sum = part2 if logw2_sum is None else torch.logaddexp(logw2_sum, part2)
        maximum_log_weight = max(maximum_log_weight, float(logw.max()))
        in0 = x[:, 0] <= coverage.topology.BRANCH_WIDTH
        in1 = x[:, 0] >= 1 - coverage.topology.BRANCH_WIDTH
        branch0 += int(in0.sum())
        branch1 += int(in1.sum())
        gap += int((~in0 & ~in1).sum())
        remaining -= count
    logn = math.log(args.samples)
    integral = math.exp(float(logw_sum) - logn)
    m2 = math.exp(float(logw2_sum) - logn)
    target_p1 = float(coverage.topology.branch_probabilities(
        benchmark, condition.reshape(1, 3)
    )[0, 1])
    return {
        "integral_estimate": integral,
        "m2": m2,
        "normalized_ess": integral * integral / m2,
        "maximum_log_weight": maximum_log_weight,
        "target_branch_1_mass": target_p1,
        "proposal_branch_0_mass": branch0 / args.samples,
        "proposal_branch_1_mass": branch1 / args.samples,
        "proposal_gap_mass": gap / args.samples,
        "absolute_branch_1_mass_error": abs(branch1 / args.samples - target_p1),
        "global_gate_weight": float(gate[0]),
        "largest_specialist_gate_weight": float(gate[1:].max()),
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    adaptive, global_model, specialists, centers, gate_args, benchmark = load_atlas(
        args.atlas, device
    )
    conditions, owners = conditions_around_centers(
        centers, args.conditions_per_center, args.condition_radius, args.seed
    )
    rows = []
    for index, (condition_cpu, owner) in enumerate(zip(conditions, owners)):
        condition = condition_cpu.to(device)
        result = audit_one(
            adaptive, global_model, specialists, centers, gate_args,
            benchmark, condition, args, device, args.seed + 1009 * index,
        )
        row = {
            "condition_index": index, "owner_specialist": owner,
            "c0": float(condition_cpu[0]), "c1": float(condition_cpu[1]),
            "c2": float(condition_cpu[2]), **result,
        }
        rows.append(row)
        print(
            f"{index + 1:3d}/{len(conditions)} specialist={owner:02d} "
            f"I={result['integral_estimate']:.6f} M2={result['m2']:.6g} "
            f"ESS={result['normalized_ess']:.4f} "
            f"branch-error={result['absolute_branch_1_mass_error']:.4f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "condition_results.csv", index=False)
    by_specialist = frame.groupby("owner_specialist").agg(
        conditions=("m2", "size"),
        median_m2=("m2", "median"),
        maximum_m2=("m2", "max"),
        median_ess=("normalized_ess", "median"),
        minimum_ess=("normalized_ess", "min"),
        maximum_branch_error=("absolute_branch_1_mass_error", "max"),
        maximum_log_weight=("maximum_log_weight", "max"),
    ).reset_index()
    by_specialist.to_csv(args.output / "specialist_summary.csv", index=False)
    summary = {
        "benchmark": benchmark, "atlas": str(args.atlas),
        "specialists": len(specialists), "conditions": len(frame),
        "conditions_per_center": args.conditions_per_center,
        "condition_radius": args.condition_radius,
        "samples_per_condition": args.samples,
        "median_m2": float(frame["m2"].median()),
        "q90_m2": float(frame["m2"].quantile(.90)),
        "maximum_m2": float(frame["m2"].max()),
        "median_normalized_ess": float(frame["normalized_ess"].median()),
        "minimum_normalized_ess": float(frame["normalized_ess"].min()),
        "maximum_absolute_integral_error": float(
            (frame["integral_estimate"] - 1).abs().max()
        ),
        "maximum_branch_mass_error": float(
            frame["absolute_branch_1_mass_error"].max()
        ),
        "maximum_gap_mass": float(frame["proposal_gap_mass"].max()),
        "maximum_log_weight": float(frame["maximum_log_weight"].max()),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nJOINT SPECIALIST ATLAS AUDIT")
    print(json.dumps(summary, indent=2))
    print("\nBY SPECIALIST")
    print(by_specialist.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
