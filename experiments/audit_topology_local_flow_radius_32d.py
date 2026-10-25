#!/usr/bin/env python3
"""Frozen shared-reference audit of a local-flow specialist across radii."""

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

import evaluate_topology_local_flow_mixture_32d as mixture  # noqa: E402
import evaluate_topology_mode_coverage_32d as coverage  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor", type=float, nargs=3, required=True)
    parser.add_argument(
        "--radii", type=float, nargs="+", default=(0.0, 0.005, 0.01, 0.025, 0.05)
    )
    parser.add_argument("--conditions-per-radius", type=int, default=8)
    parser.add_argument(
        "--mixture-fractions", type=float, nargs="+", default=(0.0, 0.995, 1.0)
    )
    parser.add_argument("--reference-local-fraction", type=float, default=0.90)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.radii = sorted(set(float(x) for x in args.radii))
    args.mixture_fractions = sorted(set(float(x) for x in args.mixture_fractions))
    if any(x < 0 for x in args.radii):
        parser.error("radii must be nonnegative")
    if any(x < 0 or x > 1 for x in args.mixture_fractions):
        parser.error("mixture fractions must lie in [0,1]")
    if not 0 < args.reference_local_fraction < 1:
        parser.error("reference-local-fraction must lie in (0,1)")
    if min(args.conditions_per_radius, args.samples, args.chunk) <= 0:
        parser.error("counts must be positive")
    if any(x < 0 or x > 1 for x in args.anchor):
        parser.error("anchor must lie in [0,1]^3")
    return args


def sample_conditions(anchor, radius, count, seed):
    if radius == 0.0:
        return torch.tensor(anchor, dtype=torch.float32)[None]
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(count, 3, generator=generator)
    direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    radial = torch.rand(count, 1, generator=generator).pow(1.0 / 3.0)
    anchor_tensor = torch.tensor(anchor, dtype=torch.float32)
    return (anchor_tensor + radius * radial * direction).clamp(0.0, 1.0)


@torch.inference_mode()
def audit_condition(
    adaptive, base_model, local_adaptive, local_model, benchmark,
    condition, args, device, seed,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    fractions = args.mixture_fractions
    reference_share = args.reference_local_fraction
    log_integral_sum = None
    log_m2_sum = {fraction: None for fraction in fractions}
    branch1_sum = {fraction: 0.0 for fraction in fractions}
    gap_sum = {fraction: 0.0 for fraction in fractions}
    max_log_weight = {fraction: -math.inf for fraction in fractions}
    remaining = args.samples
    while remaining:
        count = min(remaining, args.chunk)
        cond = condition.expand(count, 3)
        choose_local = torch.rand(count, device=device) < reference_share
        x = torch.empty(count, coverage.topology.DIMENSION, device=device)
        base_count = int((~choose_local).sum())
        local_count = int(choose_local.sum())
        if base_count:
            x[~choose_local] = adaptive.sample_from_model(
                base_model, base_count, cond[~choose_local]
            )
        if local_count:
            x[choose_local] = local_adaptive.sample_from_model(
                local_model, local_count, cond[choose_local]
            )
        logq = base_model.log_prob(x, cond).reshape(-1).double()
        logg = local_model.log_prob(x, cond).reshape(-1).double()
        logr = mixture.mixture_log_prob(logq, logg, reference_share)
        logf = coverage.topology.topology_log_integrand(
            benchmark, x, cond
        ).reshape(-1).double()
        log_integral_sum = mixture.update_logsum(
            log_integral_sum, logf - logr
        )
        branch0 = x[:, 0] <= coverage.topology.BRANCH_WIDTH
        branch1 = x[:, 0] >= 1.0 - coverage.topology.BRANCH_WIDTH
        gap = (~branch0 & ~branch1).double()
        branch1 = branch1.double()
        for fraction in fractions:
            logmix = mixture.mixture_log_prob(logq, logg, fraction)
            log_m2_sum[fraction] = mixture.update_logsum(
                log_m2_sum[fraction], 2.0 * logf - logmix - logr
            )
            ratio = torch.exp(logmix - logr)
            branch1_sum[fraction] += float((ratio * branch1).sum())
            gap_sum[fraction] += float((ratio * gap).sum())
            max_log_weight[fraction] = max(
                max_log_weight[fraction], float((logf - logmix).max())
            )
        remaining -= count
    integral = math.exp(float(log_integral_sum) - math.log(args.samples))
    target_p1 = float(coverage.topology.branch_probabilities(
        benchmark, condition.reshape(1, 3)
    )[0, 1])
    rows = []
    for fraction in fractions:
        m2 = math.exp(
            float(log_m2_sum[fraction]) - math.log(args.samples)
        )
        p1 = branch1_sum[fraction] / args.samples
        rows.append({
            "local_flow_fraction": fraction,
            "integral_estimate": integral,
            "m2": m2,
            "normalized_ess": integral * integral / m2,
            "target_branch_1_mass": target_p1,
            "proposal_branch_1_mass": p1,
            "absolute_branch_1_mass_error": abs(p1 - target_p1),
            "proposal_gap_mass": gap_sum[fraction] / args.samples,
            "maximum_log_weight": max_log_weight[fraction],
        })
    return rows


def radius_summary(frame):
    records = []
    for (radius, fraction), group in frame.groupby(
        ["radius", "local_flow_fraction"], sort=True
    ):
        records.append({
            "radius": radius,
            "local_flow_fraction": fraction,
            "conditions": len(group),
            "mean_integral": group["integral_estimate"].mean(),
            "maximum_absolute_integral_error": (
                group["integral_estimate"] - 1.0
            ).abs().max(),
            "median_m2": group["m2"].median(),
            "q90_m2": group["m2"].quantile(0.90),
            "maximum_m2": group["m2"].max(),
            "median_normalized_ess": group["normalized_ess"].median(),
            "minimum_normalized_ess": group["normalized_ess"].min(),
            "mean_branch_mass_error": group[
                "absolute_branch_1_mass_error"
            ].mean(),
            "maximum_branch_mass_error": group[
                "absolute_branch_1_mass_error"
            ].max(),
            "maximum_log_weight": group["maximum_log_weight"].max(),
        })
    return pd.DataFrame(records)


@torch.inference_mode()
def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    adaptive, base_model, benchmark = coverage.load_model(
        args.base_checkpoint, device
    )
    local_adaptive, local_model, local_benchmark = coverage.load_model(
        args.local_checkpoint, device
    )
    if benchmark != local_benchmark:
        raise ValueError(f"benchmark mismatch: {benchmark} vs {local_benchmark}")
    rows = []
    condition_index = 0
    for radius_index, radius in enumerate(args.radii):
        conditions = sample_conditions(
            args.anchor, radius, args.conditions_per_radius,
            args.seed + 100_000 * radius_index,
        )
        for within_radius_index, condition_cpu in enumerate(conditions):
            condition = condition_cpu.to(device)
            results = audit_condition(
                adaptive, base_model, local_adaptive, local_model,
                benchmark, condition, args, device,
                args.seed + 1009 * condition_index,
            )
            for result in results:
                rows.append({
                    "condition_index": condition_index,
                    "within_radius_index": within_radius_index,
                    "radius": radius,
                    "c0": float(condition_cpu[0]),
                    "c1": float(condition_cpu[1]),
                    "c2": float(condition_cpu[2]),
                    **result,
                })
            protected = min(
                results,
                key=lambda row: abs(row["local_flow_fraction"] - 0.995),
            )
            print(
                f"radius={radius:.4g} condition={within_radius_index + 1:02d}/"
                f"{len(conditions):02d} I={protected['integral_estimate']:.5f} "
                f"M2={protected['m2']:.6g} "
                f"ESS={protected['normalized_ess']:.4f}",
                flush=True,
            )
            condition_index += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "condition_results.csv", index=False)
    summary = radius_summary(frame)
    summary.to_csv(args.output / "radius_summary.csv", index=False)
    payload = {
        "benchmark": benchmark,
        "base_checkpoint": str(args.base_checkpoint),
        "local_checkpoint": str(args.local_checkpoint),
        "anchor": list(args.anchor),
        "radii": args.radii,
        "conditions_per_nonzero_radius": args.conditions_per_radius,
        "samples_per_condition": args.samples,
        "reference_local_fraction": args.reference_local_fraction,
        "mixture_fractions": args.mixture_fractions,
        "results": summary.to_dict("records"),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(payload, stream, indent=2)
    print("\nLOCAL-FLOW RADIUS AUDIT")
    print(summary.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
