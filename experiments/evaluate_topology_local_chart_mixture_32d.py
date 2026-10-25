#!/usr/bin/env python3
"""Evaluate a protected mixture of a topology NF and local zoom charts."""

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

import evaluate_topology_mode_coverage_32d as coverage  # noqa: E402
from nf.regional_m2 import (  # noqa: E402
    box_mixture_log_prob,
    sample_box_mixture,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--zoom-regions", type=Path, required=True)
    parser.add_argument(
        "--condition", type=float, nargs=3, required=True,
        metavar=("C0", "C1", "C2"),
    )
    parser.add_argument(
        "--mixture-fractions", type=float, nargs="+",
        default=(0.0, 0.001, 0.003, 0.01, 0.03, 0.10),
    )
    parser.add_argument(
        "--reference-local-fraction", type=float, default=0.50,
        help="Local-chart fraction in the shared evaluation proposal.",
    )
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.chunk <= 0:
        parser.error("samples and chunk must be positive")
    if not 0.0 < args.reference_local_fraction < 1.0:
        parser.error("reference-local-fraction must lie in (0,1)")
    fractions = sorted(set(float(x) for x in args.mixture_fractions))
    if any(x < 0.0 or x >= 1.0 for x in fractions):
        parser.error("mixture fractions must lie in [0,1)")
    args.mixture_fractions = fractions
    if any(x < 0.0 or x > 1.0 for x in args.condition):
        parser.error("condition must lie in [0,1]^3")
    return args


def read_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def mixture_log_prob(logq, logg, fraction):
    if fraction == 0.0:
        return logq
    return torch.logaddexp(
        logq + math.log1p(-fraction),
        logg + math.log(fraction),
    )


def logmeanexp(parts, count):
    return torch.logsumexp(torch.cat(parts), dim=0) - math.log(count)


@torch.inference_mode()
def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    adaptive, model, benchmark = coverage.load_model(
        args.base_checkpoint, device
    )
    region_payload = read_payload(args.zoom_regions)
    priority_conditions = region_payload["priority_conditions"].float()
    region_centers = region_payload["centers"].float()
    halfwidth = float(region_payload.get("box_halfwidth", 0.03))
    condition_cpu = torch.tensor(args.condition, dtype=torch.float32)
    nearest = int(torch.cdist(
        condition_cpu[None, :], priority_conditions
    ).argmin())
    selected_condition = priority_conditions[nearest]
    centers = region_centers[nearest].to(device)
    condition = condition_cpu.to(device)

    distance = float(torch.linalg.vector_norm(
        condition_cpu - selected_condition
    ))
    print(
        f"condition={condition_cpu.tolist()} nearest-priority={nearest} "
        f"distance={distance:.6g} centers={len(centers)} "
        f"box-halfwidth={halfwidth:.5g}",
        flush=True,
    )

    fractions = args.mixture_fractions
    local_share = args.reference_local_fraction
    log_m2_parts = {fraction: [] for fraction in fractions}
    branch0_parts = {fraction: [] for fraction in fractions}
    branch1_parts = {fraction: [] for fraction in fractions}
    gap_parts = {fraction: [] for fraction in fractions}
    max_logw = {fraction: -math.inf for fraction in fractions}
    log_integral_parts = []
    remaining = args.samples
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    while remaining:
        count = min(remaining, args.chunk)
        cond = condition.expand(count, 3)
        choose_local = torch.rand(count, device=device) < local_share
        x = torch.empty(count, coverage.topology.DIMENSION, device=device)
        global_count = int((~choose_local).sum())
        local_count = int(choose_local.sum())
        if global_count:
            x[~choose_local] = adaptive.sample_from_model(
                model, global_count, cond[~choose_local]
            )
        if local_count:
            x[choose_local] = sample_box_mixture(
                centers, local_count, halfwidth
            )

        logq = model.log_prob(x, cond).reshape(-1).double()
        logg = box_mixture_log_prob(x, centers, halfwidth).double()
        logr = torch.logaddexp(
            logq + math.log1p(-local_share),
            logg + math.log(local_share),
        )
        logf = coverage.topology.topology_log_integrand(
            benchmark, x, cond
        ).double()
        log_integral_parts.append((logf - logr).cpu())

        branch0 = (
            x[:, 0] <= coverage.topology.BRANCH_WIDTH
        ).double()
        branch1 = (
            x[:, 0] >= 1.0 - coverage.topology.BRANCH_WIDTH
        ).double()
        gap = 1.0 - branch0 - branch1
        for fraction in fractions:
            log_mix = mixture_log_prob(logq, logg, fraction)
            log_m2_parts[fraction].append(
                (2.0 * logf - log_mix - logr).cpu()
            )
            density_ratio = torch.exp(log_mix - logr)
            branch0_parts[fraction].append(
                (density_ratio * branch0).cpu()
            )
            branch1_parts[fraction].append(
                (density_ratio * branch1).cpu()
            )
            gap_parts[fraction].append((density_ratio * gap).cpu())
            max_logw[fraction] = max(
                max_logw[fraction], float((logf - log_mix).max())
            )
        remaining -= count

    log_i = logmeanexp(log_integral_parts, args.samples)
    integral = float(torch.exp(log_i))
    target_p1 = float(coverage.topology.branch_probabilities(
        benchmark, condition.reshape(1, 3)
    )[0, 1])
    rows = []
    for fraction in fractions:
        m2 = float(torch.exp(logmeanexp(
            log_m2_parts[fraction], args.samples
        )))
        p0 = float(torch.cat(branch0_parts[fraction]).mean())
        p1 = float(torch.cat(branch1_parts[fraction]).mean())
        pgap = float(torch.cat(gap_parts[fraction]).mean())
        rows.append({
            "mixture_fraction": fraction,
            "integral_estimate": integral,
            "m2": m2,
            "normalized_ess": integral * integral / m2,
            "target_branch_1_mass": target_p1,
            "proposal_branch_0_mass": p0,
            "proposal_branch_1_mass": p1,
            "absolute_branch_1_mass_error": abs(p1 - target_p1),
            "proposal_gap_mass": pgap,
            "maximum_log_weight_on_reference_samples": max_logw[fraction],
            "theoretical_maximum_m2_ratio_to_base": 1.0 / (1.0 - fraction),
        })
    frame = pd.DataFrame(rows)
    base_m2 = float(frame.loc[
        frame["mixture_fraction"] == 0.0, "m2"
    ].iloc[0]) if (frame["mixture_fraction"] == 0.0).any() else math.nan
    frame["m2_ratio_to_base"] = frame["m2"] / base_m2
    frame.to_csv(args.output / "local_chart_mixture.csv", index=False)
    summary = {
        "benchmark": benchmark,
        "base_checkpoint": str(args.base_checkpoint),
        "zoom_regions": str(args.zoom_regions),
        "condition": list(args.condition),
        "nearest_priority_condition_index": nearest,
        "nearest_priority_condition": selected_condition.tolist(),
        "condition_distance": distance,
        "centers": len(centers),
        "box_halfwidth": halfwidth,
        "reference_local_fraction": local_share,
        "samples": args.samples,
        "integral_estimate": integral,
        "results": frame.to_dict("records"),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nLOCAL-CHART PROTECTED MIXTURE")
    print(frame.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
