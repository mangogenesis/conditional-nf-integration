#!/usr/bin/env python3
"""Frozen structure-resolved audit for the heterogeneous 32D benchmark."""

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

import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.replicates, args.samples, args.chunk) <= 0:
        parser.error("budgets must be positive")
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("condition must lie in [0,1]^3")
    return args


def _update_logsum(current, values):
    value = torch.logsumexp(values.double(), dim=0).cpu()
    return value if current is None else torch.logaddexp(current, value)


def load_model(path, device):
    heterogeneous.register_benchmark()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    benchmark = checkpoint.get("benchmark")
    if benchmark != heterogeneous.BENCHMARK:
        raise ValueError(
            f"expected {heterogeneous.BENCHMARK!r}, found {benchmark!r}"
        )
    adaptive = heterogeneous.adaptive
    adaptive.DIM = heterogeneous.DIMENSION
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return adaptive, model


@torch.inference_mode()
def audit_replicate(adaptive, model, condition, args, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    logw_sum = None
    logw2_sum = None
    component_logw_sum = [None] * heterogeneous.COMPONENTS
    component_logw2_sum = [None] * heterogeneous.COMPONENTS
    proposal_counts = torch.zeros(heterogeneous.COMPONENTS, dtype=torch.int64)
    gap_count = 0
    maximum_log_weight = -math.inf
    maximum_component_log_weight = [-math.inf] * heterogeneous.COMPONENTS
    remaining = args.samples
    while remaining:
        count = min(remaining, args.chunk)
        cond = condition.expand(count, heterogeneous.CONDITION_DIMENSION)
        x = adaptive.sample_from_model(model, count, cond)
        logq = model.log_prob(x, cond).reshape(-1)
        logf = heterogeneous.heterogeneous_log_integrand(x, cond)
        logw = (logf - logq).double()
        labels = heterogeneous.structure_labels(x, cond)
        logw_sum = _update_logsum(logw_sum, logw)
        logw2_sum = _update_logsum(logw2_sum, 2 * logw)
        maximum_log_weight = max(maximum_log_weight, float(logw.max()))
        for component in range(heterogeneous.COMPONENTS):
            mask = labels == component
            proposal_counts[component] += int(mask.sum())
            if bool(mask.any()):
                component_logw_sum[component] = _update_logsum(
                    component_logw_sum[component], logw[mask]
                )
                component_logw2_sum[component] = _update_logsum(
                    component_logw2_sum[component], 2 * logw[mask]
                )
                maximum_component_log_weight[component] = max(
                    maximum_component_log_weight[component],
                    float(logw[mask].max()),
                )
        gap_count += int((labels < 0).sum())
        remaining -= count
    logn = math.log(args.samples)
    integral = math.exp(float(logw_sum) - logn)
    m2 = math.exp(float(logw2_sum) - logn)
    target_mass = heterogeneous.branch_probabilities(
        condition.reshape(1, -1)
    )[0].cpu()
    row = {
        "integral_estimate": integral,
        "m2": m2,
        "normalized_ess": integral * integral / m2,
        "maximum_log_weight": maximum_log_weight,
        "proposal_gap_mass": gap_count / args.samples,
    }
    for component, name in enumerate(heterogeneous.STRUCTURE_NAMES):
        if component_logw_sum[component] is None:
            estimated_mass = 0.0
            m2_contribution = 0.0
        else:
            estimated_mass = math.exp(
                float(component_logw_sum[component]) - logn
            )
            m2_contribution = math.exp(
                float(component_logw2_sum[component]) - logn
            )
        true_mass = float(target_mass[component])
        row.update({
            f"target_mass_{name}": true_mass,
            f"estimated_mass_{name}": estimated_mass,
            f"absolute_mass_error_{name}": abs(estimated_mass - true_mass),
            f"proposal_mass_{name}": float(proposal_counts[component]) / args.samples,
            f"m2_contribution_{name}": m2_contribution,
            f"conditional_ess_{name}": (
                true_mass * true_mass / m2_contribution
                if m2_contribution > 0 else 0.0
            ),
            f"maximum_log_weight_{name}": maximum_component_log_weight[component],
        })
    return row


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    adaptive, model = load_model(args.checkpoint, device)
    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    )
    rows = []
    for replicate in range(args.replicates):
        row = audit_replicate(
            adaptive, model, condition, args, device,
            args.seed + 1009 * replicate,
        )
        row.update({
            "replicate": replicate,
            "c0": args.fixed_condition[0],
            "c1": args.fixed_condition[1],
            "c2": args.fixed_condition[2],
        })
        rows.append(row)
        component_text = " ".join(
            f"{name}={row[f'conditional_ess_{name}']:.3f}"
            for name in heterogeneous.STRUCTURE_NAMES
        )
        print(
            f"replicate={replicate + 1:02d}/{args.replicates:02d} "
            f"I={row['integral_estimate']:.6f} M2={row['m2']:.6g} "
            f"ESS={row['normalized_ess']:.4f} | {component_text}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "replicate_results.csv", index=False)
    structure_summary = []
    for name in heterogeneous.STRUCTURE_NAMES:
        structure_summary.append({
            "structure": name,
            "target_mass": float(frame[f"target_mass_{name}"].iloc[0]),
            "mean_estimated_mass": float(frame[f"estimated_mass_{name}"].mean()),
            "maximum_absolute_mass_error": float(
                frame[f"absolute_mass_error_{name}"].max()
            ),
            "median_m2_contribution": float(
                frame[f"m2_contribution_{name}"].median()
            ),
            "median_conditional_ess": float(
                frame[f"conditional_ess_{name}"].median()
            ),
            "minimum_conditional_ess": float(
                frame[f"conditional_ess_{name}"].min()
            ),
            "maximum_log_weight": float(
                frame[f"maximum_log_weight_{name}"].max()
            ),
        })
    structure_frame = pd.DataFrame(structure_summary)
    structure_frame.to_csv(args.output / "structure_summary.csv", index=False)
    summary = {
        "benchmark": heterogeneous.BENCHMARK,
        "checkpoint": str(args.checkpoint),
        "condition": list(args.fixed_condition),
        "replicates": args.replicates,
        "samples_per_replicate": args.samples,
        "mean_integral": float(frame["integral_estimate"].mean()),
        "maximum_absolute_integral_error": float(
            (frame["integral_estimate"] - 1).abs().max()
        ),
        "median_m2": float(frame["m2"].median()),
        "maximum_m2": float(frame["m2"].max()),
        "median_normalized_ess": float(frame["normalized_ess"].median()),
        "minimum_normalized_ess": float(frame["normalized_ess"].min()),
        "maximum_log_weight": float(frame["maximum_log_weight"].max()),
        "maximum_proposal_gap_mass": float(frame["proposal_gap_mass"].max()),
        "structures": structure_summary,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nHETEROGENEOUS TOPOLOGY AUDIT")
    print(json.dumps(summary, indent=2))
    print("\nSTRUCTURE-RESOLVED SUMMARY")
    print(structure_frame.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
