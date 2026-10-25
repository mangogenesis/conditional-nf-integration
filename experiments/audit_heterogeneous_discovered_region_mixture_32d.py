#!/usr/bin/env python3
"""Frozen importance-sampling audit of the discovered-region mixture."""

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

import audit_heterogeneous_topology_32d as base_audit  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.replicates, args.samples, args.chunk) <= 0:
        parser.error("budgets must be positive")
    return args


def load_mixture(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "heterogeneous_discovered_region_mixture_v1":
        raise ValueError("unsupported mixture format")
    adaptive, template = base_audit.load_model(
        Path(payload["template_checkpoint"]), device
    )
    specialists = []
    for state in payload["specialist_state_dicts"]:
        model = copy.deepcopy(template).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        specialists.append(model)
    weights = payload["mixture_weights"].to(
        device=device, dtype=torch.float64
    )
    weights /= weights.sum()
    return adaptive, specialists, weights, float(payload["uniform_fraction"])


def mixture_log_prob(specialists, weights, uniform_fraction, x, cond):
    component_logs = torch.stack(
        [model.log_prob(x, cond).reshape(-1).double() for model in specialists],
        dim=1,
    )
    learned = torch.logsumexp(component_logs + weights.log(), dim=1)
    return torch.logaddexp(
        learned + math.log1p(-uniform_fraction),
        torch.full_like(learned, math.log(uniform_fraction)),
    )


@torch.no_grad()
def sample_mixture(
    adaptive, specialists, weights, uniform_fraction, cond
):
    count = len(cond)
    uniform = torch.rand(count, device=cond.device) < uniform_fraction
    labels = torch.multinomial(weights.float(), count, replacement=True)
    x = torch.empty(
        count, heterogeneous.DIMENSION,
        dtype=cond.dtype, device=cond.device,
    )
    if bool(uniform.any()):
        x[uniform] = torch.rand(
            int(uniform.sum()), heterogeneous.DIMENSION,
            device=cond.device, dtype=cond.dtype,
        )
    for index, model in enumerate(specialists):
        selected = (~uniform) & (labels == index)
        n = int(selected.sum())
        if n:
            x[selected] = adaptive.sample_from_model(
                model, n, cond[selected]
            )
    logq = mixture_log_prob(
        specialists, weights, uniform_fraction, x, cond
    )
    return x, logq


def update_logsum(current, values):
    part = torch.logsumexp(values.double(), dim=0).cpu()
    return part if current is None else torch.logaddexp(current, part)


@torch.inference_mode()
def audit_once(
    adaptive, specialists, weights, uniform_fraction,
    condition, args, device, seed,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    logw_sum = logw2_sum = None
    component_logw = [None] * heterogeneous.COMPONENTS
    component_logw2 = [None] * heterogeneous.COMPONENTS
    maximum_log_weight = -math.inf
    maximum_component_log_weight = [-math.inf] * heterogeneous.COMPONENTS
    gap = 0
    remaining = args.samples
    while remaining:
        n = min(remaining, args.chunk)
        cond = condition.expand(n, -1)
        x, logq = sample_mixture(
            adaptive, specialists, weights, uniform_fraction, cond
        )
        logf = heterogeneous.heterogeneous_log_integrand(x, cond).reshape(-1)
        logw = logf.double() - logq
        labels = heterogeneous.structure_labels(x, cond)
        logw_sum = update_logsum(logw_sum, logw)
        logw2_sum = update_logsum(logw2_sum, 2 * logw)
        maximum_log_weight = max(maximum_log_weight, float(logw.max()))
        for index in range(heterogeneous.COMPONENTS):
            mask = labels == index
            if bool(mask.any()):
                component_logw[index] = update_logsum(
                    component_logw[index], logw[mask]
                )
                component_logw2[index] = update_logsum(
                    component_logw2[index], 2 * logw[mask]
                )
                maximum_component_log_weight[index] = max(
                    maximum_component_log_weight[index], float(logw[mask].max())
                )
        gap += int((labels < 0).sum())
        remaining -= n
    logn = math.log(args.samples)
    integral = math.exp(float(logw_sum) - logn)
    m2 = math.exp(float(logw2_sum) - logn)
    row = {
        "integral_estimate": integral,
        "m2": m2,
        "normalized_ess": integral * integral / m2,
        "maximum_log_weight": maximum_log_weight,
        "proposal_gap_mass": gap / args.samples,
    }
    target = heterogeneous.branch_probabilities(
        condition.reshape(1, -1)
    )[0].cpu()
    for index, name in enumerate(heterogeneous.STRUCTURE_NAMES):
        estimated = (
            math.exp(float(component_logw[index]) - logn)
            if component_logw[index] is not None else 0.0
        )
        m2_part = (
            math.exp(float(component_logw2[index]) - logn)
            if component_logw2[index] is not None else 0.0
        )
        row.update({
            f"target_mass_{name}": float(target[index]),
            f"estimated_mass_{name}": estimated,
            f"absolute_mass_error_{name}": abs(estimated - float(target[index])),
            f"m2_contribution_{name}": m2_part,
            f"conditional_ess_{name}": (
                estimated * estimated / m2_part if m2_part > 0 else 0.0
            ),
            f"maximum_log_weight_{name}": maximum_component_log_weight[index],
        })
    return row


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    adaptive, specialists, weights, uniform_fraction = load_mixture(
        args.mixture, device
    )
    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    )
    rows = []
    for replicate in range(args.replicates):
        row = audit_once(
            adaptive, specialists, weights, uniform_fraction,
            condition, args, device, args.seed + 1009 * replicate,
        )
        row["replicate"] = replicate
        rows.append(row)
        details = " ".join(
            f"{name}={row[f'estimated_mass_{name}']:.3f}"
            for name in heterogeneous.STRUCTURE_NAMES
        )
        print(
            f"replicate={replicate + 1:02d}/{args.replicates:02d} "
            f"I={row['integral_estimate']:.6f} M2={row['m2']:.6g} "
            f"ESS={row['normalized_ess']:.4f} | {details}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "replicate_results.csv", index=False)
    structures = []
    for name in heterogeneous.STRUCTURE_NAMES:
        structures.append({
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
            "maximum_log_weight": float(
                frame[f"maximum_log_weight_{name}"].max()
            ),
        })
    pd.DataFrame(structures).to_csv(
        args.output / "structure_summary.csv", index=False
    )
    summary = {
        "benchmark": heterogeneous.BENCHMARK,
        "mixture": str(args.mixture),
        "specialists": len(specialists),
        "mixture_weights": weights.cpu().tolist(),
        "uniform_fraction": uniform_fraction,
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
        "structures": structures,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nDISCOVERED-REGION MIXTURE AUDIT")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
