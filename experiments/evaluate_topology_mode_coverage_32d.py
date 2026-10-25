#!/usr/bin/env python3
"""Audit whether a frozen topology-ladder NF covers every target branch."""

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

import run_topology_ladder_32d as topology  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditions", type=int, default=32)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3,
        metavar=("C0", "C1", "C2"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.conditions <= 0 or args.samples <= 0 or args.chunk <= 0:
        parser.error("conditions, samples, and chunk must be positive")
    return args


def load_model(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    benchmark = checkpoint.get("benchmark")
    if benchmark not in topology.BENCHMARKS:
        raise ValueError(f"{benchmark!r} is not a topology-ladder benchmark")
    topology.register_benchmarks()
    adaptive = topology.adaptive
    adaptive.DIM = topology.DIMENSION
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return adaptive, model, benchmark


@torch.inference_mode()
def audit_condition(adaptive, model, benchmark, condition, args, device):
    count0 = count1 = gap = 0
    logw_parts = []
    maximum_log_weight = -math.inf
    condition_tensor = torch.tensor(
        condition, dtype=torch.float32, device=device
    )
    remaining = args.samples
    while remaining:
        count = min(remaining, args.chunk)
        cond = condition_tensor.expand(count, 3)
        x = adaptive.sample_from_model(model, count, cond)
        logq = model.log_prob(x, cond).reshape(-1)
        logf = topology.topology_log_integrand(benchmark, x, cond)
        logw = (logf - logq).double()
        logw_parts.append(logw.cpu())
        maximum_log_weight = max(maximum_log_weight, float(logw.max()))
        count0 += int((x[:, 0] <= topology.BRANCH_WIDTH).sum())
        count1 += int((x[:, 0] >= 1.0 - topology.BRANCH_WIDTH).sum())
        gap += int((
            (x[:, 0] > topology.BRANCH_WIDTH)
            & (x[:, 0] < 1.0 - topology.BRANCH_WIDTH)
        ).sum())
        remaining -= count
    logw = torch.cat(logw_parts)
    logn = math.log(len(logw))
    log_sum_w = torch.logsumexp(logw, dim=0)
    log_sum_w2 = torch.logsumexp(2.0 * logw, dim=0)
    integral = float(torch.exp(log_sum_w - logn))
    m2 = float(torch.exp(log_sum_w2 - logn))
    ess = float(torch.exp(2.0 * log_sum_w - logn - log_sum_w2))
    target_p1 = float(topology.branch_probabilities(
        benchmark, condition_tensor.reshape(1, 3)
    )[0, 1])
    return {
        "c0": float(condition[0]),
        "c1": float(condition[1]),
        "c2": float(condition[2]),
        "target_branch_1_mass": target_p1,
        "proposal_branch_0_mass": count0 / args.samples,
        "proposal_branch_1_mass": count1 / args.samples,
        "proposal_gap_mass": gap / args.samples,
        "absolute_branch_1_mass_error": abs(count1 / args.samples - target_p1),
        "integral_estimate": integral,
        "m2": m2,
        "normalized_ess": ess,
        "maximum_log_weight": maximum_log_weight,
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    adaptive, model, benchmark = load_model(args.checkpoint, device)
    if args.fixed_condition is not None:
        conditions = [args.fixed_condition]
    else:
        generator = torch.Generator().manual_seed(args.seed)
        conditions = torch.rand(args.conditions, 3, generator=generator).tolist()
    rows = []
    for index, condition in enumerate(conditions):
        torch.manual_seed(args.seed + 1009 * index)
        row = audit_condition(
            adaptive, model, benchmark, condition, args, device
        )
        row["condition_index"] = index
        rows.append(row)
        print(
            f"{index + 1:3d}/{len(conditions)} "
            f"target p1={row['target_branch_1_mass']:.3f} "
            f"NF p1={row['proposal_branch_1_mass']:.3f} "
            f"gap={row['proposal_gap_mass']:.3g} "
            f"I={row['integral_estimate']:.6f} "
            f"M2={row['m2']:.5g}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "topology_mode_coverage.csv", index=False)
    summary = {
        "benchmark": benchmark,
        "conditions": len(frame),
        "samples_per_condition": args.samples,
        "mean_absolute_branch_mass_error": float(
            frame["absolute_branch_1_mass_error"].mean()
        ),
        "maximum_absolute_branch_mass_error": float(
            frame["absolute_branch_1_mass_error"].max()
        ),
        "mean_proposal_gap_mass": float(frame["proposal_gap_mass"].mean()),
        "maximum_proposal_gap_mass": float(frame["proposal_gap_mass"].max()),
        "mean_integral_estimate": float(frame["integral_estimate"].mean()),
        "maximum_m2": float(frame["m2"].max()),
        "minimum_normalized_ess": float(frame["normalized_ess"].min()),
        "interpretation": (
            "A successful topology model matches branch masses, assigns "
            "little proposal mass to the zero-density gap, and maintains "
            "stable importance weights."
        ),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nTOPOLOGY MODE-COVERAGE SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
