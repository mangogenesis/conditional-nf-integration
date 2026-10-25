#!/usr/bin/env python3
"""Evaluate a protected global-NF plus dedicated-local-flow mixture."""

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

import evaluate_topology_mode_coverage_32d as coverage  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--condition", type=float, nargs=3, required=True)
    parser.add_argument(
        "--mixture-fractions", type=float, nargs="+",
        default=(0.0, 0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 1.0),
    )
    parser.add_argument("--reference-local-fraction", type=float, default=0.50)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fractions = sorted(set(float(x) for x in args.mixture_fractions))
    if any(x < 0.0 or x > 1.0 for x in fractions):
        parser.error("mixture fractions must lie in [0,1]")
    if not 0.0 < args.reference_local_fraction < 1.0:
        parser.error("reference-local-fraction must lie in (0,1)")
    if min(args.samples, args.chunk) <= 0:
        parser.error("samples and chunk must be positive")
    if any(x < 0.0 or x > 1.0 for x in args.condition):
        parser.error("condition must lie in [0,1]^3")
    args.mixture_fractions = fractions
    return args


def mixture_log_prob(logq, logg, fraction):
    if fraction == 0.0:
        return logq
    if fraction == 1.0:
        return logg
    return torch.logaddexp(
        logq + math.log1p(-fraction), logg + math.log(fraction)
    )


def update_logsum(current, values):
    value = torch.logsumexp(values.double(), dim=0).cpu()
    return value if current is None else torch.logaddexp(current, value)


@torch.inference_mode()
def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    adaptive, base_model, base_benchmark = coverage.load_model(
        args.base_checkpoint, device
    )
    local_adaptive, local_model, local_benchmark = coverage.load_model(
        args.local_checkpoint, device
    )
    if base_benchmark != local_benchmark:
        raise ValueError(
            f"benchmark mismatch: {base_benchmark} versus {local_benchmark}"
        )
    base_model.eval()
    local_model.eval()
    condition = torch.tensor(args.condition, dtype=torch.float32, device=device)
    fractions = args.mixture_fractions
    reference_share = args.reference_local_fraction
    log_m2_sum = {fraction: None for fraction in fractions}
    branch0_sum = {fraction: 0.0 for fraction in fractions}
    branch1_sum = {fraction: 0.0 for fraction in fractions}
    gap_sum = {fraction: 0.0 for fraction in fractions}
    max_log_weight = {fraction: -math.inf for fraction in fractions}
    log_integral_sum = None
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
        logr = mixture_log_prob(logq, logg, reference_share)
        logf = coverage.topology.topology_log_integrand(
            base_benchmark, x, cond
        ).reshape(-1).double()
        log_integral_sum = update_logsum(log_integral_sum, logf - logr)
        branch0 = (
            x[:, 0] <= coverage.topology.BRANCH_WIDTH
        ).double()
        branch1 = (
            x[:, 0] >= 1.0 - coverage.topology.BRANCH_WIDTH
        ).double()
        gap = 1.0 - branch0 - branch1
        for fraction in fractions:
            logmix = mixture_log_prob(logq, logg, fraction)
            log_m2_sum[fraction] = update_logsum(
                log_m2_sum[fraction], 2.0 * logf - logmix - logr
            )
            ratio = torch.exp(logmix - logr)
            branch0_sum[fraction] += float((ratio * branch0).sum())
            branch1_sum[fraction] += float((ratio * branch1).sum())
            gap_sum[fraction] += float((ratio * gap).sum())
            max_log_weight[fraction] = max(
                max_log_weight[fraction], float((logf - logmix).max())
            )
        remaining -= count
        done = args.samples - remaining
        if done == args.samples or done % 250_000 == 0:
            print(f"evaluation {done:,}/{args.samples:,}", flush=True)

    integral = math.exp(float(log_integral_sum) - math.log(args.samples))
    target_p1 = float(coverage.topology.branch_probabilities(
        base_benchmark, condition.reshape(1, 3)
    )[0, 1])
    rows = []
    for fraction in fractions:
        m2 = math.exp(
            float(log_m2_sum[fraction]) - math.log(args.samples)
        )
        p0 = branch0_sum[fraction] / args.samples
        p1 = branch1_sum[fraction] / args.samples
        pgap = gap_sum[fraction] / args.samples
        rows.append({
            "local_flow_fraction": fraction,
            "integral_estimate": integral,
            "m2": m2,
            "normalized_ess": integral * integral / m2,
            "target_branch_1_mass": target_p1,
            "proposal_branch_0_mass": p0,
            "proposal_branch_1_mass": p1,
            "absolute_branch_1_mass_error": abs(p1 - target_p1),
            "proposal_gap_mass": pgap,
            "maximum_log_weight_on_reference_samples": max_log_weight[fraction],
            "theoretical_maximum_m2_ratio_to_base": (
                math.inf if fraction == 1.0 else 1.0 / (1.0 - fraction)
            ),
        })
    frame = pd.DataFrame(rows)
    base_m2 = float(frame[frame["local_flow_fraction"] == 0.0]["m2"].iloc[0])
    frame["m2_ratio_to_base"] = frame["m2"] / base_m2
    frame.to_csv(args.output / "local_flow_mixture.csv", index=False)
    best = frame.loc[frame["m2"].idxmin()].to_dict()
    summary = {
        "benchmark": base_benchmark,
        "base_checkpoint": str(args.base_checkpoint),
        "local_checkpoint": str(args.local_checkpoint),
        "condition": list(args.condition),
        "reference_local_fraction": reference_share,
        "samples": args.samples,
        "integral_estimate": integral,
        "best": best,
        "results": frame.to_dict("records"),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nPROTECTED LOCAL-FLOW MIXTURE")
    print(frame.to_string(index=False))
    print("\nBEST CANDIDATE")
    print(json.dumps(best, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
