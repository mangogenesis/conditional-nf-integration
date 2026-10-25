#!/usr/bin/env python3
"""Evaluate protected mixtures of base and M2-refined topology proposals.

All candidate mixture fractions are estimated from the same samples drawn from
the 50/50 reference mixture.  This gives a common-random-number comparison and
avoids repeating the expensive target evaluation for every fraction.
"""

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

import compare_topology_m2_refinement_32d as paired  # noqa: E402
import evaluate_topology_mode_coverage_32d as coverage  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint", type=Path,
        help="Defaults to the source checkpoint recorded by the refined model.",
    )
    parser.add_argument(
        "--mixture-fractions", type=float, nargs="+",
        default=(0.0, 0.05, 0.10, 0.20, 0.40, 1.0),
        help="Refined-model mass lambda in (1-lambda) q_base + lambda q_refined.",
    )
    parser.add_argument(
        "--anchor", type=float, nargs=3, default=(0.5, 0.5, 0.5),
        metavar=("C0", "C1", "C2"),
    )
    parser.add_argument("--condition-radius", type=float, default=0.15)
    parser.add_argument("--conditions", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.conditions <= 0 or args.samples <= 0 or args.chunk <= 0:
        parser.error("conditions, samples, and chunk must be positive")
    if args.condition_radius < 0:
        parser.error("condition-radius must be nonnegative")
    fractions = sorted(set(float(value) for value in args.mixture_fractions))
    if any(value < 0.0 or value > 1.0 for value in fractions):
        parser.error("every mixture fraction must lie in [0, 1]")
    args.mixture_fractions = fractions
    return args


def mixture_log_prob(logq_base, logq_refined, fraction):
    if fraction == 0.0:
        return logq_base
    if fraction == 1.0:
        return logq_refined
    return torch.logaddexp(
        logq_base + math.log1p(-fraction),
        logq_refined + math.log(fraction),
    )


def logmeanexp(parts, total):
    return torch.logsumexp(torch.cat(parts), dim=0) - math.log(total)


@torch.inference_mode()
def audit_condition(
    adaptive, base_model, refined_model, benchmark, condition, args, device
):
    condition_tensor = torch.tensor(
        condition, dtype=torch.float32, device=device
    )
    fractions = args.mixture_fractions
    log_m2_terms = {fraction: [] for fraction in fractions}
    branch0_terms = {fraction: [] for fraction in fractions}
    branch1_terms = {fraction: [] for fraction in fractions}
    gap_terms = {fraction: [] for fraction in fractions}
    maximum_log_weight = {fraction: -math.inf for fraction in fractions}
    log_integral_terms = []

    remaining = args.samples
    while remaining:
        count = min(remaining, args.chunk)
        cond = condition_tensor.expand(count, 3)

        # Draw from r = 0.5 q_base + 0.5 q_refined.
        choose_refined = torch.rand(count, device=device) < 0.5
        x = torch.empty(count, coverage.topology.DIMENSION, device=device)
        base_count = int((~choose_refined).sum())
        refined_count = int(choose_refined.sum())
        if base_count:
            x[~choose_refined] = adaptive.sample_from_model(
                base_model, base_count, cond[~choose_refined]
            )
        if refined_count:
            x[choose_refined] = adaptive.sample_from_model(
                refined_model, refined_count, cond[choose_refined]
            )

        logq_base = base_model.log_prob(x, cond).reshape(-1).double()
        logq_refined = refined_model.log_prob(x, cond).reshape(-1).double()
        logr = torch.logaddexp(logq_base, logq_refined) - math.log(2.0)
        logf = coverage.topology.topology_log_integrand(
            benchmark, x, cond
        ).double()
        log_integral_terms.append((logf - logr).cpu())

        branch0 = (x[:, 0] <= coverage.topology.BRANCH_WIDTH).double()
        branch1 = (x[:, 0] >= 1.0 - coverage.topology.BRANCH_WIDTH).double()
        gap = 1.0 - branch0 - branch1
        for fraction in fractions:
            logq = mixture_log_prob(logq_base, logq_refined, fraction)
            log_m2_terms[fraction].append((2.0 * logf - logq - logr).cpu())
            density_ratio = torch.exp(logq - logr)
            branch0_terms[fraction].append((density_ratio * branch0).cpu())
            branch1_terms[fraction].append((density_ratio * branch1).cpu())
            gap_terms[fraction].append((density_ratio * gap).cpu())
            maximum_log_weight[fraction] = max(
                maximum_log_weight[fraction], float((logf - logq).max())
            )
        remaining -= count

    log_integral = logmeanexp(log_integral_terms, args.samples)
    integral = float(torch.exp(log_integral))
    target_p1 = float(coverage.topology.branch_probabilities(
        benchmark, condition_tensor.reshape(1, 3)
    )[0, 1])
    rows = []
    for fraction in fractions:
        m2 = float(torch.exp(logmeanexp(log_m2_terms[fraction], args.samples)))
        p0 = float(torch.cat(branch0_terms[fraction]).mean())
        p1 = float(torch.cat(branch1_terms[fraction]).mean())
        pgap = float(torch.cat(gap_terms[fraction]).mean())
        rows.append({
            "mixture_fraction": fraction,
            "target_branch_1_mass": target_p1,
            "proposal_branch_0_mass": p0,
            "proposal_branch_1_mass": p1,
            "proposal_gap_mass": pgap,
            "absolute_branch_1_mass_error": abs(p1 - target_p1),
            "integral_estimate": integral,
            "m2": m2,
            "normalized_ess": integral * integral / m2,
            "maximum_log_weight_on_reference_samples": (
                maximum_log_weight[fraction]
            ),
        })
    return rows


def aggregate(frame):
    rows = []
    for fraction, group in frame.groupby("mixture_fraction", sort=True):
        m2 = group["m2"].to_numpy(float)
        ess = group["normalized_ess"].to_numpy(float)
        rows.append({
            "mixture_fraction": fraction,
            "conditions": len(group),
            "mean_m2": float(np.mean(m2)),
            "median_m2": float(np.median(m2)),
            "q90_m2": float(np.quantile(m2, 0.90)),
            "maximum_m2": float(np.max(m2)),
            "mean_normalized_ess": float(np.mean(ess)),
            "median_normalized_ess": float(np.median(ess)),
            "minimum_normalized_ess": float(np.min(ess)),
            "mean_absolute_integral_error": float(
                np.mean(np.abs(group["integral_estimate"] - 1.0))
            ),
            "mean_branch_mass_error": float(
                group["absolute_branch_1_mass_error"].mean()
            ),
            "maximum_branch_mass_error": float(
                group["absolute_branch_1_mass_error"].max()
            ),
            "mean_gap_mass": float(group["proposal_gap_mass"].mean()),
            "maximum_gap_mass": float(group["proposal_gap_mass"].max()),
            "theoretical_maximum_m2_ratio_to_base": (
                math.inf if fraction == 1.0 else 1.0 / (1.0 - fraction)
            ),
        })
    summary = pd.DataFrame(rows)
    base = summary.loc[summary["mixture_fraction"] == 0.0]
    if len(base):
        base_mean = float(base.iloc[0]["mean_m2"])
        base_maximum = float(base.iloc[0]["maximum_m2"])
        summary["mean_m2_ratio_to_base"] = summary["mean_m2"] / base_mean
        summary["maximum_m2_ratio_to_base"] = (
            summary["maximum_m2"] / base_maximum
        )
    return summary


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    refined_payload = paired.read_checkpoint(
        args.refined_checkpoint, device="cpu"
    )
    base_checkpoint = paired.resolve_base_checkpoint(args, refined_payload)
    adaptive, base_model, base_benchmark = coverage.load_model(
        base_checkpoint, device
    )
    _, refined_model, refined_benchmark = coverage.load_model(
        args.refined_checkpoint, device
    )
    if base_benchmark != refined_benchmark:
        raise ValueError(
            f"benchmark mismatch: {base_benchmark!r} versus "
            f"{refined_benchmark!r}"
        )

    conditions = paired.make_conditions(
        args.anchor, args.condition_radius, args.conditions, args.seed
    )
    rows = []
    for index, condition in enumerate(conditions):
        sample_seed = args.seed + 1009 * index
        torch.manual_seed(sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sample_seed)
        condition_rows = audit_condition(
            adaptive, base_model, refined_model, base_benchmark,
            condition, args, device
        )
        for row in condition_rows:
            row.update({
                "condition_index": index,
                "c0": condition[0],
                "c1": condition[1],
                "c2": condition[2],
            })
            rows.append(row)
        compact = " ".join(
            f"l={row['mixture_fraction']:.2f}:M2={row['m2']:.4g}"
            for row in condition_rows
        )
        print(f"{index + 1:3d}/{len(conditions)} {compact}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "condition_mixture_results.csv", index=False)
    summary = aggregate(frame)
    summary.to_csv(args.output / "mixture_summary.csv", index=False)
    payload = {
        "benchmark": base_benchmark,
        "base_checkpoint": str(base_checkpoint),
        "refined_checkpoint": str(args.refined_checkpoint),
        "reference_proposal": "0.5 * q_base + 0.5 * q_refined",
        "anchor": list(args.anchor),
        "condition_radius": args.condition_radius,
        "conditions": len(conditions),
        "reference_samples_per_condition": args.samples,
        "summary": summary.replace([np.inf, -np.inf], None).to_dict("records"),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(payload, stream, indent=2)
    print("\nPROTECTED MIXTURE AUDIT")
    print(summary.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
