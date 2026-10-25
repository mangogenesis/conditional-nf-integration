#!/usr/bin/env python3
"""Paired frozen audit of two heterogeneous proposal mixtures.

Samples come from the equal mixture r=(q_base+q_candidate)/2.  The same
points therefore estimate M2 for both proposals, strongly reducing the noise
in their difference when the proposals are protected mixtures that mostly
overlap.
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

import audit_heterogeneous_discovered_region_mixture_32d as mixture_audit  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mixture", type=Path, required=True)
    parser.add_argument("--candidate-mixture", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument(
        "--milestones", type=int, nargs="+",
        default=(100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000),
    )
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.milestones = sorted(set(args.milestones))
    if min(args.milestones) <= 0 or args.chunk <= 0:
        parser.error("milestones and chunk must be positive")
    return args


def logadd(current, values):
    part = torch.logsumexp(values.double(), dim=0).cpu()
    return part if current is None else torch.logaddexp(current, part)


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

    base = mixture_audit.load_mixture(args.base_mixture, device)
    candidate = mixture_audit.load_mixture(args.candidate_mixture, device)
    adaptive_b, models_b, weights_b, uniform_b = base
    adaptive_c, models_c, weights_c, uniform_c = candidate
    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    )

    integral_sum = None
    m2_base_sum = None
    m2_candidate_sum = None
    structure_integral = [None] * heterogeneous.COMPONENTS
    structure_m2_base = [None] * heterogeneous.COMPONENTS
    structure_m2_candidate = [None] * heterogeneous.COMPONENTS
    maximum_log_term_base = -math.inf
    maximum_log_term_candidate = -math.inf
    gap = 0
    rows = []
    total = 0
    milestone_index = 0
    maximum = args.milestones[-1]

    while total < maximum:
        next_milestone = args.milestones[milestone_index]
        n = min(args.chunk, next_milestone - total)
        cond = condition.expand(n, -1)

        # Draw exactly from r = 0.5 q_base + 0.5 q_candidate.
        choose_candidate = torch.rand(n, device=device) < 0.5
        x = torch.empty(
            n, heterogeneous.DIMENSION, dtype=cond.dtype, device=device
        )
        nb = int((~choose_candidate).sum())
        nc = int(choose_candidate.sum())
        if nb:
            xb, _ = mixture_audit.sample_mixture(
                adaptive_b, models_b, weights_b, uniform_b,
                cond[~choose_candidate],
            )
            x[~choose_candidate] = xb
        if nc:
            xc, _ = mixture_audit.sample_mixture(
                adaptive_c, models_c, weights_c, uniform_c,
                cond[choose_candidate],
            )
            x[choose_candidate] = xc

        logq_b = mixture_audit.mixture_log_prob(
            models_b, weights_b, uniform_b, x, cond
        )
        logq_c = mixture_audit.mixture_log_prob(
            models_c, weights_c, uniform_c, x, cond
        )
        log_r = torch.logaddexp(logq_b, logq_c) - math.log(2.0)
        logf = heterogeneous.heterogeneous_log_integrand(
            x, cond
        ).reshape(-1).double()
        labels = heterogeneous.structure_labels(x, cond)

        log_integral_term = logf - log_r
        log_m2_b_term = 2.0 * logf - logq_b - log_r
        log_m2_c_term = 2.0 * logf - logq_c - log_r
        integral_sum = logadd(integral_sum, log_integral_term)
        m2_base_sum = logadd(m2_base_sum, log_m2_b_term)
        m2_candidate_sum = logadd(m2_candidate_sum, log_m2_c_term)
        maximum_log_term_base = max(
            maximum_log_term_base, float(log_m2_b_term.max())
        )
        maximum_log_term_candidate = max(
            maximum_log_term_candidate, float(log_m2_c_term.max())
        )
        gap += int((labels < 0).sum())

        for index in range(heterogeneous.COMPONENTS):
            mask = labels == index
            if bool(mask.any()):
                structure_integral[index] = logadd(
                    structure_integral[index], log_integral_term[mask]
                )
                structure_m2_base[index] = logadd(
                    structure_m2_base[index], log_m2_b_term[mask]
                )
                structure_m2_candidate[index] = logadd(
                    structure_m2_candidate[index], log_m2_c_term[mask]
                )

        total += n
        if total == next_milestone:
            logn = math.log(total)
            integral = math.exp(float(integral_sum) - logn)
            m2_b = math.exp(float(m2_base_sum) - logn)
            m2_c = math.exp(float(m2_candidate_sum) - logn)
            row = {
                "samples": total,
                "integral_estimate": integral,
                "base_m2": m2_b,
                "candidate_m2": m2_c,
                "candidate_over_base_m2": m2_c / m2_b,
                "relative_m2_change": m2_c / m2_b - 1.0,
                "base_normalized_ess": integral * integral / m2_b,
                "candidate_normalized_ess": integral * integral / m2_c,
                "candidate_over_base_ess": m2_b / m2_c,
                "reference_gap_mass": gap / total,
                "maximum_log_m2_term_base": maximum_log_term_base,
                "maximum_log_m2_term_candidate": maximum_log_term_candidate,
            }
            for index, name in enumerate(heterogeneous.STRUCTURE_NAMES):
                mass = (
                    math.exp(float(structure_integral[index]) - logn)
                    if structure_integral[index] is not None else 0.0
                )
                mb = (
                    math.exp(float(structure_m2_base[index]) - logn)
                    if structure_m2_base[index] is not None else 0.0
                )
                mc = (
                    math.exp(float(structure_m2_candidate[index]) - logn)
                    if structure_m2_candidate[index] is not None else 0.0
                )
                row[f"estimated_mass_{name}"] = mass
                row[f"base_m2_{name}"] = mb
                row[f"candidate_m2_{name}"] = mc
                row[f"candidate_over_base_m2_{name}"] = (
                    mc / mb if mb > 0 else math.nan
                )
            rows.append(row)
            print(
                f"samples={total:,} I={integral:.6f} "
                f"M2 base={m2_b:.6g} candidate={m2_c:.6g} "
                f"ratio={m2_c / m2_b:.6f} | "
                + " ".join(
                    f"{name}={row[f'candidate_over_base_m2_{name}']:.4f}"
                    for name in heterogeneous.STRUCTURE_NAMES
                ),
                flush=True,
            )
            milestone_index += 1
            if milestone_index == len(args.milestones):
                break

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "paired_cumulative.csv", index=False)
    final = rows[-1]
    summary = {
        "benchmark": heterogeneous.BENCHMARK,
        "base_mixture": str(args.base_mixture),
        "candidate_mixture": str(args.candidate_mixture),
        "condition": list(args.fixed_condition),
        "reference_proposal": "0.5 * base + 0.5 * candidate",
        "common_random_numbers": True,
        "milestones": args.milestones,
        "final": final,
        "interpretation": (
            "Ratios below one favor the candidate. Stability of the paired "
            "ratio across cumulative milestones is more informative than "
            "separate noisy M2 estimates."
        ),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nPAIRED HETEROGENEOUS MIXTURE AUDIT")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
