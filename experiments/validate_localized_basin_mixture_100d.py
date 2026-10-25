#!/usr/bin/env python3
"""Fresh-sample validation of a condition-localized repair mixture."""

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

import run_irregular_mixture_100d as irregular  # noqa: E402
from explore_energy_bombs_irregular_100d import load_proposal  # noqa: E402


adaptive = irregular.adaptive


def parse_floats(value):
    return [float(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-checkpoint", type=Path, required=True)
    parser.add_argument("--gate-fractions", type=parse_floats,
                        default=[0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--gate-length-scales", type=parse_floats,
                        default=[0.03, 0.05, 0.10, 0.20])
    parser.add_argument("--conditions", type=int, default=32)
    parser.add_argument("--supported-conditions", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--evals", type=int, default=100000)
    parser.add_argument("--chunk", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_models(path, device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    base, _ = load_proposal(Path(checkpoint["base_checkpoint"]), device)
    repair = copy.deepcopy(base.components[int(checkpoint["source_component"])]).to(device)
    adaptive.load_cpu_state(repair, checkpoint["repair_state_dict"])
    repair.eval()
    for parameter in repair.parameters():
        parameter.requires_grad_(False)
    return base, repair, checkpoint["gate_centers"].to(device)


def gate_probability(cond, centers, fraction, length_scale):
    distance2 = torch.cdist(cond.double(), centers.double()).square().min(dim=1).values
    return fraction * torch.exp(-0.5 * distance2 / (length_scale ** 2))


@torch.no_grad()
def one_run(base, repair, centers, cond_value, fraction, length_scale, n, chunk):
    first = 0.0
    second = 0.0
    maximum = -torch.inf
    count = 0
    pi_value = float(gate_probability(
        cond_value[None, :], centers, fraction, length_scale
    )[0])
    while count < n:
        size = min(chunk, n - count)
        cond = cond_value[None, :].expand(size, -1)
        use_repair = torch.rand(size, device=cond.device) < pi_value
        x = base.sample(size, cond)
        if use_repair.any():
            repair_cond = cond[use_repair]
            x[use_repair] = repair.sample(len(repair_cond), repair_cond)
        log_base = base.log_prob(x, cond).reshape(-1).double()
        log_repair = repair.log_prob(x, cond).reshape(-1).double()
        if pi_value <= 0.0:
            log_q = log_base
        else:
            log_q = torch.logaddexp(
                log_base + math.log1p(-pi_value),
                log_repair + math.log(pi_value),
            )
        log_f = irregular.irregular_mixture_100_log_integrand(x, cond).double()
        log_w = log_f - log_q
        w = torch.exp(log_w)
        first += float(w.sum())
        second += float(w.square().sum())
        maximum = max(maximum, float(log_w.max()))
        count += size
    estimate = first / n
    m2 = second / n
    variance = max(m2 - estimate ** 2, 0.0)
    return estimate, m2, estimate ** 2 / max(m2, 1e-300), maximum, pi_value, variance


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    adaptive.DIM = irregular.DIMENSION
    adaptive.seed_everything(args.seed)
    base, repair, centers = load_models(args.repair_checkpoint, device)

    supported_count = min(args.supported_conditions, len(centers))
    order = torch.randperm(len(centers), device=device)[:supported_count]
    supported = centers[order].float()
    fresh_count = max(args.conditions - supported_count, 0)
    fresh = adaptive.sample_conditions(fresh_count, device)
    conditions = torch.cat([supported, fresh], dim=0)
    kinds = ["supported"] * supported_count + ["fresh"] * fresh_count
    settings = [(0.0, 1.0)] + [
        (fraction, scale)
        for fraction in args.gate_fractions
        for scale in args.gate_length_scales
    ]
    rows = []
    for fraction, scale in settings:
        for condition_index, (condition, kind) in enumerate(zip(conditions, kinds)):
            for replicate in range(args.replicates):
                estimate, m2, ess, max_logw, pi, variance = one_run(
                    base, repair, centers, condition, fraction, scale,
                    args.evals, args.chunk,
                )
                rows.append({
                    "gate_fraction": fraction,
                    "gate_length_scale": scale,
                    "condition_index": condition_index,
                    "condition_kind": kind,
                    "replicate": replicate,
                    "gate_probability": pi,
                    "estimate": estimate,
                    "absolute_error": abs(estimate - 1.0),
                    "second_moment": m2,
                    "normalized_ess": ess,
                    "maximum_log_weight": max_logw,
                    "same_run_se": math.sqrt(variance / args.evals),
                })
        print(f"finished gate fraction={fraction:.3g} length={scale:.3g}")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "replicate_results.csv", index=False)
    grouped = frame.groupby(
        ["gate_fraction", "gate_length_scale", "condition_kind"], as_index=False
    ).agg(
        mean_estimate=("estimate", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        mean_second_moment=("second_moment", "mean"),
        median_second_moment=("second_moment", "median"),
        maximum_second_moment=("second_moment", "max"),
        mean_normalized_ess=("normalized_ess", "mean"),
        minimum_normalized_ess=("normalized_ess", "min"),
        maximum_log_weight=("maximum_log_weight", "max"),
    )
    grouped.to_csv(args.output / "summary.csv", index=False)

    baseline = grouped[grouped["gate_fraction"] == 0.0].set_index("condition_kind")
    candidates = grouped[grouped["gate_fraction"] > 0.0].copy()
    candidates["safe_vs_base"] = candidates.apply(
        lambda row: row["maximum_second_moment"]
        <= 1.10 * baseline.loc[row["condition_kind"], "maximum_second_moment"],
        axis=1,
    )
    setting_safety = candidates.groupby(
        ["gate_fraction", "gate_length_scale"]
    )["safe_vs_base"].all()
    safe_settings = setting_safety[setting_safety].index.tolist()
    recommendation = None
    if safe_settings:
        scored = candidates.set_index(["gate_fraction", "gate_length_scale"])
        recommendation = min(
            safe_settings,
            key=lambda setting: float(scored.loc[setting]["maximum_second_moment"].max()),
        )
    report = {
        "conditions": len(conditions),
        "supported_conditions": supported_count,
        "fresh_conditions": fresh_count,
        "replicates": args.replicates,
        "evaluations_per_run": args.evals,
        "safe_setting_count": len(safe_settings),
        "recommended_gate_fraction": None if recommendation is None else recommendation[0],
        "recommended_gate_length_scale": None if recommendation is None else recommendation[1],
        "acceptance_rule": (
            "For both supported and fresh conditions, observed worst M2 may not "
            "exceed 1.10 times the corresponding base worst M2. Among safe "
            "settings, select the smallest observed worst M2."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nLOCALIZED MIXTURE VALIDATION")
    print(json.dumps(report, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
