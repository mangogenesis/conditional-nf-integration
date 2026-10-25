#!/usr/bin/env python3
"""Paired frozen audit of a topology model before and after M2 refinement."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint", type=Path,
        help="Defaults to the source checkpoint recorded by the refined model.",
    )
    parser.add_argument(
        "--anchor", type=float, nargs=3, default=(0.5, 0.5, 0.5),
        metavar=("C0", "C1", "C2"),
    )
    parser.add_argument("--condition-radius", type=float, default=0.15)
    parser.add_argument("--conditions", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.conditions <= 0 or args.samples <= 0 or args.chunk <= 0:
        parser.error("conditions, samples, and chunk must be positive")
    if args.condition_radius < 0:
        parser.error("condition-radius must be nonnegative")
    return args


def read_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_base_checkpoint(args, refined_payload):
    if args.base_checkpoint is not None:
        return args.base_checkpoint
    metadata = refined_payload.get("topology_within_condition_m2_refinement", {})
    source = metadata.get("source_checkpoint")
    if not source:
        raise ValueError(
            "The refined checkpoint does not record its source. "
            "Pass --base-checkpoint explicitly."
        )
    candidate = Path(source)
    if candidate.exists():
        return candidate
    candidate_from_root = ROOT / candidate
    if candidate_from_root.exists():
        return candidate_from_root
    raise FileNotFoundError(
        f"Recorded source checkpoint does not exist: {source}. "
        "Pass --base-checkpoint explicitly."
    )


def make_conditions(anchor, radius, count, seed):
    """Return the anchor plus points sampled uniformly in a 3D ball."""
    anchor = torch.tensor(anchor, dtype=torch.float32)
    if count == 1:
        return [anchor.tolist()]
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(count - 1, 3, generator=generator)
    direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    radial = torch.rand(count - 1, 1, generator=generator).pow(1.0 / 3.0)
    local = (anchor + radius * radial * direction).clamp(0.0, 1.0)
    return [anchor.tolist(), *local.tolist()]


def metric_summary(frame, prefix):
    m2 = frame[f"{prefix}_m2"].to_numpy(float)
    ess = frame[f"{prefix}_normalized_ess"].to_numpy(float)
    integral = frame[f"{prefix}_integral_estimate"].to_numpy(float)
    return {
        "mean_m2": float(np.mean(m2)),
        "median_m2": float(np.median(m2)),
        "q90_m2": float(np.quantile(m2, 0.90)),
        "maximum_m2": float(np.max(m2)),
        "mean_normalized_ess": float(np.mean(ess)),
        "median_normalized_ess": float(np.median(ess)),
        "minimum_normalized_ess": float(np.min(ess)),
        "mean_absolute_integral_error": float(np.mean(np.abs(integral - 1.0))),
        "mean_branch_mass_error": float(
            frame[f"{prefix}_absolute_branch_1_mass_error"].mean()
        ),
        "maximum_branch_mass_error": float(
            frame[f"{prefix}_absolute_branch_1_mass_error"].max()
        ),
        "mean_gap_mass": float(frame[f"{prefix}_proposal_gap_mass"].mean()),
        "maximum_gap_mass": float(frame[f"{prefix}_proposal_gap_mass"].max()),
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    refined_payload = read_checkpoint(args.refined_checkpoint, device="cpu")
    base_checkpoint = resolve_base_checkpoint(args, refined_payload)
    adaptive, base_model, base_benchmark = coverage.load_model(
        base_checkpoint, device
    )
    _, refined_model, refined_benchmark = coverage.load_model(
        args.refined_checkpoint, device
    )
    if base_benchmark != refined_benchmark:
        raise ValueError(
            f"benchmark mismatch: {base_benchmark!r} versus {refined_benchmark!r}"
        )

    conditions = make_conditions(
        args.anchor, args.condition_radius, args.conditions, args.seed
    )
    rows = []
    for index, condition in enumerate(conditions):
        sample_seed = args.seed + 1009 * index
        results = {}
        for label, model in (("base", base_model), ("refined", refined_model)):
            torch.manual_seed(sample_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(sample_seed)
            results[label] = coverage.audit_condition(
                adaptive, model, base_benchmark, condition, args, device
            )

        row = {
            "condition_index": index,
            "c0": condition[0],
            "c1": condition[1],
            "c2": condition[2],
        }
        for label in ("base", "refined"):
            for key, value in results[label].items():
                if key not in {"c0", "c1", "c2"}:
                    row[f"{label}_{key}"] = value
        row["m2_change"] = row["refined_m2"] - row["base_m2"]
        row["m2_ratio"] = row["refined_m2"] / max(
            row["base_m2"], np.finfo(float).tiny
        )
        row["ess_change"] = (
            row["refined_normalized_ess"] - row["base_normalized_ess"]
        )
        rows.append(row)
        print(
            f"{index + 1:3d}/{len(conditions)} "
            f"M2 {row['base_m2']:.6g} -> {row['refined_m2']:.6g} "
            f"ratio={row['m2_ratio']:.4f} | "
            f"ESS {row['base_normalized_ess']:.4f} -> "
            f"{row['refined_normalized_ess']:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "paired_condition_results.csv", index=False)
    base_summary = metric_summary(frame, "base")
    refined_summary = metric_summary(frame, "refined")
    m2_ratio = frame["m2_ratio"].to_numpy(float)
    summary = {
        "benchmark": base_benchmark,
        "base_checkpoint": str(base_checkpoint),
        "refined_checkpoint": str(args.refined_checkpoint),
        "anchor": list(args.anchor),
        "condition_radius": args.condition_radius,
        "conditions": len(frame),
        "samples_per_condition_per_model": args.samples,
        "common_random_numbers": True,
        "base": base_summary,
        "refined": refined_summary,
        "paired": {
            "conditions_with_lower_m2": int(np.sum(m2_ratio < 1.0)),
            "fraction_with_lower_m2": float(np.mean(m2_ratio < 1.0)),
            "median_m2_ratio": float(np.median(m2_ratio)),
            "geometric_mean_m2_ratio": float(
                math.exp(np.mean(np.log(np.maximum(
                    m2_ratio, np.finfo(float).tiny
                ))))
            ),
            "mean_ess_change": float(frame["ess_change"].mean()),
        },
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nPAIRED FROZEN M2 AUDIT")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
