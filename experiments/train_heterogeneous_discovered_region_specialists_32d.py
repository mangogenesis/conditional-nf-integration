#!/usr/bin/env python3
"""Train one full-autoregressive specialist per blindly discovered region.

The input clouds are produced by
``train_heterogeneous_from_complement_regions_32d.py``.  Region membership is
recovered from separated intervals of x0; no benchmark structure labels or
target sampler are used.  Every specialist is an independent copy of the
same full-edge checkpoint and is fit only to its region's local target cloud.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_heterogeneous_topology_32d as audit  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clouds", type=Path, required=True)
    parser.add_argument("--x0-gap-threshold", type=float, default=0.02)
    parser.add_argument("--points-per-group", type=int, default=512)
    parser.add_argument("--marginal-steps", type=int, default=20000)
    parser.add_argument("--copula-steps", type=int, default=20000)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--uniform-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = (
        args.points_per_group, args.marginal_steps, args.copula_steps,
        args.check_interval, args.patience,
    )
    if min(counts) <= 0:
        parser.error("training counts must be positive")
    if not 0 < args.x0_gap_threshold < 1:
        parser.error("--x0-gap-threshold must lie in (0,1)")
    if not 0 < args.uniform_fraction < 1:
        parser.error("--uniform-fraction must lie in (0,1)")
    return args


def load_clouds(path, device):
    try:
        clouds = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        clouds = torch.load(path, map_location=device)
    for split in ("train", "validation"):
        if split not in clouds or "x" not in clouds[split]:
            raise ValueError(f"cloud file lacks {split} samples")
    return clouds


def discover_labels(x0, gap_threshold):
    order = torch.argsort(x0)
    sorted_x0 = x0[order]
    boundaries = torch.zeros(len(x0), dtype=torch.long, device=x0.device)
    if len(x0) > 1:
        boundaries[1:] = (
            sorted_x0[1:] - sorted_x0[:-1] > gap_threshold
        ).long()
    sorted_labels = boundaries.cumsum(dim=0)
    labels = torch.empty_like(sorted_labels)
    labels[order] = sorted_labels
    return labels


def split_points(cache, gap_threshold):
    x = cache["x"].reshape(-1, heterogeneous.DIMENSION)
    cond = cache["cond"].reshape(-1, heterogeneous.CONDITION_DIMENSION)
    labels = discover_labels(x[:, 0], gap_threshold)
    regions = []
    for label in torch.unique(labels, sorted=True):
        mask = labels == label
        regions.append((x[mask], cond[mask]))
    return regions


def make_cache(x, cond, points_per_group):
    groups = len(x) // points_per_group
    if groups < 1:
        raise RuntimeError("not enough samples to form a specialist cache")
    count = groups * points_per_group
    order = torch.randperm(len(x), device=x.device)[:count]
    x = x[order].reshape(groups, points_per_group, -1)
    cond = cond[order].reshape(groups, points_per_group, -1)
    alpha = torch.full(
        (groups, points_per_group), 1.0 / points_per_group,
        dtype=x.dtype, device=x.device,
    )
    return {"x": x, "cond": cond, "alpha": alpha}


def cpu_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    adaptive, template = audit.load_model(args.checkpoint, device)
    clouds = load_clouds(args.clouds, device)
    train_regions = split_points(clouds["train"], args.x0_gap_threshold)
    validation_regions = split_points(
        clouds["validation"], args.x0_gap_threshold
    )
    if len(train_regions) != len(validation_regions):
        raise RuntimeError("train/validation region counts differ")
    if len(train_regions) < 2:
        raise RuntimeError("fewer than two regions were recovered")

    adaptive.DIM = heterogeneous.DIMENSION
    adaptive.configure_speed_profile("balanced")
    adaptive.COMPILE_TRAINING_KERNELS = False
    adaptive.MAX_MARGINAL_STEPS = args.marginal_steps
    adaptive.MIN_MARGINAL_STEPS = min(2000, args.marginal_steps)
    adaptive.MARGINAL_CHECK_INTERVAL = args.check_interval
    adaptive.MARGINAL_EARLY_STOP_PATIENCE = args.patience
    adaptive.BRIDGE_COPULA_MAX_STEPS = args.copula_steps
    adaptive.BRIDGE_COPULA_CHECK_INTERVAL = args.check_interval
    adaptive.BRIDGE_COPULA_PATIENCE = args.patience

    states = []
    rows = []
    for region, ((train_x, train_cond), (val_x, val_cond)) in enumerate(
        zip(train_regions, validation_regions)
    ):
        specialist = copy.deepcopy(template).to(device)
        for parameter in specialist.parameters():
            parameter.requires_grad_(True)
        train_cache = make_cache(
            train_x, train_cond, args.points_per_group
        )
        validation_cache = make_cache(
            val_x, val_cond, args.points_per_group
        )
        print("\n" + "=" * 88, flush=True)
        print(
            f"SPECIALIST {region}/{len(train_regions) - 1}: "
            f"x0=[{float(train_x[:, 0].min()):.6f}, "
            f"{float(train_x[:, 0].max()):.6f}] "
            f"train={len(train_x):,} validation={len(val_x):,}",
            flush=True,
        )
        marginal = adaptive.train_marginals_on_dataset(
            adaptive.get_marginal_bank(specialist),
            train_cache, validation_cache, device,
            args.seed + 1000 * (region + 1),
        )
        copula = adaptive.train_bridge_copula_on_cache(
            specialist, train_cache, validation_cache,
            label=f"discovered region {region}", optimize_joint=False,
        )
        specialist.eval()
        states.append(cpu_state(specialist))
        rows.append({
            "region": region,
            "x0_minimum": float(train_x[:, 0].min()),
            "x0_maximum": float(train_x[:, 0].max()),
            "train_samples": len(train_x),
            "validation_samples": len(val_x),
            "marginal_best_step": marginal["best_step"],
            "marginal_validation_nll": marginal["val_nll"],
            "copula_best_step": copula["best_step"],
            "copula_validation_nll": copula["val_nll"],
        })
        del specialist, train_cache, validation_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mixture_path = args.output / "heterogeneous_discovered_region_mixture.pt"
    payload = {
        "format": "heterogeneous_discovered_region_mixture_v1",
        "benchmark": heterogeneous.BENCHMARK,
        "dimension": heterogeneous.DIMENSION,
        "template_checkpoint": str(args.checkpoint),
        "clouds": str(args.clouds),
        "graph": torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )["graph"],
        "specialist_state_dicts": states,
        "mixture_weights": torch.full(
            (len(states),), 1.0 / len(states), dtype=torch.float64
        ),
        "uniform_fraction": args.uniform_fraction,
        "uses_structure_labels": False,
        "uses_target_sampler": False,
        "region_discovery": "gaps in complement-refined x0 samples",
    }
    torch.save(payload, mixture_path)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "specialist_training_summary.csv", index=False)
    summary = {
        "mixture": str(mixture_path),
        "specialists": len(states),
        "mixture_weights": [1.0 / len(states)] * len(states),
        "uniform_fraction": args.uniform_fraction,
        "specialist_training": rows,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nDISCOVERED-REGION SPECIALIST TRAINING FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"\nMixture: {mixture_path.resolve()}")


if __name__ == "__main__":
    main()
