#!/usr/bin/env python3
"""Protect the global NF and add specialists only for underfit regions.

Regions are recovered from gaps in the previously discovered x0 clouds.  The
region on which the frozen global NF has the highest held-out median log
density is assigned to that unchanged component.  Independent copies are
fine-tuned only for the remaining regions.  No benchmark structure labels or
known structure count are used.
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
import train_heterogeneous_discovered_region_specialists_32d as discovered  # noqa: E402


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
    parser.add_argument("--seed", type=int, default=20261122)
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


@torch.inference_mode()
def region_logq_summary(model, x, cond, chunk=8192):
    blocks = []
    for start in range(0, len(x), chunk):
        blocks.append(
            model.log_prob(
                x[start:start + chunk], cond[start:start + chunk]
            ).reshape(-1).double()
        )
    values = torch.cat(blocks)
    finite = values[torch.isfinite(values)]
    if not len(finite):
        return {"median_logq": float("-inf"), "mean_logq": float("-inf")}
    return {
        "median_logq": float(finite.median()),
        "mean_logq": float(finite.mean()),
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

    adaptive, global_model = audit.load_model(args.checkpoint, device)
    global_model.eval()
    for parameter in global_model.parameters():
        parameter.requires_grad_(False)
    clouds = discovered.load_clouds(args.clouds, device)
    train_regions = discovered.split_points(
        clouds["train"], args.x0_gap_threshold
    )
    validation_regions = discovered.split_points(
        clouds["validation"], args.x0_gap_threshold
    )
    if len(train_regions) != len(validation_regions):
        raise RuntimeError("train/validation region counts differ")
    if len(train_regions) < 2:
        raise RuntimeError("fewer than two regions were recovered")

    coverage = []
    for region, (_, (val_x, val_cond)) in enumerate(
        zip(train_regions, validation_regions)
    ):
        stats = region_logq_summary(global_model, val_x, val_cond)
        coverage.append({
            "region": region,
            "x0_minimum": float(val_x[:, 0].min()),
            "x0_maximum": float(val_x[:, 0].max()),
            "validation_samples": len(val_x),
            **stats,
        })
    owner = max(coverage, key=lambda row: row["median_logq"])["region"]
    print("\nGLOBAL COMPONENT COVERAGE", flush=True)
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)
    print(
        f"\nProtected global component owns discovered region {owner}; "
        "its parameters will not be changed.",
        flush=True,
    )

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

    states = [discovered.cpu_state(global_model)]
    rows = [{
        **coverage[owner],
        "role": "protected_global",
        "marginal_best_step": 0,
        "copula_best_step": 0,
    }]
    for region, ((train_x, train_cond), (val_x, val_cond)) in enumerate(
        zip(train_regions, validation_regions)
    ):
        if region == owner:
            continue
        specialist = copy.deepcopy(global_model).to(device)
        for parameter in specialist.parameters():
            parameter.requires_grad_(True)
        train_cache = discovered.make_cache(
            train_x, train_cond, args.points_per_group
        )
        validation_cache = discovered.make_cache(
            val_x, val_cond, args.points_per_group
        )
        print("\n" + "=" * 88, flush=True)
        print(
            f"NEW SPECIALIST FOR REGION {region}: "
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
            label=f"protected mixture region {region}", optimize_joint=False,
        )
        specialist.eval()
        for parameter in specialist.parameters():
            parameter.requires_grad_(False)
        states.append(discovered.cpu_state(specialist))
        rows.append({
            **coverage[region],
            "role": "new_specialist",
            "train_samples": len(train_x),
            "marginal_best_step": marginal["best_step"],
            "marginal_validation_nll": marginal["val_nll"],
            "copula_best_step": copula["best_step"],
            "copula_validation_nll": copula["val_nll"],
        })
        del specialist, train_cache, validation_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Balanced discovered clouds imply one component per recovered region.
    # The untouched global component represents its best-covered region.
    weights = torch.full(
        (len(states),), 1.0 / len(states), dtype=torch.float64
    )
    mixture_path = args.output / "heterogeneous_protected_global_mixture.pt"
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    payload = {
        "format": "heterogeneous_discovered_region_mixture_v1",
        "benchmark": heterogeneous.BENCHMARK,
        "dimension": heterogeneous.DIMENSION,
        "template_checkpoint": str(args.checkpoint),
        "clouds": str(args.clouds),
        "graph": checkpoint["graph"],
        "specialist_state_dicts": states,
        "mixture_weights": weights,
        "uniform_fraction": args.uniform_fraction,
        "uses_structure_labels": False,
        "uses_known_structure_count": False,
        "uses_target_sampler": False,
        "protected_global_component": True,
        "global_owned_region": owner,
        "region_discovery": "gaps in complement-refined x0 samples",
    }
    torch.save(payload, mixture_path)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "protected_specialist_summary.csv", index=False)
    summary = {
        "mixture": str(mixture_path),
        "discovered_regions": len(train_regions),
        "protected_global_region": owner,
        "new_specialists": len(states) - 1,
        "mixture_weights": weights.tolist(),
        "uniform_fraction": args.uniform_fraction,
        "coverage": coverage,
        "training": rows,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nPROTECTED GLOBAL SPECIALIST TRAINING FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"\nMixture: {mixture_path.resolve()}")


if __name__ == "__main__":
    main()
