#!/usr/bin/env python3
"""Train protected local repairs from fresh target neighborhoods.

The current mixture is used only to locate high-residual seeds and assign each
seed to the component with largest posterior responsibility.  Fresh,
independent MALA chains starting from those seeds generate train and validation
target neighborhoods.  No benchmark structure labels or structure count are
used.  Accepted repairs receive a small fixed total mixture share; the base
mixture proportions are otherwise preserved.
"""

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

import audit_heterogeneous_discovered_region_mixture_32d as mixture_audit  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402
import train_heterogeneous_discovered_region_specialists_32d as initial  # noqa: E402
import train_heterogeneous_from_complement_regions_32d as local_cloud  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--clouds", type=Path, required=True)
    parser.add_argument("--residual-fraction", type=float, default=0.10)
    parser.add_argument("--train-walkers", type=int, default=256)
    parser.add_argument("--validation-walkers", type=int, default=128)
    parser.add_argument("--train-draws-per-walker", type=int, default=16)
    parser.add_argument("--validation-draws-per-walker", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=400)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--mala-step-size", type=float, default=0.001)
    parser.add_argument("--mala-gradient-clip", type=float, default=50.0)
    parser.add_argument("--points-per-group", type=int, default=512)
    parser.add_argument("--marginal-steps", type=int, default=10000)
    parser.add_argument("--copula-steps", type=int, default=10000)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--repair-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = (
        args.train_walkers, args.validation_walkers,
        args.train_draws_per_walker, args.validation_draws_per_walker,
        args.burn_in, args.thin, args.points_per_group,
        args.marginal_steps, args.copula_steps,
        args.check_interval, args.patience,
    )
    if min(counts) <= 0:
        parser.error("all budgets must be positive")
    if not 0 < args.residual_fraction < 1:
        parser.error("--residual-fraction must lie in (0,1)")
    if not 0 < args.repair_fraction < 0.5:
        parser.error("--repair-fraction must lie in (0,0.5)")
    if args.mala_step_size <= 0 or args.mala_gradient_clip <= 0:
        parser.error("MALA parameters must be positive")
    return args


def load_payload(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "heterogeneous_discovered_region_mixture_v1":
        raise ValueError("unsupported mixture format")
    return payload


def flatten(cache):
    return (
        cache["x"].reshape(-1, heterogeneous.DIMENSION),
        cache["cond"].reshape(-1, heterogeneous.CONDITION_DIMENSION),
    )


@torch.no_grad()
def score_cloud(models, weights, uniform_fraction, x, cond, chunk=2048):
    component_rows = []
    logf_rows = []
    for start in range(0, len(x), chunk):
        stop = start + chunk
        component = torch.stack([
            model.log_prob(x[start:stop], cond[start:stop]).reshape(-1).double()
            for model in models
        ], dim=1)
        component_rows.append(component)
        logf_rows.append(
            heterogeneous.heterogeneous_log_integrand(
                x[start:stop], cond[start:stop]
            ).reshape(-1).double()
        )
    component = torch.cat(component_rows)
    logf = torch.cat(logf_rows)
    log_learned = torch.logsumexp(component + weights.log(), dim=1)
    logq = torch.logaddexp(
        log_learned + math.log1p(-uniform_fraction),
        torch.full_like(log_learned, math.log(uniform_fraction)),
    )
    responsibility = (component + weights.log()).argmax(dim=1)
    return {
        "component": component,
        "logf": logf,
        "logq": logq,
        "residual": 2.0 * logf - logq,
        "responsibility": responsibility,
    }


def choose_seeds(x, stats, owner, walkers, fraction, generator):
    owned = torch.nonzero(
        stats["responsibility"] == owner, as_tuple=False
    ).flatten()
    if not len(owned):
        return None, None
    keep = max(1, int(math.ceil(fraction * len(owned))))
    ranked = owned[torch.topk(stats["residual"][owned], keep).indices]
    chosen = ranked[torch.randint(
        0, len(ranked), (walkers,), generator=generator,
        device=ranked.device,
    )]
    return x[chosen].detach().clone(), {
        "owned_points": len(owned),
        "residual_seed_points": len(ranked),
        "median_seed_residual": float(stats["residual"][ranked].median()),
        "maximum_seed_residual": float(stats["residual"][ranked].max()),
    }


def cache_from_samples(x, condition, points_per_group):
    groups = len(x) // points_per_group
    if groups < 1:
        raise RuntimeError("fresh cloud cannot form a training group")
    count = groups * points_per_group
    order = torch.randperm(len(x), device=x.device)[:count]
    x = x[order].reshape(groups, points_per_group, -1)
    cond = condition.expand(count, -1).reshape(groups, points_per_group, -1)
    alpha = torch.full(
        (groups, points_per_group), 1.0 / points_per_group,
        dtype=x.dtype, device=x.device,
    )
    return {"x": x, "cond": cond.clone(), "alpha": alpha}


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

    payload = load_payload(args.mixture, device)
    adaptive, models, weights, uniform_fraction = mixture_audit.load_mixture(
        args.mixture, device
    )
    weights = weights.to(device=device, dtype=torch.float64)
    clouds = initial.load_clouds(args.clouds, device)
    train_x, train_cond = flatten(clouds["train"])
    validation_x, validation_cond = flatten(clouds["validation"])
    train_stats = score_cloud(
        models, weights, uniform_fraction, train_x, train_cond
    )
    validation_stats = score_cloud(
        models, weights, uniform_fraction, validation_x, validation_cond
    )
    condition = train_cond[:1]
    lower = torch.zeros(heterogeneous.DIMENSION, device=device)
    upper = torch.ones_like(lower)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    adaptive.DIM = heterogeneous.DIMENSION
    adaptive.configure_speed_profile("balanced")
    adaptive.COMPILE_TRAINING_KERNELS = False
    adaptive.MAX_MARGINAL_STEPS = args.marginal_steps
    adaptive.MIN_MARGINAL_STEPS = min(1000, args.marginal_steps)
    adaptive.MARGINAL_CHECK_INTERVAL = args.check_interval
    adaptive.MARGINAL_EARLY_STOP_PATIENCE = args.patience
    adaptive.BRIDGE_COPULA_MAX_STEPS = args.copula_steps
    adaptive.BRIDGE_COPULA_CHECK_INTERVAL = args.check_interval
    adaptive.BRIDGE_COPULA_PATIENCE = args.patience

    repairs = []
    rows = []
    fresh_clouds = {}
    for owner in range(len(models)):
        train_initial, train_seed_info = choose_seeds(
            train_x, train_stats, owner, args.train_walkers,
            args.residual_fraction, generator,
        )
        validation_initial, validation_seed_info = choose_seeds(
            validation_x, validation_stats, owner, args.validation_walkers,
            args.residual_fraction, generator,
        )
        if train_initial is None or validation_initial is None:
            print(f"owner={owner}: no residual seeds; skipping", flush=True)
            continue
        print("\n" + "=" * 88, flush=True)
        print(
            f"FRESH LOCAL REPAIR owner={owner} "
            f"train-seeds={train_seed_info['residual_seed_points']} "
            f"validation-seeds={validation_seed_info['residual_seed_points']}",
            flush=True,
        )
        train_samples, train_mala = local_cloud.sample_local_region(
            train_initial, lower, upper, condition,
            args.train_draws_per_walker, args.burn_in, args.thin,
            args.mala_step_size, args.mala_gradient_clip,
            f"repair owner={owner} train",
        )
        validation_samples, validation_mala = local_cloud.sample_local_region(
            validation_initial, lower, upper, condition,
            args.validation_draws_per_walker, args.burn_in, args.thin,
            args.mala_step_size, args.mala_gradient_clip,
            f"repair owner={owner} validation",
        )
        train_cache = cache_from_samples(
            train_samples, condition, args.points_per_group
        )
        validation_cache = cache_from_samples(
            validation_samples, condition, args.points_per_group
        )
        specialist = copy.deepcopy(models[owner]).to(device)
        for parameter in specialist.parameters():
            parameter.requires_grad_(True)
        marginal = adaptive.train_marginals_on_dataset(
            adaptive.get_marginal_bank(specialist),
            train_cache, validation_cache, device,
            args.seed + 1000 * (owner + 1),
        )
        copula = adaptive.train_bridge_copula_on_cache(
            specialist, train_cache, validation_cache,
            label=f"fresh local repair owner {owner}", optimize_joint=False,
        )
        learned = marginal["best_step"] > 0 or copula["best_step"] > 0
        row = {
            "owner": owner,
            **{f"train_{key}": value for key, value in train_seed_info.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_seed_info.items()
            },
            "train_mala_acceptance": train_mala["mean_acceptance"],
            "validation_mala_acceptance": validation_mala["mean_acceptance"],
            "marginal_best_step": marginal["best_step"],
            "copula_best_step": copula["best_step"],
            "accepted_for_audit": learned,
        }
        rows.append(row)
        if learned:
            specialist.eval()
            for parameter in specialist.parameters():
                parameter.requires_grad_(False)
            repairs.append(specialist)
            fresh_clouds[f"owner_{owner}"] = {
                "train": train_samples.cpu(),
                "validation": validation_samples.cpu(),
            }
        else:
            print(
                f"owner={owner}: no held-out likelihood improvement; discarded",
                flush=True,
            )

    if repairs:
        protected_base = (1.0 - args.repair_fraction) * weights
        repair_weights = torch.full(
            (len(repairs),), args.repair_fraction / len(repairs),
            dtype=torch.float64, device=device,
        )
        final_models = [*models, *repairs]
        final_weights = torch.cat((protected_base, repair_weights))
    else:
        final_models = models
        final_weights = weights

    output_path = args.output / "heterogeneous_fresh_local_repair_mixture.pt"
    result = {
        **payload,
        "specialist_state_dicts": [cpu_state(model) for model in final_models],
        "mixture_weights": final_weights.detach().cpu(),
        "fresh_local_repair": {
            "base_components": len(models),
            "accepted_repairs": len(repairs),
            "repair_fraction": args.repair_fraction if repairs else 0.0,
            "uses_structure_labels": False,
            "uses_known_structure_count": False,
            "selection": "high residual seeds by component responsibility",
            "sampling": "independent target MALA neighborhoods",
            "training": rows,
        },
    }
    torch.save(result, output_path)
    torch.save(fresh_clouds, args.output / "fresh_local_repair_clouds.pt")
    pd.DataFrame(rows).to_csv(
        args.output / "repair_training_summary.csv", index=False
    )
    summary = {
        "input_mixture": str(args.mixture),
        "output_mixture": str(output_path),
        "base_components": len(models),
        "accepted_repairs": len(repairs),
        "final_components": len(final_models),
        "repair_fraction": args.repair_fraction if repairs else 0.0,
        "weights": final_weights.detach().cpu().tolist(),
        "training": rows,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nFRESH LOCAL REPAIR TRAINING FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"\nMixture: {output_path.resolve()}")


if __name__ == "__main__":
    main()
