#!/usr/bin/env python3
"""SMC-validated sequential specialist repair for the heterogeneous 32D target.

No normalization, target sampler, structure count, or structure labels are
used.  Independent persistent-SMC populations approximate p=f/I.  On those
particles, z=log(f/q) must be constant when q=p, so the scale-free statistic

    log mean(exp(z)) + log mean(exp(-z))

is nonnegative and vanishes only at a match.  A new specialist is trained with
weights proportional to exp(z), which converts p samples into samples from the
second-moment tilt f^2/q up to an irrelevant normalization.  Its mixture mass
is selected on held-out SMC populations and accepted on a fresh external SMC
population.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heterogeneous_residual_specialists_32d as residual  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402
import run_smc_clustered_irregular_100d as smc  # noqa: E402


adaptive = heterogeneous.adaptive
BENCHMARK = heterogeneous.BENCHMARK


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--max-specialists", type=int, default=6)
    parser.add_argument("--train-populations", type=int, default=16)
    parser.add_argument("--validation-populations", type=int, default=12)
    parser.add_argument("--external-populations", type=int, default=12)
    parser.add_argument("--points-per-population", type=int, default=4096)
    parser.add_argument("--smc-target-ess", type=float, default=0.75)
    parser.add_argument("--smc-minimum-q05-ess", type=float, default=0.30)
    parser.add_argument("--smc-mala-steps", type=int, default=6)
    parser.add_argument("--smc-step-size", type=float, default=0.01)
    parser.add_argument("--smc-gradient-clip", type=float, default=50.0)
    parser.add_argument("--smc-chunk", type=int, default=1024)
    parser.add_argument("--marginal-steps", type=int, default=20000)
    parser.add_argument("--copula-steps", type=int, default=20000)
    parser.add_argument("--fit-check-interval", type=int, default=200)
    parser.add_argument("--fit-patience", type=int, default=10)
    parser.add_argument(
        "--mixture-fractions", type=float, nargs="+",
        default=(0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50),
    )
    parser.add_argument("--mismatch-tolerance", type=float, default=0.05)
    parser.add_argument("--minimum-ratio-ess", type=float, default=0.90)
    parser.add_argument("--minimum-score-improvement", type=float, default=0.01)
    parser.add_argument("--top-residual-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("fixed condition must lie in [0,1]^3")
    if min(
        args.max_specialists,
        args.train_populations,
        args.validation_populations,
        args.external_populations,
        args.points_per_population,
        args.smc_mala_steps,
        args.marginal_steps,
        args.copula_steps,
    ) <= 0:
        parser.error("all counts and budgets must be positive")
    if any(not 0 < value < 1 for value in args.mixture_fractions):
        parser.error("mixture fractions must lie strictly between zero and one")
    if sorted(args.mixture_fractions) != list(args.mixture_fractions):
        parser.error("mixture fractions must be increasing")
    if not 0 < args.minimum_ratio_ess <= 1:
        parser.error("--minimum-ratio-ess must lie in (0,1]")
    return args


def configure_smc(args):
    smc.BENCHMARK = BENCHMARK
    smc.bomb.BENCHMARK = BENCHMARK
    smc.INITIAL_BETA = 0.0
    smc.POPULATIONS.clear()
    smc.ARGS = SimpleNamespace(
        smc_target_ess=args.smc_target_ess,
        smc_minimum_q05_ess=args.smc_minimum_q05_ess,
        smc_mala_steps=args.smc_mala_steps,
        smc_step_size=args.smc_step_size,
        smc_gradient_clip=args.smc_gradient_clip,
        smc_chunk=args.smc_chunk,
        smc_forced_increment=0.02,
        smc_initial_beta=0.0,
    )


def build_smc_population(populations, points, device, seed, key):
    # The generic SMC implementation has train/validation persistent slots.
    # Clear the selected slot so every repair round and the external audit are
    # statistically fresh.
    smc.POPULATIONS.pop(key, None)
    label = "mixture train SMC repair" if key == "train" else "mixture val SMC repair"
    cache = smc.persistent_smc_cache(
        BENCHMARK,
        None,
        populations,
        points,
        device,
        seed,
        label,
        target_temperature=1.0,
        uniform_fraction=None,
    )
    cache["logf"] = smc.log_integrand(cache["x"], cache["cond"])
    return cache


def population_slice(cache, start, stop):
    total = cache["x"].shape[0]
    return {
        key: value[start:stop]
        if torch.is_tensor(value) and value.ndim and value.shape[0] == total
        else value
        for key, value in cache.items()
    }


@torch.no_grad()
def ratio_values(model, cache, condition_chunk=4):
    rows = []
    points = cache["x"].shape[1]
    for start in range(0, cache["x"].shape[0], condition_chunk):
        x = cache["x"][start:start + condition_chunk]
        cond = cache["cond"][start:start + condition_chunk]
        groups = x.shape[0]
        logq = model.log_prob(
            x.reshape(groups * points, x.shape[-1]),
            cond.reshape(groups * points, cond.shape[-1]),
        ).reshape(groups, points).double()
        rows.append(cache["logf"][start:start + groups].double() - logq)
    return torch.cat(rows, dim=0)


@torch.no_grad()
def mismatch_summary(model, cache):
    z = ratio_values(model, cache)
    logn = math.log(z.shape[1])
    score = (
        torch.logsumexp(z, dim=1) - logn
        + torch.logsumexp(-z, dim=1) - logn
    )
    weights = torch.softmax(z, dim=1)
    ratio_ess = 1.0 / (
        z.shape[1] * weights.square().sum(dim=1).clamp_min(1e-300)
    )
    return {
        "median_score": float(score.median()),
        "q90_score": float(torch.quantile(score, 0.90)),
        "maximum_score": float(score.max()),
        "median_ratio_ess": float(ratio_ess.median()),
        "q10_ratio_ess": float(torch.quantile(ratio_ess, 0.10)),
        "minimum_ratio_ess": float(ratio_ess.min()),
    }


@torch.no_grad()
def tilted_cache(model, cache):
    output = dict(cache)
    z = ratio_values(model, cache)
    output["alpha"] = torch.softmax(z, dim=1).to(cache["x"].dtype)
    output["residual_log_score"] = z
    return output


@torch.no_grad()
def set_new_component_fraction(model, index, condition, fraction):
    """Set the new component's exact gate mass at the fixed condition."""
    raw = model.gate(condition.reshape(1, -1))[0]
    old_log_total = torch.logsumexp(raw[:index], dim=0)
    desired = old_log_total + math.log(fraction) - math.log1p(-fraction)
    shift = desired - raw[index]
    model.gate[-1].bias[index].add_(shift)


@torch.no_grad()
def select_fraction(model, index, condition, fractions, validation):
    base_bias = model.gate[-1].bias[index].detach().clone()
    rows = []
    best = None
    best_bias = None
    for fraction in fractions:
        model.gate[-1].bias[index].copy_(base_bias)
        set_new_component_fraction(model, index, condition, fraction)
        metrics = mismatch_summary(model, validation)
        row = {"fraction": fraction, **metrics}
        rows.append(row)
        if best is None or metrics["median_score"] < best["median_score"]:
            best = row
            best_bias = model.gate[-1].bias[index].detach().clone()
    model.gate[-1].bias[index].copy_(best_bias)
    return best, rows


def save_checkpoint(path, source, model, graph, history, args):
    checkpoint = {
        **source,
        "benchmark": BENCHMARK,
        "dimension": heterogeneous.DIMENSION,
        "graph": graph,
        "components": model.n_components,
        "model_state_dict": adaptive.cpu_state_dict(model),
        "repair_history": history,
        "args": {
            **dict(source.get("args", {})),
            "max_components": model.max_components,
        },
        "residual_specialist_repair": {
            "condition": list(args.fixed_condition),
            "objective": "scale-free held-out SMC ratio mismatch",
            "specialist_target": "f^2 / q_current",
            "normalization_used": False,
            "training_access": "f(x|c) evaluations and SMC particles only",
            "structure_labels_used": False,
            "exact_target_sampler_used": False,
        },
    }
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    heterogeneous.register_benchmark(tuple(args.fixed_condition))
    adaptive.DIM = heterogeneous.DIMENSION
    adaptive.configure_speed_profile("balanced")
    adaptive.COMPILE_TRAINING_KERNELS = False
    adaptive.MAX_MARGINAL_STEPS = args.marginal_steps
    adaptive.MIN_MARGINAL_STEPS = min(1000, args.marginal_steps)
    adaptive.MARGINAL_CHECK_INTERVAL = args.fit_check_interval
    adaptive.MARGINAL_EARLY_STOP_PATIENCE = args.fit_patience
    adaptive.BRIDGE_COPULA_MAX_STEPS = args.copula_steps
    adaptive.BRIDGE_COPULA_CHECK_INTERVAL = args.fit_check_interval
    adaptive.BRIDGE_COPULA_PATIENCE = args.fit_patience
    adaptive.seed_everything(args.seed)
    configure_smc(args)

    source, model, graph, history = residual.load_checkpoint(
        args.checkpoint, device, args.max_specialists
    )
    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    )
    available = min(
        args.max_specialists, model.max_components - model.n_components
    )

    total_populations = (
        args.train_populations
        + args.validation_populations
        + args.external_populations
    )
    complete_smc = build_smc_population(
        total_populations,
        args.points_per_population,
        device,
        args.seed,
        "train",
    )
    train_stop = args.train_populations
    validation_stop = train_stop + args.validation_populations
    train = population_slice(complete_smc, 0, train_stop)
    validation = population_slice(complete_smc, train_stop, validation_stop)
    external = population_slice(complete_smc, validation_stop, total_populations)
    print(
        f"SMC population split: train={len(train['x'])} "
        f"validation={len(validation['x'])} external={len(external['x'])}",
        flush=True,
    )

    for round_index in range(available):
        current = residual.frozen_copy(model, device)
        before_validation = mismatch_summary(model, validation)
        before_external = mismatch_summary(model, external)
        print(
            f"\nSMC MISMATCH ROUND {round_index}: K={model.n_components} "
            f"validation score={before_validation['median_score']:.6f} "
            f"q90={before_validation['q90_score']:.6f} "
            f"ratioESS={before_validation['median_ratio_ess']:.4f}; "
            f"external score={before_external['median_score']:.6f} "
            f"ratioESS={before_external['median_ratio_ess']:.4f}",
            flush=True,
        )
        no_observed_mismatch = (
            before_validation["q90_score"] <= args.mismatch_tolerance
            and before_external["q90_score"] <= args.mismatch_tolerance
            and before_validation["q10_ratio_ess"] >= args.minimum_ratio_ess
            and before_external["q10_ratio_ess"] >= args.minimum_ratio_ess
        )
        if no_observed_mismatch:
            print(
                "STOPPING: independent SMC populations found no observed mismatch",
                flush=True,
            )
            break

        train_tilt = tilted_cache(current, train)
        validation_tilt = tilted_cache(current, validation)
        residual_info = residual.save_residual_points(
            train_tilt, args.output, round_index, args.top_residual_points
        )
        components_before = model.n_components
        round_state = adaptive.cpu_state_dict(model)
        source_index = residual.repair.residual_source_component(
            model, train_tilt
        )
        new_index = model.spawn_component(source_index, noise_scale=2e-3)
        print(
            f"SPAWNED SMC residual specialist {new_index} "
            f"from component {source_index}",
            flush=True,
        )
        fit = residual.fit_specialist(
            model, new_index, train_tilt, validation_tilt, args
        )
        best_fraction, fraction_rows = select_fraction(
            model, new_index, condition, args.mixture_fractions, validation
        )
        pd.DataFrame(fraction_rows).to_csv(
            args.output / f"round_{round_index:02d}_mixture_scan.csv",
            index=False,
        )
        after_external = mismatch_summary(model, external)
        validation_gain = (
            before_validation["median_score"] - best_fraction["median_score"]
        )
        external_gain = (
            before_external["median_score"] - after_external["median_score"]
        )
        accepted = (
            validation_gain >= args.minimum_score_improvement
            and external_gain >= args.minimum_score_improvement
            and after_external["q90_score"] <= before_external["q90_score"]
        )
        if not accepted:
            adaptive.load_cpu_state(model, round_state)
            while model.n_components > components_before:
                del model.components[-1]

        row = {
            "round": round_index,
            "components_before": components_before,
            "components_after": model.n_components,
            "accepted": bool(accepted),
            "source_component": source_index,
            "specialist_component": new_index,
            "selected_fraction": best_fraction["fraction"],
            "validation_score_before": before_validation["median_score"],
            "validation_score_after": best_fraction["median_score"],
            "external_score_before": before_external["median_score"],
            "external_score_after": after_external["median_score"],
            "external_q90_before": before_external["q90_score"],
            "external_q90_after": after_external["q90_score"],
            "external_ratio_ess_before": before_external["median_ratio_ess"],
            "external_ratio_ess_after": after_external["median_ratio_ess"],
            "maximum_log_residual": residual_info[
                "maximum_log_m2_contribution"
            ],
            "marginal_best_step": fit["marginal"]["best_step"],
            "copula_best_step": fit["copula"]["best_step"],
        }
        history.append(row)
        print(
            f"SMC SPECIALIST {'ACCEPTED' if accepted else 'REJECTED'}: "
            f"fraction={best_fraction['fraction']:.3f}; "
            f"validation {before_validation['median_score']:.6f} -> "
            f"{best_fraction['median_score']:.6f}; external "
            f"{before_external['median_score']:.6f} -> "
            f"{after_external['median_score']:.6f}",
            flush=True,
        )
        pd.DataFrame(history).to_csv(
            args.output / "smc_repair_history.csv", index=False
        )
        save_checkpoint(
            args.output / "latest_smc_specialist_checkpoint.pt",
            source, model, graph, history, args,
        )
        if not accepted:
            print(
                "STOPPING: the new specialist failed independent SMC validation",
                flush=True,
            )
            break
        del train_tilt, validation_tilt, current
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_checkpoint(
        args.output / "smc_residual_specialist_model.pt",
        source, model, graph, history, args,
    )
    with (args.output / "smc_config.json").open("w") as stream:
        json.dump(
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "normalization_used": False,
                "target_sampler_used": False,
                "structure_labels_used": False,
                "stopping_rule": "no mismatch observed on independent SMC populations",
            },
            stream,
            indent=2,
        )
    print("\nSMC-VALIDATED RESIDUAL REPAIR FINISHED")
    print(f"components = {model.n_components}")
    print(
        f"model      = "
        f"{(args.output / 'smc_residual_specialist_model.pt').resolve()}"
    )


if __name__ == "__main__":
    main()
