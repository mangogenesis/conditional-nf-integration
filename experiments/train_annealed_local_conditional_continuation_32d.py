#!/usr/bin/env python3
"""Anneal a learned 32D proposal into a local conditional target.

For every condition radius, freeze the model accepted at the preceding radius
as q_ref and follow the geometric bridge

    pi_beta(x|c) propto q_ref(x|c) ** (1 - beta) * f(x|c) ** beta.

The trainable proposal is refreshed after every accepted beta.  Temperatures
are selected with conditional ESS diagnostics, and training uses only f(x|c)
evaluations.  The exact target sampler and exact integral are never used.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_local_conditional_continuation_32d as local


DIM = local.DIM
COND_DIM = local.COND_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--anchor", type=float, nargs=COND_DIM, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument(
        "--radius-ladder", type=float, nargs="+",
        default=(0.03, 0.035, 0.04, 0.045, 0.05),
    )
    parser.add_argument("--train-conditions", type=int, default=128)
    parser.add_argument("--validation-conditions", type=int, default=64)
    parser.add_argument("--points-per-condition", type=int, default=4096)
    parser.add_argument("--anchor-fraction", type=float, default=0.25)
    parser.add_argument("--uniform-fraction", type=float, default=0.01)
    parser.add_argument("--steps-per-temperature", type=int, default=1200)
    parser.add_argument("--condition-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--anchor-penalty", type=float, default=1e-3)
    parser.add_argument("--worst-fraction", type=float, default=0.20)
    parser.add_argument("--worst-weight", type=float, default=0.50)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)

    parser.add_argument("--temperature-probe-conditions", type=int, default=64)
    parser.add_argument("--temperature-probe-points", type=int, default=2048)
    parser.add_argument("--temperature-min-increment", type=float, default=0.01)
    parser.add_argument("--temperature-max-increment", type=float, default=0.25)
    parser.add_argument("--temperature-bisection-steps", type=int, default=14)
    parser.add_argument(
        "--temperature-target-median-raw-ess", type=float, default=128.0
    )
    parser.add_argument(
        "--temperature-target-q05-raw-ess", type=float, default=20.0
    )
    parser.add_argument(
        "--temperature-hard-median-raw-ess", type=float, default=20.0
    )
    parser.add_argument(
        "--temperature-hard-q05-raw-ess", type=float, default=2.0
    )
    parser.add_argument("--maximum-temperature-stages", type=int, default=100)

    parser.add_argument("--evaluation-conditions", type=int, default=8)
    parser.add_argument("--evaluation-samples", type=int, default=100_000)
    parser.add_argument("--evaluation-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(radius <= 0.0 for radius in args.radius_ladder):
        parser.error("all continuation radii must be positive")
    if sorted(args.radius_ladder) != list(args.radius_ladder):
        parser.error("--radius-ladder must be increasing")
    if any(value < 0.0 or value > 1.0 for value in args.anchor):
        parser.error("anchor must lie in [0,1]^3")
    if not 0.0 < args.anchor_fraction < 1.0:
        parser.error("--anchor-fraction must lie in (0,1)")
    if not 0.0 < args.uniform_fraction < 1.0:
        parser.error("--uniform-fraction must lie in (0,1)")
    if not 0.0 < args.temperature_min_increment <= 1.0:
        parser.error("--temperature-min-increment must lie in (0,1]")
    if not args.temperature_min_increment <= args.temperature_max_increment <= 1.0:
        parser.error("temperature maximum increment must be in [minimum,1]")
    if args.temperature_target_median_raw_ess > args.temperature_probe_points:
        parser.error("temperature median ESS target exceeds probe points")
    if args.temperature_target_q05_raw_ess > args.temperature_probe_points:
        parser.error("temperature q05 ESS target exceeds probe points")

    # local.train_stage expects this historical name.
    args.steps_per_radius = args.steps_per_temperature
    return args


def freeze_copy(model, device):
    result = copy.deepcopy(model).to(device).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


@torch.no_grad()
def bridge_log_target(adaptive, benchmark, reference_model, x, cond, beta):
    log_reference = reference_model.log_prob(x, cond).reshape(-1).double()
    log_f = adaptive.target_log_integrand(benchmark, x, cond).reshape(-1).double()
    return (1.0 - beta) * log_reference + beta * log_f


@torch.no_grad()
def probe_temperature(
    adaptive, benchmark, proposal, reference_model, current_beta, args,
    device, seed,
):
    """Select the largest next beta satisfying preferred conditional ESS."""
    adaptive.seed_everything(seed)
    groups = args.temperature_probe_conditions
    points = args.temperature_probe_points
    anchors = adaptive.sample_conditions(groups, device)
    cond = anchors[:, None, :].expand(groups, points, COND_DIM).reshape(
        groups * points, COND_DIM
    )
    x, log_r = adaptive.sample_from_proposal(
        proposal_model=proposal,
        n=groups * points,
        cond=cond,
        device=device,
        uniform_fraction=args.uniform_fraction,
    )
    log_reference = reference_model.log_prob(x, cond).reshape(groups, points).double()
    log_f = adaptive.target_log_integrand(benchmark, x, cond).reshape(
        groups, points
    ).double()
    log_r = log_r.reshape(groups, points).double()

    def diagnostics(beta):
        log_target = (1.0 - beta) * log_reference + beta * log_f
        alpha = torch.softmax(log_target - log_r, dim=1)
        raw_ess = 1.0 / alpha.square().sum(dim=1)
        return {
            "median": float(raw_ess.median()),
            "q05": float(torch.quantile(raw_ess, 0.05)),
            "minimum": float(raw_ess.min()),
        }

    def preferred(values):
        return (
            values["median"] >= args.temperature_target_median_raw_ess
            and values["q05"] >= args.temperature_target_q05_raw_ess
        )

    upper = min(1.0, current_beta + args.temperature_max_increment)
    upper_diagnostics = diagnostics(upper)
    if preferred(upper_diagnostics):
        chosen = upper
        chosen_diagnostics = upper_diagnostics
    else:
        low = current_beta
        high = upper
        for _ in range(args.temperature_bisection_steps):
            middle = 0.5 * (low + high)
            if preferred(diagnostics(middle)):
                low = middle
            else:
                high = middle
        chosen = low
        chosen_diagnostics = diagnostics(chosen)

    if chosen <= current_beta + 1e-8:
        candidate = min(1.0, current_beta + args.temperature_min_increment)
        candidate_diagnostics = diagnostics(candidate)
        hard_safe = (
            candidate_diagnostics["median"]
            >= args.temperature_hard_median_raw_ess
            and candidate_diagnostics["q05"]
            >= args.temperature_hard_q05_raw_ess
        )
        if not hard_safe:
            return current_beta, candidate_diagnostics, False
        chosen = candidate
        chosen_diagnostics = candidate_diagnostics
        print(
            "TEMPERATURE MICRO STEP: preferred ESS was unavailable; "
            "the hard safety floor passed.",
            flush=True,
        )
    elif chosen < current_beta + args.temperature_min_increment - 1e-8:
        print(
            "TEMPERATURE SUB-MINIMUM STEP: using the largest increment that "
            "meets the preferred ESS targets.",
            flush=True,
        )

    print(
        f"\nLOCAL TEMPERATURE SELECT: {current_beta:.6f} -> {chosen:.6f} | "
        f"median raw ESS={chosen_diagnostics['median']:.1f} | "
        f"q05={chosen_diagnostics['q05']:.1f} | "
        f"min={chosen_diagnostics['minimum']:.1f}",
        flush=True,
    )
    return chosen, chosen_diagnostics, True


@torch.no_grad()
def build_bridge_cache(
    adaptive, benchmark, proposal, reference_model, beta, n_conditions,
    points_per_condition, args, device, seed, label,
):
    adaptive.seed_everything(seed)
    anchors = adaptive.sample_conditions(n_conditions, device)
    cond = anchors[:, None, :].expand(
        n_conditions, points_per_condition, COND_DIM
    ).reshape(n_conditions * points_per_condition, COND_DIM)
    x, log_r = adaptive.sample_from_proposal(
        proposal_model=proposal,
        n=n_conditions * points_per_condition,
        cond=cond,
        device=device,
        uniform_fraction=args.uniform_fraction,
    )
    log_target = bridge_log_target(
        adaptive, benchmark, reference_model, x, cond, beta
    ).reshape(n_conditions, points_per_condition)
    log_r = log_r.reshape(n_conditions, points_per_condition).double()
    alpha = torch.softmax(log_target - log_r, dim=1).float()
    raw_ess = 1.0 / alpha.double().square().sum(dim=1)
    normalized_ess = raw_ess / points_per_condition

    print(f"\nANNEALED CONDITIONAL CACHE: {label}", flush=True)
    print(f"bridge beta            = {beta:.6f}", flush=True)
    print(f"conditions             = {n_conditions:,}", flush=True)
    print(f"points / condition     = {points_per_condition:,}", flush=True)
    print(f"median raw ESS         = {float(raw_ess.median()):.2f}", flush=True)
    print(
        f"q05 raw ESS            = {float(torch.quantile(raw_ess, 0.05)):.2f}",
        flush=True,
    )
    print(f"minimum raw ESS        = {float(raw_ess.min()):.2f}", flush=True)

    return {
        "x": x.reshape(n_conditions, points_per_condition, DIM).detach(),
        "cond": cond.reshape(
            n_conditions, points_per_condition, COND_DIM
        ).detach(),
        "alpha": alpha.detach(),
        "mean_conditional_ess": float(normalized_ess.mean()),
        "mean_raw_ess": float(raw_ess.mean()),
        "median_raw_ess": float(raw_ess.median()),
        "q05_raw_ess": float(torch.quantile(raw_ess, 0.05)),
        "min_raw_ess": float(raw_ess.min()),
    }


def save_checkpoint(
    path, source_checkpoint, model, args, radius, beta, radius_results,
):
    checkpoint = {
        **source_checkpoint,
        "model_state_dict": local.adaptive_cpu_state(model),
        "annealed_conditional_continuation": {
            "anchor": list(args.anchor),
            "radius": radius,
            "beta": beta,
            "radius_ladder": list(args.radius_ladder),
            "bridge": "q_reference^(1-beta) * f(x|c)^beta",
            "training_access": "f(x|c) only",
            "exact_target_sampler_used": False,
            "temperature_results": radius_results,
        },
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    adaptive, model, source_checkpoint, benchmark = local.load_model(
        args.checkpoint, device
    )
    model.to(device)
    global_anchor_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    temperature_rows = []
    radius_rows = []
    evaluation_rows = []

    for radius_index, radius in enumerate(args.radius_ladder):
        adaptive.sample_conditions = local.make_condition_sampler(
            args.anchor, radius, args.anchor_fraction
        )
        reference_model = freeze_copy(model, device)
        current_beta = 0.0
        radius_results = []
        radius_dir = args.output / f"radius_{radius:.6g}"
        radius_dir.mkdir(exist_ok=True)
        reached_target = False

        print(
            f"\n{'=' * 88}\nANNEALED LOCAL CONDITION RADIUS {radius:.6g}\n"
            f"pi_beta propto q_reference^(1-beta) f^beta\n{'=' * 88}",
            flush=True,
        )
        for temperature_stage in range(args.maximum_temperature_stages):
            proposal = freeze_copy(model, device)
            next_beta, probe, can_advance = probe_temperature(
                adaptive, benchmark, proposal, reference_model, current_beta,
                args, device,
                args.seed + 100_000 * radius_index + 1000 * temperature_stage,
            )
            if not can_advance:
                print(
                    "STOPPING RADIUS: no positive temperature increment "
                    "passed the hard conditional ESS floor.",
                    flush=True,
                )
                break

            train_cache = build_bridge_cache(
                adaptive, benchmark, proposal, reference_model, next_beta,
                args.train_conditions, args.points_per_condition, args, device,
                args.seed + 100_000 * radius_index + 1000 * temperature_stage + 1,
                f"train radius={radius:.6g} beta={next_beta:.6f}",
            )
            validation_cache = build_bridge_cache(
                adaptive, benchmark, proposal, reference_model, next_beta,
                args.validation_conditions, args.points_per_condition, args,
                device,
                args.seed + 100_000 * radius_index + 1000 * temperature_stage + 2,
                f"validation radius={radius:.6g} beta={next_beta:.6f}",
            )
            hard_safe = (
                validation_cache["median_raw_ess"]
                >= args.temperature_hard_median_raw_ess
                and validation_cache["q05_raw_ess"]
                >= args.temperature_hard_q05_raw_ess
            )
            if not hard_safe:
                print(
                    "STOPPING RADIUS: independent validation cache failed "
                    "the hard ESS floor.",
                    flush=True,
                )
                break

            result = local.train_stage(
                model, train_cache, validation_cache,
                global_anchor_parameters, args,
                label=f"radius={radius:.6g} beta={next_beta:.6f}",
            )
            stage_record = {
                "radius": radius,
                "temperature_stage": temperature_stage,
                "beta_start": current_beta,
                "beta_end": next_beta,
                "probe_median_raw_ess": probe["median"],
                "probe_q05_raw_ess": probe["q05"],
                "train_median_raw_ess": train_cache["median_raw_ess"],
                "train_q05_raw_ess": train_cache["q05_raw_ess"],
                "validation_median_raw_ess": validation_cache["median_raw_ess"],
                "validation_q05_raw_ess": validation_cache["q05_raw_ess"],
                "best_step": result["best_step"],
                "initial_validation_score": result["initial"]["score"],
                "best_validation_score": result["best"]["score"],
            }
            temperature_rows.append(stage_record)
            radius_results.append({**stage_record, "training": result})
            pd.DataFrame(temperature_rows).to_csv(
                args.output / "temperature_summary.csv", index=False
            )

            beta_dir = radius_dir / f"beta_{next_beta:.6f}"
            beta_dir.mkdir(exist_ok=True)
            pd.DataFrame(result["history"]).to_csv(
                beta_dir / "training_history.csv", index=False
            )
            save_checkpoint(
                beta_dir / "conditional_model.pt", source_checkpoint, model,
                args, radius, next_beta, radius_results,
            )
            current_beta = next_beta
            if current_beta >= 1.0 - 1e-12:
                reached_target = True
                break

        save_checkpoint(
            radius_dir / "conditional_model.pt", source_checkpoint, model,
            args, radius, current_beta, radius_results,
        )
        radius_rows.append({
            "radius": radius,
            "reached_beta_one": reached_target,
            "final_beta": current_beta,
            "temperature_stages": len(radius_results),
        })
        pd.DataFrame(radius_rows).to_csv(
            args.output / "radius_summary.csv", index=False
        )
        if not reached_target:
            print(
                f"RADIUS {radius:.6g} WAS NOT ACCEPTED: bridge stopped at "
                f"beta={current_beta:.6f}.",
                flush=True,
            )
            break

        conditions = np.vstack([
            np.asarray(args.anchor, dtype=float),
            local.evaluation_conditions(
                args.anchor, radius, args.evaluation_conditions,
                args.seed + 10_000 + radius_index,
            ),
        ])
        for condition_index, condition in enumerate(conditions):
            metrics = local.evaluate_condition(
                adaptive, model, benchmark, condition, args, device,
                args.seed + 1_000_000 * radius_index + condition_index,
            )
            row = {
                "radius": radius,
                "condition_index": condition_index,
                "is_anchor": condition_index == 0,
                "c0": condition[0], "c1": condition[1], "c2": condition[2],
                **metrics,
            }
            evaluation_rows.append(row)
            print(
                f"radius={radius:.4g} condition={condition_index:02d} "
                f"anchor={condition_index == 0} I={metrics['estimate']:.7f} "
                f"M2={metrics['second_moment']:.5g} "
                f"ESS={metrics['normalized_ess']:.4f} "
                f"xi={metrics['hill_xi_005']:.3f}"
                + (
                    f" target-p1={metrics['target_branch_1_mass']:.3f} "
                    f"NF-p1={metrics['proposal_branch_1_mass']:.3f} "
                    f"gap={metrics['proposal_gap_mass']:.2e}"
                    if "target_branch_1_mass" in metrics else ""
                ),
                flush=True,
            )
        pd.DataFrame(evaluation_rows).to_csv(
            args.output / "evaluation.csv", index=False
        )

    with (args.output / "config.json").open("w") as stream:
        serializable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "steps_per_radius"
        }
        json.dump({
            **serializable,
            "bridge": "q_reference^(1-beta) * f(x|c)^beta",
            "training_access": "f(x|c) only",
            "exact_target_sampler_used": False,
        }, stream, indent=2)
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
