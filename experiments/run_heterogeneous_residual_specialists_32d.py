#!/usr/bin/env python3
"""Iteratively repair a frozen heterogeneous proposal with residual specialists.

The procedure is deliberately blind to the benchmark's four structure labels.
For a current normalized proposal q and target integrand f, each new specialist
is fitted to the second-moment tilt

    h_2(x | c) proportional to f(x | c)^2 / q(x | c).

This density emphasizes places where q undercovers f.  After fitting one new
component, its mixture gate and component parameters are refined against held-
out empirical M2.  A separate external cache accepts or rejects the complete
repair round.  Previous components are frozen during specialist fitting and a
small uniform share is used for discovery/auditing, so no target sampler or
structure labels are required.
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

import run_adaptive_mixture_irregular_100d as repair  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402
from nf.conditional_mixture import ConditionalMixtureProposal  # noqa: E402


adaptive = heterogeneous.adaptive
BENCHMARK = heterogeneous.BENCHMARK


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--max-specialists", type=int, default=6)
    parser.add_argument(
        "--minimum-specialists", type=int, default=1,
        help="Do not apply the ESS stopping rule before testing this many specialists.",
    )
    parser.add_argument("--train-populations", type=int, default=32)
    parser.add_argument("--validation-populations", type=int, default=16)
    parser.add_argument("--external-populations", type=int, default=24)
    parser.add_argument("--points-per-population", type=int, default=8192)
    parser.add_argument("--uniform-fraction", type=float, default=0.02)
    parser.add_argument("--marginal-steps", type=int, default=4000)
    parser.add_argument("--copula-steps", type=int, default=4000)
    parser.add_argument("--fit-check-interval", type=int, default=200)
    parser.add_argument("--fit-patience", type=int, default=10)
    parser.add_argument("--m2-steps", type=int, default=4000)
    parser.add_argument("--m2-learning-rate", type=float, default=5e-5)
    parser.add_argument("--m2-patience", type=int, default=10)
    parser.add_argument("--initial-specialist-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-log-m2-improvement", type=float, default=0.01)
    parser.add_argument("--q90-log-m2-tolerance", type=float, default=0.02)
    parser.add_argument("--target-median-ess", type=float, default=0.90)
    parser.add_argument("--target-q10-ess", type=float, default=0.20)
    parser.add_argument("--top-residual-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("fixed condition must lie in [0,1]^3")
    if args.max_specialists < 1:
        parser.error("--max-specialists must be positive")
    if not 0 <= args.minimum_specialists <= args.max_specialists:
        parser.error(
            "--minimum-specialists must lie between zero and --max-specialists"
        )
    if min(
        args.train_populations,
        args.validation_populations,
        args.external_populations,
        args.points_per_population,
        args.marginal_steps,
        args.copula_steps,
        args.m2_steps,
    ) <= 0:
        parser.error("all budgets must be positive")
    if not 0 < args.uniform_fraction < 1:
        parser.error("--uniform-fraction must lie in (0,1)")
    if not 0 < args.initial_specialist_fraction < 1:
        parser.error("--initial-specialist-fraction must lie in (0,1)")
    if not 0 < args.target_median_ess <= 1:
        parser.error("--target-median-ess must lie in (0,1]")
    if not 0 < args.target_q10_ess <= 1:
        parser.error("--target-q10-ess must lie in (0,1]")
    return args


def load_checkpoint(path, device, max_specialists):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("benchmark") != BENCHMARK:
        raise ValueError(
            f"expected benchmark {BENCHMARK!r}, "
            f"found {checkpoint.get('benchmark')!r}"
        )
    graph = checkpoint["graph"]
    count = int(checkpoint.get("components", 1))
    if count == 1 and "residual_specialist_repair" not in checkpoint:
        base = adaptive.make_real_model(graph, None, device)
        base.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = ConditionalMixtureProposal(
            [base], cond_dim=adaptive.COND_DIM,
            max_components=1 + max_specialists,
        ).to(device)
        history = []
    else:
        state = checkpoint["model_state_dict"]
        gate_capacity = int(state["gate.4.bias"].numel())
        components = [
            adaptive.make_real_model(graph, None, device) for _ in range(count)
        ]
        model = ConditionalMixtureProposal(
            components, cond_dim=adaptive.COND_DIM,
            max_components=gate_capacity,
        ).to(device)
        model.load_state_dict(state, strict=True)
        history = list(checkpoint.get("repair_history", []))
    return checkpoint, model, graph, history


@torch.no_grad()
def frozen_copy(model, device):
    result = copy.deepcopy(model).to(device).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def build_cache(model, populations, points, args, device, seed, label):
    cache = adaptive.build_weighted_cache(
        BENCHMARK,
        model,
        populations,
        points,
        device,
        seed,
        label,
        1.0,
        args.uniform_fraction,
    )
    repair.attach_log_densities(
        cache, model, 1.0, args.uniform_fraction
    )
    return cache


@torch.no_grad()
def second_moment_tilt(model, cache):
    """Replace cache weights by normalized h2/r coefficients."""
    output = dict(cache)
    groups, points, dimension = cache["x"].shape
    logq = model.log_prob(
        cache["x"].reshape(groups * points, dimension),
        cache["cond"].reshape(groups * points, -1),
    ).reshape(groups, points)
    log_score = (
        2.0 * cache["log_target"].double()
        - logq.double()
        - cache["log_source"].double()
    )
    output["alpha"] = torch.softmax(log_score, dim=1).to(cache["x"].dtype)
    output["residual_log_score"] = log_score
    return output


@torch.no_grad()
def normalized_log_m2_values(model, cache, condition_chunk=8):
    """Return log(M2/I^2), whose negative exponential is normalized ESS."""
    values = []
    points = cache["x"].shape[1]
    log_points = math.log(points)
    for start in range(0, cache["x"].shape[0], condition_chunk):
        x = cache["x"][start:start + condition_chunk]
        cond = cache["cond"][start:start + condition_chunk]
        groups, _, dimension = x.shape
        logq = repair.defensive_log_prob(
            model,
            x.reshape(groups * points, dimension),
            cond.reshape(groups * points, -1),
            cache["uniform_fraction"],
        ).reshape(groups, points).double()
        logf = cache["log_target"][start:start + groups].double()
        logr = cache["log_source"][start:start + groups].double()
        log_i = torch.logsumexp(logf - logr, dim=1) - log_points
        log_m2 = torch.logsumexp(2.0 * logf - logq - logr, dim=1) - log_points
        values.append(log_m2 - 2.0 * log_i)
    return torch.cat(values)


@torch.no_grad()
def summarize_normalized_m2(model, cache):
    values = normalized_log_m2_values(model, cache)
    points = cache["x"].shape[1]
    log_i = torch.logsumexp(
        (cache["log_target"] - cache["log_source"]).double(), dim=1
    ) - math.log(points)
    integral = torch.exp(log_i)
    return {
        "median": float(values.median()),
        "upper_quantile": float(torch.quantile(values, 0.90)),
        "mean": float(values.mean()),
        "maximum": float(values.max()),
        "median_ess": float(torch.exp(-values.median())),
        "q10_ess": float(torch.exp(-torch.quantile(values, 0.90))),
        "mean_integral": float(integral.mean()),
        "minimum_integral": float(integral.min()),
        "maximum_integral": float(integral.max()),
    }


@torch.no_grad()
def save_residual_points(cache, output, round_index, keep):
    """Record the largest observed M2 contributions without target labels."""
    score = cache["residual_log_score"].reshape(-1)
    count = min(int(keep), score.numel())
    values, indices = torch.topk(score, count)
    x = cache["x"].reshape(-1, cache["x"].shape[-1])[indices]
    cond = cache["cond"].reshape(-1, cache["cond"].shape[-1])[indices]
    frame = pd.DataFrame(
        x.detach().cpu().numpy(), columns=[f"x{j}" for j in range(x.shape[1])]
    )
    for j in range(cond.shape[1]):
        frame[f"c{j}"] = cond[:, j].detach().cpu().numpy()
    frame["log_m2_contribution"] = values.detach().cpu().numpy()
    frame.to_csv(output / f"round_{round_index:02d}_residual_points.csv", index=False)
    weighted = torch.softmax(values.double(), dim=0)
    center = (weighted[:, None] * x.double()).sum(dim=0)
    spread = torch.sqrt(
        (weighted[:, None] * (x.double() - center).square()).sum(dim=0)
    )
    return {
        "maximum_log_m2_contribution": float(values[0]),
        "residual_center": center.cpu().tolist(),
        "residual_spread": spread.cpu().tolist(),
    }


def fit_specialist(model, index, train_cache, validation_cache, args):
    specialist = model.components[index]
    bank = adaptive.get_marginal_bank(specialist)
    marginal = adaptive.train_marginals_on_dataset(
        bank,
        train_cache,
        validation_cache,
        train_cache["x"].device,
        args.seed + 1000 * (index + 1),
    )
    copula = adaptive.train_bridge_copula_on_cache(
        specialist,
        train_cache,
        validation_cache,
        label=f"residual specialist {index}",
        optimize_joint=False,
    )
    return {"marginal": marginal, "copula": copula}


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
            "objective": "held-out empirical second moment",
            "specialist_target": "f^2 / q_current",
            "training_access": "f(x|c) and proposal samples only",
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
    # Reused generic repair helpers consult their module-level benchmark name
    # when attaching target log densities.  Redirect that name explicitly to
    # the heterogeneous target registered above.
    repair.BENCHMARK = BENCHMARK
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

    source, model, graph, history = load_checkpoint(
        args.checkpoint, device, args.max_specialists
    )
    rejected_rounds = 0

    available_slots = model.max_components - model.n_components
    repair_rounds = min(args.max_specialists, available_slots)
    if repair_rounds < args.max_specialists:
        print(
            f"Specialist count limited to {repair_rounds} by checkpoint gate "
            f"capacity {model.max_components}",
            flush=True,
        )

    for round_index in range(repair_rounds):
        components_before = model.n_components
        current = frozen_copy(model, device)
        train = build_cache(
            current, args.train_populations, args.points_per_population,
            args, device, args.seed + 10_000 * round_index,
            f"residual train round={round_index}",
        )
        validation = build_cache(
            current, args.validation_populations, args.points_per_population,
            args, device, args.seed + 10_000 * round_index + 1,
            f"residual validation round={round_index}",
        )
        external = build_cache(
            current, args.external_populations, args.points_per_population,
            args, device, args.seed + 10_000 * round_index + 2,
            f"external acceptance round={round_index}",
        )
        before = summarize_normalized_m2(model, external)
        median_ess = before["median_ess"]
        print(
            f"\nRESIDUAL REPAIR ROUND {round_index}: K={model.n_components} "
            f"median logM2={before['median']:.6f} "
            f"q90={before['upper_quantile']:.6f} "
            f"median ESS={median_ess:.4f} q10 ESS={before['q10_ess']:.4f} "
            f"mean I={before['mean_integral']:.6f} "
            f"range=[{before['minimum_integral']:.6f}, "
            f"{before['maximum_integral']:.6f}]",
            flush=True,
        )
        if (
            round_index >= args.minimum_specialists
            and
            median_ess >= args.target_median_ess
            and before["q10_ess"] >= args.target_q10_ess
        ):
            print("STOPPING: target median and q10 ESS have been reached", flush=True)
            break

        train_tilt = second_moment_tilt(current, train)
        validation_tilt = second_moment_tilt(current, validation)
        residual = save_residual_points(
            train_tilt, args.output, round_index, args.top_residual_points
        )
        validation_baseline = float(
            repair.log_m2_diagnostic(model, validation).mean().item()
        )
        round_state = adaptive.cpu_state_dict(model)
        source_index = repair.residual_source_component(model, train_tilt)
        new_index = model.spawn_component(source_index, noise_scale=2e-3)
        repair.initialize_specialist_gate(
            model, new_index, source_index,
            relative_mass=args.initial_specialist_fraction,
        )
        print(
            f"SPAWNED specialist {new_index} from component {source_index}; "
            "fitting the observed f^2/q residual",
            flush=True,
        )
        fit = fit_specialist(
            model, new_index, train_tilt, validation_tilt, args
        )
        m2_result = repair.finetune_cluster_second_moment(
            model,
            new_index,
            train,
            validation,
            steps=args.m2_steps,
            lr=args.m2_learning_rate,
            minimum_improvement=args.minimum_log_m2_improvement,
            acceptance_baseline=validation_baseline,
            rejection_state=round_state,
            patience_limit=args.m2_patience,
            condition_cvar_fraction=1.0,
        )

        after = summarize_normalized_m2(model, external)
        median_gain = before["median"] - after["median"]
        q90_change = after["upper_quantile"] - before["upper_quantile"]
        accepted = (
            m2_result["accepted"]
            and median_gain >= args.minimum_log_m2_improvement
            and q90_change <= args.q90_log_m2_tolerance
        )
        if not accepted:
            adaptive.load_cpu_state(model, round_state)
            while model.n_components > components_before:
                del model.components[-1]
            rejected_rounds += 1
        else:
            rejected_rounds = 0
        row = {
            "round": round_index,
            "components_before": components_before,
            "components_after": model.n_components,
            "source_component": source_index,
            "specialist_component": new_index,
            "accepted": bool(accepted),
            "median_log_m2_before": before["median"],
            "median_log_m2_after": after["median"],
            "q90_log_m2_before": before["upper_quantile"],
            "q90_log_m2_after": after["upper_quantile"],
            "median_ess_before": math.exp(-before["median"]),
            "median_ess_after": math.exp(-after["median"]),
            "q10_ess_before": before["q10_ess"],
            "q10_ess_after": after["q10_ess"],
            "mean_integral_before": before["mean_integral"],
            "mean_integral_after": after["mean_integral"],
            "maximum_log_m2_contribution": residual[
                "maximum_log_m2_contribution"
            ],
            "marginal_best_step": fit["marginal"]["best_step"],
            "copula_best_step": fit["copula"]["best_step"],
        }
        history.append(row)
        print(
            f"EXTERNAL REPAIR {'ACCEPTED' if accepted else 'REJECTED'}: "
            f"median logM2 {before['median']:.6f} -> {after['median']:.6f}; "
            f"q90 {before['upper_quantile']:.6f} -> "
            f"{after['upper_quantile']:.6f}",
            flush=True,
        )
        pd.DataFrame(history).to_csv(
            args.output / "repair_history.csv", index=False
        )
        save_checkpoint(
            args.output / "latest_residual_specialist_checkpoint.pt",
            source, model, graph, history, args,
        )
        if rejected_rounds >= 1:
            print(
                "STOPPING: the independently audited specialist was rejected",
                flush=True,
            )
            break
        del train, validation, external, train_tilt, validation_tilt, current
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_checkpoint(
        args.output / "residual_specialist_model.pt",
        source, model, graph, history, args,
    )
    with (args.output / "config.json").open("w") as stream:
        json.dump(
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "training_access": "f(x|c) and proposal samples only",
                "structure_labels_used": False,
                "exact_target_sampler_used": False,
            },
            stream,
            indent=2,
        )
    print("\nADAPTIVE RESIDUAL-SPECIALIST REPAIR FINISHED")
    print(f"components = {model.n_components}")
    print(
        f"model      = "
        f"{(args.output / 'residual_specialist_model.pt').resolve()}"
    )


if __name__ == "__main__":
    main()
