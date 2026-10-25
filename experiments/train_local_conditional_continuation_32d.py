#!/usr/bin/env python3
"""Continue a fixed-condition 32D NF into a local conditional model.

Training uses only evaluations of f(x|c).  The fixed-condition checkpoint is
the proposal and initialization.  A radius ladder expands a Euclidean ball in
condition space while anchor conditions, grouped validation, and a worst-group
loss reduce catastrophic forgetting and average-condition masking.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_geometry_ladder_32d as geometry  # noqa: E402
import run_topology_ladder_32d as topology  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402


DIM = 32
COND_DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--anchor", type=float, nargs=COND_DIM, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument(
        "--radius-ladder", type=float, nargs="+",
        default=(0.01, 0.025, 0.05, 0.10),
    )
    parser.add_argument("--train-conditions", type=int, default=128)
    parser.add_argument("--validation-conditions", type=int, default=64)
    parser.add_argument("--points-per-condition", type=int, default=2048)
    parser.add_argument("--anchor-fraction", type=float, default=0.25)
    parser.add_argument("--uniform-fraction", type=float, default=0.01)
    parser.add_argument("--steps-per-radius", type=int, default=2000)
    parser.add_argument("--condition-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--anchor-penalty", type=float, default=1e-3)
    parser.add_argument("--worst-fraction", type=float, default=0.20)
    parser.add_argument("--worst-weight", type=float, default=0.50)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument(
        "--minimum-validation-median-raw-ess", type=float, default=20.0,
        help="Stop before training a radius whose proposal cache is too weak.",
    )
    parser.add_argument(
        "--stop-on-no-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not expand farther when a radius cannot improve validation.",
    )
    parser.add_argument("--evaluation-conditions", type=int, default=8)
    parser.add_argument("--evaluation-samples", type=int, default=100_000)
    parser.add_argument("--evaluation-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261013)
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
    if not 0.0 < args.worst_fraction <= 1.0:
        parser.error("--worst-fraction must lie in (0,1]")
    return args


def load_model(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    benchmark = checkpoint.get("benchmark")
    geometry_benchmarks = {
        geometry.AUTOREGRESSIVE_BROAD,
        geometry.AUTOREGRESSIVE_SHARP,
    }
    if benchmark == heterogeneous.BENCHMARK:
        heterogeneous.register_benchmark()
        adaptive = heterogeneous.adaptive
    elif benchmark in topology.BENCHMARKS:
        topology.register_benchmarks()
        adaptive = topology.adaptive
    elif benchmark in geometry_benchmarks:
        geometry.register_benchmarks()
        adaptive = geometry.adaptive
    else:
        raise ValueError(
            "checkpoint must be a supported geometry or topology 32D "
            "benchmark"
        )
    adaptive.DIM = DIM
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return adaptive, model, checkpoint, benchmark


def make_condition_sampler(anchor, radius: float, anchor_fraction: float):
    anchor = tuple(float(value) for value in anchor)

    def sample_conditions(n: int, device: torch.device):
        count_anchor = min(n, int(round(anchor_fraction * n)))
        count_local = n - count_anchor
        center = torch.tensor(anchor, device=device, dtype=torch.float32)
        pieces = []
        if count_anchor:
            pieces.append(center.expand(count_anchor, COND_DIM).clone())
        if count_local:
            direction = torch.randn(count_local, COND_DIM, device=device)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
            radial = torch.rand(count_local, 1, device=device).pow(1.0 / COND_DIM)
            local = center + radius * radial * direction
            pieces.append(local.clamp(0.0, 1.0))
        result = torch.cat(pieces, dim=0)
        return result[torch.randperm(n, device=device)]

    return sample_conditions


def grouped_losses(model, cache, indices=None):
    if indices is None:
        x, cond, alpha = cache["x"], cache["cond"], cache["alpha"]
    else:
        x, cond, alpha = (
            cache["x"][indices], cache["cond"][indices], cache["alpha"][indices]
        )
    groups, points, _ = x.shape
    logq = model.log_prob(
        x.reshape(groups * points, DIM),
        cond.reshape(groups * points, COND_DIM),
    ).reshape(groups, points)
    return -(alpha * logq).sum(dim=1)


def robust_objective(losses, worst_fraction: float, worst_weight: float):
    count = max(1, int(math.ceil(worst_fraction * len(losses))))
    worst = torch.topk(losses, count).values.mean()
    return losses.mean() + worst_weight * worst


@torch.no_grad()
def validation_score(model, cache, worst_fraction, worst_weight):
    model.eval()
    values = []
    for start in range(0, len(cache["x"]), 8):
        values.append(grouped_losses(
            model, cache, torch.arange(
                start, min(start + 8, len(cache["x"])),
                device=cache["x"].device,
            )
        ).double().cpu())
    losses = torch.cat(values)
    count = max(1, int(math.ceil(worst_fraction * len(losses))))
    return {
        "mean_nll": float(losses.mean()),
        "worst_nll": float(torch.topk(losses, count).values.mean()),
        "score": float(losses.mean() + worst_weight * torch.topk(
            losses, count
        ).values.mean()),
    }


def parameter_anchor(model, initial_parameters):
    total = torch.zeros((), device=next(model.parameters()).device)
    count = 0
    for parameter, initial in zip(model.parameters(), initial_parameters):
        total = total + (parameter - initial).square().sum()
        count += parameter.numel()
    return total / max(count, 1)


def train_stage(model, train_cache, validation_cache, base_parameters, args, label):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial = validation_score(
        model, validation_cache, args.worst_fraction, args.worst_weight
    )
    best = initial
    best_state = adaptive_cpu_state(model)
    best_step = 0
    patience = 0
    history = []
    n_conditions = len(train_cache["x"])

    print(
        f"\nLOCAL CONDITIONAL STAGE {label}: initial mean={initial['mean_nll']:.6f} "
        f"worst={initial['worst_nll']:.6f} score={initial['score']:.6f}",
        flush=True,
    )
    for start in range(0, args.steps_per_radius, args.check_interval):
        block = min(args.check_interval, args.steps_per_radius - start)
        train_values = []
        model.train()
        for _ in range(block):
            indices = torch.randint(
                0, n_conditions, (args.condition_batch,),
                device=train_cache["x"].device,
            )
            losses = grouped_losses(model, train_cache, indices)
            loss = robust_objective(
                losses, args.worst_fraction, args.worst_weight
            )
            loss = loss + args.anchor_penalty * parameter_anchor(
                model, base_parameters
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_values.append(float(loss.detach()))

        step = start + block
        current = validation_score(
            model, validation_cache, args.worst_fraction, args.worst_weight
        )
        improved = current["score"] < (
            best["score"]
            - args.minimum_improvement * max(abs(best["score"]), 1.0)
        )
        if improved:
            best = current
            best_state = adaptive_cpu_state(model)
            best_step = step
            patience = 0
        else:
            patience += 1
        row = {
            "step": step,
            "train_loss": float(np.mean(train_values)),
            **{f"validation_{key}": value for key, value in current.items()},
            "best_score": best["score"],
        }
        history.append(row)
        print(
            f"step {step:5d}/{args.steps_per_radius}: "
            f"train={row['train_loss']:.6f} mean={current['mean_nll']:.6f} "
            f"worst={current['worst_nll']:.6f} best={best['score']:.6f} "
            f"patience={patience}/{args.patience}",
            flush=True,
        )
        if patience >= args.patience:
            break

    model.load_state_dict(best_state, strict=True)
    return {
        "initial": initial,
        "best": best,
        "best_step": best_step,
        "history": history,
    }


def adaptive_cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def hill_xi(logw: np.ndarray, fraction=0.005):
    ordered = np.sort(logw)[::-1]
    k = min(max(2, int(math.ceil(fraction * len(ordered)))), len(ordered) - 1)
    return float(np.mean(ordered[:k] - ordered[k]))


@torch.no_grad()
def evaluate_condition(adaptive, model, benchmark, condition, args, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    pieces = []
    branch_0_count = 0
    branch_1_count = 0
    gap_count = 0
    center = torch.tensor(condition, device=device, dtype=torch.float32)
    remaining = args.evaluation_samples
    while remaining:
        count = min(remaining, args.evaluation_chunk)
        cond = center.expand(count, COND_DIM)
        x = adaptive.sample_from_model(model, count, cond)
        logq = model.log_prob(x, cond).reshape(-1)
        logf = adaptive.target_log_integrand(benchmark, x, cond).reshape(-1)
        pieces.append((logf - logq).double().cpu().numpy())
        if benchmark in topology.BENCHMARKS:
            x0 = x[:, 0]
            branch_0_count += int((x0 <= topology.BRANCH_WIDTH).sum())
            branch_1_count += int((x0 >= 1.0 - topology.BRANCH_WIDTH).sum())
            gap_count += int((
                (x0 > topology.BRANCH_WIDTH)
                & (x0 < 1.0 - topology.BRANCH_WIDTH)
            ).sum())
        remaining -= count
    logw = np.concatenate(pieces)
    log_sum = float(logsumexp(logw))
    log_sum2 = float(logsumexp(2.0 * logw))
    n = len(logw)
    estimate = math.exp(log_sum - math.log(n))
    m2 = math.exp(log_sum2 - math.log(n))
    result = {
        "estimate": estimate,
        "second_moment": m2,
        "normalized_ess": estimate * estimate / m2,
        "max_weight_share": math.exp(float(logw.max()) - log_sum),
        "hill_xi_005": hill_xi(logw),
    }
    if benchmark in topology.BENCHMARKS:
        condition_tensor = center.reshape(1, COND_DIM)
        target_branch_1 = float(
            topology.branch_probabilities(benchmark, condition_tensor)[0, 1]
        )
        proposal_branch_1 = branch_1_count / n
        result.update({
            "target_branch_1_mass": target_branch_1,
            "proposal_branch_0_mass": branch_0_count / n,
            "proposal_branch_1_mass": proposal_branch_1,
            "proposal_gap_mass": gap_count / n,
            "absolute_branch_1_mass_error": abs(
                proposal_branch_1 - target_branch_1
            ),
        })
    return result


def evaluation_conditions(anchor, radius, count, seed):
    generator = np.random.default_rng(seed)
    direction = generator.normal(size=(count, COND_DIM))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    local = np.asarray(anchor)[None, :] + radius * direction
    return np.clip(local, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    adaptive, model, source_checkpoint, benchmark = load_model(
        args.checkpoint, device
    )
    model.to(device)
    base_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    stage_summaries = []
    evaluation_rows = []

    for stage_index, radius in enumerate(args.radius_ladder):
        adaptive.sample_conditions = make_condition_sampler(
            args.anchor, radius, args.anchor_fraction
        )
        proposal = copy.deepcopy(model).to(device).eval()
        for parameter in proposal.parameters():
            parameter.requires_grad_(False)
        train_cache = adaptive.build_weighted_cache(
            benchmark=benchmark,
            proposal_model=proposal,
            n_conditions=args.train_conditions,
            points_per_condition=args.points_per_condition,
            device=device,
            seed=args.seed + 1000 * stage_index,
            label=f"local continuation train radius={radius:.6g}",
            target_temperature=1.0,
            uniform_fraction=args.uniform_fraction,
        )
        validation_cache = adaptive.build_weighted_cache(
            benchmark=benchmark,
            proposal_model=proposal,
            n_conditions=args.validation_conditions,
            points_per_condition=args.points_per_condition,
            device=device,
            seed=args.seed + 1000 * stage_index + 1,
            label=f"local continuation validation radius={radius:.6g}",
            target_temperature=1.0,
            uniform_fraction=args.uniform_fraction,
        )
        if (
            validation_cache["median_raw_ess"]
            < args.minimum_validation_median_raw_ess
        ):
            stage_summaries.append({
                "stage": stage_index,
                "radius": radius,
                "accepted": False,
                "stop_reason": "validation proposal ESS below safety floor",
                "validation_median_raw_ess": validation_cache["median_raw_ess"],
                "validation_min_raw_ess": validation_cache["min_raw_ess"],
            })
            pd.DataFrame(stage_summaries).to_csv(
                args.output / "stage_summary.csv", index=False
            )
            print(
                "STOPPING RADIUS EXPANSION: validation median raw ESS "
                f"{validation_cache['median_raw_ess']:.2f} is below "
                f"{args.minimum_validation_median_raw_ess:.2f}",
                flush=True,
            )
            break
        result = train_stage(
            model, train_cache, validation_cache, base_parameters, args,
            label=f"radius={radius:.6g}",
        )

        stage_dir = args.output / f"radius_{radius:.6g}"
        stage_dir.mkdir(exist_ok=True)
        checkpoint = {
            **source_checkpoint,
            "model_state_dict": adaptive_cpu_state(model),
            "conditional_continuation": {
                "anchor": list(args.anchor),
                "radius": radius,
                "radius_ladder": list(args.radius_ladder),
                "training_access": "f(x|c) only",
                "exact_target_sampler_used": False,
                "stage_result": result,
            },
        }
        torch.save(checkpoint, stage_dir / "conditional_model.pt")
        pd.DataFrame(result["history"]).to_csv(
            stage_dir / "training_history.csv", index=False
        )

        conditions = np.vstack([
            np.asarray(args.anchor, dtype=float),
            evaluation_conditions(
                args.anchor, radius, args.evaluation_conditions,
                args.seed + 10_000 + stage_index,
            ),
        ])
        for condition_index, condition in enumerate(conditions):
            metrics = evaluate_condition(
                adaptive, model, benchmark, condition, args, device,
                args.seed + 100_000 * stage_index + condition_index,
            )
            row = {
                "stage": stage_index,
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
                f"xi={metrics['hill_xi_005']:.3f}",
                flush=True,
            )
        stage_summaries.append({
            "stage": stage_index,
            "radius": radius,
            "accepted": result["best_step"] > 0,
            "stop_reason": "",
            "train_median_raw_ess": train_cache["median_raw_ess"],
            "train_min_raw_ess": train_cache["min_raw_ess"],
            "validation_median_raw_ess": validation_cache["median_raw_ess"],
            "validation_min_raw_ess": validation_cache["min_raw_ess"],
            "best_step": result["best_step"],
            "initial_validation_score": result["initial"]["score"],
            "best_validation_score": result["best"]["score"],
        })
        pd.DataFrame(stage_summaries).to_csv(
            args.output / "stage_summary.csv", index=False
        )
        pd.DataFrame(evaluation_rows).to_csv(
            args.output / "evaluation.csv", index=False
        )
        if args.stop_on_no_improvement and result["best_step"] == 0:
            stage_summaries[-1]["stop_reason"] = (
                "grouped validation did not improve"
            )
            pd.DataFrame(stage_summaries).to_csv(
                args.output / "stage_summary.csv", index=False
            )
            print(
                "STOPPING RADIUS EXPANSION: grouped validation did not improve",
                flush=True,
            )
            break

    with (args.output / "config.json").open("w") as stream:
        json.dump({
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "output": str(args.output),
            "training_access": "f(x|c) only",
        }, stream, indent=2)
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
