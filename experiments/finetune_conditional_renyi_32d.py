#!/usr/bin/env python3
"""Conditional Renyi-alpha tail fine tuning for the 32D geometry ladder.

The input checkpoint is treated as a frozen, already-trained bulk proposal.
For alpha > 1, the optimized quantity for each condition is

    R_alpha = [log int f^alpha q^(1-alpha) dx - alpha log int f dx]
              / (alpha - 1).

At alpha=2 this is log(M2 / I^2), so exp(-R_2) is normalized ESS.
Training samples come from a defensive frozen-proposal mixture r.  The score
weights are therefore proportional to f^alpha q^(1-alpha) / r, not simply to
(f/r)^alpha.  Fixed independent caches determine stage acceptance.
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


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_local_conditional_continuation_32d as local  # noqa: E402


DIM = local.DIM
COND_DIM = local.COND_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--anchor", type=float, nargs=COND_DIM, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument(
        "--condition-radius", type=float, default=0.05,
        help=(
            "Radius of the local condition neighborhood. Set this to zero "
            "with --anchor-fraction 1 to fine-tune one fixed condition."
        ),
    )
    parser.add_argument(
        "--alpha-ladder", type=float, nargs="+",
        default=(1.25, 1.50, 1.75, 2.00),
    )
    parser.add_argument("--train-conditions", type=int, default=128)
    parser.add_argument("--validation-conditions", type=int, default=64)
    parser.add_argument("--points-per-condition", type=int, default=8192)
    parser.add_argument("--anchor-fraction", type=float, default=0.25)
    parser.add_argument(
        "--priority-conditions-file", type=Path,
        help=(
            "Optional CSV containing c0,c1,c2 conditions found by an external "
            "tail search. A fraction of every cache is drawn near these "
            "conditions; the remainder keeps broad local coverage."
        ),
    )
    parser.add_argument(
        "--priority-condition-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--priority-jitter-radius", type=float, default=0.005,
        help="Uniform coordinate jitter around priority conditions.",
    )
    parser.add_argument("--uniform-fraction", type=float, default=0.02)
    parser.add_argument(
        "--cache-condition-block", type=int, default=8,
        help="Number of condition groups sampled at once while building caches.",
    )
    parser.add_argument(
        "--replay-rounds", type=int, default=2,
        help="Number of recent independent train/validation caches retained.",
    )
    parser.add_argument("--steps-per-alpha", type=int, default=1200)
    parser.add_argument("--condition-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--anchor-penalty", type=float, default=1e-3)
    parser.add_argument("--worst-fraction", type=float, default=0.20)
    parser.add_argument("--worst-weight", type=float, default=0.50)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--minimum-improvement", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.condition_radius < 0.0:
        parser.error("--condition-radius must be nonnegative")
    if any(value < 0.0 or value > 1.0 for value in args.anchor):
        parser.error("anchor must lie in [0,1]^3")
    if any(alpha <= 1.0 or alpha > 2.0 for alpha in args.alpha_ladder):
        parser.error("every alpha must lie in (1,2]")
    if sorted(args.alpha_ladder) != list(args.alpha_ladder):
        parser.error("--alpha-ladder must be increasing")
    if not 0.0 < args.anchor_fraction <= 1.0:
        parser.error("--anchor-fraction must lie in (0,1]")
    if not 0.0 <= args.priority_condition_fraction <= 1.0:
        parser.error("--priority-condition-fraction must lie in [0,1]")
    if args.priority_jitter_radius < 0.0:
        parser.error("--priority-jitter-radius must be nonnegative")
    if not 0.0 < args.uniform_fraction < 1.0:
        parser.error("--uniform-fraction must lie in (0,1)")
    if not 0.0 < args.worst_fraction <= 1.0:
        parser.error("--worst-fraction must lie in (0,1]")
    if args.worst_weight < 0.0:
        parser.error("--worst-weight must be nonnegative")
    if min(
        args.train_conditions, args.validation_conditions,
        args.points_per_condition, args.cache_condition_block,
        args.replay_rounds, args.steps_per_alpha, args.condition_batch,
    ) <= 0:
        parser.error("all count and budget arguments must be positive")
    return args


def cpu_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def freeze_copy(model, device):
    frozen = copy.deepcopy(model).to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def load_priority_conditions(path):
    if path is None:
        return None
    frame = pd.read_csv(path)
    missing = {"c0", "c1", "c2"}.difference(frame.columns)
    if missing:
        raise ValueError(f"priority condition file lacks {sorted(missing)}")
    values = frame[["c0", "c1", "c2"]].to_numpy(np.float32)
    if len(values) == 0:
        raise ValueError("priority condition file is empty")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("priority conditions must lie in [0,1]^3")
    return values


def parameter_anchor(model, reference_parameters):
    total = torch.zeros((), device=next(model.parameters()).device)
    count = 0
    for parameter, reference in zip(model.parameters(), reference_parameters):
        total = total + (parameter - reference).square().sum()
        count += parameter.numel()
    return total / max(count, 1)


@torch.no_grad()
def build_cache(
    adaptive, benchmark, proposal, n_conditions, points_per_condition,
    args, device, seed, label, priority_conditions=None,
):
    """Build a fixed cache with the exact defensive proposal log density."""
    adaptive.seed_everything(seed)
    condition_sampler = local.make_condition_sampler(
        args.anchor, args.condition_radius, args.anchor_fraction
    )
    anchors = condition_sampler(n_conditions, device)
    priority_count = 0
    if priority_conditions is not None and args.priority_condition_fraction > 0:
        priority_count = min(
            n_conditions,
            max(1, int(round(
                n_conditions * args.priority_condition_fraction
            ))),
        )
        priority = torch.as_tensor(
            priority_conditions, dtype=torch.float32, device=device
        )
        indices = torch.randint(
            0, len(priority), (priority_count,), device=device
        )
        selected = priority[indices]
        if args.priority_jitter_radius > 0.0:
            selected = selected + (
                2.0 * torch.rand_like(selected) - 1.0
            ) * args.priority_jitter_radius
        anchor = torch.as_tensor(
            args.anchor, dtype=torch.float32, device=device
        )
        lower = torch.clamp(anchor - args.condition_radius, min=0.0)
        upper = torch.clamp(anchor + args.condition_radius, max=1.0)
        anchors[:priority_count] = torch.maximum(
            torch.minimum(selected, upper), lower
        )
    xs, conds, logfs, logrs = [], [], [], []
    for start in range(0, n_conditions, args.cache_condition_block):
        block_anchors = anchors[start:start + args.cache_condition_block]
        groups = len(block_anchors)
        cond = block_anchors[:, None, :].expand(
            groups, points_per_condition, COND_DIM
        ).reshape(groups * points_per_condition, COND_DIM)
        x, log_r = adaptive.sample_from_proposal(
            proposal_model=proposal,
            n=groups * points_per_condition,
            cond=cond,
            device=device,
            uniform_fraction=args.uniform_fraction,
        )
        log_f = adaptive.target_log_integrand(benchmark, x, cond).reshape(
            groups, points_per_condition
        )
        xs.append(x.reshape(groups, points_per_condition, DIM).detach())
        conds.append(cond.reshape(
            groups, points_per_condition, COND_DIM
        ).detach())
        logfs.append(log_f.detach())
        logrs.append(log_r.reshape(groups, points_per_condition).detach())

    cache = {
        "x": torch.cat(xs, dim=0),
        "cond": torch.cat(conds, dim=0),
        "logf": torch.cat(logfs, dim=0),
        "logr": torch.cat(logrs, dim=0),
        "label": label,
    }
    ordinary_logw = (cache["logf"] - cache["logr"]).double()
    ordinary_alpha = torch.softmax(ordinary_logw, dim=1)
    raw_ess = 1.0 / ordinary_alpha.square().sum(dim=1)
    max_share = ordinary_alpha.max(dim=1).values
    print(f"\nRENYI CACHE: {label}", flush=True)
    print(f"conditions             = {n_conditions:,}", flush=True)
    print(f"priority conditions    = {priority_count:,}", flush=True)
    print(f"points / condition     = {points_per_condition:,}", flush=True)
    print(f"uniform fraction       = {args.uniform_fraction:.4f}", flush=True)
    print(f"median raw ESS         = {float(raw_ess.median()):.2f}", flush=True)
    print(
        f"q05 raw ESS            = {float(torch.quantile(raw_ess, 0.05)):.2f}",
        flush=True,
    )
    print(f"minimum raw ESS        = {float(raw_ess.min()):.2f}", flush=True)
    print(
        f"q90 maximum share      = {float(torch.quantile(max_share, 0.90)):.4f}",
        flush=True,
    )
    return cache


def robust_mean(values, worst_fraction, worst_weight):
    if worst_weight == 0.0:
        return values.mean()
    count = max(1, int(math.ceil(worst_fraction * len(values))))
    worst = torch.topk(values, count).values.mean()
    return values.mean() + worst_weight * worst


def objective_name(args):
    if args.worst_weight == 0.0:
        return "mean conditional Renyi-alpha divergence"
    return "mean plus worst-group conditional Renyi-alpha divergence"


def training_group_losses(model, cache, indices, alpha):
    x = cache["x"][indices]
    cond = cache["cond"][indices]
    logf = cache["logf"][indices]
    logr = cache["logr"][indices]
    groups, points, _ = x.shape
    logq = model.log_prob(
        x.reshape(groups * points, DIM),
        cond.reshape(groups * points, COND_DIM),
    ).reshape(groups, points)
    # h_alpha(x) propto f(x)^alpha q(x)^(1-alpha).
    # Since x ~ r, the correct fixed-cache coefficient is h_alpha / r.
    log_score_weight = (
        alpha * logf.double()
        + (1.0 - alpha) * logq.detach().double()
        - logr.double()
    )
    score_weight = torch.softmax(log_score_weight, dim=1).to(logq.dtype)
    return -(score_weight * logq).sum(dim=1)


@torch.no_grad()
def cache_renyi_values(model, cache, alpha, condition_chunk=4):
    model.eval()
    output_alpha = []
    output_two = []
    groups_total = len(cache["x"])
    log_points = math.log(cache["x"].shape[1])
    for start in range(0, groups_total, condition_chunk):
        x = cache["x"][start:start + condition_chunk]
        cond = cache["cond"][start:start + condition_chunk]
        logf = cache["logf"][start:start + condition_chunk].double()
        logr = cache["logr"][start:start + condition_chunk].double()
        groups, points, _ = x.shape
        logq = model.log_prob(
            x.reshape(groups * points, DIM),
            cond.reshape(groups * points, COND_DIM),
        ).reshape(groups, points).double()
        log_i = torch.logsumexp(logf - logr, dim=1) - log_points

        def values(order):
            log_j = torch.logsumexp(
                order * logf + (1.0 - order) * logq - logr,
                dim=1,
            ) - log_points
            return (log_j - order * log_i) / (order - 1.0)

        output_alpha.append(values(alpha).cpu())
        output_two.append(values(2.0).cpu())
    return torch.cat(output_alpha), torch.cat(output_two)


@torch.no_grad()
def validation_score(model, caches, alpha, args):
    alpha_values, two_values = [], []
    for cache in caches:
        current_alpha, current_two = cache_renyi_values(model, cache, alpha)
        alpha_values.append(current_alpha)
        two_values.append(current_two)
    alpha_values = torch.cat(alpha_values)
    two_values = torch.cat(two_values)
    score = robust_mean(
        alpha_values, args.worst_fraction, args.worst_weight
    )
    return {
        "score": float(score),
        "mean_r_alpha": float(alpha_values.mean()),
        "median_r_alpha": float(alpha_values.median()),
        "q90_r_alpha": float(torch.quantile(alpha_values, 0.90)),
        "mean_r2": float(two_values.mean()),
        "median_r2": float(two_values.median()),
        "q90_r2": float(torch.quantile(two_values, 0.90)),
        "median_ess_proxy": float(torch.exp(-two_values.median())),
        "q10_ess_proxy": float(torch.exp(-torch.quantile(two_values, 0.90))),
    }


def train_alpha_stage(
    model, train_caches, validation_caches, bulk_parameters, alpha, args,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial = validation_score(model, validation_caches, alpha, args)
    best = initial
    best_state = cpu_state(model)
    best_step = 0
    patience = 0
    history = []
    generator = np.random.default_rng(args.seed + int(round(alpha * 10_000)))

    print(
        f"\nRENYI STAGE alpha={alpha:.3f}: "
        f"initial score={initial['score']:.6f} "
        f"median R2={initial['median_r2']:.4f} "
        f"ESS proxy={initial['median_ess_proxy']:.4f}",
        flush=True,
    )
    for start in range(0, args.steps_per_alpha, args.check_interval):
        block = min(args.check_interval, args.steps_per_alpha - start)
        train_values = []
        model.train()
        for _ in range(block):
            cache = train_caches[int(generator.integers(len(train_caches)))]
            indices = torch.randint(
                0, len(cache["x"]), (args.condition_batch,),
                device=cache["x"].device,
            )
            group_losses = training_group_losses(model, cache, indices, alpha)
            loss = robust_mean(
                group_losses, args.worst_fraction, args.worst_weight
            )
            loss = loss + args.anchor_penalty * parameter_anchor(
                model, bulk_parameters
            )
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            optimizer.step()
            train_values.append(float(loss.detach()))

        step = start + block
        current = validation_score(model, validation_caches, alpha, args)
        improved = current["score"] < (
            best["score"]
            - args.minimum_improvement * max(abs(best["score"]), 1.0)
        )
        if improved:
            best = current
            best_state = cpu_state(model)
            best_step = step
            patience = 0
        else:
            patience += 1
        row = {
            "step": step,
            "train_loss": (
                float(np.mean(train_values)) if train_values else float("nan")
            ),
            **{f"validation_{key}": value for key, value in current.items()},
            "best_score": best["score"],
        }
        history.append(row)
        print(
            f"alpha={alpha:.2f} step {step:4d}/{args.steps_per_alpha}: "
            f"train={row['train_loss']:.6f} score={current['score']:.6f} "
            f"medianR2={current['median_r2']:.4f} "
            f"q90R2={current['q90_r2']:.4f} "
            f"ESS={current['median_ess_proxy']:.4f} "
            f"best={best['score']:.6f} patience={patience}/{args.patience}",
            flush=True,
        )
        if patience >= args.patience:
            break

    model.load_state_dict(best_state, strict=True)
    restored = validation_score(model, validation_caches, alpha, args)
    return {
        "alpha": alpha,
        "initial": initial,
        "best": best,
        "restored": restored,
        "best_step": best_step,
        "history": history,
    }


def save_checkpoint(path, source_checkpoint, model, args, stage_results):
    checkpoint = {
        **source_checkpoint,
        "model_state_dict": cpu_state(model),
        "conditional_renyi_finetuning": {
            "anchor": list(args.anchor),
            "condition_radius": args.condition_radius,
            "alpha_ladder": list(args.alpha_ladder),
            "stage_results": stage_results,
            "objective": objective_name(args),
            "priority_conditions_file": (
                None if args.priority_conditions_file is None
                else str(args.priority_conditions_file)
            ),
            "priority_condition_fraction": args.priority_condition_fraction,
            "priority_jitter_radius": args.priority_jitter_radius,
            "training_access": "f(x|c) only",
            "exact_target_sampler_used": False,
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
        torch.backends.cudnn.allow_tf32 = True

    adaptive, model, source_checkpoint, benchmark = local.load_model(
        args.checkpoint, device
    )
    priority_conditions = load_priority_conditions(
        args.priority_conditions_file
    )
    model.to(device)
    bulk_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    train_replay = []
    validation_replay = []
    stage_results = []
    summary_rows = []

    for stage_index, alpha in enumerate(args.alpha_ladder):
        proposal = freeze_copy(model, device)
        train_replay.append(build_cache(
            adaptive, benchmark, proposal,
            args.train_conditions, args.points_per_condition,
            args, device, args.seed + 10_000 * stage_index,
            f"train alpha={alpha:.3f} round={stage_index}",
            priority_conditions,
        ))
        validation_replay.append(build_cache(
            adaptive, benchmark, proposal,
            args.validation_conditions, args.points_per_condition,
            args, device, args.seed + 10_000 * stage_index + 1,
            f"validation alpha={alpha:.3f} round={stage_index}",
            priority_conditions,
        ))
        train_replay = train_replay[-args.replay_rounds:]
        validation_replay = validation_replay[-args.replay_rounds:]

        result = train_alpha_stage(
            model, train_replay, validation_replay,
            bulk_parameters, alpha, args,
        )
        stage_results.append(result)
        stage_dir = args.output / f"alpha_{alpha:.3f}"
        stage_dir.mkdir(exist_ok=True)
        pd.DataFrame(result["history"]).to_csv(
            stage_dir / "training_history.csv", index=False
        )
        save_checkpoint(
            stage_dir / "conditional_model.pt", source_checkpoint,
            model, args, stage_results,
        )
        summary_rows.append({
            "stage": stage_index,
            "alpha": alpha,
            "best_step": result["best_step"],
            "initial_score": result["initial"]["score"],
            "best_score": result["best"]["score"],
            "initial_median_r2": result["initial"]["median_r2"],
            "final_median_r2": result["restored"]["median_r2"],
            "initial_q90_r2": result["initial"]["q90_r2"],
            "final_q90_r2": result["restored"]["q90_r2"],
            "final_median_ess_proxy": result["restored"][
                "median_ess_proxy"
            ],
            "final_q10_ess_proxy": result["restored"]["q10_ess_proxy"],
        })
        pd.DataFrame(summary_rows).to_csv(
            args.output / "stage_summary.csv", index=False
        )

    save_checkpoint(
        args.output / "conditional_renyi_model.pt", source_checkpoint,
        model, args, stage_results,
    )
    with (args.output / "config.json").open("w") as stream:
        serializable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        json.dump({
            **serializable,
            "objective": objective_name(args),
            "training_access": "f(x|c) only",
            "exact_target_sampler_used": False,
        }, stream, indent=2)
    print("\nRENYI CONDITIONAL FINE-TUNING SUMMARY")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nModel: {(args.output / 'conditional_renyi_model.pt').resolve()}")


if __name__ == "__main__":
    main()
