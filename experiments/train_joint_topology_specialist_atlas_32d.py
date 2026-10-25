#!/usr/bin/env python3
"""Jointly train a protected atlas of conditional topology specialists.

The global proposal is frozen.  K trainable copies of an existing local-flow
checkpoint are attached to K condition-space centers.  Inside the atlas
domain their fixed smooth gate is

    q_atlas(x|c) = eps q_global(x|c)
                 + (1-eps) sum_k g_k(c) q_k(x|c),

where g_k is a normalized Gaussian kernel in condition space.  All specialist
log densities enter the same mixture denominator, so every update optimizes
the deployed mixture rather than an isolated expert.

Training uses only f(x|c).  Frozen caches are drawn from the current atlas and
an alpha ladder ends at alpha=2, where the objective is log(M2/I^2).
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-specialist-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--centers-file", type=Path, required=True,
        help="CSV containing c0,c1,c2 and preferably normalized_ess.",
    )
    parser.add_argument("--specialists", type=int, default=4)
    parser.add_argument(
        "--center-mixture-fraction", type=float, default=0.995,
        help="If the CSV has local_flow_fraction, select this audit row.",
    )
    parser.add_argument(
        "--condition-radius", type=float, default=0.01,
        help="Training-ball radius around each selected center.",
    )
    parser.add_argument(
        "--gate-length-scale", type=float, default=0.015,
        help="Gaussian gate length scale in condition space.",
    )
    parser.add_argument(
        "--global-fraction", type=float, default=0.005,
        help="Fixed protective weight of the frozen global proposal.",
    )
    parser.add_argument(
        "--alpha-ladder", type=float, nargs="+",
        default=(1.10, 1.25, 1.50, 1.75, 2.00),
    )
    parser.add_argument("--train-conditions", type=int, default=64)
    parser.add_argument("--validation-conditions", type=int, default=32)
    parser.add_argument("--points-per-condition", type=int, default=8192)
    parser.add_argument("--cache-condition-block", type=int, default=4)
    parser.add_argument("--replay-rounds", type=int, default=2)
    parser.add_argument("--steps-per-alpha", type=int, default=2000)
    parser.add_argument("--condition-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--anchor-penalty", type=float, default=1e-5)
    parser.add_argument("--worst-fraction", type=float, default=0.20)
    parser.add_argument("--worst-weight", type=float, default=0.50)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-improvement", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.specialists <= 0:
        parser.error("--specialists must be positive")
    if args.condition_radius < 0 or args.gate_length_scale <= 0:
        parser.error("condition radius must be nonnegative and gate scale positive")
    if not 0 < args.global_fraction < 1:
        parser.error("--global-fraction must lie in (0,1)")
    if any(alpha <= 1 or alpha > 2 for alpha in args.alpha_ladder):
        parser.error("alpha values must lie in (1,2]")
    if sorted(args.alpha_ladder) != list(args.alpha_ladder):
        parser.error("--alpha-ladder must be increasing")
    if not 0 < args.worst_fraction <= 1 or args.worst_weight < 0:
        parser.error("invalid worst-group settings")
    counts = (
        args.train_conditions, args.validation_conditions,
        args.points_per_condition, args.cache_condition_block,
        args.replay_rounds, args.steps_per_alpha, args.condition_batch,
    )
    if min(counts) <= 0:
        parser.error("all count and budget arguments must be positive")
    return args


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def frozen_copy(model, device):
    result = copy.deepcopy(model).to(device).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def load_centers(path, count, mixture_fraction):
    frame = pd.read_csv(path)
    missing = {"c0", "c1", "c2"}.difference(frame.columns)
    if missing:
        raise ValueError(f"centers file lacks {sorted(missing)}")
    if "local_flow_fraction" in frame.columns:
        values = frame["local_flow_fraction"].to_numpy(float)
        available = np.unique(values)
        chosen = available[np.argmin(np.abs(available - mixture_fraction))]
        frame = frame[np.isclose(values, chosen)].copy()
    if "normalized_ess" in frame.columns:
        frame = frame.sort_values("normalized_ess", ascending=True)
    elif "m2" in frame.columns:
        frame = frame.sort_values("m2", ascending=False)
    frame = frame.drop_duplicates(["c0", "c1", "c2"])
    if len(frame) < count:
        raise ValueError(f"requested {count} centers but only {len(frame)} are available")
    selected = frame.head(count).copy().reset_index(drop=True)
    centers = torch.tensor(
        selected[["c0", "c1", "c2"]].to_numpy(np.float32)
    )
    if bool(((centers < 0) | (centers > 1)).any()):
        raise ValueError("centers must lie in [0,1]^3")
    return centers, selected


def specialist_gate(cond, centers, length_scale, global_fraction):
    """Return [global, specialist_0, ..., specialist_K] weights."""
    distance2 = (cond[:, None, :] - centers[None, :, :]).square().sum(dim=2)
    logits = -0.5 * distance2 / (length_scale * length_scale)
    local_weights = torch.softmax(logits, dim=1) * (1.0 - global_fraction)
    global_weights = torch.full(
        (len(cond), 1), global_fraction, dtype=cond.dtype, device=cond.device
    )
    return torch.cat((global_weights, local_weights), dim=1)


def atlas_log_prob(global_model, specialists, x, cond, centers, args):
    weights = specialist_gate(
        cond, centers, args.gate_length_scale, args.global_fraction
    ).clamp_min(torch.finfo(cond.dtype).tiny)
    terms = [global_model.log_prob(x, cond).reshape(-1)]
    terms.extend(model.log_prob(x, cond).reshape(-1) for model in specialists)
    log_components = torch.stack(terms, dim=1)
    return torch.logsumexp(log_components + weights.log(), dim=1)


@torch.no_grad()
def sample_atlas(adaptive, global_model, specialists, cond, centers, args):
    weights = specialist_gate(
        cond, centers, args.gate_length_scale, args.global_fraction
    )
    component = torch.multinomial(weights, 1).reshape(-1)
    x = torch.empty(len(cond), DIM, dtype=cond.dtype, device=cond.device)
    models = [global_model, *specialists]
    for index, model in enumerate(models):
        mask = component == index
        count = int(mask.sum())
        if count:
            x[mask] = adaptive.sample_from_model(model, count, cond[mask])
    logq = atlas_log_prob(
        global_model, specialists, x, cond, centers, args
    )
    return x, logq


def sample_conditions(centers, count, radius, device):
    indices = torch.arange(count, device=device) % len(centers)
    indices = indices[torch.randperm(count, device=device)]
    conditions = centers[indices].clone()
    if radius > 0:
        direction = torch.randn(count, COND_DIM, device=device)
        direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
        radial = torch.rand(count, 1, device=device).pow(1.0 / COND_DIM)
        conditions += radius * radial * direction
    return conditions.clamp(0, 1), indices


@torch.no_grad()
def build_cache(
    adaptive, benchmark, global_model, proposal_specialists, centers,
    n_conditions, args, device, seed, label,
):
    adaptive.seed_everything(seed)
    conditions, owner = sample_conditions(
        centers, n_conditions, args.condition_radius, device
    )
    xs, conds, logfs, logrs, owners = [], [], [], [], []
    for start in range(0, n_conditions, args.cache_condition_block):
        block_conditions = conditions[start:start + args.cache_condition_block]
        groups = len(block_conditions)
        cond = block_conditions[:, None, :].expand(
            groups, args.points_per_condition, COND_DIM
        ).reshape(-1, COND_DIM)
        x, logr = sample_atlas(
            adaptive, global_model, proposal_specialists, cond, centers, args
        )
        logf = adaptive.target_log_integrand(benchmark, x, cond)
        xs.append(x.reshape(groups, args.points_per_condition, DIM))
        conds.append(cond.reshape(groups, args.points_per_condition, COND_DIM))
        logfs.append(logf.reshape(groups, args.points_per_condition))
        logrs.append(logr.reshape(groups, args.points_per_condition))
        owners.append(owner[start:start + groups])
    cache = {
        "x": torch.cat(xs), "cond": torch.cat(conds),
        "logf": torch.cat(logfs), "logr": torch.cat(logrs),
        "owner": torch.cat(owners), "label": label,
    }
    normalized = torch.softmax((cache["logf"] - cache["logr"]).double(), dim=1)
    raw_ess = 1.0 / normalized.square().sum(dim=1)
    print(f"\nJOINT ATLAS CACHE: {label}", flush=True)
    print(f"conditions             = {n_conditions:,}", flush=True)
    print(f"points / condition     = {args.points_per_condition:,}", flush=True)
    print(f"median raw ESS         = {float(raw_ess.median()):.2f}", flush=True)
    print(f"q05 raw ESS            = {float(torch.quantile(raw_ess, .05)):.2f}", flush=True)
    print(f"minimum raw ESS        = {float(raw_ess.min()):.2f}", flush=True)
    return cache


def robust_mean(values, args):
    if args.worst_weight == 0:
        return values.mean()
    count = max(1, int(math.ceil(args.worst_fraction * len(values))))
    return values.mean() + args.worst_weight * torch.topk(values, count).values.mean()


def parameter_anchor(specialists, references):
    total = torch.zeros((), device=next(specialists[0].parameters()).device)
    count = 0
    for model, model_reference in zip(specialists, references):
        for parameter, reference in zip(model.parameters(), model_reference):
            total += (parameter - reference).square().sum()
            count += parameter.numel()
    return total / max(count, 1)


def group_losses(global_model, specialists, cache, indices, centers, alpha, args):
    x, cond = cache["x"][indices], cache["cond"][indices]
    logf, logr = cache["logf"][indices], cache["logr"][indices]
    groups, points, _ = x.shape
    logq = atlas_log_prob(
        global_model, specialists, x.reshape(-1, DIM),
        cond.reshape(-1, COND_DIM), centers, args,
    ).reshape(groups, points)
    log_score = (
        alpha * logf.double()
        + (1.0 - alpha) * logq.detach().double()
        - logr.double()
    )
    score = torch.softmax(log_score, dim=1).to(logq.dtype)
    return -(score * logq).sum(dim=1)


@torch.no_grad()
def cache_r2(global_model, specialists, cache, centers, args):
    output = []
    logn = math.log(cache["x"].shape[1])
    for start in range(0, len(cache["x"]), 4):
        x = cache["x"][start:start + 4]
        cond = cache["cond"][start:start + 4]
        logf = cache["logf"][start:start + 4].double()
        logr = cache["logr"][start:start + 4].double()
        groups, points, _ = x.shape
        logq = atlas_log_prob(
            global_model, specialists, x.reshape(-1, DIM),
            cond.reshape(-1, COND_DIM), centers, args,
        ).reshape(groups, points).double()
        log_i = torch.logsumexp(logf - logr, dim=1) - logn
        log_m2 = torch.logsumexp(2 * logf - logq - logr, dim=1) - logn
        output.append((log_m2 - 2 * log_i).cpu())
    return torch.cat(output)


@torch.no_grad()
def validation(global_model, specialists, caches, centers, args):
    values = torch.cat([
        cache_r2(global_model, specialists, cache, centers, args)
        for cache in caches
    ])
    score = robust_mean(values, args)
    return {
        "score": float(score), "mean_r2": float(values.mean()),
        "median_r2": float(values.median()),
        "q90_r2": float(torch.quantile(values, .90)),
        "median_ess": float(torch.exp(-values.median())),
        "q10_ess": float(torch.exp(-torch.quantile(values, .90))),
        "minimum_ess": float(torch.exp(-values.max())),
    }


def train_stage(
    global_model, specialists, train_caches, validation_caches,
    centers, references, alpha, args,
):
    parameters = [p for model in specialists for p in model.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial = validation(global_model, specialists, validation_caches, centers, args)
    best, best_step = initial, 0
    best_states = [cpu_state(model) for model in specialists]
    patience, history = 0, []
    generator = np.random.default_rng(args.seed + int(alpha * 10000))
    print(
        f"\nJOINT ATLAS RENYI STAGE alpha={alpha:.2f}: "
        f"score={initial['score']:.5f} medianESS={initial['median_ess']:.4f} "
        f"q10ESS={initial['q10_ess']:.4f}", flush=True,
    )
    for start in range(0, args.steps_per_alpha, args.check_interval):
        block = min(args.check_interval, args.steps_per_alpha - start)
        train_values = []
        for model in specialists:
            model.train()
        for _ in range(block):
            cache = train_caches[int(generator.integers(len(train_caches)))]
            indices = torch.randint(
                0, len(cache["x"]), (args.condition_batch,),
                device=cache["x"].device,
            )
            losses = group_losses(
                global_model, specialists, cache, indices, centers, alpha, args
            )
            loss = robust_mean(losses, args)
            loss += args.anchor_penalty * parameter_anchor(specialists, references)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()
            train_values.append(float(loss.detach()))
        step = start + block
        current = validation(
            global_model, specialists, validation_caches, centers, args
        )
        improved = current["score"] < (
            best["score"] - args.minimum_improvement * max(abs(best["score"]), 1)
        )
        if improved:
            best, best_step = current, step
            best_states = [cpu_state(model) for model in specialists]
            patience = 0
        else:
            patience += 1
        history.append({
            "step": step,
            "train_loss": float(np.mean(train_values)) if train_values else math.nan,
            **{f"validation_{key}": value for key, value in current.items()},
            "best_score": best["score"],
        })
        print(
            f"alpha={alpha:.2f} step {step:5d}/{args.steps_per_alpha}: "
            f"score={current['score']:.5f} medianESS={current['median_ess']:.4f} "
            f"q10ESS={current['q10_ess']:.4f} minESS={current['minimum_ess']:.4f} "
            f"best={best['score']:.5f} patience={patience}/{args.patience}",
            flush=True,
        )
        if patience >= args.patience:
            break
    for model, state in zip(specialists, best_states):
        model.load_state_dict(state, strict=True)
    restored = validation(global_model, specialists, validation_caches, centers, args)
    return {
        "alpha": alpha, "initial": initial, "best": best,
        "restored": restored, "best_step": best_step, "history": history,
    }


def save_outputs(
    output, global_checkpoint_path, initial_checkpoint_path,
    initial_checkpoint, specialists, centers, selected_centers, args,
    stage_results,
):
    output.mkdir(parents=True, exist_ok=True)
    specialist_dir = output / "specialists"
    specialist_dir.mkdir(exist_ok=True)
    specialist_paths = []
    for index, model in enumerate(specialists):
        path = specialist_dir / f"specialist_{index:02d}.pt"
        torch.save({
            **initial_checkpoint,
            "model_state_dict": cpu_state(model),
            "joint_atlas_specialist": {
                "index": index, "center": centers[index].detach().cpu().tolist(),
                "atlas_output": str(output),
            },
        }, path)
        specialist_paths.append(str(path))
    atlas = {
        "format": "joint_topology_specialist_atlas_v1",
        "benchmark": initial_checkpoint.get("benchmark"),
        "global_checkpoint": str(global_checkpoint_path),
        "initial_specialist_checkpoint": str(initial_checkpoint_path),
        "centers": centers.detach().cpu(),
        "gate_length_scale": args.gate_length_scale,
        "global_fraction": args.global_fraction,
        "specialist_state_dicts": [cpu_state(model) for model in specialists],
        "specialist_checkpoints": specialist_paths,
        "stage_results": stage_results,
    }
    torch.save(atlas, output / "joint_specialist_atlas.pt")
    selected_centers.to_csv(output / "selected_centers.csv", index=False)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    adaptive, global_model, global_checkpoint, benchmark = local.load_model(
        args.global_checkpoint, device
    )
    local_adaptive, initial_model, initial_checkpoint, local_benchmark = local.load_model(
        args.initial_specialist_checkpoint, device
    )
    if benchmark != local_benchmark:
        raise ValueError(f"benchmark mismatch: {benchmark} vs {local_benchmark}")
    del local_adaptive
    global_model = frozen_copy(global_model, device)
    specialists = [copy.deepcopy(initial_model).to(device) for _ in range(args.specialists)]
    centers_cpu, selected_centers = load_centers(
        args.centers_file, args.specialists, args.center_mixture_fraction
    )
    centers = centers_cpu.to(device)
    references = [
        [parameter.detach().clone() for parameter in model.parameters()]
        for model in specialists
    ]
    print("\nSELECTED SPECIALIST CENTERS")
    print(selected_centers.to_string(index=False))
    print(
        f"\nprotected atlas: global={args.global_fraction:.4f}, "
        f"local total={1-args.global_fraction:.4f}, "
        f"gate scale={args.gate_length_scale:.4g}", flush=True,
    )
    train_replay, validation_replay = [], []
    stage_results, summary_rows = [], []
    for stage, alpha in enumerate(args.alpha_ladder):
        proposal_specialists = [frozen_copy(model, device) for model in specialists]
        train_replay.append(build_cache(
            adaptive, benchmark, global_model, proposal_specialists, centers,
            args.train_conditions, args, device, args.seed + stage * 10000,
            f"train alpha={alpha:.3f}",
        ))
        validation_replay.append(build_cache(
            adaptive, benchmark, global_model, proposal_specialists, centers,
            args.validation_conditions, args, device,
            args.seed + stage * 10000 + 1, f"validation alpha={alpha:.3f}",
        ))
        train_replay = train_replay[-args.replay_rounds:]
        validation_replay = validation_replay[-args.replay_rounds:]
        result = train_stage(
            global_model, specialists, train_replay, validation_replay,
            centers, references, alpha, args,
        )
        stage_results.append(result)
        stage_dir = args.output / f"alpha_{alpha:.3f}"
        stage_dir.mkdir(exist_ok=True)
        pd.DataFrame(result["history"]).to_csv(
            stage_dir / "training_history.csv", index=False
        )
        summary_rows.append({
            "stage": stage, "alpha": alpha, "best_step": result["best_step"],
            "initial_score": result["initial"]["score"],
            "final_score": result["restored"]["score"],
            "initial_median_ess": result["initial"]["median_ess"],
            "final_median_ess": result["restored"]["median_ess"],
            "initial_q10_ess": result["initial"]["q10_ess"],
            "final_q10_ess": result["restored"]["q10_ess"],
            "final_minimum_ess": result["restored"]["minimum_ess"],
        })
        pd.DataFrame(summary_rows).to_csv(args.output / "stage_summary.csv", index=False)
        save_outputs(
            args.output, args.global_checkpoint, args.initial_specialist_checkpoint,
            initial_checkpoint, specialists, centers, selected_centers,
            args, stage_results,
        )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "benchmark": benchmark,
        "objective": "mean plus worst-group within-condition Renyi divergence",
        "training_access": "f(x|c) only",
        "exact_target_sampler_used": False,
        "deployment_scope": "use this gate inside the union of its local charts; use the global proposal outside",
    })
    with (args.output / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2)
    print("\nJOINT SPECIALIST ATLAS SUMMARY")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nAtlas: {(args.output / 'joint_specialist_atlas.pt').resolve()}")


if __name__ == "__main__":
    main()
