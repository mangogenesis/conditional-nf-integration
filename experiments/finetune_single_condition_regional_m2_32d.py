#!/usr/bin/env python3
"""Iteratively reduce global and worst-regional M2 at one 32D condition.

Each round has three statistically separate pieces:

1. defensive discovery samples locate high contributions to integral f^2/q;
2. frozen-proposal train/validation caches optimize
       M2 + C * max_region M2_region;
3. a fresh pure-NF audit measures M2 convergence at increasing sample sizes.

The regions are a Voronoi partition whose centers are fitted only to retained
high-contribution discovery points.  They are rediscovered after every round.
No exact target sampler or true integral is used for training or selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
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
from nf.regional_m2 import (  # noqa: E402
    assign_nearest_region,
    box_mixture_log_prob,
    combined_log_objective,
    diverse_weighted_centers,
    merge_diverse_centers,
    regional_log_moments,
    sample_box_mixture,
)


DIM = local.DIM
COND_DIM = local.COND_DIM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--condition", type=float, nargs=COND_DIM, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument(
        "--initial-discovery-data", type=Path,
        help=(
            "Optional largest_weight_points.pt from the frozen M2 audit. "
            "These points seed round zero; later rounds always rediscover."
        ),
    )
    parser.add_argument(
        "--initial-region-data", type=Path, nargs="*", default=(),
        help=(
            "Prior regional_discovery.pt files whose centers seed the "
            "cumulative regional atlas."
        ),
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument(
        "--maximum-regions", type=int, default=64,
        help="Maximum number of old plus newly discovered atlas centers.",
    )
    parser.add_argument(
        "--region-bank-minimum-distance", type=float, default=0.05,
        help="Minimum distance required to append a new atlas center.",
    )
    parser.add_argument("--minimum-center-distance", type=float, default=0.10)
    parser.add_argument("--discovery-replicates", type=int, default=4)
    parser.add_argument("--discovery-evals", type=int, default=1_000_000)
    parser.add_argument("--discovery-keep", type=int, default=512)
    parser.add_argument("--train-groups", type=int, default=16)
    parser.add_argument("--validation-groups", type=int, default=8)
    parser.add_argument("--points-per-group", type=int, default=32768)
    parser.add_argument("--steps-per-round", type=int, default=2400)
    parser.add_argument("--group-batch", type=int, default=2)
    parser.add_argument("--regional-weight", type=float, default=1.0)
    parser.add_argument(
        "--regional-temperature", type=float, default=0.02,
        help="Smooth-max temperature in regional/global M2-fraction units.",
    )
    parser.add_argument("--uniform-fraction", type=float, default=0.02)
    parser.add_argument(
        "--regional-sample-fraction", type=float, default=0.25,
        help=(
            "Fraction of train/validation cache points drawn from the local "
            "regional box mixture. Discovery itself still uses the broad "
            "defensive proposal."
        ),
    )
    parser.add_argument(
        "--regional-box-halfwidth", type=float, default=0.05,
        help="Coordinate half-width of each clipped regional uniform box.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--anchor-penalty", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-improvement", type=float, default=1e-3)
    parser.add_argument("--audit-replicates", type=int, default=10)
    parser.add_argument("--audit-evals", type=int, default=1_000_000)
    parser.add_argument(
        "--audit-milestones", type=int, nargs="+",
        default=(100_000, 250_000, 500_000, 1_000_000),
    )
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(value < 0.0 or value > 1.0 for value in args.condition):
        parser.error("condition must lie in [0,1]^3")
    counts = (
        args.rounds, args.regions, args.maximum_regions,
        args.discovery_replicates,
        args.discovery_evals, args.discovery_keep, args.train_groups,
        args.validation_groups, args.points_per_group, args.steps_per_round,
        args.group_batch, args.check_interval, args.patience,
        args.audit_replicates, args.audit_evals, args.chunk,
    )
    if min(counts) <= 0:
        parser.error("all count and budget arguments must be positive")
    if not 0.0 < args.uniform_fraction < 1.0:
        parser.error("uniform-fraction must lie in (0,1)")
    if not 0.0 < args.regional_sample_fraction < 1.0:
        parser.error("regional-sample-fraction must lie in (0,1)")
    if not 0.0 < args.regional_box_halfwidth <= 1.0:
        parser.error("regional-box-halfwidth must lie in (0,1]")
    if args.regional_weight < 0 or args.regional_temperature < 0:
        parser.error("regional weight and temperature must be nonnegative")
    if args.minimum_center_distance < 0:
        parser.error("minimum center distance must be nonnegative")
    if args.region_bank_minimum_distance < 0:
        parser.error("region-bank minimum distance must be nonnegative")
    if any(value <= 1 or value > args.audit_evals for value in args.audit_milestones):
        parser.error("audit milestones must lie in [2, audit-evals]")
    return args


def cpu_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def freeze_copy(model, device):
    result = copy.deepcopy(model).to(device).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def parameter_anchor(model, reference_parameters):
    total = torch.zeros((), device=next(model.parameters()).device)
    count = 0
    for parameter, reference in zip(model.parameters(), reference_parameters):
        total = total + (parameter - reference).square().sum()
        count += parameter.numel()
    return total / max(count, 1)


def fixed_condition(args, count, device):
    return torch.tensor(
        args.condition, dtype=torch.float32, device=device
    ).expand(count, COND_DIM)


@torch.no_grad()
def defensive_sample(
    adaptive, proposal, benchmark, count, args, device, centers=None,
):
    cond = fixed_condition(args, count, device)
    x, _ = adaptive.sample_from_proposal(
        proposal_model=proposal,
        n=count,
        cond=cond,
        device=device,
        uniform_fraction=args.uniform_fraction,
    )
    if centers is not None:
        use_regional = (
            torch.rand(count, device=device) < args.regional_sample_fraction
        )
        if use_regional.any():
            x[use_regional] = sample_box_mixture(
                centers,
                int(use_regional.sum()),
                args.regional_box_halfwidth,
            )
    logq = proposal.log_prob(x, cond).reshape(-1)
    logr = torch.logaddexp(
        logq + math.log1p(-args.uniform_fraction),
        torch.full_like(logq, math.log(args.uniform_fraction)),
    )
    if centers is not None:
        logg = box_mixture_log_prob(
            x, centers, args.regional_box_halfwidth
        )
        logs = torch.logaddexp(
            logr + math.log1p(-args.regional_sample_fraction),
            logg + math.log(args.regional_sample_fraction),
        )
    else:
        logs = logr
    logf = adaptive.target_log_integrand(benchmark, x, cond).reshape(-1)
    # x ~ s, where s is either the broad defensive proposal or its exact
    # mixture with regional boxes. This estimates integral f^2/q.
    log_contribution = 2.0 * logf.double() - logq.double() - logs.double()
    return x, logf, logq, logs, log_contribution


@torch.no_grad()
def discover_regions(
    adaptive, proposal, benchmark, args, device, round_index,
    initial_data=None, previous_centers=None,
):
    kept = []
    kept_log = []
    if round_index == 0 and initial_data is not None:
        x = initial_data["x"].to(device=device, dtype=torch.float32)
        if "log_f" in initial_data and "log_q" in initial_data:
            # Existing retained points were normally sampled from pure q.
            logc = 2.0 * (
                initial_data["log_f"].double()
                - initial_data["log_q"].double()
            ).to(device)
        else:
            logc = 2.0 * initial_data["log_weight"].double().to(device)
        kept.append(x)
        kept_log.append(logc)

    for replicate in range(args.discovery_replicates):
        torch.manual_seed(
            args.seed + 100_000 * round_index + 10_007 * replicate
        )
        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                args.seed + 100_000 * round_index + 10_007 * replicate
            )
        best_x = None
        best_log = None
        remaining = args.discovery_evals
        while remaining:
            count = min(args.chunk, remaining)
            x, _, _, _, logc = defensive_sample(
                adaptive, proposal, benchmark, count, args, device
            )
            take = min(args.discovery_keep, count)
            values, indices = torch.topk(logc, take)
            candidate_x = x[indices]
            if best_x is None:
                best_x, best_log = candidate_x, values
            else:
                joined_log = torch.cat((best_log, values))
                joined_x = torch.cat((best_x, candidate_x))
                indices = torch.topk(
                    joined_log, min(args.discovery_keep, len(joined_log))
                ).indices
                best_log = joined_log[indices]
                best_x = joined_x[indices]
            remaining -= count
        kept.append(best_x)
        kept_log.append(best_log)
        print(
            f"discovery round={round_index} replicate={replicate + 1}/"
            f"{args.discovery_replicates} max log contribution="
            f"{float(best_log.max()):.6f}",
            flush=True,
        )

    x = torch.cat(kept, dim=0)
    logc = torch.cat(kept_log, dim=0)
    take = min(len(x), max(args.regions * 16, args.discovery_keep * 2))
    selected = torch.topk(logc, take).indices
    x = x[selected]
    logc = logc[selected]
    if len(x) < args.regions:
        raise RuntimeError("too few retained points to construct regions")
    centers = diverse_weighted_centers(
        x,
        logc,
        args.regions,
        minimum_distance=args.minimum_center_distance,
    )
    # Local proposals must be centered on genuine high-contribution samples,
    # not at an average that might lie off a curved target structure.
    new_centers = x[torch.cdist(centers, x).argmin(dim=1)]
    centers = merge_diverse_centers(
        previous_centers,
        new_centers,
        args.maximum_regions,
        args.region_bank_minimum_distance,
    )
    labels = assign_nearest_region(x, centers)
    return {
        "x": x.detach().cpu(),
        "log_contribution": logc.detach().cpu(),
        "region_index": labels.detach().cpu(),
        "centers": centers.detach().cpu(),
        "new_centers": new_centers.detach().cpu(),
        "previous_center_count": (
            0 if previous_centers is None else len(previous_centers)
        ),
        "active_center_count": len(centers),
        "condition": torch.tensor(args.condition, dtype=torch.float32),
    }


@torch.no_grad()
def build_cache(
    adaptive, proposal, benchmark, groups, centers, args, device, seed, label,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    xs, logfs, logrs, regions = [], [], [], []
    for _ in range(groups):
        group_x, group_logf, _, group_logr, _ = defensive_sample(
            adaptive,
            proposal,
            benchmark,
            args.points_per_group,
            args,
            device,
            centers=centers,
        )
        xs.append(group_x)
        logfs.append(group_logf)
        logrs.append(group_logr)
        regions.append(assign_nearest_region(group_x, centers))
    result = {
        "x": torch.stack(xs),
        "logf": torch.stack(logfs),
        "logr": torch.stack(logrs),
        "region": torch.stack(regions),
        "label": label,
    }
    print(
        f"{label}: groups={groups} points/group={args.points_per_group:,} "
        f"total={groups * args.points_per_group:,} "
        f"regional-share={args.regional_sample_fraction:.3f} "
        f"box-halfwidth={args.regional_box_halfwidth:.4f}",
        flush=True,
    )
    return result


def cache_objective(model, cache, indices, centers, args, smooth):
    x = cache["x"][indices]
    logf = cache["logf"][indices].double()
    logr = cache["logr"][indices].double()
    groups, points, _ = x.shape
    cond = fixed_condition(args, groups * points, x.device)
    flat_x = x.reshape(groups * points, DIM)
    logq = model.log_prob(flat_x, cond).reshape(-1).double()
    logc = 2.0 * logf.reshape(-1) - logq - logr.reshape(-1)
    if "region" in cache:
        region = cache["region"][indices].reshape(-1)
    else:
        region = assign_nearest_region(flat_x.detach(), centers)
    log_global, log_regions = regional_log_moments(
        logc, region, len(centers)
    )
    objective, log_regional_max = combined_log_objective(
        log_global,
        log_regions,
        args.regional_weight,
        args.regional_temperature if smooth else 0.0,
    )
    regional_fraction = torch.exp(log_regional_max - log_global)
    return objective, log_global, log_regional_max, regional_fraction


@torch.no_grad()
def validate(model, cache, centers, args):
    model.eval()
    indices = torch.arange(len(cache["x"]), device=cache["x"].device)
    objective, global_m2, regional_max, fraction = cache_objective(
        model, cache, indices, centers, args, smooth=False
    )
    return {
        "combined_log_objective": float(objective),
        "log_m2": float(global_m2),
        "m2": math.exp(min(float(global_m2), 700.0)),
        "maximum_regional_log_m2": float(regional_max),
        "maximum_regional_fraction": float(fraction),
    }


def train_round(
    model, train_cache, validation_cache, centers, reference_parameters,
    args, round_index,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    before = validate(model, validation_cache, centers, args)
    best = before
    best_state = cpu_state(model)
    best_step = 0
    patience = 0
    history = []
    generator = torch.Generator(device=train_cache["x"].device)
    generator.manual_seed(args.seed + 1_000_000 + round_index)
    print(
        f"REGIONAL M2 ROUND {round_index}: initial logM2={before['log_m2']:.6f} "
        f"max-region-fraction={before['maximum_regional_fraction']:.6f} "
        f"score={before['combined_log_objective']:.6f}",
        flush=True,
    )

    for start in range(0, args.steps_per_round, args.check_interval):
        block = min(args.check_interval, args.steps_per_round - start)
        train_losses = []
        model.train()
        for _ in range(block):
            indices = torch.randint(
                0,
                len(train_cache["x"]),
                (args.group_batch,),
                generator=generator,
                device=train_cache["x"].device,
            )
            objective, _, _, _ = cache_objective(
                model, train_cache, indices, centers, args, smooth=True
            )
            loss = objective + args.anchor_penalty * parameter_anchor(
                model, reference_parameters
            )
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        step = start + block
        current = validate(model, validation_cache, centers, args)
        threshold = args.minimum_improvement * max(
            abs(best["combined_log_objective"]), 1.0
        )
        if current["combined_log_objective"] < (
            best["combined_log_objective"] - threshold
        ):
            best = current
            best_state = cpu_state(model)
            best_step = step
            patience = 0
        else:
            patience += 1
        row = {
            "step": step,
            "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
            **current,
            "best_combined_log_objective": best["combined_log_objective"],
        }
        history.append(row)
        print(
            f"step {step:4d}/{args.steps_per_round}: "
            f"train={row['train_loss']:.6f} logM2={current['log_m2']:.6f} "
            f"max-region={current['maximum_regional_fraction']:.6f} "
            f"score={current['combined_log_objective']:.6f} "
            f"patience={patience}/{args.patience}",
            flush=True,
        )
        if patience >= args.patience:
            break

    model.load_state_dict(best_state, strict=True)
    after = validate(model, validation_cache, centers, args)
    return {
        "before": before,
        "after": after,
        "best_step": best_step,
        "history": history,
    }


@torch.no_grad()
def audit_model(adaptive, model, benchmark, args, device, round_index):
    milestones = sorted(set(args.audit_milestones + [args.audit_evals]))
    rows = []
    model.eval()
    for replicate in range(args.audit_replicates):
        seed = args.seed + 5_000_000 + 100_000 * round_index + replicate
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        pieces = []
        sampled = 0
        milestone_index = 0
        maximum_logw = -math.inf
        logsumw = -math.inf
        started = time.perf_counter()
        while sampled < args.audit_evals:
            target = milestones[milestone_index]
            count = min(args.chunk, target - sampled)
            cond = fixed_condition(args, count, device)
            x = adaptive.sample_from_model(model, count, cond)
            logq = model.log_prob(x, cond).reshape(-1)
            logf = adaptive.target_log_integrand(benchmark, x, cond).reshape(-1)
            logw = (logf - logq).double().cpu()
            pieces.append(logw)
            maximum_logw = max(maximum_logw, float(logw.max()))
            chunk_sum = float(torch.logsumexp(logw, dim=0))
            logsumw = float(np.logaddexp(logsumw, chunk_sum))
            sampled += count
            if sampled == target:
                all_logw = torch.cat(pieces)
                logm2 = float(torch.logsumexp(2.0 * all_logw, dim=0)) - math.log(sampled)
                logi = float(torch.logsumexp(all_logw, dim=0)) - math.log(sampled)
                rows.append({
                    "round": round_index,
                    "replicate": replicate,
                    "samples": sampled,
                    "first_moment": math.exp(min(logi, 700.0)),
                    "log_m2": logm2,
                    "second_moment": math.exp(min(logm2, 700.0)),
                    "normalized_log_m2": logm2 - 2.0 * logi,
                    "maximum_weight_share": math.exp(maximum_logw - logsumw),
                    "seconds": time.perf_counter() - started,
                })
                if sampled == args.audit_evals:
                    print(
                        f"audit round={round_index} replicate="
                        f"{replicate + 1}/{args.audit_replicates} "
                        f"I={math.exp(min(logi, 700.0)):.7f} "
                        f"M2={math.exp(min(logm2, 700.0)):.6g} "
                        f"max-share={math.exp(maximum_logw - logsumw):.4g}",
                        flush=True,
                    )
                milestone_index += 1
    frame = pd.DataFrame(rows)
    summary = frame.groupby("samples", as_index=False).agg(
        median_m2=("second_moment", "median"),
        q90_m2=("second_moment", lambda value: value.quantile(0.90)),
        maximum_m2=("second_moment", "max"),
        median_maximum_weight_share=("maximum_weight_share", "median"),
        q90_maximum_weight_share=("maximum_weight_share", lambda value: value.quantile(0.90)),
        mean_integral=("first_moment", "mean"),
    )
    return frame, summary


def save_checkpoint(
    path, source_checkpoint, model, args, rounds, region_centers,
):
    checkpoint = {
        **source_checkpoint,
        "model_state_dict": cpu_state(model),
        "single_condition_regional_m2": {
            "condition": list(args.condition),
            "regional_weight": args.regional_weight,
            "regions": args.regions,
            "rounds": rounds,
            "training_access": "f(x|c) only",
            "exact_target_sampler_used": False,
        },
        "regional_m2_region_centers": region_centers.detach().cpu(),
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

    adaptive, model, source_checkpoint, benchmark = local.load_model(
        args.checkpoint, device
    )
    model.to(device)
    reference_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    initial_data = None
    if args.initial_discovery_data is not None:
        try:
            initial_data = torch.load(
                args.initial_discovery_data,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            initial_data = torch.load(args.initial_discovery_data, map_location="cpu")

    def json_value(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        return value

    config = {key: json_value(value) for key, value in vars(args).items()}
    config.update({
        "benchmark": benchmark,
        "objective": "M2 + C * maximum regional M2",
        "training_access": "f(x|c) only",
        "exact_target_sampler_used": False,
    })
    with (args.output / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2)

    region_bank = None
    checkpoint_centers = source_checkpoint.get("regional_m2_region_centers")
    if checkpoint_centers is not None:
        region_bank = checkpoint_centers.to(
            device=device, dtype=torch.float32
        )
    for region_path in args.initial_region_data:
        try:
            region_data = torch.load(
                region_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            region_data = torch.load(region_path, map_location="cpu")
        region_bank = merge_diverse_centers(
            region_bank,
            region_data["centers"].to(device=device, dtype=torch.float32),
            args.maximum_regions,
            args.region_bank_minimum_distance,
        )
    if region_bank is not None:
        print(
            f"INITIAL CUMULATIVE REGION BANK: {len(region_bank)} centers",
            flush=True,
        )

    round_results = []
    audit_frames = []
    audit_summaries = []
    initial_frame, initial_summary = audit_model(
        adaptive, model, benchmark, args, device, round_index=-1
    )
    audit_frames.append(initial_frame)
    initial_summary.insert(0, "round", -1)
    audit_summaries.append(initial_summary)
    print("\nINITIAL FROZEN M2 AUDIT")
    print(initial_summary.to_string(index=False), flush=True)

    for round_index in range(args.rounds):
        round_dir = args.output / f"round_{round_index:02d}"
        round_dir.mkdir(exist_ok=True)
        proposal = freeze_copy(model, device)
        discovery = discover_regions(
            adaptive,
            proposal,
            benchmark,
            args,
            device,
            round_index,
            initial_data=initial_data,
            previous_centers=region_bank,
        )
        torch.save(discovery, round_dir / "regional_discovery.pt")
        centers = discovery["centers"].to(device)
        region_bank = centers
        print(
            f"CUMULATIVE REGION BANK: previous="
            f"{discovery['previous_center_count']} new="
            f"{len(discovery['new_centers'])} active="
            f"{discovery['active_center_count']}",
            flush=True,
        )

        train_cache = build_cache(
            adaptive,
            proposal,
            benchmark,
            args.train_groups,
            centers,
            args,
            device,
            args.seed + 2_000_000 + 2 * round_index,
            f"train round={round_index}",
        )
        validation_cache = build_cache(
            adaptive,
            proposal,
            benchmark,
            args.validation_groups,
            centers,
            args,
            device,
            args.seed + 2_000_001 + 2 * round_index,
            f"validation round={round_index}",
        )
        result = train_round(
            model,
            train_cache,
            validation_cache,
            centers,
            reference_parameters,
            args,
            round_index,
        )
        pd.DataFrame(result["history"]).to_csv(
            round_dir / "training_history.csv", index=False
        )
        round_results.append({
            "round": round_index,
            "best_step": result["best_step"],
            **{f"before_{key}": value for key, value in result["before"].items()},
            **{f"after_{key}": value for key, value in result["after"].items()},
        })
        save_checkpoint(
            round_dir / "regional_m2_model.pt",
            source_checkpoint,
            model,
            args,
            round_results,
            region_bank,
        )

        audit_frame, audit_summary = audit_model(
            adaptive, model, benchmark, args, device, round_index=round_index
        )
        audit_frames.append(audit_frame)
        audit_summary.insert(0, "round", round_index)
        audit_summaries.append(audit_summary)
        print(f"\nFROZEN M2 AUDIT AFTER ROUND {round_index}")
        print(audit_summary.to_string(index=False), flush=True)
        pd.concat(audit_frames, ignore_index=True).to_csv(
            args.output / "audit_runs.csv", index=False
        )
        pd.concat(audit_summaries, ignore_index=True).to_csv(
            args.output / "audit_summary.csv", index=False
        )
        pd.DataFrame(round_results).to_csv(
            args.output / "round_summary.csv", index=False
        )

    save_checkpoint(
        args.output / "single_condition_regional_m2_model.pt",
        source_checkpoint,
        model,
        args,
        round_results,
        region_bank,
    )
    print("\nSINGLE-CONDITION REGIONAL M2 SUMMARY")
    print(pd.DataFrame(round_results).to_string(index=False))
    print(
        f"\nModel: {(args.output / 'single_condition_regional_m2_model.pt').resolve()}"
    )


if __name__ == "__main__":
    main()
