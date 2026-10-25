#!/usr/bin/env python3
"""Residual-adaptive mixture-of-copula annealing on the irregular 100D target."""

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

import run_irregular_mixture_100d as irregular  # noqa: E402
from nf.conditional_mixture import (  # noqa: E402
    ConditionalMixtureProposal,
    cluster_worst_conditions,
    subset_condition_cache,
)


adaptive = irregular.adaptive
BENCHMARK = irregular.BENCHMARK


def weighted_nll(model, cache, condition_chunk=8):
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, cache["x"].shape[0], condition_chunk):
            x = cache["x"][start : start + condition_chunk]
            cond = cache["cond"][start : start + condition_chunk]
            alpha = cache["alpha"][start : start + condition_chunk]
            b, p, d = x.shape
            logq = model.log_prob(x.reshape(b * p, d), cond.reshape(b * p, -1)).reshape(b, p)
            total += float((-(alpha * logq).sum(1)).double().sum().item())
            count += b
    return total / max(count, 1)


def train_mixture_on_cache(model, train_cache, validation_cache, steps, check_interval, lr):
    optimizer = adaptive.make_adam(model.parameters(), lr=lr)
    best_nll = weighted_nll(model, validation_cache)
    best_state = adaptive.cpu_state_dict(model)
    best_step = 0
    history = []
    patience = 0
    points = train_cache["x"].shape[1]
    condition_batch = max(1, min(8, 4096 // points))

    for start_step in range(0, steps, check_interval):
        block = min(check_interval, steps - start_step)
        losses = []
        model.train()
        for _ in range(block):
            index = torch.randint(
                0, train_cache["x"].shape[0], (condition_batch,),
                device=train_cache["x"].device,
            )
            x = train_cache["x"][index]
            cond = train_cache["cond"][index]
            alpha = train_cache["alpha"][index]
            b, p, d = x.shape
            logq = model.log_prob(
                x.reshape(b * p, d), cond.reshape(b * p, -1)
            ).reshape(b, p)
            loss = (-(alpha * logq).sum(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

        step = start_step + block
        val = weighted_nll(model, validation_cache)
        if val < best_nll - 1e-5:
            best_nll = val
            best_state = adaptive.cpu_state_dict(model)
            best_step = step
            patience = 0
        else:
            patience += 1
        row = {
            "step": step, "train_nll": sum(losses) / len(losses),
            "val_nll": val, "best_nll": best_nll,
        }
        history.append(row)
        print(
            f"mixture step {step:5d}/{steps} K={model.n_components} "
            f"train={row['train_nll']:.6f} val={val:.6f} "
            f"best={best_nll:.6f} patience={patience}/2"
        )
        if patience >= 2:
            break

    adaptive.load_cpu_state(model, best_state)
    return {"best_step": best_step, "val_nll": best_nll, "history": history}


@torch.no_grad()
def residual_source_component(model, cache):
    """Choose the component closest to the worst weighted residual samples."""
    alpha = cache["alpha"]
    per_condition = max(1, min(8, alpha.shape[1]))
    selected = torch.topk(alpha, per_condition, dim=1).indices
    gather_x = selected[..., None].expand(-1, -1, cache["x"].shape[-1])
    gather_c = selected[..., None].expand(-1, -1, cache["cond"].shape[-1])
    x = torch.gather(cache["x"], 1, gather_x).reshape(-1, cache["x"].shape[-1])
    cond = torch.gather(cache["cond"], 1, gather_c).reshape(-1, cache["cond"].shape[-1])
    score = model.component_log_probs(x, cond).mean(0)
    return int(score.argmax().item())


def residual_training_set(cache, tail_points_per_condition=16):
    """Extract a weighted sample concentrated on proposal-undercovered mass."""
    count = min(int(tail_points_per_condition), cache["alpha"].shape[1])
    values, indices = torch.topk(cache["alpha"], count, dim=1)
    gather_x = indices[..., None].expand(-1, -1, cache["x"].shape[-1])
    gather_c = indices[..., None].expand(-1, -1, cache["cond"].shape[-1])
    x = torch.gather(cache["x"], 1, gather_x).reshape(-1, cache["x"].shape[-1])
    cond = torch.gather(cache["cond"], 1, gather_c).reshape(-1, cache["cond"].shape[-1])
    weights = values.reshape(-1)
    weights = weights / weights.sum()
    return x, cond, weights


@torch.no_grad()
def attach_log_densities(
    cache, source_proposal, temperature, uniform_fraction, chunk_size=65536,
):
    """Attach log f_beta and the exact frozen sampling density to a cache."""
    flat_x = cache["x"].reshape(-1, cache["x"].shape[-1])
    flat_cond = cache["cond"].reshape(-1, cache["cond"].shape[-1])
    log_target = []
    log_source = []
    for start in range(0, flat_x.shape[0], chunk_size):
        x = flat_x[start : start + chunk_size]
        cond = flat_cond[start : start + chunk_size]
        log_f = adaptive.target_log_integrand(BENCHMARK, x, cond).reshape(-1)
        log_target.append(float(temperature) * log_f)
        if source_proposal is None:
            log_source.append(torch.zeros_like(log_f))
        else:
            log_q = source_proposal.log_prob(x, cond).reshape(-1)
            log_r = torch.logaddexp(
                log_q + math.log1p(-uniform_fraction),
                torch.full_like(log_q, math.log(uniform_fraction)),
            )
            log_source.append(log_r)
    shape = cache["alpha"].shape
    cache["log_target"] = torch.cat(log_target).reshape(shape).detach()
    cache["log_source"] = torch.cat(log_source).reshape(shape).detach()
    cache["uniform_fraction"] = float(uniform_fraction)
    return cache


def defensive_log_prob(model, x, cond, uniform_fraction):
    """Evaluate the exact NF/uniform proposal density used for sampling."""
    log_q = model.log_prob(x, cond).reshape(-1)
    if not 0.0 < uniform_fraction < 1.0:
        return log_q
    return torch.logaddexp(
        log_q + math.log1p(-uniform_fraction),
        torch.full_like(log_q, math.log(uniform_fraction)),
    )


def split_cache_points(cache, validation_fraction=0.20):
    """Deterministically split every condition's samples into train and validation."""
    points = cache["x"].shape[1]
    validation_points = max(1, int(round(points * validation_fraction)))
    train_points = points - validation_points
    if train_points < 1:
        raise ValueError("cache needs at least two points per condition")
    train = {}
    validation = {}
    for key, value in cache.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == (cache["x"].shape[0], points):
            train[key] = value[:, :train_points]
            validation[key] = value[:, train_points:]
        else:
            train[key] = value
            validation[key] = value
    return train, validation


@torch.no_grad()
def log_m2_diagnostic(model, cache, condition_chunk=8):
    """Return per-condition log empirical second moments on a frozen cache."""
    values = []
    for start in range(0, cache["x"].shape[0], condition_chunk):
        x = cache["x"][start : start + condition_chunk]
        cond = cache["cond"][start : start + condition_chunk]
        b, p, d = x.shape
        log_q = defensive_log_prob(
            model, x.reshape(b * p, d), cond.reshape(b * p, -1),
            cache["uniform_fraction"],
        ).reshape(b, p)
        log_term = 2.0 * cache["log_target"][start : start + b] - log_q - cache["log_source"][start : start + b]
        values.append(torch.logsumexp(log_term.double(), dim=1) - math.log(p))
    return torch.cat(values)


def finetune_cluster_second_moment(
    model, component_index, train_cache, validation_cache, steps=600,
    batch_conditions=8, points_per_condition=256, lr=1e-4,
    minimum_improvement=1e-3, acceptance_baseline=None,
    rejection_state=None, patience_limit=5, condition_cvar_fraction=0.25,
):
    """Optimize a specialist and its gate, accepting only held-out M2 gains."""
    component_index = int(component_index)
    saved_state = (
        rejection_state if rejection_state is not None
        else adaptive.cpu_state_dict(model)
    )
    original_requires_grad = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    for index, component in enumerate(model.components):
        enabled = index == component_index
        for parameter in component.parameters():
            parameter.requires_grad_(enabled)
    for parameter in model.gate.parameters():
        parameter.requires_grad_(True)

    parameters = list(model.components[component_index].parameters()) + list(model.gate.parameters())
    optimizer = adaptive.make_adam(parameters, lr=lr)
    current_values = log_m2_diagnostic(model, validation_cache)
    current = float(current_values.mean().item())
    before = current if acceptance_baseline is None else float(acceptance_baseline)
    best = current
    best_state = (
        adaptive.cpu_state_dict(model)
        if current < before - minimum_improvement else None
    )
    patience = 0

    for start_step in range(0, steps, 100):
        block = min(100, steps - start_step)
        model.train()
        for _ in range(block):
            condition_index = torch.randint(
                0, train_cache["x"].shape[0],
                (min(batch_conditions, train_cache["x"].shape[0]),),
                device=train_cache["x"].device,
            )
            available_points = train_cache["x"].shape[1]
            point_index = torch.randint(
                0, available_points,
                (condition_index.shape[0], min(points_per_condition, available_points)),
                device=train_cache["x"].device,
            )
            x = train_cache["x"][condition_index]
            cond = train_cache["cond"][condition_index]
            gather_x = point_index[..., None].expand(-1, -1, x.shape[-1])
            gather_c = point_index[..., None].expand(-1, -1, cond.shape[-1])
            x = torch.gather(x, 1, gather_x)
            cond = torch.gather(cond, 1, gather_c)
            log_f = torch.gather(train_cache["log_target"][condition_index], 1, point_index)
            log_r = torch.gather(train_cache["log_source"][condition_index], 1, point_index)
            b, p, d = x.shape
            log_q = defensive_log_prob(
                model, x.reshape(b * p, d), cond.reshape(b * p, -1),
                train_cache["uniform_fraction"],
            ).reshape(b, p)
            # This is the gradient of log M2, estimated under the frozen
            # exploration density. Softmax normalization makes it stable while
            # preserving the per-condition second-moment descent direction.
            importance = torch.softmax(
                (2.0 * log_f - log_q.detach() - log_r).double(), dim=1
            ).to(log_q.dtype)
            condition_loss = -(importance * log_q).sum(dim=1)
            condition_log_m2 = (
                torch.logsumexp(
                    (2.0 * log_f - log_q.detach() - log_r).double(), dim=1
                ) - math.log(p)
            )
            tail_count = max(
                1, int(math.ceil(condition_cvar_fraction * condition_loss.shape[0]))
            )
            difficult = torch.topk(condition_log_m2, tail_count).indices
            loss = condition_loss[difficult].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()

        step = start_step + block
        model.eval()
        validation_values = log_m2_diagnostic(model, validation_cache)
        value = float(validation_values.mean().item())
        if value < best - minimum_improvement:
            best = value
            best_state = adaptive.cpu_state_dict(model)
            patience = 0
        else:
            patience += 1
        print(
            f"  M2 fine tune {step:4d}/{steps} component={component_index} "
            f"heldout logM2={value:.6f} best={best:.6f} before={before:.6f}"
        )
        if patience >= patience_limit:
            break

    accepted = best_state is not None and best < before - minimum_improvement
    adaptive.load_cpu_state(model, best_state if accepted else saved_state)
    for parameter, required in original_requires_grad.items():
        parameter.requires_grad_(required)
    print(
        f"  specialist {component_index} {'ACCEPTED' if accepted else 'REJECTED'}: "
        f"heldout logM2 {before:.6f} -> {best:.6f}"
    )
    return {"accepted": accepted, "before_log_m2": before, "after_log_m2": best}


@torch.no_grad()
def summarize_condition_log_m2(model, cache, quantile=0.95):
    """Summarize integration difficulty across an independent condition set."""
    values = log_m2_diagnostic(model, cache)
    return {
        "median": float(values.median().item()),
        "upper_quantile": float(torch.quantile(values, quantile).item()),
        "mean": float(values.mean().item()),
        "maximum": float(values.max().item()),
    }


def specialize_residual_component(
    model, component_index, cache, steps=600, batch_size=512, lr=3e-4,
    tail_points_per_condition=64,
):
    """Fit a newly spawned component directly to high-weight residual mass."""
    component = model.components[int(component_index)]
    x, cond, weights = residual_training_set(
        cache, tail_points_per_condition=tail_points_per_condition,
    )
    optimizer = adaptive.make_adam(component.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = adaptive.cpu_state_dict(component)

    component.train()
    for step in range(steps):
        index = torch.multinomial(weights, batch_size, replacement=True)
        loss = -component.log_prob(x[index], cond[index]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(component.parameters(), 10.0)
        optimizer.step()
        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_state = adaptive.cpu_state_dict(component)
        if (step + 1) % 150 == 0:
            print(
                f"  residual specialization {step + 1:4d}/{steps} "
                f"component={component_index} NLL={value:.6f} best={best_loss:.6f}"
            )

    adaptive.load_cpu_state(component, best_state)
    # Give the specialist enough initial gate mass to affect the next fit,
    # without allowing it to dominate conditions it has not yet learned.
    with torch.no_grad():
        model.gate[-1].bias[int(component_index)] = -1.0
    return best_loss


@torch.no_grad()
def initialize_specialist_gate(model, specialist_index, source_index, relative_mass=0.05):
    """Give a specialist a small condition-dependent copy of its source gate."""
    if not 0.0 < relative_mass < 1.0:
        raise ValueError("relative_mass must lie strictly between zero and one")
    final = model.gate[-1]
    final.weight[int(specialist_index)].copy_(final.weight[int(source_index)])
    final.bias[int(specialist_index)].copy_(
        final.bias[int(source_index)] + math.log(relative_mass)
    )


def warm_start_specialist_gate(
    model, assignments, background_conditions=None, steps=200, lr=1e-3,
    retention_weight=0.1,
):
    """Teach the conditional gate which failure region each specialist owns."""
    if not assignments:
        return
    conditions = torch.cat([item[0] for item in assignments], dim=0)
    labels = torch.cat([
        torch.full(
            (item[0].shape[0],), int(item[1]), dtype=torch.long,
            device=item[0].device,
        )
        for item in assignments
    ])
    parameters = list(model.gate.parameters())
    teacher = None
    if background_conditions is not None:
        with torch.no_grad():
            teacher = model.log_gate(background_conditions).detach()
    optimizer = adaptive.make_adam(parameters, lr=lr)
    best_loss = float("inf")
    best_state = adaptive.cpu_state_dict(model.gate)
    model.gate.train()
    for step in range(steps):
        index = torch.randint(0, conditions.shape[0], (min(256, conditions.shape[0]),),
                              device=conditions.device)
        loss = torch.nn.functional.nll_loss(model.log_gate(conditions[index]), labels[index])
        if teacher is not None:
            background_index = torch.randint(
                0, background_conditions.shape[0],
                (min(256, background_conditions.shape[0]),),
                device=background_conditions.device,
            )
            student_log = model.log_gate(background_conditions[background_index])
            teacher_log = teacher[background_index]
            retention = torch.nn.functional.kl_div(
                student_log, teacher_log.exp(), reduction="batchmean",
            )
            loss = loss + retention_weight * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_state = adaptive.cpu_state_dict(model.gate)
    adaptive.load_cpu_state(model.gate, best_state)
    print(f"  specialist gate warm start: assignments={conditions.shape[0]} NLL={best_loss:.6f}")


@torch.no_grad()
def recycle_component_for_residual(model, source_index, cache, excluded=()):
    """Reuse the least-active component when the configured mixture is full."""
    mean_gate = torch.exp(
        model.log_gate(cache["cond"].reshape(-1, cache["cond"].shape[-1]))
    ).mean(0)
    candidates = [
        index for index in range(model.n_components)
        if index not in set(int(value) for value in excluded)
    ]
    if not candidates:
        raise RuntimeError("no mixture component is available for recycling")
    target_index = min(candidates, key=lambda index: float(mean_gate[index].item()))
    source_state = adaptive.cpu_state_dict(model.components[int(source_index)])
    adaptive.load_cpu_state(model.components[target_index], source_state)
    for parameter in model.components[target_index].parameters():
        parameter.add_(2e-3 * torch.randn_like(parameter))
    initialize_specialist_gate(model, target_index, source_index)
    return target_index


def save_stage(output, model, beta, stage, history, args):
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "benchmark": BENCHMARK,
            "dimension": irregular.DIMENSION,
            "beta": beta,
            "stage": stage,
            "components": model.n_components,
            "model_state_dict": adaptive.cpu_state_dict(model),
            "graph": irregular.irregular_dependency_superset(irregular.DIMENSION),
            "history": history,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        output / "latest_mixture_checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(output / "annealing_history.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--steps-per-stage", type=int, default=1200)
    parser.add_argument("--check-interval", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--repair-sample-growth", type=float, default=2.0)
    parser.add_argument("--repair-max-point-multiplier", type=int, default=8)
    parser.add_argument("--specialists-per-repair", type=int, default=3)
    parser.add_argument("--worst-condition-fraction", type=float, default=0.10)
    parser.add_argument("--specialist-gate-steps", type=int, default=0)
    parser.add_argument("--specialist-pretrain-steps", type=int, default=0)
    parser.add_argument("--specialist-m2-steps", type=int, default=1800)
    parser.add_argument("--specialist-m2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--specialist-m2-minimum-improvement", type=float, default=1e-3)
    parser.add_argument("--specialist-m2-patience", type=int, default=5)
    parser.add_argument("--condition-cvar-fraction", type=float, default=0.25)
    parser.add_argument("--repair-validation-conditions", type=int, default=512)
    parser.add_argument("--repair-validation-points", type=int, default=512)
    parser.add_argument("--repair-validation-quantile", type=float, default=0.95)
    parser.add_argument("--repair-validation-minimum-improvement", type=float, default=1e-3)
    parser.add_argument("--repair-validation-median-tolerance", type=float, default=5e-3)
    parser.add_argument("--max-stages", type=int, default=200)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/irregular_mixture_100_adaptive_mixture"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    irregular.register_benchmark()
    adaptive.DIM = irregular.DIMENSION
    adaptive.configure_speed_profile("balanced")
    adaptive.TRAIN_BATCH_SIZE = 4096
    adaptive.COMPILE_TRAINING_KERNELS = False
    adaptive.seed_everything(args.seed)

    checkpoint = None
    if args.resume is not None:
        # This is a trusted, locally generated stage checkpoint containing
        # metadata as well as tensors. PyTorch 2.6 defaults to tensor-only
        # loading, which rejects the Path metadata in older checkpoints.
        try:
            checkpoint = torch.load(
                args.resume, map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch versions predating weights_only.
            checkpoint = torch.load(args.resume, map_location="cpu")
    component_count = (
        int(checkpoint["components"])
        if checkpoint is not None else args.initial_components
    )
    graph = irregular.irregular_dependency_superset(irregular.DIMENSION)
    components = [
        adaptive.make_real_model(graph, None, device)
        for _ in range(component_count)
    ]
    model = ConditionalMixtureProposal(
        components, cond_dim=adaptive.COND_DIM,
        max_components=args.max_components,
    ).to(device)
    if checkpoint is None:
        # Break initial symmetry without changing the model family.
        with torch.no_grad():
            for index, component in enumerate(model.components[1:], start=1):
                for parameter in component.parameters():
                    parameter.add_(1e-3 * index * torch.randn_like(parameter))
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(
            f"RESUMED mixture checkpoint at beta={checkpoint['beta']:.6f}, "
            f"stage={checkpoint['stage']}, K={component_count}"
        )

    beta = float(checkpoint["beta"]) if checkpoint is not None else 0.0
    stage = int(checkpoint["stage"]) + 1 if checkpoint is not None else 0
    repairs = 0
    base_probe_points = adaptive.TEMPERATURE_PROBE_POINTS
    history = list(checkpoint.get("history", [])) if checkpoint is not None else []
    proposal = copy.deepcopy(model).to(device).eval() if checkpoint is not None else None
    if proposal is not None:
        for parameter in proposal.parameters():
            parameter.requires_grad_(False)
    while beta < 1.0 - 1e-12 and stage < args.max_stages:
        # A failed repair is evidence that the ordinary pilot missed important
        # tail structure. Increase samples per condition geometrically, capped
        # to keep memory and runtime predictable.
        repair_point_multiplier = min(
            args.repair_max_point_multiplier,
            max(1, int(round(args.repair_sample_growth ** repairs))),
        )
        # Acceptance always uses the same fresh probe size. Increasing the
        # discovery budget must not make a fixed raw-ESS threshold easier.
        adaptive.TEMPERATURE_PROBE_POINTS = base_probe_points
        next_beta, uniform_fraction, passed = adaptive.select_next_temperature(
            BENCHMARK, proposal, beta, device, 70000 + 100 * stage
        )
        if proposal is not None and not passed:
            repairs += 1
            repair_point_multiplier = min(
                args.repair_max_point_multiplier,
                max(1, int(round(args.repair_sample_growth ** repairs))),
            )
            adaptive.TEMPERATURE_PROBE_POINTS = base_probe_points
            if repairs > args.max_repairs:
                raise RuntimeError(
                    f"Mixture annealing stalled at beta={beta:.6f} with "
                    f"K={model.n_components} after {args.max_repairs} "
                    "residual-specialization rounds"
                )
            probe = adaptive.build_weighted_cache(
                BENCHMARK, proposal, 512,
                adaptive.MARGINAL_ADAPTIVE_POINTS * repair_point_multiplier,
                device, 30303 + stage, "residual component discovery",
                beta, uniform_fraction,
            )
            attach_log_densities(
                probe, proposal, beta, uniform_fraction,
            )
            # This cache contains entirely new conditions and is never used
            # for specialist fitting or early stopping. It authorizes or
            # rejects the complete repair round.
            external_validation = adaptive.build_weighted_cache(
                BENCHMARK, proposal, args.repair_validation_conditions,
                args.repair_validation_points, device, 40404 + 100 * stage,
                "external condition repair validation", beta, uniform_fraction,
            )
            attach_log_densities(
                external_validation, proposal, beta, uniform_fraction,
            )
            round_state = adaptive.cpu_state_dict(model)
            external_before = summarize_condition_log_m2(
                model, external_validation,
                quantile=args.repair_validation_quantile,
            )
            print(
                "EXTERNAL CONDITION BASELINE: "
                f"median logM2={external_before['median']:.6f} "
                f"q{int(100 * args.repair_validation_quantile):02d}="
                f"{external_before['upper_quantile']:.6f}"
            )
            groups, condition_ess = cluster_worst_conditions(
                probe, clusters=args.specialists_per_repair,
                worst_fraction=args.worst_condition_fraction,
            )
            print(
                "CLUSTERED RESIDUAL REPAIR: "
                f"worst={sum(group.numel() for group in groups)} "
                f"clusters={len(groups)} "
                f"minimum normalized ESS={float(condition_ess.min().item()):.6g}"
            )
            protected = set()
            accepted_specialists = 0
            for cluster_index, group in enumerate(groups):
                cluster_cache = subset_condition_cache(probe, group)
                cluster_train, cluster_validation = split_cache_points(cluster_cache)
                baseline_log_m2 = float(
                    log_m2_diagnostic(model, cluster_validation).mean().item()
                )
                pre_repair_state = adaptive.cpu_state_dict(model)
                source = residual_source_component(model, cluster_cache)
                if model.n_components < args.max_components:
                    new_index = model.spawn_component(source)
                    initialize_specialist_gate(model, new_index, source)
                    action = "SPAWNED"
                else:
                    new_index = recycle_component_for_residual(
                        model, source, cluster_cache,
                        excluded=protected | {source},
                    )
                    action = "RECYCLED"
                protected.add(new_index)
                print(
                    f"{action} SPECIALIST {new_index} from component {source} "
                    f"for cluster {cluster_index + 1}/{len(groups)} "
                    f"({group.numel()} conditions) at held beta={beta:.6f}"
                )
                if args.specialist_pretrain_steps > 0:
                    specialize_residual_component(
                        model, new_index, cluster_train,
                        steps=args.specialist_pretrain_steps,
                        batch_size=512,
                        lr=max(args.learning_rate, 3e-4),
                        tail_points_per_condition=min(
                            256, max(32, cluster_train["x"].shape[1] // 32),
                        ),
                    )
                if args.specialist_gate_steps > 0:
                    warm_start_specialist_gate(
                        model, [(cluster_train["cond"][:, 0], new_index)],
                        steps=args.specialist_gate_steps,
                        background_conditions=probe["cond"][:, 0],
                        lr=max(args.learning_rate, 1e-3),
                    )
                m2_result = finetune_cluster_second_moment(
                    model, new_index, cluster_train, cluster_validation,
                    steps=args.specialist_m2_steps,
                    lr=args.specialist_m2_learning_rate,
                    minimum_improvement=args.specialist_m2_minimum_improvement,
                    acceptance_baseline=baseline_log_m2,
                    rejection_state=pre_repair_state,
                    patience_limit=args.specialist_m2_patience,
                    condition_cvar_fraction=args.condition_cvar_fraction,
                )
                accepted_specialists += int(m2_result["accepted"])
            print(
                f"CLUSTERED REPAIR ACCEPTANCE: {accepted_specialists}/"
                f"{len(groups)} specialists reduced held-out cluster M2"
            )
            external_after = summarize_condition_log_m2(
                model, external_validation,
                quantile=args.repair_validation_quantile,
            )
            upper_improvement = (
                external_before["upper_quantile"]
                - external_after["upper_quantile"]
            )
            median_change = external_after["median"] - external_before["median"]
            round_accepted = (
                accepted_specialists > 0
                and upper_improvement
                >= args.repair_validation_minimum_improvement
                and median_change
                <= args.repair_validation_median_tolerance
            )
            if not round_accepted:
                adaptive.load_cpu_state(model, round_state)
            print(
                f"EXTERNAL CONDITION REPAIR {'ACCEPTED' if round_accepted else 'REJECTED'}: "
                f"median {external_before['median']:.6f} -> "
                f"{external_after['median']:.6f}; "
                f"q{int(100 * args.repair_validation_quantile):02d} "
                f"{external_before['upper_quantile']:.6f} -> "
                f"{external_after['upper_quantile']:.6f}"
            )
            print(
                "SKIPPING GLOBAL LIKELIHOOD FIT: stalled-temperature repairs "
                "are controlled only by external-condition M2."
            )
            proposal = copy.deepcopy(model).to(device).eval()
            for parameter in proposal.parameters():
                parameter.requires_grad_(False)
            history.append({
                "stage": stage, "beta": beta,
                "components": model.n_components,
                "advanced": False, "repairs": repairs,
                "accepted_specialists": accepted_specialists,
                "external_repair_accepted": bool(round_accepted),
                "external_median_log_m2_before": external_before["median"],
                "external_median_log_m2_after": external_after["median"],
                "external_upper_log_m2_before": external_before["upper_quantile"],
                "external_upper_log_m2_after": external_after["upper_quantile"],
                "val_nll": math.nan, "best_step": 0,
            })
            save_stage(args.output, model, beta, stage, history, args)
            del probe, external_validation
            torch.cuda.empty_cache()
            stage += 1
            continue
        else:
            repairs = 0
            repair_point_multiplier = 1

        source_proposal = proposal
        points = (
            adaptive.UNIFORM_BOOTSTRAP_POINTS
            if source_proposal is None
            else adaptive.MARGINAL_ADAPTIVE_POINTS * repair_point_multiplier
        )
        if not passed:
            print(
                f"REPAIR SAMPLE BUDGET: {repair_point_multiplier}x points/condition "
                f"({points:,} points; probe={adaptive.TEMPERATURE_PROBE_POINTS:,})"
            )
        train_cache = adaptive.build_weighted_cache(
            BENCHMARK, source_proposal, adaptive.MARGINAL_TRAIN_CONDITIONS,
            points, device, 10101 + 100 * stage, f"mixture train beta={next_beta:.6f}",
            next_beta, uniform_fraction,
        )
        val_cache = adaptive.build_weighted_cache(
            BENCHMARK, source_proposal, adaptive.VAL_CONDITIONS,
            points, device, 20202 + 100 * stage, f"mixture val beta={next_beta:.6f}",
            next_beta, uniform_fraction,
        )
        result = train_mixture_on_cache(
            model, train_cache, val_cache, args.steps_per_stage,
            args.check_interval, args.learning_rate,
        )
        if passed:
            beta = next_beta
        proposal = copy.deepcopy(model).to(device).eval()
        for parameter in proposal.parameters():
            parameter.requires_grad_(False)
        history.append({
            "stage": stage, "beta": beta, "components": model.n_components,
            "advanced": bool(passed), "repairs": repairs,
            "val_nll": result["val_nll"], "best_step": result["best_step"],
        })
        save_stage(args.output, model, beta, stage, history, args)
        del train_cache, val_cache
        torch.cuda.empty_cache()
        stage += 1

    metrics = adaptive.evaluate_ess(
        model, BENCHMARK, device, n=adaptive.FINAL_ESS_SIZE,
        seed=adaptive.EVAL_SEED + 999,
    )
    with (args.output / "final_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print("\nADAPTIVE MIXTURE COMPLETE")
    print(f"beta={beta:.6f} K={model.n_components}")
    print(f"ESS={metrics['ess']:.6f} estimate={metrics['integral_estimate']:.12g}")


if __name__ == "__main__":
    main()
