#!/usr/bin/env python3
"""Automatically grow residual specialists for a heterogeneous NF mixture.

No structure count or benchmark labels are supplied.  At each round the
current deployed mixture is evaluated on independent local target clouds.
Points with the largest log second-moment contribution are clustered in the
full standardized 32-dimensional space with a data-derived neighborhood
scale.  One specialist is spawned for every stable residual cluster, mixture
weights are optimized on held-out samples, and growth stops when the held-out
log-M2 proxy ceases to improve.
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

import audit_heterogeneous_discovered_region_mixture_32d as mixture_audit  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402
import train_heterogeneous_discovered_region_specialists_32d as initial  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--clouds", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--residual-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-residual-points", type=int, default=2048)
    parser.add_argument("--minimum-cluster-points", type=int, default=128)
    parser.add_argument("--neighbor-rank", type=int, default=8)
    parser.add_argument("--cluster-scale", type=float, default=1.5)
    parser.add_argument("--maximum-new-specialists-per-round", type=int, default=4)
    parser.add_argument("--points-per-group", type=int, default=256)
    parser.add_argument("--marginal-steps", type=int, default=10000)
    parser.add_argument("--copula-steps", type=int, default=10000)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--weight-steps", type=int, default=1000)
    parser.add_argument("--weight-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--maximum-repair-weight", type=float, default=0.10,
        help="Maximum total mixture mass assigned to specialists from one round.",
    )
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = (
        args.rounds, args.maximum_residual_points, args.minimum_cluster_points,
        args.neighbor_rank, args.maximum_new_specialists_per_round,
        args.points_per_group, args.marginal_steps, args.copula_steps,
        args.check_interval, args.patience, args.weight_steps,
    )
    if min(counts) <= 0:
        parser.error("all budgets must be positive")
    if not 0 < args.residual_fraction < 1:
        parser.error("--residual-fraction must lie in (0,1)")
    if args.cluster_scale <= 0 or args.weight_learning_rate <= 0:
        parser.error("scales and learning rates must be positive")
    if not 0 < args.maximum_repair_weight < 1:
        parser.error("--maximum-repair-weight must lie in (0,1)")
    if not 0 <= args.minimum_relative_improvement < 1:
        parser.error("invalid minimum relative improvement")
    return args


def load_payload(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "heterogeneous_discovered_region_mixture_v1":
        raise ValueError("unsupported mixture format")
    return payload


def flatten_cloud(cache):
    return (
        cache["x"].reshape(-1, heterogeneous.DIMENSION),
        cache["cond"].reshape(-1, heterogeneous.CONDITION_DIMENSION),
    )


@torch.no_grad()
def component_logs(models, x, cond, chunk=2048):
    rows = []
    for start in range(0, len(x), chunk):
        stop = start + chunk
        rows.append(torch.stack([
            model.log_prob(x[start:stop], cond[start:stop]).reshape(-1).double()
            for model in models
        ], dim=1))
    return torch.cat(rows)


def safe_logq(log_components, weights, uniform_fraction):
    learned = torch.logsumexp(
        log_components + weights.double().log(), dim=1
    )
    return torch.logaddexp(
        learned + math.log1p(-uniform_fraction),
        torch.full_like(learned, math.log(uniform_fraction)),
    )


@torch.no_grad()
def cloud_statistics(models, weights, uniform_fraction, x, cond):
    logs = component_logs(models, x, cond)
    logq = safe_logq(logs, weights, uniform_fraction)
    logf = heterogeneous.heterogeneous_log_integrand(x, cond).reshape(-1).double()
    residual = 2.0 * logf - logq
    log_m2_proxy = torch.logsumexp(logf - logq, dim=0) - math.log(len(x))
    return {
        "component_logs": logs,
        "logq": logq,
        "logf": logf,
        "residual": residual,
        "log_m2_proxy": float(log_m2_proxy),
    }


def connected_components(adjacency):
    adjacency = adjacency.cpu().numpy()
    seen = np.zeros(len(adjacency), dtype=bool)
    groups = []
    for start in range(len(adjacency)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        group = []
        while stack:
            node = stack.pop()
            group.append(node)
            neighbors = np.flatnonzero(adjacency[node] & ~seen)
            if len(neighbors):
                seen[neighbors] = True
                stack.extend(neighbors.tolist())
        groups.append(np.asarray(group, dtype=np.int64))
    return groups


def discover_residual_clusters(x, residual, args):
    count = min(
        args.maximum_residual_points,
        max(args.minimum_cluster_points, int(math.ceil(
            args.residual_fraction * len(x)
        ))),
    )
    selected = torch.topk(residual, count).indices
    points = x[selected].double()
    center = x.double().median(dim=0).values
    q25 = torch.quantile(x.double(), 0.25, dim=0)
    q75 = torch.quantile(x.double(), 0.75, dim=0)
    scale = (q75 - q25).clamp_min(0.02)
    z = (points - center) / scale
    distance = torch.cdist(z, z)
    rank = min(args.neighbor_rank, len(z) - 1)
    kth = torch.topk(distance, rank + 1, largest=False).values[:, -1]
    epsilon = float(kth.median()) * args.cluster_scale
    adjacency = distance <= epsilon
    groups = connected_components(adjacency)
    stable = [
        selected[torch.as_tensor(group, device=selected.device)]
        for group in groups if len(group) >= args.minimum_cluster_points
    ]
    if not stable:
        stable = [selected]
    stable.sort(key=lambda idx: float(residual[idx].median()), reverse=True)
    stable = stable[:args.maximum_new_specialists_per_round]
    return stable, center, scale, epsilon


def validation_subset(x, residual, train_cluster, train_x, center, scale, args):
    selected_count = min(
        args.maximum_residual_points,
        max(args.minimum_cluster_points, int(math.ceil(
            args.residual_fraction * len(x)
        ))),
    )
    pool = torch.topk(residual, selected_count).indices
    centroid = ((train_x[train_cluster].double() - center) / scale).mean(dim=0)
    z = (x[pool].double() - center) / scale
    distance = (z - centroid).square().sum(dim=1)
    count = min(len(pool), max(args.minimum_cluster_points, len(train_cluster) // 4))
    return pool[torch.topk(distance, count, largest=False).indices]


def make_cache(x, cond, indices, points_per_group):
    x = x[indices]
    cond = cond[indices]
    effective = min(points_per_group, len(x))
    groups = len(x) // effective
    if groups < 1:
        raise RuntimeError("residual cluster cannot form a training group")
    count = groups * effective
    order = torch.randperm(len(x), device=x.device)[:count]
    x = x[order].reshape(groups, effective, -1)
    cond = cond[order].reshape(groups, effective, -1)
    alpha = torch.full(
        (groups, effective), 1.0 / effective,
        dtype=x.dtype, device=x.device,
    )
    return {"x": x, "cond": cond, "alpha": alpha}


def optimize_weights(component_log_prob, logf, base_weights, new_count,
                     uniform_fraction, args):
    """Optimize a protected repair gate without changing base proportions."""
    base_weights = base_weights.double() / base_weights.double().sum()
    repair_logits = torch.zeros(
        new_count, dtype=torch.float64, device=component_log_prob.device,
        requires_grad=True,
    )
    repair_fraction_logit = torch.tensor(
        -2.0, dtype=torch.float64, device=component_log_prob.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [repair_logits, repair_fraction_logit], lr=args.weight_learning_rate
    )
    best_loss = math.inf
    best = torch.cat((base_weights, torch.zeros_like(repair_logits))).detach()
    for step in range(1, args.weight_steps + 1):
        repair_fraction = (
            args.maximum_repair_weight * torch.sigmoid(repair_fraction_logit)
        )
        weights = torch.cat((
            (1.0 - repair_fraction) * base_weights,
            repair_fraction * torch.softmax(repair_logits, dim=0),
        ))
        logq = safe_logq(component_log_prob, weights, uniform_fraction)
        loss = torch.logsumexp(logf - logq, dim=0) - math.log(len(logf))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best = weights.detach().clone()
        if step % 200 == 0:
            print(
                f"mixture-weight step={step:4d}/{args.weight_steps} "
                f"logM2-proxy={value:.6f} best={best_loss:.6f}",
                flush=True,
            )
    return best / best.sum(), best_loss


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
    clouds = initial.load_clouds(args.clouds, device)
    train_x, train_cond = flatten_cloud(clouds["train"])
    val_x, val_cond = flatten_cloud(clouds["validation"])
    weights = weights.to(device=device, dtype=torch.float64)

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

    history = []
    for round_index in range(args.rounds):
        base_models_count = len(models)
        base_weights = weights.detach().clone()
        train_stats = cloud_statistics(
            models, weights, uniform_fraction, train_x, train_cond
        )
        val_stats = cloud_statistics(
            models, weights, uniform_fraction, val_x, val_cond
        )
        before = val_stats["log_m2_proxy"]
        clusters, center, scale, epsilon = discover_residual_clusters(
            train_x, train_stats["residual"], args
        )
        print("\n" + "#" * 88, flush=True)
        print(
            f"RESIDUAL ROUND {round_index}: components={len(models)} "
            f"held-out logM2 proxy={before:.6f} "
            f"clusters={len(clusters)} eps={epsilon:.6f}",
            flush=True,
        )
        new_models = []
        cluster_rows = []
        for cluster_index, train_indices in enumerate(clusters):
            val_indices = validation_subset(
                val_x, val_stats["residual"], train_indices, train_x,
                center, scale, args,
            )
            # Initialize from whichever current component already gives the
            # highest average density on this residual cluster.
            owner = int(
                train_stats["component_logs"][train_indices].mean(dim=0).argmax()
            )
            specialist = copy.deepcopy(models[owner]).to(device)
            for parameter in specialist.parameters():
                parameter.requires_grad_(True)
            train_cache = make_cache(
                train_x, train_cond, train_indices, args.points_per_group
            )
            validation_cache = make_cache(
                val_x, val_cond, val_indices, args.points_per_group
            )
            print(
                f"cluster={cluster_index} train={len(train_indices)} "
                f"validation={len(val_indices)} owner={owner} "
                f"median-residual="
                f"{float(train_stats['residual'][train_indices].median()):.6f}",
                flush=True,
            )
            marginal = adaptive.train_marginals_on_dataset(
                adaptive.get_marginal_bank(specialist),
                train_cache, validation_cache, device,
                args.seed + 10000 * (round_index + 1) + cluster_index,
            )
            copula = adaptive.train_bridge_copula_on_cache(
                specialist, train_cache, validation_cache,
                label=f"residual round {round_index} cluster {cluster_index}",
                optimize_joint=False,
            )
            specialist.eval()
            for parameter in specialist.parameters():
                parameter.requires_grad_(False)
            new_models.append(specialist)
            cluster_rows.append({
                "cluster": cluster_index,
                "owner": owner,
                "train_points": len(train_indices),
                "validation_points": len(val_indices),
                "median_residual": float(
                    train_stats["residual"][train_indices].median()
                ),
                "marginal_best_step": marginal["best_step"],
                "copula_best_step": copula["best_step"],
            })
        models.extend(new_models)
        new_count = len(new_models)
        val_logs = component_logs(models, val_x, val_cond)
        weights, after = optimize_weights(
            val_logs, val_stats["logf"], base_weights, new_count,
            uniform_fraction, args,
        )
        relative = 1.0 - math.exp(after - before)
        learned_update = any(
            row["marginal_best_step"] > 0 or row["copula_best_step"] > 0
            for row in cluster_rows
        )
        accepted = (
            learned_update
            and relative >= args.minimum_relative_improvement
        )
        if not accepted:
            # A rejected repair must not alter the deployed proposal.
            del models[base_models_count:]
            weights = base_weights
            after = before
            relative = 0.0
        round_row = {
            "round": round_index,
            "components_before": base_models_count,
            "components_after": len(models),
            "accepted": accepted,
            "clusters": cluster_rows,
            "clustering_epsilon": epsilon,
            "held_out_log_m2_before": before,
            "held_out_log_m2_after": after,
            "relative_m2_improvement": relative,
            "weights": weights.detach().cpu().tolist(),
        }
        history.append(round_row)
        print("\nRESIDUAL ROUND RESULT")
        print(json.dumps(round_row, indent=2))
        checkpoint = {
            **payload,
            "specialist_state_dicts": [cpu_state(model) for model in models],
            "mixture_weights": weights.detach().cpu(),
            "residual_repair_history": history,
            "residual_discovery": (
                "automatic connected components of high residual points in "
                "standardized 32D space"
            ),
        }
        round_path = args.output / f"residual_mixture_round_{round_index:02d}.pt"
        torch.save(checkpoint, round_path)
        if not accepted:
            print(
                "STOPPING: rejected repair was rolled back; the deployed "
                "mixture is unchanged",
                flush=True,
            )
            break

    final_path = args.output / "heterogeneous_residual_specialist_mixture.pt"
    final_payload = {
        **payload,
        "specialist_state_dicts": [cpu_state(model) for model in models],
        "mixture_weights": weights.detach().cpu(),
        "residual_repair_history": history,
        "residual_discovery": (
            "automatic connected components of high residual points in "
            "standardized 32D space"
        ),
    }
    torch.save(final_payload, final_path)
    pd.DataFrame([{
        "round": row["round"],
        "components_before": row["components_before"],
        "components_after": row["components_after"],
        "clusters": len(row["clusters"]),
        "held_out_log_m2_before": row["held_out_log_m2_before"],
        "held_out_log_m2_after": row["held_out_log_m2_after"],
        "relative_m2_improvement": row["relative_m2_improvement"],
    } for row in history]).to_csv(
        args.output / "iteration_summary.csv", index=False
    )
    with (args.output / "summary.json").open("w") as stream:
        json.dump({
            "input_mixture": str(args.mixture),
            "output_mixture": str(final_path),
            "initial_components": len(payload["specialist_state_dicts"]),
            "final_components": len(models),
            "uniform_fraction": uniform_fraction,
            "history": history,
        }, stream, indent=2)
    print("\nAUTOMATIC RESIDUAL-SPECIALIST REPAIR FINISHED")
    print(f"components = {len(models)}")
    print(f"mixture    = {final_path.resolve()}")


if __name__ == "__main__":
    main()
