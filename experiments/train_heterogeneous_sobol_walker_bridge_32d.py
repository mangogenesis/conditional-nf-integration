#!/usr/bin/env python3
"""Blind Sobol + potential-walker repair of a heterogeneous NF mixture.

The deployed mixture defines the start of an ESS-controlled geometric bridge

    pi_beta(x) proportional to g(x)^(1-beta) f(x)^beta,

where g is a defensively protected version of the current proposal.  A formal
SMC population follows this bridge.  At every bridge temperature an independent
scrambled-Sobol population searches for places where the current proposal is
small relative to the tempered target.  Gradient-free random-walk Metropolis
chains refine the best scouts.  Only after beta reaches one are the accumulated
discoveries clustered and used to train protected local specialists.

The scout population never contributes to an integral estimate.  The script
uses neither benchmark structure labels, the number of structures, nor the
known value of the integral.
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
import train_heterogeneous_discovered_region_specialists_32d as training  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--ess-fraction", type=float, default=0.75)
    parser.add_argument("--maximum-bridge-stages", type=int, default=128)
    parser.add_argument("--main-walker-steps", type=int, default=8)
    parser.add_argument("--sobol-points-per-stage", type=int, default=65536)
    parser.add_argument("--scout-seeds-per-stage", type=int, default=256)
    parser.add_argument("--scout-walker-steps", type=int, default=40)
    parser.add_argument("--main-walker-step-size", type=float, default=0.008)
    parser.add_argument("--scout-walker-step-size", type=float, default=0.015)
    parser.add_argument("--local-walker-step-size", type=float, default=0.008)
    parser.add_argument("--defensive-uniform-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-discovery-beta", type=float, default=0.10)
    parser.add_argument("--maximum-discovery-points", type=int, default=2048)
    parser.add_argument("--neighbor-rank", type=int, default=8)
    parser.add_argument("--cluster-scale", type=float, default=1.5)
    parser.add_argument("--minimum-cluster-points", type=int, default=32)
    parser.add_argument("--maximum-new-specialists", type=int, default=8)
    parser.add_argument("--train-walkers-per-cluster", type=int, default=256)
    parser.add_argument("--validation-walkers-per-cluster", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--train-draws-per-walker", type=int, default=16)
    parser.add_argument("--validation-draws-per-walker", type=int, default=8)
    parser.add_argument("--points-per-group", type=int, default=512)
    parser.add_argument("--marginal-steps", type=int, default=20000)
    parser.add_argument("--copula-steps", type=int, default=20000)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--repair-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = (
        args.particles, args.maximum_bridge_stages, args.main_walker_steps,
        args.sobol_points_per_stage, args.scout_seeds_per_stage,
        args.scout_walker_steps, args.maximum_discovery_points,
        args.neighbor_rank, args.minimum_cluster_points,
        args.maximum_new_specialists, args.train_walkers_per_cluster,
        args.validation_walkers_per_cluster, args.burn_in, args.thin,
        args.train_draws_per_walker, args.validation_draws_per_walker,
        args.points_per_group, args.marginal_steps, args.copula_steps,
        args.check_interval, args.patience,
    )
    if min(counts) <= 0:
        parser.error("all budgets must be positive")
    for name in (
        "ess_fraction", "defensive_uniform_fraction", "repair_fraction"
    ):
        value = getattr(args, name)
        if not 0 < value < 1:
            parser.error(f"--{name.replace('_', '-')} must lie in (0,1)")
    if not 0 <= args.minimum_discovery_beta <= 1:
        parser.error("--minimum-discovery-beta must lie in [0,1]")
    if min(
        args.main_walker_step_size,
        args.scout_walker_step_size,
        args.local_walker_step_size,
        args.cluster_scale,
    ) <= 0:
        parser.error("walker and cluster scales must be positive")
    return args


def load_payload(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "heterogeneous_discovered_region_mixture_v1":
        raise ValueError("unsupported mixture format")
    return payload


def reflect_unit(x):
    y = torch.remainder(x, 2.0)
    return torch.where(y <= 1.0, y, 2.0 - y)


@torch.no_grad()
def log_target(x, condition):
    cond = condition.expand(len(x), -1)
    return heterogeneous.heterogeneous_log_integrand(x, cond).reshape(-1).double()


@torch.no_grad()
def base_logq(models, weights, uniform_fraction, x, condition, chunk=4096):
    rows = []
    for start in range(0, len(x), chunk):
        stop = min(len(x), start + chunk)
        cond = condition.expand(stop - start, -1)
        rows.append(mixture_audit.mixture_log_prob(
            models, weights, uniform_fraction, x[start:stop], cond
        ))
    return torch.cat(rows)


@torch.no_grad()
def protected_logq(models, weights, uniform_fraction, extra_uniform, x, condition):
    logq = base_logq(models, weights, uniform_fraction, x, condition)
    if extra_uniform == 0:
        return logq
    return torch.logaddexp(
        logq + math.log1p(-extra_uniform),
        torch.full_like(logq, math.log(extra_uniform)),
    )


@torch.no_grad()
def bridge_log_density(models, weights, uniform_fraction, extra_uniform,
                       x, condition, beta):
    logg = protected_logq(
        models, weights, uniform_fraction, extra_uniform, x, condition
    )
    logf = log_target(x, condition)
    return (1.0 - beta) * logg + beta * logf, logf, logg


def relative_ess(log_weights):
    shifted = log_weights.double() - log_weights.double().max()
    weights = shifted.exp()
    return float(weights.sum().square() / weights.square().sum() / len(weights))


def choose_next_beta(log_ratio, beta, target_ess):
    if beta >= 1.0:
        return 1.0
    if relative_ess((1.0 - beta) * log_ratio) >= target_ess:
        return 1.0
    low, high = beta, 1.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if relative_ess((mid - beta) * log_ratio) >= target_ess:
            low = mid
        else:
            high = mid
    return low


@torch.no_grad()
def sample_protected(adaptive, models, weights, uniform_fraction,
                     extra_uniform, condition, count):
    cond = condition.expand(count, -1)
    if extra_uniform == 0:
        return mixture_audit.sample_mixture(
            adaptive, models, weights, uniform_fraction, cond
        )[0]
    choose_uniform = torch.rand(count, device=condition.device) < extra_uniform
    x = torch.empty(
        count, heterogeneous.DIMENSION,
        dtype=condition.dtype, device=condition.device,
    )
    if bool(choose_uniform.any()):
        x[choose_uniform] = torch.rand(
            int(choose_uniform.sum()), heterogeneous.DIMENSION,
            dtype=condition.dtype, device=condition.device,
        )
    if bool((~choose_uniform).any()):
        selected = ~choose_uniform
        x[selected] = mixture_audit.sample_mixture(
            adaptive, models, weights, uniform_fraction, cond[selected]
        )[0]
    return x


@torch.no_grad()
def random_walk(models, weights, uniform_fraction, extra_uniform,
                x, condition, beta, steps, step_size):
    current, _, _ = bridge_log_density(
        models, weights, uniform_fraction, extra_uniform, x, condition, beta
    )
    accepted = 0
    for _ in range(steps):
        proposal = reflect_unit(x + step_size * torch.randn_like(x))
        proposed, _, _ = bridge_log_density(
            models, weights, uniform_fraction, extra_uniform,
            proposal, condition, beta,
        )
        log_acceptance = proposed - current
        accept = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
        x = torch.where(accept[:, None], proposal, x)
        current = torch.where(accept, proposed, current)
        accepted += int(accept.sum())
    return x, accepted / (steps * len(x))


@torch.no_grad()
def collect_walk(models, weights, uniform_fraction, extra_uniform,
                 initial, condition, beta, burn_in, thin, draws, step_size):
    x, first_acceptance = random_walk(
        models, weights, uniform_fraction, extra_uniform,
        initial.clone(), condition, beta, burn_in, step_size,
    )
    samples = []
    acceptance = [first_acceptance]
    for _ in range(draws):
        x, value = random_walk(
            models, weights, uniform_fraction, extra_uniform,
            x, condition, beta, thin, step_size,
        )
        samples.append(x.clone())
        acceptance.append(value)
    return torch.cat(samples), float(np.mean(acceptance))


def kmeans(z, clusters, restarts=6, iterations=60):
    """Small deterministic-budget k-means used only for discovery."""
    best_labels = None
    best_inertia = math.inf
    for restart in range(restarts):
        first = (restart * max(1, len(z) // restarts)) % len(z)
        centers = [z[first]]
        for _ in range(1, clusters):
            distance = torch.cdist(z, torch.stack(centers)).square().min(dim=1).values
            centers.append(z[int(distance.argmax())])
        centers = torch.stack(centers)
        labels = torch.zeros(len(z), dtype=torch.long, device=z.device)
        for _ in range(iterations):
            new_labels = torch.cdist(z, centers).argmin(dim=1)
            if torch.equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels
            updated = []
            for index in range(clusters):
                mask = labels == index
                updated.append(z[mask].mean(dim=0) if bool(mask.any()) else centers[index])
            centers = torch.stack(updated)
        inertia = float((z - centers[labels]).square().sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
    return best_labels


def silhouette_score(distance, labels, clusters):
    values = []
    for index in range(clusters):
        own = labels == index
        count = int(own.sum())
        if count <= 1:
            return -math.inf
        within = distance[own][:, own].sum(dim=1) / (count - 1)
        other_means = []
        for other in range(clusters):
            if other == index:
                continue
            mask = labels == other
            if not bool(mask.any()):
                return -math.inf
            other_means.append(distance[own][:, mask].mean(dim=1))
        between = torch.stack(other_means).min(dim=0).values
        values.append((between - within) / torch.maximum(within, between).clamp_min(1e-12))
    return float(torch.cat(values).mean())


def discover_clusters(points, scores, args):
    count = min(args.maximum_discovery_points, len(points))
    selected = torch.topk(scores, count).indices
    x = points[selected].double()
    center = x.median(dim=0).values
    q25 = torch.quantile(x, 0.25, dim=0)
    q75 = torch.quantile(x, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(0.02)
    z = (x - center) / scale
    distance = torch.cdist(z, z)

    # Single-linkage connected components can merge genuinely separate modes
    # through a thin chain of points.  Instead, infer K by held-out geometry:
    # try every admissible K and retain the partition with best silhouette.
    # K=1 remains available through a conservative score threshold.
    maximum = min(
        args.maximum_new_specialists,
        len(z) // args.minimum_cluster_points,
    )
    best_labels = torch.zeros(len(z), dtype=torch.long, device=z.device)
    best_k = 1
    best_score = 0.08
    candidate_scores = {1: best_score}
    for clusters in range(2, maximum + 1):
        labels = kmeans(z, clusters)
        sizes = torch.bincount(labels, minlength=clusters)
        if int(sizes.min()) < args.minimum_cluster_points:
            continue
        value = silhouette_score(distance, labels, clusters)
        candidate_scores[clusters] = value
        if value > best_score:
            best_score = value
            best_k = clusters
            best_labels = labels

    groups = [
        torch.nonzero(best_labels == index, as_tuple=False).flatten()
        for index in range(best_k)
    ]
    groups.sort(
        key=lambda group: float(scores[selected[group]].median()), reverse=True
    )
    clusters = [selected[group] for group in groups]
    diagnostics = {
        "selected_k": best_k,
        "selected_silhouette": best_score,
        "candidate_silhouettes": candidate_scores,
    }
    return clusters, diagnostics


def cache_from_samples(x, condition, points_per_group):
    groups = len(x) // points_per_group
    if groups < 1:
        raise RuntimeError("walker cloud cannot form a training group")
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
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    payload = load_payload(args.mixture, device)
    adaptive, models, weights, uniform_fraction = mixture_audit.load_mixture(
        args.mixture, device
    )
    weights = weights.to(device=device, dtype=torch.float64)
    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    ).reshape(1, -1)

    # Formal bridge population.  Sobol scouts below are deliberately separate.
    particles = sample_protected(
        adaptive, models, weights, uniform_fraction,
        args.defensive_uniform_fraction, condition, args.particles,
    )
    beta = 0.0
    stage_rows = []
    discoveries = []
    discovery_scores = []
    sobol = torch.quasirandom.SobolEngine(
        heterogeneous.DIMENSION, scramble=True, seed=args.seed + 17
    )

    for stage in range(args.maximum_bridge_stages):
        _, logf, logg = bridge_log_density(
            models, weights, uniform_fraction,
            args.defensive_uniform_fraction, particles, condition, beta,
        )
        log_ratio = logf - logg
        next_beta = choose_next_beta(log_ratio, beta, args.ess_fraction)
        if next_beta <= beta + 1e-10:
            raise RuntimeError(
                f"bridge stalled at beta={beta:.8f}; increase particles or "
                "reduce --ess-fraction"
            )
        incremental = (next_beta - beta) * log_ratio
        incremental -= incremental.max()
        resampling_weights = incremental.exp()
        resampling_weights /= resampling_weights.sum()
        bridge_ess = relative_ess(incremental)
        indices = torch.multinomial(
            resampling_weights.float(), len(particles), replacement=True
        )
        particles = particles[indices]
        particles, main_acceptance = random_walk(
            models, weights, uniform_fraction,
            args.defensive_uniform_fraction, particles, condition,
            next_beta, args.main_walker_steps, args.main_walker_step_size,
        )

        scouts = sobol.draw(
            args.sobol_points_per_stage, dtype=torch.float32
        ).to(device)
        _, scout_logf, scout_logg = bridge_log_density(
            models, weights, uniform_fraction,
            args.defensive_uniform_fraction, scouts, condition, next_beta,
        )
        finite = torch.isfinite(scout_logf) & torch.isfinite(scout_logg)
        finite_count = int(finite.sum())
        scout_acceptance = math.nan
        maximum_score = -math.inf
        if finite_count:
            candidates = scouts[finite]
            logf_candidate = scout_logf[finite]
            logg_candidate = scout_logg[finite]
            # Large values indicate a large local contribution to the
            # tempered target's second moment under the current proposal.
            score = (
                2.0 * ((1.0 - next_beta) * logg_candidate
                       + next_beta * logf_candidate)
                - logg_candidate
            )
            keep = min(args.scout_seeds_per_stage, len(candidates))
            chosen = torch.topk(score, keep).indices
            seeds = candidates[chosen]
            walked, scout_acceptance = random_walk(
                models, weights, uniform_fraction,
                args.defensive_uniform_fraction, seeds, condition,
                next_beta, args.scout_walker_steps,
                args.scout_walker_step_size,
            )
            _, walked_logf, walked_logg = bridge_log_density(
                models, weights, uniform_fraction,
                args.defensive_uniform_fraction, walked, condition, next_beta,
            )
            walked_score = (
                2.0 * ((1.0 - next_beta) * walked_logg
                       + next_beta * walked_logf)
                - walked_logg
            )
            maximum_score = float(walked_score.max())
            if next_beta >= args.minimum_discovery_beta:
                discoveries.append(walked.detach())
                # Put all stages on the final residual scale before clustering.
                final_logf = log_target(walked, condition)
                final_logg = protected_logq(
                    models, weights, uniform_fraction,
                    args.defensive_uniform_fraction, walked, condition,
                )
                discovery_scores.append(2.0 * final_logf - final_logg)

        row = {
            "stage": stage,
            "beta_before": beta,
            "beta_after": next_beta,
            "incremental_relative_ess": bridge_ess,
            "main_walker_acceptance": main_acceptance,
            "sobol_points": args.sobol_points_per_stage,
            "finite_target_sobol_points": finite_count,
            "scout_walker_acceptance": scout_acceptance,
            "maximum_tempered_residual_score": maximum_score,
        }
        stage_rows.append(row)
        print(
            f"BRIDGE {stage:02d}: beta={beta:.6f}->{next_beta:.6f} "
            f"ESS={bridge_ess:.4f} main-RW={main_acceptance:.3f} "
            f"Sobol-support={finite_count:,}/{args.sobol_points_per_stage:,} "
            f"scout-RW={scout_acceptance:.3f}",
            flush=True,
        )
        beta = next_beta
        if beta >= 1.0 - 1e-10:
            break
    else:
        raise RuntimeError("maximum bridge stages reached before beta=1")

    if not discoveries:
        raise RuntimeError("bridge produced no eligible discovery points")
    discovery_x = torch.cat(discoveries)
    discovery_score = torch.cat(discovery_scores)
    clusters, cluster_diagnostics = discover_clusters(
        discovery_x, discovery_score, args
    )
    print(
        f"DISCOVERY: {len(discovery_x):,} points -> {len(clusters)} "
        f"automatic clusters "
        f"(silhouette={cluster_diagnostics['selected_silhouette']:.4f})",
        flush=True,
    )

    adaptive.DIM = heterogeneous.DIMENSION
    adaptive.configure_speed_profile("balanced")
    adaptive.COMPILE_TRAINING_KERNELS = False
    adaptive.MAX_MARGINAL_STEPS = args.marginal_steps
    adaptive.MIN_MARGINAL_STEPS = min(2000, args.marginal_steps)
    adaptive.MARGINAL_CHECK_INTERVAL = args.check_interval
    adaptive.MARGINAL_EARLY_STOP_PATIENCE = args.patience
    adaptive.BRIDGE_COPULA_MAX_STEPS = args.copula_steps
    adaptive.BRIDGE_COPULA_CHECK_INTERVAL = args.check_interval
    adaptive.BRIDGE_COPULA_PATIENCE = args.patience

    repairs = []
    training_rows = []
    cloud_payload = {"clusters": []}
    for cluster_index, indices in enumerate(clusters):
        seeds = discovery_x[indices]
        scores = discovery_score[indices]
        component_logs = torch.stack([
            model.log_prob(
                seeds, condition.expand(len(seeds), -1)
            ).reshape(-1).double()
            for model in models
        ], dim=1)
        owner = int((component_logs + weights.log()).mean(dim=0).argmax())
        train_initial = seeds[torch.randint(
            len(seeds), (args.train_walkers_per_cluster,), device=device
        )]
        validation_initial = seeds[torch.randint(
            len(seeds), (args.validation_walkers_per_cluster,), device=device
        )]
        train_samples, train_acceptance = collect_walk(
            models, weights, uniform_fraction,
            args.defensive_uniform_fraction, train_initial, condition, 1.0,
            args.burn_in, args.thin, args.train_draws_per_walker,
            args.local_walker_step_size,
        )
        validation_samples, validation_acceptance = collect_walk(
            models, weights, uniform_fraction,
            args.defensive_uniform_fraction, validation_initial, condition, 1.0,
            args.burn_in, args.thin, args.validation_draws_per_walker,
            args.local_walker_step_size,
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
        print("\n" + "=" * 88, flush=True)
        print(
            f"SPECIALIST cluster={cluster_index} owner={owner} "
            f"seeds={len(seeds)} score-median={float(scores.median()):.6f}",
            flush=True,
        )
        marginal = adaptive.train_marginals_on_dataset(
            adaptive.get_marginal_bank(specialist),
            train_cache, validation_cache, device,
            args.seed + 1000 * (cluster_index + 1),
        )
        copula = adaptive.train_bridge_copula_on_cache(
            specialist, train_cache, validation_cache,
            label=f"Sobol-walker cluster {cluster_index}", optimize_joint=False,
        )
        accepted = marginal["best_step"] > 0 or copula["best_step"] > 0
        row = {
            "cluster": cluster_index,
            "owner": owner,
            "seeds": len(seeds),
            "median_final_residual": float(scores.median()),
            "train_walker_acceptance": train_acceptance,
            "validation_walker_acceptance": validation_acceptance,
            "marginal_best_step": marginal["best_step"],
            "copula_best_step": copula["best_step"],
            "accepted": accepted,
        }
        training_rows.append(row)
        if accepted:
            specialist.eval()
            for parameter in specialist.parameters():
                parameter.requires_grad_(False)
            repairs.append(specialist)
            cloud_payload["clusters"].append({
                "cluster": cluster_index,
                "train": train_samples.cpu(),
                "validation": validation_samples.cpu(),
            })

    if repairs:
        base_weights = (1.0 - args.repair_fraction) * weights
        repair_weights = torch.full(
            (len(repairs),), args.repair_fraction / len(repairs),
            dtype=torch.float64, device=device,
        )
        final_models = [*models, *repairs]
        final_weights = torch.cat((base_weights, repair_weights))
    else:
        final_models = models
        final_weights = weights

    output_path = args.output / "heterogeneous_sobol_walker_repair_mixture.pt"
    result = {
        **payload,
        "specialist_state_dicts": [cpu_state(model) for model in final_models],
        "mixture_weights": final_weights.detach().cpu(),
        "sobol_walker_repair": {
            "input_mixture": str(args.mixture),
            "uses_structure_labels": False,
            "uses_known_structure_count": False,
            "uses_known_integral": False,
            "formal_population": "ESS-controlled SMC bridge",
            "discovery_population": "independent scrambled Sobol plus potential RWM",
            "bridge_stages": stage_rows,
            "automatic_clusters": len(clusters),
            "cluster_selection": cluster_diagnostics,
            "accepted_repairs": len(repairs),
            "repair_fraction": args.repair_fraction if repairs else 0.0,
            "training": training_rows,
        },
    }
    torch.save(result, output_path)
    torch.save(cloud_payload, args.output / "sobol_walker_repair_clouds.pt")
    pd.DataFrame(stage_rows).to_csv(
        args.output / "bridge_summary.csv", index=False
    )
    pd.DataFrame(training_rows).to_csv(
        args.output / "specialist_training_summary.csv", index=False
    )
    summary = result["sobol_walker_repair"] | {
        "output_mixture": str(output_path),
        "initial_components": len(models),
        "final_components": len(final_models),
        "weights": final_weights.detach().cpu().tolist(),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nSOBOL + POTENTIAL-WALKER REPAIR FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"\nMixture: {output_path.resolve()}")


if __name__ == "__main__":
    main()
