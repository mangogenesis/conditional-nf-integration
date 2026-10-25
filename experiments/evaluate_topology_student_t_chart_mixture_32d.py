#!/usr/bin/env python3
"""Test covariance-aligned local Student-t charts for a difficult condition.

The base NF remains a protected component.  High-contribution samples from the
base/defensive proposal are clustered around previously discovered zoom
centres.  A full-covariance Student-t distribution is fitted in logit space to
each cluster, giving a normalized proposal on the open unit cube with exact
density.  Candidate protected mixtures are evaluated with common random
numbers from a broader reference mixture.
"""

from __future__ import annotations

import argparse
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

import evaluate_topology_mode_coverage_32d as coverage  # noqa: E402


DIMENSION = coverage.topology.DIMENSION
LOG_2PI = math.log(2.0 * math.pi)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--zoom-regions", type=Path, required=True)
    parser.add_argument("--condition", type=float, nargs=3, required=True)
    parser.add_argument("--discovery-evals", type=int, default=2_000_000)
    parser.add_argument("--discovery-keep", type=int, default=4096)
    parser.add_argument("--discovery-chunk", type=int, default=10_000)
    parser.add_argument("--uniform-fraction", type=float, default=0.01)
    parser.add_argument("--fit-temperature", type=float, default=4.0)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.10)
    parser.add_argument("--covariance-floor", type=float, default=1e-4)
    parser.add_argument("--degrees-of-freedom", type=float, default=5.0)
    parser.add_argument(
        "--chart-scales", type=float, nargs="+", default=(0.5, 1.0, 2.0)
    )
    parser.add_argument(
        "--mixture-fractions", type=float, nargs="+",
        default=(0.0, 0.001, 0.003, 0.01, 0.03, 0.10),
    )
    parser.add_argument("--reference-local-fraction", type=float, default=0.50)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.discovery_evals, args.discovery_keep, args.discovery_chunk,
           args.samples, args.chunk) <= 0:
        parser.error("sample counts and chunks must be positive")
    if not 0.0 <= args.uniform_fraction < 1.0:
        parser.error("uniform-fraction must lie in [0,1)")
    if not 0.0 < args.reference_local_fraction < 1.0:
        parser.error("reference-local-fraction must lie in (0,1)")
    if not 0.0 <= args.covariance_shrinkage <= 1.0:
        parser.error("covariance-shrinkage must lie in [0,1]")
    if args.covariance_floor <= 0 or args.degrees_of_freedom <= 2:
        parser.error("covariance floor must be positive and df must exceed 2")
    if args.fit_temperature <= 0 or min(args.chart_scales) <= 0:
        parser.error("fit temperature and chart scales must be positive")
    fractions = sorted(set(float(x) for x in args.mixture_fractions))
    if any(x < 0.0 or x >= 1.0 for x in fractions):
        parser.error("mixture fractions must lie in [0,1)")
    args.mixture_fractions = fractions
    args.chart_scales = sorted(set(float(x) for x in args.chart_scales))
    if any(x < 0.0 or x > 1.0 for x in args.condition):
        parser.error("condition must lie in [0,1]^3")
    return args


def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def logit(x):
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def defensive_log_prob(logq, uniform_fraction):
    if uniform_fraction == 0.0:
        return logq
    return torch.logaddexp(
        logq + math.log1p(-uniform_fraction),
        torch.full_like(logq, math.log(uniform_fraction)),
    )


@torch.inference_mode()
def discover_high_contribution_samples(
    adaptive, model, benchmark, condition, args, device
):
    best_x = None
    best_score = None
    remaining = args.discovery_evals
    while remaining:
        count = min(remaining, args.discovery_chunk)
        cond = condition.expand(count, 3)
        x, logr = adaptive.sample_from_proposal(
            proposal_model=model,
            n=count,
            cond=cond,
            device=device,
            uniform_fraction=args.uniform_fraction,
        )
        logq = model.log_prob(x, cond).reshape(-1)
        logf = coverage.topology.topology_log_integrand(
            benchmark, x, cond
        ).reshape(-1)
        score = 2.0 * logf.double() - logq.double() - logr.double()
        finite = torch.isfinite(score)
        x, score = x[finite], score[finite]
        take = min(args.discovery_keep, len(score))
        values, indices = torch.topk(score, take)
        if best_x is None:
            best_x, best_score = x[indices], values
        else:
            joined_x = torch.cat((best_x, x[indices]))
            joined_score = torch.cat((best_score, values))
            keep = torch.topk(
                joined_score, min(args.discovery_keep, len(joined_score))
            ).indices
            best_x, best_score = joined_x[keep], joined_score[keep]
        remaining -= count
        done = args.discovery_evals - remaining
        if done == args.discovery_evals or done % 250_000 == 0:
            print(
                f"discovery {done:,}/{args.discovery_evals:,} "
                f"max-log-contribution={float(best_score.max()):.5f}",
                flush=True,
            )
    return best_x, best_score


def stable_cholesky(covariance, floor):
    eye = torch.eye(len(covariance), dtype=covariance.dtype,
                    device=covariance.device)
    jitter = floor
    for _ in range(8):
        factor, info = torch.linalg.cholesky_ex(covariance + jitter * eye)
        if int(info.max()) == 0:
            return factor, jitter
        jitter *= 10.0
    raise RuntimeError("could not stabilize local chart covariance")


@torch.inference_mode()
def fit_charts(samples, scores, initial_centers, args):
    z = logit(samples.double())
    center_z = logit(initial_centers.double())
    # Standardization prevents a few naturally broad coordinates from
    # dominating cluster assignment.
    global_scale = z.std(dim=0).clamp_min(0.05)
    labels = torch.cdist(z / global_scale, center_z / global_scale).argmin(1)
    locations, factors, counts, jitters = [], [], [], []
    for index in range(len(initial_centers)):
        selected = labels == index
        if int(selected.sum()) < DIMENSION + 2:
            # Borrow nearest discoveries if a centre has too few assignments.
            distance = torch.linalg.vector_norm(
                (z - center_z[index]) / global_scale, dim=1
            )
            selected_indices = torch.topk(
                distance, min(max(DIMENSION * 4, DIMENSION + 2), len(z)),
                largest=False,
            ).indices
            cluster_z = z[selected_indices]
            cluster_score = scores[selected_indices]
        else:
            cluster_z = z[selected]
            cluster_score = scores[selected]
        tempered = (cluster_score - cluster_score.max()) / args.fit_temperature
        weights = torch.softmax(tempered, dim=0).double()
        # Cap domination by any single rare observation while preserving rank.
        cap = 10.0 / len(weights)
        weights = weights.clamp_max(cap)
        weights = weights / weights.sum()
        location = (weights[:, None] * cluster_z).sum(0)
        delta = cluster_z - location
        covariance = torch.einsum("n,ni,nj->ij", weights, delta, delta)
        diagonal = torch.diag(torch.diagonal(covariance))
        covariance = (
            (1.0 - args.covariance_shrinkage) * covariance
            + args.covariance_shrinkage * diagonal
        )
        factor, jitter = stable_cholesky(covariance, args.covariance_floor)
        locations.append(location)
        factors.append(factor)
        counts.append(int(len(cluster_z)))
        jitters.append(jitter)
    return (
        torch.stack(locations), torch.stack(factors), counts, jitters
    )


def student_t_mixture_log_prob(x, locations, factors, degrees, scale):
    x64 = x.double()
    inside = ((x64 > 0.0) & (x64 < 1.0)).all(dim=1)
    result = torch.full(
        (len(x64),), -torch.inf, dtype=torch.float64, device=x.device
    )
    if not inside.any():
        return result
    xin = x64[inside]
    z = logit(xin)
    component_logs = []
    dimension = z.shape[1]
    constant = (
        torch.lgamma(torch.tensor((degrees + dimension) / 2.0,
                                  device=x.device, dtype=torch.float64))
        - torch.lgamma(torch.tensor(degrees / 2.0, device=x.device,
                                    dtype=torch.float64))
        - 0.5 * dimension * math.log(degrees * math.pi)
    )
    for location, factor in zip(locations, factors):
        scaled_factor = factor * math.sqrt(scale)
        solved = torch.linalg.solve_triangular(
            scaled_factor, (z - location).T, upper=False
        ).T
        mahalanobis = solved.square().sum(1)
        logdet = 2.0 * torch.log(torch.diagonal(scaled_factor)).sum()
        logpz = constant - 0.5 * logdet - 0.5 * (
            degrees + dimension
        ) * torch.log1p(mahalanobis / degrees)
        log_jacobian = -torch.log(xin).sum(1) - torch.log1p(-xin).sum(1)
        component_logs.append(logpz + log_jacobian)
    result[inside] = torch.logsumexp(torch.stack(component_logs), dim=0) \
        - math.log(len(component_logs))
    return result


def sample_student_t_mixture(count, locations, factors, degrees, scale):
    device = locations.device
    indices = torch.randint(len(locations), (count,), device=device)
    normal = torch.randn(
        count, locations.shape[1], device=device, dtype=torch.float64
    )
    chi2 = torch.distributions.Chi2(
        torch.tensor(degrees, device=device, dtype=torch.float64)
    ).sample((count,))
    standardized = normal / torch.sqrt(chi2[:, None] / degrees)
    chosen_factor = factors[indices] * math.sqrt(scale)
    z = locations[indices] + torch.bmm(
        chosen_factor, standardized[:, :, None]
    ).squeeze(-1)
    return torch.sigmoid(z).float()


def mixture_log_prob(logq, logg, fraction):
    if fraction == 0.0:
        return logq
    return torch.logaddexp(
        logq + math.log1p(-fraction), logg + math.log(fraction)
    )


def update_logsum(current, values):
    value = torch.logsumexp(values.double(), dim=0).cpu()
    return value if current is None else torch.logaddexp(current, value)


@torch.inference_mode()
def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    adaptive, model, benchmark = coverage.load_model(
        args.base_checkpoint, device
    )
    payload = load_payload(args.zoom_regions)
    priority = payload["priority_conditions"].float()
    all_centers = payload["centers"].float()
    condition_cpu = torch.tensor(args.condition, dtype=torch.float32)
    nearest = int(torch.cdist(condition_cpu[None], priority).argmin())
    initial_centers = all_centers[nearest].to(device)
    condition = condition_cpu.to(device)
    distance = float(torch.linalg.vector_norm(condition_cpu - priority[nearest]))
    print(
        f"condition={condition_cpu.tolist()} nearest-priority={nearest} "
        f"distance={distance:.6g} initial-centers={len(initial_centers)}",
        flush=True,
    )

    samples, scores = discover_high_contribution_samples(
        adaptive, model, benchmark, condition, args, device
    )
    locations, factors, counts, jitters = fit_charts(
        samples, scores, initial_centers, args
    )
    print(
        f"fitted {len(locations)} Student-t charts; cluster-counts={counts}; "
        f"jitter={jitters}", flush=True
    )
    torch.save({
        "condition": condition_cpu,
        "nearest_priority_index": nearest,
        "locations_logit": locations.cpu(),
        "cholesky_factors_logit": factors.cpu(),
        "degrees_of_freedom": args.degrees_of_freedom,
        "cluster_counts": counts,
        "discovery_samples": samples.cpu(),
        "discovery_scores": scores.cpu(),
    }, args.output / "student_t_charts.pt")

    scales = args.chart_scales
    fractions = args.mixture_fractions
    reference_share = args.reference_local_fraction
    keys = [(scale, fraction) for scale in scales for fraction in fractions]
    log_m2_sum = {key: None for key in keys}
    proposal_branch0 = {key: 0.0 for key in keys}
    proposal_branch1 = {key: 0.0 for key in keys}
    proposal_gap = {key: 0.0 for key in keys}
    max_log_weight = {key: -math.inf for key in keys}
    log_integral_sum = None
    remaining = args.samples
    while remaining:
        count = min(remaining, args.chunk)
        cond = condition.expand(count, 3)
        choose_local = torch.rand(count, device=device) < reference_share
        x = torch.empty(count, DIMENSION, device=device)
        global_count = int((~choose_local).sum())
        local_count = int(choose_local.sum())
        if global_count:
            x[~choose_local] = adaptive.sample_from_model(
                model, global_count, cond[~choose_local]
            )
        if local_count:
            scale_indices = torch.randint(
                len(scales), (local_count,), device=device
            )
            local_x = torch.empty(local_count, DIMENSION, device=device)
            for scale_index, scale in enumerate(scales):
                selected = scale_indices == scale_index
                if selected.any():
                    local_x[selected] = sample_student_t_mixture(
                        int(selected.sum()), locations, factors,
                        args.degrees_of_freedom, scale,
                    )
            x[choose_local] = local_x

        logq = model.log_prob(x, cond).reshape(-1).double()
        logg = {
            scale: student_t_mixture_log_prob(
                x, locations, factors, args.degrees_of_freedom, scale
            ) for scale in scales
        }
        logg_reference = torch.logsumexp(
            torch.stack([logg[scale] for scale in scales]), dim=0
        ) - math.log(len(scales))
        logr = torch.logaddexp(
            logq + math.log1p(-reference_share),
            logg_reference + math.log(reference_share),
        )
        logf = coverage.topology.topology_log_integrand(
            benchmark, x, cond
        ).reshape(-1).double()
        log_integral_sum = update_logsum(log_integral_sum, logf - logr)
        branch0 = (x[:, 0] <= coverage.topology.BRANCH_WIDTH).double()
        branch1 = (x[:, 0] >= 1.0 - coverage.topology.BRANCH_WIDTH).double()
        gap = 1.0 - branch0 - branch1
        for scale, fraction in keys:
            key = (scale, fraction)
            logmix = mixture_log_prob(logq, logg[scale], fraction)
            log_m2_sum[key] = update_logsum(
                log_m2_sum[key], 2.0 * logf - logmix - logr
            )
            ratio = torch.exp(logmix - logr)
            proposal_branch0[key] += float((ratio * branch0).sum())
            proposal_branch1[key] += float((ratio * branch1).sum())
            proposal_gap[key] += float((ratio * gap).sum())
            max_log_weight[key] = max(
                max_log_weight[key], float((logf - logmix).max())
            )
        remaining -= count
        done = args.samples - remaining
        if done == args.samples or done % 250_000 == 0:
            print(f"evaluation {done:,}/{args.samples:,}", flush=True)

    integral = math.exp(float(log_integral_sum) - math.log(args.samples))
    target_p1 = float(coverage.topology.branch_probabilities(
        benchmark, condition.reshape(1, 3)
    )[0, 1])
    rows = []
    for scale, fraction in keys:
        key = (scale, fraction)
        m2 = math.exp(float(log_m2_sum[key]) - math.log(args.samples))
        p0 = proposal_branch0[key] / args.samples
        p1 = proposal_branch1[key] / args.samples
        pgap = proposal_gap[key] / args.samples
        rows.append({
            "chart_scale": scale,
            "mixture_fraction": fraction,
            "integral_estimate": integral,
            "m2": m2,
            "normalized_ess": integral * integral / m2,
            "target_branch_1_mass": target_p1,
            "proposal_branch_0_mass": p0,
            "proposal_branch_1_mass": p1,
            "absolute_branch_1_mass_error": abs(p1 - target_p1),
            "proposal_gap_mass": pgap,
            "maximum_log_weight_on_reference_samples": max_log_weight[key],
            "theoretical_maximum_m2_ratio_to_base": 1.0 / (1.0 - fraction),
        })
    frame = pd.DataFrame(rows)
    base_m2 = float(frame[frame["mixture_fraction"] == 0.0]["m2"].iloc[0])
    frame["m2_ratio_to_base"] = frame["m2"] / base_m2
    frame.to_csv(args.output / "student_t_chart_mixture.csv", index=False)
    best = frame.loc[frame["m2"].idxmin()].to_dict()
    summary = {
        "benchmark": benchmark,
        "base_checkpoint": str(args.base_checkpoint),
        "zoom_regions": str(args.zoom_regions),
        "condition": list(args.condition),
        "nearest_priority_condition_index": nearest,
        "condition_distance": distance,
        "discovery_evals": args.discovery_evals,
        "discovery_keep": args.discovery_keep,
        "degrees_of_freedom": args.degrees_of_freedom,
        "cluster_counts": counts,
        "reference_local_fraction": reference_share,
        "samples": args.samples,
        "integral_estimate": integral,
        "best": best,
        "results": frame.to_dict("records"),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nCOVARIANCE-ALIGNED STUDENT-T CHART MIXTURE")
    print(frame.to_string(index=False))
    print("\nBEST CANDIDATE")
    print(json.dumps(best, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
