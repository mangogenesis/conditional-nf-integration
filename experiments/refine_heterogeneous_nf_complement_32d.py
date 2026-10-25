#!/usr/bin/env python3
"""Discover missed target regions by locally refining the NF complement.

This is deliberately simpler than SMC.  Uniform candidates are retained only
when the frozen NF assigns low density and the target log integrand is finite.
Their branch coordinate x0 is frozen, while x1,...,x31 are independently
optimized to increase log f.  The refined endpoints are ranked afterward by

    2 log f(x|c) - log q(x|c),

so the NF does not guide the local target search itself.  No target sampler,
known normalization, structure count, or structure labels are used.
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

import audit_heterogeneous_topology_32d as audit  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--calibration-samples", type=int, default=100_000)
    parser.add_argument("--complement-quantile", type=float, default=0.01)
    parser.add_argument("--refinement-seeds", type=int, default=4096)
    parser.add_argument("--candidate-chunk", type=int, default=10_000)
    parser.add_argument("--refinement-batch", type=int, default=512)
    parser.add_argument("--refinement-steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=50.0)
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--x0-gap-threshold", type=float, default=0.02)
    parser.add_argument("--keep", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.calibration_samples,
        args.refinement_seeds,
        args.candidate_chunk,
        args.refinement_batch,
        args.refinement_steps,
        args.check_interval,
        args.keep,
    ) <= 0:
        parser.error("all sample counts, batches, and steps must be positive")
    if not 0 < args.complement_quantile < 1:
        parser.error("--complement-quantile must lie in (0,1)")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        parser.error("learning rate and gradient clip must be positive")
    if not 0 < args.x0_gap_threshold < 1:
        parser.error("--x0-gap-threshold must lie in (0,1)")
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("fixed condition must lie in [0,1]^3")
    return args


@torch.inference_mode()
def calibrate_threshold(adaptive, model, condition, args):
    rows = []
    remaining = args.calibration_samples
    while remaining:
        count = min(remaining, args.candidate_chunk)
        cond = condition.expand(count, -1)
        x = adaptive.sample_from_model(model, count, cond)
        rows.append(model.log_prob(x, cond).reshape(-1).cpu())
        remaining -= count
    values = torch.cat(rows).double()
    return float(torch.quantile(values, args.complement_quantile)), {
        "nf_logq_minimum": float(values.min()),
        "nf_logq_median": float(values.median()),
        "nf_logq_maximum": float(values.max()),
    }


@torch.inference_mode()
def draw_complement_seeds(model, condition, threshold, args, device):
    accepted = []
    attempted = 0
    low_q = 0
    finite_target = 0
    while sum(len(block) for block in accepted) < args.refinement_seeds:
        x = torch.rand(
            args.candidate_chunk,
            heterogeneous.DIMENSION,
            dtype=torch.float32,
            device=device,
        )
        cond = condition.expand(len(x), -1)
        logq = model.log_prob(x, cond).reshape(-1)
        complement = logq <= threshold
        attempted += len(x)
        low_q += int(complement.sum())
        if bool(complement.any()):
            candidate = x[complement]
            candidate_cond = cond[complement]
            logf = heterogeneous.heterogeneous_log_integrand(
                candidate, candidate_cond
            )
            finite = torch.isfinite(logf)
            finite_target += int(finite.sum())
            if bool(finite.any()):
                accepted.append(candidate[finite].cpu())
        if attempted % max(args.candidate_chunk, 100_000) == 0:
            print(
                f"seed scan attempted={attempted:,} low-q={low_q:,} "
                f"finite-target={finite_target:,}",
                flush=True,
            )
    seeds = torch.cat(accepted, dim=0)[:args.refinement_seeds]
    return seeds, {
        "uniform_candidates_attempted": attempted,
        "low_q_candidates": low_q,
        "finite_target_candidates": finite_target,
    }


def refine_batch(initial, condition, args, device, batch_index, batches):
    initial = initial.to(device)
    fixed_x0 = initial[:, :1].detach().clone()
    coordinates = torch.nn.Parameter(initial[:, 1:].detach().clone())
    optimizer = torch.optim.Adam([coordinates], lr=args.learning_rate)
    cond = condition.expand(len(initial), -1)
    with torch.no_grad():
        initial_logf = heterogeneous.heterogeneous_log_integrand(
            initial, cond
        ).double().cpu()
    for step in range(1, args.refinement_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x = torch.cat((fixed_x0, coordinates), dim=1)
        logf = heterogeneous.heterogeneous_log_integrand(x, cond)
        loss = -logf.mean()
        loss.backward()
        with torch.no_grad():
            gradient = coordinates.grad
            norms = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
            gradient.mul_((args.gradient_clip / norms).clamp(max=1.0))
        optimizer.step()
        with torch.no_grad():
            coordinates.clamp_(1e-6, 1.0 - 1e-6)
        if step % args.check_interval == 0 or step == args.refinement_steps:
            with torch.no_grad():
                current = torch.cat((fixed_x0, coordinates), dim=1)
                current_logf = heterogeneous.heterogeneous_log_integrand(
                    current, cond
                )
                print(
                    f"refinement batch={batch_index + 1}/{batches} "
                    f"step={step}/{args.refinement_steps} "
                    f"median-logf={float(current_logf.median()):.4f} "
                    f"max-logf={float(current_logf.max()):.4f}",
                    flush=True,
                )
    with torch.no_grad():
        refined = torch.cat((fixed_x0, coordinates), dim=1)
        refined_logf = heterogeneous.heterogeneous_log_integrand(
            refined, cond
        ).double()
    # q is evaluated by the caller in larger inference-only chunks.  Keeping
    # it out of the optimization graph ensures that the frozen NF cannot guide
    # the target-mode search.
    return refined.detach().cpu(), initial_logf, refined_logf.cpu()


@torch.inference_mode()
def evaluate_nf(model, points, condition, chunk, device):
    rows = []
    for start in range(0, len(points), chunk):
        x = points[start:start + chunk].to(device)
        cond = condition.expand(len(x), -1)
        rows.append(model.log_prob(x, cond).reshape(-1).double().cpu())
    return torch.cat(rows)


def discover_x0_regions(x0, gap_threshold):
    order = torch.argsort(x0)
    sorted_x0 = x0[order]
    new_region = torch.zeros(len(x0), dtype=torch.int64)
    if len(x0) > 1:
        new_region[1:] = (sorted_x0[1:] - sorted_x0[:-1] > gap_threshold).long()
    sorted_region = torch.cumsum(new_region, dim=0)
    region = torch.empty_like(sorted_region)
    region[order] = sorted_region
    return region


def quantiles(values):
    values = values.double()
    return {
        "minimum": float(values.min()),
        "q10": float(torch.quantile(values, 0.10)),
        "median": float(values.median()),
        "q90": float(torch.quantile(values, 0.90)),
        "maximum": float(values.max()),
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    adaptive, model = audit.load_model(args.checkpoint, device)
    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    ).reshape(1, -1)
    threshold, calibration = calibrate_threshold(
        adaptive, model, condition, args
    )
    print(
        f"NF complement threshold={threshold:.6f} "
        f"(lower quantile={args.complement_quantile:.4f})",
        flush=True,
    )
    seeds, discovery = draw_complement_seeds(
        model, condition, threshold, args, device
    )
    refined_rows = []
    initial_logf_rows = []
    refined_logf_rows = []
    batches = math.ceil(len(seeds) / args.refinement_batch)
    for batch_index, start in enumerate(
        range(0, len(seeds), args.refinement_batch)
    ):
        refined, initial_logf, refined_logf = refine_batch(
            seeds[start:start + args.refinement_batch],
            condition,
            args,
            device,
            batch_index,
            batches,
        )
        refined_rows.append(refined)
        initial_logf_rows.append(initial_logf)
        refined_logf_rows.append(refined_logf)
    refined = torch.cat(refined_rows)
    initial_logf = torch.cat(initial_logf_rows)
    refined_logf = torch.cat(refined_logf_rows)
    refined_logq = evaluate_nf(
        model, refined, condition, args.candidate_chunk, device
    )
    log_ratio = refined_logf - refined_logq
    log_m2_contribution = 2 * refined_logf - refined_logq
    region = discover_x0_regions(refined[:, 0], args.x0_gap_threshold)
    order = torch.argsort(log_m2_contribution, descending=True)
    data = {
        "rank": torch.arange(1, len(refined) + 1).numpy(),
        "discovered_region": region[order].numpy(),
        "initial_logf": initial_logf[order].numpy(),
        "refined_logf": refined_logf[order].numpy(),
        "logq": refined_logq[order].numpy(),
        "log_f_over_q": log_ratio[order].numpy(),
        "log_second_moment_contribution": log_m2_contribution[order].numpy(),
    }
    for coordinate in range(heterogeneous.DIMENSION):
        data[f"x{coordinate}"] = refined[order, coordinate].numpy()
    frame = pd.DataFrame(data)
    frame.to_csv(args.output / "refined_complement_endpoints.csv", index=False)
    keep = min(args.keep, len(order))
    torch.save(
        {
            "x": refined[order[:keep]],
            "condition": condition.cpu(),
            "initial_logf": initial_logf[order[:keep]],
            "logf": refined_logf[order[:keep]],
            "logq": refined_logq[order[:keep]],
            "log_f_over_q": log_ratio[order[:keep]],
            "log_second_moment_contribution": log_m2_contribution[order[:keep]],
            "discovered_region": region[order[:keep]],
        },
        args.output / "top_refined_complement_endpoints.pt",
    )
    region_rows = []
    for index in torch.unique(region, sorted=True).tolist():
        mask = region == index
        region_rows.append({
            "discovered_region": index,
            "seeds": int(mask.sum()),
            "x0_minimum": float(refined[mask, 0].min()),
            "x0_maximum": float(refined[mask, 0].max()),
            "median_refined_logf": float(refined_logf[mask].median()),
            "maximum_refined_logf": float(refined_logf[mask].max()),
            "median_log_f_over_q": float(log_ratio[mask].median()),
            "maximum_log_f_over_q": float(log_ratio[mask].max()),
            "maximum_log_second_moment_contribution": float(
                log_m2_contribution[mask].max()
            ),
        })
    region_frame = pd.DataFrame(region_rows)
    region_frame.to_csv(args.output / "discovered_regions.csv", index=False)
    summary = {
        "benchmark": heterogeneous.BENCHMARK,
        "checkpoint": str(args.checkpoint),
        "condition": list(args.fixed_condition),
        "method": "uniform NF-complement seeds plus independent projected log-f ascent",
        "uses_smc": False,
        "uses_target_sampler": False,
        "uses_structure_labels": False,
        "nf_complement_threshold": threshold,
        "refinement_seeds": len(seeds),
        "refinement_steps": args.refinement_steps,
        "automatically_discovered_x0_regions": len(region_rows),
        "initial_logf": quantiles(initial_logf),
        "refined_logf": quantiles(refined_logf),
        "refined_log_f_over_q": quantiles(log_ratio),
        "refined_log_second_moment_contribution": quantiles(
            log_m2_contribution
        ),
        **calibration,
        **discovery,
        "regions": region_rows,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nCOMPLEMENT-SEEDED LOCAL REFINEMENT")
    print(json.dumps(summary, indent=2))
    print("\nDISCOVERED REGIONS")
    print(region_frame.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
