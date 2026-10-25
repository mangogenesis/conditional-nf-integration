#!/usr/bin/env python3
"""Blind validation of a frozen NF using low-density uniform candidates.

The learned proposal is used only to define where *not* to validate.  A lower
tail threshold is calibrated from ordinary NF samples.  Uniform candidates
whose NF log density lies below that threshold form the proposal complement.
The target oracle is evaluated only on this complement.  Candidate points are
ranked by

    2 log f(x|c) - log q(x|c),

the logarithm of their local contribution to the importance-sampling second
moment.  No target sampler, normalization, structure labels, or SMC is used.
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
    parser.add_argument("--calibration-samples", type=int, default=200_000)
    parser.add_argument("--uniform-candidates", type=int, default=2_000_000)
    parser.add_argument("--complement-quantile", type=float, default=0.01)
    parser.add_argument("--keep", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.calibration_samples,
        args.uniform_candidates,
        args.keep,
        args.chunk,
    ) <= 0:
        parser.error("sample counts and chunk must be positive")
    if not 0 < args.complement_quantile < 1:
        parser.error("--complement-quantile must lie strictly between 0 and 1")
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("fixed condition must lie in [0,1]^3")
    return args


def safe_exp(value):
    return math.exp(value) if value < math.log(sys.float_info.max) else math.inf


@torch.inference_mode()
def calibrate(adaptive, model, condition, args):
    remaining = args.calibration_samples
    logq_rows = []
    logw_rows = []
    while remaining:
        count = min(remaining, args.chunk)
        cond = condition.expand(count, -1)
        x = adaptive.sample_from_model(model, count, cond)
        logq = model.log_prob(x, cond).reshape(-1)
        logf = heterogeneous.heterogeneous_log_integrand(x, cond)
        logq_rows.append(logq.cpu())
        logw_rows.append((logf - logq).cpu())
        remaining -= count
    logq = torch.cat(logq_rows).double()
    logw = torch.cat(logw_rows).double()
    threshold = float(torch.quantile(logq, args.complement_quantile))
    finite = torch.isfinite(logw)
    finite_logw = logw[finite]
    logn = math.log(logw.numel())
    log_i = (
        float(torch.logsumexp(finite_logw, dim=0)) - logn
        if bool(finite.any()) else -math.inf
    )
    log_m2 = (
        float(torch.logsumexp(2 * finite_logw, dim=0)) - logn
        if bool(finite.any()) else math.inf
    )
    ess = math.exp(2 * log_i - log_m2) if math.isfinite(log_m2) else 0.0
    return threshold, {
        "nf_logq_complement_threshold": threshold,
        "nf_logq_minimum": float(logq.min()),
        "nf_logq_median": float(logq.median()),
        "nf_logq_maximum": float(logq.max()),
        "nf_integral_estimate": safe_exp(log_i),
        "nf_m2_estimate": safe_exp(log_m2),
        "nf_normalized_ess": ess,
        "nf_maximum_log_weight": float(finite_logw.max()),
    }


def merge_top(previous, x, logq, logf, limit):
    score = 2 * logf - logq
    if previous is not None:
        x = torch.cat((previous["x"], x), dim=0)
        logq = torch.cat((previous["logq"], logq), dim=0)
        logf = torch.cat((previous["logf"], logf), dim=0)
        score = torch.cat((previous["score"], score), dim=0)
    count = min(limit, len(score))
    indices = torch.topk(score, count, sorted=True).indices
    return {
        "x": x[indices],
        "logq": logq[indices],
        "logf": logf[indices],
        "score": score[indices],
    }


@torch.inference_mode()
def scan_complement(model, condition, threshold, args, device):
    remaining = args.uniform_candidates
    candidates = 0
    complement = 0
    finite_target = 0
    top = None
    maximum_logf = -math.inf
    maximum_log_ratio = -math.inf
    while remaining:
        count = min(remaining, args.chunk)
        x = torch.rand(
            count, heterogeneous.DIMENSION, device=device, dtype=torch.float32
        )
        cond = condition.expand(count, -1)
        logq = model.log_prob(x, cond).reshape(-1)
        mask = logq <= threshold
        candidates += count
        complement += int(mask.sum())
        if bool(mask.any()):
            selected_x = x[mask]
            selected_cond = cond[mask]
            selected_logq = logq[mask]
            selected_logf = heterogeneous.heterogeneous_log_integrand(
                selected_x, selected_cond
            )
            finite = torch.isfinite(selected_logf)
            finite_target += int(finite.sum())
            if bool(finite.any()):
                selected_x = selected_x[finite].cpu()
                selected_logq = selected_logq[finite].double().cpu()
                selected_logf = selected_logf[finite].double().cpu()
                maximum_logf = max(maximum_logf, float(selected_logf.max()))
                maximum_log_ratio = max(
                    maximum_log_ratio,
                    float((selected_logf - selected_logq).max()),
                )
                top = merge_top(
                    top,
                    selected_x,
                    selected_logq,
                    selected_logf,
                    args.keep,
                )
        remaining -= count
        if candidates % max(args.chunk, 250_000) == 0 or remaining == 0:
            print(
                f"uniform scan {candidates:,}/{args.uniform_candidates:,} "
                f"complement={complement / candidates:.4f} "
                f"finite-target={finite_target:,}",
                flush=True,
            )
    return top, {
        "uniform_candidates": candidates,
        "complement_candidates": complement,
        "complement_fraction": complement / candidates,
        "finite_target_candidates": finite_target,
        "finite_target_fraction_of_uniform": finite_target / candidates,
        "maximum_complement_logf": maximum_logf,
        "maximum_complement_log_ratio": maximum_log_ratio,
    }


def save_top(top, output):
    if top is None:
        return 0
    data = {
        "rank": torch.arange(1, len(top["score"]) + 1).numpy(),
        "logf": top["logf"].numpy(),
        "logq": top["logq"].numpy(),
        "log_f_over_q": (top["logf"] - top["logq"]).numpy(),
        "log_second_moment_contribution": top["score"].numpy(),
    }
    for coordinate in range(heterogeneous.DIMENSION):
        data[f"x{coordinate}"] = top["x"][:, coordinate].numpy()
    pd.DataFrame(data).to_csv(output / "top_complement_candidates.csv", index=False)
    torch.save(top, output / "top_complement_candidates.pt")
    return len(top["score"])


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
    threshold, calibration = calibrate(adaptive, model, condition, args)
    print("\nNF CALIBRATION")
    print(json.dumps(calibration, indent=2))
    top, scan = scan_complement(model, condition, threshold, args, device)
    retained = save_top(top, args.output)
    summary = {
        "benchmark": heterogeneous.BENCHMARK,
        "checkpoint": str(args.checkpoint),
        "condition": list(args.fixed_condition),
        "selection": "uniform candidates with log q below the calibrated NF lower-tail threshold",
        "complement_quantile": args.complement_quantile,
        "retained_top_candidates": retained,
        **calibration,
        **scan,
    }
    if top is not None:
        summary.update({
            "best_log_second_moment_contribution": float(top["score"][0]),
            "best_logf": float(top["logf"][0]),
            "best_logq": float(top["logq"][0]),
            "best_log_f_over_q": float(top["logf"][0] - top["logq"][0]),
        })
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nNF-COMPLEMENT VALIDATION")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
