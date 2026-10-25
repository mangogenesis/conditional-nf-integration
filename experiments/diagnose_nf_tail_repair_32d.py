#!/usr/bin/env python3
"""Pilot-trained local tail repair for the frozen conditional 32D NF.

The pilot and production samples are independent.  High-importance-weight
pilot points become centers of a mixture of truncated Gaussian kernels.  The
production proposal is the exact mixture

    (1 - epsilon) q_nf + epsilon q_tail.

This is an exploratory diagnostic, not a replacement for independent
validation across conditions.
"""

from __future__ import annotations

import argparse
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
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_adaptive_integrand_structure_search as adaptive  # noqa: E402
import run_final_checkpoint_integration_comparison as common  # noqa: E402
from diagnose_nf_weight_tails_32d import (  # noqa: E402
    COND_DIM,
    DIM,
    TAIL_FRACTIONS,
    checkpoints,
    hill_estimate,
    load_condition,
    moment_row,
    regime_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "results/adaptive_integrand_all_weak/"
            "sparse_wave_32_final_model.pt"
        ),
    )
    parser.add_argument(
        "--conditions-file", type=Path,
        default=Path("results/nf_vs_vegas_160/conditions.csv"),
    )
    parser.add_argument("--condition-index", type=int, default=0)
    parser.add_argument("--pilot-evals", type=int, default=1_000_000)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--bandwidth", type=float, default=0.03)
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--evals", type=int, default=1_000_000)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--anchor-chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/nf_weight_tails_32d_c0_tail_repair"),
    )
    return parser.parse_args()


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def normal_icdf(p: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


class TruncatedKernelMixture:
    """Equally weighted product-truncated-normal kernels on [0,1]^d."""

    def __init__(
        self, centers: torch.Tensor, bandwidth: float, anchor_chunk: int,
    ) -> None:
        if centers.ndim != 2:
            raise ValueError("centers must be a two-dimensional tensor")
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        self.centers = centers
        self.bandwidth = float(bandwidth)
        self.anchor_chunk = int(anchor_chunk)
        lower = -centers / self.bandwidth
        upper = (1.0 - centers) / self.bandwidth
        self.cdf_lower = normal_cdf(lower)
        self.cdf_upper = normal_cdf(upper)
        self.log_normalizer = torch.log(
            (self.cdf_upper - self.cdf_lower).clamp_min(1e-12)
        )

    @property
    def count(self) -> int:
        return self.centers.shape[0]

    @torch.inference_mode()
    def sample(self, n: int) -> torch.Tensor:
        indices = torch.randint(self.count, (n,), device=self.centers.device)
        centers = self.centers[indices]
        lo = self.cdf_lower[indices]
        hi = self.cdf_upper[indices]
        probability = lo + torch.rand_like(centers) * (hi - lo)
        eps = torch.finfo(probability.dtype).eps
        z = normal_icdf(probability.clamp(eps, 1.0 - eps))
        return (centers + self.bandwidth * z).clamp(0.0, 1.0)

    @torch.inference_mode()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        partials = []
        constant = -math.log(self.bandwidth) - 0.5 * math.log(2.0 * math.pi)
        for start in range(0, self.count, self.anchor_chunk):
            stop = min(start + self.anchor_chunk, self.count)
            centers = self.centers[start:stop]
            z = (x[:, None, :] - centers[None, :, :]) / self.bandwidth
            log_component = (
                constant - 0.5 * z.square()
                - self.log_normalizer[start:stop][None, :, :]
            ).sum(dim=2)
            partials.append(log_component)
        return torch.logsumexp(torch.cat(partials, dim=1), dim=1) - math.log(
            self.count
        )


@torch.inference_mode()
def collect_pilot(model, condition, evaluations, chunk, seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    best_x, best_logw = [], []
    remaining = evaluations
    while remaining:
        count = min(remaining, chunk)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)
        x = adaptive.sample_from_model(model, count, cond)
        logq = model.log_prob(x, cond).reshape(-1)
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)
        best_x.append(x.detach().cpu())
        best_logw.append((logf - logq).detach().cpu())
        remaining -= count
    return torch.cat(best_x), torch.cat(best_logw)


def choose_diverse_anchors(
    pilot_x: torch.Tensor, pilot_logw: torch.Tensor, count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose high-weight points while avoiding exact local duplicates."""
    if count > len(pilot_x):
        raise ValueError("more anchors requested than pilot observations")
    candidate_count = min(len(pilot_x), max(20 * count, count))
    values, indices = torch.topk(pilot_logw, candidate_count)
    candidates = pilot_x[indices]

    # Greedy farthest-point selection seeded by the largest-weight point.
    selected = [0]
    minimum_distance = torch.full((candidate_count,), torch.inf)
    for _ in range(1, count):
        newest = candidates[selected[-1]]
        distance = (candidates - newest).square().mean(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        # Retain emphasis on the upper tail while rewarding coverage.
        score = minimum_distance.sqrt() * (1.0 + 0.05 * (values - values.min()))
        score[selected] = -torch.inf
        selected.append(int(torch.argmax(score)))
    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    return candidates[selected_tensor], values[selected_tensor]


@torch.inference_mode()
def production_logweights(
    model, tail, condition, evaluations, chunk, seed, device, tail_fraction,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    parts = []
    remaining = evaluations
    while remaining:
        count = min(remaining, chunk)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)
        use_tail = torch.rand(count, device=device) < tail_fraction
        x = adaptive.sample_from_model(model, count, cond)
        if use_tail.any():
            x[use_tail] = tail.sample(int(use_tail.sum().item()))
        log_nf = model.log_prob(x, cond).reshape(-1)
        log_tail = tail.log_prob(x)
        log_proposal = torch.logaddexp(
            log_nf + math.log1p(-tail_fraction),
            log_tail + math.log(tail_fraction),
        )
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)
        parts.append((logf - log_proposal).double().cpu().numpy())
        remaining -= count
    return np.concatenate(parts).astype(np.float64, copy=False)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.tail_fraction < 1.0:
        raise ValueError("tail fraction must lie in (0,1)")
    args.output.mkdir(parents=True, exist_ok=True)
    condition = load_condition(args.conditions_file, args.condition_index)
    device = torch.device(args.device)
    model, _, checkpoint = common.load_nf(args.checkpoint, device)
    adaptive.DIM = DIM

    pilot_x, pilot_logw = collect_pilot(
        model, condition, args.pilot_evals, args.gpu_chunk,
        args.seed, device,
    )
    centers, center_logw = choose_diverse_anchors(
        pilot_x, pilot_logw, args.anchors
    )
    centers = centers.to(device)
    tail = TruncatedKernelMixture(
        centers, args.bandwidth, args.anchor_chunk
    )
    pd.DataFrame({
        "anchor": np.arange(args.anchors),
        "pilot_log_weight": center_logw.numpy(),
        **{
            f"x{j}": centers[:, j].detach().cpu().numpy()
            for j in range(DIM)
        },
    }).to_csv(args.output / "tail_anchors.csv", index=False)
    print(
        f"pilot complete: max logW={float(pilot_logw.max()):.4f}; "
        f"anchors={args.anchors}; bandwidth={args.bandwidth}",
        flush=True,
    )

    tail_rows, moment_rows = [], []
    for replicate in range(args.replicates):
        seed = args.seed + 10_000_019 + 10_007 * replicate
        logw = production_logweights(
            model, tail, condition, args.evals, args.gpu_chunk, seed,
            device, args.tail_fraction,
        )
        for fraction in TAIL_FRACTIONS:
            k, threshold, xi, se = hill_estimate(logw, fraction)
            tail_rows.append({
                "condition_index": args.condition_index,
                "replicate": replicate,
                "seed": seed,
                "evaluations": args.evals,
                "tail_fraction_threshold": fraction,
                "tail_count": k,
                "threshold_log_weight": threshold,
                "hill_xi": xi,
                "hill_standard_error": se,
                "suggested_regime": regime_label(xi),
            })
        for count in checkpoints(args.evals):
            row = moment_row(logw, count, replicate)
            row["condition_index"] = args.condition_index
            moment_rows.append(row)
        final = moment_rows[-1]
        print(
            f"replicate {replicate + 1}/{args.replicates}: "
            f"I={final['first_moment']:.7f} "
            f"M2={final['second_moment']:.6g} "
            f"ESS={final['normalized_ess']:.4g} "
            f"max-share={final['max_weight_share']:.4g}",
            flush=True,
        )

    tails = pd.DataFrame(tail_rows)
    moments = pd.DataFrame(moment_rows)
    tails.to_csv(args.output / "hill_estimates.csv", index=False)
    moments.to_csv(args.output / "moment_convergence.csv", index=False)
    grouped = tails.groupby("tail_fraction_threshold", as_index=False).agg(
        replicates=("replicate", "count"),
        mean_hill_xi=("hill_xi", "mean"),
        sd_hill_xi=("hill_xi", "std"),
        minimum_hill_xi=("hill_xi", "min"),
        maximum_hill_xi=("hill_xi", "max"),
    )
    representative = float(grouped[
        grouped["tail_fraction_threshold"].isin([0.002, 0.005, 0.01])
    ]["mean_hill_xi"].median())
    final = moments[moments["samples"] == args.evals]
    summary = pd.DataFrame([{
        "condition_index": args.condition_index,
        "replicates": args.replicates,
        "evaluations_per_replicate": args.evals,
        "pilot_evaluations": args.pilot_evals,
        "anchors": args.anchors,
        "bandwidth": args.bandwidth,
        "tail_mixture_fraction": args.tail_fraction,
        "representative_hill_xi": representative,
        "suggested_regime": regime_label(representative),
        "minimum_final_ess": float(final["normalized_ess"].min()),
        "maximum_final_weight_share": float(final["max_weight_share"].max()),
        "minimum_final_second_moment": float(final["second_moment"].min()),
        "maximum_final_second_moment": float(final["second_moment"].max()),
    }])
    grouped.to_csv(args.output / "hill_summary_by_threshold.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "condition_index": args.condition_index,
            "condition": condition.tolist(),
            "pilot_evaluations": args.pilot_evals,
            "anchors": args.anchors,
            "bandwidth": args.bandwidth,
            "tail_fraction": args.tail_fraction,
            "replicates": args.replicates,
            "evaluations": args.evals,
            "pilot_production_independent": True,
            "training_performed": False,
            "checkpoint_benchmark": checkpoint.get("benchmark"),
        }, stream, indent=2)
    print("\nTAIL-REPAIR DIAGNOSTIC", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nHILL THRESHOLD SENSITIVITY", flush=True)
    print(grouped.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
