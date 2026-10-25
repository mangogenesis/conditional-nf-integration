#!/usr/bin/env python3
"""Diagnose conditional importance-weight tails for the frozen 32D NF.

This is a finite-sample diagnostic, not a proof that a mathematical moment
exists.  It reports Hill tail-index estimates over several thresholds and
tracks empirical first through fourth moments as sample size grows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


DIM = 32
COND_DIM = 3
TAIL_FRACTIONS = (0.001, 0.002, 0.005, 0.01, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/adaptive_integrand_all_weak/"
            "sparse_wave_32_final_model.pt"
        ),
    )
    parser.add_argument(
        "--conditions-file",
        type=Path,
        default=Path("results/nf_vs_vegas_160/conditions.csv"),
    )
    parser.add_argument("--condition-index", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--evals", type=int, default=1_000_000)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument(
        "--proposal",
        choices=("pure", "mixture", "marginal_mixture", "copula_mixture"),
        default="pure",
        help=(
            "pure NF, defensive NF/uniform mixture, or defensive mixture "
            "of the full NF with the product of its learned marginals, or "
            "a mixture with a coordinate-wise broadened copula"
        ),
    )
    parser.add_argument(
        "--uniform-fraction",
        type=float,
        help="Override the checkpoint mixture fraction when --proposal mixture.",
    )
    parser.add_argument(
        "--marginal-fraction",
        type=float,
        default=0.1,
        help=(
            "Product-marginal component probability for "
            "--proposal marginal_mixture (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--broad-mixture-fraction",
        type=float,
        default=0.1,
        help=(
            "Probability of the broadened-copula component in the final "
            "NF/broad proposal (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--copula-broad-fraction",
        type=float,
        default=0.1,
        help=(
            "At each autoregressive copula coordinate, probability assigned "
            "to a uniform conditional instead of the learned conditional "
            "(default: 0.1)."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("results/nf_weight_tails_32d_c0")
    )
    return parser.parse_args()


def load_condition(path: Path, index: int) -> np.ndarray:
    frame = pd.read_csv(path)
    if "condition_index" not in frame:
        frame.insert(0, "condition_index", np.arange(len(frame), dtype=int))
    selected = frame[frame["condition_index"].astype(int) == index]
    if len(selected) != 1:
        raise ValueError(f"condition index {index} occurs {len(selected)} times")
    return selected[["c0", "c1", "c2"]].to_numpy(dtype=np.float64)[0]


@torch.inference_mode()
def sample_from_product_marginals(model, n: int, cond: torch.Tensor):
    """Sample exactly from prod_j q_j(x_j | c)."""
    if cond.shape[0] == 1:
        cond = cond.expand(n, -1)
    if cond.shape[0] != n:
        raise ValueError("condition batch must have one row or n rows")

    coordinates = []
    for marginal in model.marginals:
        base = torch.rand(n, 1, device=cond.device, dtype=cond.dtype)
        xj, _ = marginal.forward(base, cond)
        coordinates.append(xj.clamp(0.0, 1.0))
    return torch.cat(coordinates, dim=1)


@torch.inference_mode()
def sample_from_marginal_mixture(
    model, n: int, cond: torch.Tensor, marginal_fraction: float,
):
    """Draw from and evaluate (1-eps) q_full + eps prod_j q_j."""
    if not 0.0 < marginal_fraction < 1.0:
        raise ValueError("marginal fraction must lie strictly between 0 and 1")

    use_marginal = torch.rand(n, device=cond.device) < marginal_fraction
    x = adaptive.sample_from_model(model, n, cond)
    marginal_count = int(use_marginal.sum().item())
    if marginal_count:
        x[use_marginal] = sample_from_product_marginals(
            model, marginal_count, cond[use_marginal]
        )

    log_q_full = model.log_prob(x, cond).reshape(-1)
    _, log_q_marginal = model.x_to_u_and_log_marginal(x, cond)
    log_q_mixture = torch.logaddexp(
        log_q_full + math.log1p(-marginal_fraction),
        log_q_marginal.reshape(-1) + math.log(marginal_fraction),
    )
    return x.detach(), log_q_mixture.detach()


def broadened_copula_log_prob_u(
    model, u: torch.Tensor, cond: torch.Tensor, broad_fraction: float,
) -> torch.Tensor:
    """Log density of a coordinate-wise broadened autoregressive copula."""
    logq = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
    for j in range(1, model.dim):
        context = model.copula.context(u, cond, j)
        _, forward_logdet = model.copula.conditionals[str(j)].inverse(
            u[:, j:j + 1], context
        )
        learned_logq = -forward_logdet.squeeze(-1)
        logq = logq + torch.logaddexp(
            learned_logq + math.log1p(-broad_fraction),
            torch.full_like(learned_logq, math.log(broad_fraction)),
        )
    return logq


@torch.inference_mode()
def sample_from_broadened_copula(
    model, n: int, cond: torch.Tensor, broad_fraction: float,
) -> torch.Tensor:
    """Sample marginals coupled by softened autoregressive conditionals."""
    if not 0.0 < broad_fraction < 1.0:
        raise ValueError("copula broad fraction must lie strictly between 0 and 1")
    if cond.shape[0] == 1:
        cond = cond.expand(n, -1)
    if cond.shape[0] != n:
        raise ValueError("condition batch must have one row or n rows")

    u = torch.zeros(n, model.dim, device=cond.device, dtype=cond.dtype)
    u[:, 0] = torch.rand(n, device=cond.device, dtype=cond.dtype)
    for j in range(1, model.dim):
        context = model.copula.context(u, cond, j)
        base = torch.rand(n, 1, device=cond.device, dtype=cond.dtype)
        learned_u, _ = model.copula.conditionals[str(j)].forward(base, context)
        use_uniform = torch.rand(n, device=cond.device) < broad_fraction
        uj = learned_u.clamp(0.0, 1.0)
        if use_uniform.any():
            uj[use_uniform] = torch.rand(
                int(use_uniform.sum().item()), 1,
                device=cond.device, dtype=cond.dtype,
            )
        u[:, j:j + 1] = uj

    coordinates = []
    for j, marginal in enumerate(model.marginals):
        xj, _ = marginal.forward(u[:, j:j + 1], cond)
        coordinates.append(xj.clamp(0.0, 1.0))
    return torch.cat(coordinates, dim=1)


@torch.inference_mode()
def sample_from_copula_mixture(
    model, n: int, cond: torch.Tensor,
    broad_mixture_fraction: float, copula_broad_fraction: float,
):
    """Draw from and evaluate the full-NF/broadened-copula mixture."""
    if not 0.0 < broad_mixture_fraction < 1.0:
        raise ValueError("broad mixture fraction must lie strictly between 0 and 1")
    use_broad = torch.rand(n, device=cond.device) < broad_mixture_fraction
    x = adaptive.sample_from_model(model, n, cond)
    broad_count = int(use_broad.sum().item())
    if broad_count:
        x[use_broad] = sample_from_broadened_copula(
            model, broad_count, cond[use_broad], copula_broad_fraction
        )

    log_q_full = model.log_prob(x, cond).reshape(-1)
    u, log_marginal = model.x_to_u_and_log_marginal(x, cond)
    log_q_broad = log_marginal + broadened_copula_log_prob_u(
        model, u, cond, copula_broad_fraction
    )
    log_q_mixture = torch.logaddexp(
        log_q_full + math.log1p(-broad_mixture_fraction),
        log_q_broad + math.log(broad_mixture_fraction),
    )
    return x.detach(), log_q_mixture.detach()


@torch.inference_mode()
def sample_logweights(
    model, condition, evaluations, chunk, seed, device,
    proposal, uniform_fraction, marginal_fraction,
    broad_mixture_fraction, copula_broad_fraction,
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
        if proposal == "pure":
            x = adaptive.sample_from_model(model, count, cond)
            logq = model.log_prob(x, cond).reshape(-1)
        elif proposal == "mixture":
            x, logq = adaptive.sample_from_proposal(
                proposal_model=model,
                n=count,
                cond=cond,
                device=device,
                uniform_fraction=uniform_fraction,
            )
        elif proposal == "marginal_mixture":
            x, logq = sample_from_marginal_mixture(
                model=model,
                n=count,
                cond=cond,
                marginal_fraction=marginal_fraction,
            )
        else:
            x, logq = sample_from_copula_mixture(
                model=model,
                n=count,
                cond=cond,
                broad_mixture_fraction=broad_mixture_fraction,
                copula_broad_fraction=copula_broad_fraction,
            )
        logf = adaptive.sparse_wave_32_log_integrand(x, cond)
        parts.append((logf - logq).double().cpu().numpy())
        remaining -= count
    return np.concatenate(parts).astype(np.float64, copy=False)


def hill_estimate(logw: np.ndarray, fraction: float):
    n = len(logw)
    k = max(20, int(round(fraction * n)))
    k = min(k, n - 1)
    partitioned = np.partition(logw, n - k - 1)
    threshold = float(partitioned[n - k - 1])
    top = partitioned[n - k:]
    xi = float(np.mean(top - threshold))
    standard_error = xi / math.sqrt(k)
    return k, threshold, xi, standard_error


def checkpoints(total: int) -> list[int]:
    candidates = [
        10_000, 30_000, 100_000, 300_000, 1_000_000,
        3_000_000, 10_000_000,
    ]
    values = [value for value in candidates if value < total]
    values.append(total)
    return sorted(set(values))


def moment_row(logw: np.ndarray, count: int, replicate: int) -> dict:
    values = logw[:count]
    logs = {
        order: float(logsumexp(order * values) - math.log(count))
        for order in (1, 2, 3, 4)
    }
    log_sum = float(logsumexp(values))
    log_sum2 = float(logsumexp(2.0 * values))

    def safe_exp(value: float) -> float:
        return math.exp(value) if value < 700 else math.inf

    return {
        "replicate": replicate,
        "samples": count,
        "first_moment": safe_exp(logs[1]),
        "second_moment": safe_exp(logs[2]),
        "third_moment": safe_exp(logs[3]),
        "fourth_moment": safe_exp(logs[4]),
        "log_first_moment": logs[1],
        "log_second_moment": logs[2],
        "log_third_moment": logs[3],
        "log_fourth_moment": logs[4],
        "normalized_ess": math.exp(2.0 * log_sum - math.log(count) - log_sum2),
        "max_weight_share": math.exp(float(np.max(values)) - log_sum),
        "maximum_log_weight": float(np.max(values)),
    }


def regime_label(xi: float) -> str:
    if xi < 0.25:
        return "finite fourth moment suggested"
    if xi < 0.50:
        return "finite second but unstable second-moment estimation suggested"
    if xi < 1.00:
        return "infinite second moment suggested"
    return "even first moment may be problematic"


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    condition = load_condition(args.conditions_file, args.condition_index)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, checkpoint_uniform_fraction, checkpoint = common.load_nf(
        args.checkpoint, device
    )
    uniform_fraction = (
        checkpoint_uniform_fraction
        if args.uniform_fraction is None else args.uniform_fraction
    )
    if args.proposal == "mixture" and not 0.0 < uniform_fraction < 1.0:
        raise ValueError("mixture uniform fraction must lie in (0,1)")
    if args.proposal == "marginal_mixture" and not 0.0 < args.marginal_fraction < 1.0:
        raise ValueError("marginal mixture fraction must lie in (0,1)")
    if args.proposal == "copula_mixture":
        if not 0.0 < args.broad_mixture_fraction < 1.0:
            raise ValueError("broad mixture fraction must lie in (0,1)")
        if not 0.0 < args.copula_broad_fraction < 1.0:
            raise ValueError("copula broad fraction must lie in (0,1)")
    adaptive.DIM = DIM

    tail_rows, moment_rows = [], []
    for replicate in range(args.replicates):
        seed = args.seed + 1_000_003 * args.condition_index + 10_007 * replicate
        logw = sample_logweights(
            model, condition, args.evals, args.gpu_chunk, seed, device,
            args.proposal, uniform_fraction, args.marginal_fraction,
            args.broad_mixture_fraction, args.copula_broad_fraction,
        )
        for fraction in TAIL_FRACTIONS:
            k, threshold, xi, se = hill_estimate(logw, fraction)
            tail_rows.append({
                "condition_index": args.condition_index,
                "replicate": replicate,
                "seed": seed,
                "evaluations": args.evals,
                "tail_fraction": fraction,
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
        print(
            f"replicate {replicate + 1}/{args.replicates}: "
            f"I={moment_rows[-1]['first_moment']:.7f} "
            f"M2={moment_rows[-1]['second_moment']:.6g} "
            f"ESS={moment_rows[-1]['normalized_ess']:.4g} "
            f"max-share={moment_rows[-1]['max_weight_share']:.4g}",
            flush=True,
        )

    tails = pd.DataFrame(tail_rows)
    moments = pd.DataFrame(moment_rows)
    tails.to_csv(args.output / "hill_estimates.csv", index=False)
    moments.to_csv(args.output / "moment_convergence.csv", index=False)

    grouped = tails.groupby("tail_fraction", as_index=False).agg(
        replicates=("replicate", "count"),
        mean_hill_xi=("hill_xi", "mean"),
        sd_hill_xi=("hill_xi", "std"),
        minimum_hill_xi=("hill_xi", "min"),
        maximum_hill_xi=("hill_xi", "max"),
    )
    central = grouped[
        grouped["tail_fraction"].isin([0.002, 0.005, 0.01])
    ]["mean_hill_xi"]
    representative_xi = float(central.median())
    summary = pd.DataFrame([{
        "condition_index": args.condition_index,
        "replicates": args.replicates,
        "evaluations_per_replicate": args.evals,
        "representative_hill_xi": representative_xi,
        "suggested_regime": regime_label(representative_xi),
        "minimum_final_ess": float(
            moments[moments["samples"] == args.evals]["normalized_ess"].min()
        ),
        "maximum_final_weight_share": float(
            moments[moments["samples"] == args.evals]["max_weight_share"].max()
        ),
        "minimum_final_second_moment": float(
            moments[moments["samples"] == args.evals]["second_moment"].min()
        ),
        "maximum_final_second_moment": float(
            moments[moments["samples"] == args.evals]["second_moment"].max()
        ),
    }])
    grouped.to_csv(args.output / "hill_summary_by_threshold.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "conditions_file": str(args.conditions_file),
            "condition_index": args.condition_index,
            "condition": condition.tolist(),
            "proposal": args.proposal,
            "uniform_fraction": uniform_fraction,
            "marginal_fraction": args.marginal_fraction,
            "broad_mixture_fraction": args.broad_mixture_fraction,
            "copula_broad_fraction": args.copula_broad_fraction,
            "replicates": args.replicates,
            "evaluations": args.evals,
            "tail_fractions": TAIL_FRACTIONS,
            "training_performed": False,
            "warning": "finite-sample tail diagnostics do not prove moment existence",
            "checkpoint_benchmark": checkpoint.get("benchmark"),
        }, stream, indent=2)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for replicate, group in tails.groupby("replicate"):
        ax.plot(
            100.0 * group["tail_fraction"], group["hill_xi"],
            "o-", alpha=0.75, label=f"replicate {replicate}",
        )
    ax.axhline(0.25, color="tab:orange", linestyle="--", label="fourth-moment boundary")
    ax.axhline(0.50, color="tab:red", linestyle="--", label="second-moment boundary")
    ax.set(xlabel="Upper-tail fraction (%)", ylabel="Hill tail index xi")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "hill_tail_index.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for replicate, group in moments.groupby("replicate"):
        ax.plot(
            group["samples"], group["second_moment"], "o-",
            alpha=0.75, label=f"replicate {replicate}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Samples", ylabel="Empirical second moment of W")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "second_moment_convergence.png", dpi=180)
    plt.close(fig)

    print("\nWEIGHT-TAIL DIAGNOSTIC", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nHILL THRESHOLD SENSITIVITY", flush=True)
    print(grouped.to_string(index=False), flush=True)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
