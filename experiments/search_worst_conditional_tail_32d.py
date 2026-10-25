#!/usr/bin/env python3
"""Automatically mine conditions with the worst frozen-proposal tails.

The search uses successive halving.  A broad pool of conditions receives a
small sampling budget; only the riskiest conditions receive the next, larger
budget.  The primary score is normalized log M2 (Renyi-2 divergence).  A
one-sided, importance-weighted Kolmogorov discrepancy is added as a bounded
shape diagnostic:

    KS_tail = sup_t [F_q(R <= t) - F_p(R <= t)],  R = log(f / q).

Thus KS helps find conditions where target mass has shifted toward large
importance ratios, while log M2 remains the quantity that determines ranking
and eventual optimization.  No exact target samples or true integral are
used.
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
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_geometry_ladder_coverage_32d as frozen  # noqa: E402


COND_DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--anchor", type=float, nargs=COND_DIM, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--condition-radius", type=float, default=0.05)
    parser.add_argument("--candidate-conditions", type=int, default=256)
    parser.add_argument(
        "--condition-pool-file", type=Path,
        help=(
            "Evaluate exactly the c0,c1,c2 conditions in this CSV instead of "
            "generating a new random pool. Useful for comparable re-audits."
        ),
    )
    parser.add_argument(
        "--include-conditions-file", type=Path,
        help="Optional c0,c1,c2 CSV whose conditions are inserted in the pool.",
    )
    parser.add_argument(
        "--cumulative-budgets", type=int, nargs="+",
        default=(4096, 16384, 65536, 262144),
        help="Cumulative samples per surviving condition at each round.",
    )
    parser.add_argument(
        "--keep-fraction", type=float, default=0.25,
        help="Fraction promoted after each non-final round.",
    )
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument(
        "--ks-weight", type=float, default=1.0,
        help="Score = normalized log M2 + ks_weight * tail KS.",
    )
    parser.add_argument(
        "--proposal", choices=("pure", "mixture"), default="pure"
    )
    parser.add_argument("--uniform-fraction", type=float, default=0.02)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.condition_pool_file is not None
        and args.include_conditions_file is not None
    ):
        parser.error(
            "use either --condition-pool-file or --include-conditions-file"
        )
    if args.candidate_conditions <= 0 or args.finalists <= 0:
        parser.error("candidate conditions and finalists must be positive")
    if not 0.0 < args.keep_fraction < 1.0:
        parser.error("--keep-fraction must lie in (0,1)")
    if args.ks_weight < 0.0:
        parser.error("--ks-weight must be nonnegative")
    if args.condition_radius <= 0.0:
        parser.error("--condition-radius must be positive")
    if any(value <= 1 for value in args.cumulative_budgets):
        parser.error("all cumulative budgets must exceed one")
    if sorted(set(args.cumulative_budgets)) != list(args.cumulative_budgets):
        parser.error("cumulative budgets must be strictly increasing")
    if args.finalists > args.candidate_conditions:
        parser.error("finalists cannot exceed candidate conditions")
    if args.proposal == "mixture" and not 0.0 < args.uniform_fraction < 1.0:
        parser.error("--uniform-fraction must lie in (0,1)")
    return args


def condition_pool(args: argparse.Namespace) -> np.ndarray:
    if args.condition_pool_file is not None:
        frame = pd.read_csv(args.condition_pool_file)
        missing = {"c0", "c1", "c2"}.difference(frame.columns)
        if missing:
            raise ValueError(f"condition pool lacks columns {sorted(missing)}")
        values = frame[["c0", "c1", "c2"]].to_numpy(np.float64)
        if len(values) < args.finalists:
            raise ValueError(
                f"condition pool has {len(values)} rows but "
                f"--finalists={args.finalists}"
            )
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("condition pool must lie in [0,1]^3")
        return values
    rng = np.random.default_rng(args.seed)
    anchor = np.asarray(args.anchor, dtype=np.float64)
    lower = np.maximum(anchor - args.condition_radius, 0.0)
    upper = np.minimum(anchor + args.condition_radius, 1.0)
    inserted = []
    if args.include_conditions_file is not None:
        frame = pd.read_csv(args.include_conditions_file)
        missing = {"c0", "c1", "c2"}.difference(frame.columns)
        if missing:
            raise ValueError(
                f"condition file lacks columns {sorted(missing)}"
            )
        inserted.extend(frame[["c0", "c1", "c2"]].to_numpy(float))
    inserted.append(anchor)
    result = []
    seen = set()
    for value in inserted:
        value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
        key = tuple(np.round(value, 12))
        if key not in seen:
            result.append(value)
            seen.add(key)
    while len(result) < args.candidate_conditions:
        value = rng.uniform(lower, upper)
        key = tuple(np.round(value, 12))
        if key not in seen:
            result.append(value)
            seen.add(key)
    return np.asarray(result[:args.candidate_conditions])


def tail_metrics(logw: np.ndarray, ks_weight: float) -> dict[str, float]:
    n = len(logw)
    log_n = math.log(n)
    log_i = float(logsumexp(logw) - log_n)
    log_m2 = float(logsumexp(2.0 * logw) - log_n)
    normalized_log_m2 = log_m2 - 2.0 * log_i
    ordered = np.sort(logw)
    target_weight = np.exp(ordered - float(logsumexp(ordered)))
    target_cdf = np.cumsum(target_weight)
    proposal_cdf = np.arange(1, n + 1, dtype=np.float64) / n
    tail_ks = float(np.max(proposal_cdf - target_cdf))
    two_sided_ks = float(np.max(np.abs(proposal_cdf - target_cdf)))
    max_share = float(np.exp(logw.max() - float(logsumexp(logw))))
    descending = ordered[::-1]
    k = min(max(2, int(math.ceil(0.005 * n))), n - 1)
    hill = float(np.mean(descending[:k] - descending[k]))
    return {
        "samples": n,
        "log_first_moment": log_i,
        "first_moment": float(math.exp(log_i)),
        "log_second_moment": log_m2,
        "normalized_log_m2": normalized_log_m2,
        "normalized_ess": float(math.exp(-normalized_log_m2)),
        "tail_ks": tail_ks,
        "two_sided_ks": two_sided_ks,
        "maximum_weight_share": max_share,
        "hill_xi_005": hill,
        "search_score": normalized_log_m2 + ks_weight * tail_ks,
    }


@torch.inference_mode()
def sample_log_weights(
    adaptive, model, benchmark, condition, count, mixture_fraction,
    proposal_kind, chunk, device,
) -> np.ndarray:
    pieces = []
    condition_tensor = torch.as_tensor(
        condition, dtype=torch.float32, device=device
    )
    remaining = count
    while remaining:
        current = min(remaining, chunk)
        cond = condition_tensor.expand(current, COND_DIM)
        if proposal_kind == "pure":
            x = adaptive.sample_from_model(model, current, cond)
            logq = model.log_prob(x, cond).reshape(-1)
        else:
            x, logq = adaptive.sample_from_proposal(
                proposal_model=model,
                n=current,
                cond=cond,
                device=device,
                uniform_fraction=mixture_fraction,
            )
        logf = adaptive.target_log_integrand(benchmark, x, cond).reshape(-1)
        pieces.append((logf - logq).double().cpu().numpy())
        remaining -= current
    return np.concatenate(pieces)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    adaptive, model, _, benchmark, checkpoint_fraction = frozen.load_model(
        args.checkpoint, device
    )
    mixture_fraction = (
        args.uniform_fraction if args.proposal == "mixture"
        else checkpoint_fraction
    )
    conditions = condition_pool(args)
    pd.DataFrame(conditions, columns=["c0", "c1", "c2"]).rename_axis(
        "condition_index"
    ).reset_index().to_csv(args.output / "candidate_conditions.csv", index=False)

    active = list(range(len(conditions)))
    samples = {index: np.empty(0, dtype=np.float64) for index in active}
    history = []
    previous_budget = 0
    for round_index, budget in enumerate(args.cumulative_budgets):
        additional = budget - previous_budget
        print(
            f"\nTAIL SEARCH ROUND {round_index + 1}: "
            f"conditions={len(active)} cumulative budget={budget:,}",
            flush=True,
        )
        rows = []
        for position, condition_index in enumerate(active, start=1):
            torch.manual_seed(
                args.seed + 1_000_003 * round_index + 10_007 * condition_index
            )
            if device.type == "cuda":
                torch.cuda.manual_seed_all(
                    args.seed + 1_000_003 * round_index
                    + 10_007 * condition_index
                )
            new = sample_log_weights(
                adaptive, model, benchmark, conditions[condition_index],
                additional, mixture_fraction, args.proposal, args.chunk, device,
            )
            samples[condition_index] = np.concatenate(
                [samples[condition_index], new]
            )
            metrics = tail_metrics(samples[condition_index], args.ks_weight)
            row = {
                "round": round_index + 1,
                "condition_index": condition_index,
                "c0": conditions[condition_index, 0],
                "c1": conditions[condition_index, 1],
                "c2": conditions[condition_index, 2],
                **metrics,
            }
            rows.append(row)
            history.append(row)
            if position % max(1, min(16, len(active))) == 0:
                print(
                    f"  evaluated {position:4d}/{len(active)} conditions",
                    flush=True,
                )
        frame = pd.DataFrame(rows).sort_values(
            ["search_score", "normalized_log_m2"], ascending=False
        )
        frame.insert(1, "rank", np.arange(1, len(frame) + 1))
        frame.to_csv(
            args.output / f"round_{round_index + 1:02d}.csv", index=False
        )
        leader = frame.iloc[0]
        print(
            f"  worst: c={int(leader.condition_index)} "
            f"logR2={leader.normalized_log_m2:.4f} "
            f"tailKS={leader.tail_ks:.4f} "
            f"xi={leader.hill_xi_005:.3f} "
            f"share={leader.maximum_weight_share:.4g}",
            flush=True,
        )
        if round_index + 1 < len(args.cumulative_budgets):
            keep = max(
                args.finalists,
                int(math.ceil(len(active) * args.keep_fraction)),
            )
            active = frame.head(keep)["condition_index"].astype(int).tolist()
        else:
            active = frame.head(args.finalists)[
                "condition_index"
            ].astype(int).tolist()
        previous_budget = budget

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(args.output / "search_history.csv", index=False)
    final_frame = pd.read_csv(
        args.output / f"round_{len(args.cumulative_budgets):02d}.csv"
    ).head(args.finalists)
    final_frame.to_csv(args.output / "worst_conditions.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        serializable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        json.dump({
            **serializable,
            "benchmark": benchmark,
            "ranking_primary": "normalized_log_m2",
            "ranking_secondary": "importance_weighted_tail_ks",
            "true_integral_used": False,
            "exact_target_sampler_used": False,
        }, stream, indent=2)
    print("\nWORST CONDITIONAL TAILS")
    print(final_frame[[
        "rank", "condition_index", "c0", "c1", "c2",
        "normalized_log_m2", "tail_ks", "maximum_weight_share",
        "hill_xi_005",
    ]].to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
