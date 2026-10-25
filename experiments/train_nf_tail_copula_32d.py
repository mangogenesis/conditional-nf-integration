#!/usr/bin/env python3
"""Fine-tune a copied 32D copula on stabilized residual-measure samples.

The original proposal is never modified.  Its marginal transforms are frozen,
and only the copied autoregressive copula is trained.  Residual importance
weights are tempered automatically until they reach a requested training ESS;
this prevents a handful of extreme observations from being memorized.
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
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_final_checkpoint_integration_comparison as common  # noqa: E402


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
        "--training-data", type=Path,
        default=Path(
            "results/nf_tail_training_32d_c0/tail_training_points.pt"
        ),
    )
    parser.add_argument("--target-training-ess", type=float, default=5000.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--anchor-penalty", type=float, default=1e-5)
    parser.add_argument("--precompute-chunk", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/nf_tail_copula_32d_c0"),
    )
    return parser.parse_args()


def normalized_weights(log_weights: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.softmax(beta * log_weights.double(), dim=0)


def effective_sample_size(weights: torch.Tensor) -> float:
    return float(1.0 / weights.square().sum())


def select_temperature(
    log_weights: torch.Tensor, target_ess: float,
) -> tuple[float, torch.Tensor]:
    n = len(log_weights)
    target = min(max(float(target_ess), 1.0), float(n))
    full = normalized_weights(log_weights, 1.0)
    if effective_sample_size(full) >= target:
        return 1.0, full
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        weights = normalized_weights(log_weights, mid)
        if effective_sample_size(weights) > target:
            low = mid
        else:
            high = mid
    beta = 0.5 * (low + high)
    return beta, normalized_weights(log_weights, beta)


def residual_log_weights(data: dict) -> tuple[torch.Tensor, dict[int, float]]:
    logw_nf = data["logw_nf"].double()
    logw_explorer = data["logw_explorer"].double()
    condition_index = data["condition_index"].long()
    log_residual = torch.full_like(logw_nf, -torch.inf)
    thresholds = {}
    for value in torch.unique(condition_index).tolist():
        mask = condition_index == value
        tau_log = float(logw_nf[mask].min())
        ratio = torch.exp((tau_log - logw_nf[mask]).clamp(max=0.0))
        # At the exact cutoff log1p(-1)=-inf, assigning zero residual mass.
        log_residual[mask] = logw_explorer[mask] + torch.log1p(-ratio)
        thresholds[int(value)] = math.exp(tau_log)
    return log_residual, thresholds


def balanced_tempered_weights(
    log_weights: torch.Tensor,
    condition_index: torch.Tensor,
    target_total_ess: float,
) -> tuple[torch.Tensor, dict[int, float], dict[int, float]]:
    """Temper within conditions and give every condition equal total mass."""
    values = sorted(int(v) for v in torch.unique(condition_index).tolist())
    target_each = target_total_ess / len(values)
    result = torch.zeros_like(log_weights, dtype=torch.double)
    betas, achieved = {}, {}
    for value in values:
        mask = condition_index == value
        beta, local = select_temperature(log_weights[mask], target_each)
        result[mask] = local / len(values)
        betas[value] = beta
        achieved[value] = effective_sample_size(local)
    result = result / result.sum()
    return result, betas, achieved


@torch.inference_mode()
def precompute_u(model, x, cond, chunk, device):
    parts = []
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        xb = x[start:stop].to(device)
        cb = cond[start:stop].to(device)
        u, _ = model.x_to_u_and_log_marginal(xb, cb)
        parts.append(u.detach().cpu())
    return torch.cat(parts, dim=0)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    base_model, _, checkpoint = common.load_nf(args.checkpoint, device)
    base_model.eval()
    data = torch.load(args.training_data, map_location="cpu")
    x = data["x"].float()
    cond = data["condition"].float()
    log_residual, thresholds = residual_log_weights(data)
    condition_index = data["condition_index"].long()
    sampling_weights, betas, condition_ess = balanced_tempered_weights(
        log_residual, condition_index, args.target_training_ess
    )
    training_ess = effective_sample_size(sampling_weights)

    raw_condition_ess = {}
    for value in sorted(int(v) for v in torch.unique(condition_index).tolist()):
        mask = condition_index == value
        raw_condition_ess[value] = effective_sample_size(
            normalized_weights(log_residual[mask], 1.0)
        )

    print(
        f"conditions={len(thresholds)}; balanced training ESS={training_ess:.2f}",
        flush=True,
    )
    for value in sorted(thresholds):
        print(
            f"  c={value:3d}: tau={thresholds[value]:.6g} "
            f"raw_ESS={raw_condition_ess[value]:.2f} "
            f"beta={betas[value]:.6f} trained_ESS={condition_ess[value]:.2f}",
            flush=True,
        )
    u = precompute_u(
        base_model, x, cond, args.precompute_chunk, device
    )

    tail_model = copy.deepcopy(base_model).to(device)
    for parameter in tail_model.marginals.parameters():
        parameter.requires_grad_(False)
    tail_model.train()
    trainable = [
        parameter for parameter in tail_model.copula.parameters()
        if parameter.requires_grad
    ]
    reference = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    for step in range(1, args.steps + 1):
        indices = torch.multinomial(
            sampling_weights, args.batch_size, replacement=True
        )
        ub = u[indices].to(device)
        cb = cond[indices].to(device)
        logq = tail_model.copula.log_prob_u(ub, cb)
        nll = -logq.mean()
        anchor = torch.zeros((), device=device)
        if args.anchor_penalty:
            numerator = torch.zeros((), device=device)
            denominator = 0
            for parameter, initial in zip(trainable, reference):
                numerator = numerator + (parameter - initial).square().sum()
                denominator += parameter.numel()
            anchor = numerator / max(denominator, 1)
        loss = nll + args.anchor_penalty * anchor

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "copula_nll": float(nll.detach()),
                "anchor_mse": float(anchor.detach()),
            }
            history.append(row)
            print(
                f"step {step:5d}/{args.steps}: "
                f"NLL={row['copula_nll']:.6f} "
                f"anchor={row['anchor_mse']:.3e}",
                flush=True,
            )

    tail_model.eval()
    output_checkpoint = args.output / "tail_model.pt"
    torch.save({
        "tail_model_state_dict": tail_model.state_dict(),
        "base_checkpoint": str(args.checkpoint),
        "training_data": str(args.training_data),
        "residual_threshold_tau_by_condition": thresholds,
        "weight_temperature_beta_by_condition": betas,
        "training_weight_ess_by_condition": condition_ess,
        "training_weight_ess": training_ess,
        "marginals_frozen": True,
        "condition_indices": sorted(
            set(int(value) for value in data["condition_index"].tolist())
        ),
    }, output_checkpoint)
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "training_data": str(args.training_data),
            "target_training_ess": args.target_training_ess,
            "actual_training_ess": training_ess,
            "weight_temperature_beta_by_condition": betas,
            "residual_threshold_tau_by_condition": thresholds,
            "training_weight_ess_by_condition": condition_ess,
            "raw_residual_ess_by_condition": raw_condition_ess,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "anchor_penalty": args.anchor_penalty,
            "marginals_frozen": True,
            "base_benchmark": checkpoint.get("benchmark"),
        }, stream, indent=2)
    print(f"\nTail model: {output_checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
