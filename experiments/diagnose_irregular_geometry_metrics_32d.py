#!/usr/bin/env python3
"""Oracle-side coverage diagnostics for the irregular 32D benchmark.

Training is not performed here.  The normalized benchmark construction gives
us an exact target sampler, which is used only as a diagnostic oracle.  This
lets us compare proposal-side importance-sampling diagnostics with target-side
coverage metrics and identify modes that the proposal almost never visits.
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

import run_irregular_mixture_32d as target32  # noqa: E402
from explore_energy_bombs_irregular_100d import load_proposal  # noqa: E402


family = target32.family
adaptive = target32.adaptive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditions", type=int, default=16)
    parser.add_argument("--oracle-evals", type=int, default=100000)
    parser.add_argument("--proposal-evals", type=int, default=100000)
    parser.add_argument("--chunk", type=int, default=5000)
    parser.add_argument("--tail-thresholds", default="2,5,10,20")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normal_cdf(value):
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def normal_icdf(value):
    epsilon = torch.finfo(value.dtype).eps
    value = value.clamp(epsilon, 1.0 - epsilon)
    return math.sqrt(2.0) * torch.erfinv(2.0 * value - 1.0)


def sample_truncated_normal(mu, sigma):
    low = normal_cdf(-mu / sigma)
    high = normal_cdf((1.0 - mu) / sigma)
    uniform = low + torch.rand_like(mu) * (high - low)
    return (mu + sigma * normal_icdf(uniform)).clamp(0.0, 1.0)


@torch.inference_mode()
def sample_target(cond):
    """Draw exact samples from the normalized hidden target mixture."""
    parameters = family.parameters_on(cond.device, cond.dtype)
    count = cond.shape[0]
    labels = torch.distributions.Categorical(
        logits=family.mixture_log_weights(cond, parameters)
    ).sample()
    x = torch.empty(count, target32.DIMENSION, device=cond.device,
                    dtype=cond.dtype)
    row = torch.arange(count, device=cond.device)

    for child in range(target32.DIMENSION):
        if child == 0:
            a = cond[:, 0]
            b = cond[:, 1]
        else:
            parent_a = parameters["parent_a"][labels, child]
            parent_b = parameters["parent_b"][labels, child]
            a = x[row, parent_a]
            b = x[row, parent_b]

        frequency = parameters["frequency"][labels, child]
        phase = parameters["phase"][labels, child]
        wave = torch.sin(2 * math.pi * frequency * a + phase + 0.8 * cond[:, 0])
        cross = torch.cos(2 * math.pi * (a - b) + 0.5 * phase - 0.7 * cond[:, 1])
        switch = torch.tanh(4.0 * (cond[:, 2] - 0.5) + 1.2 * cross)
        signal = (
            parameters["amplitude"][labels] * wave
            + parameters["interaction"][labels] * cross * switch
        )
        drift = 0.035 * torch.sin(
            0.17 * child + labels.to(cond.dtype) + math.pi * cond[:, 1]
        )
        mu = (0.5 + signal + drift).clamp(0.055, 0.945)
        sigma = parameters["base_sigma"][labels] * (
            0.78 + 0.38 * cond[:, 2]
            + 0.06 * ((child + labels) % 5).to(cond.dtype)
        )
        x[:, child] = sample_truncated_normal(mu, sigma)
    return x, labels


def logmeanexp(values):
    return float(torch.logsumexp(values.double(), dim=0) - math.log(len(values)))


def safe_exp(value):
    return math.exp(value) if value < 700 else float("inf")


@torch.inference_mode()
def oracle_metrics(model, condition, evaluations, chunk, thresholds):
    ratios = []
    labels = []
    remaining = evaluations
    while remaining:
        size = min(chunk, remaining)
        cond = condition[None].expand(size, -1)
        x, component = sample_target(cond)
        log_p = family.irregular_mixture_100_log_integrand(x, cond).double()
        log_q = model.log_prob(x, cond).reshape(-1).double()
        ratios.append((log_p - log_q).cpu())
        labels.append(component.cpu())
        remaining -= size

    ratio = torch.cat(ratios)
    label = torch.cat(labels)
    log_m2 = logmeanexp(ratio)
    row = {
        "forward_kl": float(ratio.mean()),
        "log_m2_oracle": log_m2,
        "m2_oracle": safe_exp(log_m2),
        "ratio_q50": float(torch.quantile(ratio, 0.50)),
        "ratio_q90": float(torch.quantile(ratio, 0.90)),
        "ratio_q99": float(torch.quantile(ratio, 0.99)),
        "ratio_max": float(ratio.max()),
    }
    for threshold in thresholds:
        selected = ratio > threshold
        target_mass = float(selected.double().mean())
        # q(A) = E_p[(q/p) 1_A].  This estimates how hard region A is to
        # discover using samples from q, without requiring q to find it first.
        proposal_mass = float(
            torch.where(selected, torch.exp(-ratio), torch.zeros_like(ratio)).mean()
        )
        n95 = (
            math.log(0.05) / math.log1p(-proposal_mass)
            if 0.0 < proposal_mass < 1.0 else
            (float("inf") if proposal_mass == 0.0 else 1.0)
        )
        key = str(threshold).replace(".", "p")
        row[f"target_mass_r_gt_{key}"] = target_mass
        row[f"proposal_mass_r_gt_{key}"] = proposal_mass
        row[f"samples_for_95pct_hit_r_gt_{key}"] = n95

    component_rows = []
    for component in range(family.COMPONENTS):
        selected = label == component
        if not selected.any():
            continue
        values = ratio[selected]
        component_rows.append({
            "component": component,
            "oracle_samples": int(selected.sum()),
            "mean_log_p_over_q": float(values.mean()),
            "q90_log_p_over_q": float(torch.quantile(values, 0.90)),
            "q99_log_p_over_q": float(torch.quantile(values, 0.99)),
            "maximum_log_p_over_q": float(values.max()),
        })
    return row, component_rows


@torch.inference_mode()
def proposal_metrics(model, condition, evaluations, chunk):
    log_weights = []
    remaining = evaluations
    while remaining:
        size = min(chunk, remaining)
        cond = condition[None].expand(size, -1)
        x = model.sample(size, cond)
        log_q = model.log_prob(x, cond).reshape(-1).double()
        log_p = family.irregular_mixture_100_log_integrand(x, cond).double()
        log_weights.append((log_p - log_q).cpu())
        remaining -= size
    log_weight = torch.cat(log_weights)
    log_i = logmeanexp(log_weight)
    log_m2 = logmeanexp(2.0 * log_weight)
    return {
        "integral_is": safe_exp(log_i),
        "log_m2_proposal": log_m2,
        "m2_proposal": safe_exp(log_m2),
        "normalized_ess": safe_exp(2.0 * log_i - log_m2),
        "proposal_max_log_weight": float(log_weight.max()),
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    target32.configure_family()
    adaptive.DIM = target32.DIMENSION
    model, checkpoint = load_proposal(args.checkpoint, device)
    thresholds = [float(value) for value in args.tail_thresholds.split(",")]
    conditions = adaptive.sample_conditions(args.conditions, device)

    summary_rows = []
    component_rows = []
    for index, condition in enumerate(conditions):
        oracle, by_component = oracle_metrics(
            model, condition, args.oracle_evals, args.chunk, thresholds
        )
        proposal = proposal_metrics(
            model, condition, args.proposal_evals, args.chunk
        )
        row = {
            "condition_index": index,
            "c0": float(condition[0]),
            "c1": float(condition[1]),
            "c2": float(condition[2]),
            **proposal,
            **oracle,
            "m2_underestimate_factor": (
                oracle["m2_oracle"] / max(proposal["m2_proposal"], 1e-300)
            ),
        }
        summary_rows.append(row)
        for component_row in by_component:
            component_rows.append({"condition_index": index, **component_row})
        print(
            f"{index + 1:3d}/{len(conditions)} "
            f"I_IS={row['integral_is']:.4g} "
            f"logM2(q)={row['log_m2_proposal']:.3f} "
            f"logM2(p)={row['log_m2_oracle']:.3f} "
            f"R99={row['ratio_q99']:.3f}",
            flush=True,
        )

    frame = pd.DataFrame(summary_rows)
    components = pd.DataFrame(component_rows)
    frame.to_csv(args.output / "condition_metrics.csv", index=False)
    components.to_csv(args.output / "component_metrics.csv", index=False)

    aggregate = {
        "conditions": len(frame),
        "oracle_evals_per_condition": args.oracle_evals,
        "proposal_evals_per_condition": args.proposal_evals,
        "mean_is_integral": float(frame["integral_is"].mean()),
        "minimum_is_integral": float(frame["integral_is"].min()),
        "mean_forward_kl": float(frame["forward_kl"].mean()),
        "mean_oracle_log_m2": float(frame["log_m2_oracle"].mean()),
        "maximum_oracle_log_m2": float(frame["log_m2_oracle"].max()),
        "median_m2_underestimate_factor": float(
            frame["m2_underestimate_factor"].median()
        ),
        "maximum_target_ratio_q99": float(frame["ratio_q99"].max()),
        "checkpoint_beta": float(checkpoint.get("beta", float("nan"))),
        "interpretation": (
            "Large oracle/proposal M2 disagreement means proposal-side samples "
            "are missing important target regions."
        ),
    }
    with (args.output / "aggregate.json").open("w") as stream:
        json.dump(aggregate, stream, indent=2)
    print("\nORACLE-SIDE 32D METRIC SUMMARY")
    print(json.dumps(aggregate, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
