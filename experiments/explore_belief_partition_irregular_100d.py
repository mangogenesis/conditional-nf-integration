#!/usr/bin/env python3
"""Sequential belief-guided rare-tail exploration for the irregular 100D target.

The explored landscape is

    h(x,c) = 2 log f(x|c) - log q(x|c),

whose exponential is the second-moment integrand f^2/q.  A recursive
partition in a fixed random-projection feature space maintains local empirical
beliefs about h.  Regions receive stochastic ``bombs'' according to an upper
confidence acquisition score.  Newly observed envelope violations update the
belief and cause further refinement.

The reported envelopes are empirical unless --known-lipschitz is supplied.
Bomb endpoints are discovery/training data, never unbiased integration data.
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

import run_irregular_mixture_100d as irregular  # noqa: E402
from explore_energy_bombs_irregular_100d import load_proposal, logit  # noqa: E402


adaptive = irregular.adaptive
BENCHMARK = irregular.BENCHMARK


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--initial-conditions", type=int, default=32)
    parser.add_argument("--initial-points-per-condition", type=int, default=512)
    parser.add_argument("--initial-broad-fraction", type=float, default=0.10)
    parser.add_argument("--projection-dimension", type=int, default=16)
    parser.add_argument("--maximum-leaves", type=int, default=32)
    parser.add_argument("--minimum-leaf-points", type=int, default=128)
    parser.add_argument("--bombs-per-round", type=int, default=16)
    parser.add_argument("--condition-balanced-fraction", type=float, default=0.50)
    parser.add_argument("--walkers-per-bomb", type=int, default=64)
    parser.add_argument("--bomb-steps", type=int, default=100)
    parser.add_argument("--bomb-step-size", type=float, default=0.002)
    parser.add_argument("--bomb-start-temperature", type=float, default=0.5)
    parser.add_argument("--bomb-end-temperature", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=50.0)
    parser.add_argument("--ucb-beta", type=float, default=3.0)
    parser.add_argument("--novelty-weight", type=float, default=1.0)
    parser.add_argument("--known-lipschitz", type=float)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, x, cond, chunk):
    log_f_parts = []
    log_q_parts = []
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        log_f_parts.append(
            irregular.irregular_mixture_100_log_integrand(
                x[start:stop], cond[start:stop]
            ).reshape(-1).double()
        )
        log_q_parts.append(
            model.log_prob(x[start:stop], cond[start:stop]).reshape(-1).double()
        )
    log_f = torch.cat(log_f_parts)
    log_q = torch.cat(log_q_parts)
    return log_f, log_q, 2.0 * log_f - log_q


@torch.no_grad()
def initial_population(model, args, device):
    conditions = adaptive.sample_conditions(args.initial_conditions, device)
    cond = conditions[:, None, :].expand(
        -1, args.initial_points_per_condition, -1
    ).reshape(-1, conditions.shape[-1])
    x = model.sample(len(cond), cond)
    broad = torch.rand(len(x), device=device) < args.initial_broad_fraction
    if broad.any():
        x[broad] = torch.rand(int(broad.sum()), x.shape[-1], device=device)
    log_f, log_q, h = evaluate(model, x, cond, args.chunk)
    return {
        "x": x.detach(), "cond": cond.detach(),
        "log_f": log_f, "log_q": log_q, "h": h,
        "round": torch.zeros(len(x), dtype=torch.long, device=device),
        "source_broad": broad,
    }


def make_features(x, cond, projection):
    y = logit(x).double()
    projected = y @ projection.T
    projected = projected / math.sqrt(x.shape[-1])
    return torch.cat([projected, cond.double()], dim=1)


@torch.no_grad()
def two_means(features, indices, iterations=20):
    values = features[indices]
    center_a = values[0]
    distance = (values - center_a).square().sum(dim=1)
    center_b = values[int(distance.argmax())]
    labels = None
    for _ in range(iterations):
        new_labels = (
            (values - center_b).square().sum(dim=1)
            < (values - center_a).square().sum(dim=1)
        )
        if labels is not None and torch.equal(labels, new_labels):
            break
        labels = new_labels
        if labels.any() and (~labels).any():
            center_a = values[~labels].mean(dim=0)
            center_b = values[labels].mean(dim=0)
        else:
            break
    if labels is None or not labels.any() or labels.all():
        return None
    return indices[~labels], indices[labels]


@torch.no_grad()
def leaf_statistics(features, h, indices, args, leaf_id):
    values = h[indices]
    leaf_features = features[indices]
    center = leaf_features.mean(dim=0)
    radii = (leaf_features - center).square().sum(dim=1).sqrt()
    maximum = float(values.max())
    mean = float(values.mean())
    sd = float(values.std(unbiased=True)) if len(values) > 1 else 0.0
    q95 = float(torch.quantile(values, 0.95))
    uncertainty = sd / math.sqrt(max(len(values), 1))

    pair_count = min(4096, max(len(indices) * 2, 1))
    left = torch.randint(0, len(indices), (pair_count,), device=indices.device)
    right = torch.randint(0, len(indices), (pair_count,), device=indices.device)
    distance = (
        leaf_features[left] - leaf_features[right]
    ).square().sum(dim=1).sqrt().clamp_min(1e-8)
    slopes = (values[left] - values[right]).abs() / distance
    observed_lipschitz = float(torch.quantile(slopes, 0.99))
    lipschitz = (
        args.known_lipschitz
        if args.known_lipschitz is not None
        else observed_lipschitz
    )
    covering_radius = float(radii.max())
    envelope = maximum + lipschitz * covering_radius
    novelty = covering_radius / math.sqrt(max(len(values), 1))
    acquisition = maximum + args.ucb_beta * uncertainty + args.novelty_weight * novelty
    return {
        "leaf_id": leaf_id,
        "count": len(indices),
        "mean_h": mean,
        "sd_h": sd,
        "q95_h": q95,
        "maximum_h": maximum,
        "standard_error_h": uncertainty,
        "observed_lipschitz_q99": observed_lipschitz,
        "covering_radius": covering_radius,
        "empirical_envelope_h": envelope,
        "acquisition": acquisition,
        "center": center,
        "indices": indices,
    }


@torch.no_grad()
def build_partition(features, h, args):
    leaves = [torch.arange(len(h), device=h.device)]
    while len(leaves) < args.maximum_leaves:
        candidates = []
        for position, indices in enumerate(leaves):
            if len(indices) < 2 * args.minimum_leaf_points:
                continue
            values = h[indices]
            priority = float(values.std(unbiased=True)) * math.log1p(len(indices))
            candidates.append((priority, position))
        if not candidates:
            break
        _, position = max(candidates)
        split = two_means(features, leaves[position])
        if split is None or min(len(split[0]), len(split[1])) < args.minimum_leaf_points:
            break
        leaves[position:position + 1] = [split[0], split[1]]
    statistics = [
        leaf_statistics(features, h, indices, args, leaf_id)
        for leaf_id, indices in enumerate(leaves)
    ]
    return statistics


def tail_objective_and_gradient(model, y, cond, gradient_clip):
    y = y.detach().requires_grad_(True)
    x = torch.sigmoid(y)
    log_f = irregular.irregular_mixture_100_log_integrand(x, cond).reshape(-1)
    log_q = model.log_prob(x, cond).reshape(-1)
    h = 2.0 * log_f - log_q
    gradient = torch.autograd.grad(h.sum(), y)[0]
    norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gradient = gradient * torch.clamp(gradient_clip / norm, max=1.0)
    return h.detach(), gradient.detach()


def launch_bombs(model, data, leaves, args, round_id):
    scores = torch.tensor(
        [leaf["acquisition"] for leaf in leaves],
        dtype=torch.double,
        device=data["x"].device,
    )
    probabilities = torch.softmax(
        (scores - scores.max()) / max(scores.std().item(), 1.0), dim=0
    )
    balanced_count = min(
        args.bombs_per_round,
        int(round(args.bombs_per_round * args.condition_balanced_fraction)),
    )
    acquisition_count = args.bombs_per_round - balanced_count
    allocations = torch.multinomial(
        probabilities.float(), acquisition_count, replacement=True
    ) if acquisition_count else torch.empty(
        0, dtype=torch.long, device=data["x"].device
    )
    centers = []
    center_cond = []
    for selected_leaf in allocations.tolist():
        indices = leaves[selected_leaf]["indices"]
        candidate_h = data["h"][indices]
        top = min(16, len(indices))
        candidates = indices[torch.topk(candidate_h, top).indices]
        chosen = candidates[torch.randint(0, len(candidates), (1,), device=indices.device)]
        centers.append(data["x"][chosen])
        center_cond.append(data["cond"][chosen])
    # Reserve part of every round for condition coverage.  These bombs start
    # from the most dangerous known point at distinct conditions, preventing
    # the acquisition rule from spending the entire budget on a few easy-to-
    # discover conditions.
    unique_cond, inverse = torch.unique(data["cond"], dim=0, return_inverse=True)
    condition_order = torch.randperm(len(unique_cond), device=data["x"].device)
    for condition_index in condition_order[:balanced_count].tolist():
        candidates = torch.nonzero(
            inverse == condition_index, as_tuple=False
        ).flatten()
        chosen = candidates[torch.argmax(data["h"][candidates])][None]
        centers.append(data["x"][chosen])
        center_cond.append(data["cond"][chosen])
    centers = torch.cat(centers, dim=0)
    center_cond = torch.cat(center_cond, dim=0)
    y = logit(centers).repeat_interleave(args.walkers_per_bomb, dim=0)
    y = y + 0.20 * torch.randn_like(y)
    cond = center_cond.repeat_interleave(args.walkers_per_bomb, dim=0)
    initial_h = None
    for step in range(args.bomb_steps):
        fraction = step / max(args.bomb_steps - 1, 1)
        temperature = args.bomb_start_temperature * (
            args.bomb_end_temperature / args.bomb_start_temperature
        ) ** fraction
        h, gradient = tail_objective_and_gradient(
            model, y, cond, args.gradient_clip
        )
        if initial_h is None:
            initial_h = h
        eta = args.bomb_step_size * (0.25 + 0.75 * temperature / args.bomb_start_temperature)
        with torch.no_grad():
            y = y + eta * gradient + math.sqrt(2.0 * eta * temperature) * torch.randn_like(y)
        if step == 0 or step + 1 == args.bomb_steps:
            print(
                f"round {round_id:02d} bomb step {step + 1:03d}/{args.bomb_steps}: "
                f"mean h={float(h.mean()):.4f} q95={float(torch.quantile(h, .95)):.4f} "
                f"max={float(h.max()):.4f}"
            )
    x = torch.sigmoid(y.detach())
    log_f, log_q, h = evaluate(model, x, cond, args.chunk)
    return {
        "x": x,
        "cond": cond,
        "log_f": log_f,
        "log_q": log_q,
        "h": h,
        "round": torch.full(
            (len(x),), round_id, dtype=torch.long, device=x.device
        ),
        "source_broad": torch.zeros(len(x), dtype=torch.bool, device=x.device),
    }, float(initial_h.max()), float(h.max())


def append_data(first, second):
    return {key: torch.cat([first[key], second[key]], dim=0) for key in first}


def serializable_leaf(row, round_id):
    return {
        "round": round_id,
        **{
            key: value for key, value in row.items()
            if key not in {"center", "indices"}
        },
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    adaptive.DIM = irregular.DIMENSION
    model, checkpoint = load_proposal(args.checkpoint, device)
    data = initial_population(model, args, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 101)
    projection = torch.randn(
        args.projection_dimension, irregular.DIMENSION,
        generator=generator, device=device, dtype=torch.double,
    )
    projection = torch.linalg.qr(projection.T, mode="reduced").Q.T
    leaf_history = []
    round_history = []
    previous_envelopes = {}

    for round_id in range(1, args.rounds + 1):
        features = make_features(data["x"], data["cond"], projection)
        leaves = build_partition(features, data["h"], args)
        for row in leaves:
            leaf_history.append(serializable_leaf(row, round_id))
        envelope = max(row["empirical_envelope_h"] for row in leaves)
        maximum_before = float(data["h"].max())
        new_data, initial_max, final_max = launch_bombs(
            model, data, leaves, args, round_id
        )
        violation = final_max > envelope
        data = append_data(data, new_data)
        round_row = {
            "round": round_id,
            "points": len(data["x"]),
            "leaves": len(leaves),
            "maximum_h_before": maximum_before,
            "empirical_global_envelope_h": envelope,
            "bomb_initial_maximum_h": initial_max,
            "bomb_final_maximum_h": final_max,
            "envelope_violated": bool(violation),
        }
        round_history.append(round_row)
        print("BELIEF UPDATE", json.dumps(round_row, indent=2))
        previous_envelopes[round_id] = envelope

        torch.save(
            {
                **{key: value.detach().cpu() for key, value in data.items()},
                "projection": projection.cpu(),
                "checkpoint": str(args.checkpoint),
                "checkpoint_beta": float(checkpoint.get("beta", 1.0)),
                "landscape": "h = 2 log f - log q",
                "envelope_status": (
                    "assumption_informed_empirical"
                    if args.known_lipschitz is not None else "empirical"
                ),
            },
            args.output / "belief_tail_points.pt",
        )
        pd.DataFrame(leaf_history).to_csv(
            args.output / "leaf_history.csv", index=False
        )
        pd.DataFrame(round_history).to_csv(
            args.output / "round_history.csv", index=False
        )

    summary = {
        "rounds": args.rounds,
        "total_points": len(data["x"]),
        "maximum_h": float(data["h"].max()),
        "maximum_log_f_over_q": float((data["log_f"] - data["log_q"]).max()),
        "envelope_violations": sum(row["envelope_violated"] for row in round_history),
        "envelope_status": (
            "assumption_informed_empirical"
            if args.known_lipschitz is not None else "empirical"
        ),
        "warning": (
            "Observed local Lipschitz values are not global bounds; even a supplied "
            "constant is not a certificate until the partition has a verified "
            "full-space covering radius."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nFINAL BELIEF-PARTITION SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
