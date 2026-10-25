#!/usr/bin/env python3
"""Pure sliced-Wasserstein copula fine-tuning for the 32D benchmark.

The marginal maps are frozen.  Target points are converted to copula space,
where every coordinate is approximately uniform, and the copula is optimized
only through joint one-dimensional projections.  No benchmark structure label
or known integral is used by training.
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
import train_heterogeneous_discovered_region_specialists_32d as cloud_io  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clouds", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--random-projections", type=int, default=96)
    parser.add_argument("--pairwise-projections", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-4)
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = (
        args.steps, args.batch_size, args.random_projections,
        args.pairwise_projections, args.check_interval, args.patience,
        args.validation_samples,
    )
    if min(counts) <= 0:
        parser.error("all budgets must be positive")
    if args.learning_rate <= 0 or args.minimum_relative_improvement < 0:
        parser.error("invalid optimization parameters")
    return args


def flatten_cloud(cache):
    return cache["x"].reshape(-1, heterogeneous.DIMENSION)


def marginal_x_to_u(bank, x, cond):
    marginals = bank.marginals if hasattr(bank, "marginals") else bank
    pieces = []
    for coordinate, marginal in enumerate(marginals):
        u, _ = marginal.inverse(x[:, coordinate:coordinate + 1], cond)
        pieces.append(u.clamp(0.0, 1.0))
    return torch.cat(pieces, dim=1)


def call_sample(module, count, cond):
    errors = []
    for call in (
        lambda: module.sample(count, cond),
        lambda: module.sample(cond),
        lambda: module.sample(cond, count),
    ):
        try:
            value = call()
            if isinstance(value, tuple):
                value = value[0]
            if value.ndim == 3 and value.shape[0] == 1:
                value = value.reshape(-1, value.shape[-1])
            if value.shape == (count, heterogeneous.DIMENSION):
                return value
        except (AttributeError, TypeError, RuntimeError, ValueError) as error:
            errors.append(str(error))
    raise RuntimeError("no supported differentiable sampling signature: " + " | ".join(errors))


def sample_copula_u(model, bank, count, condition):
    cond = condition.expand(count, -1)
    # Prefer sampling the copula directly.  Fall back to sampling x and
    # applying the frozen marginal inverse if the copula has no public sampler.
    try:
        u = call_sample(model.copula, count, cond)
        if not u.requires_grad:
            raise RuntimeError("copula sampler is not reparameterized")
    except RuntimeError:
        x = call_sample(model, count, cond)
        u = marginal_x_to_u(bank, x, cond)
    if not u.requires_grad:
        raise RuntimeError(
            "the model sampler is not reparameterized: generated copula "
            "samples do not carry gradients"
        )
    return u.clamp(0.0, 1.0)


def projection_directions(dimension, random_count, pair_count, device, dtype):
    random = torch.randn(random_count, dimension, device=device, dtype=dtype)
    random = random / random.norm(dim=1, keepdim=True).clamp_min(1e-12)
    first = torch.randint(0, dimension, (pair_count,), device=device)
    offset = torch.randint(1, dimension, (pair_count,), device=device)
    second = (first + offset) % dimension
    signs = torch.where(
        torch.rand(pair_count, device=device) < 0.5,
        torch.ones(pair_count, device=device, dtype=dtype),
        -torch.ones(pair_count, device=device, dtype=dtype),
    )
    pairwise = torch.zeros(pair_count, dimension, device=device, dtype=dtype)
    pairwise[torch.arange(pair_count, device=device), first] = 1.0 / math.sqrt(2.0)
    pairwise[torch.arange(pair_count, device=device), second] = signs / math.sqrt(2.0)
    return torch.cat((random, pairwise), dim=0)


def sliced_wasserstein_squared(target_u, proposal_u, directions):
    target_projection = target_u @ directions.T
    proposal_projection = proposal_u @ directions.T
    target_sorted = torch.sort(target_projection, dim=0).values
    proposal_sorted = torch.sort(proposal_projection, dim=0).values
    return (target_sorted - proposal_sorted).square().mean()


@torch.no_grad()
def validation_loss(model, bank, target_u, condition, directions, count, seed):
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if condition.is_cuda else None
    torch.manual_seed(seed)
    if condition.is_cuda:
        torch.cuda.manual_seed_all(seed)
    model.eval()
    proposal_u = sample_copula_u_for_validation(model, bank, count, condition)
    value = float(sliced_wasserstein_squared(
        target_u[:count], proposal_u, directions
    ))
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return value


def sample_copula_u_for_validation(model, bank, count, condition):
    # Sampling under no_grad is intentional for validation, so this version
    # omits the training-time reparameterization check.
    cond = condition.expand(count, -1)
    try:
        return call_sample(model.copula, count, cond).clamp(0.0, 1.0)
    except RuntimeError:
        x = call_sample(model, count, cond)
        return marginal_x_to_u(bank, x, cond).clamp(0.0, 1.0)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    try:
        source = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        source = torch.load(args.checkpoint, map_location=device)
    adaptive, model = audit.load_model(args.checkpoint, device)
    bank = adaptive.get_marginal_bank(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.copula.parameters():
        parameter.requires_grad_(True)

    clouds = cloud_io.load_clouds(args.clouds, device)
    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    ).reshape(1, -1)
    train_x = flatten_cloud(clouds["train"])
    validation_x = flatten_cloud(clouds["validation"])
    with torch.no_grad():
        train_u = marginal_x_to_u(
            bank, train_x, condition.expand(len(train_x), -1)
        ).detach()
        validation_u = marginal_x_to_u(
            bank, validation_x, condition.expand(len(validation_x), -1)
        ).detach()

    validation_count = min(args.validation_samples, len(validation_u))
    validation_directions = projection_directions(
        heterogeneous.DIMENSION, args.random_projections,
        args.pairwise_projections, device, train_u.dtype,
    )
    # Fix the target validation subset once.
    validation_order = torch.randperm(len(validation_u), device=device)
    validation_u = validation_u[validation_order[:validation_count]]

    optimizer = torch.optim.Adam(model.copula.parameters(), lr=args.learning_rate)
    initial = validation_loss(
        model, bank, validation_u, condition, validation_directions,
        validation_count, args.seed + 1,
    )
    best = initial
    best_step = 0
    best_state = adaptive.cpu_state_dict(model.copula)
    no_improve = 0
    history = []
    print(
        f"PURE COPULA SLICED-WASSERSTEIN: initial validation SW2={initial:.8g}",
        flush=True,
    )

    for start in range(0, args.steps, args.check_interval):
        block = min(args.check_interval, args.steps - start)
        losses = []
        model.train()
        for _ in range(block):
            indices = torch.randint(
                0, len(train_u), (args.batch_size,), device=device
            )
            target = train_u[indices]
            proposal = sample_copula_u(
                model, bank, args.batch_size, condition
            )
            directions = projection_directions(
                heterogeneous.DIMENSION, args.random_projections,
                args.pairwise_projections, device, target.dtype,
            )
            loss = sliced_wasserstein_squared(target, proposal, directions)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.copula.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.detach())

        step = start + block
        value = validation_loss(
            model, bank, validation_u, condition, validation_directions,
            validation_count, args.seed + 1,
        )
        improved = value < best * (1.0 - args.minimum_relative_improvement)
        if improved:
            best = value
            best_step = step
            best_state = adaptive.cpu_state_dict(model.copula)
            no_improve = 0
        else:
            no_improve += 1
        train_value = float(torch.stack(losses).mean()) if losses else math.nan
        history.append({
            "step": step,
            "train_sw2": train_value,
            "validation_sw2": value,
            "best_validation_sw2": best,
        })
        print(
            f"step {step:5d}/{args.steps}: train SW2={train_value:.8g} "
            f"validation SW2={value:.8g} best={best:.8g} "
            f"patience={no_improve}/{args.patience}",
            flush=True,
        )
        if no_improve >= args.patience:
            print(f"SLICED-WASSERSTEIN EARLY STOP at step {step}", flush=True)
            break

    adaptive.load_cpu_state(model.copula, best_state)
    model.eval()
    output_path = args.output / "heterogeneous_four_structure_32_copula_sw_model.pt"
    checkpoint = {
        **source,
        "benchmark": heterogeneous.BENCHMARK,
        "dimension": heterogeneous.DIMENSION,
        "graph": source["graph"],
        "model_state_dict": adaptive.cpu_state_dict(model),
        "sliced_wasserstein_finetuning": {
            "objective": "pure joint copula sliced Wasserstein-2",
            "marginals_frozen": True,
            "uses_structure_labels": False,
            "uses_known_integral": False,
            "source_clouds": str(args.clouds),
            "initial_validation_sw2": initial,
            "best_validation_sw2": best,
            "best_step": best_step,
            "random_projections": args.random_projections,
            "pairwise_projections": args.pairwise_projections,
        },
    }
    torch.save(checkpoint, output_path)
    pd.DataFrame(history).to_csv(
        args.output / "sliced_wasserstein_history.csv", index=False
    )
    summary = checkpoint["sliced_wasserstein_finetuning"] | {
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(output_path),
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nPURE COPULA SLICED-WASSERSTEIN FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"\nModel: {output_path.resolve()}")


if __name__ == "__main__":
    main()
