#!/usr/bin/env python3
"""Fine-tune the full-edge NF from blindly discovered complement regions.

Refined modes from ``refine_heterogeneous_nf_complement_32d.py`` initialize
independent local MALA walkers.  Each walker explores all 32 coordinates while
remaining inside its automatically discovered x0 region.  Every region
receives the same number of training samples, preventing the already-learned
region from dominating again.  The target sampler, target normalization,
structure count, and structure labels are never used.

This pilot assumes equal prior mass for each discovered region.  That is the
least informative allocation and is correct at the benchmark's central fixed
condition; general conditional mass estimation is a later stage.
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
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--train-walkers-per-region", type=int, default=256)
    parser.add_argument("--validation-walkers-per-region", type=int, default=128)
    parser.add_argument("--train-draws-per-walker", type=int, default=16)
    parser.add_argument("--validation-draws-per-walker", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--mala-step-size", type=float, default=0.003)
    parser.add_argument("--mala-gradient-clip", type=float, default=50.0)
    parser.add_argument("--points-per-group", type=int, default=512)
    parser.add_argument("--marginal-steps", type=int, default=20000)
    parser.add_argument("--copula-steps", type=int, default=20000)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = (
        args.train_walkers_per_region,
        args.validation_walkers_per_region,
        args.train_draws_per_walker,
        args.validation_draws_per_walker,
        args.burn_in,
        args.thin,
        args.points_per_group,
        args.marginal_steps,
        args.copula_steps,
        args.check_interval,
        args.patience,
    )
    if min(counts) <= 0:
        parser.error("all counts and training budgets must be positive")
    if args.mala_step_size <= 0 or args.mala_gradient_clip <= 0:
        parser.error("MALA step size and gradient clip must be positive")
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("fixed condition must lie in [0,1]^3")
    return args


def logit(x):
    epsilon = torch.finfo(x.dtype).eps
    x = x.clamp(epsilon, 1.0 - epsilon)
    return torch.log(x) - torch.log1p(-x)


def local_log_density_and_gradient(y, lower, upper, cond, gradient_clip):
    """Target density in a logit chart restricted to one discovered region."""
    y = y.detach().requires_grad_(True)
    unit = torch.sigmoid(y)
    width = upper - lower
    x = lower + width * unit
    logf = heterogeneous.heterogeneous_log_integrand(x, cond).reshape(-1)
    log_jacobian = (
        torch.log(width)
        + torch.nn.functional.logsigmoid(y)
        + torch.nn.functional.logsigmoid(-y)
    ).sum(dim=1)
    logpi = logf + log_jacobian
    gradient = torch.autograd.grad(logpi.sum(), y)[0]
    norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gradient = gradient * torch.clamp(gradient_clip / norm, max=1.0)
    return logpi.detach().double(), gradient.detach()


def local_mala_step(y, lower, upper, cond, step_size, gradient_clip):
    log_current, gradient_current = local_log_density_and_gradient(
        y, lower, upper, cond, gradient_clip
    )
    mean_forward = y + 0.5 * step_size * gradient_current
    proposed = mean_forward + math.sqrt(step_size) * torch.randn_like(y)
    log_proposed, gradient_proposed = local_log_density_and_gradient(
        proposed, lower, upper, cond, gradient_clip
    )
    mean_reverse = proposed + 0.5 * step_size * gradient_proposed
    log_forward = -(
        proposed - mean_forward
    ).double().square().sum(dim=1) / (2 * step_size)
    log_reverse = -(
        y - mean_reverse
    ).double().square().sum(dim=1) / (2 * step_size)
    log_acceptance = log_proposed - log_current + log_reverse - log_forward
    finite = torch.isfinite(log_acceptance)
    accepted = finite & (
        torch.log(torch.rand_like(log_acceptance))
        < torch.minimum(log_acceptance, torch.zeros_like(log_acceptance))
    )
    updated = torch.where(accepted[:, None], proposed, y).detach()
    return updated, float(accepted.double().mean())


def choose_initial_points(frame, region, count, generator, device):
    block = frame[
        (frame["discovered_region"] == region)
        & frame["refined_logf"].map(math.isfinite)
    ]
    if block.empty:
        raise RuntimeError(f"region {region} has no finite refined endpoints")
    indices = torch.randint(
        0, len(block), (count,), generator=generator
    ).numpy()
    columns = [f"x{coordinate}" for coordinate in range(heterogeneous.DIMENSION)]
    return torch.tensor(
        block.iloc[indices][columns].to_numpy(),
        dtype=torch.float32,
        device=device,
    )


def sample_local_region(
    initial, lower, upper, condition, draws, burn_in, thin, step_size,
    gradient_clip, label
):
    lower = lower.reshape(1, -1)
    upper = upper.reshape(1, -1)
    scaled = (initial - lower) / (upper - lower)
    y = logit(scaled).detach()
    cond = condition.expand(len(initial), -1)
    samples = []
    acceptances = []
    total_steps = burn_in + draws * thin
    current_step_size = step_size
    for step in range(1, total_steps + 1):
        y, acceptance = local_mala_step(
            y, lower, upper, cond, current_step_size, gradient_clip
        )
        acceptances.append(acceptance)
        if step <= burn_in:
            if acceptance < 0.35:
                current_step_size *= 0.85
            elif acceptance > 0.80:
                current_step_size *= 1.10
            current_step_size = min(max(current_step_size, 1e-5), 0.03)
        elif (step - burn_in) % thin == 0:
            samples.append(
                (lower + (upper - lower) * torch.sigmoid(y)).detach()
            )
        if step % 50 == 0 or step == total_steps:
            recent = acceptances[-50:]
            print(
                f"{label} local step={step:4d}/{total_steps} "
                f"acceptance={sum(recent) / len(recent):.4f} "
                f"h={current_step_size:.6g}",
                flush=True,
            )
    return torch.cat(samples, dim=0), {
        "mean_acceptance": sum(acceptances) / len(acceptances),
        "final_step_size": current_step_size,
        "samples": sum(len(block) for block in samples),
    }


def build_balanced_cache(region_samples, condition, points_per_group, device):
    region_count = len(region_samples)
    samples_per_region = min(len(block) for block in region_samples)
    generator = torch.Generator(device=device)
    generator.manual_seed(7319 + samples_per_region)
    shuffled = []
    for block in region_samples:
        order = torch.randperm(len(block), generator=generator, device=device)
        shuffled.append(block[order[:samples_per_region]])
    # Interleave regions so every group has essentially equal representation.
    flat = torch.stack(shuffled, dim=1).reshape(
        samples_per_region * region_count, heterogeneous.DIMENSION
    )
    effective_points = max(
        region_count,
        (points_per_group // region_count) * region_count,
    )
    groups = len(flat) // effective_points
    if groups < 1:
        raise RuntimeError("not enough local samples to form one training group")
    flat = flat[:groups * effective_points]
    x = flat.reshape(groups, effective_points, heterogeneous.DIMENSION)
    cond = condition.expand(groups * effective_points, -1).reshape(
        groups, effective_points, -1
    ).clone()
    alpha = torch.full(
        (groups, effective_points),
        1.0 / effective_points,
        dtype=x.dtype,
        device=device,
    )
    return {"x": x, "cond": cond, "alpha": alpha}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    adaptive, model = audit.load_model(args.checkpoint, device)
    # The audit loader deliberately freezes every parameter.  This script
    # reuses its checkpoint reconstruction, but the model is being fine-tuned,
    # so gradients must be restored explicitly.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise RuntimeError("loaded model has no trainable parameters")
    print(f"Trainable model parameters: {trainable_parameters:,}", flush=True)
    try:
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        source = torch.load(args.checkpoint, map_location="cpu")
    frame = pd.read_csv(args.regions)
    required = {"discovered_region", "refined_logf"} | {
        f"x{coordinate}" for coordinate in range(heterogeneous.DIMENSION)
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"region file is missing columns: {sorted(missing)}")
    region_ids = sorted(int(value) for value in frame["discovered_region"].unique())
    if len(region_ids) < 2:
        raise RuntimeError("fewer than two complement regions were discovered")
    print(f"Discovered regions used for balanced learning: {region_ids}")

    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    ).reshape(1, -1)
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(args.seed)
    train_regions = []
    validation_regions = []
    sampling_rows = []
    for region in region_ids:
        finite_region = frame[
            (frame["discovered_region"] == region)
            & frame["refined_logf"].map(math.isfinite)
        ]
        x0_minimum = float(finite_region["x0"].min())
        x0_maximum = float(finite_region["x0"].max())
        if not x0_minimum < x0_maximum:
            raise RuntimeError(f"region {region} has a degenerate x0 interval")
        lower = torch.zeros(
            heterogeneous.DIMENSION, dtype=torch.float32, device=device
        )
        upper = torch.ones_like(lower)
        lower[0] = x0_minimum
        upper[0] = x0_maximum
        print(
            f"region={region} local chart x0 in "
            f"[{x0_minimum:.6f}, {x0_maximum:.6f}]",
            flush=True,
        )
        train_initial = choose_initial_points(
            frame, region, args.train_walkers_per_region,
            cpu_generator, device,
        )
        validation_initial = choose_initial_points(
            frame, region, args.validation_walkers_per_region,
            cpu_generator, device,
        )
        train_samples, train_info = sample_local_region(
            train_initial,
            lower,
            upper,
            condition,
            args.train_draws_per_walker,
            args.burn_in,
            args.thin,
            args.mala_step_size,
            args.mala_gradient_clip,
            f"train region={region}",
        )
        validation_samples, validation_info = sample_local_region(
            validation_initial,
            lower,
            upper,
            condition,
            args.validation_draws_per_walker,
            args.burn_in,
            args.thin,
            args.mala_step_size,
            args.mala_gradient_clip,
            f"validation region={region}",
        )
        train_regions.append(train_samples)
        validation_regions.append(validation_samples)
        sampling_rows.append({
            "region": region,
            "split": "train",
            **train_info,
        })
        sampling_rows.append({
            "region": region,
            "split": "validation",
            **validation_info,
        })

    train_cache = build_balanced_cache(
        train_regions, condition, args.points_per_group, device
    )
    validation_cache = build_balanced_cache(
        validation_regions, condition, args.points_per_group, device
    )
    print(
        f"Balanced local train cache: groups={len(train_cache['x'])} "
        f"points/group={train_cache['x'].shape[1]} "
        f"total={train_cache['x'].numel() // heterogeneous.DIMENSION:,}"
    )
    print(
        f"Balanced local validation cache: groups={len(validation_cache['x'])} "
        f"points/group={validation_cache['x'].shape[1]} "
        f"total={validation_cache['x'].numel() // heterogeneous.DIMENSION:,}"
    )

    adaptive.DIM = heterogeneous.DIMENSION
    adaptive.configure_speed_profile("balanced")
    adaptive.COMPILE_TRAINING_KERNELS = False
    adaptive.MAX_MARGINAL_STEPS = args.marginal_steps
    adaptive.MIN_MARGINAL_STEPS = min(2000, args.marginal_steps)
    adaptive.MARGINAL_CHECK_INTERVAL = args.check_interval
    adaptive.MARGINAL_EARLY_STOP_PATIENCE = args.patience
    adaptive.BRIDGE_COPULA_MAX_STEPS = args.copula_steps
    adaptive.BRIDGE_COPULA_CHECK_INTERVAL = args.check_interval
    adaptive.BRIDGE_COPULA_PATIENCE = args.patience

    bank = adaptive.get_marginal_bank(model)
    marginal_result = adaptive.train_marginals_on_dataset(
        bank,
        train_cache,
        validation_cache,
        device,
        args.seed + 1000,
    )
    copula_result = adaptive.train_bridge_copula_on_cache(
        model,
        train_cache,
        validation_cache,
        label="balanced complement regions",
        optimize_joint=False,
    )
    model.eval()
    output_model = args.output / "heterogeneous_four_structure_32_complement_trained_model.pt"
    checkpoint = {
        **source,
        "benchmark": heterogeneous.BENCHMARK,
        "dimension": heterogeneous.DIMENSION,
        "graph": source["graph"],
        "model_state_dict": adaptive.cpu_state_dict(model),
        "complement_region_training": {
            "condition": list(args.fixed_condition),
            "regions": region_ids,
            "region_allocation": "equal",
            "uses_smc": False,
            "uses_target_sampler": False,
            "uses_structure_labels": False,
            "local_sampler": (
                "MALA in all 32 coordinates, with x0 restricted to each "
                "automatically discovered region"
            ),
            "marginal_best_step": marginal_result["best_step"],
            "copula_best_step": copula_result["best_step"],
        },
    }
    torch.save(checkpoint, output_model)
    torch.save(
        {
            "train": {key: value.cpu() for key, value in train_cache.items()},
            "validation": {
                key: value.cpu() for key, value in validation_cache.items()
            },
            "region_ids": region_ids,
        },
        args.output / "balanced_local_clouds.pt",
    )
    pd.DataFrame(sampling_rows).to_csv(
        args.output / "local_sampling_summary.csv", index=False
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "output_model": str(output_model),
        "condition": list(args.fixed_condition),
        "regions": region_ids,
        "train_samples": int(
            train_cache["x"].numel() // heterogeneous.DIMENSION
        ),
        "validation_samples": int(
            validation_cache["x"].numel() // heterogeneous.DIMENSION
        ),
        "marginal_best_step": marginal_result["best_step"],
        "marginal_validation_nll": marginal_result["val_nll"],
        "copula_best_step": copula_result["best_step"],
        "copula_validation_nll": copula_result["val_nll"],
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nCOMPLEMENT-REGION TRAINING FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"\nModel: {output_model.resolve()}")


if __name__ == "__main__":
    main()
