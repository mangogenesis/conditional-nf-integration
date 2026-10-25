#!/usr/bin/env python3
"""Fresh annealed 32D training with KL marginals and a pure SW copula.

No checkpoint or discovered target cloud is loaded.  Each annealing stage is
constructed from the usual defensive uniform/NF proposal.  Marginals use the
existing weighted likelihood fit; the copula is trained only with joint sliced
Wasserstein-2 in copula space.  Training stops after the annealed bridge, before
the generic runner's formal KL and chi-square stages.
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

import finetune_heterogeneous_copula_sliced_wasserstein_32d as sw  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--train-conditions", type=int, default=64)
    parser.add_argument("--validation-conditions", type=int, default=32)
    parser.add_argument("--points-per-condition", type=int, default=8192)
    parser.add_argument("--marginal-steps", type=int, default=20000)
    parser.add_argument("--copula-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--random-projections", type=int, default=96)
    parser.add_argument("--pairwise-projections", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-uniform-fraction", type=float, default=0.01)
    parser.add_argument("--temperature-repair-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = (
        args.train_conditions, args.validation_conditions,
        args.points_per_condition, args.marginal_steps, args.copula_steps,
        args.batch_size, args.random_projections, args.pairwise_projections,
        args.check_interval, args.patience,
    )
    if min(counts) <= 0:
        parser.error("all budgets must be positive")
    if not 0 < args.minimum_uniform_fraction <= args.uniform_fraction < 1:
        parser.error("invalid uniform fractions")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def weighted_target_u(model, cache, count, seed):
    """Resample the fixed weighted cache and transform it to copula space."""
    x = cache["x"].reshape(-1, heterogeneous.DIMENSION)
    cond = cache["cond"].reshape(-1, heterogeneous.CONDITION_DIMENSION)
    groups = cache["alpha"].shape[0]
    probability = cache["alpha"].reshape(-1).double() / groups
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    indices = torch.multinomial(
        probability, count, replacement=True, generator=generator
    )
    bank = heterogeneous.adaptive.get_marginal_bank(model)
    with torch.no_grad():
        u = sw.marginal_x_to_u(bank, x[indices], cond[indices]).detach()
    return u


def make_sw_bridge_trainer(args, condition, history):
    adaptive = heterogeneous.adaptive

    def train_sw_copula(model, train_cache, validation_cache, label,
                        optimize_joint=False):
        if optimize_joint:
            raise RuntimeError(
                "fresh SW experiment requires separately trained marginals"
            )
        bank = adaptive.get_marginal_bank(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.copula.parameters():
            parameter.requires_grad_(True)

        validation_count = min(
            4096, validation_cache["x"].numel() // heterogeneous.DIMENSION
        )
        validation_target = weighted_target_u(
            model, validation_cache, validation_count,
            args.seed + 900_000 + len(history),
        )
        directions = sw.projection_directions(
            heterogeneous.DIMENSION, args.random_projections,
            args.pairwise_projections, validation_target.device,
            validation_target.dtype,
        )
        optimizer = torch.optim.Adam(
            model.copula.parameters(), lr=args.learning_rate
        )
        initial = sw.validation_loss(
            model, bank, validation_target, condition, directions,
            validation_count, args.seed + 910_000 + len(history),
        )
        best = initial
        best_step = 0
        best_state = adaptive.cpu_state_dict(model.copula)
        no_improve = 0
        rows = []
        print(
            f"PURE SW COPULA BRIDGE: {label} initial SW2={initial:.8g}",
            flush=True,
        )

        train_x = train_cache["x"].reshape(-1, heterogeneous.DIMENSION)
        train_cond = train_cache["cond"].reshape(
            -1, heterogeneous.CONDITION_DIMENSION
        )
        train_probability = train_cache["alpha"].reshape(-1).double()
        train_probability /= train_cache["alpha"].shape[0]
        for start in range(0, args.copula_steps, args.check_interval):
            block = min(args.check_interval, args.copula_steps - start)
            losses = []
            model.train()
            for _ in range(block):
                indices = torch.multinomial(
                    train_probability, args.batch_size, replacement=True
                )
                with torch.no_grad():
                    target_u = sw.marginal_x_to_u(
                        bank, train_x[indices], train_cond[indices]
                    ).detach()
                proposal_u = sw.sample_copula_u(
                    model, bank, args.batch_size, condition
                )
                projection = sw.projection_directions(
                    heterogeneous.DIMENSION, args.random_projections,
                    args.pairwise_projections, target_u.device, target_u.dtype,
                )
                loss = sw.sliced_wasserstein_squared(
                    target_u, proposal_u, projection
                )
                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.copula.parameters(), 10.0)
                optimizer.step()
                losses.append(loss.detach())

            step = start + block
            value = sw.validation_loss(
                model, bank, validation_target, condition, directions,
                validation_count, args.seed + 910_000 + len(history),
            )
            improved = value < best * (1.0 - 1e-4)
            if improved:
                best = value
                best_step = step
                best_state = adaptive.cpu_state_dict(model.copula)
                no_improve = 0
            else:
                no_improve += 1
            train_value = (
                float(torch.stack(losses).mean()) if losses else math.nan
            )
            rows.append({
                "step": step,
                "train_sw2": train_value,
                "validation_sw2": value,
                "best_validation_sw2": best,
            })
            print(
                f"SW copula step {step:5d}: train={train_value:.8g} "
                f"validation={value:.8g} best={best:.8g} "
                f"patience={no_improve}/{args.patience}",
                flush=True,
            )
            if no_improve >= args.patience:
                print(f"SW COPULA EARLY STOP at {step}", flush=True)
                break

        adaptive.load_cpu_state(model.copula, best_state)
        record = {
            "label": label,
            "best_step": best_step,
            "initial_sw2": initial,
            "validation_sw2": best,
            "history": rows,
            "optimize_joint": False,
            # Preserve the generic result key while making its meaning clear.
            "val_nll": None,
        }
        history.append(record)
        return record

    return train_sw_copula


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    heterogeneous.register_benchmark(tuple(args.fixed_condition))
    adaptive = heterogeneous.adaptive
    adaptive.DIM = heterogeneous.DIMENSION
    adaptive.configure_speed_profile("balanced")
    adaptive.COMPILE_TRAINING_KERNELS = False
    adaptive.MAX_MARGINAL_STEPS = args.marginal_steps
    adaptive.MIN_MARGINAL_STEPS = min(2000, args.marginal_steps)
    adaptive.MARGINAL_CHECK_INTERVAL = args.check_interval
    adaptive.MARGINAL_EARLY_STOP_PATIENCE = args.patience
    adaptive.MARGINAL_TRAIN_CONDITIONS = args.train_conditions
    adaptive.VAL_CONDITIONS = args.validation_conditions
    adaptive.MARGINAL_ADAPTIVE_POINTS = args.points_per_condition
    adaptive.UNIFORM_BOOTSTRAP_POINTS = args.points_per_condition
    adaptive.TEMPERATURE_PROBE_CONDITIONS = args.validation_conditions
    adaptive.TEMPERATURE_PROBE_POINTS = args.points_per_condition
    adaptive.PROPOSAL_UNIFORM_FRACTION = args.uniform_fraction
    adaptive.PROPOSAL_MIN_UNIFORM_FRACTION = args.minimum_uniform_fraction

    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    ).reshape(1, -1)
    sw_history = []
    original_bridge_trainer = adaptive.train_bridge_copula_on_cache
    adaptive.train_bridge_copula_on_cache = make_sw_bridge_trainer(
        args, condition, sw_history
    )
    bridge_args = argparse.Namespace(
        speed_profile="balanced",
        temperature_repair_rounds=args.temperature_repair_rounds,
        joint_marginal_copula=False,
    )
    try:
        marginal_state, marginal_result, model = adaptive.train_initial_marginals(
            heterogeneous.BENCHMARK, bridge_args, device
        )
    finally:
        adaptive.train_bridge_copula_on_cache = original_bridge_trainer

    graph = adaptive.full_ar_graph(heterogeneous.DIMENSION)
    output_path = args.output / "heterogeneous_four_structure_32_fresh_sw_model.pt"
    checkpoint = {
        "benchmark": heterogeneous.BENCHMARK,
        "dimension": heterogeneous.DIMENSION,
        "graph": graph,
        "model_state_dict": adaptive.cpu_state_dict(model),
        "fresh_sliced_wasserstein_training": {
            "condition": list(args.fixed_condition),
            "starts_from_checkpoint": False,
            "uses_previous_clouds": False,
            "uses_structure_labels": False,
            "uses_known_integral": False,
            "marginal_objective": "importance-weighted likelihood",
            "copula_objective": "pure sliced Wasserstein-2",
            "annealing": marginal_result,
            "sw_stages": sw_history,
        },
    }
    torch.save(checkpoint, output_path)
    rows = []
    for stage, record in enumerate(sw_history):
        for row in record["history"]:
            rows.append({"stage": stage, "label": record["label"], **row})
    pd.DataFrame(rows).to_csv(
        args.output / "sliced_wasserstein_history.csv", index=False
    )
    summary = {
        "checkpoint": str(output_path),
        "condition": list(args.fixed_condition),
        "annealing_stages": len(sw_history),
        "stage_best_sw2": [record["validation_sw2"] for record in sw_history],
        "stage_best_steps": [record["best_step"] for record in sw_history],
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nFRESH ANNEALED SLICED-WASSERSTEIN TRAINING FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"\nModel: {output_path.resolve()}")


if __name__ == "__main__":
    main()
