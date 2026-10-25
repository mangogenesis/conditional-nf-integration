#!/usr/bin/env python3
"""Fresh annealed 32D training with a hybrid KL + sliced-Wasserstein copula.

The usual importance-weighted likelihood remains the density-fitting backbone.
At every copula step, a sliced-Wasserstein term is added in copula space so
that matching joint geometry and separated modes is rewarded explicitly.
Neither a checkpoint, target labels, nor the known integral is used.
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
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument("--random-projections", type=int, default=96)
    parser.add_argument("--pairwise-projections", type=int, default=32)
    parser.add_argument("--sw-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-kl-improvement", type=float, default=1e-3)
    parser.add_argument("--minimum-sw-improvement", type=float, default=1e-6)
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-uniform-fraction", type=float, default=0.01)
    parser.add_argument("--temperature-repair-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = (
        args.train_conditions, args.validation_conditions,
        args.points_per_condition, args.marginal_steps, args.copula_steps,
        args.batch_size, args.validation_samples, args.random_projections,
        args.pairwise_projections, args.check_interval, args.patience,
    )
    if min(counts) <= 0:
        parser.error("all budgets must be positive")
    if not 0 < args.minimum_uniform_fraction <= args.uniform_fraction < 1:
        parser.error("invalid uniform fractions")
    if args.learning_rate <= 0 or args.sw_weight < 0:
        parser.error("learning rate must be positive and SW weight nonnegative")
    if args.minimum_kl_improvement < 0:
        parser.error("--minimum-kl-improvement must be nonnegative")
    if args.minimum_sw_improvement < 0:
        parser.error("--minimum-sw-improvement must be nonnegative")
    return args


def strictly_improves_both(
    nll, sw2, best_nll, best_sw2,
    minimum_kl_improvement, minimum_sw_improvement,
):
    """Require one checkpoint to improve both validation objectives."""
    kl_improved = nll < best_nll - minimum_kl_improvement
    sw_improved = sw2 < best_sw2 - minimum_sw_improvement
    return kl_improved, sw_improved, kl_improved and sw_improved


def weighted_target_sample(cache, count, seed):
    """Draw a reproducible sample from the cache's empirical target law."""
    x = cache["x"].reshape(-1, heterogeneous.DIMENSION)
    cond = cache["cond"].reshape(-1, heterogeneous.CONDITION_DIMENSION)
    groups = cache["alpha"].shape[0]
    probability = cache["alpha"].reshape(-1).double() / groups
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    indices = torch.multinomial(
        probability, count, replacement=True, generator=generator
    )
    return x[indices], cond[indices]


def fixed_validation_metrics(
    model, bank, target_x, target_cond, target_u, condition, directions,
    proposal_count, seed,
):
    """Evaluate both terms on fixed target data and fixed random samples."""
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if condition.is_cuda else None
    torch.manual_seed(seed)
    if condition.is_cuda:
        torch.cuda.manual_seed_all(seed)
    model.eval()
    with torch.no_grad():
        nll = float(-model.log_prob(target_x, target_cond).mean())
        proposal_u = sw.sample_copula_u_for_validation(
            model, bank, proposal_count, condition
        )
        sw2 = float(sw.sliced_wasserstein_squared(
            target_u, proposal_u, directions
        ))
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return nll, sw2


def make_hybrid_bridge_trainer(args, condition, stage_history):
    adaptive = heterogeneous.adaptive

    def train_hybrid(model, train_cache, validation_cache, label,
                     optimize_joint=False):
        if optimize_joint:
            raise RuntimeError(
                "fresh KL+SW experiment requires separately trained marginals"
            )
        bank = adaptive.get_marginal_bank(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.copula.parameters():
            parameter.requires_grad_(True)

        validation_count = min(
            args.validation_samples,
            validation_cache["x"].numel() // heterogeneous.DIMENSION,
        )
        validation_x, validation_cond = weighted_target_sample(
            validation_cache, validation_count,
            args.seed + 900_000 + len(stage_history),
        )
        with torch.no_grad():
            validation_u = sw.marginal_x_to_u(
                bank, validation_x, validation_cond
            ).detach()
        directions = sw.projection_directions(
            heterogeneous.DIMENSION, args.random_projections,
            args.pairwise_projections, validation_x.device,
            validation_x.dtype,
        )
        metric_seed = args.seed + 910_000 + len(stage_history)
        initial_nll, initial_sw2 = fixed_validation_metrics(
            model, bank, validation_x, validation_cond, validation_u,
            condition, directions, validation_count, metric_seed,
        )

        # KL and SW live on very different numerical scales.  Fixed scales
        # make --sw-weight meaningful while preserving each loss's gradient.
        kl_scale = max(abs(initial_nll), 1.0)
        sw_scale = max(initial_sw2, 1e-6)

        def score(nll, sw2):
            return (
                (nll - initial_nll) / kl_scale
                + args.sw_weight * (sw2 - initial_sw2) / sw_scale
            )

        optimizer = torch.optim.Adam(
            model.copula.parameters(), lr=args.learning_rate
        )
        best_score = 0.0
        best_nll = initial_nll
        best_sw2 = initial_sw2
        best_step = 0
        best_state = adaptive.cpu_state_dict(model.copula)
        no_improve = 0
        rows = []
        print(
            f"HYBRID KL+SW COPULA BRIDGE: {label} "
            f"initial NLL={initial_nll:.6f} SW2={initial_sw2:.8g} "
            f"SW-weight={args.sw_weight:g}",
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
            kl_losses = []
            sw_losses = []
            total_losses = []
            model.train()
            for _ in range(block):
                indices = torch.multinomial(
                    train_probability, args.batch_size, replacement=True
                )
                target_x = train_x[indices]
                target_cond = train_cond[indices]
                with torch.no_grad():
                    target_u = sw.marginal_x_to_u(
                        bank, target_x, target_cond
                    ).detach()
                kl_loss = -model.log_prob(target_x, target_cond).mean()
                proposal_u = sw.sample_copula_u(
                    model, bank, args.batch_size, condition
                )
                projection = sw.projection_directions(
                    heterogeneous.DIMENSION, args.random_projections,
                    args.pairwise_projections, target_x.device,
                    target_x.dtype,
                )
                sw_loss = sw.sliced_wasserstein_squared(
                    target_u, proposal_u, projection
                )
                loss = kl_loss / kl_scale + args.sw_weight * sw_loss / sw_scale
                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.copula.parameters(), 10.0)
                optimizer.step()
                kl_losses.append(kl_loss.detach())
                sw_losses.append(sw_loss.detach())
                total_losses.append(loss.detach())

            step = start + block
            val_nll, val_sw2 = fixed_validation_metrics(
                model, bank, validation_x, validation_cond, validation_u,
                condition, directions, validation_count, metric_seed,
            )
            val_score = score(val_nll, val_sw2)
            kl_improved, sw_improved, jointly_improved = strictly_improves_both(
                val_nll,
                val_sw2,
                best_nll,
                best_sw2,
                args.minimum_kl_improvement,
                args.minimum_sw_improvement,
            )
            if jointly_improved:
                best_score = val_score
                best_nll = val_nll
                best_sw2 = val_sw2
                best_step = step
                best_state = adaptive.cpu_state_dict(model.copula)
                no_improve = 0
            else:
                no_improve += 1
            train_kl = float(torch.stack(kl_losses).mean()) if kl_losses else math.nan
            train_sw = float(torch.stack(sw_losses).mean()) if sw_losses else math.nan
            train_total = (
                float(torch.stack(total_losses).mean())
                if total_losses else math.nan
            )
            rows.append({
                "step": step,
                "train_nll": train_kl,
                "train_sw2": train_sw,
                "train_combined": train_total,
                "validation_nll": val_nll,
                "validation_sw2": val_sw2,
                "validation_combined_score": val_score,
                "best_combined_score": best_score,
                "best_validation_nll": best_nll,
                "best_validation_sw2": best_sw2,
                "kl_improved": kl_improved,
                "sw_improved": sw_improved,
                "jointly_improved": jointly_improved,
            })
            print(
                f"KL+SW step {step:5d}: train NLL={train_kl:.6f} "
                f"SW2={train_sw:.8g} | val NLL={val_nll:.6f} "
                f"SW2={val_sw2:.8g} score={val_score:.6f} "
                f"joint-best={'yes' if jointly_improved else 'no'} "
                f"best NLL={best_nll:.6f} best SW2={best_sw2:.8g} "
                f"patience={no_improve}/{args.patience}",
                flush=True,
            )
            if no_improve >= args.patience:
                print(f"KL+SW COPULA EARLY STOP at {step}", flush=True)
                break

        adaptive.load_cpu_state(model.copula, best_state)
        record = {
            "label": label,
            "best_step": best_step,
            "initial_nll": initial_nll,
            "validation_nll": best_nll,
            "initial_sw2": initial_sw2,
            "validation_sw2": best_sw2,
            "best_combined_score": best_score,
            "kl_scale": kl_scale,
            "sw_scale": sw_scale,
            "sw_weight": args.sw_weight,
            "checkpoint_rule": "strict simultaneous KL and SW improvement",
            "minimum_kl_improvement": args.minimum_kl_improvement,
            "minimum_sw_improvement": args.minimum_sw_improvement,
            "history": rows,
            "optimize_joint": False,
            "val_nll": best_nll,
        }
        stage_history.append(record)
        return record

    return train_hybrid


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
    history = []
    original = adaptive.train_bridge_copula_on_cache
    adaptive.train_bridge_copula_on_cache = make_hybrid_bridge_trainer(
        args, condition, history
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
        adaptive.train_bridge_copula_on_cache = original

    graph = adaptive.full_ar_graph(heterogeneous.DIMENSION)
    output_path = (
        args.output / "heterogeneous_four_structure_32_fresh_kl_sw_model.pt"
    )
    torch.save({
        "benchmark": heterogeneous.BENCHMARK,
        "dimension": heterogeneous.DIMENSION,
        "graph": graph,
        "model_state_dict": adaptive.cpu_state_dict(model),
        "fresh_kl_sliced_wasserstein_training": {
            "condition": list(args.fixed_condition),
            "starts_from_checkpoint": False,
            "uses_previous_clouds": False,
            "uses_structure_labels": False,
            "uses_known_integral": False,
            "marginal_objective": "importance-weighted likelihood",
            "copula_objective": "normalized KL + sliced Wasserstein-2",
            "checkpoint_rule": "strict simultaneous KL and SW improvement",
            "sw_weight": args.sw_weight,
            "minimum_kl_improvement": args.minimum_kl_improvement,
            "minimum_sw_improvement": args.minimum_sw_improvement,
            "annealing": marginal_result,
            "stages": history,
        },
    }, output_path)

    rows = []
    for stage, record in enumerate(history):
        for row in record["history"]:
            rows.append({"stage": stage, "label": record["label"], **row})
    pd.DataFrame(rows).to_csv(args.output / "kl_sw_history.csv", index=False)
    summary = {
        "checkpoint": str(output_path),
        "condition": list(args.fixed_condition),
        "annealing_stages": len(history),
        "sw_weight": args.sw_weight,
        "checkpoint_rule": "strict simultaneous KL and SW improvement",
        "minimum_kl_improvement": args.minimum_kl_improvement,
        "minimum_sw_improvement": args.minimum_sw_improvement,
        "stage_best_nll": [r["validation_nll"] for r in history],
        "stage_best_sw2": [r["validation_sw2"] for r in history],
        "stage_best_steps": [r["best_step"] for r in history],
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nFRESH ANNEALED KL + SLICED-WASSERSTEIN TRAINING FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"\nModel: {output_path.resolve()}")


if __name__ == "__main__":
    main()
