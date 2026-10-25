#!/usr/bin/env python3
"""Run protected annealed KL followed by residual-weighted Renyi refinement.

Stage 1 trains the usual full autoregressive NF from scratch while retaining a
fixed uniform exploration share.  Stage 2 freezes successive proposals, keeps
their independent sample sets for replay, and advances a Renyi-alpha ladder to
alpha=2.  The score weights emphasize samples with large target/proposal
residual without using structure labels or the known value of the integral.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--points-per-condition", type=int, default=8192)
    parser.add_argument("--marginal-steps", type=int, default=20000)
    parser.add_argument("--bridge-steps", type=int, default=20000)
    parser.add_argument("--renyi-steps", type=int, default=4000)
    parser.add_argument(
        "--alpha-ladder", type=float, nargs="+",
        default=(1.10, 1.25, 1.50, 1.75, 2.00),
    )
    parser.add_argument("--train-groups", type=int, default=64)
    parser.add_argument("--validation-groups", type=int, default=32)
    parser.add_argument("--replay-rounds", type=int, default=3)
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-bulk-checkpoint", action="store_true",
        help="Skip stage 1 when output/bulk already contains its checkpoint.",
    )
    args = parser.parse_args()
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("the fixed condition must lie in [0,1]^3")
    if not 0 < args.uniform_fraction < 1:
        parser.error("--uniform-fraction must lie in (0,1)")
    if any(alpha <= 1 or alpha > 2 for alpha in args.alpha_ladder):
        parser.error("the alpha ladder must lie in (1,2]")
    return args


def run(command):
    print("\n$ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bulk_dir = args.output / "bulk_kl"
    renyi_dir = args.output / "renyi_replay"
    bulk_dir.mkdir(exist_ok=True)
    renyi_dir.mkdir(exist_ok=True)
    checkpoint = bulk_dir / "heterogeneous_four_structure_32_final_model.pt"

    if not args.reuse_bulk_checkpoint or not checkpoint.exists():
        run([
            sys.executable,
            "-u",
            EXPERIMENTS / "run_heterogeneous_protected_kl_32d.py",
            "--protected-uniform-fraction", args.uniform_fraction,
            "--benchmarks", "heterogeneous_four_structure_32",
            "--fixed-condition", *args.fixed_condition,
            "--speed-profile", "balanced",
            "--points-per-condition", args.points_per_condition,
            "--marginal-max-steps", args.marginal_steps,
            "--bridge-max-steps", args.bridge_steps,
            "--marginal-patience", args.patience,
            "--bridge-patience", args.patience,
            "--temperature-repair-rounds", 2,
            "--allow-micro-annealing",
            "--skip-structure-search",
            "--device", args.device,
            "--output", bulk_dir,
        ])
    if not checkpoint.exists():
        raise FileNotFoundError(f"bulk checkpoint was not created: {checkpoint}")

    run([
        sys.executable,
        "-u",
        EXPERIMENTS / "finetune_conditional_renyi_32d.py",
        "--checkpoint", checkpoint,
        "--anchor", *args.fixed_condition,
        "--condition-radius", 0,
        "--anchor-fraction", 1,
        "--train-conditions", args.train_groups,
        "--validation-conditions", args.validation_groups,
        "--points-per-condition", args.points_per_condition,
        "--alpha-ladder", *args.alpha_ladder,
        "--steps-per-alpha", args.renyi_steps,
        "--condition-batch", 4,
        "--uniform-fraction", args.uniform_fraction,
        "--replay-rounds", args.replay_rounds,
        "--worst-weight", 0,
        "--learning-rate", args.learning_rate,
        "--check-interval", 100,
        "--patience", args.patience,
        "--seed", args.seed,
        "--device", args.device,
        "--output", renyi_dir,
    ])

    final_checkpoint = renyi_dir / "conditional_renyi_model.pt"
    summary = {
        "condition": list(args.fixed_condition),
        "bulk_checkpoint": str(checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "training": (
            "protected annealed KL followed by residual-weighted replay and "
            "a Renyi-alpha ladder ending at alpha=2"
        ),
        "structure_labels_used": False,
        "known_integral_used": False,
    }
    with (args.output / "pipeline_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nPROTECTED KL + REPLAY + RENYI PIPELINE FINISHED", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
