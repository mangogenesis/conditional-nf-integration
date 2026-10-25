#!/usr/bin/env python3
"""Closed-loop worst-condition search and Renyi-2 repair in 32D.

Each round audits a fixed condition pool with successive-halving tail search,
then fine-tunes around the worst condition neighborhoods.  The next round
re-audits the same pool with the updated frozen proposal.  State is written
after every phase, so an interrupted run resumes without repeating completed
work.

Stopping states are deliberately distinct:

* stable_safe: all requested tail thresholds pass;
* stalled_unsafe: the worst score stopped changing but thresholds still fail;
* maximum_rounds: the configured training budget was exhausted.

No exact target sampler or true integral is used by the loop.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEARCH_SCRIPT = ROOT / "experiments" / "search_worst_conditional_tail_32d.py"
TRAIN_SCRIPT = ROOT / "experiments" / "finetune_conditional_renyi_32d.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initial-conditions-file", type=Path,
        help="Optional known conditions inserted into the first search pool.",
    )
    parser.add_argument(
        "--anchor", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--condition-radius", type=float, default=0.05)
    parser.add_argument("--maximum-training-rounds", type=int, default=3)
    parser.add_argument("--candidate-conditions", type=int, default=256)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument(
        "--search-budgets", type=int, nargs="+",
        default=(4096, 16384, 65536, 1_000_000),
    )
    parser.add_argument("--search-keep-fraction", type=float, default=0.25)
    parser.add_argument("--ks-weight", type=float, default=1.0)
    parser.add_argument("--search-chunk", type=int, default=4096)
    parser.add_argument("--train-conditions", type=int, default=128)
    parser.add_argument("--validation-conditions", type=int, default=64)
    parser.add_argument("--points-per-condition", type=int, default=32768)
    parser.add_argument(
        "--alpha-ladder", type=float, nargs="+", default=(1.5, 1.75, 2.0)
    )
    parser.add_argument("--steps-per-alpha", type=int, default=1800)
    parser.add_argument("--condition-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--priority-condition-fraction", type=float, default=0.75)
    parser.add_argument("--priority-jitter-radius", type=float, default=0.005)
    parser.add_argument("--uniform-fraction", type=float, default=0.02)
    parser.add_argument("--replay-rounds", type=int, default=2)
    parser.add_argument("--worst-fraction", type=float, default=0.10)
    parser.add_argument("--worst-weight", type=float, default=2.0)
    parser.add_argument("--target-normalized-m2", type=float, default=10.0)
    parser.add_argument("--target-tail-ks", type=float, default=0.20)
    parser.add_argument("--target-hill-xi", type=float, default=0.50)
    parser.add_argument("--target-max-weight-share", type=float, default=0.01)
    parser.add_argument("--stability-relative-change", type=float, default=0.05)
    parser.add_argument("--stability-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_training_rounds <= 0:
        parser.error("--maximum-training-rounds must be positive")
    if args.target_normalized_m2 <= 1.0:
        parser.error("--target-normalized-m2 must exceed one")
    if not 0.0 <= args.target_tail_ks <= 1.0:
        parser.error("--target-tail-ks must lie in [0,1]")
    if args.stability_patience <= 0:
        parser.error("--stability-patience must be positive")
    return args


def run(command: list[str]) -> None:
    print("\nRUNNING:\n" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2)
    temporary.replace(path)


def audit_metrics(path: Path, args: argparse.Namespace) -> dict:
    frame = pd.read_csv(path)
    result = {
        "worst_search_score": float(frame["search_score"].max()),
        "worst_normalized_log_m2": float(frame["normalized_log_m2"].max()),
        "worst_normalized_m2": float(
            math.exp(min(frame["normalized_log_m2"].max(), 700.0))
        ),
        "worst_tail_ks": float(frame["tail_ks"].max()),
        "worst_hill_xi_005": float(frame["hill_xi_005"].max()),
        "worst_maximum_weight_share": float(
            frame["maximum_weight_share"].max()
        ),
    }
    result["safe"] = bool(
        result["worst_normalized_m2"] <= args.target_normalized_m2
        and result["worst_tail_ks"] <= args.target_tail_ks
        and result["worst_hill_xi_005"] <= args.target_hill_xi
        and result["worst_maximum_weight_share"]
        <= args.target_max_weight_share
    )
    return result


def search_command(
    args: argparse.Namespace, checkpoint: Path, search_output: Path,
    round_index: int, pool_file: Optional[Path],
) -> list[str]:
    command = [
        sys.executable, "-u", str(SEARCH_SCRIPT),
        "--checkpoint", str(checkpoint),
        "--anchor", *(str(value) for value in args.anchor),
        "--condition-radius", str(args.condition_radius),
        "--candidate-conditions", str(args.candidate_conditions),
        "--cumulative-budgets", *(str(value) for value in args.search_budgets),
        "--keep-fraction", str(args.search_keep_fraction),
        "--finalists", str(args.finalists),
        "--ks-weight", str(args.ks_weight),
        "--proposal", "pure",
        "--chunk", str(args.search_chunk),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output", str(search_output),
    ]
    if pool_file is not None:
        command.extend(["--condition-pool-file", str(pool_file)])
    elif args.initial_conditions_file is not None:
        command.extend([
            "--include-conditions-file", str(args.initial_conditions_file)
        ])
    return command


def train_command(
    args: argparse.Namespace, checkpoint: Path, worst_file: Path,
    train_output: Path, round_index: int,
) -> list[str]:
    return [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "--checkpoint", str(checkpoint),
        "--anchor", *(str(value) for value in args.anchor),
        "--condition-radius", str(args.condition_radius),
        "--alpha-ladder", *(str(value) for value in args.alpha_ladder),
        "--priority-conditions-file", str(worst_file),
        "--priority-condition-fraction", str(args.priority_condition_fraction),
        "--priority-jitter-radius", str(args.priority_jitter_radius),
        "--train-conditions", str(args.train_conditions),
        "--validation-conditions", str(args.validation_conditions),
        "--points-per-condition", str(args.points_per_condition),
        "--anchor-fraction", "0.10",
        "--uniform-fraction", str(args.uniform_fraction),
        "--replay-rounds", str(args.replay_rounds),
        "--steps-per-alpha", str(args.steps_per_alpha),
        "--condition-batch", str(args.condition_batch),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", "1e-6",
        "--anchor-penalty", "1e-3",
        "--worst-fraction", str(args.worst_fraction),
        "--worst-weight", str(args.worst_weight),
        "--check-interval", "100",
        "--patience", "8",
        "--seed", str(args.seed + 100_000 * (round_index + 1)),
        "--device", args.device,
        "--output", str(train_output),
    ]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "loop_state.json"
    if state_path.exists():
        with state_path.open() as stream:
            state = json.load(stream)
        print(
            f"RESUMING round={state['round_index']} "
            f"checkpoint={state['current_checkpoint']}",
            flush=True,
        )
    else:
        state = {
            "round_index": 0,
            "current_checkpoint": str(args.checkpoint),
            "condition_pool_file": None,
            "audits": [],
            "stability_count": 0,
            "status": "running",
        }
        write_state(state_path, state)

    while state["round_index"] <= args.maximum_training_rounds:
        round_index = int(state["round_index"])
        round_dir = args.output / f"round_{round_index:02d}"
        search_output = round_dir / "search"
        search_output.mkdir(parents=True, exist_ok=True)
        worst_file = search_output / "worst_conditions.csv"
        pool_file = (
            None if state["condition_pool_file"] is None
            else Path(state["condition_pool_file"])
        )
        if not worst_file.exists():
            run(search_command(
                args, Path(state["current_checkpoint"]), search_output,
                round_index, pool_file,
            ))
        if state["condition_pool_file"] is None:
            state["condition_pool_file"] = str(
                search_output / "candidate_conditions.csv"
            )

        metrics = audit_metrics(worst_file, args)
        previous = state["audits"][-1] if state["audits"] else None
        if previous is not None:
            relative_change = abs(
                metrics["worst_search_score"]
                - previous["worst_search_score"]
            ) / max(abs(previous["worst_search_score"]), 1.0)
            metrics["relative_score_change"] = relative_change
            state["stability_count"] = (
                state["stability_count"] + 1
                if relative_change <= args.stability_relative_change else 0
            )
        else:
            metrics["relative_score_change"] = None
        metrics["round"] = round_index
        metrics["checkpoint"] = state["current_checkpoint"]
        if not state["audits"] or state["audits"][-1]["round"] != round_index:
            state["audits"].append(metrics)
        else:
            state["audits"][-1] = metrics

        print("\nTAIL AUDIT")
        print(json.dumps(metrics, indent=2), flush=True)
        if metrics["safe"]:
            state["status"] = "stable_safe"
            write_state(state_path, state)
            break
        if state["stability_count"] >= args.stability_patience:
            state["status"] = "stalled_unsafe"
            write_state(state_path, state)
            break
        if round_index == args.maximum_training_rounds:
            state["status"] = "maximum_rounds"
            write_state(state_path, state)
            break

        train_output = round_dir / "train"
        train_output.mkdir(parents=True, exist_ok=True)
        candidate_checkpoint = train_output / "conditional_renyi_model.pt"
        if not candidate_checkpoint.exists():
            run(train_command(
                args, Path(state["current_checkpoint"]), worst_file,
                train_output, round_index,
            ))
        state["current_checkpoint"] = str(candidate_checkpoint)
        state["round_index"] = round_index + 1
        write_state(state_path, state)

    pd.DataFrame(state["audits"]).to_csv(
        args.output / "iteration_summary.csv", index=False
    )
    with (args.output / "final_status.json").open("w") as stream:
        json.dump(state, stream, indent=2)
    print("\nITERATIVE TAIL MINIMAX FINISHED")
    print(f"status             = {state['status']}")
    print(f"final checkpoint   = {state['current_checkpoint']}")
    print(f"summary            = {args.output / 'iteration_summary.csv'}")


if __name__ == "__main__":
    main()
