#!/usr/bin/env python3
"""Generate frozen-proposal coverage data for a 32D geometry benchmark.

No training or interval tuning is performed.  Conditions are deterministic
interstitial points (or supplied explicitly), and every replicate uses an
independent random seed.  Output ``moment_convergence.csv`` files are directly
consumable by ``evaluate_empirical_m2_coverage.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_geometry_ladder_32d as geometry  # noqa: E402
import run_topology_ladder_32d as topology  # noqa: E402


DIM = 32
COND_DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--condition-design",
        choices=("interstitial", "random"),
        default="interstitial",
    )
    parser.add_argument("--conditions-file", type=Path)
    parser.add_argument(
        "--fixed-condition",
        type=float,
        nargs=COND_DIM,
        metavar=("C0", "C1", "C2"),
        help=(
            "Evaluate one fixed anchor condition. This overrides the condition "
            "design and --conditions count, while every replicate still uses "
            "independent proposal samples."
        ),
    )
    parser.add_argument("--conditions", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--evals", type=int, default=100_000)
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        help=(
            "Optional cumulative sample sizes at which to save moments. "
            "The final --evals value is always included."
        ),
    )
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument(
        "--keep-largest-weights", type=int, default=0,
        help=(
            "Retain this many largest-weight phase-space points per "
            "condition/replicate in a PyTorch file for later tail repair. "
            "Zero disables point retention."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--proposal", choices=("pure", "mixture"), default="pure"
    )
    parser.add_argument(
        "--uniform-fraction",
        type=float,
        default=None,
        help="For mixture mode; defaults to the checkpoint value.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard partial per-condition results instead of resuming them.",
    )
    args = parser.parse_args()
    if args.conditions <= 0 or args.replicates <= 0 or args.evals <= 1:
        parser.error("conditions/replicates must be positive and evals > 1")
    if args.chunk <= 0:
        parser.error("--chunk must be positive")
    if args.keep_largest_weights < 0:
        parser.error("--keep-largest-weights must be nonnegative")
    if args.milestones is not None and any(
        value <= 1 or value > args.evals for value in args.milestones
    ):
        parser.error("every milestone must lie in [2, --evals]")
    if args.uniform_fraction is not None and not (
        0.0 < args.uniform_fraction < 1.0
    ):
        parser.error("--uniform-fraction must lie in (0,1)")
    return args


def make_conditions(args: argparse.Namespace) -> np.ndarray:
    if args.fixed_condition is not None and args.conditions_file is not None:
        raise ValueError(
            "Use either --fixed-condition or --conditions-file, not both"
        )
    if args.fixed_condition is not None:
        values = np.asarray([args.fixed_condition], dtype=np.float64)
    elif args.conditions_file is not None:
        frame = pd.read_csv(args.conditions_file)
        missing = {"c0", "c1", "c2"}.difference(frame.columns)
        if missing:
            raise ValueError(
                f"conditions file lacks columns {sorted(missing)}"
            )
        values = frame[["c0", "c1", "c2"]].to_numpy(np.float64)
    elif args.condition_design == "interstitial":
        # Centers of a 4 x 4 x 2 partition of [0,1]^3.  Under continuous
        # random training-condition sampling, these exact values are unseen
        # with probability one and test interpolation within the same range.
        values = np.asarray([
            ((i + 0.5) / 4.0, (j + 0.5) / 4.0, (k + 0.5) / 2.0)
            for i in range(4)
            for j in range(4)
            for k in range(2)
        ], dtype=np.float64)
        if args.conditions != len(values):
            raise ValueError(
                "interstitial design has exactly 32 conditions; use "
                "--conditions 32 or select --condition-design random"
            )
    else:
        values = np.random.default_rng(args.seed).random(
            (args.conditions, COND_DIM)
        )
    if args.fixed_condition is None and len(values) != args.conditions:
        raise ValueError(
            f"requested {args.conditions} conditions but received {len(values)}"
        )
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("all conditions must lie in [0,1]^3")
    return values


def load_model(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    benchmark = checkpoint.get("benchmark")
    if benchmark in topology.BENCHMARKS:
        topology.register_benchmarks()
    elif benchmark in geometry.BENCHMARKS:
        geometry.register_benchmarks()
    else:
        raise ValueError(
            f"checkpoint benchmark {benchmark!r} is not a geometry/topology "
            "ladder target"
        )
    adaptive = geometry.adaptive
    adaptive.DIM = DIM
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fraction = float(
        checkpoint.get("final_metrics", {}).get("uniform_fraction", 0.10)
    )
    return adaptive, model, checkpoint, benchmark, fraction


def moments(logw: np.ndarray) -> dict[str, float]:
    n = len(logw)
    log_sum = float(logsumexp(logw))
    log_sum2 = float(logsumexp(2.0 * logw))
    first = math.exp(log_sum - math.log(n))
    second = math.exp(log_sum2 - math.log(n))
    normalized_ess = math.exp(2.0 * log_sum - math.log(n) - log_sum2)
    max_share = math.exp(float(logw.max()) - log_sum)
    variance_mean = max(second - first * first, 0.0) / max(n - 1, 1)
    ordered = np.sort(logw)[::-1]

    def hill_xi(fraction: float) -> float:
        # Hill's estimator is computed directly from log weights.  The next
        # order statistic is used as the threshold, so it remains stable even
        # when the weights themselves would overflow.
        k = max(2, int(math.ceil(fraction * n)))
        k = min(k, n - 1)
        return float(np.mean(ordered[:k] - ordered[k]))

    return {
        "first_moment": first,
        "second_moment": second,
        "same_run_se": math.sqrt(variance_mean),
        "normalized_ess": normalized_ess,
        "max_weight_share": max_share,
        "maximum_log_weight": float(logw.max()),
        "hill_xi_001": hill_xi(0.001),
        "hill_xi_005": hill_xi(0.005),
    }


@torch.inference_mode()
def run_replicate(
    adaptive,
    model,
    benchmark: str,
    condition: np.ndarray,
    condition_index: int,
    replicate: int,
    mixture_fraction: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    list[dict[str, float]], Optional[dict[str, torch.Tensor]]
]:
    run_seed = args.seed + 1_000_003 * condition_index + 10_007 * replicate
    torch.manual_seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
        torch.cuda.synchronize()
    started = time.perf_counter()
    pieces = []
    retained = None
    rows = []
    milestones = sorted(set((args.milestones or []) + [args.evals]))
    milestone_index = 0
    sampled = 0
    remaining = args.evals
    condition_tensor = torch.as_tensor(
        condition, dtype=torch.float32, device=device
    )
    while remaining:
        next_milestone = milestones[milestone_index]
        count = min(remaining, args.chunk, next_milestone - sampled)
        cond = condition_tensor.expand(count, COND_DIM)
        if args.proposal == "pure":
            x = adaptive.sample_from_model(model, count, cond)
            logq = model.log_prob(x, cond).reshape(-1)
        else:
            x, logq = adaptive.sample_from_proposal(
                proposal_model=model,
                n=count,
                cond=cond,
                device=device,
                uniform_fraction=mixture_fraction,
            )
        logf = adaptive.target_log_integrand(benchmark, x, cond)
        logw = (logf - logq).double()
        pieces.append(logw.cpu().numpy())
        if args.keep_largest_weights:
            keep = min(args.keep_largest_weights, count)
            values, indices = torch.topk(logw, keep)
            candidate = {
                "x": x[indices].detach().cpu(),
                "log_f": logf[indices].double().detach().cpu(),
                "log_q": logq[indices].double().detach().cpu(),
                "log_weight": values.detach().cpu(),
            }
            if retained is None:
                retained = candidate
            else:
                combined = {
                    key: torch.cat((retained[key], candidate[key]), dim=0)
                    for key in retained
                }
                selected = torch.topk(
                    combined["log_weight"],
                    min(args.keep_largest_weights, len(combined["log_weight"])),
                ).indices
                retained = {
                    key: value[selected] for key, value in combined.items()
                }
        remaining -= count
        sampled += count
        if sampled == next_milestone:
            result = moments(np.concatenate(pieces))
            result.update({
                "condition_index": condition_index,
                "replicate": replicate,
                "seed": run_seed,
                "samples": sampled,
                "c0": float(condition[0]),
                "c1": float(condition[1]),
                "c2": float(condition[2]),
                "seconds": time.perf_counter() - started,
            })
            rows.append(result)
            milestone_index += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    return rows, retained


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = make_conditions(args)
    pd.DataFrame(
        conditions, columns=["c0", "c1", "c2"]
    ).rename_axis("condition_index").reset_index().to_csv(
        args.output / "conditions.csv", index=False
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    adaptive, model, checkpoint, benchmark, checkpoint_fraction = load_model(
        args.checkpoint, device
    )
    mixture_fraction = (
        checkpoint_fraction
        if args.uniform_fraction is None
        else args.uniform_fraction
    )

    print(
        "FROZEN GEOMETRY/TOPOLOGY COVERAGE DATA\n"
        f"benchmark={benchmark} conditions={len(conditions)} "
        f"replicates={args.replicates} evals/run={args.evals:,}\n"
        f"proposal={args.proposal} mixture_fraction={mixture_fraction:.6g}\n"
        f"checkpoint={args.checkpoint}",
        flush=True,
    )

    all_rows = []
    total = len(conditions) * args.replicates
    done = 0
    for condition_index, condition in enumerate(conditions):
        condition_dir = args.output / f"c{condition_index}"
        condition_dir.mkdir(exist_ok=True)
        moment_path = condition_dir / "moment_convergence.csv"
        if moment_path.exists() and not args.restart:
            existing = pd.read_csv(moment_path)
            condition_rows = existing.to_dict("records")
            completed = set(
                existing.loc[
                    existing["samples"] == args.evals, "replicate"
                ].astype(int)
            )
        else:
            condition_rows = []
            completed = set()
        all_rows.extend(condition_rows)
        done += len(completed)
        for replicate in range(args.replicates):
            if replicate in completed:
                continue
            rows, retained = run_replicate(
                adaptive,
                model,
                benchmark,
                condition,
                condition_index,
                replicate,
                mixture_fraction,
                args,
                device,
            )
            condition_rows.extend(rows)
            all_rows.extend(rows)
            if retained is not None:
                torch.save({
                    **retained,
                    "condition": torch.as_tensor(
                        condition, dtype=torch.float32
                    ),
                    "condition_index": condition_index,
                    "replicate": replicate,
                    "seed": int(rows[-1]["seed"]),
                    "checkpoint": str(args.checkpoint),
                    "proposal": args.proposal,
                    "uniform_fraction": mixture_fraction,
                    "samples_searched": args.evals,
                }, condition_dir / (
                    f"largest_weights_r{replicate:04d}.pt"
                ))
            row = rows[-1]
            done += 1
            pd.DataFrame(condition_rows).to_csv(
                moment_path, index=False
            )
            print(
                f"{done:4d}/{total} c={condition_index:2d} "
                f"r={replicate:2d} I={row['first_moment']:.7f} "
                f"M2={row['second_moment']:.5g} "
                f"ESS={row['normalized_ess']:.4f} "
                f"max-share={row['max_weight_share']:.3g}",
                flush=True,
            )

    detail = pd.DataFrame(all_rows)
    detail.to_csv(args.output / "replicate_results.csv", index=False)
    if args.keep_largest_weights:
        retained_files = sorted(
            args.output.glob("c*/largest_weights_r*.pt")
        )
        retained_runs = []
        for path in retained_files:
            try:
                retained_runs.append(torch.load(
                    path, map_location="cpu", weights_only=False
                ))
            except TypeError:
                retained_runs.append(torch.load(path, map_location="cpu"))
        if retained_runs:
            combined = {
                key: torch.cat([run[key] for run in retained_runs], dim=0)
                for key in ("x", "log_f", "log_q", "log_weight")
            }
            combined["condition"] = torch.cat([
                run["condition"].reshape(1, COND_DIM).expand(
                    len(run["log_weight"]), COND_DIM
                )
                for run in retained_runs
            ], dim=0)
            combined["condition_index"] = torch.cat([
                torch.full(
                    (len(run["log_weight"]),),
                    int(run["condition_index"]), dtype=torch.long,
                )
                for run in retained_runs
            ])
            combined["replicate"] = torch.cat([
                torch.full(
                    (len(run["log_weight"]),),
                    int(run["replicate"]), dtype=torch.long,
                )
                for run in retained_runs
            ])
            combined.update({
                "checkpoint": str(args.checkpoint),
                "proposal": args.proposal,
                "uniform_fraction": mixture_fraction,
                "evaluations_per_run": args.evals,
                "retained_per_run": args.keep_largest_weights,
                "source_files": [str(path) for path in retained_files],
                "warning": (
                    "Largest-weight points are biased tail-discovery data, "
                    "not independent target samples or integration samples."
                ),
            })
            torch.save(
                combined, args.output / "largest_weight_points.pt"
            )
            print(
                "Retained tail-discovery points: "
                f"{len(combined['log_weight']):,} -> "
                f"{args.output / 'largest_weight_points.pt'}",
                flush=True,
            )
    with (args.output / "config.json").open("w") as stream:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "checkpoint_benchmark": benchmark,
            "condition_design": (
                "fixed" if args.fixed_condition is not None
                else args.condition_design
            ),
            "fixed_condition": args.fixed_condition,
            "conditions": len(conditions),
            "replicates": args.replicates,
            "evaluations_per_run": args.evals,
            "milestones": sorted(set((args.milestones or []) + [args.evals])),
            "proposal": args.proposal,
            "uniform_fraction": mixture_fraction,
            "true_integral": 1.0,
            "training_performed": False,
            "interval_tuning_performed": False,
            "largest_weights_kept_per_replicate": (
                args.keep_largest_weights
            ),
            "held_out_interpolation_conditions": (
                args.fixed_condition is None
                and
                args.conditions_file is None
                and args.condition_design == "interstitial"
            ),
            "checkpoint_final_metrics": checkpoint.get("final_metrics", {}),
        }, stream, indent=2)
    print(f"\nResults: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
