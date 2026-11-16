#!/usr/bin/env python3
"""Train the established conditional copula workflow for a configured process."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import run_adaptive_integrand_structure_search as baseline
from madgraph_process_integrand import ConfiguredMadGraphIntegrand, load_process_spec

_WORKER_EVALUATOR = None


def _init_me_worker(spec, root, alpha_s):
    """Load an independent matrix element in a pool worker.

    Each worker inherits the parent's CWD, which must be inside the generated
    standalone directory: MadGraph's lha_read resolves Cards/ from the CWD.
    """
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = ConfiguredMadGraphIntegrand(spec, Path(root), alpha_s)


def _evaluate_chunk(payload):
    unit, energy = payload
    return _WORKER_EVALUATOR.log_integrand(unit, energy)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--channel",
        default=None,
        help="Channel name from the config's `channels` list (required for multi-subprocess configs).",
    )
    parser.add_argument("--alpha-s", type=float, default=None)
    parser.add_argument("--me-chunk", type=int, default=512)
    parser.add_argument(
        "--me-workers",
        type=int,
        default=0,
        help=(
            "Processes used to evaluate the CPU matrix element. 0 keeps the "
            "original serial loop. The matrix-element calls are independent, "
            "so this is a pure throughput change: results are bit-identical "
            "and the training trajectory is unchanged. 48 measured optimal on "
            "the 192-thread odyssey node (20x on the matrix element)."
        ),
    )
    process_args, remaining = parser.parse_known_args()

    config_data = json.loads(Path(process_args.config).read_text(encoding="utf-8"))
    if config_data.get("channels") and process_args.channel is None:
        raise ValueError(
            "This config defines multiple channels; pass --channel with one of: "
            + ", ".join(item["name"] for item in config_data["channels"])
        )
    spec = load_process_spec(process_args.config, channel=process_args.channel)
    alpha_s = (
        process_args.alpha_s
        if process_args.alpha_s is not None
        else float(spec.get("alpha_s", 0.118))
    )
    evaluator = ConfiguredMadGraphIntegrand(spec, PROJECT, alpha_s)

    pool = None
    if process_args.me_workers > 0:
        # Fork is safe here: the workers only run numpy and the Fortran matrix
        # element; they never touch torch/ROCm. Fork also avoids paying the
        # 3.6k-line engine import in every worker.
        try:
            context = mp.get_context("fork")
        except ValueError:  # Windows
            context = mp.get_context("spawn")
        pool = context.Pool(
            processes=process_args.me_workers,
            initializer=_init_me_worker,
            initargs=(spec, str(PROJECT), alpha_s),
        )
    condition = spec["condition"]
    minimum = float(condition["minimum"])
    maximum = float(condition["maximum"])
    if minimum <= evaluator.phase_space.minimum_energy:
        raise ValueError(
            f"Condition minimum must exceed threshold "
            f"{evaluator.phase_space.minimum_energy:g} GeV"
        )
    benchmark = f"madgraph_{spec['name']}_{evaluator.dimension}"

    def sample_conditions(n, device):
        z = torch.rand(n, 1, device=device)
        return torch.cat((z, z * z, torch.log1p(z) / np.log(2.0)), dim=1)

    def target_log_integrand(name, unit, cond):
        if name != benchmark:
            raise ValueError(f"Unexpected benchmark {name!r}")
        z = cond[:, 0].detach().cpu().double().numpy()
        energy = minimum + (maximum - minimum) * z
        unit_numpy = unit.detach().cpu().double().numpy()
        payloads = []
        for start in range(0, len(unit_numpy), process_args.me_chunk):
            stop = min(start + process_args.me_chunk, len(unit_numpy))
            payloads.append((unit_numpy[start:stop], energy[start:stop]))
        if pool is None:
            pieces = [
                evaluator.log_integrand(chunk_unit, chunk_energy)
                for chunk_unit, chunk_energy in payloads
            ]
        else:
            # pool.map preserves order, and the Fortran matrix element is
            # deterministic, so the returned integrand is bit-identical to the
            # serial path above.
            pieces = pool.map(_evaluate_chunk, payloads, chunksize=1)
        return torch.as_tensor(np.concatenate(pieces), device=unit.device, dtype=unit.dtype)

    baseline.SUPPORTED_BENCHMARKS = {benchmark}
    baseline.BENCHMARK_DIMENSIONS = {benchmark: evaluator.dimension}
    baseline.sample_conditions = sample_conditions
    baseline.target_log_integrand = target_log_integrand

    filtered = []
    skip = False
    for value in remaining:
        if skip:
            skip = False
            continue
        if value in ("--benchmarks", "--max-rounds"):
            skip = True
            continue
        if value.startswith(("--benchmarks=", "--max-rounds=")):
            continue
        if value in ("--compile-kernels", "--compile-model"):
            continue
        filtered.append(value)
    if not any(item == "--output" or item.startswith("--output=") for item in filtered):
        filtered.extend(["--output", f"results/{spec['name']}_conditional_copula"])
    filtered.extend([
        "--benchmarks", benchmark, "--max-rounds", "0",
        "--no-compile-kernels", "--no-compile-model",
    ])
    sys.argv = [sys.argv[0], *filtered]
    print(f"Configured process       = {spec['name']}")
    print(f"MadGraph process         = {spec['process']}")
    print(f"MadGraph subprocess      = {spec.get('subprocess', '(single)')}")
    print(f"alpha_s                  = {alpha_s}")
    print(f"dimension                = {evaluator.dimension}")
    print(f"sqrt(s_hat) range        = [{minimum:g}, {maximum:g}] GeV")
    print(f"matrix-element workers   = {process_args.me_workers}")
    try:
        baseline.main()
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
