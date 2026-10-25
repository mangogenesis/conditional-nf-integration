#!/usr/bin/env python3
"""Train the established marginal-then-copula workflow on MadGraph ttH.

The learner sees only f(u | sqrt(s_hat)); MadGraph event generation is never
called.  The full autoregressive copula is trained without graph selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

try:
    import run_adaptive_integrand_structure_search as baseline
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The original adaptive trainer and its project modules must be available."
    ) from exc

from madgraph_tth_integrand import MadGraphTTHIntegrand


BENCHMARK = "madgraph_tth_20"


def extract_scattering_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--standalone",
        type=Path,
        default=Path("generated/gg_tth_full_standalone"),
    )
    parser.add_argument("--sqrt-s-min", type=float, default=600.0)
    parser.add_argument("--sqrt-s-max", type=float, default=2000.0)
    parser.add_argument("--width-window", type=float, default=15.0)
    parser.add_argument("--me-chunk", type=int, default=512)
    return parser.parse_known_args(argv)


class TorchMadGraphTarget:
    def __init__(self, evaluator, sqrt_s_min, sqrt_s_max, chunk):
        self.evaluator = evaluator
        self.sqrt_s_min = float(sqrt_s_min)
        self.sqrt_s_max = float(sqrt_s_max)
        self.chunk = int(chunk)

    def energy(self, cond: torch.Tensor) -> np.ndarray:
        z = cond[:, 0].detach().cpu().double().numpy()
        return self.sqrt_s_min + (self.sqrt_s_max - self.sqrt_s_min) * z

    def __call__(self, benchmark, x, cond):
        if benchmark != BENCHMARK:
            raise ValueError(f"Unexpected benchmark: {benchmark}")
        x_cpu = x.detach().cpu().double().numpy()
        energy = self.energy(cond)
        pieces = []
        for start in range(0, x_cpu.shape[0], self.chunk):
            end = min(start + self.chunk, x_cpu.shape[0])
            pieces.append(self.evaluator.log_integrand(x_cpu[start:end], energy[start:end]))
        result = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)
        return torch.as_tensor(result, device=x.device, dtype=x.dtype)


def main() -> None:
    scattering, remaining = extract_scattering_args(sys.argv[1:])
    if scattering.sqrt_s_max <= scattering.sqrt_s_min:
        raise ValueError("--sqrt-s-max must exceed --sqrt-s-min")

    evaluator = MadGraphTTHIntegrand(
        scattering.standalone,
        width_window=scattering.width_window,
    )
    if scattering.sqrt_s_min <= evaluator.phase_space.minimum_energy:
        raise ValueError(
            f"--sqrt-s-min must exceed {evaluator.phase_space.minimum_energy:.3f} GeV"
        )
    target = TorchMadGraphTarget(
        evaluator,
        scattering.sqrt_s_min,
        scattering.sqrt_s_max,
        scattering.me_chunk,
    )

    def sample_scattering_conditions(n, device):
        # Three smooth features preserve the existing conditioner interface.
        z = torch.rand(n, 1, device=device)
        return torch.cat((z, z * z, torch.log1p(z) / np.log(2.0)), dim=1)

    baseline.SUPPORTED_BENCHMARKS = {BENCHMARK}
    baseline.BENCHMARK_DIMENSIONS = {BENCHMARK: 20}
    baseline.sample_conditions = sample_scattering_conditions
    baseline.target_log_integrand = target

    # Run only the stable full autoregressive model: annealed marginals,
    # annealed bridge copula, formal KL refinement, then chi^2 fine-tuning.
    filtered = []
    skip_next = False
    for index, value in enumerate(remaining):
        if skip_next:
            skip_next = False
            continue
        if value == "--benchmarks":
            skip_next = True
            continue
        if value.startswith("--benchmarks="):
            continue
        if value == "--max-rounds":
            skip_next = True
            continue
        if value.startswith("--max-rounds="):
            continue
        if value in ("--compile-kernels", "--compile-model"):
            continue
        filtered.append(value)

    if not any(v == "--output" or v.startswith("--output=") for v in filtered):
        filtered.extend(["--output", "results/madgraph_tth_conditional_copula"])
    filtered.extend(
        ["--benchmarks", BENCHMARK, "--max-rounds", "0", "--no-compile-kernels", "--no-compile-model"]
    )
    sys.argv = [sys.argv[0], *filtered]

    print("MADGRAPH CONDITIONAL ttH COPULA")
    print(f"standalone              = {scattering.standalone.resolve()}")
    print(f"dimension               = 20")
    print(f"condition               = sqrt(s_hat) in [{scattering.sqrt_s_min:g}, {scattering.sqrt_s_max:g}] GeV")
    print(f"resonance mapping       = +/- {scattering.width_window:g} Gamma")
    print("target samples          = NEVER USED")
    print("structure search        = disabled (full autoregressive copula)")
    baseline.main()


if __name__ == "__main__":
    main()
