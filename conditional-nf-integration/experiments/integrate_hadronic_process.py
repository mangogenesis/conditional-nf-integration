#!/usr/bin/env python3
"""Hadronic pp -> Z + j integration from the conditional partonic NFs.

The partonic NFs are trained per subprocess group and per sqrt(s-hat) range.
For each channel this script

  1. evaluates the NF partonic cross section on a log-spaced sqrt(s-hat) grid,
  2. multiplies by the LHAPDF parton luminosity dL/dsqrt(s-hat),
  3. integrates over sqrt(s-hat) and sums the channels,
  4. adds the measured [2000, 13000] GeV tail as a constant.

The PDF set is NNPDF23_lo_as_0130_qed (LHAPDF ID 247000), the set behind
MG5's built-in `nn23lo1`.  The NF matrix elements and the MadEvent reference
both use alpha_s = 0.130 at the fixed scale 91.188 GeV.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
for _path in (PROJECT, PROJECT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_adaptive_integrand_structure_search as adaptive
from madgraph_process_integrand import ConfiguredMadGraphIntegrand, load_process_spec
from validate_madgraph_process_integral import GEV2_TO_PB, load_model, nf_estimate

import lhapdf


def trapz_with_error(grid: np.ndarray, values: np.ndarray, errors: np.ndarray):
    """Trapezoid integral and its independent-point error."""
    grid = np.asarray(grid, dtype=np.float64)
    weights = np.zeros_like(grid)
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    value = float(np.sum(weights * values))
    error = float(np.sqrt(np.sum((weights * errors) ** 2)))
    return value, error


def luminosity(
    pdf,
    sqrts_hat: float,
    sqrts: float,
    q2: float,
    pairs: list[list[int]],
    y_nodes: np.ndarray,
    y_weights: np.ndarray,
) -> float:
    """dL/dsqrt(s-hat) for one channel, summed over pairs and beam orderings.

    dL/dsqrt(s-hat) = (2 sqrt(s-hat)/s) int dY f_a(x1) f_b(x2),
    x1,2 = sqrt(s-hat/s) exp(+-Y), |Y| <= ln(sqrt(s/s-hat)).
    """
    s = sqrts * sqrts
    shat = sqrts_hat * sqrts_hat
    xroot = math.sqrt(shat / s)
    ymax = 0.5 * math.log(s / shat)
    if ymax <= 0.0:
        return 0.0
    y = ymax * y_nodes
    weights = ymax * y_weights
    total = 0.0
    for first, second in pairs:
        # For identical partons the two beam assignments are the same
        # configuration; summing both would double count the luminosity.
        orderings = (
            ((first, second),)
            if first == second
            else ((first, second), (second, first))
        )
        for left, right in orderings:
            x1 = xroot * np.exp(y)
            x2 = xroot * np.exp(-y)
            # q2 is a squared scale; use xfxQ2 (xfxQ would evaluate at Q=q2).
            xf1 = np.array([pdf.xfxQ2(int(left), float(x), q2) for x in x1])
            xf2 = np.array([pdf.xfxQ2(int(right), float(x), q2) for x in x2])
            total += float(np.sum(weights * (xf1 / x1) * (xf2 / x2)))
    return (2.0 * sqrts_hat / s) * total


def checkpoint_path(
    config_name: str,
    channel: str,
    dimension: int = 2,
    override: str | None = None,
) -> Path:
    """Locate a channel checkpoint.

    ``dimension`` is the phase-space dimension carried by the benchmark name
    (2 for the Z+1 jet channels, 5 for Z+2 jets).  ``override`` points a
    channel at a shared model, e.g. a C-conjugate channel.
    """
    if override:
        path = Path(override)
        return path if path.is_absolute() else PROJECT / path
    return (
        PROJECT / "results" / f"{config_name}_{channel}_copula_balanced_gpu"
        / f"madgraph_{config_name}_{channel}_{dimension}_final_model.pt"
    )


def load_acceptance_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a lab/CM acceptance table into (bin centers, factors)."""
    import csv

    centers = []
    ratios = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ratio = float(row["lab_over_cm"])
            if not math.isfinite(ratio) or ratio <= 0.0:
                continue
            centers.append(
                0.5 * (float(row["shat_low_gev"]) + float(row["shat_high_gev"]))
            )
            ratios.append(ratio)
    order = np.argsort(centers)
    return (
        np.asarray(centers, dtype=float)[order],
        np.asarray(ratios, dtype=float)[order],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low-config", type=Path,
        default=PROJECT / "configs/processes/zj_hadronic_low.json",
    )
    parser.add_argument(
        "--high-config", type=Path,
        default=PROJECT / "configs/processes/zj_hadronic_high.json",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nf-evals", type=int, default=100_000)
    parser.add_argument("--gpu-chunk", type=int, default=20_000)
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--grid-low-points", type=int, default=60)
    parser.add_argument("--grid-high-points", type=int, default=60)
    parser.add_argument(
        "--split", type=float, default=None,
        help=(
            "sqrt(s-hat) split between the low and high models. Defaults to "
            "the low config maximum (250 GeV). The low model condition range "
            "is unchanged; this only moves the integration boundary."
        ),
    )
    parser.add_argument(
        "--acceptance-csv", type=Path, default=None,
        help=(
            "CSV from scripts/zjj_acceptance_from_lhe.py with "
            "shat_low_gev,shat_high_gev,lab_over_cm. Multiplies each partonic "
            "cross section by the interpolated lab/CM factor."
        ),
    )
    parser.add_argument(
        "--acceptance-dir", type=Path, default=None,
        help=(
            "Directory holding eta_frame_<P1_group>.csv. Each channel uses the "
            "table named by its `group` field; overrides --acceptance-csv."
        ),
    )
    parser.add_argument("--y-points", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--tail-pb", type=float, default=0.1389)
    parser.add_argument("--tail-error-pb", type=float, default=0.00026)
    parser.add_argument("--interpolation-points", type=int, default=50)
    parser.add_argument("--interpolation-evals", type=int, default=50_000)
    parser.add_argument("--skip-interpolation", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/zj_hadronic_integral",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The driver chdirs into the standalone so MadGraph can resolve Cards/;
    # anchor the acceptance inputs before that happens.
    if args.acceptance_csv is not None:
        args.acceptance_csv = args.acceptance_csv.resolve()
    if args.acceptance_dir is not None:
        args.acceptance_dir = args.acceptance_dir.resolve()
    args.low_config = args.low_config.resolve()
    args.high_config = args.high_config.resolve()
    args.output = args.output.resolve()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    low_config = json.loads(args.low_config.read_text(encoding="utf-8"))
    high_config = json.loads(args.high_config.read_text(encoding="utf-8"))
    low_channels = {item["name"]: item for item in low_config["channels"]}
    high_channels = {item["name"]: item for item in high_config["channels"]}
    if set(low_channels) != set(high_channels):
        raise ValueError("Low and high configs must define the same channel names")

    standalone = (PROJECT / low_config["standalone"]).resolve()
    if not standalone.is_dir():
        raise FileNotFoundError(f"Standalone directory not found: {standalone}")
    os.chdir(standalone)

    beam = low_config["beam"]
    pdf = lhapdf.mkPDF(beam["pdf"], int(beam.get("member", 0)))
    q2 = float(beam["scale"]) ** 2
    sqrts = float(beam["sqrt_s"])
    y_nodes, y_weights = np.polynomial.legendre.leggauss(args.y_points)

    acceptance_centers = None
    acceptance_ratios = None
    if args.acceptance_csv is not None:
        acceptance_centers, acceptance_ratios = load_acceptance_csv(
            args.acceptance_csv
        )
    acceptance_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    low_min = float(low_config["condition"]["minimum"])
    low_max = float(low_config["condition"]["maximum"])
    high_min = float(high_config["condition"]["minimum"])
    high_max = float(high_config["condition"]["maximum"])
    split = float(args.split) if args.split is not None else low_max
    if not (low_min < split <= low_max):
        raise ValueError(
            f"--split must lie in the low model range ({low_min}, {low_max}], "
            f"got {split}"
        )
    if not (high_min <= split < high_max):
        raise ValueError(
            f"--split must lie in the high model range [{high_min}, {high_max}), "
            f"got {split}"
        )
    low_grid = np.geomspace(low_min, split, args.grid_low_points)
    # The low model is trained on [114, 250] and the high model on
    # [200, 2000].  The integration boundary is normally 250 GeV; --split
    # moves it (e.g. 200 GeV to match the native slices) without changing
    # either model's condition normalization.
    high_grid = np.geomspace(split, high_max, args.grid_high_points)

    device = torch.device(args.device)
    nf_args = SimpleNamespace(
        nf_evals=args.nf_evals,
        gpu_chunk=args.gpu_chunk,
        uniform_fraction=args.uniform_fraction,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    alpha_s = float(low_config.get("alpha_s", 0.118))
    channel_rows = []
    total_integral = 0.0
    total_variance = 0.0
    nf_calls = 0
    pdf_calls = 0
    started = time.perf_counter()

    for channel_name, channel in low_channels.items():
        channel_centers, channel_ratios = acceptance_centers, acceptance_ratios
        if args.acceptance_dir is not None and channel.get("group"):
            group = channel["group"]
            if group not in acceptance_cache:
                acceptance_cache[group] = load_acceptance_csv(
                    args.acceptance_dir / f"eta_frame_{group}.csv"
                )
            channel_centers, channel_ratios = acceptance_cache[group]
        low_spec = load_process_spec(args.low_config, channel=channel_name)
        high_spec = load_process_spec(args.high_config, channel=channel_name)
        evaluator = ConfiguredMadGraphIntegrand(low_spec, PROJECT, alpha_s)
        dimension = evaluator.dimension
        low_checkpoint = checkpoint_path(
            low_config["name"], channel_name, dimension, channel.get("checkpoint")
        )
        high_checkpoint = checkpoint_path(
            high_config["name"], channel_name, dimension, channel.get("checkpoint")
        )
        for path in (low_checkpoint, high_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
        low_model, _ = load_model(low_checkpoint, evaluator.dimension, device)
        high_model, _ = load_model(high_checkpoint, evaluator.dimension, device)

        rows = []
        for grid, model, minimum, maximum, label in (
            (low_grid, low_model, low_min, low_max, "low"),
            (high_grid, high_model, high_min, high_max, "high"),
        ):
            for sqrts_hat in grid:
                row = nf_estimate(
                    model, evaluator, float(sqrts_hat), minimum, maximum, nf_args, device
                )
                lum = float(channel.get("multiplicity", 1.0)) * luminosity(
                    pdf, float(sqrts_hat), sqrts, q2,
                    channel["pdf_pairs"], y_nodes, y_weights,
                )
                acceptance = 1.0
                if channel_centers is not None:
                    acceptance = float(
                        np.interp(
                            float(sqrts_hat),
                            channel_centers,
                            channel_ratios,
                            left=channel_ratios[0],
                            right=channel_ratios[-1],
                        )
                    )
                rows.append({
                    "range": label,
                    "sqrts_hat": float(sqrts_hat),
                    "luminosity": lum,
                    "acceptance": acceptance,
                    "sigma_gev2": row["estimate_gev2"],
                    "sigma_err_gev2": row["stderr_gev2"],
                    "ess": row["normalized_ess"],
                    "nf_seconds": row["seconds"],
                })
                nf_calls += args.nf_evals
                pdf_calls += args.y_points * len(channel["pdf_pairs"]) * 2

        low_rows = [row for row in rows if row["range"] == "low"]
        high_rows = [row for row in rows if row["range"] == "high"]
        low_integral, low_error = trapz_with_error(
            np.array([row["sqrts_hat"] for row in low_rows]),
            np.array([
                row["luminosity"] * row["acceptance"] * row["sigma_gev2"]
                for row in low_rows
            ]),
            np.array([
                row["luminosity"] * row["acceptance"] * row["sigma_err_gev2"]
                for row in low_rows
            ]),
        )
        high_integral, high_error = trapz_with_error(
            np.array([row["sqrts_hat"] for row in high_rows]),
            np.array([
                row["luminosity"] * row["acceptance"] * row["sigma_gev2"]
                for row in high_rows
            ]),
            np.array([
                row["luminosity"] * row["acceptance"] * row["sigma_err_gev2"]
                for row in high_rows
            ]),
        )
        integral = low_integral + high_integral
        error = math.sqrt(low_error ** 2 + high_error ** 2)
        integral_pb = integral * GEV2_TO_PB
        error_pb = error * GEV2_TO_PB
        total_integral += integral_pb
        total_variance += error_pb ** 2
        ess = float(np.mean([row["ess"] for row in rows]))
        channel_rows.append({
            "channel": channel_name,
            "integral_pb": integral_pb,
            "error_pb": error_pb,
            "ess_mean": ess,
            "grid_points": len(rows),
            "nf_samples": args.nf_evals * len(rows),
        })
        print(
            f"{channel_name:10s} {integral_pb:12.6g} +/- {error_pb:10.4g} pb  "
            f"ESS={ess:.4f}  grid={len(rows)}",
            flush=True,
        )
        (output / f"grid_{channel_name}.csv").write_text(
            "range,sqrts_hat,luminosity,acceptance,sigma_gev2,"
            "sigma_err_gev2,ess,nf_seconds\n"
            + "\n".join(
                f"{row['range']},{row['sqrts_hat']:.12g},{row['luminosity']:.12g},"
                f"{row['acceptance']:.12g},"
                f"{row['sigma_gev2']:.12g},{row['sigma_err_gev2']:.12g},"
                f"{row['ess']:.12g},{row['nf_seconds']:.12g}"
                for row in rows
            ) + "\n",
            encoding="utf-8",
        )

    total_before_tail = total_integral
    total_with_tail = total_integral + args.tail_pb
    total_error = math.sqrt(total_variance + args.tail_error_pb ** 2)
    elapsed = time.perf_counter() - started

    summary = {
        "low_config": str(args.low_config),
        "high_config": str(args.high_config),
        "pdf": beam["pdf"],
        "lhaid": beam.get("lhaid"),
        "alpha_s": alpha_s,
        "acceptance_csv": str(args.acceptance_csv) if args.acceptance_csv else None,
        "scale_gev": beam["scale"],
        "sqrt_s_gev": sqrts,
        "split_gev": split,
        "low_range_gev": [low_min, split],
        "high_range_gev": [split, high_max],
        "grid_low_points": args.grid_low_points,
        "grid_high_points": args.grid_high_points,
        "nf_evals_per_point": args.nf_evals,
        "nf_samples": nf_calls,
        "pdf_calls": pdf_calls,
        "generation_seconds": elapsed,
        "total_before_tail_pb": total_before_tail,
        "tail_pb": args.tail_pb,
        "tail_error_pb": args.tail_error_pb,
        "total_pb": total_with_tail,
        "total_error_pb": total_error,
        "channels": channel_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "per_channel.csv").write_text(
        "channel,integral_pb,error_pb,ess_mean,grid_points,nf_samples\n"
        + "\n".join(
            f"{row['channel']},{row['integral_pb']:.12g},{row['error_pb']:.12g},"
            f"{row['ess_mean']:.12g},{row['grid_points']},{row['nf_samples']}"
            for row in channel_rows
        ) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"integral [114,2000] GeV : {total_before_tail:.6g} +/- {total_error:.4g} pb")
    print(f"tail     [2000,13000] GeV: {args.tail_pb:.6g} +/- {args.tail_error_pb:.3g} pb")
    print(f"total                   : {total_with_tail:.6g} +/- {total_error:.4g} pb")
    print(f"NF generation time      : {elapsed:.1f} s")
    print(f"NF samples              : {nf_calls:,}")
    print(f"PDF calls               : {pdf_calls:,}")

    if not args.skip_interpolation and args.interpolation_points > 0:
        interp_args = SimpleNamespace(
            nf_evals=args.interpolation_evals,
            gpu_chunk=args.gpu_chunk,
            uniform_fraction=args.uniform_fraction,
        )
        interp_low = np.linspace(low_min, low_max, args.interpolation_points)
        interp_high = np.linspace(high_min, high_max, args.interpolation_points)
        for label, grid, model, minimum, maximum, config_name in (
            ("low", interp_low, None, low_min, low_max, low_config["name"]),
            ("high", interp_high, None, high_min, high_max, high_config["name"]),
        ):
            lines = ["sqrts_hat_gev,channel,nf_pb,nf_err_pb,ess,seconds"]
            for channel_name in low_channels:
                low_spec = load_process_spec(args.low_config, channel=channel_name)
                evaluator = ConfiguredMadGraphIntegrand(low_spec, PROJECT, alpha_s)
                checkpoint = checkpoint_path(
                    config_name,
                    channel_name,
                    evaluator.dimension,
                    low_channels[channel_name].get("checkpoint"),
                )
                model, _ = load_model(checkpoint, evaluator.dimension, device)
                for sqrts_hat in grid:
                    row = nf_estimate(
                        model, evaluator, float(sqrts_hat), minimum, maximum,
                        interp_args, device,
                    )
                    lines.append(
                        f"{sqrts_hat:.12g},{channel_name},"
                        f"{row['estimate_pb']:.12g},{row['stderr_pb']:.12g},"
                        f"{row['normalized_ess']:.12g},{row['seconds']:.12g}"
                    )
            (output / f"interpolation_{label}.csv").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            print(f"wrote interpolation_{label}.csv")


if __name__ == "__main__":
    main()
