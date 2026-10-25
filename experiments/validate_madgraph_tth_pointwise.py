#!/usr/bin/env python3
"""Compare two independently generated ttH matrix elements point by point.

The candidate is the standalone library used by the conditional NF.  The
reference is a fresh standalone export made by the same MG5 installation used
for MadEvent.  Both receive the same parameter card and exactly the same
external four-momenta.  Thus this test contains no Monte Carlo integration and
no phase-space-normalization ambiguity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from madgraph_tth_integrand import MadGraphTTHIntegrand


PROCESS = (
    "g g > t t~ h, "
    "(t > b w+, w+ > e+ ve), "
    "(t~ > b~ w-, w- > mu- vm~), "
    "(h > b b~) @1"
)


def resolve_executable(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    found = shutil.which(value)
    if found:
        return Path(found).resolve()
    raise FileNotFoundError(f"Could not find {value!r}; pass --mg5 /path/to/bin/mg5_aMC")


def generate_reference(mg5: Path, output: Path) -> None:
    if output.exists():
        libraries = list((output / "SubProcesses").glob("P*/matrix2py*.so"))
        if not libraries:
            raise FileExistsError(
                f"Reference path exists but has no matrix2py library: {output}"
            )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        "set automatic_html_opening False\n"
        "import model sm\n"
        f"generate {PROCESS}\n"
        f"output standalone {output.resolve()} --prefix=int\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as handle:
        handle.write(commands)
        command_card = Path(handle.name)
    try:
        subprocess.run([str(mg5), str(command_card)], check=True)
    finally:
        command_card.unlink(missing_ok=True)

    process_dirs = sorted(
        path for path in (output / "SubProcesses").glob("P*") if path.is_dir()
    )
    failures = []
    for process_dir in process_dirs:
        completed = subprocess.run(
            ["make", "matrix2py.so"], cwd=process_dir, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            failures.append(f"{process_dir}:\n{completed.stdout}")
    if not list((output / "SubProcesses").glob("P*/matrix2py*.so")):
        raise RuntimeError("Could not compile reference matrix2py library:\n" + "\n".join(failures))


def install_parameter_card(source: Path, standalone: Path) -> None:
    destination = standalone / "Cards" / "param_card.dat"
    if not source.is_file():
        raise FileNotFoundError(f"Parameter card not found: {source}")
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def momenta_from_unit_cube(evaluator, unit: np.ndarray, energies: np.ndarray):
    final, jacobian = evaluator.phase_space.map(unit, energies)
    initial = np.zeros((len(unit), 2, 4), dtype=np.float64)
    initial[:, 0, 0] = energies / 2.0
    initial[:, 0, 3] = energies / 2.0
    initial[:, 1, 0] = energies / 2.0
    initial[:, 1, 3] = -energies / 2.0
    return np.concatenate((initial, final), axis=1), jacobian


def evaluate_matrix(evaluator, momenta: np.ndarray, alpha_s: float) -> np.ndarray:
    values = np.empty(len(momenta), dtype=np.float64)
    for index, event in enumerate(momenta):
        p = np.asfortranarray(event.T)
        if evaluator._evaluate_name == "get_value":
            values[index] = float(evaluator._evaluate(p, alpha_s, -1))
        elif evaluator._evaluate_name == "smatrix":
            values[index] = float(evaluator._evaluate(p, alpha_s))
        else:
            values[index] = float(evaluator._evaluate(p, alpha_s, -1))
    return values


def relative_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mg5", required=True)
    parser.add_argument("--candidate", type=Path,
                        default=Path("generated/gg_tth_full_standalone"))
    parser.add_argument("--reference", type=Path,
                        default=Path("generated/gg_tth_pointwise_reference"))
    parser.add_argument("--param-card", type=Path, default=None)
    parser.add_argument("--energies", type=float, nargs="+", default=(800, 1400, 1990, 2000))
    parser.add_argument("--points-per-energy", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--alpha-s", type=float, default=0.118)
    parser.add_argument("--width-window", type=float, default=15.0)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path,
                        default=Path("results/madgraph_tth_pointwise_validation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mg5 = resolve_executable(args.mg5)
    candidate = args.candidate.resolve()
    reference = args.reference.resolve()
    param_card = (args.param_card or candidate / "Cards" / "param_card.dat").resolve()

    started = time.perf_counter()
    generate_reference(mg5, reference)
    install_parameter_card(param_card, reference)

    # Load and evaluate sequentially.  Both f2py exports use the module name
    # matrix2py, so retaining both module objects is safe after initialization.
    candidate_me = MadGraphTTHIntegrand(candidate, args.width_window, args.alpha_s)
    reference_me = MadGraphTTHIntegrand(reference, args.width_window, args.alpha_s)

    rng = np.random.default_rng(args.seed)
    rows = []
    for energy in args.energies:
        unit = rng.random((args.points_per_energy, 20))
        energies = np.full(args.points_per_energy, energy, dtype=np.float64)
        momenta, jacobian = momenta_from_unit_cube(candidate_me, unit, energies)
        candidate_values = evaluate_matrix(candidate_me, momenta, args.alpha_s)
        reference_values = evaluate_matrix(reference_me, momenta, args.alpha_s)
        rel = relative_difference(candidate_values, reference_values)
        for point in range(args.points_per_energy):
            rows.append({
                "energy_gev": energy,
                "point": point,
                "candidate_me2": candidate_values[point],
                "reference_me2": reference_values[point],
                "relative_difference": rel[point],
                "phase_space_jacobian": jacobian[point],
                "finite": bool(
                    np.isfinite(candidate_values[point])
                    and np.isfinite(reference_values[point])
                    and np.isfinite(rel[point])
                ),
            })

    frame = pd.DataFrame(rows)
    summary = frame.groupby("energy_gev", as_index=False).agg(
        points=("point", "count"),
        finite_fraction=("finite", "mean"),
        max_abs_relative_difference=("relative_difference", lambda x: float(np.max(np.abs(x)))),
        rms_relative_difference=("relative_difference", lambda x: float(np.sqrt(np.mean(x * x)))),
        mean_ratio=("candidate_me2", lambda x: math.nan),
    )
    ratios = frame.assign(ratio=frame.candidate_me2 / frame.reference_me2).groupby("energy_gev")["ratio"].mean()
    summary["mean_ratio"] = summary["energy_gev"].map(ratios)
    maximum = float(np.max(np.abs(frame["relative_difference"])))
    passed = bool(frame["finite"].all() and maximum <= args.tolerance)

    frame.to_csv(args.output / "pointwise_results.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    config = {
        "process": PROCESS,
        "mg5": str(mg5),
        "candidate": str(candidate),
        "reference": str(reference),
        "param_card": str(param_card),
        "energies_gev": list(args.energies),
        "points_per_energy": args.points_per_energy,
        "seed": args.seed,
        "alpha_s": args.alpha_s,
        "width_window": args.width_window,
        "tolerance": args.tolerance,
        "max_abs_relative_difference": maximum,
        "passed": passed,
        "seconds": time.perf_counter() - started,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print("\nPOINTWISE MATRIX-ELEMENT COMPARISON")
    print(summary.to_string(index=False))
    print(f"\nmax |relative difference| = {maximum:.6e}")
    print(f"tolerance                 = {args.tolerance:.6e}")
    print(f"result                    = {'PASS' if passed else 'FAIL'}")
    print(f"results                   = {args.output.resolve()}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
