#!/usr/bin/env python3
"""Run an independent MadEvent normalization check for the ttH benchmark.

This generates (or reuses) a normal MadEvent process directory, integrates the
same decay-chain process at fixed partonic center-of-mass energies, and compares
MadEvent's cross sections with the fixed-energy NF/Sobol validation results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import pandas as pd


PROCESS = (
    "g g > t t~ h, "
    "(t > b w+, w+ > e+ ve), "
    "(t~ > b~ w-, w- > mu- vm~), "
    "(h > b b~) @1"
)


def resolve_executable(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Could not find {value!r}; pass --mg5 /path/to/bin/mg5_aMC")


def generate_process(mg5: str, process_dir: Path) -> None:
    if process_dir.exists():
        if not (process_dir / "bin" / "madevent").is_file():
            raise FileExistsError(f"Existing path is not a MadEvent directory: {process_dir}")
        return
    process_dir.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        "set automatic_html_opening False\n"
        "import model sm\n"
        f"generate {PROCESS}\n"
        f"output {process_dir.resolve()}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as handle:
        handle.write(commands)
        card = Path(handle.name)
    try:
        subprocess.run([mg5, str(card)], check=True)
    finally:
        card.unlink(missing_ok=True)


def install_param_card(source: Path, process_dir: Path) -> None:
    """Use the exact parameter card employed by the standalone evaluator."""
    if not source.is_file():
        raise FileNotFoundError(f"Standalone parameter card not found: {source}")
    destination = process_dir / "Cards" / "param_card.dat"
    shutil.copy2(source, destination)


def run_card_settings(energy: float, args) -> dict[str, str | int]:
    beam = energy / 2.0
    # cut_decays=False is essential: every final particle belongs to a declared
    # decay chain, and the reference integral has no analysis-level cuts.
    return {
        "lpp1": 0,
        "lpp2": 0,
        "ebeam1": f"{beam:.12g}",
        "ebeam2": f"{beam:.12g}",
        "bwcutoff": f"{args.width_window:.12g}",
        "cut_decays": "False",
        "fixed_ren_scale": "True",
        "fixed_fac_scale": "True",
        "scale": f"{args.ren_scale:.12g}",
        "dsqrt_q2fact1": f"{args.ren_scale:.12g}",
        "dsqrt_q2fact2": f"{args.ren_scale:.12g}",
        "nevents": args.nevents,
        "req_acc": f"{args.required_accuracy:.12g}",
        "iseed": args.seed + int(round(energy)),
        "use_syst": "False",
        "ickkw": 0,
    }


def update_run_card(process_dir: Path, settings: dict[str, str | int]) -> None:
    """Update MadEvent's value-before-name run-card format in place."""
    path = process_dir / "Cards" / "run_card.dat"
    text = path.read_text(encoding="utf-8")
    missing = []
    for key, value in settings.items():
        pattern = re.compile(
            rf"(?im)^(?P<indent>\s*)(?P<old>[^#!\n=]+?)(?P<sep>\s*=\s*{re.escape(key)}\b)"
        )
        text, count = pattern.subn(
            lambda match: f"{match.group('indent')}{value}{match.group('sep')}",
            text,
            count=1,
        )
        if count == 0:
            missing.append(key)
    # Scale-factorization fields vary across MG5 releases and are immaterial
    # when lpp1=lpp2=0.  All other requested controls must be present.
    optional = {
        "dsqrt_q2fact1",
        "dsqrt_q2fact2",
        "use_syst",
        # MG5 3.8 omits these from some process-specific LO run cards.  The
        # process has no matching and MadEvent then uses its generated survey
        # and refinement defaults.
        "ickkw",
        "req_acc",
    }
    required_missing = [key for key in missing if key not in optional]
    if required_missing:
        raise RuntimeError(
            f"Run card {path} lacks required fields: {required_missing}. "
            "No integration was started."
        )
    optional_missing = [key for key in missing if key in optional]
    if optional_missing:
        print(
            f"MadEvent run card omits optional fields {optional_missing}; "
            "using process defaults.",
            flush=True,
        )
    path.write_text(text, encoding="utf-8")


def madevent_commands(run_name: str, args) -> str:
    if args.integration_only:
        commands = (
            f"survey {run_name} --points={args.survey_points} "
            f"--iterations={args.survey_iterations} "
            f"--accuracy={args.survey_accuracy:.12g}\n"
        )
        if args.refine_accuracy is not None:
            commands += (
                f"refine {args.refine_accuracy:.12g} {args.refine_max_channels}\n"
            )
        return commands
    return f"generate_events {run_name} -f\n"


def parse_cross_section(text: str) -> tuple[float, float]:
    number = r"[+\-0-9.eEdD]+"
    patterns = (
        rf"Cross[- ]section\s*:\s*({number})\s*(?:\+/-|\+-|±)\s*({number})(?:\s*pb)?",
        rf"cross[- ]section\s*=\s*({number})\s*(?:\+/-|\+-|±)\s*({number})(?:\s*pb)?",
        rf"Current estimate of cross[- ]section\s*:\s*({number})\s*(?:\+/-|\+-|±)\s*({number})(?:\s*pb)?",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            value, error = matches[-1]
            return float(value.replace("D", "E").replace("d", "e")), float(
                error.replace("D", "E").replace("d", "e")
            )
    raise RuntimeError("Could not parse a MadEvent cross section and uncertainty from its output")


def banner_cross_section(process_dir: Path, run_name: str) -> tuple[float, float] | None:
    banners = sorted((process_dir / "Events" / run_name).glob("*_banner.txt"))
    if not banners:
        return None
    text = banners[-1].read_text(encoding="utf-8", errors="replace")
    init = re.search(r"<init>(.*?)</init>", text, flags=re.IGNORECASE | re.DOTALL)
    if init:
        lines = [
            line.strip() for line in init.group(1).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        # First data line is the beam/PDF header.  Each following subprocess
        # line is XSECUP XERRUP XMAXUP LPRUP in the Les Houches convention.
        subprocesses = []
        for line in lines[1:]:
            fields = line.replace("D", "E").replace("d", "e").split()
            if len(fields) >= 4:
                try:
                    subprocesses.append((float(fields[0]), float(fields[1])))
                except ValueError:
                    continue
        if subprocesses:
            cross = sum(item[0] for item in subprocesses)
            error = math.sqrt(sum(item[1] ** 2 for item in subprocesses))
            return cross, error
    match = re.search(
        r"Integrated weight \(pb\)\s*:\s*([+\-0-9.eEdD]+)", text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1).replace("D", "E").replace("d", "e")), math.nan
    return None


def current_results_dat(process_dir: Path) -> tuple[float, float] | None:
    """Read MadEvent's current total cross section and integration error."""
    path = process_dir / "SubProcesses" / "results.dat"
    if not path.is_file():
        return None
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    fields = first.replace("D", "E").replace("d", "e").split()
    if len(fields) < 2:
        return None
    try:
        return float(fields[0]), float(fields[1])
    except ValueError:
        return None


def run_energy(process_dir: Path, energy: float, args) -> dict:
    run_name = f"{args.run_prefix}_{int(round(energy))}"
    update_run_card(process_dir, run_card_settings(energy, args))
    commands = madevent_commands(run_name, args)
    command_path = args.output / f"{run_name}_commands.txt"
    command_path.write_text(commands, encoding="utf-8")
    executable = process_dir / "bin" / "madevent"
    results_path = process_dir / "SubProcesses" / "results.dat"
    previous_results_mtime = results_path.stat().st_mtime_ns if results_path.is_file() else None
    started = time.perf_counter()
    completed = subprocess.run(
        [str(executable), str(command_path.resolve())], cwd=process_dir,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    log_path = args.output / f"{run_name}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MadEvent failed at {energy:g} GeV; see {log_path}")
    if args.integration_only:
        current_mtime = results_path.stat().st_mtime_ns if results_path.is_file() else None
        if current_mtime is None or current_mtime == previous_results_mtime:
            raise RuntimeError(
                f"MadEvent did not refresh SubProcesses/results.dat at {energy:g} GeV; "
                f"the survey/refine command may have failed. See {log_path}"
            )
    results_dat = current_results_dat(process_dir)
    if results_dat is not None:
        cross_pb, error_pb = results_dat
        (args.output / f"{run_name}_results.dat").write_text(
            (process_dir / "SubProcesses" / "results.dat").read_text(
                encoding="utf-8", errors="replace"
            ),
            encoding="utf-8",
        )
    else:
        try:
            cross_pb, error_pb = parse_cross_section(completed.stdout)
        except RuntimeError:
            banner_result = banner_cross_section(process_dir, run_name)
            if banner_result is None:
                raise
            cross_pb, error_pb = banner_result
    print(f"MadEvent {energy:g} GeV: {cross_pb * 1000:.6g} +/- {error_pb * 1000:.2g} fb", flush=True)
    return {
        "energy_gev": energy, "madevent_pb": cross_pb,
        "madevent_error_pb": error_pb, "madevent_fb": cross_pb * 1000.0,
        "madevent_error_fb": error_pb * 1000.0, "run_name": run_name,
        "seconds": elapsed, "log": str(log_path.resolve()),
    }


def attach_existing_results(rows: list[dict], validation_summary: Path) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if not validation_summary.is_file():
        return result
    previous = pd.read_csv(validation_summary)
    previous = previous[previous["scope"].eq("fixed_energy")]
    if "normalized_ess" not in previous:
        previous["normalized_ess"] = math.nan
    previous = previous[[
        "energy_gev", "method", "estimate_fb", "stderr_fb", "normalized_ess"
    ]]
    wide = previous.pivot(index="energy_gev", columns="method")
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    wide = wide.reset_index()
    result = result.merge(wide, on="energy_gev", how="left")
    for method in ("nf_mixture", "sobol_qmc"):
        estimate = f"{method}_estimate_fb"
        error = f"{method}_stderr_fb"
        if estimate in result:
            result[f"madevent_minus_{method}_fb"] = result["madevent_fb"] - result[estimate]
            denom = (result["madevent_error_fb"] ** 2 + result[error] ** 2) ** 0.5
            result[f"madevent_vs_{method}_pull"] = (
                result[f"madevent_minus_{method}_fb"] / denom
            )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mg5", default=os.environ.get("MG5AMC", "mg5_aMC"))
    parser.add_argument("--process-dir", type=Path, default=Path("generated/gg_tth_full_madevent"))
    parser.add_argument("--param-card", type=Path,
                        default=Path("generated/gg_tth_full_standalone/Cards/param_card.dat"))
    parser.add_argument("--energies", type=float, nargs="+", default=(600, 800, 1000, 1400, 2000))
    parser.add_argument("--width-window", type=float, default=15.0)
    parser.add_argument("--ren-scale", type=float, default=91.188,
                        help="fixed renormalization scale chosen to reproduce alpha_s(MZ) approximately")
    parser.add_argument("--required-accuracy", type=float, default=0.002)
    parser.add_argument("--nevents", type=int, default=10000)
    parser.add_argument(
        "--integration-only", action="store_true",
        help="run explicit MadEvent survey/refine integration without event generation",
    )
    parser.add_argument("--survey-points", type=int, default=10000)
    parser.add_argument("--survey-iterations", type=int, default=8)
    parser.add_argument("--survey-accuracy", type=float, default=0.001)
    parser.add_argument(
        "--refine-accuracy", type=float, default=0.0002,
        help="target relative integration error; use a negative value to skip refinement",
    )
    parser.add_argument("--refine-max-channels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260919)
    parser.add_argument("--run-prefix", default="validate")
    parser.add_argument("--validation-summary", type=Path,
                        default=Path("results/madgraph_tth_validation/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/madgraph_tth_madevent_validation"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refine_accuracy is not None and args.refine_accuracy < 0:
        args.refine_accuracy = None
    args.output.mkdir(parents=True, exist_ok=True)
    mg5 = resolve_executable(args.mg5)
    generate_process(mg5, args.process_dir)
    install_param_card(args.param_card, args.process_dir)
    rows = [run_energy(args.process_dir.resolve(), energy, args) for energy in args.energies]
    comparison = attach_existing_results(rows, args.validation_summary)
    comparison.to_csv(args.output / "comparison.csv", index=False)
    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    config["process"] = PROCESS
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("\nMADEVENT COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
