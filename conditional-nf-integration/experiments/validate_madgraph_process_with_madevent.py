#!/usr/bin/env python3
"""Independent MadEvent normalization check for a configured partonic process.

This is the low-multiplicity analogue of
``validate_madgraph_tth_with_madevent.py``.  It generates (or reuses) a
MadEvent directory for the same partonic process and diagram filter used by
the NF standalone, installs the standalone parameter card, runs MadEvent's
explicit ``survey``/``refine`` integration at fixed partonic sqrt(s), and
merges the result with the NF/Sobol validation summary.

The beam cards are set to ``lpp1 = lpp2 = 0`` with
``ebeam1 = ebeam2 = sqrt(s_hat)/2``.  This is essential: the NF integrand is a
fixed-initial-state partonic integral, so a proton-beam (PDF-convolved)
MadEvent run would integrate a different observable.
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
import sys
import tempfile
import time

import pandas as pd

from madgraph_process_integrand import load_process_spec


def resolve_executable(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Could not find {value!r}; pass --mg5 /path/to/bin/mg5_aMC")


def generate_process(mg5: str, spec: dict, process_dir: Path) -> None:
    if process_dir.exists():
        if not (process_dir / "bin" / "madevent").is_file():
            raise FileExistsError(f"Existing path is not a MadEvent directory: {process_dir}")
        return
    process_dir.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        "set automatic_html_opening False\n"
        f"import model {spec.get('model', 'sm')}\n"
        f"generate {spec['process']}\n"
        f"output {process_dir.resolve()}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as handle:
        handle.write(commands)
        card = Path(handle.name)
    command = [mg5, str(card)]
    entry = Path(mg5)
    if entry.is_file() and entry.read_bytes()[:2] == b"#!":
        command = [sys.executable, str(entry), str(card)]
    try:
        subprocess.run(command, check=True)
    finally:
        card.unlink(missing_ok=True)


def install_param_card(source: Path, process_dir: Path, alpha_s: float | None = None) -> None:
    """Use the exact parameter card employed by the standalone evaluator.

    For a partonic (lpp=0) MadEvent run the strong coupling comes from the
    parameter card, not from a PDF set.  The NF passes alpha_s directly to
    matrix2py, so the card's SMINPUTS alpha_s(M_Z) must be set to the same
    value for the two calculations to use the same coupling.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Standalone parameter card not found: {source}")
    destination = process_dir / "Cards" / "param_card.dat"
    shutil.copy2(source, destination)
    if alpha_s is None:
        return
    lines = destination.read_text(encoding="utf-8").splitlines()
    in_sminputs = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("block sminputs"):
            in_sminputs = True
            continue
        if in_sminputs and stripped.lower().startswith("block"):
            break
        if in_sminputs and stripped:
            fields = stripped.split()
            if len(fields) >= 2 and fields[0] == "3":
                comment = ""
                if "#" in line:
                    comment = " #" + line.split("#", 1)[1]
                lines[index] = f"  3 {alpha_s:.12g}{comment}"
                break
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_card_settings(energy: float, cuts: dict, args) -> dict[str, str | int | float]:
    beam = energy / 2.0
    # The NF matrix element is evaluated at the config's fixed alpha_s.  The
    # same value is written into SMINPUTS of the MadEvent parameter card, and
    # fixed_ren_scale=True at scale=91.188 makes the partonic run use it.  The
    # factorization scale is immaterial with lpp1 = lpp2 = 0.
    return {
        "lpp1": 0,
        "lpp2": 0,
        "ebeam1": f"{beam:.12g}",
        "ebeam2": f"{beam:.12g}",
        "bwcutoff": f"{args.width_window:.12g}",
        "fixed_ren_scale": "True",
        "fixed_fac_scale1": "True",
        "fixed_fac_scale2": "True",
        "scale": f"{args.ren_scale:.12g}",
        "dsqrt_q2fact1": f"{args.ren_scale:.12g}",
        "dsqrt_q2fact2": f"{args.ren_scale:.12g}",
        "ptj": f"{float(cuts.get('pt_j_min', 0.0)):.12g}",
        "ptjmax": -1.0,
        "etaj": f"{float(cuts.get('eta_j_max', -1.0)):.12g}",
        "drjj": f"{float(cuts.get('dr_jj_min', 0.0)):.12g}",
        "mmjj": f"{float(cuts.get('m_jj_min', 0.0)):.12g}",
        "nevents": args.nevents,
        "req_acc": f"{args.required_accuracy:.12g}",
        "iseed": args.seed + int(round(energy)),
        "use_syst": "False",
        "ickkw": 0,
    }


def update_run_card(process_dir: Path, settings: dict[str, str | int | float]) -> None:
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
    # MG5 3.6.7's LO run card does not carry every historical field.  These
    # are either irrelevant at lpp1 = lpp2 = 0 or controlled by the
    # survey/refine command line.
    optional = {
        "dsqrt_q2fact1",
        "dsqrt_q2fact2",
        "use_syst",
        "ickkw",
        "req_acc",
        "cut_decays",
        "fixed_fac_scale1",
        "fixed_fac_scale2",
        "fixed_fac_scale",
        "drjj",
        "mmjj",
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
    commands = (
        f"survey {run_name} --points={args.survey_points} "
        f"--iterations={args.survey_iterations} "
        f"--accuracy={args.survey_accuracy:.12g}\n"
    )
    if args.refine_accuracy is not None and args.refine_accuracy >= 0:
        commands += f"refine {args.refine_accuracy:.12g} {args.refine_max_channels}\n"
    return commands


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


def madevent_command(executable: Path, command_path: Path) -> list[str]:
    """Run the MadEvent entry point through its interpreter on Windows.

    ``bin/madevent`` is a Python script with a shebang; CreateProcess cannot
    execute it directly (WinError 193).  On POSIX the executable runs as is.
    """
    try:
        first = executable.read_bytes()[:2]
    except OSError:
        first = b""
    if first == b"#!":
        return [sys.executable, str(executable), str(command_path.resolve())]
    return [str(executable), str(command_path.resolve())]


def run_energy(process_dir: Path, energy: float, args, cuts: dict) -> dict:
    run_name = f"{args.run_prefix}_{int(round(energy))}"
    update_run_card(process_dir, run_card_settings(energy, cuts, args))
    commands = madevent_commands(run_name, args)
    command_path = args.output / f"{run_name}_commands.txt"
    command_path.write_text(commands, encoding="utf-8")
    executable = process_dir / "bin" / "madevent"
    results_path = process_dir / "SubProcesses" / "results.dat"
    previous_results_mtime = results_path.stat().st_mtime_ns if results_path.is_file() else None
    started = time.perf_counter()
    completed = subprocess.run(
        madevent_command(executable, command_path), cwd=process_dir,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    log_path = args.output / f"{run_name}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MadEvent failed at {energy:g} GeV; see {log_path}")
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
    print(f"MadEvent {energy:g} GeV: {cross_pb:.8g} +/- {error_pb:.3g} pb", flush=True)
    return {
        "energy_gev": energy,
        "madevent_pb": cross_pb,
        "madevent_error_pb": error_pb,
        "run_name": run_name,
        "seconds": elapsed,
        "log": str(log_path.resolve()),
    }


def attach_validation(summary_path: Path, rows: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if not summary_path.is_file():
        print(f"Validation summary not found: {summary_path}", flush=True)
        return result
    previous = pd.read_csv(summary_path)
    methods = [item for item in ("conditional_nf", "sobol_qmc", "uniform_mc")
               if item in set(previous["method"])]
    for method in methods:
        subset = previous[previous["method"].eq(method)].set_index("energy_gev")
        result[f"{method}_pb"] = result["energy_gev"].map(subset["estimate_pb"])
        result[f"{method}_stderr_pb"] = result["energy_gev"].map(subset["stderr_pb"])
        denom = (result["madevent_error_pb"] ** 2 + result[f"{method}_stderr_pb"] ** 2) ** 0.5
        result[f"madevent_vs_{method}_pull"] = (
            (result["madevent_pb"] - result[f"{method}_pb"]) / denom
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--channel",
        default=None,
        help="Channel name from the config's `channels` list (required for multi-subprocess configs).",
    )
    parser.add_argument("--alpha-s", type=float, default=None)
    parser.add_argument("--mg5", default=os.environ.get("MG5AMC", "mg5_aMC"))
    parser.add_argument("--process-dir", type=Path, default=None)
    parser.add_argument("--param-card", type=Path, default=None)
    parser.add_argument("--energies", type=float, nargs="+", default=None)
    parser.add_argument(
        "--allow-outside-condition",
        action="store_true",
        help="Allow energies outside the config's condition range (used for the interpolation scan).",
    )
    parser.add_argument("--width-window", type=float, default=15.0)
    parser.add_argument("--ren-scale", type=float, default=91.188)
    parser.add_argument("--required-accuracy", type=float, default=0.002)
    parser.add_argument("--nevents", type=int, default=10000)
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
    parser.add_argument("--validation-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config_data.get("channels") and args.channel is None:
        raise ValueError(
            "This config defines multiple channels; pass --channel with one of: "
            + ", ".join(item["name"] for item in config_data["channels"])
        )
    spec = load_process_spec(args.config, channel=args.channel)
    alpha_s = (
        args.alpha_s
        if args.alpha_s is not None
        else float(spec.get("alpha_s", 0.118))
    )
    channel = spec.get("channel", spec["name"])
    default_dir = Path(spec["standalone"]).parent / f"zj_hadronic_madevent_{channel}"
    process_dir = (args.process_dir or default_dir).resolve()
    param_card = (args.param_card or Path(spec["standalone"]) / "Cards" / "param_card.dat").resolve()
    if not param_card.is_file():
        raise FileNotFoundError(f"Parameter card not found: {param_card}")
    energies = args.energies or [float(item) for item in spec["condition"]["held_out"]]
    minimum = float(spec["condition"]["minimum"])
    maximum = float(spec["condition"]["maximum"])
    if not args.allow_outside_condition and any(
        item < minimum or item > maximum for item in energies
    ):
        raise ValueError("Every validation energy must lie inside the condition range")
    cuts = spec["phase_space"].get("cuts", {})

    args.output.mkdir(parents=True, exist_ok=True)
    mg5 = resolve_executable(args.mg5)
    generate_process(mg5, spec, process_dir)
    if os.name == "nt":
        # MG5's Windows output needs the shared-file copies, make_opts temp
        # fix and single-core setting before MadEvent can compile a process
        # directory (otherwise run_config.inc is missing).
        prepare = (
            Path(__file__).resolve().parents[2]
            / "madgraph_baselines" / "prepare_run.py"
        )
        if prepare.is_file():
            subprocess.run(
                [sys.executable, str(prepare), str(process_dir)], check=True
            )
    install_param_card(param_card, process_dir, alpha_s)

    rows = [run_energy(process_dir, energy, args, cuts) for energy in energies]
    comparison = attach_validation(
        args.validation_summary or args.output.parent
        / f"{spec['name']}_validation_balanced_gpu_sobol" / "summary.csv",
        rows,
    )
    comparison.to_csv(args.output / "comparison.csv", index=False)
    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    config["process"] = spec["process"]
    config["resolved_energies"] = energies
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("\nMADEVENT COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
