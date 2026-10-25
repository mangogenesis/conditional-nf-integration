#!/usr/bin/env python3
"""Build a MadGraph standalone matrix-element library for the ttH benchmark.

This is a setup/validation utility.  The resulting shared library evaluates the
unnormalized integrand; it does not provide target samples to the NF trainer.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROCESSES = {
    "smoke": {
        "process": "g g > t t~ h @1",
        "description": "LO gg -> tt~H (2 -> 3 interface and normalization check)",
        "final_particles": 3,
        "phase_space_dimension": 5,
    },
    "full": {
        "process": (
            "g g > t t~ h, "
            "(t > b w+, w+ > e+ ve), "
            "(t~ > b~ w-, w- > mu- vm~), "
            "(h > b b~) @1"
        ),
        "description": "LO fully decayed gg -> tt~H benchmark",
        "final_particles": 8,
        "phase_space_dimension": 20,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and compile a standalone MadGraph ttH matrix element."
    )
    parser.add_argument(
        "--mg5",
        default=os.environ.get("MG5AMC", "mg5_aMC"),
        help="Path to MG5_aMC/bin/mg5_aMC (default: $MG5AMC or mg5_aMC).",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(PROCESSES),
        default="full",
        help="Use 'smoke' to validate the interface before building the full target.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Standalone output directory (must not already exist).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Generate the standalone code but do not compile allmatrix2py.so.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Compile/inspect an existing generated directory without regenerating it.",
    )
    return parser.parse_args()


def resolve_executable(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find {value!r}. Pass --mg5 /path/to/MG5_aMC/bin/mg5_aMC."
    )


def write_command_card(path: Path, output: Path, process: str) -> None:
    # An absolute output path makes the result independent of the launch folder.
    commands = (
        "set automatic_html_opening False\n"
        "import model sm\n"
        f"generate {process}\n"
        f"output standalone {output.resolve()} --prefix=int\n"
    )
    path.write_text(commands, encoding="utf-8")


def compile_libraries(output: Path) -> list[Path]:
    subprocess_root = output / "SubProcesses"
    combined = subprocess.run(
        ["make", "allmatrix2py.so"],
        cwd=subprocess_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    candidates = sorted((output / "SubProcesses").glob("allmatrix2py*.so"))
    if candidates:
        return candidates

    print("Combined allmatrix2py target is unavailable; trying per-process wrappers.")
    process_dirs = sorted(
        path
        for path in subprocess_root.glob("P*")
        if path.is_dir()
        and ((path / "makefile").exists() or (path / "Makefile").exists())
    )
    failures = []
    for process_dir in process_dirs:
        result = subprocess.run(
            ["make", "matrix2py.so"],
            cwd=process_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            failures.append(f"{process_dir.name}:\n{result.stdout}")

    candidates = sorted(subprocess_root.glob("P*/matrix2py*.so"))
    if not candidates:
        details = "\n".join(failures) or combined.stdout
        raise RuntimeError(
            "MadGraph generated the process, but no Python matrix-element wrapper "
            f"could be compiled. Build output:\n{details}"
        )
    return candidates


def inspect_library(output: Path, library: Path) -> dict:
    module_dir = library.parent
    sys.path.insert(0, str(module_dir.resolve()))
    try:
        module_name = "allmatrix2py" if library.name.startswith("allmatrix2py") else "matrix2py"
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        param_card = output / "Cards" / "param_card.dat"
        initializer = getattr(module, "initialise", None) or getattr(module, "setpara", None)
        if initializer is not None:
            try:
                initializer(str(param_card))
            except TypeError:
                initializer(str(param_card).encode())

        diagnostics = {"module": module_name, "directory": str(module_dir.resolve())}
        for name in ("get_pdg_order", "get_prefix", "get_process_ids"):
            function = getattr(module, name, None)
            if function is not None:
                try:
                    value = function()
                    diagnostics[name] = (
                        value.tolist() if hasattr(value, "tolist") else str(value)
                    )
                except Exception as exc:  # diagnostic only; API varies by MG version
                    diagnostics[name] = f"unavailable: {exc}"
        return diagnostics
    finally:
        sys.path.pop(0)


def main() -> None:
    args = parse_args()
    spec = PROCESSES[args.mode]
    output = (
        args.output
        if args.output is not None
        else Path("generated") / f"gg_tth_{args.mode}_standalone"
    ).expanduser()

    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new --output path so an "
            "existing MadGraph build is not overwritten, or pass --resume."
        )
    if args.resume and not output.is_dir():
        raise FileNotFoundError(f"Cannot resume because the output does not exist: {output}")

    mg5 = None if args.resume else resolve_executable(args.mg5)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("MADGRAPH CONDITIONAL SCATTERING BENCHMARK SETUP")
    print(f"mode                    = {args.mode}")
    print(f"process                 = {spec['process']}")
    print(f"physical phase-space dim= {spec['phase_space_dimension']}")
    print("proposed condition      = partonic sqrt(s_hat)")
    print(f"output                  = {output.resolve()}")

    if not args.resume:
        with tempfile.TemporaryDirectory(prefix="mg5_tth_") as temporary:
            card = Path(temporary) / "proc_card.dat"
            write_command_card(card, output, spec["process"])
            subprocess.run([mg5, str(card)], check=True)

    metadata = {
        "mode": args.mode,
        **spec,
        "condition": "sqrt_s_hat_GeV",
        "training_access": "matrix element / integrand evaluations only",
        "target_samples_used_for_training": False,
    }

    if not args.no_compile:
        libraries = compile_libraries(output)
        metadata["libraries"] = [str(path.resolve()) for path in libraries]
        metadata["madgraph_api"] = [inspect_library(output, path) for path in libraries]

    metadata_path = output / "conditional_benchmark.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("\nSETUP COMPLETE")
    if not args.no_compile:
        print(f"matrix-element libraries= {metadata['libraries']}")
        print(f"MadGraph API diagnostics= {metadata['madgraph_api']}")
    print(f"metadata                = {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
