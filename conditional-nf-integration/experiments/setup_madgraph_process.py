#!/usr/bin/env python3
"""Generate a standalone MadGraph library from a process JSON specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from madgraph_process_integrand import load_process_spec
from setup_madgraph_tth_benchmark import compile_libraries, inspect_library, resolve_executable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mg5", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    spec = load_process_spec(args.config)
    output = (args.output or Path(spec["standalone"])).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    mg5 = resolve_executable(args.mg5)
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        "set automatic_html_opening False\n"
        f"import model {spec.get('model', 'sm')}\n"
        f"generate {spec['process']}\n"
        f"output standalone {output} --prefix=int\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as handle:
        handle.write(commands)
        command_card = Path(handle.name)
    try:
        # Run MG5 from a throwaway directory.  Its ALOHA/PLY code generator
        # writes a generated LALR parser table named `py.py` into the *current
        # working directory*, along with `nsqso_born.inc` and `param.log`.
        # Because Python puts the CWD first on sys.path, a stray `py.py` then
        # shadows the `py` shim that pytest ships and every `pytest` run from
        # that directory dies with
        #     AttributeError: module 'py' has no attribute 'path'
        # The output location is absolute in the command card, so the working
        # directory is free to be somewhere disposable.
        # On Windows the MG5 entry point is a Python script carrying a
        # shebang; CreateProcess cannot execute it directly (WinError 193),
        # so run it through this interpreter when that is what we were given.
        command = [mg5, str(command_card)]
        entry = Path(mg5)
        if entry.is_file() and entry.read_bytes()[:2] == b"#!":
            command = [sys.executable, str(entry), str(command_card)]
        with tempfile.TemporaryDirectory(prefix="mg5_run_") as run_dir:
            subprocess.run(command, check=True, cwd=run_dir)
    finally:
        command_card.unlink(missing_ok=True)

    metadata = {**spec, "config": str(args.config.resolve())}
    if not args.no_compile:
        libraries = compile_libraries(output)
        metadata["libraries"] = [str(item.resolve()) for item in libraries]
        metadata["madgraph_api"] = [inspect_library(output, item) for item in libraries]
    (output / "conditional_benchmark.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {spec['name']} in {output}")


if __name__ == "__main__":
    main()
