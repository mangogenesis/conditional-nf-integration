#!/usr/bin/env python3
"""Apply the Windows fixes needed before launching a generated MG5 directory."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def copy_shared_files(me_dir: Path) -> None:
    subprocesses = me_dir / "SubProcesses"
    pdirs = sorted(p for p in subprocesses.glob("P*") if p.is_dir())
    if not pdirs:
        raise SystemExit(f"No SubProcesses/P* directories below {subprocesses}")
    shared = [p for p in subprocesses.iterdir() if p.is_file()]
    for pdir in pdirs:
        for source in shared:
            target = pdir / source.name
            if not target.exists():
                shutil.copy2(source, target)
    model_coupl = me_dir / "Source" / "MODEL" / "coupl.inc"
    if model_coupl.exists():
        for target in (me_dir / "Source" / "coupl.inc", subprocesses / "coupl.inc"):
            if not target.exists():
                shutil.copy2(model_coupl, target)


def patch_make_opts(me_dir: Path) -> None:
    make_opts = me_dir / "Source" / "make_opts"
    text = make_opts.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "export TMP :=" in text:
        return
    temp_dir = Path(tempfile.gettempdir()).as_posix()
    header = (
        "# Windows gfortran needs a writable temp directory.  The MSYS2 shell\n"
        "# used by make clears TMP/TEMP/TMPDIR, so set them explicitly.\n"
        f"export TMP := {temp_dir}\n"
        "export TEMP := $(TMP)\n"
        "export TMPDIR := $(TMP)\n\n"
    )
    if not text.endswith("\n"):
        text += "\n"
    make_opts.write_text(header + text, encoding="utf-8")


def patch_makefiles(me_dir: Path) -> None:
    for makefile in (me_dir / "SubProcesses").glob("P*/makefile"):
        text = makefile.read_text(encoding="utf-8", errors="replace")
        fixed = text.replace("-I../../Source/ ", "-I../../Source ")
        if fixed != text:
            makefile.write_text(fixed, encoding="utf-8")


def seed_optimized_matrices(me_dir: Path) -> None:
    for pdir in (me_dir / "SubProcesses").glob("P*"):
        if not pdir.is_dir():
            continue
        for original in pdir.glob("matrix*_orig.f"):
            optimized = original.with_name(original.name.replace("_orig.f", "_optim.f"))
            if not optimized.exists():
                shutil.copy2(original, optimized)


def force_single_core(me_dir: Path) -> None:
    config = me_dir / "Cards" / "me5_configuration.txt"
    text = config.read_text(encoding="utf-8", errors="replace")
    if any(line.strip().startswith("nb_core") for line in text.splitlines()):
        return
    config.write_text(text.rstrip() + "\nnb_core = 1\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("me_dir", type=Path)
    args = parser.parse_args()
    me_dir = args.me_dir.resolve()
    copy_shared_files(me_dir)
    patch_make_opts(me_dir)
    patch_makefiles(me_dir)
    seed_optimized_matrices(me_dir)
    force_single_core(me_dir)
    print(f"Prepared {me_dir}")


if __name__ == "__main__":
    main()
