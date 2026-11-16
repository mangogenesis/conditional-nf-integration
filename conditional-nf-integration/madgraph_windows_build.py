"""Build MadGraph's `matrix2py` Python wrapper on Windows.

Why this module exists
----------------------
`experiments/setup_madgraph_process.py` compiles the generated standalone by
running MadGraph's own makefiles (`make allmatrix2py.so`, falling back to
`make matrix2py.so` per process directory).  Those makefiles are POSIX-shaped:
they set ``SHELL = /bin/bash``, call ``uname``, decide the shared-library
suffix from it, and link with ``-Wl,-rpath``.  On Windows they produce nothing
usable, and the loader is left with no importable wrapper at all.

This module replaces only that final step.  Everything before it — the MG5
generation, and the ``lib/libdhelas.a`` / ``lib/libmodel.a`` archives that MG5's
own make builds for ``Source/`` during generation — is unchanged.

What it does
------------
For each ``SubProcesses/P*/`` directory containing ``f2py_matrix_wrapper.f``:

1. make sure ``coupl.inc`` sits next to ``matrix.f`` (MG5 tries to *symlink* it
   there, which silently fails on Windows without Developer Mode);
2. run f2py to compile ``f2py_matrix_wrapper.f`` + ``matrix.f`` and link them
   against the two archives.

The result is ``SubProcesses/P*/matrix2py.cp310-win_amd64.pyd``.

Two details that are not optional
---------------------------------
* ``--compiler=mingw32 --fcompiler=gnu95``
      numpy's f2py otherwise assumes MSVC.  With these flags it uses the MinGW
      toolchain *and* generates the ``libpython310.a`` import library it needs
      from ``python310.dll``, which is what lets a MinGW-built extension be
      imported by conda's MSVC-built CPython.
* ``LDFLAGS=-static``
      Without it the extension links against ``libgfortran-5.dll``,
      ``libgcc_s_seh-1.dll``, ``libquadmath-0.dll`` and ``libwinpthread-1.dll``
      from the MSYS2 tree.  CPython 3.8+ does not search ``PATH`` for the
      dependencies of extension modules, so the import then fails with
      "DLL load failed while importing matrix2py".  Linking statically leaves
      the module depending only on ``KERNEL32.dll``, ``msvcrt.dll`` and
      ``python310.dll`` — all of which are always resolvable.

Requires the MSYS2 MinGW-w64 toolchain (``gcc``/``gfortran``/``ar``) on PATH.
Set ``NFI_MINGW_BIN`` to override where it is looked for.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

# Where the MinGW-w64 toolchain usually lives on this machine.
_MINGW_CANDIDATES = (
    r"C:\msys64\mingw64\bin",
    r"C:\msys64\ucrt64\bin",
    r"C:\Strawberry\c\bin",
)

# Extra compile flags.  MG5 builds its Fortran with -ffixed-line-length-132
# (see Source/make_opts); the generated MadGraph sources exceed 72 columns and
# do not compile without it.  -w silences the very noisy legacy-Fortran output.
_F77FLAGS = "-ffixed-line-length-132 -w"


def find_mingw_bin() -> Path:
    """Locate a directory holding gfortran/gcc, or raise with what to install."""
    override = os.environ.get("NFI_MINGW_BIN")
    candidates = ([override] if override else []) + list(_MINGW_CANDIDATES)
    for candidate in candidates:
        if candidate and (Path(candidate) / "gfortran.exe").is_file():
            return Path(candidate)
    found = shutil.which("gfortran")
    if found:
        return Path(found).parent
    raise FileNotFoundError(
        "Could not find gfortran.  Install the MSYS2 MinGW-w64 Fortran compiler\n"
        "    pacman -S mingw-w64-x86_64-gcc-fortran\n"
        "or point NFI_MINGW_BIN at the directory holding gfortran.exe."
    )


def _build_environment(mingw_bin: Path) -> dict:
    env = dict(os.environ)
    env["PATH"] = str(mingw_bin) + os.pathsep + env.get("PATH", "")
    # Static-link the Fortran runtime; see the module docstring.
    env["LDFLAGS"] = "-static"
    return env


def _ensure_source_archives(standalone: Path, env: dict) -> None:
    """Build lib/libdhelas.a and lib/libmodel.a if MG5's own make did not.

    MG5 normally builds these while writing the standalone output, so this is
    only a safety net for a generation that was interrupted part-way.
    """
    archives = (standalone / "lib" / "libdhelas.a", standalone / "lib" / "libmodel.a")
    if all(item.is_file() for item in archives):
        return
    make = shutil.which("make")
    if not make:
        raise FileNotFoundError(
            "lib/libdhelas.a or lib/libmodel.a is missing and `make` is not on "
            "PATH, so they cannot be rebuilt.  Install MSYS2 make, or regenerate "
            "the standalone."
        )
    for source_dir in ("DHELAS", "MODEL"):
        subprocess.run(
            [make], cwd=standalone / "Source" / source_dir, env=env,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    missing = [item for item in archives if not item.is_file()]
    if missing:
        raise RuntimeError(f"Failed to build {[str(item) for item in missing]}")


def _prepare_process_dir(process_dir: Path, standalone: Path) -> None:
    """Copy coupl.inc next to matrix.f (MG5's symlink silently fails here).

    A failed symlink still *exists* as a tiny text file holding the target
    path, so an existence test is not enough: anything under 100 bytes is
    treated as a broken link and replaced with the real include.
    """
    include = process_dir / "coupl.inc"
    source = standalone / "Source" / "MODEL" / "coupl.inc"
    if not include.is_file() or include.stat().st_size < 100:
        if not source.is_file():
            raise FileNotFoundError(f"Neither {include} nor {source} exists")
        shutil.copyfile(source, include)


def _run_f2py(process_dir: Path, standalone: Path, env: dict) -> Path:
    library_dir = (standalone / "lib").resolve()
    before = set(process_dir.glob("matrix2py*.pyd"))
    command = [
        sys.executable, "-m", "numpy.f2py", "-c",
        "f2py_matrix_wrapper.f",
        "matrix.f",
        f"-L{library_dir}",
        "-ldhelas",
        "-lmodel",
        "-m", "matrix2py",
        "--compiler=mingw32",
        "--fcompiler=gnu95",
        f"--f77flags={_F77FLAGS}",
    ]
    result = subprocess.run(
        command, cwd=process_dir, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"f2py failed to build matrix2py in {process_dir}:\n{result.stdout}"
        )
    built = sorted(set(process_dir.glob("matrix2py*.pyd")) - before)
    if not built:
        # f2py overwrote an existing module rather than creating a new one.
        built = sorted(process_dir.glob("matrix2py*.pyd"))
    if not built:
        raise RuntimeError(
            f"f2py reported success but produced no matrix2py*.pyd in {process_dir}:\n"
            f"{result.stdout}"
        )
    return built[0]


def build_matrix2py(standalone: Path) -> list[Path]:
    """Build every per-process matrix2py wrapper below one standalone tree."""
    if sys.platform != "win32":
        raise RuntimeError("build_matrix2py is the Windows replacement for MG5's make path")

    standalone = Path(standalone).resolve()
    subprocesses = standalone / "SubProcesses"
    if not subprocesses.is_dir():
        raise FileNotFoundError(f"No SubProcesses directory below {standalone}")

    mingw_bin = find_mingw_bin()
    env = _build_environment(mingw_bin)
    _ensure_source_archives(standalone, env)

    targets = sorted(
        path for path in subprocesses.glob("P*")
        if path.is_dir() and (path / "f2py_matrix_wrapper.f").is_file()
    )
    if not targets:
        raise FileNotFoundError(
            f"No SubProcesses/P*/f2py_matrix_wrapper.f below {standalone}; the "
            "standalone was not generated with the f2py (--prefix=int) output mode."
        )

    built: list[Path] = []
    for process_dir in targets:
        _prepare_process_dir(process_dir, standalone)
        built.append(_run_f2py(process_dir, standalone, env))
    return built


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--standalone", type=Path, required=True,
                        help="root of the generated MadGraph standalone")
    args = parser.parse_args()
    for library in build_matrix2py(args.standalone):
        print(f"Built {library}")


if __name__ == "__main__":
    main()
