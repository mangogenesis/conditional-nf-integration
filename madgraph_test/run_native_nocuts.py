#!/usr/bin/env python3
"""Run the native p p > z j MadEvent integration with no cuts at all.

The only change relative to the earlier native baseline is ``ptj = 0`` (the
baseline used ``ptj = 20``).  The eta cut is already absent (``etaj = -1``).

Usage:
    python run_native_nocuts.py --points 2000 --iterations 2
    python run_native_nocuts.py --points 20000 --iterations 8
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ME_DIR = HERE / "native_zj_nocuts"
RUN_CARD = ME_DIR / "Cards" / "run_card.dat"
PY = Path(sys.executable)


def set_value(path: Path, name: str, value: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(rf"^(\s*)(\S+)(\s*=\s*{re.escape(name)}\b)", re.MULTILINE)
    text, count = pattern.subn(
        lambda match: f"{match.group(1)}{value}{match.group(3)}", text, count=1
    )
    if count != 1:
        raise RuntimeError(f"{path}: could not set {name}")
    path.write_text(text, encoding="utf-8")


def set_card(run_card: Path, ptj: float, etaj: float, dsqrt_shat: float,
             pdlabel: str, lhaid: int, dsqrt_shatmax: float = -1.0) -> None:
    set_value(run_card, "lpp1", "1")
    set_value(run_card, "lpp2", "1")
    set_value(run_card, "ebeam1", "6500.0")
    set_value(run_card, "ebeam2", "6500.0")
    set_value(run_card, "pdlabel1", pdlabel)
    set_value(run_card, "pdlabel2", pdlabel)
    set_value(run_card, "lhaid", str(lhaid))
    set_value(run_card, "fixed_ren_scale", "True")
    set_value(run_card, "fixed_fac_scale1", "True")
    set_value(run_card, "fixed_fac_scale2", "True")
    set_value(run_card, "scale", "91.188")
    set_value(run_card, "dsqrt_q2fact1", "91.188")
    set_value(run_card, "dsqrt_q2fact2", "91.188")
    set_value(run_card, "dsqrt_shat", f"{dsqrt_shat:.6g}")
    set_value(run_card, "dsqrt_shatmax", f"{dsqrt_shatmax:.6g}")
    set_value(run_card, "ptj", f"{ptj:.1f}")
    set_value(run_card, "etaj", f"{etaj:.1f}")
    set_value(run_card, "maxjetflavor", "4")
    set_value(run_card, "use_syst", "False")


def read_results(path: Path) -> tuple[float, float]:
    line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    fields = line.split()
    return float(fields[0].replace("D", "E")), float(fields[1].replace("D", "E"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--me-dir", type=Path, default=ME_DIR)
    parser.add_argument("--pdlabel", default="nn23lo1")
    parser.add_argument("--lhaid", type=int, default=247000)
    parser.add_argument("--lhapdf-data-path", default="",
                        help="LHAPDF_DATA_PATH; empty uses the local install")
    parser.add_argument("--no-make-opts", action="store_true",
                        help="do not patch Source/make_opts")
    parser.add_argument("--ptj", type=float, default=0.0)
    parser.add_argument("--etaj", type=float, default=-1.0)
    parser.add_argument("--dsqrt-shat", type=float, default=0.0,
                        help="minimum sqrt(s-hat) in GeV; 0 means no cut")
    parser.add_argument("--dsqrt-shatmax", type=float, default=-1.0,
                        help="maximum sqrt(s-hat) in GeV; -1 means no cut")
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--accuracy", type=float, default=0.01)
    parser.add_argument("--run-name", default="nocuts")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    me_dir = args.me_dir.resolve()
    run_card = me_dir / "Cards" / "run_card.dat"
    # If using external LHAPDF, tell the build which lhapdf-config to use.
    if not args.no_make_opts:
        make_opts = me_dir / "Source" / "make_opts"
        text = make_opts.read_text(encoding="utf-8", errors="replace")
        lhapdf_config = (
            os.environ.get("LHAPDF_CONFIG", "lhapdf-config")
            if args.pdlabel == "lhapdf" else ""
        )
        text, count = re.subn(r"^lhapdf=.*$", f"lhapdf={lhapdf_config}", text,
                              count=1, flags=re.MULTILINE)
        if count == 0:
            text = f"lhapdf={lhapdf_config}\n" + text
        make_opts.write_text(text, encoding="utf-8")

    set_card(run_card, args.ptj, args.etaj, args.dsqrt_shat,
             args.pdlabel, args.lhaid, args.dsqrt_shatmax)
    command_file = me_dir / f"{args.run_name}_commands.txt"
    command_file.write_text(
        f"survey {args.run_name} --points={args.points} "
        f"--iterations={args.iterations} --accuracy={args.accuracy}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    tool_dirs = [str(HERE / "toolbin")]
    for key in ("MSYS_BIN", "LHAPDF_BIN"):
        if os.environ.get(key):
            tool_dirs.append(os.environ[key])
    env["PATH"] = os.pathsep.join(tool_dirs + [env.get("PATH", "")])
    env["TMP"] = os.environ.get("TMP") or tempfile.gettempdir()
    env["TEMP"] = env["TMP"]
    env["TMPDIR"] = env["TMP"]
    if args.pdlabel == "lhapdf":
        env["LHAPDF_DATA_PATH"] = (
            args.lhapdf_data_path
            or os.environ.get("LHAPDF_DATA_PATH", "")
        )
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(PY), "bin/madevent", command_file.name],
            cwd=me_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        log = HERE / "logs" / f"{args.run_name}_timeout.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(exc.output or "", encoding="utf-8", errors="replace")
        print(f"TIMEOUT after {args.timeout:.0f}s; see {log}")
        raise SystemExit(2)
    elapsed = time.perf_counter() - started
    log = HERE / "logs" / f"{args.run_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout, encoding="utf-8", errors="replace")
    print(proc.stdout[-4000:])
    print(f"exit={proc.returncode} elapsed={elapsed:.1f}s log={log}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    total, error = read_results(me_dir / "SubProcesses" / "results.dat")
    print(f"NATIVE ptj={args.ptj} etaj={args.etaj} dsqrt_shat={args.dsqrt_shat} "
          f"pdlabel={args.pdlabel}: {total:.6g} +/- {error:.6g} pb")
    for pdir in sorted((me_dir / "SubProcesses").glob("P*")):
        files = sorted(pdir.glob("G*/results.dat"))
        if not files:
            continue
        s = 0.0
        e2 = 0.0
        for result in files:
            si, ei = read_results(result)
            s += si
            e2 += ei * ei
        print(f"  {pdir.name}: {s:.6g} +/- {e2 ** 0.5:.6g} pb")


if __name__ == "__main__":
    main()
