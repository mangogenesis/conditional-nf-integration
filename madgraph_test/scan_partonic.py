#!/usr/bin/env python3
"""Run a continuous partonic MadGraph scan of Z+jet with no eta cut.

The MadEvent directory ``partonic_zj`` contains six explicit partonic
processes:

    P1 g u  > z u       P2 g d  > z d
    P3 g u~ > z u~      P4 g d~ > z d~
    P5 u u~ > z g       P6 d d~ > z g

This script sets ``lpp1 = lpp2 = 0``, ``ebeam1 = ebeam2 = sqrt(s-hat)/2``,
``ptj = 20``, ``etaj = -1`` (no eta cut), fixed scale 91.188 GeV, and
alpha_s = 0.130.  It runs one MadEvent survey per energy and records the
per-channel cross sections.

Usage:
    python scan_partonic.py --energies 114 500 1000 2000
    python scan_partonic.py            # 50 linear points 114..2000
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ME_DIR = HERE / "partonic_zj"
RUN_CARD = ME_DIR / "Cards" / "run_card.dat"
PARAM_CARD = ME_DIR / "Cards" / "param_card.dat"
OUT = HERE / "results"
PY = Path(sys.executable)

PROCESSES = {
    "P1_gq_zq": "gu_zu",
    "P2_gq_zq": "gd_zd",
    "P3_gq_zq": "gux_zux",
    "P4_gq_zq": "gdx_zdx",
    "P5_qq_zg": "uux_zg",
    "P6_qq_zg": "ddx_zg",
}


def set_value(path: Path, name: str, value: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(rf"^(\s*)(\S+)(\s*=\s*{re.escape(name)}\b)", re.MULTILINE)
    text, count = pattern.subn(
        lambda match: f"{match.group(1)}{value}{match.group(3)}", text, count=1
    )
    if count != 1:
        raise RuntimeError(f"{path}: could not set {name}")
    path.write_text(text, encoding="utf-8")


def set_energy(energy: float, ptj: float = 20.0) -> None:
    beam = 0.5 * energy
    set_value(RUN_CARD, "lpp1", "0")
    set_value(RUN_CARD, "lpp2", "0")
    set_value(RUN_CARD, "ebeam1", f"{beam:.12g}")
    set_value(RUN_CARD, "ebeam2", f"{beam:.12g}")
    set_value(RUN_CARD, "pdlabel1", "none")
    set_value(RUN_CARD, "pdlabel2", "none")
    set_value(RUN_CARD, "fixed_ren_scale", "True")
    set_value(RUN_CARD, "fixed_fac_scale1", "True")
    set_value(RUN_CARD, "fixed_fac_scale2", "True")
    set_value(RUN_CARD, "ptj", f"{ptj:.12g}")
    set_value(RUN_CARD, "etaj", "-1.0")
    set_value(RUN_CARD, "maxjetflavor", "4")
    set_value(RUN_CARD, "use_syst", "False")
    # The six processes are partonic; aS comes from the param card because
    # pdlabel = none.  Keep it at the campaign value.
    text = PARAM_CARD.read_text(encoding="utf-8", errors="replace")
    text, count = re.subn(
        r"^(\s*3\s+)[0-9.eE+-]+(\s*#\s*aS)",
        r"\g<1>1.300000e-01\g<2>",
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("could not set alpha_s in param card")
    PARAM_CARD.write_text(text, encoding="utf-8")


def read_results(path: Path) -> tuple[float, float]:
    line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    fields = line.split()
    return float(fields[0].replace("D", "E")), float(fields[1].replace("D", "E"))


def run_energy(
    energy: float,
    points: int,
    iterations: int,
    accuracy: float,
    ptj: float = 20.0,
    run_prefix: str = "scan",
) -> list[dict]:
    set_energy(energy, ptj)
    run_name = f"{run_prefix}_{energy:.3f}".replace(".", "p")
    (ME_DIR / "scan_commands.txt").write_text(
        f"survey {run_name} --points={points} --iterations={iterations} --accuracy={accuracy}\n",
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
    started = __import__("time").perf_counter()
    proc = subprocess.run(
        [str(PY), "bin/madevent", "scan_commands.txt"],
        cwd=ME_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    )
    elapsed = __import__("time").perf_counter() - started
    log = HERE / "logs" / f"scan_{energy:.3f}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"madevent failed at {energy} GeV; see {log}")
    rows = []
    total = 0.0
    total_err2 = 0.0
    for pdir, channel in PROCESSES.items():
        sigma = 0.0
        err2 = 0.0
        for result in sorted((ME_DIR / "SubProcesses" / pdir).glob("G*/results.dat")):
            s, e = read_results(result)
            sigma += s
            err2 += e * e
        err = err2 ** 0.5
        rows.append({
            "energy_gev": energy,
            "channel": channel,
            "sigma_pb": sigma,
            "error_pb": err,
            "seconds": elapsed,
        })
        total += sigma
        total_err2 += err * err
    rows.append({
        "energy_gev": energy,
        "channel": "sum",
        "sigma_pb": total,
        "error_pb": total_err2 ** 0.5,
        "seconds": elapsed,
    })
    print(
        f"{energy:9.3f} GeV  sum={total:10.4f} +/- {total_err2**0.5:.4f} pb  "
        f"({elapsed:.1f} s)",
        flush=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energies", type=float, nargs="+", default=None)
    parser.add_argument("--energy-file", type=Path, default=None,
                        help="whitespace/comma-separated energy list (GeV)")
    parser.add_argument("--points", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--accuracy", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=OUT / "scan_noeta.csv")
    parser.add_argument("--ptj", type=float, default=20.0)
    parser.add_argument("--run-prefix", default="scan")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.energy_file is not None:
        energies = [
            float(token)
            for token in args.energy_file.read_text(
                encoding="utf-8", errors="replace"
            ).replace(",", " ").split()
        ]
    elif args.energies is not None:
        energies = args.energies
    else:
        # 50 linear points, the same point density as each previous 50-point
        # interpolation scan, now continuous over the full range.
        energies = list(__import__("numpy").linspace(114.0, 2000.0, 50))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    done = set()
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        done = {float(row["energy_gev"]) for row in all_rows}
        print(f"resume: {len(done)} energies already present", flush=True)
    for energy in energies:
        if float(energy) in done:
            continue
        all_rows.extend(
            run_energy(
                float(energy), args.points, args.iterations, args.accuracy,
                ptj=args.ptj, run_prefix=args.run_prefix,
            )
        )
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
