#!/usr/bin/env python3
"""Parse native LHE events into a flat per-event table for acceptance checks."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The LHAPDF run is the clean no-eta sample: it has the correct CM fraction
# (0.9897) and weight sum (11,913.95 pb).  Some later native_zj_nocuts event
# files came from other run-card settings and are not a no-eta sample.
ME = HERE.parent / "native_zj_lhapdf" / "SubProcesses"
EBEAM = 6500.0
OUT = HERE / "events_flat.csv"


def rap(pz: float, e: float) -> float:
    if abs(e - pz) < 1e-12:
        return math.copysign(50.0, pz)
    return 0.5 * math.log((e + pz) / (e - pz))


def eta(px: float, py: float, pz: float) -> float:
    p = math.sqrt(px * px + py * py + pz * pz)
    if abs(p - pz) < 1e-12:
        return math.copysign(50.0, pz)
    return 0.5 * math.log((p + pz) / (p - pz))


def channel(p1: int, p2: int) -> str:
    if p1 == 21 and p2 == 21:
        return "gg"
    if p1 == 21 or p2 == 21:
        q = p2 if p1 == 21 else p1
        if q in (2, 4):
            return "gu_zu"
        if q in (-2, -4):
            return "gux_zux"
        if q in (1, 3):
            return "gd_zd"
        if q in (-1, -3):
            return "gdx_zdx"
        return f"other_g_{q}"
    a, b = p1, p2
    if a * b < 0:
        flavor = sorted((abs(a), abs(b)))
        if flavor[0] in (2, 4) and flavor[1] in (2, 4):
            return "uux_zg"
        if flavor[0] in (1, 3) and flavor[1] in (1, 3):
            return "ddx_zg"
    return f"other_{p1}_{p2}"

def main() -> None:
    files = sorted(ME.glob("P*/G*/events.lhe"))
    print("files:", len(files))
    rows = []
    for path in files:
        label = f"{path.parent.parent.name}/{path.parent.name}"
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() != "<event>":
                i += 1
                continue
            i += 1
            header = lines[i].split()
            npart = int(header[0])
            weight = float(header[2])
            particles = []
            for j in range(npart):
                f = lines[i + 1 + j].split()
                particles.append({
                    "pdg": int(f[0]), "status": int(f[1]),
                    "px": float(f[6]), "py": float(f[7]),
                    "pz": float(f[8]), "e": float(f[9]),
                })
            i += npart + 1
            ini = [p for p in particles if p["status"] == -1]
            fin = [p for p in particles if p["status"] == 1]
            if len(ini) != 2:
                continue
            ini.sort(key=lambda p: p["pz"])
            x2 = ini[0]["e"] / EBEAM
            x1 = ini[1]["e"] / EBEAM
            if x1 <= 0 or x2 <= 0:
                continue
            Y = 0.5 * math.log(x1 / x2)
            shat = math.sqrt(x1 * x2) * 2.0 * EBEAM
            jets = [p for p in fin if abs(p["pdg"]) != 23]
            if not jets:
                continue
            jet = max(jets, key=lambda p: math.hypot(p["px"], p["py"]))
            ylab = rap(jet["pz"], jet["e"])
            etalab = eta(jet["px"], jet["py"], jet["pz"])
            # boost jet to the partonic CM
            beta = math.tanh(Y)
            gamma = math.cosh(Y)
            ep = gamma * (jet["e"] - beta * jet["pz"])
            pzp = gamma * (jet["pz"] - beta * jet["e"])
            ycm = rap(pzp, ep)
            pxc, pyc, pzc = jet["px"], jet["py"], pzp
            etacm = eta(pxc, pyc, pzc)
            # final-state Z+jet system
            z = [p for p in fin if p["pdg"] == 23]
            if z:
                zs = z[0]
                es = zs["e"] + jet["e"]
                pzs = zs["pz"] + jet["pz"]
                pxs = zs["px"] + jet["px"]
                pys = zs["py"] + jet["py"]
                m2 = es * es - pxs * pxs - pys * pys - pzs * pzs
                shat_final = math.sqrt(max(m2, 0.0))
                Y_final = rap(pzs, es)
            else:
                shat_final = float("nan")
                Y_final = float("nan")
            rows.append({
                "source": label,
                "weight": weight,
                "x1": x1, "x2": x2, "Y": Y, "shat": shat,
                "shat_final": shat_final, "Y_final": Y_final,
                "jet_pdg": jet["pdg"], "jet_pt": math.hypot(jet["px"], jet["py"]),
                "y_lab": ylab, "eta_lab": etalab,
                "y_cm": ycm, "eta_cm": etacm,
                "init_pdg1": ini[0]["pdg"], "init_pdg2": ini[1]["pdg"],
                "channel": channel(ini[0]["pdg"], ini[1]["pdg"]),
                "n_final": len(fin), "n_jets": len(jets),
            })
    print("events:", len(rows))
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print("weight sum:", arr["weight"].sum())
    print("lab frac:", np.average(arr["y_lab"].__abs__() < 2.5, weights=arr["weight"]))
    print("cm  frac:", np.average(arr["y_cm"].__abs__() < 2.5, weights=arr["weight"]))
    print("eta lab frac:", np.average(arr["eta_lab"].__abs__() < 2.5, weights=arr["weight"]))
    print("eta cm  frac:", np.average(arr["eta_cm"].__abs__() < 2.5, weights=arr["weight"]))
    print("max |shat-shat_final|:", np.nanmax(np.abs(arr["shat"] - arr["shat_final"])))
    print("max |Y-Y_final|:", np.nanmax(np.abs(arr["Y"] - arr["Y_final"])))
    print("max |y_cm-(y_lab-Y)|:", np.max(np.abs(arr["y_cm"] - (arr["y_lab"] - arr["Y"]))))
    for ch in sorted(set(arr["channel"])):
        mask = arr["channel"] == ch
        print(f"{ch:10s} n={mask.sum():6d} w={arr['weight'][mask].sum():10.2f} "
              f"lab={np.average(arr['y_lab'][mask].__abs__() < 2.5, weights=arr['weight'][mask]):.4f} "
              f"cm={np.average(arr['y_cm'][mask].__abs__() < 2.5, weights=arr['weight'][mask]):.4f}")


if __name__ == "__main__":
    main()
