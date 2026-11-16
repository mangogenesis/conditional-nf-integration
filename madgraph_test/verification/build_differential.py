#!/usr/bin/env python3
"""Build the hadronic differential cross section and cumulative integrals."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]


def load():
    data = defaultdict(dict)
    err = defaultdict(dict)
    with (HERE / "channel_integrands.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            e = float(row["energy_gev"])
            data[row["channel"]][e] = float(row["integrand_pb_per_gev"])
            err[row["channel"]][e] = float(row["integrand_err_pb_per_gev"])
    energies = np.array(sorted(data[CHANNELS[0]]))
    vals = {c: np.array([data[c][e] for e in energies]) for c in CHANNELS}
    errs = {c: np.array([err[c][e] for e in energies]) for c in CHANNELS}
    return energies, vals, errs


def main() -> None:
    energies, vals, errs = load()
    isum = sum(vals[c] for c in CHANNELS)
    esum = np.sqrt(sum(errs[c] ** 2 for c in CHANNELS))
    acc = list(csv.DictReader((HERE.parent / "results" / "eta_frame_events.csv").open()))
    b = np.array([float(r["shat_gev"]) for r in acc])
    fl = np.array([float(r["lab_cut_fraction"]) for r in acc])
    fc = np.array([float(r["cm_cut_fraction"]) for r in acc])
    fli = np.interp(energies, b, fl, left=fl[0], right=fl[-1])
    fci = np.interp(energies, b, fc, left=fc[0], right=fc[-1])
    rows = []
    for e, i, ie, flv, fcv in zip(energies, isum, esum, fli, fci):
        rows.append({
            "energy_gev": e,
            "dsigma_dsqrts_gev_pb_per_gev": i,
            "error_pb_per_gev": ie,
            "f_lab": flv,
            "f_cm": fcv,
            "lab_integrand_pb_per_gev": i * flv,
            "lab_error_pb_per_gev": ie * flv,
            "cm_integrand_pb_per_gev": i * fcv,
            "cm_error_pb_per_gev": ie * fcv,
        })
    out = HERE / "differential.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")

    def cum(y):
        w = np.zeros_like(energies)
        w[1:-1] = 0.5 * (energies[2:] - energies[:-2])
        w[0] = 0.5 * (energies[1] - energies[0])
        w[-1] = 0.5 * (energies[-1] - energies[-2])
        seg = 0.5 * (y[1:] + y[:-1]) * np.diff(energies)
        return np.concatenate([[0.0], np.cumsum(seg)]), float(np.sum(w * y))

    print("\ncumulative fractions")
    print("E     no-eta      lab        CM")
    for label, y in (("noeta", isum), ("lab", isum * fli), ("cm", isum * fci)):
        c, tot = cum(y)
        print(f"{label}: total {tot:.2f} pb")
        for e in (150, 200, 250, 500, 1000, 2000):
            j = int(np.argmin(np.abs(energies - e)))
            print(f"  <= {e:5.0f}: {c[j]:9.2f} pb ({c[j]/tot*100:5.1f}%)")


if __name__ == "__main__":
    main()
