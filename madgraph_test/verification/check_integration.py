#!/usr/bin/env python3
"""Check the sqrt(s-hat) integration, interpolation, and differential cross section."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson

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


def trapz(x, y):
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return float(np.sum(w * y))


def main() -> None:
    energies, vals, errs = load()
    isum = sum(vals[c] for c in CHANNELS)
    esum = np.sqrt(sum(errs[c] ** 2 for c in CHANNELS))
    official = trapz(energies, isum)
    print(f"official 100-point trapezoid: {official:.4f} pb")
    print(f"differential at 114, 150, 200, 250, 500, 1000, 2000 GeV:")
    for e in (114, 150, 200, 250, 500, 1000, 2000):
        i = int(np.argmin(np.abs(energies - e)))
        print(f"  {e:5.0f}  {isum[i]:12.6e} +/- {esum[i]:.3e} pb/GeV")

    # fine-grid interpolation and integration
    fine = np.arange(114.0, 2000.0 + 0.5, 0.5)
    print("\nfine-grid integration (0.5 GeV):")
    for method in ("linear", "pchip"):
        tot = 0.0
        tot_lab = 0.0
        for c in CHANNELS:
            if method == "linear":
                y = np.interp(fine, energies, vals[c])

            else:
                y = PchipInterpolator(energies, vals[c])(fine)
                ylog = np.exp(PchipInterpolator(np.log(energies), np.log(vals[c]))(np.log(fine)))
            tot += trapz(fine, y)
        print(f"  {method:6s} linear-y: {tot:.4f} pb  (delta {tot-official:+.4f})")
    # log-linear and pchip in log space
    for method in ("log-linear", "log-pchip"):
        tot = 0.0
        for c in CHANNELS:
            if method == "log-linear":
                y = np.exp(np.interp(np.log(fine), np.log(energies), np.log(vals[c])))
            else:
                y = np.exp(PchipInterpolator(np.log(energies), np.log(vals[c]))(np.log(fine)))
            tot += trapz(fine, y)
        print(f"  {method:10s}: {tot:.4f} pb  (delta {tot-official:+.4f})")

    # Simpson on the fine grid using PCHIP in log space
    ysum = sum(np.exp(PchipInterpolator(np.log(energies), np.log(vals[c]))(np.log(fine))) for c in CHANNELS)
    print(f"  Simpson (log-PCHIP): {simpson(ysum, x=fine):.4f} pb")

    # cumulative integral
    print("\ncumulative integral (official grid):")
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (isum[1:] + isum[:-1]) * np.diff(energies))])
    for e in (150, 200, 250, 500, 1000, 2000):
        i = int(np.argmin(np.abs(energies - e)))
        print(f"  <= {e:5.0f} GeV: {cum[i]:9.2f} pb ({cum[i]/official*100:5.1f}%)")


if __name__ == "__main__":
    main()
