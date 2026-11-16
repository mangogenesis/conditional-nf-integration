#!/usr/bin/env python3
"""Double-check the fine integration grid and its spacing convergence."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]


def load():
    data = defaultdict(dict)
    with (HERE / "channel_integrands.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data[row["channel"]][float(row["energy_gev"])] = float(row["integrand_pb_per_gev"])
    e = np.array(sorted(data[CHANNELS[0]]))
    vals = {c: np.array([data[c][x] for x in e]) for c in CHANNELS}
    return e, vals


def trapz(x, y):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def main() -> None:
    e, vals = load()
    isum = sum(vals[c] for c in CHANNELS)
    official = trapz(e, isum)
    print(f"scan nodes: {len(e)}; first={e[0]:.6f}, last={e[-1]:.6f}")
    print(f"official 100-point trapezoid: {official:.6f} pb")
    print()
    print("spacing  n_points   linear integral      delta vs official   pchip integral     delta")
    for h in (2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        fine = np.arange(114.0, 2000.0 + h, h)
        # guard against a floating-point last point slightly above 2000
        fine = fine[fine <= 2000.0 + 1e-9]
        lin = 0.0
        pch = 0.0
        for c in CHANNELS:
            lin += trapz(fine, np.interp(fine, e, vals[c]))
            pch += trapz(fine, PchipInterpolator(e, vals[c])(fine))
        print(f"{h:7.3f}  {len(fine):7d}   {lin:16.6f}   {lin-official:+12.6f}   "
              f"{pch:16.6f}   {pch-official:+12.6f}")

    # Explicitly show the 0.5 GeV grid properties.
    h = 0.5
    fine = np.arange(114.0, 2000.0 + h, h)
    fine = fine[fine <= 2000.0 + 1e-9]
    print()
    print(f"0.5 GeV grid: n={len(fine)}, first={fine[0]}, last={fine[-1]}, "
          f"min spacing={np.min(np.diff(fine)):.12f}, max spacing={np.max(np.diff(fine)):.12f}")


if __name__ == "__main__":
    main()
