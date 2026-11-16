#!/usr/bin/env python3
"""Recompute the per-channel hadronic integrands and save them."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import integrate_and_plot as ip

HERE = Path(__file__).resolve().parent
SCAN = HERE.parent / "results" / "scan_noeta_fine.csv"
OUT = HERE / "channel_integrands.csv"


def main() -> None:
    scan = ip.read_scan(SCAN)
    pdf = ip.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    q2 = 91.188 ** 2
    yn, yw = np.polynomial.legendre.leggauss(200)
    rows = []
    for ch in ip.CHANNELS:
        for e, sigma, err in scan[ch]:
            lum = ip.luminosity(pdf, e, 13000.0, q2, ip.PAIRS[ch], yn, yw)
            rows.append({
                "energy_gev": e, "channel": ch,
                "sigma_pb": sigma, "sigma_err_pb": err,
                "luminosity_per_gev": lum,
                "integrand_pb_per_gev": lum * sigma,
                "integrand_err_pb_per_gev": lum * err,
            })
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} with {len(rows)} rows")


if __name__ == "__main__":
    main()
