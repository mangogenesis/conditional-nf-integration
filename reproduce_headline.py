#!/usr/bin/env python3
"""Reproduce the Z+1 NF lab-frame total from the packaged tables.

This needs no trained checkpoints, no MadGraph build, and no LHE events: it
reads `madgraph_test/verification/nf_lab_acceptance.csv` and the NF
interpolation tables under `conditional-nf-integration/results/`.

Expected final line: 9155.14 +/- 8.92 pb (NF lab + threshold).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MADGRAPH_TEST = ROOT / "madgraph_test"
sys.path.insert(0, str(MADGRAPH_TEST / "verification"))

import apply_lab_frame_nf as nfmod  # noqa: E402


def main() -> None:
    acceptance = MADGRAPH_TEST / "verification" / "nf_lab_acceptance.csv"
    rows = [
        {
            "channel": row["channel"],
            "energy_gev": float(row["energy_gev"]),
            "f_lab_given_cm": float(row["f_lab_given_cm"]),
            "error": float(row["error"]),
        }
        for row in csv.DictReader(acceptance.open(encoding="utf-8"))
    ]
    nf = nfmod.read_nf()
    pdf = nfmod.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    nfmod.integrate_nf_lab(rows, nf, pdf)


if __name__ == "__main__":
    main()
