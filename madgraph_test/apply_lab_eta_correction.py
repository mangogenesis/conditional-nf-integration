#!/usr/bin/env python3
"""Apply the lab-frame eta acceptance to the partonic no-eta convolution.

The acceptance is extracted from the native LHE events by
``analyze_eta_frame_events.py``: for each sqrt(s-hat) bin it records the
weighted fraction of events passing |eta_lab| < 2.5 and |eta_CM| < 2.5.
The partonic no-eta integrand is then multiplied by that acceptance and
integrated with the same trapezoid rule as the main convolution.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import integrate_and_plot as ip

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def read_acceptance(path: Path):
    shat = []
    lab = []
    cm = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            shat.append(float(row["shat_gev"]))
            lab.append(float(row["lab_cut_fraction"]))
            cm.append(float(row["cm_cut_fraction"]))
    return np.array(shat), np.array(lab), np.array(cm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=RESULTS / "scan_noeta.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--cm-with-cuts", type=float, default=None)
    parser.add_argument("--note", default="acceptance from native LHE events")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scan = ip.read_scan(args.scan)
    shat, lab_acc, cm_acc = read_acceptance(RESULTS / "eta_frame_events.csv")
    pdf = ip.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    q2 = 91.188 ** 2
    sqrts = 13000.0
    y_nodes, y_weights = np.polynomial.legendre.leggauss(200)

    energies = np.array([e for e, _, _ in scan[ip.CHANNELS[0]]])
    integrand = np.zeros_like(energies)
    for ch in ip.CHANNELS:
        sigma = np.array([s for _, s, _ in scan[ch]])
        lum = np.array([
            ip.luminosity(pdf, e, sqrts, q2, ip.PAIRS[ch], y_nodes, y_weights)
            for e in energies
        ])
        integrand += lum * sigma

    weights = np.zeros_like(energies)
    weights[1:-1] = 0.5 * (energies[2:] - energies[:-2])
    weights[0] = 0.5 * (energies[1] - energies[0])
    weights[-1] = 0.5 * (energies[-1] - energies[-2])

    lab = np.interp(energies, shat, lab_acc, left=lab_acc[0], right=lab_acc[-1])
    cm = np.interp(energies, shat, cm_acc, left=cm_acc[0], right=cm_acc[-1])

    total = float(np.sum(weights * integrand))
    total_lab = float(np.sum(weights * integrand * lab))
    total_cm = float(np.sum(weights * integrand * cm))

    rows = [
        {"quantity": "no eta", "total_pb": total, "ratio_to_noeta": 1.0},
        {"quantity": "lab |eta|<2.5", "total_pb": total_lab,
         "ratio_to_noeta": total_lab / total},
        {"quantity": "partonic CM |eta|<2.5", "total_pb": total_cm,
         "ratio_to_noeta": total_cm / total},
    ]
    with (args.output_dir / "lab_eta_correction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scan": str(args.scan),
        "no_eta_pb": total,
        "lab_cut_pb": total_lab,
        "cm_cut_pb": total_cm,
        "lab_fraction": total_lab / total,
        "cm_fraction": total_cm / total,
        "native_both_cuts_pb": 9142.3,
        "note": args.note,
    }
    if args.cm_with_cuts is not None:
        summary["partonic_cm_with_cuts_pb"] = args.cm_with_cuts
    (args.output_dir / "lab_eta_correction.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(f"{row['quantity']:25s} {row['total_pb']:10.2f} pb  "
              f"ratio={row['ratio_to_noeta']:.4f}")
    print(f"native both cuts: 9142.3 pb")
    print(f"wrote {args.output_dir / 'lab_eta_correction.csv'}")


if __name__ == "__main__":
    main()
