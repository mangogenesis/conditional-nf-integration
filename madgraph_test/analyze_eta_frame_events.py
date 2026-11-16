#!/usr/bin/env python3
"""Apply the eta cut in the lab and partonic-CM frames to native LHE events.

The native events are from the no-eta LHAPDF run (etaj=-1, ptj=20), i.e.
almost no eta cut.  The event sample is validated by the weight sum
(11,913.95 pb) and the CM fraction (0.9897).
For each event:
    x1 = E(initial +z parton) / E_beam
    x2 = E(initial -z parton) / E_beam
    Y  = 0.5 * ln(x1 / x2)          (partonic CM rapidity in the lab)
    y_lab = rapidity of the final jet
    y_CM  = y_lab - Y

Then compare the weighted fractions with |y_lab| < 2.5 and |y_CM| < 2.5.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ME = HERE / "native_zj_lhapdf" / "SubProcesses"
EBEAM = 6500.0
OUT = HERE / "results" / "eta_frame_events.csv"


def rapidity(pz: float, energy: float) -> float:
    if abs(energy - pz) < 1e-12:
        return math.copysign(50.0, pz)
    return 0.5 * math.log((energy + pz) / (energy - pz))


def main() -> None:
    files = sorted(ME.glob("P*/G*/events.lhe"))
    if not files:
        raise SystemExit("no native events.lhe files found")
    total = 0.0
    lab_cut = 0.0
    cm_cut = 0.0
    both_cut = 0.0
    n_events = 0
    bins = {}
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as handle:
            in_event = False
            weight = 0.0
            particles = []
            for line in handle:
                text = line.strip()
                if text == "<event>":
                    in_event = True
                    particles = []
                    weight = 0.0
                    continue
                if text == "</event>":
                    if not in_event or not particles:
                        in_event = False
                        continue
                    # initial partons: status -1; final jet: status 1, pdg != 23
                    ini = [p for p in particles if p[1] == -1]
                    jet = [p for p in particles if p[1] == 1 and abs(p[0]) != 23]
                    if len(ini) != 2 or not jet:
                        in_event = False
                        continue
                    ini.sort(key=lambda p: p[6])  # pz
                    x1 = ini[1][7] / EBEAM
                    x2 = ini[0][7] / EBEAM
                    if x1 <= 0 or x2 <= 0:
                        in_event = False
                        continue
                    Y = 0.5 * math.log(x1 / x2)
                    shat = math.sqrt(x1 * x2) * 2.0 * EBEAM
                    j = max(jet, key=lambda p: math.hypot(p[4], p[5]))
                    ylab = rapidity(j[6], j[7])
                    ycm = ylab - Y
                    in_lab = abs(ylab) < 2.5
                    in_cm = abs(ycm) < 2.5
                    total += weight
                    lab_cut += weight * in_lab
                    cm_cut += weight * in_cm
                    both_cut += weight * (in_lab and in_cm)
                    n_events += 1
                    key = round(shat / 25.0) * 25
                    row = bins.setdefault(key, [0.0, 0.0, 0.0])
                    row[0] += weight
                    row[1] += weight * in_lab
                    row[2] += weight * in_cm
                    in_event = False
                    continue
                if not in_event:
                    continue
                fields = text.split()
                if len(fields) == 6 and not particles:
                    weight = float(fields[2])
                    continue
                if len(fields) >= 13:
                    try:
                        particles.append([
                            int(fields[0]), int(fields[1]), int(fields[2]),
                            int(fields[3]), float(fields[6]), float(fields[7]),
                            float(fields[8]), float(fields[9]), float(fields[10]),
                        ])
                    except ValueError:
                        continue
    print(f"files: {len(files)}  events: {n_events}")
    print(f"weighted total: {total:.6g}")
    print(f"lab |eta|<2.5 : {lab_cut/total:.4f}")
    print(f"CM  |eta|<2.5 : {cm_cut/total:.4f}")
    print(f"both          : {both_cut/total:.4f}")

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shat_gev", "weight", "lab_cut_fraction", "cm_cut_fraction"])
        for key in sorted(bins):
            w, lab, cm = bins[key]
            if w > 0:
                writer.writerow([key, w, lab / w, cm / w])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
