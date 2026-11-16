#!/usr/bin/env python3
"""Test whether the lab acceptance is channel-dependent and quantify the bias."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SCAN = HERE.parent / "results" / "scan_noeta_fine.csv"
EVENTS = HERE / "events_flat.csv"
ACC = HERE.parent / "results" / "eta_frame_events.csv"
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]


def load_integrands():
    data = defaultdict(dict)
    with (HERE / "channel_integrands.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data[row["channel"]][float(row["energy_gev"])] = (
                float(row["integrand_pb_per_gev"]),
                float(row["integrand_err_pb_per_gev"]),
            )
    energies = np.array(sorted(data[CHANNELS[0]]))
    return energies, {ch: np.array([data[ch][e][0] for e in energies]) for ch in CHANNELS}


def load_events():
    rows = list(csv.DictReader(EVENTS.open()))
    w = np.array([float(r["weight"]) for r in rows])
    sh = np.array([float(r["shat"]) for r in rows])
    lab = np.abs(np.array([float(r["y_lab"]) for r in rows])) < 2.5
    cm = np.abs(np.array([float(r["y_cm"]) for r in rows])) < 2.5
    ch = np.array([r["channel"] for r in rows])
    return w, sh, lab, cm, ch


def binned_fraction(sh, lab, w, edges, ch_mask=None):
    """Weighted fraction in each bin; returns centers and fractions."""
    if ch_mask is not None:
        sh, lab, w = sh[ch_mask], lab[ch_mask], w[ch_mask]
    centers = []
    frac = []
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sh >= lo) & (sh < hi)
        if m.sum() == 0:
            centers.append(0.5 * (lo + hi))
            frac.append(np.nan)
            counts.append(0)
            continue
        centers.append(0.5 * (lo + hi))
        frac.append(float(np.average(lab[m], weights=w[m])))
        counts.append(int(m.sum()))
    return np.array(centers), np.array(frac), np.array(counts)


def interp_nan(x, y, xnew):
    good = np.isfinite(y)
    return np.interp(xnew, x[good], y[good], left=y[good][0], right=y[good][-1])


def main() -> None:
    energies, integrands = load_integrands()
    w, sh, lab, cm, ch = load_events()
    wgrid = np.zeros_like(energies)
    wgrid[1:-1] = 0.5 * (energies[2:] - energies[:-2])
    wgrid[0] = 0.5 * (energies[1] - energies[0])
    wgrid[-1] = 0.5 * (energies[-1] - energies[-2])
    isum = sum(integrands[c] for c in CHANNELS)
    t_noeta = float(np.sum(wgrid * isum))

    # Official 25-GeV all-channel acceptance.
    acc = list(csv.DictReader(ACC.open()))
    ab = np.array([float(r["shat_gev"]) for r in acc])
    af = np.array([float(r["lab_cut_fraction"]) for r in acc])
    f_all = np.interp(energies, ab, af, left=af[0], right=af[-1])
    t_single = float(np.sum(wgrid * isum * f_all))
    print(f"no-eta total: {t_noeta:.4f} pb")
    print(f"official 25-GeV single acceptance: {t_single:.4f} pb (eff {t_single/t_noeta:.6f})")

    # Per-channel acceptance with different binnings.
    for width_low, width_high, split in [(50, 100, 500), (25, 50, 500), (100, 200, 500)]:
        edges = list(np.arange(100, split + width_low, width_low))
        edges += list(np.arange(split + width_high, 2500 + width_high, width_high))
        edges = np.array(edges, dtype=float)
        print(f"\nbinning: {width_low} GeV below {split}, {width_high} GeV above")
        total_chan = 0.0
        for c in CHANNELS:
            centers, frac, counts = binned_fraction(sh, lab, w, edges, ch == c)
            fc = interp_nan(centers, frac, energies)
            contrib = float(np.sum(wgrid * integrands[c] * fc))
            total_chan += contrib
            print(f"  {c:8s} contrib={contrib:9.2f} pb  f_eff={contrib/float(np.sum(wgrid*integrands[c])):.4f}  "
                  f"n_min={counts.min()} n_med={int(np.median(counts))}")
        print(f"  channel-dependent total: {total_chan:.4f} pb  "
              f"diff from single: {total_chan-t_single:+.2f} pb ({(total_chan/t_single-1)*100:+.2f}%)")

    # Event-sample channel mix vs partonic channel integrals.
    print("\nchannel mix: event sample vs partonic integrand")
    ev_w = {c: float(w[ch == c].sum()) for c in CHANNELS}
    tot_ev = sum(ev_w.values())
    for c in CHANNELS:
        part = float(np.sum(wgrid * integrands[c]))
        print(f"  {c:8s} event {ev_w[c]/tot_ev*100:6.2f}%  partonic {part/t_noeta*100:6.2f}%")


if __name__ == "__main__":
    main()
