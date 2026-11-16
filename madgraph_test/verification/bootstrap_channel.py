#!/usr/bin/env python3
"""Bootstrap the channel-dependent vs single acceptance correction."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]


def load_integrands():
    data = defaultdict(dict)
    with (HERE / "channel_integrands.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data[row["channel"]][float(row["energy_gev"])] = float(row["integrand_pb_per_gev"])
    energies = np.array(sorted(data[CHANNELS[0]]))
    return energies, {ch: np.array([data[ch][e] for e in energies]) for ch in CHANNELS}


def main() -> None:
    energies, integrands = load_integrands()
    rows = list(csv.DictReader((HERE / "events_flat.csv").open()))
    w = np.array([float(r["weight"]) for r in rows])
    sh = np.array([float(r["shat"]) for r in rows])
    lab = (np.abs(np.array([float(r["y_lab"]) for r in rows])) < 2.5).astype(float)
    ch = np.array([r["channel"] for r in rows])
    wgrid = np.zeros_like(energies)
    wgrid[1:-1] = 0.5 * (energies[2:] - energies[:-2])
    wgrid[0] = 0.5 * (energies[1] - energies[0])
    wgrid[-1] = 0.5 * (energies[-1] - energies[-2])
    isum = sum(integrands[c] for c in CHANNELS)
    edges = np.arange(112.5, 2025.0 + 25.0, 25.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(sh, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(centers))
    ch_idx = np.array([CHANNELS.index(c) for c in ch])

    def totals(w_override):
        # per-channel bin sums
        total_chan = 0.0
        total_single = 0.0
        # all-channel
        sw = np.bincount(bin_idx[valid], weights=w_override[valid], minlength=len(centers))
        swl = np.bincount(bin_idx[valid], weights=w_override[valid] * lab[valid], minlength=len(centers))
        with np.errstate(invalid="ignore", divide="ignore"):
            fall = np.where(sw > 0, swl / sw, np.nan)
        good = np.isfinite(fall)
        if good.sum() < 2:
            return np.nan, np.nan
        f_all = np.interp(energies, centers[good], fall[good], left=fall[good][0], right=fall[good][-1])
        total_single = float(np.sum(wgrid * isum * f_all))
        for ci, c in enumerate(CHANNELS):
            m = valid & (ch_idx == ci)
            sw = np.bincount(bin_idx[m], weights=w_override[m], minlength=len(centers))
            swl = np.bincount(bin_idx[m], weights=w_override[m] * lab[m], minlength=len(centers))
            with np.errstate(invalid="ignore", divide="ignore"):
                fc = np.where(sw > 0, swl / sw, np.nan)
            good = np.isfinite(fc)
            if good.sum() < 2:
                return np.nan, total_single
            f = np.interp(energies, centers[good], fc[good], left=fc[good][0], right=fc[good][-1])
            total_chan += float(np.sum(wgrid * integrands[c] * f))
        return total_chan, total_single

    t_chan, t_single = totals(w)
    print(f"point estimate: single={t_single:.3f}  channel={t_chan:.3f}  "
          f"ratio={t_chan/t_single:.6f}  diff={t_chan-t_single:+.2f} pb")

    rng = np.random.default_rng(12345)
    ratios = []
    diffs = []
    for b in range(300):
        # Poisson bootstrap (weighted events): multiply each event by a Poisson(1) count
        mult = rng.poisson(1.0, size=len(w)).astype(float)
        wo = w * mult
        tc, ts = totals(wo)
        if np.isfinite(tc) and np.isfinite(ts) and ts > 0:
            ratios.append(tc / ts)
            diffs.append(tc - ts)
    ratios = np.array(ratios)
    diffs = np.array(diffs)
    print(f"bootstrap ({len(ratios)}): ratio mean={ratios.mean():.6f} std={ratios.std():.6f}")
    print(f"  diff mean={diffs.mean():+.2f} pb std={diffs.std():.2f} pb")
    lab_total = json.loads(
        (HERE.parent / "results" / "noeta_fine" / "lab_eta_correction.json").read_text()
    )["lab_cut_pb"]
    print(f"  applied to corrected stored-acceptance lab total {lab_total:.2f}: "
          f"{lab_total*ratios.mean():.2f} +/- {lab_total*ratios.std():.2f} pb")


if __name__ == "__main__":
    main()
