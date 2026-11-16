#!/usr/bin/env python3
"""Plot the hadronic differential cross section, acceptance, and cumulative integral."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

NOETA = "#2a78d6"
LAB = "#eb6834"
CM = "#1baf7a"
GRAY = "#666666"


def main() -> None:
    rows = list(csv.DictReader((HERE / "differential.csv").open()))
    e = np.array([float(r["energy_gev"]) for r in rows])
    i = np.array([float(r["dsigma_dsqrts_gev_pb_per_gev"]) for r in rows])
    flab = np.array([float(r["f_lab"]) for r in rows])
    fcm = np.array([float(r["f_cm"]) for r in rows])

    acc = list(csv.DictReader((HERE.parent / "results" / "eta_frame_events.csv").open()))
    b = np.array([float(r["shat_gev"]) for r in acc])
    fl_raw = np.array([float(r["lab_cut_fraction"]) for r in acc])
    fc_raw = np.array([float(r["cm_cut_fraction"]) for r in acc])
    wb = np.array([float(r["weight"]) for r in acc])
    ilab = i * flab
    icm = i * fcm

    def cum(y):
        seg = 0.5 * (y[1:] + y[:-1]) * np.diff(e)
        return np.concatenate([[0.0], np.cumsum(seg)])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    ax = axes[0, 0]
    ax.plot(e, i, color=NOETA, lw=1.8, label="no eta")
    ax.plot(e, ilab, color=LAB, lw=1.8, label=r"lab $|\eta|<2.5$ (stored)")
    ax.plot(e, icm, color=CM, lw=1.2, ls="--", label=r"CM $|\eta|<2.5$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(114, 500)
    ax.set_ylim(0.3, 300)
    ax.set_xticks([114, 150, 200, 250, 300, 400, 500])
    ax.set_xticklabels(["114", "150", "200", "250", "300", "400", "500"])
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel(r"$d\sigma/d\sqrt{\hat s}$ (pb GeV$^{-1}$)")
    ax.set_title("Hadronic differential cross section")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.03, 0.05, "tail above 500 GeV is < 1.0% of the total",
            transform=ax.transAxes, fontsize=7, color="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    # cumulative with the official raw acceptance
    ax.plot(e, cum(i), color=NOETA, lw=1.8,
            label=f"no eta {cum(i)[-1]:,.0f} pb")
    ax.plot(e, cum(i * flab), color=LAB, lw=1.8,
            label=f"lab {cum(i * flab)[-1]:,.0f} pb (stored)")
    ax.plot(e, cum(i * fcm), color=CM, lw=1.2, ls="--",
            label=f"CM {cum(i * fcm)[-1]:,.0f} pb")
    ax.set_xscale("log")
    ax.set_xlim(114, 2000)
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel("cumulative integral (pb)")
    ax.set_title("Cumulative integral: the total is the area")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.text(0.03, 0.05,
            "per-channel acceptance + threshold: 9,142 pb",
            transform=ax.transAxes, fontsize=7, color="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    ax.plot(e, flab, color=LAB, lw=1.8, label=r"$f_{\rm lab}$ (stored, interpolated)")
    ax.plot(e, fcm, color=CM, lw=1.8, ls="--", label=r"$f_{\rm CM}$ (stored, interpolated)")
    show = (b <= 550) & (wb > 1.0)
    ax.plot(b[show], fl_raw[show], "o", color=LAB, ms=3.0, alpha=0.45, label="stored bins")
    ax.plot(b[show], fc_raw[show], "o", color=CM, ms=3.0, alpha=0.45)
    # fresh no-eta sample, all-channel 25-GeV bins with enough effective events
    ev = list(csv.DictReader((HERE / "events_flat.csv").open()))
    w = np.array([float(r["weight"]) for r in ev])
    sh = np.array([float(r["shat"]) for r in ev])
    lab = np.abs(np.array([float(r["y_lab"]) for r in ev])) < 2.5
    edges = np.arange(112.5, 2025.0 + 25.0, 25.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fcur, ncur = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sh >= lo) & (sh < hi)
        if m.any():
            fcur.append(np.average(lab[m], weights=w[m]))
            ncur.append(m.sum())
        else:
            fcur.append(np.nan); ncur.append(0)
    fcur = np.array(fcur); ncur = np.array(ncur)
    good = np.isfinite(fcur) & (ncur >= 30)
    ax.plot(centers[good], fcur[good], "s", color=GRAY, ms=3.0, alpha=0.7,
            label=r"$f_{\rm lab}$ (fresh no-eta sample)")
    ax.set_xscale("log")
    ax.set_xlim(114, 500)
    ax.set_ylim(0.7, 1.02)
    ax.set_xticks([114, 150, 200, 250, 300, 400, 500])
    ax.set_xticklabels(["114", "150", "200", "250", "300", "400", "500"])
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel("acceptance")
    ax.set_title("Lab and CM acceptance")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    frac_lab = cum(i * flab) / cum(i)[-1]
    frac_noeta = cum(i) / cum(i)[-1]
    ax.plot(e, frac_noeta * 100, color=NOETA, lw=1.5, label="no eta")
    ax.plot(e, frac_lab * 100, color=LAB, lw=1.5, label="lab")
    ax.set_xscale("log")
    ax.set_xlim(114, 2000)
    ax.set_ylim(0, 105)
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel("fraction of total (%)")
    ax.set_title("Cumulative fraction of the total")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("From the partonic scan to the lab-frame hadronic total", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"hadronic_differential.{suffix}", dpi=180, bbox_inches="tight")
    print("wrote hadronic_differential.png/pdf")


if __name__ == "__main__":
    main()
