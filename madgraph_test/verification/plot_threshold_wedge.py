#!/usr/bin/env python3
"""Show the missing threshold wedge 113.3555-114.0 GeV.

The true on-shell threshold for Z (M=91.188 GeV) plus a jet with pT>20 GeV is

    sqrt(MZ^2 + pT^2) + pT = 113.3555 GeV,

not MZ + pT = 111.188 GeV.  The partonic scan started at 114.0 GeV, so the
wedge contributes 33.84 pb no-eta and about 24.4 pb in the lab.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import integrate_and_plot as ip  # noqa: E402

MZ = 91.188
PT = 20.0
THRESHOLD = np.sqrt(MZ * MZ + PT * PT) + PT
SCAN_START = 114.0
COLORS = {"noeta": "#2a78d6", "lab": "#eb6834"}


def load_sigma():
    sig = {}
    for name in ("results/scan_noeta_fine.csv", "verification/scan_threshold.csv"):
        for row in csv.DictReader((ROOT / name).open(newline="", encoding="utf-8")):
            e = float(row["energy_gev"])
            sig.setdefault(e, {})[row["channel"]] = float(row["sigma_pb"])
    return sig


def main() -> None:
    sig = load_sigma()
    pdf = ip.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    q2 = MZ * MZ
    yn, yw = np.polynomial.legendre.leggauss(200)
    acc = list(csv.DictReader((ROOT / "results/eta_frame_events.csv").open(
        newline="", encoding="utf-8")))
    b = np.array([float(r["shat_gev"]) for r in acc])
    fl = np.array([float(r["lab_cut_fraction"]) for r in acc])

    energies = np.array([THRESHOLD, 113.4, 113.6, 113.8, 114.0, 116.77551,
                         119.55102, 122.326531, 125.102041, 127.877551])
    noeta, lab = [], []
    for e in energies:
        if e <= THRESHOLD + 1e-9:
            noeta.append(0.0)
            lab.append(0.0)
            continue
        nearest = min(sig, key=lambda x: abs(x - e))
        f = float(np.interp(e, b, fl, left=fl[0], right=fl[-1]))
        total = 0.0
        for ch in ip.CHANNELS:
            if ch not in sig[nearest]:
                continue
            lum = ip.luminosity(pdf, e, 13000.0, q2, ip.PAIRS[ch], yn, yw)
            total += lum * sig[nearest][ch]
        noeta.append(total)
        lab.append(total * f)
    noeta = np.array(noeta)
    lab = np.array(lab)

    # wedge integrals from threshold to 114, using the same trapezoid
    mask = energies <= SCAN_START + 1e-9
    e_w = energies[mask]
    noeta_area = float(np.sum(0.5 * (noeta[mask][1:] + noeta[mask][:-1]) * np.diff(e_w)))
    lab_area = float(np.sum(0.5 * (lab[mask][1:] + lab[mask][:-1]) * np.diff(e_w)))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    for ax, values, color, label, area in (
        (axes[0], noeta, COLORS["noeta"], "no $\\eta$ cut", noeta_area),
        (axes[1], lab, COLORS["lab"], "lab $|\\eta|<2.5$", lab_area),
    ):
        ax.plot(energies, values, color=color, lw=2.0, marker="o", ms=4)
        ax.fill_between(e_w, 0, values[mask], color=color, alpha=0.18, lw=0)
        ax.axvline(THRESHOLD, color="0.35", lw=1.0, ls="--")
        ax.axvline(SCAN_START, color="0.35", lw=1.0, ls=":")
        ax.set_xlim(112.8, 128.5)
        ax.set_yscale("log")
        ax.set_ylim(8, 260)
        ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
        ax.set_ylabel(r"$dL/d\sqrt{\hat s}\times\sigma$ (pb GeV$^{-1}$)")
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(THRESHOLD + 0.03, 150, "true threshold\n113.3555",
                fontsize=7.5, color="0.35", va="top")
        ax.text(SCAN_START + 0.10, 112, "scan start\n114.0",
                fontsize=7.5, color="0.35", va="top")
        ax.annotate(f"missing wedge\n{area:.1f} pb",
                    xy=(113.68, 35), xytext=(115.0, 18),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9))
    axes[0].text(0.98, 0.05,
                 "native run card: $\\sqrt{\\hat s}_{\\min}=111.188$ GeV\n"
                 "($=M_Z+p_T$, below the true threshold)",
                 transform=axes[0].transAxes, fontsize=7.2, color="#555555",
                 ha="right", va="bottom")
    fig.suptitle("The missing threshold wedge: the scan started 0.6445 GeV too high",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"threshold_wedge.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"threshold {THRESHOLD:.4f} GeV")
    print(f"no-eta wedge {noeta_area:.3f} pb; lab wedge {lab_area:.3f} pb")
    print("wrote verification/threshold_wedge.png/pdf")


if __name__ == "__main__":
    main()
