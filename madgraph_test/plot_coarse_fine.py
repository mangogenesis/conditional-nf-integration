#!/usr/bin/env python3
"""Visualise the coarse and fine sqrt(s-hat) grids and their effect on the total."""
from __future__ import annotations

import json
import os
from pathlib import Path

if "LHAPDF_DATA_PATH" not in os.environ:
    _fallback = (Path(__file__).resolve().parent.parent
                 / "lhapdf-local" / "share" / "LHAPDF")
    if _fallback.is_dir():
        os.environ["LHAPDF_DATA_PATH"] = str(_fallback).replace("\\", "/")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import integrate_and_plot as ip

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

COARSE_COLOR = "#2a78d6"
FINE_COLOR = "#eb6834"


def integrand(scan):
    energies = np.array([e for e, _, _ in scan[ip.CHANNELS[0]]])
    pdf = ip.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    q2 = 91.188 ** 2
    sqrts = 13000.0
    y_nodes, y_weights = np.polynomial.legendre.leggauss(200)
    values = np.zeros_like(energies)
    for ch in ip.CHANNELS:
        sigma = np.array([s for _, s, _ in scan[ch]])
        lum = np.array([
            ip.luminosity(pdf, e, sqrts, q2, ip.PAIRS[ch], y_nodes, y_weights)
            for e in energies
        ])
        values += lum * sigma
    return energies, values


def trapezoid_cumulative(x, y):
    dx = np.diff(x)
    seg = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])


def main() -> None:
    coarse = ip.read_scan(RESULTS / "scan_noeta.csv")
    fine = ip.read_scan(RESULTS / "scan_noeta_fine.csv")
    e_coarse, i_coarse = integrand(coarse)
    e_fine, i_fine = integrand(fine)

    coarse_total = float(np.sum(0.5 * (i_coarse[1:] + i_coarse[:-1]) * np.diff(e_coarse)))
    fine_total = float(np.sum(0.5 * (i_fine[1:] + i_fine[:-1]) * np.diff(e_fine)))

    coarse_json = json.loads((RESULTS / "total_integral.json").read_text())
    fine_json = json.loads((RESULTS / "noeta_fine" / "total_integral.json").read_text())
    coarse_lab = json.loads((RESULTS / "lab_eta_correction.json").read_text())
    fine_lab = json.loads((RESULTS / "noeta_fine" / "lab_eta_correction.json").read_text())
    nf = json.loads((HERE / "verification" / "nf_lab_total.json").read_text())
    totals = {
        "no eta": (coarse_json["total_noeta_pb"], fine_json["total_noeta_pb"]),
        "CM |eta|<2.5": (coarse_lab["cm_cut_pb"], fine_lab["cm_cut_pb"]),
        "lab |eta|<2.5": (coarse_lab["lab_cut_pb"], nf["exact_lab_pb"]),
    }

    # ------------------------------------------------------------------ grids
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(9.2, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.6]},
    )
    ax.scatter(e_coarse, np.ones_like(e_coarse), marker="|", s=140,
               color=COARSE_COLOR, label="coarse (50 points)")
    ax.scatter(e_fine, np.zeros_like(e_fine), marker="|", s=140,
               color=FINE_COLOR, label="fine (100 points)")
    ax.axvspan(114, 250, color="#999999", alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlim(114, 2000)
    ticks = [114, 150, 200, 300, 500, 700, 1000, 2000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.tick_params(labelbottom=False)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["fine", "coarse"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_ylabel("grid")
    ax.set_title(r"Coarse vs fine $\sqrt{\hat s}$ grids (114–2000 GeV, log x)")
    ax.grid(True, axis="x", which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center left")
    ax.text(120, 1.38,
            "coarse: 4 points below 250 GeV\nfine: 52 points below 250 GeV",
            fontsize=8, color="#333333", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))

    mid_c = 0.5 * (e_coarse[1:] + e_coarse[:-1])
    d_c = np.diff(e_coarse)
    mid_f = 0.5 * (e_fine[1:] + e_fine[:-1])
    d_f = np.diff(e_fine)
    mask = d_f > 0.1  # drop the one near-duplicate point at 200 GeV
    axb.plot(mid_c, d_c, color=COARSE_COLOR, lw=1.6, marker="o", ms=3,
             label="coarse")
    axb.plot(mid_f[mask], d_f[mask], color=FINE_COLOR, lw=1.6, marker="o", ms=3,
             label="fine")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(114, 2000)
    axb.set_ylim(1.5, 80)
    axb.set_xticks(ticks)
    axb.set_xticklabels([str(t) for t in ticks])
    axb.set_xlabel(r"$\sqrt{\hat s}$ midpoint (GeV)")
    axb.set_ylabel("grid spacing (GeV)")
    axb.set_title("Grid spacing: coarse vs fine")
    axb.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.legend(fontsize=8, frameon=False, loc="lower right")
    axb.text(126, 2.1, "2.78 GeV", color=FINE_COLOR, fontsize=8)
    axb.text(520, 31, "36.73 GeV", color=FINE_COLOR, fontsize=8)
    axb.text(1000, 44, "38.49 GeV", color=COARSE_COLOR, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"coarse_fine_grids.{suffix}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------- effect
    fig = plt.figure(figsize=(10.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(e_fine, i_fine, color=FINE_COLOR, lw=1.8, label="fine grid")
    ax.plot(e_coarse, i_coarse, color=COARSE_COLOR, marker="o", ms=4, lw=1.2,
            label="coarse grid")
    ax.axvspan(114, 250, color="#999999", alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(114, 2000)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"$dL/d\sqrt{\hat s}\times\sigma$ (pb GeV$^{-1}$)")
    ax.set_title("Hadronic integrand on the coarse and fine grids")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="best")

    axc = fig.add_subplot(gs[1, 0])
    cum_c = trapezoid_cumulative(e_coarse, i_coarse)
    cum_f = trapezoid_cumulative(e_fine, i_fine)
    axc.plot(e_coarse, cum_c, color=COARSE_COLOR, lw=1.6, label=f"coarse {coarse_total:,.0f}")
    axc.plot(e_fine, cum_f, color=FINE_COLOR, lw=1.6, label=f"fine {fine_total:,.0f}")
    axc.set_xscale("log")
    axc.set_xlim(114, 2000)
    axc.set_xticks(ticks)
    axc.set_xticklabels([str(t) for t in ticks], fontsize=7.5)
    axc.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    axc.set_ylabel("cumulative integral (pb)")
    axc.set_title("Cumulative integral")
    axc.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    axc.spines["top"].set_visible(False)
    axc.spines["right"].set_visible(False)
    axc.legend(fontsize=8, frameon=False, loc="best")

    axb2 = fig.add_subplot(gs[1, 1])
    labels = list(totals)
    x = np.arange(len(labels))
    width = 0.34
    cvals = [totals[k][0] for k in labels]
    fvals = [totals[k][1] for k in labels]
    axb2.bar(x - width / 2, cvals, width, color=COARSE_COLOR, label="coarse")
    axb2.bar(x + width / 2, fvals, width, color=FINE_COLOR, label="fine")
    for xi, v in zip(x - width / 2, cvals):
        axb2.text(xi, v + 250, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + width / 2, fvals):
        axb2.text(xi, v + 250, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
    for xi, k in zip(x, labels):
        c, f = totals[k]
        axb2.text(xi, max(c, f) + 1600, f"{(f-c)/f*100:.1f}%", ha="center",
                  fontsize=8, color="#333333")
    axb2.set_xticks(x)
    axb2.set_xticklabels(labels)
    axb2.set_ylabel("total cross section (pb)")
    axb2.set_ylim(0, 18000)
    axb2.set_title("Coarse vs fine totals")
    axb2.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    axb2.spines["top"].set_visible(False)
    axb2.spines["right"].set_visible(False)
    axb2.legend(fontsize=8, frameon=False, loc="upper right")

    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"coarse_fine_effect.{suffix}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"coarse no-eta: {coarse_total:.2f} pb")
    print(f"fine no-eta:   {fine_total:.2f} pb")
    print("wrote coarse_fine_grids and coarse_fine_effect")


if __name__ == "__main__":
    main()
