#!/usr/bin/env python3
"""Justify the pT > 20 GeV and |eta| < 2.5 cuts.

Four panels:
  1. native pT-regulator scan: the cross section diverges as pT -> 0;
  2. partonic pT-regulator scan at sqrt(s-hat) = 200 GeV;
  3. native eta scan: no eta, |eta| < 5, |eta| < 2.5;
  4. event-level |eta_lab| distribution with the 2.5 cut.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"


def read_sequence(path: Path):
    rows = []
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        rows.append((float(row["ptj_gev"]), float(row["sigma_pb"]),
                     float(row["error_pb"]), row["note"]))
    return rows


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))

    # ---------------------------------------------------------- native pT scan
    ax = axes[0, 0]
    rows = read_sequence(RESULTS / "native_ptj_sequence.csv")
    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    e = [r[2] for r in rows]
    ax.errorbar(x, y, yerr=e, color=BLUE, lw=1.8, marker="o", ms=5,
                capsize=3, label="native $p p \\to Z j$, no $\\eta$ cut")
    ax.set_yscale("log")
    ax.set_xticks([20, 10, 5, 2, 1, 0])
    ax.set_xlim(22, -1)
    ax.set_ylim(7e3, 3e6)
    ax.set_xlabel(r"jet $p_T$ regulator (GeV)")
    ax.set_ylabel(r"total cross section (pb)")
    ax.set_title("Native: the cross section diverges as $p_T\\to0$", fontsize=10)
    ax.axvline(20, color=GREEN, lw=1.4, ls="--")
    ax.text(19.2, 2.2e4, "$p_T>20$\n(analysis cut)", color=GREEN, fontsize=8,
            va="bottom", ha="left")
    ax.annotate("$p_T=0$: $1.49\\times10^6$ pb\n(grid-dependent, not converged)",
                xy=(0.0, 1.485e6), xytext=(3.0, 2.2e5), fontsize=7.5,
                color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # -------------------------------------------------------- partonic pT scan
    ax = axes[0, 1]
    # Values from plot_divergence_endpoint.py; the pT=0 entry is a sum of
    # grid configurations, not a converged cross section.
    x = [20, 10, 5, 2, 0]
    y = [303.11, 416.13, 527.96, 675.71, 1268.76]
    ax.plot(x[:4], y[:4], color=ORANGE, lw=1.8, marker="o", ms=5,
            label="partonic $g q\\to Z q$ etc.")
    ax.plot(x[3:], y[3:], color=ORANGE, lw=1.2, ls="--")
    ax.plot([0], [y[-1]], marker="o", ms=7, mfc="white", mec=ORANGE,
            mew=1.8, ls="none", label="$p_T=0$: grid sum, combined $=0$")
    ax.set_yscale("log")
    ax.set_xticks([20, 10, 5, 2, 0])
    ax.set_xlim(22, -1)
    ax.set_ylim(220, 2200)
    ax.set_xlabel(r"jet $p_T$ regulator (GeV)")
    ax.set_ylabel(r"$\sigma$ at $\sqrt{\hat s}=200$ GeV (pb)")
    ax.set_title("Partonic: the same $p_T\\to0$ divergence", fontsize=10)
    ax.axvline(20, color=GREEN, lw=1.4, ls="--")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # -------------------------------------------------------------- eta scan
    ax = axes[1, 0]
    labels = ["no $\\eta$ cut", "$|\\eta|<5$", "$|\\eta|<2.5$"]
    values = [11913.0, 11892.0, 9142.3]
    errors = [10.866, 10.534, 6.7709]
    x = np.arange(len(values))
    bars = ax.bar(x, values, yerr=errors, capsize=4, width=0.58,
                  color=[BLUE, BLUE, ORANGE])
    for xi, v in zip(x, values):
        ax.text(xi, v + 250, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi, v / 2, f"{v/11913*100:.1f}%", ha="center", va="center",
                fontsize=9, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("native total (pb)")
    ax.set_ylim(0, 14200)
    ax.set_title("Native lab-frame $\\eta$ scan", fontsize=10)
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(1.0, 13500,
            "$|\\eta|<2.5$ removes 23.3% of the $p_T>20$ cross section",
            ha="center", fontsize=7.5, color="#444444")

    # ------------------------------------------------------- event eta shapes
    ax = axes[1, 1]
    rows = list(csv.DictReader((HERE / "events_flat.csv").open(newline="", encoding="utf-8")))
    w = np.array([float(r["weight"]) for r in rows])
    eta = np.array([float(r["eta_lab"]) for r in rows])
    good = np.isfinite(eta) & (np.abs(eta) < 5.0)
    bins = np.linspace(-5.0, 5.0, 61)
    hist, edges = np.histogram(eta[good], bins=bins, weights=w[good])
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.step(centers, hist, where="mid", color=BLUE, lw=1.6)
    ax.fill_between(centers, hist, step="mid", color=BLUE, alpha=0.18, lw=0)
    ax.axvline(-2.5, color=ORANGE, lw=1.4, ls="--")
    ax.axvline(2.5, color=ORANGE, lw=1.4, ls="--")
    ax.axvspan(-5, -2.5, color=ORANGE, alpha=0.10, lw=0)
    ax.axvspan(2.5, 5, color=ORANGE, alpha=0.10, lw=0)
    # Fraction over the full sample; the histogram is clipped at |eta|=5 only
    # for display.  The native eta scan uses rapidity and gives 76.7%; the
    # event-level eta fraction is 76.2% (the small difference is eta vs y).
    full = np.isfinite(eta)
    inside = np.average(np.abs(eta[full]) < 2.5, weights=w[full])
    ax.set_xlim(-5, 5)
    ax.set_xlabel(r"jet $\eta_{\rm lab}$")
    ax.set_ylabel("weighted events / bin")
    ax.set_title(r"Event-level $\eta$ distribution (native, $p_T>20$)", fontsize=10)
    ax.text(0.02, 0.95,
            f"inside $|\\eta|<2.5$: {inside*100:.1f}%\n"
            f"removed: {(1-inside)*100:.1f}%\n"
            f"(independent scan: 76.7%)",
            transform=ax.transAxes, fontsize=7.5, color="#333333", va="top")
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Why the analysis uses $p_T>20$ GeV and $|\\eta|<2.5$", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"cuts_justification.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote verification/cuts_justification.png/pdf")


if __name__ == "__main__":
    main()
