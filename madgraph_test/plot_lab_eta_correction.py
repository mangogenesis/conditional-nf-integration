#!/usr/bin/env python3
"""Compare the coarse and fine scans for every cut scenario."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

COARSE = "#9a9a9a"
FINE = "#2a78d6"
NATIVE = "#1baf7a"


def main() -> None:
    coarse = json.loads((RESULTS / "lab_eta_correction.json").read_text())
    fine = json.loads((RESULTS / "noeta_fine" / "lab_eta_correction.json").read_text())
    nf = json.loads((HERE / "verification" / "nf_lab_total.json").read_text())
    groups = ["no $\\eta$", "CM $|\\eta|<2.5$", "lab $|\\eta|<2.5$"]
    cvals = [coarse["no_eta_pb"], coarse["cm_cut_pb"], coarse["lab_cut_pb"]]
    fvals = [fine["no_eta_pb"], fine["cm_cut_pb"], nf["exact_lab_pb"]]

    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x - width / 2, cvals, width, color=COARSE, label="coarse 50-point scan")
    ax.bar(x + width / 2, fvals, width, color=FINE, label="fine 100-point scan")
    for xi, v in zip(x - width / 2, cvals):
        ax.text(xi, v + 180, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, fvals):
        ax.text(xi, v + 180, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    for xi, c, f in zip(x, cvals, fvals):
        ax.text(xi, max(c, f) + 1050, f"coarse\n{(c/f-1)*100:+.1f}%", ha="center",
                fontsize=7.5, color="#333333")
    ax.axhline(9142.3, color=NATIVE, lw=1.3, ls="--",
               label="native lab $|\\eta|<2.5$ = 9,142.3 pb")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("total cross section (pb)")
    ax.set_ylim(0, 14500)
    ax.set_title("Coarse-grid truncation affects every cut scenario; the fine lab matches native",
                 fontsize=10)
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"lab_eta_correction.{suffix}", dpi=180)
    plt.close(fig)
    print("wrote lab_eta_correction.png/pdf")


if __name__ == "__main__":
    main()
