#!/usr/bin/env python3
"""Compare native and partonic totals for the two finite cut scenarios."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    scenarios = ["pT only\n(no eta)", "both cuts\n$|\\eta|<2.5$"]
    native = [11913.0, 9142.3]
    native_err = [10.866, 6.7709]
    partonic = [11876.96, 9142.25]
    partonic_err = [0.6136, 4.24]

    x = np.arange(len(scenarios))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(x - width / 2, native, width, yerr=native_err, capsize=3,
           color="#2a78d6", label="native MadGraph (lab frame)")
    ax.bar(x + width / 2, partonic, width, yerr=partonic_err, capsize=3,
           color="#eb6834", label="partonic convolution (corrected PDF scale)")
    for xi, value in zip(x - width / 2, native):
        ax.text(xi, value + 250, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    for xi, value in zip(x + width / 2, partonic):
        ax.text(xi, value + 250, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("total cross section (pb)")
    ax.set_ylim(0, 14000)
    ax.set_title("MadGraph consistency after the PDF-scale fix: both scenarios agree")
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.text(0.02, 0.96, "no cuts ($p_T=0$): divergent, not shown",
            transform=ax.transAxes, fontsize=8, color="#555555", va="top")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"consistency_scenarios.{suffix}", dpi=180)
    plt.close(fig)
    print("wrote consistency_scenarios.png/pdf")


if __name__ == "__main__":
    main()
