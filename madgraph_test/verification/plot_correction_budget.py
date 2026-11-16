#!/usr/bin/env python3
"""Waterfall of the corrections from last week's result to the native value.

Left:  exact partonic convolution.
Right: conditional NF.
The bars answer "where did the 13,700 -> 9,100 change come from?".
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXACT_COLOR = "#2a78d6"
NF_COLOR = "#eb6834"
GRAY = "#9a9a9a"
NATIVE = 9142.3
NATIVE_ERR = 6.8


def waterfall(ax, labels, start, deltas, final, color, title):
    """Draw a waterfall: start bar, signed deltas, final bar, native line."""
    n = len(labels)
    bottoms = np.zeros(n)
    heights = np.zeros(n)
    running = start
    bottoms[0] = 0.0
    heights[0] = start
    for i, delta in enumerate(deltas, start=1):
        if delta >= 0:
            bottoms[i] = running
            heights[i] = delta
        else:
            bottoms[i] = running + delta
            heights[i] = -delta
        running += delta
    bottoms[-1] = 0.0
    heights[-1] = final

    colors = [GRAY] + [color] * (n - 2) + [color]
    ax.bar(np.arange(n), heights, bottom=bottoms, color=colors,
           width=0.62, edgecolor="white", linewidth=1.0)
    # connectors
    running = start
    for i, delta in enumerate(deltas, start=1):
        ax.plot([i - 1 + 0.31, i - 0.31], [running, running],
                color="0.45", lw=0.8, ls=":")
        running += delta
    ax.axhline(NATIVE, color="#1baf7a", lw=1.4, ls="--",
               label=f"native {NATIVE:,.1f} $\\pm$ {NATIVE_ERR:.1f} pb")
    ax.axhspan(NATIVE - NATIVE_ERR, NATIVE + NATIVE_ERR,
               color="#1baf7a", alpha=0.12, lw=0)
    # labels
    for i in range(n):
        top = bottoms[i] + heights[i]
        value = heights[i] if i not in (0, n - 1) else heights[i]
        if i == 0:
            text = f"{start:,.0f}"
        elif i == n - 1:
            text = f"{final:,.0f}"
        else:
            sign = "+" if deltas[i - 1] >= 0 else "−"
            text = f"{sign}{abs(deltas[i - 1]):,.0f}"
        ax.text(i, top + 120, text, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylim(0, 15200)
    ax.set_ylabel("cross section (pb)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)

    exact_start = 13761.97
    exact_deltas = [11753.90 - exact_start, 9117.66 - 11753.90, 24.59]
    exact_final = 9142.25
    exact_labels = ["old exact\nCM cut", "PDF scale\nfix", "lab acceptance\n(per-channel)",
                    "threshold\nwedge", "corrected\nexact"]
    waterfall(axes[0], exact_labels, exact_start, exact_deltas, exact_final,
              EXACT_COLOR, "Exact partonic convolution")

    nf_start = 13765.5
    nf_deltas = [11752.44 - nf_start, 9041.84 - 11752.44, 88.71, 24.59]
    nf_final = 9155.14
    nf_labels = ["old NF\nCM cut", "PDF scale\nfix", "lab acceptance\n(CM-cut NF)",
                 "CM-fail\nlab-pass tail", "threshold\nwedge", "corrected\nNF"]
    waterfall(axes[1], nf_labels, nf_start, nf_deltas, nf_final,
              NF_COLOR, "Conditional NF")

    fig.suptitle("Correction budget: from last week's 13,700 pb to the native 9,142 pb",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"correction_budget.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote verification/correction_budget.png/pdf")


if __name__ == "__main__":
    main()
