#!/usr/bin/env python3
"""Plot native vs fine-partonic differential cross section and their ratio."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    rows = []
    with (RESULTS / "differential_compare.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "shat": float(row["shat_center_gev"]),
                "native": float(row["native_pb"]),
                "partonic": float(row["partonic_pb"]),
                "ratio": float(row["ratio_native_over_partonic"]),
            })
    rows.sort(key=lambda r: r["shat"])
    # keep bins with a visible partonic and native contribution
    rows = [r for r in rows if r["partonic"] > 0.01 and r["native"] > 0.005]
    x = np.array([r["shat"] for r in rows])
    native = np.array([r["native"] for r in rows])
    partonic = np.array([r["partonic"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax.plot(x, native, color="#2a78d6", marker="o", ms=3.5, lw=1.4,
            label="native LHE (lab, etaj<5)")
    ax.plot(x, partonic, color="#eb6834", marker="s", ms=3.5, lw=1.4,
            label="fine partonic no-eta convolution")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"cross section per 25 GeV bin (pb)")
    ax.set_title(r"Native vs fine partonic $d\sigma/d\sqrt{\hat{s}}$")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # The upper panel keeps all bins; the ratio panel stops at 700 GeV so the
    # statistics-limited tail does not dominate the visual.
    stat = (native > 0.05) & (x < 700)
    xr = x[stat]
    rr = ratio[stat]
    pr = partonic[stat]
    axr.axhline(1.0, color="#555555", lw=0.9, ls="--")
    axr.axhspan(0.95, 1.05, color="#1baf7a", alpha=0.10, lw=0)
    axr.plot(xr, rr, color="#1baf7a", marker="o", ms=3.2, lw=1.3)
    axr.set_xscale("log")
    axr.set_xlim(114, 700)
    axr.set_ylim(0.80, 1.10)
    axr.set_xlabel(r"$\sqrt{\hat{s}}$ bin center (GeV)")
    axr.set_ylabel("native / partonic")
    axr.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(False)
    lines = []
    for lo, hi in ((114, 500), (500, 700)):
        m = (xr >= lo) & (xr < hi)
        if m.any():
            mean = float(np.average(rr[m], weights=pr[m]))
            lines.append(f"{lo}–{hi}: {mean:.3f}")
    lines.append("700–2000: statistics-limited")
    axr.text(0.02, 0.95, "\n".join(lines), transform=axr.transAxes,
             fontsize=7.5, color="#333333", va="top")
    axr.text(0.02, 0.05,
             "ratio panel shown for $\\sqrt{\\hat{s}}<700$ GeV",
             transform=axr.transAxes, fontsize=7.0, color="#777777")

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"differential_compare.{suffix}", dpi=180)
    plt.close(fig)
    print("wrote differential_compare.png/pdf")


if __name__ == "__main__":
    main()
