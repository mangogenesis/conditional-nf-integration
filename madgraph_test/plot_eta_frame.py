#!/usr/bin/env python3
"""Show why the eta cut is frame-dependent: lab frame vs partonic CM."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"


def main() -> None:
    with (RESULTS / "native_eta_sequence.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    native_labels = ["no $\\eta$", "$|\\eta|<5$", "$|\\eta|<2.5$"]
    native_values = [float(row["sigma_pb"]) for row in rows]

    fine = json.loads((RESULTS / "noeta_fine" / "lab_eta_correction.json").read_text())
    nf = json.loads((HERE / "verification" / "nf_lab_total.json").read_text())
    partonic_labels = ["no $\\eta$", "CM $|\\eta|<2.5$", "lab $|\\eta|<2.5$"]
    partonic_values = [fine["no_eta_pb"], fine["cm_cut_pb"], nf["exact_lab_pb"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True)
    colors = [BLUE, BLUE, ORANGE]

    ax = axes[0]
    x = range(len(native_values))
    bars = ax.bar(x, native_values, color=colors, width=0.62)
    for xi, value in zip(x, native_values):
        ax.text(xi, value + 200, f"{value:,.0f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi, value / 2, f"{value/native_values[0]*100:.1f}%", ha="center",
                va="center", fontsize=9, color="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(native_labels, fontsize=9)
    ax.set_title("Native MadGraph (lab-frame $\\eta$)", fontsize=10)
    ax.set_ylabel("total cross section (pb)")
    ax.set_ylim(0, 13500)
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    x = range(len(partonic_values))
    ax.bar(x, partonic_values, color=colors, width=0.62)
    for xi, value in zip(x, partonic_values):
        ax.text(xi, value + 200, f"{value:,.0f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi, value / 2, f"{value/partonic_values[0]*100:.1f}%", ha="center",
                va="center", fontsize=9, color="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(partonic_labels, fontsize=9)
    ax.set_title("Partonic scan + LHAPDF (corrected PDF scale)", fontsize=10)
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("The same $|\\eta|<2.5$ cut removes 23% in the lab and only 1% in the partonic CM",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"eta_frame_effect.{suffix}", dpi=180)
    plt.close(fig)
    print("wrote eta_frame_effect.png/pdf")


if __name__ == "__main__":
    main()
