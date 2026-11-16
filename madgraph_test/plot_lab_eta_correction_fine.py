#!/usr/bin/env python3
"""Fine-grid comparison: CM, exact lab, NF lab, and native lab."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

EXACT = "#2a78d6"
NF = "#eb6834"
NATIVE = "#1baf7a"
CM = "#eda100"


def main() -> None:
    nf = json.loads((HERE / "verification" / "nf_lab_total.json").read_text())
    fine = json.loads((RESULTS / "noeta_fine" / "lab_eta_correction.json").read_text())
    exact_lab = nf["exact_lab_pb"]
    nf_lab = nf["nf_lab_total_pb"]
    native = nf["native_lab_pb"]
    native_err = nf["native_lab_error_pb"]
    cm = fine["cm_cut_pb"]

    fig = plt.figure(figsize=(12.4, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.25)

    # ------------------------------------------------------------ overview
    ax = fig.add_subplot(gs[0, 0])
    labels = ["native\nlab", "partonic\nCM", "partonic\nlab", "NF\nlab"]
    values = [native, cm, exact_lab, nf_lab]
    errors = [native_err, 0.0, 0.0, nf["nf_lab_total_error_pb"]]
    colors = [NATIVE, CM, EXACT, NF]
    x = np.arange(len(values))
    ax.bar(x, values, yerr=errors, capsize=4, color=colors, width=0.60)
    for xi, v in zip(x, values):
        ax.text(xi, v + 220, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("total cross section (pb)")
    ax.set_ylim(0, 14000)
    ax.set_title("All fine-grid results", fontsize=10)
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------------- zoom
    ax = fig.add_subplot(gs[0, 1])
    labels = ["native lab", "exact partonic lab", "NF lab"]
    values = [native, exact_lab, nf_lab]
    errors = [native_err, 0.0, nf["nf_lab_total_error_pb"]]
    colors = [NATIVE, EXACT, NF]
    x = np.arange(len(values))
    ax.bar(x, values, yerr=errors, capsize=5, color=colors, width=0.52)
    for xi, v, e in zip(x, values, errors):
        ax.text(xi, v + e + 1.5, f"{v:,.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhspan(native - native_err, native + native_err, color=NATIVE,
               alpha=0.12, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("total cross section (pb)")
    ax.set_ylim(9090, 9185)
    ax.set_title("Zoom: exact and native agree; the NF is 0.14% high", fontsize=10)
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.95,
            f"NF/exact = {nf_lab/exact_lab:.5f}\n"
            f"NF/native = {nf_lab/native:.5f}",
            transform=ax.transAxes, fontsize=8, color="#333333", va="top")

    fig.suptitle("Fine grid: the CM cut is not the lab cut; adding the lab acceptance fixes it",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(RESULTS / f"lab_eta_correction_fine.{suffix}", dpi=180)
    plt.close(fig)
    print("wrote lab_eta_correction_fine.png/pdf")


if __name__ == "__main__":
    main()
