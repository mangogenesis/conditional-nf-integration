#!/usr/bin/env python3
"""Visualise the lab-frame NF: per-channel acceptance and the final budget."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]
LABELS = {
    "gu_zu": r"$gu\to Zu$",
    "gd_zd": r"$gd\to Zd$",
    "gux_zux": r"$g\bar u\to Z\bar u$",
    "gdx_zdx": r"$g\bar d\to Z\bar d$",
    "uux_zg": r"$u\bar u\to Zg$",
    "ddx_zg": r"$d\bar d\to Zg$",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NF = "#eb6834"
EXACT = "#2a78d6"
NATIVE = "#1baf7a"


def main() -> None:
    rows = list(csv.DictReader((HERE / "nf_lab_acceptance.csv").open(
        newline="", encoding="utf-8")))
    by = defaultdict(list)
    for row in rows:
        by[row["channel"]].append((float(row["energy_gev"]),
                                   float(row["f_lab_given_cm"]),
                                   float(row["error"])))
    summary = json.loads((HERE / "nf_lab_total.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---------------------------------------------------------- acceptance
    ax = axes[0]
    for ch, color in zip(CHANNELS, COLORS):
        pts = sorted(by[ch])
        e = np.array([p[0] for p in pts])
        f = np.array([p[1] for p in pts])
        ax.plot(e, f, color=color, lw=1.7, marker="o", ms=2.8, label=LABELS[ch])
    ax.set_xscale("log")
    ax.set_xlim(114, 2000)
    ax.set_ylim(0.55, 0.92)
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel(r"$f_{\rm lab|CM}(\sqrt{\hat s})$")
    ax.set_title("Lab acceptance of the CM-cut NF sample", fontsize=10)
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="lower right")
    ax.text(0.02, 0.96,
            "both beam orderings; $Y$ drawn from the PDF luminosity",
            transform=ax.transAxes, fontsize=7.2, color="#555555", va="top")

    # ------------------------------------------------------------- budget
    ax = axes[1]
    start = summary["nf_lab_cm_cut_pb"]
    tail = summary["cm_fail_lab_pb"]
    threshold = summary["threshold_pb"]
    total = summary["nf_lab_total_pb"]
    baseline = 9000.0
    xs = np.arange(4)
    # bars: start, +tail, +threshold, final
    ax.bar(0, start - baseline, bottom=baseline, color=NF, width=0.62,
           alpha=0.85, zorder=2)
    ax.bar(1, tail, bottom=start, color=NF, width=0.62, alpha=0.85, zorder=2)
    ax.bar(2, threshold, bottom=start + tail, color=NF, width=0.62,
           alpha=0.85, zorder=2)
    ax.bar(3, total - baseline, bottom=baseline, color=NF, width=0.62,
           alpha=0.85, zorder=2)
    labels = [f"{start:,.2f}", f"+{tail:.2f}", f"+{threshold:.2f}", f"{total:,.2f}"]
    for x, lab in zip(xs, labels):
        if x == 1:
            # Put the tail label inside its bar so it does not collide with
            # the exact/native horizontal lines.
            ax.text(x, start + 0.5 * tail, lab, ha="center", va="center",
                    fontsize=8, color="white")
            continue
        if x == 0:
            y = start
        elif x == 2:
            y = start + tail + threshold
        else:
            y = total
        ax.text(x, y + 7, lab, ha="center", va="bottom", fontsize=8)
    ax.axhline(summary["exact_lab_pb"], color=EXACT, lw=1.6, ls="--", zorder=6,
               label=f"exact {summary['exact_lab_pb']:,.2f} pb")
    ax.axhline(summary["native_lab_pb"], color=NATIVE, lw=1.6, ls=":", zorder=6,
               label=(f"native {summary['native_lab_pb']:,.1f} "
                      f"$\\pm$ {summary['native_lab_error_pb']:.1f} pb"))
    ax.axhspan(summary["native_lab_pb"] - summary["native_lab_error_pb"],
               summary["native_lab_pb"] + summary["native_lab_error_pb"],
               color=NATIVE, alpha=0.12, lw=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["NF lab\n(CM cut)", "CM-fail\nlab-pass tail",
                        "threshold\nwedge", "NF lab\ntotal"], fontsize=8)
    ax.set_ylim(baseline, 9200)
    ax.set_ylabel("cross section (pb)")
    ax.set_title("From the CM-cut NF to the lab-frame total", fontsize=10)
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.suptitle(f"Lab-frame NF: {total:,.2f} $\\pm$ "
                 f"{summary['nf_lab_total_error_pb']:.2f} pb "
                 f"({(total/summary['native_lab_pb']-1)*100:+.3f}% vs native)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"nf_lab_frame.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote verification/nf_lab_frame.png/pdf")


if __name__ == "__main__":
    main()
