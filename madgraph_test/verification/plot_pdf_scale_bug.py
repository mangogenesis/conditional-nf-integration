#!/usr/bin/env python3
"""Show the Q vs Q^2 PDF-scale bug and its effect on the totals.

The old Python convolution called ``pdf.xfxQ(pid, x, q2)`` with
``q2 = 91.188**2``.  LHAPDF's ``xfxQ`` takes Q in GeV, so the PDFs were
evaluated at Q = 8315.25 GeV.  This figure shows the resulting DGLAP tilt and
the old-vs-corrected totals.
"""
from __future__ import annotations

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
Q_WRONG = MZ * MZ
X = np.logspace(-5.0, -0.30, 260)

OLD = "#9a9a9a"
NEW = "#2a78d6"
GLUON = "#2a78d6"
UP = "#eb6834"
DOWN = "#1baf7a"


def main() -> None:
    pdf = ip.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)

    fig = plt.figure(figsize=(14.0, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.30)

    # ------------------------------------------------------------------ PDFs
    ax = fig.add_subplot(gs[0, 0])
    for pid, label, color in ((21, "gluon", GLUON), (2, r"$u$ quark", UP)):
        good = np.array([pdf.xfxQ(pid, float(x), MZ) for x in X])
        wrong = np.array([pdf.xfxQ(pid, float(x), Q_WRONG) for x in X])
        ax.plot(X, good, color=color, lw=2.0, label=f"{label}, Q=91.188 GeV")
        ax.plot(X, wrong, color=color, lw=2.0, ls="--",
                label=f"{label}, Q=8,315 GeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 0.5)
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel(r"parton momentum fraction $x$")
    ax.set_ylabel(r"$x f(x,Q)$")
    ax.set_title("PDFs at the two scales")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    # ----------------------------------------------------------------- ratio
    ax = fig.add_subplot(gs[0, 1])
    for pid, label, color in ((21, "gluon", GLUON), (2, r"$u$", UP),
                              (1, r"$d$", DOWN)):
        good = np.array([pdf.xfxQ(pid, float(x), MZ) for x in X])
        wrong = np.array([pdf.xfxQ(pid, float(x), Q_WRONG) for x in X])
        ratio = wrong / good
        ax.plot(X, ratio, color=color, lw=2.0, label=label)
    ax.axhline(1.0, color="0.35", lw=0.9, ls=":")
    ax.axvspan(0.008, 0.15, color="#999999", alpha=0.13, lw=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 0.5)
    ax.set_ylim(0.2, 2.3)
    ax.set_xlabel(r"parton momentum fraction $x$")
    ax.set_ylabel("wrong scale / correct scale")
    ax.set_title(r"Wrong-scale / correct-scale PDF ratio")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.text(0.009, 0.28,
            "13 TeV $Z+j$\n$x\\approx0.008$–$0.15$",
            fontsize=7.5, color="#444444")

    # ---------------------------------------------------------------- totals
    ax = fig.add_subplot(gs[0, 2])
    scenarios = ["no $\\eta$\n(CM scan)", "CM\n$|\\eta|<2.5$", "lab\n$|\\eta|<2.5$"]
    old = [13866.59, 13761.97, 10599.26]
    new = [11876.96, 11753.90, 9142.25]
    x = np.arange(len(scenarios))
    width = 0.34
    ax.bar(x - width / 2, old, width, color=OLD, label="old (wrong $Q$)")
    ax.bar(x + width / 2, new, width, color=NEW, label="corrected ($Q=M_Z$)")
    for xi, v in zip(x - width / 2, old):
        ax.text(xi, v - 420, f"{v:,.0f}", ha="center", va="top",
                fontsize=7.2, color="white")
    for xi, v in zip(x + width / 2, new):
        ax.text(xi, v - 420, f"{v:,.0f}", ha="center", va="top",
                fontsize=7.2, color="white")
    for xi, o, n in zip(x, old, new):
        ax.text(xi, max(o, n) + 650, f"{(n/o-1)*100:+.1f}%", ha="center",
                fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("cross section (pb)")
    ax.set_ylim(0, 17200)
    ax.set_title("Old vs corrected totals")
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.98, "gray: old ($Q=8{,}315$ GeV)\nblue: corrected ($Q=M_Z$)",
            transform=ax.transAxes, fontsize=7.2, color="#333333", va="top")

    fig.suptitle("The dominant bug: PDFs evaluated at $Q=8{,}315$ GeV instead of $Q=M_Z$",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"pdf_scale_bug.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote verification/pdf_scale_bug.png/pdf")


if __name__ == "__main__":
    main()
