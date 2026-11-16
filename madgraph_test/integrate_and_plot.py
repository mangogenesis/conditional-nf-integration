#!/usr/bin/env python3
"""Convolve the continuous partonic no-eta scan with LHAPDF and plot it.

This script is deliberately independent of the NF repository: it reads only
MadGraph outputs and LHAPDF.  It also reads the previous *with-eta-cut*
partonic MadEvent comparison CSVs for the overlay; those are MadGraph outputs
from the earlier scan.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = ROOT / "conditional-nf-integration"
LHAPDF_PY = ROOT / "lhapdf-local" / "python"
LHAPDF_DATA = ROOT / "lhapdf-local" / "share" / "LHAPDF"
if "LHAPDF_DATA_PATH" not in os.environ and LHAPDF_DATA.is_dir():
    os.environ["LHAPDF_DATA_PATH"] = str(LHAPDF_DATA).replace("\\", "/")
try:
    import lhapdf  # noqa: E402
except ImportError:
    if LHAPDF_PY.is_dir():
        sys.path.insert(0, str(LHAPDF_PY))
    import lhapdf  # noqa: E402

CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]
PAIRS = {
    "gu_zu": [[21, 2], [21, 4]],
    "gd_zd": [[21, 1], [21, 3]],
    "gux_zux": [[21, -2], [21, -4]],
    "gdx_zdx": [[21, -1], [21, -3]],
    "uux_zg": [[2, -2], [4, -4]],
    "ddx_zg": [[1, -1], [3, -3]],
}
CHANNEL_LABELS = {
    "gu_zu": "g(u+c) → Z(u+c)",
    "gd_zd": "g(d+s) → Z(d+s)",
    "gux_zux": "g(u~+c~) → Z(u~+c~)",
    "gdx_zdx": "g(d~+s~) → Z(d~+s~)",
    "uux_zg": "(u+c)(u~+c~) → Zg",
    "ddx_zg": "(d+s)(d~+s~) → Zg",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def luminosity(pdf, sqrts_hat, sqrts, q2, pairs, y_nodes, y_weights):
    s = sqrts * sqrts
    shat = sqrts_hat * sqrts_hat
    xroot = math.sqrt(shat / s)
    ymax = 0.5 * math.log(s / shat)
    if ymax <= 0:
        return 0.0
    y = ymax * y_nodes
    weights = ymax * y_weights
    total = 0.0
    for first, second in pairs:
        for left, right in ((first, second), (second, first)):
            x1 = xroot * np.exp(y)
            x2 = xroot * np.exp(-y)
            xf1 = np.array([pdf.xfxQ2(int(left), float(x), q2) for x in x1])
            xf2 = np.array([pdf.xfxQ2(int(right), float(x), q2) for x in x2])
            total += float(np.sum(weights * (xf1 / x1) * (xf2 / x2)))
    return (2.0 * sqrts_hat / s) * total


def trapz(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return float(np.sum(w * y))


def read_scan(path: Path):
    data = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ch = row["channel"]
            data.setdefault(ch, []).append(
                (float(row["energy_gev"]), float(row["sigma_pb"]), float(row["error_pb"]))
            )
    return {ch: sorted(v) for ch, v in data.items()}


def read_exact_with_cuts():
    """Read the previous MadEvent with-etaj comparison scans, continuous."""
    out = {ch: [] for ch in CHANNELS}
    groups = {ch: {} for ch in CHANNELS}
    for ch in CHANNELS:
        path = UPSTREAM / "results/zj_hadronic_madevent_interpolation" / ch / "comparison.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                e = float(row["energy_gev"])
                groups[ch].setdefault(e, []).append(
                    (float(row["madevent_pb"]), float(row["madevent_error_pb"]))
                )
        # comparison.csv contains a low scan [114,250] and a high scan
        # [200,2000].  Use the low scan below 250 and the high scan above it.
        low = [(e, sum(v for v, _ in rows)) for e, rows in groups[ch].items() if e <= 250]
        high = [(e, sum(v for v, _ in rows)) for e, rows in groups[ch].items() if e > 250]
        out[ch] = sorted(low + high)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=HERE / "results/scan_noeta.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scan = read_scan(args.scan)
    exact = read_exact_with_cuts()
    pdf = lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    q2 = 91.188 ** 2
    sqrts = 13000.0
    y_nodes, y_weights = np.polynomial.legendre.leggauss(200)

    # Per-channel convolution and total.
    rows = []
    total_value = 0.0
    total_err2 = 0.0
    for ch in CHANNELS:
        energies = np.array([e for e, _, _ in scan[ch]])
        sigma = np.array([s for _, s, _ in scan[ch]])
        err = np.array([e for _, _, e in scan[ch]])
        lum = np.array([luminosity(pdf, e, sqrts, q2, PAIRS[ch], y_nodes, y_weights) for e in energies])
        weights = np.zeros_like(energies)
        weights[1:-1] = 0.5 * (energies[2:] - energies[:-2])
        weights[0] = 0.5 * (energies[1] - energies[0])
        weights[-1] = 0.5 * (energies[-1] - energies[-2])
        value = float(np.sum(weights * lum * sigma))
        error = float(np.sqrt(np.sum((weights * lum * err) ** 2)))
        total_value += value
        total_err2 += error ** 2
        rows.append({
            "channel": ch,
            "integral_pb": value,
            "error_pb": error,
            "n_points": len(energies),
        })
        print(f"{ch:8s} {value:10.2f} +/- {error:.4f} pb")

    total_error = math.sqrt(total_err2)
    print(f"TOTAL    {total_value:.2f} +/- {total_error:.4f} pb")
    with (args.output_dir / "total_integral.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["channel", "integral_pb", "error_pb", "n_points"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"channel": "total", "integral_pb": total_value,
                         "error_pb": total_error, "n_points": sum(r["n_points"] for r in rows)})

    # A direct convolution of the previous with-eta-cut scan, same code.
    exact_total = 0.0
    exact_err2 = 0.0
    for ch in CHANNELS:
        energies = np.array([e for e, _ in exact[ch]])
        sigma = np.array([s for _, s in exact[ch]])
        lum = np.array([luminosity(pdf, e, sqrts, q2, PAIRS[ch], y_nodes, y_weights) for e in energies])
        w = np.zeros_like(energies)
        w[1:-1] = 0.5 * (energies[2:] - energies[:-2])
        w[0] = 0.5 * (energies[1] - energies[0])
        w[-1] = 0.5 * (energies[-1] - energies[-2])
        exact_total += float(np.sum(w * lum * sigma))
    print(f"PREVIOUS WITH-CUTS TOTAL {exact_total:.2f} pb")

    # The same no-eta 50-point grid applied to the with-cut exact curve.
    # This separates the effect of eta acceptance from the effect of the
    # coarse low-energy grid.
    same_grid_total = 0.0
    for ch in CHANNELS:
        e_scan = np.array([e for e, _, _ in scan[ch]])
        e_exact = np.array([e for e, _ in exact[ch]])
        s_exact = np.array([s for _, s in exact[ch]])
        s_interp = np.interp(e_scan, e_exact, s_exact)
        lum = np.array([
            luminosity(pdf, e, sqrts, q2, PAIRS[ch], y_nodes, y_weights)
            for e in e_scan
        ])
        w = np.zeros_like(e_scan)
        w[1:-1] = 0.5 * (e_scan[2:] - e_scan[:-2])
        w[0] = 0.5 * (e_scan[1] - e_scan[0])
        w[-1] = 0.5 * (e_scan[-1] - e_scan[-2])
        same_grid_total += float(np.sum(w * lum * s_interp))
    print(f"WITH-CUTS ON SAME 50-POINT GRID {same_grid_total:.2f} pb")

    summary = {
        "scan": str(args.scan),
        "energy_range_gev": [float(min(e for e, _, _ in scan[CHANNELS[0]])),
                             float(max(e for e, _, _ in scan[CHANNELS[0]]))],
        "total_noeta_pb": total_value,
        "total_noeta_error_pb": total_error,
        "total_with_cuts_pb": exact_total,
        "total_with_cuts_same_grid_pb": same_grid_total,
        "nf_reference_pb": 11752.4414,
        "nf_reference_error_pb": 0.8248,
        "native_noeta_pb": 11913.0,
        "native_noeta_error_pb": 10.9,
        "native_both_cuts_pb": 9142.3,
        "native_both_cuts_error_pb": 6.8,
        "pdf_scale": "xfxQ2 at Q=MZ; corrected 2026-11-10",
    }
    (args.output_dir / "total_integral.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # Figure 1: per-energy partonic cross section, no eta vs with eta.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, ch in enumerate(CHANNELS):
        e = [x for x, _, _ in scan[ch]]
        s = [x for _, x, _ in scan[ch]]
        ax.plot(e, s, color=COLORS[i], lw=1.1, alpha=0.55,
                label=CHANNEL_LABELS[ch])
    e_sum = [x for x, _, _ in scan["sum"]]
    s_sum = [x for _, x, _ in scan["sum"]]
    ax.plot(e_sum, s_sum, color="black", lw=2.2, label="six-channel sum, no eta")
    e_cut = [x for x, _ in exact["gu_zu"]]
    s_cut = []
    for e in e_cut:
        s_cut.append(sum(dict((x, v) for x, v in exact[ch]).get(e, 0.0) for ch in CHANNELS))
    ax.plot(e_cut, s_cut, color="#555555", lw=1.6, ls="--",
            label="six-channel sum, |eta|<2.5 (fine low/high)")

    # The with-cut curve interpolated onto the same coarse 50-point grid.
    s_cut_same = []
    for e in e_sum:
        total = 0.0
        for ch in CHANNELS:
            xe = [x for x, _ in exact[ch]]
            ys = [s for _, s in exact[ch]]
            total += float(np.interp(e, xe, ys))
        s_cut_same.append(total)
    ax.plot(e_sum, s_cut_same, color="#2a78d6", lw=1.4, ls=":",
            label="six-channel sum, |eta|<2.5 (same 50-grid)")
    ax.set_xlim(114, 2000)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel(r"partonic $\sigma$ (pb)")
    ax.set_title("MadGraph partonic Z + j: no eta cut vs |eta| < 2.5")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"partonic_cross_section_vs_energy.{suffix}", dpi=180)
    plt.close(fig)

    # Figure 2: hadronic integrand L * sigma vs energy.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    energies = np.array(e_sum)
    sigma = np.array(s_sum)
    lum_sum = np.zeros_like(energies)
    for ch in CHANNELS:
        lum = np.array([
            luminosity(pdf, e, sqrts, q2, PAIRS[ch], y_nodes, y_weights)
            for e in energies
        ])
        # The no-eta sum is a sum of channels; use its own luminosity factor
        # only illustratively by adding each channel's L*sigma.
        scan_ch = np.array([s for _, s, _ in scan[ch]])
        lum_sum += lum * scan_ch
    ax.plot(energies, lum_sum, color="#2a78d6", lw=1.8)
    ax.set_xlim(114, 2000)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel(r"$dL/d\sqrt{\hat s}\times\sigma$ (pb GeV$^{-1}$)")
    ax.set_title("Hadronic integrand of the no-eta MadGraph scan")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"hadronic_integrand_vs_energy.{suffix}", dpi=180)
    plt.close(fig)

    # Figure 3: total comparison.
    labels = ["MadGraph\nno eta (fine)", "MadGraph\nCM cuts (fine)",
              "NF\nCM cuts", "NF\nlab cuts (event-level)",
              "native\nno eta", "native\nlab cuts"]
    values = [total_value, exact_total, 11752.4414, 9155.14, 11913.0, 9142.3]
    errors = [total_error, 0.0, 0.8248, 8.92, 10.9, 6.8]
    colors = ["#2a78d6", "#78a9e3", "#eb6834", "#eda100", "#1baf7a", "#008300"]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(values))
    ax.bar(x, values, yerr=errors, color=colors, width=0.62, capsize=3)
    for xi, v in zip(x, values):
        ax.text(xi, v + 180, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("cross section (pb)")
    ax.set_ylim(0, 13800)
    ax.set_title("Total integrated $Z+j$: partonic, NF, and native in their correct frames")
    ax.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"total_comparison.{suffix}", dpi=180)
    plt.close(fig)

    print("wrote plots and total_integral.csv/json")


if __name__ == "__main__":
    main()
