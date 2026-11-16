#!/usr/bin/env python3
"""Systematic diagnostic of the native/partonic normalization gap.

Steps
1. Audit LHE event headers (SCALUP, AQEDUP, AQCDUP), run cards, and param cards.
2. Check the subprocess / flavor / symmetry mapping between native and partonic.
3. Compare the native and partonic differential cross sections bin by bin and
   per channel.
4. Quantify the kinematic threshold 113.356 GeV and the missing 113.356-114
   piece from the threshold scan.

Run from madgraph_test with LHAPDF_DATA_PATH set externally.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]
CHANNEL_LABELS = {
    "gu_zu": r"$gu\to Zu$",
    "gd_zd": r"$gd\to Zd$",
    "gux_zux": r"$g\bar u\to Z\bar u$",
    "gdx_zdx": r"$g\bar d\to Z\bar d$",
    "uux_zg": r"$u\bar u\to Zg$",
    "ddx_zg": r"$d\bar d\to Zg$",
}


def read_card(path: Path) -> dict:
    """Parse a run card (value = name)."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.split("#")[0].strip()
        if "=" not in text:
            continue
        left, right = text.split("=", 1)
        fields = right.split()
        if fields:
            out[fields[0].strip()] = left.strip()
    return out


def read_param_card(path: Path) -> dict:
    """Parse a param card (id value # name)."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "#" not in line:
            continue
        left, comment = line.split("#", 1)
        fields = left.split()
        name = comment.strip().split()[0] if comment.strip() else ""
        if fields and name:
            out[name] = fields[-1].strip()
    return out


def lHe_headers() -> dict:
    """Extract the first 20 event-header lines per LHE file."""
    out = {}
    for path in sorted((ROOT / "native_zj_nocuts" / "SubProcesses").glob("P*/G*/events.lhe")):
        rows = []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for j, line in enumerate(lines):
            if line.strip() == "<event>":
                fields = lines[j + 1].split()
                if len(fields) >= 6:
                    rows.append((float(fields[3]), float(fields[4]), float(fields[5])))
                if len(rows) >= 20:
                    break
        out[str(path.relative_to(ROOT))] = rows
    return out


def step1() -> None:
    print("=" * 78)
    print("STEP 1 - LHE headers, scales, couplings")
    print("=" * 78)
    native_param = read_param_card(ROOT / "native_zj_nocuts" / "Cards" / "param_card.dat")
    part_param = read_param_card(ROOT / "partonic_zj" / "Cards" / "param_card.dat")
    native_run = read_card(ROOT / "native_zj_nocuts" / "Cards" / "run_card.dat")
    part_run = read_card(ROOT / "partonic_zj" / "Cards" / "run_card.dat")
    print("\nparam_card:")
    for key in ("MZ", "WZ", "aEWM1", "Gf", "aS"):
        print(f"  {key:6s} native={native_param.get(key, '?'):>14s}  partonic={part_param.get(key, '?'):>14s}")
    print("\nrun_card:")
    for key in ("lpp1", "lpp2", "ebeam1", "ebeam2", "pdlabel1", "pdlabel2",
                "lhaid", "scale", "dsqrt_q2fact1", "dsqrt_q2fact2",
                "dsqrt_shat", "ptj", "etaj", "maxjetflavor"):
        print(f"  {key:14s} native={native_run.get(key, '?'):>12s}  partonic={part_run.get(key, '?'):>12s}")

    print("\nLHE event headers (SCALUP, AQEDUP, AQCDUP), first 20 events/file:")
    headers = lHe_headers()
    for name, rows in headers.items():
        if not rows:
            print(f"  {name}: no event headers")
            continue
        arr = np.array(rows)
        print(f"  {name}: n={len(rows)} unique scale={np.unique(arr[:,0])} "
              f"aqed={np.unique(arr[:,1])} aqcd={np.unique(arr[:,2])}")
    # check whether any <init> block exists
    print("\nLHE <init> blocks:")
    for path in sorted((ROOT / "native_zj_nocuts" / "SubProcesses").glob("P*/G*/events.lhe")):
        text = path.read_text(encoding="utf-8", errors="replace")
        print(f"  {path.relative_to(ROOT)}: {'yes' if '<init>' in text else 'NO (<init> absent)'}")

    mz = float(native_param.get("MZ", 91.188))
    pt = float(native_run.get("ptj", 20.0))
    threshold = math.sqrt(mz * mz + pt * pt) + pt
    print("\nkinematic threshold:")
    print(f"  MZ={mz}, ptj={pt}: sqrt(MZ^2+ptj^2)+ptj = {threshold:.6f} GeV")
    print(f"  native dsqrt_shat = {native_run.get('dsqrt_shat', '?')} GeV")
    print(f"  partonic scan starts at 114.0 GeV; missing interval = {114.0 - threshold:.6f} GeV")
    print(f"  native dsqrt_shat is below the true threshold, so it does not cut phase space.")


def parse_processes(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            out.append(text)
    return out


def step2() -> None:
    print()
    print("=" * 78)
    print("STEP 2 - subprocess, flavor, symmetry mapping")
    print("=" * 78)
    print("\npartonic processes.dat:")
    for i in range(1, 7):
        p = ROOT / "partonic_zj" / "SubProcesses" / f"P{i}_" / "processes.dat"
        # find the actual directory (P1_gq_zq, ...)
        matches = list((ROOT / "partonic_zj" / "SubProcesses").glob(f"P{i}_*/processes.dat"))
        if not matches:
            continue
        print(f"  P{i}: {matches[0].parent.name}")
        for line in parse_processes(matches[0]):
            print(f"      {line}")
    print("\nnative processes.dat:")
    for name in ("P1_gq_zq", "P1_qq_zg"):
        path = ROOT / "native_zj_nocuts" / "SubProcesses" / name / "processes.dat"
        print(f"  {name}:")
        for line in parse_processes(path):
            print(f"      {line}")
    print("\nsymfact.dat:")
    for name in ("P5_qq_zg", "P6_qq_zg"):
        path = ROOT / "partonic_zj" / "SubProcesses" / name / "symfact.dat"
        print(f"  partonic {name}: {parse_processes(path)}")
    for name in ("P1_gq_zq", "P1_qq_zg"):
        path = ROOT / "native_zj_nocuts" / "SubProcesses" / name / "symfact.dat"
        print(f"  native {name}: {parse_processes(path)}")

    # Per-channel cumulative ratios using the official partonic integrals.
    events = list(csv.DictReader((HERE / "events_flat.csv").open()))
    w = np.array([float(r["weight"]) for r in events])
    ch = np.array([r["channel"] for r in events])
    sh = np.array([float(r["shat"]) for r in events])
    part = {}
    with (HERE / "channel_integrands.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_ch = defaultdict(dict)
    for r in rows:
        by_ch[r["channel"]][float(r["energy_gev"])] = float(r["integrand_pb_per_gev"])
    for c in CHANNELS:
        e = np.array(sorted(by_ch[c]))
        y = np.array([by_ch[c][x] for x in e])
        part[c] = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(e)))
    print("\ncumulative per-channel native/partonic ratios (sqrt(s-hat) >= 114):")
    native_total = 0.0
    part_total = 0.0
    for c in CHANNELS:
        nat = float(w[(ch == c) & (sh >= 114.0)].sum())
        ratio = nat / part[c]
        native_total += nat
        part_total += part[c]
        print(f"  {c:8s} native={nat:9.2f}  partonic={part[c]:9.2f}  ratio={ratio:.4f}")
    print(f"  {'total':8s} native={native_total:9.2f}  partonic={part_total:9.2f}  "
          f"ratio={native_total/part_total:.4f}")
    gq = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx"]
    qq = ["uux_zg", "ddx_zg"]
    for label, group in (("gq", gq), ("qqbar", qq)):
        nat = float(w[(np.isin(ch, group)) & (sh >= 114.0)].sum())
        par = sum(part[c] for c in group)
        print(f"  {label:8s} native={nat:9.2f}  partonic={par:9.2f}  ratio={nat/par:.4f}")


def load_integrands():
    data = defaultdict(dict)
    with (HERE / "channel_integrands.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data[row["channel"]][float(row["energy_gev"])] = float(row["integrand_pb_per_gev"])
    e = np.array(sorted(data[CHANNELS[0]]))
    return e, {c: np.array([data[c][x] for x in e]) for c in CHANNELS}


def step3() -> None:
    print()
    print("=" * 78)
    print("STEP 3 - differential ratio native/partonic")
    print("=" * 78)
    events = list(csv.DictReader((HERE / "events_flat.csv").open()))
    w = np.array([float(r["weight"]) for r in events])
    sh = np.array([float(r["shat"]) for r in events])
    ch = np.array([r["channel"] for r in events])
    e, vals = load_integrands()
    fine = np.arange(114.0, 2000.0 + 0.1, 0.1)
    fvals = {c: np.interp(fine, e, vals[c]) for c in CHANNELS}

    bins = [(114, 150), (150, 200), (200, 250), (250, 300), (300, 400),
            (400, 500), (500, 700), (700, 1000), (1000, 2000)]
    rows = []
    print(f"\n{'bin':>12s} " + "".join(f"{c:>10s}" for c in CHANNELS) + f"{'all':>10s}")
    for lo, hi in bins:
        m = (sh >= lo) & (sh < hi)
        part = {}
        for c in CHANNELS:
            mm = (fine >= lo) & (fine <= hi)
            x = fine[mm]
            y = fvals[c][mm]
            part[c] = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) if len(x) > 1 else 0.0
        native_all = float(w[m].sum())
        part_all = sum(part.values())
        ratios = {}
        for c in CHANNELS:
            nat = float(w[m & (ch == c)].sum())
            ratios[c] = nat / part[c] if part[c] > 0 else float("nan")
            rows.append({"bin_lo": lo, "bin_hi": hi, "channel": c,
                         "native_pb": nat, "partonic_pb": part[c],
                         "ratio": ratios[c]})
        rows.append({"bin_lo": lo, "bin_hi": hi, "channel": "all",
                     "native_pb": native_all, "partonic_pb": part_all,
                     "ratio": native_all / part_all if part_all > 0 else float("nan")})
        print(f"{lo:5.0f}-{hi:5.0f} " +
              "".join(f"{ratios[c]:10.4f}" if np.isfinite(ratios[c]) else f"{'--':>10s}" for c in CHANNELS) +
              f"{native_all/part_all:10.4f}")

    out = HERE / "normalization_gap.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")

    # Plot: ratio vs sqrt(s-hat), total and groups.
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                  gridspec_kw={"width_ratios": [2.0, 1.0]})
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    # Re-read rows to plot
    data = defaultdict(list)
    for r in rows:
        center = 0.5 * (float(r["bin_lo"]) + float(r["bin_hi"]))
        data[r["channel"]].append((center, float(r["ratio"])))
    for c, color in zip(CHANNELS, colors):
        pts = sorted(data[c])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3.5,
                lw=1.1, color=color, label=CHANNEL_LABELS[c])
    pts = sorted(data["all"])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", ms=5,
            lw=2.0, color="black", label="all channels")
    ax.axhline(1.0, color="0.35", lw=0.8, ls=":")
    ax.axhspan(0.95, 1.05, color="#1baf7a", alpha=0.10, lw=0)
    ax.set_xscale("log")
    ax.set_xlim(114, 500)
    ax.set_ylim(0.85, 1.10)
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel("native / partonic")
    ax.set_title("Corrected differential ratio (114–500 GeV)")
    ax.grid(True, which="both", color="black", alpha=0.10, lw=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.05, "700–2000 GeV: statistics-limited, omitted",
            transform=ax.transAxes, ha="right", fontsize=7, color="#777777")

    # Cumulative bars
    gq = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx"]
    qq = ["uux_zg", "ddx_zg"]
    events_ch = ch
    native = {}
    part = {}
    for group, label in ((gq, "gq"), (qq, "qqbar")):
        native[label] = float(w[np.isin(events_ch, group) & (sh >= 114.0)].sum())
        part[label] = sum(
            float(np.sum(0.5 * (vals[c][1:] + vals[c][:-1]) * np.diff(e))) for c in group
        )
    labels = ["gq", "qqbar", "all"]
    native["all"] = float(w[sh >= 114.0].sum())
    part["all"] = sum(part[k] for k in ("gq", "qqbar"))
    vals_bar = [native[k] / part[k] for k in labels]
    axb.bar(labels, [v - 0.95 for v in vals_bar], bottom=0.95,
            color=["#2a78d6", "#eb6834", "black"], width=0.6)
    for i, v in enumerate(vals_bar):
        axb.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    axb.axhline(1.0, color="0.35", lw=0.8, ls=":")
    axb.set_ylim(0.95, 1.05)
    axb.set_ylabel("native / partonic")
    axb.set_title("Cumulative by group (corrected)")
    axb.grid(True, axis="y", color="black", alpha=0.10, lw=0.6)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"normalization_gap.{suffix}", dpi=180, bbox_inches="tight")
    print(f"wrote {HERE / 'normalization_gap.png'}")


def threshold() -> None:
    print()
    print("=" * 78)
    print("THRESHOLD - 113.356 to 114 GeV")
    print("=" * 78)
    import integrate_and_plot as ip

    native_param = read_param_card(ROOT / "native_zj_nocuts" / "Cards" / "param_card.dat")
    native_run = read_card(ROOT / "native_zj_nocuts" / "Cards" / "run_card.dat")
    mz = float(native_param.get("MZ", 91.188))
    pt = float(native_run.get("ptj", 20.0))
    threshold = math.sqrt(mz * mz + pt * pt) + pt
    print(f"exact threshold: {threshold:.6f} GeV")
    path = HERE / "scan_threshold.csv"
    if not path.exists():
        print("no threshold scan found; skipping numerical integral")
        return
    th = {}
    for r in csv.DictReader(path.open()):
        th.setdefault(float(r["energy_gev"]), {})[r["channel"]] = float(r["sigma_pb"])
    # add the 114 GeV point from the fine scan
    fine = {}
    for r in csv.DictReader((ROOT / "results" / "scan_noeta_fine.csv").open()):
        e = float(r["energy_gev"])
        if abs(e - 114.0) < 1e-6:
            fine[r["channel"]] = float(r["sigma_pb"])
    if fine:
        th[114.0] = fine
    pdf = ip.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    q2 = 91.188 ** 2
    yn, yw = np.polynomial.legendre.leggauss(200)
    energies = np.array([threshold, 113.4, 113.6, 113.8, 114.0])
    integrand = []
    for e in energies:
        if e <= threshold + 1e-9:
            integrand.append(0.0)
            continue
        total = 0.0
        for c in ip.CHANNELS:
            sigma = th[float(e)][c]
            lum = ip.luminosity(pdf, e, 13000.0, q2, ip.PAIRS[c], yn, yw)
            total += lum * sigma
        integrand.append(total)
    integrand = np.array(integrand)
    for e, v in zip(energies, integrand):
        print(f"  {e:8.3f} GeV  dsigma/dsqrt(s) = {v:10.4f} pb/GeV")
    area = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(energies)))
    print(f"threshold contribution ({threshold:.3f}-114): {area:.3f} pb")
    noeta = json.loads(
        (ROOT / "results" / "noeta_fine" / "total_integral.json").read_text()
    )["total_noeta_pb"]
    acc = json.loads(
        (ROOT / "results" / "noeta_fine" / "lab_eta_correction.json").read_text()
    )
    print(f"adjusted no-eta: {noeta + area:.3f} pb")
    print(f"adjusted lab (f_lab=0.71989): {acc['lab_cut_pb'] + area * 0.71989:.3f} pb")
    print(f"adjusted CM (f_CM=1): {acc['cm_cut_pb'] + area:.3f} pb")


def main() -> None:
    step1()
    step2()
    step3()
    threshold()


if __name__ == "__main__":
    main()
