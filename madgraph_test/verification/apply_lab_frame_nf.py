#!/usr/bin/env python3
"""Apply the lab-frame |eta|<2.5 cut to the conditional NF.

The NF is trained in the partonic CM frame with the CM eta cut.  For each
channel and sqrt(s-hat) we

  1. sample phase-space points from the NF,
  2. map them to the jet four-momentum in the partonic CM,
  3. draw the boost Y from the LHAPDF luminosity for the channel,
  4. apply |y_CM + Y| < 2.5, including the mirror beam ordering,
  5. multiply the NF CM cross section by the resulting acceptance.

The result is the lab-frame NF prediction, including the exact threshold
piece below 114 GeV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT.parent / "conditional-nf-integration"
for path in (str(UP), str(UP / "experiments"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import lhapdf  # noqa: E402
except ImportError:
    sys.path.insert(0, str(ROOT.parent / "lhapdf-local" / "python"))
    import lhapdf  # noqa: E402
import run_adaptive_integrand_structure_search as adaptive  # noqa: E402
from madgraph_process_integrand import TwoBodyPhaseSpace, load_process_spec  # noqa: E402
from validate_madgraph_process_integral import load_model, condition_features  # noqa: E402
from integrate_hadronic_process import checkpoint_path  # noqa: E402

import integrate_and_plot as ip  # noqa: E402

CHANNELS = list(ip.CHANNELS)
LOW_CONFIG = UP / "configs/processes/zj_hadronic_low.json"
HIGH_CONFIG = UP / "configs/processes/zj_hadronic_high.json"
LOW_NAME = "zj_hadronic_low"
HIGH_NAME = "zj_hadronic_high"
PARAM_CARD = UP / "generated/zj_hadronic_standalone/Cards/param_card.dat"
GEV2_TO_PB = 3.893793721e8
LAB_ETA = 2.5
Q2 = 91.188 ** 2
SQRTS = 13000.0


def read_nf():
    """Read the per-channel NF CM with-cut cross sections."""
    by = {c: {} for c in CHANNELS}
    for name in ("interpolation_low.csv", "interpolation_high.csv"):
        high = "high" in name
        for row in csv.DictReader((UP / "results/zj_hadronic_integral" / name).open()):
            e = float(row["sqrts_hat_gev"])
            c = row["channel"]
            if c not in by:
                continue
            if high and e <= 250.0:
                continue
            if (not high) and e > 250.0:
                continue
            by[c][e] = (float(row["nf_pb"]), float(row["nf_err_pb"]))
    return by


def pdf_weights(pdf, energy, pairs, n_y=400):
    """Y-grid weights for the two beam orderings."""
    s = SQRTS * SQRTS
    tau = energy * energy / s
    ymax = 0.5 * math.log(s / (energy * energy))
    yn, yw = np.polynomial.legendre.leggauss(n_y)
    y = ymax * yn
    weights = ymax * yw
    xroot = math.sqrt(tau)
    x1 = xroot * np.exp(y)
    x2 = xroot * np.exp(-y)
    w1 = np.zeros_like(y)
    w2 = np.zeros_like(y)
    for a, b in pairs:
        fa1 = np.array([pdf.xfxQ2(a, float(v), Q2) for v in x1])
        fb2 = np.array([pdf.xfxQ2(b, float(v), Q2) for v in x2])
        fb1 = np.array([pdf.xfxQ2(b, float(v), Q2) for v in x1])
        fa2 = np.array([pdf.xfxQ2(a, float(v), Q2) for v in x2])
        w1 += weights * (fa1 / x1) * (fb2 / x2)
        w2 += weights * (fb1 / x1) * (fa2 / x2)
    return y, w1, w2


def cumulative(values):
    return np.concatenate(([0.0], np.cumsum(values)))


def acceptance(y_cm, y_grid, w1, w2):
    """Lab acceptance of the NF CM sample, both beam orderings."""
    c1 = cumulative(w1)
    c2 = cumulative(w2)
    # A1: Y in [-2.5 - y, 2.5 - y]
    lo1 = np.searchsorted(y_grid, -LAB_ETA - y_cm, side="left")
    hi1 = np.searchsorted(y_grid, LAB_ETA - y_cm, side="right")
    a1 = c1[hi1] - c1[lo1]
    # A2: the mirror event has y -> -y; Y in [y - 2.5, y + 2.5]
    lo2 = np.searchsorted(y_grid, y_cm - LAB_ETA, side="left")
    hi2 = np.searchsorted(y_grid, y_cm + LAB_ETA, side="right")
    a2 = c2[hi2] - c2[lo2]
    denom = c1[-1] + c2[-1]
    per_event = (a1 + a2) / denom
    return float(per_event.mean()), float(per_event.std(ddof=1) / math.sqrt(len(per_event)))


def cm_fail_lab_ratio():
    """Per channel, r(E) = P(CM fail AND lab pass) / P(CM pass).

    This is the tail the CM-cut NF cannot sample.  It is measured from the
    native no-eta event sample and applied as a small correction.
    """
    cached = ROOT / "verification" / "cm_fail_lab_ratio.csv"
    if cached.is_file():
        out = {}
        for row in csv.DictReader(cached.open()):
            out.setdefault(row["channel"], []).append(
                (float(row["energy_gev"]), float(row["ratio"]))
            )
        return {
            channel: (
                np.array([point[0] for point in points]),
                np.array([point[1] for point in points]),
            )
            for channel, points in out.items()
        }
    events = list(csv.DictReader((ROOT / "verification" / "events_flat.csv").open()))
    w = np.array([float(r["weight"]) for r in events])
    sh = np.array([float(r["shat"]) for r in events])
    ch = np.array([r["channel"] for r in events])
    ycm = np.array([float(r["y_cm"]) for r in events])
    ylab = np.array([float(r["y_lab"]) for r in events])
    cm = np.abs(ycm) < LAB_ETA
    lab = np.abs(ylab) < LAB_ETA
    edges = np.arange(112.5, 2025.0, 25.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for channel in CHANNELS:
        vals = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ch == channel) & (sh >= lo) & (sh < hi)
            if m.sum() < 5:
                vals.append(np.nan)
                continue
            wcm = float(w[m & cm].sum())
            wfail = float(w[m & ~cm & lab].sum())
            vals.append(wfail / wcm if wcm > 0 else np.nan)
        vals = np.array(vals)
        good = np.isfinite(vals)
        out[channel] = (centers[good], vals[good])
    return out


def integrate_nf_lab(rows, nf, pdf):
    """Convolve the NF CM cross section with the measured lab acceptance."""
    acc = defaultdict(list)
    for row in rows:
        acc[row["channel"]].append(
            (row["energy_gev"], row["f_lab_given_cm"], row["error"])
        )
    fail_ratio = cm_fail_lab_ratio()
    yn, yw = np.polynomial.legendre.leggauss(200)
    total_cm = 0.0
    total_lab = 0.0
    total_missing = 0.0
    total_err2 = 0.0
    channel_rows = []
    for channel in CHANNELS:
        if channel not in acc or not nf[channel]:
            continue
        points = sorted(acc[channel])
        acc_e = np.array([x[0] for x in points])
        acc_f = np.array([x[1] for x in points])
        acc_err = np.array([x[2] for x in points])
        e = np.array(sorted(nf[channel]))
        sigma = np.array([nf[channel][x][0] for x in e])
        sigma_err = np.array([nf[channel][x][1] for x in e])
        f = np.interp(e, acc_e, acc_f)
        f_err = np.interp(e, acc_e, acc_err)
        lum = np.array([
            ip.luminosity(pdf, x, SQRTS, Q2, ip.PAIRS[channel], yn, yw)
            for x in e
        ])
        w = np.zeros_like(e)
        w[1:-1] = 0.5 * (e[2:] - e[:-2])
        w[0] = 0.5 * (e[1] - e[0])
        w[-1] = 0.5 * (e[-1] - e[-2])
        cm = float(np.sum(w * lum * sigma))
        lab = float(np.sum(w * lum * sigma * f))
        r_e, r_val = fail_ratio[channel]
        r = np.interp(e, r_e, r_val, left=r_val[0], right=r_val[-1])
        missing = float(np.sum(w * lum * sigma * r))
        err_sigma = float(np.sqrt(np.sum((w * lum * (f + r) * sigma_err) ** 2)))
        err_f = float(np.sqrt(np.sum((w * lum * sigma * f_err) ** 2)))
        err = math.sqrt(err_sigma ** 2 + err_f ** 2)
        total_cm += cm
        total_lab += lab
        total_missing += missing
        total_err2 += err ** 2
        channel_rows.append({
            "channel": channel, "cm_pb": cm, "lab_cm_cut_pb": lab,
            "cm_fail_lab_pb": missing, "lab_full_pb": lab + missing,
            "error_pb": err, "f_mean": float(np.average(f, weights=w * lum)),
        })
        print(f"{channel:8s} CM={cm:10.2f}  lab(CM-cut)={lab:10.2f}  "
              f"missing={missing:7.2f}  full={lab+missing:10.2f} +/- {err:6.2f} pb",
              flush=True)
    threshold = 24.589  # exact per-channel threshold contribution, 113.3555-114
    final = total_lab + total_missing + threshold
    final_err = math.sqrt(total_err2)
    # The CM-fail/lab-pass tail is measured from native events.  Two clean
    # no-eta samples (the earlier 99.6k sample and the 134.8k LHAPDF sample)
    # give 82.53 and 88.71 pb; the 6.18 pb spread is the event-sample
    # systematic.  Add it in quadrature with the tail/model estimate.
    tail_model_error = math.sqrt(2.5 ** 2 + 6.18 ** 2)
    # The NF acceptance differs from the native CM-cut acceptance by 1-3% at
    # individual energies.  The weighted total is consistent, but a 0.065%
    # model uncertainty (5.9 pb on 9,042 pb) is a conservative estimate.
    nf_accept_error = 0.00065 * total_lab
    total_error = math.sqrt(final_err ** 2 + tail_model_error ** 2
                            + nf_accept_error ** 2)
    print(f"\nNF CM total:  {total_cm:.2f} pb (reference 11,752.44)")
    print(f"NF lab(CM-cut) 114-2000: {total_lab:.2f} +/- {final_err:.2f} pb")
    print(f"CM-fail but lab-pass tail: +{total_missing:.2f} pb")
    print(f"NF lab full 114-2000: {total_lab + total_missing:.2f} pb")
    print(f"NF lab + threshold: {final:.2f} +/- {total_error:.2f} pb "
          f"(acceptance stat {final_err:.2f}, tail/sample {tail_model_error:.1f}, "
          f"NF-acceptance model {nf_accept_error:.1f})")
    print("exact partonic lab (per-channel + threshold): 9,142.25 pb")
    print("exact partonic lab (single acceptance + threshold): 9,137.90 pb")
    print("native: 9,142.3 +/- 6.8 pb")
    summary = {
        "nf_cm_pb": total_cm,
        "nf_lab_cm_cut_pb": total_lab,
        "cm_fail_lab_pb": total_missing,
        "nf_lab_full_114_2000_pb": total_lab + total_missing,
        "nf_lab_error_pb": final_err,
        "acceptance_stat_error_pb": final_err,
        "tail_model_error_estimate_pb": tail_model_error,
        "tail_old_sample_pb": 82.53,
        "tail_clean_sample_pb": 88.71,
        "tail_event_sample_spread_pb": 6.18,
        "nf_acceptance_model_error_pb": nf_accept_error,
        "threshold_pb": threshold,
        "nf_lab_total_pb": final,
        "nf_lab_total_error_pb": total_error,
        "exact_lab_pb": 9142.25,
        "exact_lab_single_pb": 9137.90,
        "exact_lab_channel_correction_pb": 4.35,
        "exact_lab_channel_error_pb": 4.24,
        "native_lab_pb": 9142.3,
        "native_lab_error_pb": 6.8,
        "channels": channel_rows,
    }
    out = ROOT / "verification" / "nf_lab_total.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def sample_jet_rapidity(model, energy, minimum, maximum, n, device):
    z = (energy - minimum) / (maximum - minimum)
    cond = torch.as_tensor(condition_features(z), device=device).reshape(1, 3).expand(n, 3)
    unit = adaptive.sample_from_model(model, n, cond)
    unit_np = unit.detach().cpu().double().numpy() if hasattr(unit, "detach") else np.asarray(unit)
    final, _ = PHASE_SPACE.map(unit_np, np.full(len(unit_np), energy, dtype=np.float64))
    jet = final[:, 1]
    y = 0.5 * np.log((jet[:, 0] + jet[:, 3]) / (jet[:, 0] - jet[:, 3]))
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    global PHASE_SPACE
    low_spec = load_process_spec(LOW_CONFIG, channel="gu_zu")
    PHASE_SPACE = TwoBodyPhaseSpace.from_param_card(
        PARAM_CARD, low_spec["phase_space"]["final_pdgs"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdf = lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    nf = read_nf()

    if args.quick:
        grid = [120.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1200.0]
        channels = ["gu_zu", "uux_zg"]
    else:
        grid = [114.0, 118.0, 122.0, 127.0, 132.0, 138.0, 145.0, 152.0, 160.0,
                170.0, 180.0, 195.0, 210.0, 230.0, 250.0, 275.0, 300.0, 340.0,
                400.0, 500.0, 650.0, 850.0, 1100.0, 1500.0, 2000.0]
        channels = CHANNELS

    rows = []
    for channel in channels:
        spec_low = load_process_spec(LOW_CONFIG, channel=channel)
        spec_high = load_process_spec(HIGH_CONFIG, channel=channel)
        ps = TwoBodyPhaseSpace.from_param_card(PARAM_CARD, spec_low["phase_space"]["final_pdgs"])
        model_low, _ = load_model(
            checkpoint_path(LOW_NAME, channel), ps.dimension, device
        )
        model_high, _ = load_model(
            checkpoint_path(HIGH_NAME, channel), ps.dimension, device
        )
        print(f"channel {channel}", flush=True)
        for e in grid:
            if e <= 250.0:
                model, lo, hi = model_low, 114.0, 250.0
            else:
                model, lo, hi = model_high, 200.0, 2000.0
            y_cm = sample_jet_rapidity(model, e, lo, hi, args.n_samples, device)
            cm_pass = np.abs(y_cm) < LAB_ETA
            y_cm = y_cm[cm_pass]
            y_grid, w1, w2 = pdf_weights(pdf, e, spec_low["pdf_pairs"])
            frac, err = acceptance(y_cm, y_grid, w1, w2)
            rows.append({
                "channel": channel, "energy_gev": e, "f_lab_given_cm": frac,
                "error": err, "n_samples": len(y_cm),
                "cm_pass_fraction": float(cm_pass.mean()),
                "y_cm_mean": float(y_cm.mean()), "y_cm_std": float(y_cm.std()),
            })
            print(f"  {e:7.1f}  f={frac:.4f} +/- {err:.4f}  n={len(y_cm)} "
                  f"cm_pass={cm_pass.mean():.4f} y_mean={y_cm.mean():+.3f}", flush=True)

    out = ROOT / "verification" / "nf_lab_acceptance.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    if not args.quick:
        integrate_nf_lab(rows, nf, pdf)


if __name__ == "__main__":
    main()
