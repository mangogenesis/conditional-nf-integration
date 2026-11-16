#!/usr/bin/env python3
"""Convergence and independent checks of the parton-luminosity integral."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import integrate_and_plot as ip

SCAN = HERE.parent / "results" / "scan_noeta_fine.csv"
ACC = HERE.parent / "results" / "eta_frame_events.csv"


def load_scan():
    return ip.read_scan(SCAN)


def lum_nodes(pdf, e, ch, nodes):
    yn, yw = np.polynomial.legendre.leggauss(nodes)
    return ip.luminosity(pdf, e, 13000.0, 91.188 ** 2, ip.PAIRS[ch], yn, yw)


def lum_quad(pdf, e, ch):
    """Adaptive quadrature version of the same y integral."""
    s = 13000.0 ** 2
    shat = e * e
    xroot = math.sqrt(shat / s)
    ymax = 0.5 * math.log(s / shat)

    def integrand(y):
        total = 0.0
        for first, second in ip.PAIRS[ch]:
            for left, right in ((first, second), (second, first)):
                x1 = xroot * math.exp(y)
                x2 = xroot * math.exp(-y)
                total += (pdf.xfxQ2(int(left), x1, 91.188 ** 2) / x1) * (
                    pdf.xfxQ2(int(right), x2, 91.188 ** 2) / x2
                )
        return total

    val, err = quad(integrand, -ymax, ymax, epsabs=1e-12, epsrel=1e-10, limit=200)
    return (2.0 * e / s) * val, err


def main() -> None:
    scan = load_scan()
    pdf = ip.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
    acc = list(csv.DictReader(ACC.open()))
    b = np.array([float(r["shat_gev"]) for r in acc])
    fl = np.array([float(r["lab_cut_fraction"]) for r in acc])

    # Totals for different Gauss-Legendre node counts.
    print("nodes   no-eta total (pb)   lab total (pb)")
    for nodes in (10, 20, 40, 80, 160, 200, 320, 640):
        total = 0.0
        total_lab = 0.0
        for ch in ip.CHANNELS:
            e = np.array([x[0] for x in scan[ch]])
            s = np.array([x[1] for x in scan[ch]])
            lum = np.array([lum_nodes(pdf, x, ch, nodes) for x in e])
            integrand = lum * s
            w = np.zeros_like(e)
            w[1:-1] = 0.5 * (e[2:] - e[:-2])
            w[0] = 0.5 * (e[1] - e[0])
            w[-1] = 0.5 * (e[-1] - e[-2])
            f = np.interp(e, b, fl, left=fl[0], right=fl[-1])
            total += float(np.sum(w * integrand))
            total_lab += float(np.sum(w * integrand * f))
        print(f"{nodes:5d}   {total:16.4f}   {total_lab:14.4f}")

    # Adaptive-quadrature cross-check at selected points.
    print("\nadaptive quad vs 200-node Gauss-Legendre")
    print("energy channel   L_quad              L_gl200             rel.diff")
    for e in (114.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0):
        for ch in ("gu_zu", "uux_zg"):
            lq, lerr = lum_quad(pdf, e, ch)
            lg = lum_nodes(pdf, e, ch, 200)
            print(f"{e:7.1f} {ch:6s} {lq:18.12e} {lg:18.12e} {abs(lq-lg)/lg:10.2e}")

    # Exact independent check of the Jacobian:
    #   u = x1, v = x1*x2; delta( sqrt(v*s) - sqrt(shat) ) = 2 sqrt(shat)/s delta(v-tau)
    #   L = (2 sqrt(shat)/s) int_tau^1 du/u [ f_a(u) f_b(tau/u) + f_b(u) f_a(tau/u) ]
    print("\nexact u-integral check (gu_zu)")
    s = 13000.0 ** 2
    q2 = 91.188 ** 2
    for e0 in (200.0, 1000.0):
        tau = e0 * e0 / s
        def integrand(u):
            total = 0.0
            for a, b in ip.PAIRS["gu_zu"]:
                v = tau / u
                if v > 1.0:
                    continue
                total += (1.0 / u) * (pdf.xfxQ2(a, u, q2) / u) * (pdf.xfxQ2(b, v, q2) / v)
                total += (1.0 / u) * (pdf.xfxQ2(b, u, q2) / u) * (pdf.xfxQ2(a, v, q2) / v)
            return total
        uv, _ = quad(integrand, tau, 1.0, epsabs=1e-12, epsrel=1e-10, limit=300)
        uv *= 2.0 * e0 / s
        yv = lum_nodes(pdf, e0, "gu_zu", 400)
        print(f"  {e0:7.1f} GeV  u={uv:.9e}  y={yv:.9e}  ratio={uv/yv:.8f}")


if __name__ == "__main__":
    main()
