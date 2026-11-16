# Normalization gap: resolved — `xfxQ` vs `xfxQ2` scale bug

**Status:** resolved. The ~14% gap was a **bug in our Python LHAPDF calls**, not
in native MadEvent. Native MadEvent was correct all along.

**Old results (wrong):** partonic no-eta 13,866.59 pb, lab 10,599.26 pb,
native/partonic 0.859.

**Corrected results (clean no-eta event sample):** partonic no-eta
**11,876.96 ± 0.61 pb**, lab single acceptance **9,113.31 pb** (per-channel +
threshold: **9,142.25 pb**), native no-eta 11,913 ± 10.9 pb, native both-cuts
9,142.3 ± 6.8 pb. The corrected native/partonic ratios are **0.9998**
(no-eta) and **0.99999** (lab, per-channel).

---

## 1. The bug

Our Python shim exposed

```python
pdf.xfxQ(pid, x, q2)     # argument named q2
```

but the C wrapper calls LHAPDF's `PDF::xfxQ(id, x, q)`, which takes **q in
GeV** and squares it internally. Passing `q2 = 91.188 ** 2 = 8315.25`
therefore evaluated the PDFs at

```text
Q = 8315.25 GeV, Q^2 = 6.91×10^7 GeV^2,
```

not at `Q=M_Z=91.188` GeV. Native MadEvent sets
`q2fact(1) = q2fact(2) = 91.188²` and its `pdg2pdf` passes `Q=91.188`, so
the native PDF was correct.

The effect is a DGLAP tilt:

- low `x` (low `√ŝ`): sea quarks and gluons inflated by
  15–35% → partonic convolution too high;
- high `x` (high `√ŝ`): valence/gluon PDFs depleted by
  25–30% → partonic convolution too low.

That is exactly the observed differential tilt: native/partonic 0.791 at
114–150 GeV, crossing 1 near 300 GeV, and 1.36 at 1000–2000 GeV.

### Fix

The shim now has both methods:

```python
def xfxQ(self, pid, x, q):      # q in GeV
def xfxQ2(self, pid, x, q2):    # q2 in GeV^2
```

`integrate_and_plot.py`, `verification/check_luminosity.py`, and the upstream
`conditional-nf-integration/experiments/integrate_hadronic_process.py` now use
`xfxQ2(..., q2)` with `q2 = 91.188 ** 2`.

---

## 2. Corrected totals

### No-eta

| quantity | corrected (pb) | native (pb) | ratio |
|---|---:|---:|---:|
| 114–2000 GeV | **11,876.96 ± 0.61** | 11,876.23 (events ≥114) | 1.00006 |
| 113.3555–2000 GeV (threshold-adjusted) | **11,910.79** | 11,912.39 ± 10.9 | 0.99987 |

Threshold contribution 113.3555–114: **33.84 pb** (was 45.46 pb with the wrong
PDF scale).

### Lab frame, `∣η∣<2.5`

| quantity | corrected (pb) | native (pb) | ratio |
|---|---:|---:|---:|
| 114–2000, single stored acceptance | **9,113.31** | — | — |
| single acceptance + threshold | **9,137.90** | 9,142.3 ± 6.8 | 0.99952 |
| per-channel acceptance + threshold | **9,142.25** | 9,142.3 ± 6.8 | 0.99999 |
| effective `f_lab` (event-level / weighted) | 0.7666 / 0.7673 | 0.7674 | — |

The channel-dependent correction is now **+4.35 ± 4.24 pb** (0.048%), not
−31 pb; the method spread is ±5–10 pb. The partonic channel mix now matches
the native event sample:

| channel | native % | partonic % |
|---|---:|---:|
| `gu` | 27.31 | 27.26 |
| `gd` | 23.45 | 23.53 |
| `gū` | 8.52 | 8.49 |
| `gd̄` | 12.41 | 12.54 |
| `uū` | 14.38 | 14.31 |
| `dd̄` | 13.94 | 13.88 |

### CM frame, `∣η_CM∣<2.5`

| quantity | corrected (pb) |
|---|---:|
| 114–2000, stored acceptance | **11,753.55** |
| threshold-adjusted | **11,787.39** |
| exact with-cut partonic convolution (upstream scan, corrected PDF) | **11,753.90** |
| NF + LHAPDF, CM with-cut | **11,752.44 ± 0.82** |

The NF/exact agreement is **−1.46 pb (−0.012%)**, unchanged from the previous
0.025%.

### Windowed integrals

| window | corrected partonic (pb) | native (pb) | native/partonic |
|---|---:|---:|---:|
| 114–150 GeV | 5,331.18 | 5,365.5 ± 4.7 | 1.0064 |
| 500–700 GeV | 113.38 | 112.35 ± 0.12 | 0.9909 |

The old values were 0.792 and 1.259. The residual 0.6–0.9% is consistent with
the native window integration and the partonic-scan interpolation; the
full-range totals agree to 0.01%.

---

## 3. Corrected differential ratios

Using `verification/channel_integrands.csv` (recomputed with `xfxQ2`) and the
native event sample:

| `√ŝ` bin | old native/partonic | corrected native/partonic |
|---|---:|---:|
| 114–150 | 0.791 | **1.006** |
| 150–200 | 0.877 | **1.014** |
| 200–250 | 0.930 | **0.989** |
| 250–300 | 0.969 | **0.969** |
| 300–400 | 1.031 | **0.962** |
| 400–500 | 1.164 | **0.998** |
| 500–700 | 1.287 | **1.013** |
| 700–1000 | 1.319 | 0.923 (statistics-limited) |
| 1000–2000 | 1.363 | 0.815 (statistics-limited) |

Cumulative per channel (native/partonic):

| channel | corrected ratio |
|---|---:|
| `gu→Zu` | 1.0029 |
| `gd→Zd` | 0.9876 |
| `gū→Zū` | 0.9904 |
| `gd̄→Zd̄` | 1.0149 |
| `uū→Zg` | 1.0072 |
| `dd̄→Zg` | 0.9998 |
| **total** | **0.9999** |
| `gq` group | 0.9985 |
| `qq̄` group | 1.0035 |

The old `qq̄` deficit (0.741) was entirely the PDF scale bug.

---

## 4. What was *not* the cause

The native side was audited and found correct:

- **Scales:** `fixed_ren_scale=True`, `fixed_fac_scale1/2=True`,
  `dsqrt_q2fact1/2=91.188`; `setrun.f` sets `q2fact=sf²`, and `cuts.f` does
  not call `set_fac_scale` when both fixed flags are true.
- **Couplings:** LHE event headers show SCALUP=91.188, AQEDUP=0.00754677,
  AQCDUP=0.130; param cards agree on `M_Z`, `Γ_Z`, `a_EW^-1`,
  `G_F`.
- **PDF:** external LHAPDF 247000 and internal `nn23lo1` give the same native
  window and full totals (5,372 pb for 114–150; 11,914 pb full). The problem was
  our convolution's Q/Q² argument, not the native PDF.
- **Flavor/symmetry:** native sums u,c,d,s; partonic processes are
  single-flavor; the luminosity pairs u+c and d+s; mirrors are included;
  `symfact=1`.
- **Windowed native tests:** native 114–150 = 5,365.5 ± 4.7 pb and
  500–700 = 112.35 ± 0.12 pb; these are correct. The old partonic windows
  (6,778.9 and 89.24) were wrong because of the PDF scale.

---

## 5. Documentation / upstream impact

The same `xfxQ(..., 91.188**2)` convention was used in:

- `conditional-nf-integration/experiments/integrate_hadronic_process.py`
  (patched to `xfxQ2`);
- the upstream NF + LHAPDF number **13,765.5 pb** (CM with-cut) — corrected to
  **11,752.44 ± 0.82 pb**;
- the upstream with-cut partonic convolution **13,761.97 pb** — corrected to
  **11,753.90 pb**.

The NF partonic predictions themselves are unaffected; only the LHAPDF
convolution scale changes. The NF/exact agreement remains 0.012%.

The coarse 50-point scan total is also corrected: **10,062.31 pb** (was
11,561.17 pb). The fine 100-point total is the reference: **11,876.96 pb**.
