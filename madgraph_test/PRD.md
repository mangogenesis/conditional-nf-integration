# MadGraph-only Z + 1 jet: cut definitions and integral comparison

**Date:** 2026-11-09  
**Scope:** `madgraph_test` only. No file outside this
directory is edited. The upstream NF/MMC repositories are read-only references.

**Summary:** the main finding is the eta-frame mismatch and the lab-frame
correction; see [`FINDINGS.md`](FINDINGS.md).

> **Correction 2026-11-10 — PDF scale bug.** The convolution totals quoted
> below (13,866.59, 10,599.26, 13,761.97, 13,765.5) used
> `pdf.xfxQ(..., 91.188**2)`, which evaluates at Q = 8315.25 GeV instead of
> Q = M_Z. The corrected values are: no-eta **11,876.96 ± 0.61 pb**; lab
> single acceptance **9,113.31 pb**, per-channel + threshold **9,142.25 pb**
> (native 9,142.3 ± 6.8); CM **11,753.55 pb**; exact with-cut
> **11,753.90 pb**; NF + LHAPDF CM **11,752.44 ± 0.82 pb**; NF lab
> **9,155.14 ± 8.92 pb**. Native MadEvent was correct. See
> [`FINDINGS.md`](FINDINGS.md) and
> [`verification/NORMALIZATION_GAP.md`](verification/NORMALIZATION_GAP.md).

## 1. The question

The earlier comparison showed that the per-energy partonic cross sections from
MadGraph and the trained conditional NF are close, while the total integrated
cross sections differ:

| quantity | corrected value (2026-11-10) |
|---|---:|
| NF + LHAPDF, `∣eta_CM∣<2.5` | **11,752.44 ± 0.82 pb** |
| exact partonic MadEvent + LHAPDF, with cuts | **11,753.90 pb** |
| native MadEvent fixed scale, `pT,j>20`, `∣eta_j∣<2.5` | **9,142.3 ± 6.8 pb** |
| native MadEvent fixed scale, `pT,j>20`, no eta cut | **11,913 ± 10.9 pb** |

The pre-correction values (13,765.5, 13,761.97, 9,147.5, 11,915) used the
`xfxQ`/`Q^2` bug and are superseded; see the correction note above.

This test now asks two questions:

1. **Literal no-cut test.** Remove both the eta cut and the jet `p_T` cut. Does
   the native MadGraph total agree with the total obtained by scanning the
   partonic cross section at individual energies and convolving with PDFs?
2. **Finite-regulator comparison.** If the literal no-cut limit is not finite,
   what is the smallest regulator at which the two methods can be compared, and
   do they agree there?

The short answer is in §5: the literal no-cut limit is divergent on both sides,
so there is no finite no-cut total to compare. At a finite regulator
`pT,j > 5 GeV`, the native total is **40,116 ± 46 pb**; the partonic convolution
on the same regulator is **22,919 ± 4.4 pb** on the coarse 50-point grid.

## 2. Two different meanings of "no cuts"

The earlier scan used **no eta cut** but kept the jet `p_T` regulator
`pT,j > 20 GeV`. This document separates two cases:

| label | eta | pT | status |
|---|---|---|---|
| no eta, regulated | `etaj = -1` | `ptj = 20` | finite; the 50-point scan |
| literal no cuts | `etaj = -1` | `ptj = 0` | divergent; no finite result |

A truly cut-free LO `Z + jet` cross section is collinear/soft divergent. The
matrix element has initial-state collinear singularities and a soft-gluon
singularity, and at LO there is no cancellation against virtual corrections or
PDF counterterms. MadGraph's phase-space grid also requires a positive jet
`p_T`: with `ptj = 0` it prints

```text
Warning: pt or E min of a jet should in general be >0
Error opening grid
```

and the per-grid integrals are not convergent. The same happens on the partonic
side. There is therefore no finite literal no-cut total to compare.

"Literal no cuts" in this document means exactly `ptj = 0` and `etaj = -1`.
The finite comparison uses `ptj = 5 GeV`, which is explicitly labelled as a
regulator, not as "no cuts".

## 3. Method

### 3.1 Processes

Six explicit partonic MadGraph processes are generated in one MadEvent
directory:

```text
P1  g u   > z u       P2  g d   > z d
P3  g u~  > z u~      P4  g d~  > z d~
P5  u u~  > z g       P6  d d~  > z g
```

The channel names are `gu_zu`, `gd_zd`, `gux_zux`, `gdx_zdx`, `uux_zg`,
`ddx_zg`. The native test uses the single process `p p > z j` in a separate
MadEvent directory.

### 3.2 Fixed settings

Partonic scan:

```text
lpp1 = lpp2 = 0                 # partonic; no PDF in the matrix-element run
ebeam1 = ebeam2 = sqrt(s-hat)/2
pdlabel1 = pdlabel2 = none
fixed_ren_scale = fixed_fac_scale1 = fixed_fac_scale2 = True
scale = dsqrt_q2fact1 = dsqrt_q2fact2 = 91.188 GeV
alpha_s = 0.130                 # param_card aS, pdlabel = none
etaj = -1                       # no eta cut
maxjetflavor = 4
use_syst = False
```

The `ptj` value is varied: `20` for the earlier scan, `0` for the literal
no-cut probe, and `1,2,5,10` for the regulator sequence.

Native scan:

```text
p p > z j
lpp1 = lpp2 = 1
ebeam1 = ebeam2 = 6500 GeV
pdlabel1 = pdlabel2 = nn23lo1
fixed_ren_scale = fixed_fac_scale1 = fixed_fac_scale2 = True
scale = dsqrt_q2fact1 = dsqrt_q2fact2 = 91.188 GeV
etaj = -1                       # no eta cut
maxjetflavor = 4
use_syst = False
```

The native `ptj` is varied in the same way.

### 3.3 Energy grid

The partonic cross section is evaluated at **50 linear points from 114 to
2000 GeV**:

```text
√ŝ_i = 114 + i(2000-114)/(49),
 i=0,…,49,
```

spacing **38.49 GeV**. This is the same 50-point density as each of the
earlier low/high interpolation scans, now continuous without a high/low split.
The low-energy region is under-sampled: only four points lie below 250 GeV.

### 3.4 Total integrated cross section

For each channel `c`, the partonic MadGraph values `σ̂_c(√ŝ)`
are convolved with the LHAPDF luminosity:

```text
σ_c =
∫_114^2000 d√ŝ 
dL_c/d√ŝ 
σ̂_c(√ŝ),
```

with the same beam energy, scale, and PDF pairs as the earlier campaign. The
trapezoid rule is used on the 50-point grid. The total is the sum of the six
channels. No `[2000,13000]` tail is added in this test; it is 0.139 pb in the
previous campaign and is negligible at this precision.

### 3.5 Native regulator sequence

The native `p p > z j` integration is repeated at

```text
ptj = 20, 10, 5, 2, 1, 0 GeV
```

with the same PDF, scale, and no eta cut. The nonzero points show whether the
native total approaches a finite limit as `ptj -> 0`; the `ptj = 0` point shows
what MadGraph actually returns at the literal no-cut setting.

## 4. Commands

All commands are run from `madgraph_test`.

### Generate the six-process partonic directory

```bash
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
export TMP="/tmp"
export TEMP="$TMP"; export TMPDIR="$TMP"

"mg5_aMC" generate_partonic.mg5
"python3" prepare_run.py partonic_zj
```

### Generate the native `p p > z j` directory

```bash
"mg5_aMC" generate_native_nocuts.mg5
"python3" prepare_run.py native_zj_nocuts
```

### Native regulator sequence

```bash
for p in 20 10 5 2 1; do
  "python3" run_native_nocuts.py \
    --ptj $p --points 20000 --iterations 8 --accuracy 0.002 \
    --run-name ptj${p}_20k --timeout 1800
done

# Literal no-cut attempt
"python3" run_native_nocuts.py \
  --ptj 0 --points 20000 --iterations 8 --accuracy 0.002 \
  --run-name nocuts_20k --timeout 1800
```

### Partonic no-cut probe

```bash
"python3" scan_partonic.py \
  --energies 200 --points 2000 --iterations 2 --accuracy 0.01 \
  --ptj 0 --run-prefix nocut_probe \
  --output results/scan_nocut_probe.csv
```

### Partonic `ptj = 5` scan

```bash
"python3" scan_partonic.py \
  --points 5000 --iterations 3 --accuracy 0.005 \
  --ptj 5 --run-prefix ptj5_scan \
  --output results/scan_ptj5.csv
```

### Integrate and plot

```bash
export LHAPDF_DATA_PATH="/path/to/lhapdf/share/LHAPDF"
"python3" integrate_and_plot.py \
  --scan results/scan_ptj5.csv --output-dir results/ptj5
"python3" plot_ptj_divergence.py
"python3" plot_divergence_endpoint.py
```

### Fine no-eta scan

The 100-point split grid is in `results/fine_energy_grid.txt`, split into five
20-point chunks. Run one chunk at a time with `--resume` between them:

```bash
for i in 1 2 3 4 5; do
  "python3" scan_partonic.py \
    --energy-file results/fine_energy_grid_chunk${i}.txt \
    --points 10000 --iterations 6 --accuracy 0.002 \
    --ptj 20 --run-prefix noeta_fine \
    --output results/scan_noeta_fine.csv --resume
done

export LHAPDF_DATA_PATH="/path/to/lhapdf/share/LHAPDF"
"python3" integrate_and_plot.py \
  --scan results/scan_noeta_fine.csv --output-dir results/noeta_fine
"python3" apply_lab_eta_correction.py \
  --scan results/scan_noeta_fine.csv --output-dir results/noeta_fine \
  --cm-with-cuts 13761.97 \
  --note "fine low/high no-eta scan; acceptance from native LHE events"
```

### Frame diagnostics

The native LHE events used below are from the `etaj=5`, `ptj=20` run; they are
the last `events.lhe` files in `native_zj_nocuts/SubProcesses/P*/G*/`.

```bash
"python3" run_native_nocuts.py \
  --ptj 20 --etaj 5 --points 50000 --iterations 8 --accuracy 0.002 \
  --run-name etaj5_50k
"python3" analyze_eta_frame_events.py

export LHAPDF_DATA_PATH="/path/to/lhapdf/share/LHAPDF"
"python3" apply_lab_eta_correction.py
"python3" plot_eta_frame.py
"python3" plot_lab_eta_correction.py
"python3" plot_consistency_scenarios.py
```

## 5. Results

> **Pre-fix section.** The convolution totals below use the old
> `xfxQ(..., 91.188**2)` convention. Authoritative corrected values (clean
> event sample): no-eta **11,876.96 pb**; lab single acceptance **9,113.31 pb**,
> per-channel + threshold **9,142.25 pb**; CM **11,753.55 pb**; exact with-cut
> **11,753.90 pb**; NF + LHAPDF CM **11,752.44 pb**; NF lab
> **9,155.14 ± 8.92 pb**. See `FINDINGS.md`.

### 5.1 Literal no-cut attempt

**Native `p p > z j`, `etaj = -1`, `ptj = 0`, 20,000 points × 8 iterations:**

MadGraph returns

```text
Cross-section :   1.485e+06 +- 4.331e+04 pb
```

but this is **not a converged cross section**. The two phase-space grid copies
disagree by orders of magnitude:

| process | grid G1 (pb) | grid G2 (pb) |
|---|---:|---:|
| `P1_gq_zq` | 995.1 | 146,100 |
| `P1_qq_zg` | 535,900 | 610,100 |

The 2,000 × 2 attempt returns `0 ± 0` and the same `Error opening grid`
warning. The returned 1.485e6 pb is therefore a grid-dependent artefact of the
divergent integral, not a physical total.

**Partonic, `ptj = 0`, 200 GeV, 2,000 points × 2 iterations:**

The same grid warning appears, and the per-grid channel values are not
convergent:

| process | grid G1 (pb) | grid G2 (pb) |
|---|---:|---:|
| `P1_gq_zq` | 1.671 | 49.72 |
| `P2_gq_zq` | 2.135 | 47.65 |
| `P3_gq_zq` | 1.660 | 72.25 |
| `P4_gq_zq` | 2.136 | 59.64 |
| `P5_qq_zg` | 224.7 | 176.0 |
| `P6_qq_zg` | 346.6 | 284.6 |

The combined result is `0 ± 0`; there is no finite per-energy cross section,
and therefore no convolution total. Summing the two grid configs by hand gives
about **1,269 pb** at 200 GeV (1.88× the `ptj = 2` value), but this is not a
converged no-cut result; the combined MadGraph `results.dat` is 0 because the
grid step failed. The raw grid records are in
`results/nocut_grid_dependence.csv`.

**Conclusion:** the literal no-cut comparison is undefined. The LO
`Z + jet` cross section is divergent; a regulator (or a matched NLO/parton-
shower calculation) is mandatory.

### 5.2 Regulator sequence

Native `p p > z j`, no eta cut, 20,000 points × 8 iterations:

| `ptj` (GeV) | native total (pb) | note |
|---:|---:|---|
| 20 | 11,913 ± 10.9 | control |
| 10 | 23,692 ± 30.1 | |
| 5 | 40,116 ± 45.9 | |
| 2 | 68,856 ± 79.6 | |
| 1 | 95,661 ± 108.3 | |
| 0 | 1,484,700 ± 43,311 | grid-dependent; not converged |

The native total grows monotonically as `ptj` decreases. There is no plateau:
the ratio from `ptj = 20` to `ptj = 1` is **8.0×**, and the `ptj = 0` point is
another **15.5×** above `ptj = 1` but is not a converged number.

Partonic `Z + j` at `√ŝ=200` GeV, no eta cut:

| `ptj` (GeV) | partonic sum (pb) |
|---:|---:|
| 10 | 416.1 ± 0.07 |
| 5 | 528.0 ± 0.09 |
| 2 | 675.7 ± 0.09 |

The partonic cross section also grows as `ptj` decreases. The growth is milder
at fixed `√ŝ=200` GeV than in the native total, because the native
total is dominated by the low-`√ŝ` region where the divergence is
strongest.

### 5.3 Finite-regulator comparison at `ptj = 5`

At the smallest regulator used for a full scan,

```text
native total, no eta, pT,j > 5 GeV : 40,116 ± 46 pb
partonic convolution, same regulator, coarse grid : 22,919 ± 4.4 pb
```

The ratio is **1.75**. The difference is concentrated in the `qq` channel:

| channel | native (pb) | partonic (pb) | native/partonic |
|---|---:|---:|---:|
| `gq` | 22,141 | 15,839 | 1.40 |
| `qq` | 17,974 | 7,079 | 2.54 |
| total | 40,116 | 22,919 | 1.75 |

The partonic values use the full 50-point grid (§3.3), which under-samples the
low-energy threshold region; the comparison is therefore indicative, not final;
a fine low-energy scan would be needed for a quantitative statement.

### 5.4 The earlier `ptj = 20` results

These are the values from the earlier scan. They are **not** literal no-cut
results; they keep `pT,j > 20 GeV`.

| method | total (pb) | note |
|---|---:|---|
| MadGraph, no eta, **continuous 50-point grid** | **11,561.17 ± 1.13** | `ptj = 20` |
| MadGraph, `∣eta∣<2.5`, **same 50-point grid** | **11,457.05** | `ptj = 20` |
| MadGraph, `∣eta∣<2.5`, **fine low/high grid** | **13,761.97** | previous exact convolution |
| NF + LHAPDF, `∣eta∣<2.5` | **13,765.5 ± 0.59** | reference only |
| native MadEvent, no eta | 11,913 ± 10.9 | `ptj = 20`, this test |
| native MadEvent, both cuts | 9,147.5 ± 10.32 | campaign baseline |

The controlled comparison at `ptj = 20`:

```text
13,761.97 (fine with cuts)
-
11,457.05 (same 50-grid with cuts)
=
2,304.92 pb
```

is a **16.7% grid effect**. The eta effect on the same coarse grid is only

```text
11,561.17 - 11,457.05 = 104.12 pb (0.9%).
```

The fine-grid with-cut MadGraph convolution (13,761.97 pb) agrees with the NF
total (13,765.5 pb); both use the partonic CM-frame eta cut, so this agreement
does not test the native lab-frame with-cut total (see §5.6). The coarse no-eta total (11,561 pb) is close to the native
no-eta total (11,913 pb), but both are on different integration grids and
should not be treated as a controlled comparison.

### 5.5 What the test shows

The literal no-cut limit cannot settle the earlier discrepancy because it is
divergent. Both methods show the divergence:

- native: 11,913 pb at `ptj = 20` -> 95,661 pb at `ptj = 1` -> 1.48e6 pb at
  `ptj = 0` (grid-dependent);
- partonic: 416 pb at `ptj = 10` -> 528 pb at `ptj = 5` -> 676 pb at
  `ptj = 2` at 200 GeV.

There is no finite plateau, so there is no literal no-cut total to compare.
At `ptj = 20`, the no-eta comparison is the clean one: native 11,913 pb vs the
partonic coarse 11,561 pb (3%). The both-cuts comparison has an additional
frame mismatch, described in §5.6.

### 5.6 Why the both-cuts comparison disagrees: eta is frame-dependent

The native both-cuts total (9,142 ± 6.8 pb) is **23% below** the native no-eta
total (11,913 ± 10.9 pb). On the same coarse 50-point grid, the partonic
with-cut total (11,457.05 pb) is only **0.9% below** the partonic no-eta total
(11,561.17 pb). The fine with-cut total is 13,761.97 pb; the fine no-eta scan
gives **13,866.59 ± 0.73 pb**, a 0.76% eta reduction. The reason for the
different eta effects is the frame in which the cut is applied:

- the native `p p > z j` run applies `etaj = 2.5` in the **lab frame** (the
  hadronic CM frame);
- the partonic scan uses `lpp1 = lpp2 = 0`, so its `etaj = 2.5` is the
  **partonic CM frame** rapidity.

These are not the same observable. The partonic CM is boosted along the beam
by rapidity `Y`, so

```text
y_lab = y_CM + Y.
```

A jet that is central in the partonic CM can be forward in the lab, and vice
versa. The current convolution uses the total partonic cross section
`σ̂(√ŝ)` with a cut on `y_CM`; the physical lab
cut requires the double-differential `dσ̂/dY` and the constraint

```text
∣η_CM + Y∣ < 2.5.
```

The native eta scan shows the lab-frame effect:

| native cut | total (pb) |
|---|---:|
| no eta | 11,913 ± 10.9 |
| lab `∣η∣<5` | 11,892 ± 10.5 |
| lab `∣η∣<2.5` | 9,142 ± 6.8 |

The partonic CM-frame scan on the same 50-point grid:

| partonic cut | total (pb) |
|---|---:|
| no eta | 11,561.17 ± 1.13 |
| CM `∣η∣<2.5` | 11,457.05 |

The native LHE events (100,990 events, weight sum 11,891.7 pb) show the same
effect directly. Applying the cut to the same events gives:

| frame of the `∣η∣<2.5` cut | weighted fraction | implied total (pb) |
|---|---:|---:|
| lab frame | 0.7669 | 9,119 |
| partonic CM frame | 0.9899 | 11,772 |

The lab-frame fraction reproduces the native both-cuts total (9,142 pb); the
CM fraction reproduces the partonic CM eta effect (11,457.05/11,561.17 =
0.9909). The same events give both answers depending on the frame.

The native 9,142 pb is the **lab-frame** with-cut total. The partonic 13,761.97
pb (and the NF 13,765.5 pb) is the **partonic-CM-frame** with-cut total. They
are different observables; comparing them directly is the source of the ~1.5×
with-cut discrepancy. This is a definition/frame mismatch, not a numerical bug
in MadGraph or the NF.

#### Correcting the partonic total to the lab frame

The event-level lab acceptance can be applied as a function of
`√ŝ` to the partonic no-eta integrand. The results are:

| convolution | total (pb) | ratio to no-eta |
|---|---:|---:|
| coarse no eta | 11,561.17 | 1.0000 |
| coarse CM `∣η∣<2.5` | 11,460.53 | 0.9913 |
| coarse lab-corrected `∣η∣<2.5` | 8,951.23 | 0.7742 |
| **fine no eta** | **13,866.59** | 1.0000 |
| **fine CM `∣η∣<2.5`** | **13,764.87** | 0.9927 |
| **fine lab-corrected `∣η∣<2.5`** | **10,599.26** | 0.7644 |

The fine CM correction matches the direct fine with-cut total (13,764.87 vs
13,761.97 pb, 0.02%), and the fine lab fraction (0.7644) matches the native
event-level lab fraction (0.7669). The eta frame is confirmed.

The coarse lab-corrected **8,951 pb** is within 2.1% of native **9,142.3 pb**,
but the fine lab-corrected **10,599.26 pb** is 13.7% above native. The coarse
agreement was a coincidence of two low-biased calculations. The fine no-eta
scan (**13,866.59 pb**) shows the native no-eta **11,913 pb** is 14.1% low. The
remaining issue is the native absolute normalization, not the eta frame.

A differential check (`results/differential_compare.csv`) shows the deficit is
concentrated at low `√ŝ`: native/partonic is **0.834** in
114–250 GeV, **1.003** in 250–500 GeV, and **1.274** in 500–1000 GeV
(statistics-limited above 1000 GeV). A threshold-aware native run with
`dsqrt_shat = 111.188` GeV gives 11,912 pb, so the `dsqrt_shat` setting is not
the cause.

The native run was repeated with external **LHAPDF 247000** using patched
`madevent.exe` binaries. The result is **11,914.0 ± 10.9 pb**, identical to the
internal `nn23lo1` **11,913 ± 10.9 pb**. The PDF is therefore **not** the
cause; the remaining ~14% is the native phase-space integration/grid. The fix
is a same-length binary replacement of the embedded POSIX path
`/path/to/lhapdf/share` with
`/path/to/lhapdf/share` in
`native_zj_lhapdf/SubProcesses/P*/madevent.exe`.

To make a consistent both-cuts comparison, the partonic convolution must apply
the eta cut in the **lab frame**. That requires the double-differential
partonic cross section `dσ̂/dy_CM` (or event-level information
from the partonic calculation), not just the total partonic cross section with
a CM eta cut. Alternatively, one can compare native and partonic with an eta
cut defined in the partonic CM frame, but that is not the physical hadronic
jet cut.

#### Three-scenario summary

| scenario | native MadGraph | partonic scan + convolution | consistent? |
|---|---:|---:|---|
| no cuts (`etaj=-1`, `ptj=0`) | 1.485e6 ± 43,311 pb (grid-dependent) | no finite result | no; divergent |
| pt only (`etaj=-1`, `ptj=20`) | 11,913 ± 10.9 pb | coarse 11,561.17 ± 1.13; fine **13,866.59 ± 0.73** | native **14.1% low** vs fine |
| both cuts (`etaj=2.5`, `ptj=20`) | 9,142.3 ± 6.8 pb (lab frame) | coarse lab-corrected 8,951.23; fine lab-corrected **10,599.26**; CM fine 13,761.97 | eta frame matches; native **13.7% low** vs fine |

The machine-readable version is `results/madgraph_consistency.csv`.

## 6. Plots

- `results/nocut_divergence.png` — native total vs `ptj` (left) and partonic
  cross section at 200 GeV vs `ptj` (right). Both grow as `ptj` decreases; the
  `ptj = 0` native point is annotated as grid-dependent.
- `results/divergence_endpoint.png` — the same data with `ptj = 0` included
  explicitly: native 1.49e6 pb (open marker, grid-dependent) and partonic
  1,269 pb (config sum; combined result 0).
- `results/eta_frame_effect.png` — native lab-frame eta scan (left) vs
  partonic CM-frame eta scan (right). The lab cut removes 23%; the partonic
  CM cut removes 0.9%.
- `results/consistency_scenarios.png` — native vs partonic totals for
  `pT`-only and both-cuts. The `pT`-only bars agree; the both-cuts bars differ
  because the eta cut is in different frames.
- `results/eta_frame_events.csv` — native-event `√ŝ` bins with the
  weighted lab-frame and partonic-CM-frame `∣eta∣<2.5` fractions.
- `results/lab_eta_correction.csv` and `results/lab_eta_correction.json` — the
  coarse partonic no-eta, CM-cut, and lab-corrected totals.
- `results/lab_eta_correction.png` — coarse: native both-cuts 9,142 pb,
  partonic CM 11,457 pb, lab-corrected 8,951 pb.
- `results/lab_eta_correction_fine.png` — fine: native 9,142 pb, partonic CM
  13,765 pb, lab-corrected 10,599 pb; native is 13.7% low.
- `results/scan_noeta_fine.csv`, `results/noeta_fine/total_integral.csv` /
  `.json`, and `results/noeta_fine/lab_eta_correction.csv` / `.json` — the
  fine 100-point no-eta scan, integral, and lab/CM correction.
- `results/differential_compare.csv` / `.json` / `.png` — native vs fine
  partonic `dσ/d√ŝ`; native/partonic 0.834 in 114–250 GeV,
  1.003 in 250–500, 1.274 in 500–1000 (statistics-limited above).
- `results/madgraph_consistency.csv` — the three cut scenarios, native vs
  partonic, with the frame note.
- `results/partonic_cross_section_vs_energy.png` — six channels and the sum,
  no eta cut, with the `∣eta∣<2.5` curve on the same coarse 50-point grid and
  on the fine low/high grid.
- `results/hadronic_integrand_vs_energy.png` — `dL/d√ŝ×σ`
  for the no-eta scan; the area is the total.
- `results/total_comparison.png` — total integrated cross sections at
  `ptj = 20`: MadGraph no-eta, MadGraph with cuts, NF reference, native no-eta,
  native both cuts.
- `results/ptj5/total_integral.csv` and `results/ptj5/total_integral.json` — the
  `ptj = 5` convolution total and per-channel breakdown.
- `results/ptj5/comparison.csv` and `results/ptj5/comparison.json` — the native
  vs partonic `ptj = 5` comparison by `gq`/`qq`/total. The generated
  `total_comparison.png` in that directory mixes the `ptj = 5` scan with the
  earlier `ptj = 20` reference bars; use the JSON/CSV for the number.

## 7. Interpretation

1. **Per-energy consistency** — the partonic curve is smooth and continuous
   across 114–2000 GeV.
2. **Grid/split effect** — one continuous 50-point scan versus the earlier
   fine low/high split: **16.7%** at `ptj = 20`.
3. **Eta effect** — on the same coarse grid, removing eta changes the
   partonic total by only **0.9%**, because the partonic eta cut is in the
   partonic CM frame. The native lab-frame eta cut changes the total by
   **23%**. The with-cut native and partonic totals are different observables.
4. **Literal no-cut limit** — not finite. The native and partonic calculations
   both diverge as `ptj -> 0`; the returned numbers are regulator- and
   grid-dependent.
5. **Both-cuts discrepancy** — the native 9,142 pb (lab frame) and the
   partonic/NF 13,762 pb (partonic CM frame) differ because of the eta frame,
   not because of a MadGraph integration bug.
6. **Lab-frame correction and native normalization** — the fine lab-corrected
   total is 10,599.26 pb; the fine lab fraction (0.7644) matches the native
   event fraction (0.7669). The eta frame is resolved. The fine no-eta total
   (13,866.59 pb) shows the native no-eta 11,913 pb is 14.1% low. The native
   LHAPDF 247000 cross-check gives 11,914.0 pb, so the PDF is ruled out; the
   remaining cause is the native phase-space integration/grid.

The correct conclusion is not that the native and convolution methods disagree
at no cuts. It is that **the no-cut LO cross section is not a well-defined
number**, and that **the eta cut must be defined in the same frame on both
sides**. A physical hadronic jet cut is in the lab frame; the native result is
the relevant one, and the partonic convolution must include the boost. After
that correction, the eta-frame discrepancy is resolved. The native LHAPDF
cross-check rules out the PDF; the remaining ~14% is the native phase-space
integration, which is the next debugging target. A comparison also requires a
common finite regulator (and, for the convolution, a grid fine enough to
resolve the low-energy threshold).

## 8. Caveats

- The native `ptj = 0` number (1.485e6 pb) is grid-dependent and must not be
  quoted as a physical cross section.
- The partonic `ptj = 0` probe used 2,000 points × 2 iterations; higher
  statistics would also be grid-dependent because the integral is divergent.
- The `ptj = 5` convolution uses the coarse 50-point grid; the low-energy
  region is under-sampled. A fine low-energy scan is required for a
  quantitative comparison.
- `ptj = 5` is a regulator, not "no cuts".
- The native and partonic setups use different PDF treatments (native
  `nn23lo1`, partonic LHAPDF convolution). The native LHAPDF 247000
  cross-check gives **11,914.0 ± 10.9 pb**, identical to the internal
  **11,913 ± 10.9 pb**, so the PDF is not the cause of the 14% deficit. The
  local Windows fix is a same-length replacement of the embedded POSIX LHAPDF
  path in `madevent.exe`; the remaining ~14% is the native phase-space
  integration.
- The native both-cuts and no-eta runs are the same setup except for `etaj`;
  their 23% difference is a lab-frame eta effect. The partonic 0.9% eta effect
  is a partonic-CM-frame effect. The two with-cut totals are different
  observables.
- The lab-frame correction uses the native event-level acceptance, not a pure
  partonic differential calculation. A pure partonic implementation would
  generate partonic events at each `√ŝ` and boost by `Y` sampled
  from the luminosity. The fine lab-corrected 10,599.26 pb vs native 9,142.3 pb
  is a 13.7% native deficit; the frame fraction matches (0.7644 vs 0.7669).
- The fine no-eta total is measured at **13,866.59 ± 0.73 pb**. The native
  no-eta 11,913 pb is 14.1% low; the coarse 11,561 pb is 16.6% low. The
  remaining issue is the native normalization.
- The earlier with-cut native baseline (9,147.5 pb) and the no-eta native
  baseline (11,915 pb) come from the campaign run cards; they are consistent
  with the fresh fixed-scale runs (9,142.3 and 11,913 pb).
