# Verification of the lab-frame row

**Row audited:** `partonic convolution × native lab acceptance | lab | pT>20, |eta|<2.5 | fine`

**Old result (superseded):** 10,599.26 pb, computed with
`pdf.xfxQ(..., 91.188**2)`.  `xfxQ` takes `Q` in GeV, so this evaluated the
PDFs at `Q=8315.25` GeV instead of `Q=M_Z=91.188` GeV.

**Corrected result (2026-11-10, clean event sample):** after switching to
`pdf.xfxQ2`: no-eta **11,876.96 ± 0.61 pb**; lab single acceptance
**9,113.31 pb**, per-channel + threshold **9,142.25 pb** vs native
**9,142.3 ± 6.8 pb**; CM **11,753.55 pb**; exact with-cut **11,753.90 pb**;
NF + LHAPDF CM **11,752.44 ± 0.82 pb**; NF lab **9,155.14 ± 8.92 pb**.
The old apparent 14% native deficit was the PDF-scale bug, not a native
integration error. See [`../FINDINGS.md`](../FINDINGS.md) for the full
bug/fix record and [`NORMALIZATION_GAP.md`](NORMALIZATION_GAP.md) for the
scale-bug report.

The detailed methodology below remains valid (luminosity Jacobian, event
kinematics, acceptance extraction, channel checks). Where it quotes
10,599.26 / 13,866.59 / 13,764.87, those are **pre-fix** numbers.

**Verdict on the old audit:** the old number is reproducible exactly from the
old inputs, and every numerical step passed its check. The missing piece was
the `xfxQ`/`xfxQ2` scale convention, which is now fixed.

### Corrected summary

| quantity | corrected (pb) | native (pb) |
|---|---:|---:|
| no-eta, 114–2000 | 11,876.96 ± 0.61 | 11,876.23 (events ≥114) |
| no-eta, threshold-adjusted | 11,910.79 | 11,912.39 ± 10.9 |
| lab, 114–2000, single acceptance | 9,113.31 | — |
| lab, per-channel + threshold | **9,142.25** | 9,142.3 ± 6.8 |
| CM, 114–2000 | 11,753.55 | — |
| CM, threshold-adjusted | 11,787.39 | — |
| exact with-cut convolution | 11,753.90 | — |
| NF + LHAPDF, CM with-cut | 11,752.44 ± 0.82 | — |
| NF + LHAPDF, lab (event-level + tail) | 9,155.14 ± 8.92 | 9,142.3 ± 6.8 |
| window 114–150 | 5,331.18 | 5,365.5 ± 4.7 |
| window 500–700 | 113.38 | 112.35 ± 0.12 |

All files are in `madgraph_test/verification/`.

---

## 1. Inputs and what each one does

| input | file | role |
|---|---|---|
| partonic scan | `results/scan_noeta_fine.csv` | 100 energies, 114–2000 GeV, six channels, no eta, `pT>20` |
| per-channel integrands | `verification/channel_integrands.csv` | `I_c(E)=L_c(E)σ_c(E)` recomputed from the scan |
| PDF | LHAPDF 6.5.6 `NNPDF23_lo_as_0130_qed` (247000) | luminosity `L_c(E)` |
| stored acceptance | `results/eta_frame_events.csv` | 25-GeV bins of `f_lab`, `f_CM` |
| fresh event sample | `verification/events_flat.csv` | 99,611 native LHE events used to re-derive the acceptance |
| native results | logs `ptj20_check.log`, `bothcuts_100k.log` | 11,913 ± 10.9 pb and 9,142.3 ± 6.8 pb |

The partonic scan settings are unchanged: `lpp1=lpp2=0`, `ebeam=√ŝ/2`, `pdlabel=none`, `α_s=0.130`, fixed scale 91.188 GeV, `maxjetflavor=4`, `ptj=20`, `etaj=-1`, 10k points, 6 iterations, accuracy 0.002.

The fine grid has 52 points at or below 250 GeV (2.78 GeV spacing) and 48 above 250 GeV (36.73 GeV spacing), with near-duplicate pairs at 200.000/200.041 and 236.122/236.735 GeV. Their trapezoid weights are tiny.

---

## 2. Step 1 — event parsing and kinematic identities

`verification/parse_events.py` parses every LHE event in the current `native_zj_nocuts/SubProcesses/P*/G*/events.lhe` sample (this is a later no-eta run with `dsqrt_shat=111.188`, not the etaj=5 sample used for the stored acceptance; it is used as an independent test sample).

| check | result |
|---|---|
| events | 99,611 |
| weight sum | 11,912.391 ± 51.146 pb |
| final-state particles per event | exactly 2 for all events |
| jets per event (excluding Z) | exactly 1 for all events |
| jet PDG | gluon 51,855; quarks 47,756 |
| `x_1` range | `1.05×10^-4` to 0.881 |
| `x_2` range | `9.47×10^-5` to 0.842 |
| max `lvert√ŝ_LHE-√ŝ_Zjrvert` | `5.1×10^-6` GeV |
| max `lvert Y_LHE-Y_Zjrvert` | `4.4×10^-8` |
| max `lvert y_CM-(y_lab-Y)rvert` | `1.1×10^-12` |
| max `lvert y-ηrvert` | `4.2×10^-7` (lab), `4.0×10^-7` (CM) |

The definitions are therefore consistent:

```text
x_1=(E_+z)/(6500), x_2=(E_-z)/(6500), 
Y=1/2 ln x_1/x_2, 
√ŝ=√(x_1x_2) 13 TeV,
```
```text
y_lab=y(jet), y_CM=y_lab-Y.
```

The jet is the highest-`p_T` final-state particle with PDG `≠ 23`. At LO there is exactly one such particle, so the selection is unambiguous. Because the final-state parton is massless, `y=η` to `4×10^-7`; the acceptance therefore matches MadGraph's `etaj` cut.

The event weights are constant per grid, except `P1_qq_zg` which has 19 distinct weights. All fractions below are weighted; the effective sample size is about 54k, not 100k.

---

## 3. Step 2 — luminosity integral

`verification/check_luminosity.py` tests the 1D rapidity integral

```text
dL_c/d√ŝ =
2√ŝ/s 
Σ_((a,b)∈ c)∫_-y_max^y_maxdy 
(f_a(x_1,Q^2))/(x_1) (f_b(x_2,Q^2))/(x_2),
 
x_(1,2)=√ŝ/√s e^(± y),
```

with `Q^2=91.188^2` GeV`^2`, `√s=13` TeV, and both orderings.

### Gauss–Legendre convergence

| nodes | no-eta total (pb) | lab total (pb) |
|---:|---:|---:|
| 10 | 13,870.5209 | 10,602.1738 |
| 20 | 13,866.3435 | 10,599.0946 |
| 40 | 13,866.8393 | 10,599.4473 |
| 80 | 13,866.5973 | 10,599.2675 |
| 160 | 13,866.5920 | 10,599.2638 |
| **200 (official)** | **13,866.5928** | **10,599.2645** |
| 320 | 13,866.5933 | 10,599.2648 |
| 640 | 13,866.5932 | 10,599.2647 |

80 nodes already agree with 640 to better than 0.001 pb. The official 200 nodes are converged.

### Independent adaptive quadrature

`scipy.integrate.quad` on the same `y`-integral agrees with the 200-node Gauss–Legendre result at the `10^-7` level for all tested energies and channels (114, 150, 200, 300, 500, 1000, 2000 GeV; `gu_zu` and `uux_zg`).

### Independent Jacobian check

A direct change of variables `u=x_1`, `v=x_1x_2` gives

```text
L_c(√ŝ)=2√ŝ/s 
∫_τ^1du/u 
[
f_a(u)f_b(τ/u)+f_b(u)f_a(τ/u)
],
 τ=ŝ/s.
```

This integral agrees with the `y`-integral to **1.00000000** for all six channels at 200 and 1000 GeV; one ordering alone is exactly half. This validates both the Jacobian `1/u` and the factor of 2 from the two beam orderings.

A first 2D histogram check gave a ratio of 0.878. That check was too coarse (a 1 GeV curved band on a 4000-point grid) and was discarded; the exact `u`-integral above is the reliable independent check.

---

## 4. Step 3 — partonic convolution and the hadronic differential cross section

The hadronic differential cross section is

```text
dσ_had/d√ŝ (√ŝ)
=
Σ_c L_c(√ŝ) σ_c(√ŝ)
```

in pb/GeV. It is the **integrand** of the total:

```text
σ_had=∫_114^2000 d√ŝ 
dσ_had/d√ŝ.
```

The lab-frame version multiplies the differential before integration:

```text
dσ_lab/d√ŝ 
=f_lab(√ŝ) 
dσ_had/d√ŝ,
 
σ_lab=∫ d√ŝ 
f_lab(√ŝ) 
dσ_had/d√ŝ.
```

That is how the acceptance is used: it is not applied to the total after the fact; it weights the differential at each `√ŝ`, and the total is the area under the corrected curve.

### Reconstructed differential at key energies

From `verification/differential.csv`:

| `√ŝ` (GeV) | `dσ/d√ŝ` (pb/GeV) | `f_lab` | `f_CM` | lab differential (pb/GeV) | CM differential (pb/GeV) |
|---:|---:|---:|---:|---:|---:|
| 114.00 | 106.027 | 0.71989 | 1.00000 | 76.328 | 106.027 |
| 150.08 | 133.820 | 0.76460 | 1.00000 | 102.319 | 133.820 |
| 200.00 | 46.6817 | 0.80030 | 1.00000 | 37.359 | 46.682 |
| 250.00 | 19.3343 | 0.80818 | 1.00000 | 15.626 | 19.334 |
| 310.20 | 7.84995 | 0.82164 | 0.95522 | 6.4499 | 7.4984 |
| 383.67 | 3.09679 | 0.82857 | 0.88341 | 2.5659 | 2.7357 |
| 493.88 | 0.978905 | 0.81289 | 0.83445 | 0.79574 | 0.81685 |
| 1008.16 | 0.0286398 | 0.89141 | 0.66129 | 0.025530 | 0.018939 |

Above about 500 GeV the stored 25-GeV acceptance bins are statistics-limited and the individual fractions scatter; the table's 1008 GeV row is shown only to indicate the size of the tail. The tail above 500 GeV is **0.93%** of the lab total, and above 1000 GeV is **0.04%**. The differential and acceptance plots are therefore truncated at 500 GeV, with the cumulative plots showing the full range.

### Cumulative integral

| cut | `≤ 150` GeV | `≤ 200` GeV | `≤ 250` GeV | `≤ 500` GeV | `≤ 1000` GeV |
|---|---:|---:|---:|---:|---:|
| no eta | 6,791.21 pb (49.0%) | 10,853.52 pb (78.3%) | 12,386.88 pb (89.3%) | 13,740.60 pb (99.1%) | 13,860.42 pb (100.0%) |
| lab | 4,980.72 pb (47.0%) | 8,149.20 pb (76.9%) | 9,381.80 pb (88.5%) | 10,500.25 pb (99.1%) | 10,594.87 pb (100.0%) |
| CM | 6,791.21 pb (49.3%) | 10,853.52 pb (78.8%) | 12,386.88 pb (90.0%) | 13,673.24 pb (99.3%) | 13,761.51 pb (100.0%) |

The lab cut removes a roughly uniform fraction across the dominant region: the cumulative fractions are close to the no-eta ones. That is why the lab correction is close to a global factor, and why the differential shape alone cannot explain the 14% normalization gap.

### Integration and interpolation checks

The official 100-point trapezoid is **13,866.5928 ± 0.7333 pb**. Fine-grid (0.5 GeV) alternatives:

| method | total (pb) | difference from official |
|---|---:|---:|
| official 100-point trapezoid | 13,866.593 | — |
| fine-grid linear, 0.5 GeV (3,773 points) | 13,865.661 | −0.932 pb (−0.0067%) |
| fine-grid linear, 0.02 GeV (94,301 points) | 13,866.591 | −0.0013 pb (−0.00001%) |
| fine-grid PCHIP, 0.5 GeV | 13,869.784 | +3.19 pb (+0.023%) |
| fine-grid PCHIP, converged (0.02 GeV) | 13,870.768 | +4.18 pb (+0.030%) |
| fine-grid log-linear, 0.5 GeV | 13,818.392 | −48.2 pb (−0.35%) |
| fine-grid log-PCHIP, 0.5 GeV | 13,859.399 | −7.19 pb (−0.052%) |
| Simpson on log-PCHIP, 0.5 GeV | 13,860.205 | −6.39 pb (−0.046%) |

The 0.5 GeV grid is exactly **3,773 points** from 114.0 to 2000.0, with uniform 0.5 GeV spacing. It does not include the scan nodes, so its linear-trapezoid integral differs from the official exact piecewise-linear integral by −0.93 pb. As the spacing is refined, the linear check converges to the official value:

| spacing (GeV) | points | linear − official (pb) | PCHIP − official (pb) |
|---:|---:|---:|---:|
| 2.0 | 944 | −15.132 | −11.343 |
| 1.0 | 1,887 | −3.303 | +0.097 |
| 0.5 | 3,773 | −0.932 | +3.191 |
| 0.25 | 7,545 | −0.132 | +3.934 |
| 0.1 | 18,861 | −0.0338 | +4.138 |
| 0.05 | 37,721 | −0.0085 | +4.167 |
| 0.02 | 94,301 | −0.0013 | +4.176 |

At 0.02 GeV the linear check is within 0.0013 pb of the official result. This confirms that the official 100-point trapezoid is the exact integral of the piecewise-linear interpolant; the 0.5 GeV difference is a discretization of the check, not an error in the official number. The residual interpolation-model uncertainty is the converged PCHIP difference, **+4.18 pb (+0.030%)**, which is still negligible. The log-linear model is a bad description of the integrand and is not used.

The per-channel integrals reproduce the official CSV exactly:

| channel | integral (pb) |
|---|---:|
| `gu→Zu` | 3,497.527 |
| `gd→Zd` | 3,013.248 |
| `gū→Zū` | 1,159.738 |
| `gd̄→Zd̄` | 1,664.827 |
| `uū→Zg` | 2,299.753 |
| `dd̄→Zg` | 2,231.499 |
| **total** | **13,866.593 ± 0.733** |

### Threshold correction: 113.356–114 GeV

The exact kinematic minimum for an on-shell `Z` recoiling against a massless
parton with `p_T>20` GeV is at `y^(*)=0`:

```text
√ŝ_min
=
√(M_Z^2+p_T^2)+p_T
=
√(91.188^2+20^2)+20
=
113.3555 GeV.
```

The native run card's `dsqrt_shat = 111.188` is `M_Z+p_T`, **not** the true
threshold. It is 2.1675 GeV below the true phase-space boundary, so it has no
effect; the native phase space starts at 113.3555. The partonic scan started at
114.0 and therefore missed 0.6445 GeV.

A dedicated threshold scan at 113.4, 113.6, 113.8 GeV gives the partonic
cross sections 79.94, 184.97, 246.26 pb. After luminosity convolution:

| `√ŝ` (GeV) | `dσ/d√ŝ` (pb/GeV) |
|---:|---:|
| 113.3555 | 0.000 |
| 113.400 | 29.08 |
| 113.600 | 67.18 |
| 113.800 | 89.33 |
| 114.000 | 106.03 |

Trapezoid integral 113.3555–114: **45.46 pb**.

| quantity | 114–2000 | threshold | adjusted |
|---|---:|---:|---:|
| no eta | 13,866.59 | +45.46 | **13,912.05 pb** |
| lab, stored acceptance | 10,599.26 | +32.73 | **10,631.99 pb** |
| CM | 13,764.87 | +45.46 | **13,810.33 pb** |

The threshold correction is **0.33%**. It does not close the normalization gap:
the threshold-adjusted no-eta ratio is `13,912.05/11,913=1.168`, or 14.4%.
For the lab-frame total, the threshold correction (+32.7 pb) and the
channel-dependent acceptance correction (−31 pb) nearly cancel, giving
`10,631.99×0.99706=10,600.75` pb — within 1.5 pb of the old 10,599.26.

---

## 5. Step 4 — acceptance extraction and cross-checks

### Stored acceptance

`results/eta_frame_events.csv` has 64 bins of 25 GeV from 125 to 2425 GeV, derived from the etaj=5, `ptj=20` sample (100,990 events, weight sum 11,891.7 pb). The event-level definitions are the same as in Step 1.

Applying the stored 25-GeV fractions with linear interpolation to the fine scan:

| quantity | value |
|---|---:|
| no-eta total | 13,866.5928 pb |
| lab-corrected total | **10,599.2645 pb** |
| effective `f_lab` | **0.764374** |
| CM-corrected total | 13,764.8684 pb |
| effective `f_CM` | 0.992664 |
| direct partonic with-cut scan | 13,761.97 pb |
| CM-corrected / direct | **1.00021** |

The CM cross-check is the strongest validation of the extraction: applying the separately measured CM acceptance to the no-eta scan reproduces the direct with-cut partonic scan to **0.02%**.

### Lab cross-check against native

The native both-cuts survey (`bothcuts_100k`, 100k points × 10 iterations) gives 9,142.3 ± 6.8 pb; the independent 20k×8 survey gives 9,122 ± 15.1 pb, consistent within 1.3σ. The native no-eta survey gives 11,913 ± 10.9 pb. Their ratio is

```text
(9,142.3)/(11,913) =0.7674.
```

The stored acceptance's global lab fraction is **0.7669**. Agreement: **0.07%**. This independently validates the stored acceptance normalization.

**Etaj=5 sample bias.** The stored sample was generated with `etaj=5`, not `etaj=-1`. The eta=5 cut removes 0.174% of the event weight, and every removed event would fail `∣y∣<2.5`, so the stored acceptance is high by 0.174% relative to a true no-eta sample (about 18 pb on the total). The native both-cuts/no-eta ratio is 0.7674 while the stored fraction is 0.7669; the 0.07% difference is within the roughly 0.2% statistical error of the stored sample. This bias is covered by the ±30 pb acceptance-normalization uncertainty and is not applied as a separate shift.

### Fresh no-eta sample

The current no-eta LHE sample (99,611 events, weight 11,912.391 ± 51.146 pb) gives:

| quantity | value |
|---|---:|
| global `f_lab` | 0.762090 ± 0.001765 |
| lab-cut total | 9,078.31 ± 45.34 pb |
| global `f_CM` | 0.989391 ± 0.000469 |
| CM-cut total | 11,786.01 ± 50.84 pb |
| 25-GeV binned lab total | 10,532.62 pb |

The fresh sample's lab total is 0.63% below the stored 10,599.26, but the difference from the native both-cuts 9,142.3 is only **1.4σ** given the fresh sample's 45 pb weighted-event error.

The fresh sample's **CM** fraction is more decisive: 0.989391 ± 0.000469, versus the direct partonic with-cut ratio 13,761.97/13,866.59 = **0.99245**. The difference is `0.00306`, or **6.5σ**. The stored acceptance's effective CM fraction is 0.992664, which matches the direct scan to 0.02%. Therefore the fresh no-eta sample is biased low in the acceptance (its phase-space grid under-samples the region where the CM cut rejects events), and the **stored acceptance is the reliable one**. The fresh sample is used below only for the channel-dependence **shape**, where its overall normalization cancels.

### Why the fresh sample is biased

The fresh sample comes from the `dsqrt_shat=111.188` no-eta run, whose MadEvent grid is optimized for the no-eta integrand. That grid under-samples the forward/high-`√ŝ` events that the CM and lab cuts remove, so both event-level fractions come out low. The etaj=5 sample has a grid adapted to a forward-sensitive cut and reproduces both the direct CM scan and the native lab both-cuts ratio. The stored acceptance is therefore the correct input; the fresh sample is a useful independent stress test, not a replacement.

---

## 6. Step 5 — channel dependence

The single-acceptance approximation applies one `f(√ŝ)` to the sum of all six channels. The native events show that the acceptance differs by channel:

| channel | event-weighted `f_lab` | event-weight share | partonic integral share |
|---|---:|---:|---:|
| `gu→Zu` | 0.7355 | 27.33% | 25.22% |
| `gd→Zd` | 0.7972 | 23.23% | 21.73% |
| `gū→Zū` | 0.8376 | 8.40% | 8.36% |
| `gd̄→Zd̄` | 0.8405 | 12.73% | 12.01% |
| `uū→Zg` | 0.6605 | 14.42% | 16.58% |
| `dd̄→Zg` | 0.7435 | 13.89% | 16.09% |

The `qq̄→Zg` channels have a lower lab acceptance, and the fresh event sample under-weights them relative to the partonic integrands (14.42/13.89% versus 16.58/16.09%). A single acceptance therefore does not exactly reproduce the partonic channel mix.

Applying the per-channel acceptance from the fresh sample to each channel's integrand, and taking the ratio to the single-acceptance result from the same sample:

| method | channel-dependent total (pb) | ratio to single |
|---|---:|---:|
| 25-GeV bins | 10,501.28 | 0.99703 |
| 25-GeV polynomial deg 2 | 10,508.34 | 0.99846 |
| 25-GeV polynomial deg 3 | 10,499.06 | 0.99758 |
| 25-GeV polynomial deg 4 | 10,496.14 | 0.99745 |
| 25-GeV polynomial deg 5 | 10,488.41 | 0.99690 |
| kernel `h=50` GeV | 10,495.47 | 0.99756 |
| kernel `h=100` GeV | 10,498.52 | 0.99724 |
| kernel adaptive, max(25, 0.1E) | 10,497.80 | 0.99742 |
| 50-GeV polynomial deg 3 | 10,473.97 | 0.99929 (unstable) |

The 25-GeV and kernel methods cluster at **0.9969–0.9985**, with the direct
25-GeV binning and the bootstrap at **0.9970**. The 50-GeV polynomial fits are
unstable (0.998–1.002 depending on degree) because the coarse bins smear the
channel differences; they are excluded from the central value. A Poisson
bootstrap of the events (300 resamples) gives

```text
T_channel/T_single =0.997062±0.000456,
```

i.e. a correction of `-31.3±4.8` pb on the fresh sample. Applying that ratio to the official stored-acceptance result:

```text
10,599.2645 × 0.997062 = 10,568.12 pb,
 
± 4.84 pb (stat).
```

The method spread across binning/fit choices is about **10 pb**. The channel-dependent correction is therefore

```text
Δ_channel=-31±5 (stat)±10 (method) pb,
```

or **−0.30% ± 0.11%**.

The correction is real but small. It does not change the conclusion.

---

## 7. Final result and uncertainty

> **Pre-fix section.** The totals below use the old `xfxQ(..., Q²)` convention.
> The corrected values are in the banner above and in
> [`../FINDINGS.md`](../FINDINGS.md): lab 9,113.31 pb (single acceptance),
> 9,142.25 pb (per-channel + threshold), native 9,142.3 pb.

| contribution | value |
|---|---:|
| official single-acceptance result, 114–2000 | **10,599.26 pb** |
| + threshold 113.356–114 | +32.73 pb → **10,631.99 pb** |
| channel correction (114–2000 piece) | −31.1 pb (−0.30%) |
| **threshold + channel result** | **10,600.75 pb** |
| channel stat error | ±4.8 pb |
| channel method spread | ±10 pb |
| acceptance-sample normalization uncertainty | ≈ ±30 pb (0.3%) |
| **recommended final value** | **10,601 ± 30 pb** |

The old 10,599.26 is within **1.5 pb** of the threshold+channel result. The two corrections nearly cancel: the threshold adds 32.7 pb, the channel-dependent acceptance removes 31.1 pb. The no-eta reference after the threshold correction is **13,912.05 pb**. The old result is therefore more robust than the separate corrections suggested; the correct statement is that the old number was accurate but carried an unquoted ~0.3% acceptance systematic.

The separate `etaj=5` bias is not applied as an additional shift. If it were applied naively (−0.174%, about −18 pb), the central value would move to about **10,550 pb**, still within the quoted ±30 pb acceptance uncertainty. The native both-cuts/no-eta ratio (0.7674) is 0.07% above the stored fraction (0.7669), so the two effects are consistent with the stored sample's statistical error; they should not be double-counted.

### Comparison to native

Native both-cuts: **9,142.3 ± 6.8 pb**.

| comparison | ratio | native deficit |
|---|---:|---:|
| old hybrid / native | 10,599.26 / 9,142.3 = 1.1594 | 13.7% |
| threshold + channel / native | 10,600.75 / 9,142.3 = 1.1596 | 13.8% |
| no-eta partonic (threshold-adjusted) / native no-eta | 13,912.05 / 11,913 = 1.1678 | 14.4% |

The final acceptance factor is `10,600.75/13,912.05=0.7620`, and the native factor is `9,142.3/11,913=0.7674`; their ratio is 0.9929. The no-eta 16.8% gap times this 0.7% acceptance difference gives the observed 15.96% gap. The eta frame is not the cause of the normalization difference; the native low-`√ŝ` integration deficit is.

---

## 8. Reproduction commands

All commands are run from `madgraph_test` with

```bash
export LHAPDF_DATA_PATH='C:\Users\laiji\test\lhapdf-local\share\LHAPDF'
PY="python3"
```

```bash
$PY verification/parse_events.py
$PY verification/build_integrands.py
$PY verification/check_luminosity.py
$PY verification/check_integration.py
$PY verification/channel_acceptance.py
$PY verification/kernel_acceptance.py
$PY verification/fit_acceptance.py
$PY verification/bootstrap_channel.py
$PY verification/build_differential.py
$PY verification/plot_differential.py
```

Outputs:

- `verification/events_flat.csv` — per-event kinematics and weights
- `verification/channel_integrands.csv` — `L_cσ_c` at all 100 energies
- `verification/differential.csv` — `dσ/d√ŝ`, `f_lab`, `f_CM`, lab/CM differentials
- `verification/hadronic_differential.png` / `.pdf` — four-panel figure
- `verification/bootstrap_channel.py` output — channel-correction bootstrap

---

## 9. Caveats

1. **Hybrid, not pure partonic.** The acceptance is measured from native events and applied as a function of `√ŝ`. It assumes the acceptance depends only on `√ŝ` within a 25-GeV bin, not on the partonic scattering angle. The channel-dependent test relaxes the channel part of that assumption but not the angular part.
2. **Stored sample vs fresh sample.** The stored etaj=5 sample is validated by the direct CM scan (0.02%) and the native lab both-cuts ratio (0.07%). The fresh no-eta sample is biased low (CM fraction 6.5σ below the direct scan) and is used only for the channel shape.
3. **High-`√ŝ` acceptance.** The 25-GeV bins above about 800 GeV have few events and the fractions scatter (including 0 at 2000 GeV). The contribution above 500 GeV is 0.93% of the lab total; above 1000 GeV it is 0.04%. The channel-dependent correction and the acceptance fits are insensitive to this tail at the few-pb level.
4. **Weighted events.** The event weights are not uniform (`P1_qq_zg` has 19 distinct weights), so the effective sample size is about 54k, not 100k. All quoted errors account for this where possible.
5. **The 2D histogram check was discarded.** It was too coarse; the exact `u`-integral is the valid Jacobian check.
6. **The channel-dependent correction uses the fresh sample's shape.** Its overall normalization is biased, but the channel/single ratio cancels that normalization. The residual shape bias is covered by the ±10 pb method spread.
