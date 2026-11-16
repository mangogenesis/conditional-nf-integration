# madgraph_test

A self-contained MadGraph test of `Z + 1 jet` at 13 TeV: native
`p p > Z j` versus a fixed-`√ŝ` partonic scan convolved with
LHAPDF. Read [`FINDINGS.md`](FINDINGS.md) for the summary and
[`PRD.md`](PRD.md) for the full method, commands, and caveats. The detailed
lab-frame audit is in [`verification/VERIFICATION.md`](verification/VERIFICATION.md);
the normalization-gap resolution is in
[`verification/NORMALIZATION_GAP.md`](verification/NORMALIZATION_GAP.md).

## Latest result

The earlier ~14% "native deficit" was a **PDF-scale bug in our convolution**:
`pdf.xfxQ(..., q2)` evaluates at `Q=q2` (GeV), so passing
`q2 = 91.188**2` used `Q=8315.25` GeV instead of `Q=M_Z`. After switching
to `pdf.xfxQ2(..., q2)`:

| scenario | native (pb) | corrected partonic (pb) | agreement |
|---|---:|---:|---:|
| no eta, 114–2000 | 11,876.23 (events ≥114) | **11,876.96 ± 0.61** | 0.99994 |
| no eta, threshold-adjusted | 11,912.39 ± 10.9 | **11,910.79** | 0.99987 |
| lab, per-channel acceptance + threshold | 9,142.3 ± 6.8 | **9,142.25** | 0.99999 |
| lab, NF event-level + CM-fail tail | 9,142.3 ± 6.8 | **9,155.14 ± 8.92** | 1.00140 |
| CM with-cut, exact convolution | — | **11,753.90** | — |
| NF + LHAPDF, CM with-cut | — | **11,752.44 ± 0.82** | NF/exact 0.99988 |

Old wrong values: no-eta 13,866.59, lab 10,599.26, CM 13,761.97, NF 13,765.5.
The full-resolution report is
[`verification/NORMALIZATION_GAP.md`](verification/NORMALIZATION_GAP.md).

The lab-frame NF result is **9,155.14 ± 8.92 pb**: the NF CM samples are
boosted event-by-event, the lab cut is applied, and the small CM-fail tail is
added from native events. It is 0.140% above native 9,142.3 ± 6.8 pb, i.e.
within the NF-acceptance and event-sample systematics. The exact partonic
per-channel lab result is 9,142.25 pb. See
[`FINDINGS.md`](FINDINGS.md) and
[`verification/NF_LAB_FRAME.md`](verification/NF_LAB_FRAME.md).

## Quick start

```bash
cd madgraph_test
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
export TMP="/tmp"
export TEMP="$TMP"; export TMPDIR="$TMP"
export LHAPDF_DATA_PATH="/path/to/lhapdf/share/LHAPDF"

"python3" scan_partonic.py
"python3" integrate_and_plot.py \
  --scan results/scan_noeta_fine.csv --output-dir results/noeta_fine
"python3" apply_lab_eta_correction.py \
  --scan results/scan_noeta_fine.csv --output-dir results/noeta_fine
```

## Key outputs

```text
results/scan_noeta_fine.csv
results/noeta_fine/total_integral.csv / .json
results/noeta_fine/lab_eta_correction.csv / .json
results/differential_compare.csv / .json / .png
results/total_integral.csv / .json
verification/differential.csv
verification/channel_integrands.csv
verification/events_flat.csv
verification/hadronic_differential.png
verification/normalization_gap.png
```

## What the test shows

### Three cut scenarios

| scenario | native MadGraph | partonic convolution | status |
|---|---:|---:|---|
| no cuts (`etaj=-1`, `ptj=0`) | 1.485e6 ± 43,311 pb | no finite result | divergent |
| `p_T` only (`etaj=-1`, `ptj=20`) | 11,913 ± 10.9 pb | coarse 10,062.31; **fine 11,876.96 ± 0.61** | 0.3% |
| both cuts (`etaj=2.5`, `ptj=20`) | 9,142.3 ± 6.8 pb (lab) | CM 11,753.55; **single lab 9,113.31**; **per-channel + threshold 9,142.25** | 0.0005% |

### Eta frame

The native run applies `∣η_j∣<2.5` in the **lab frame**. The partonic scan
uses `lpp=0`, so its `etaj=2.5` is the **partonic CM** rapidity. The native
event-level lab fraction is **0.7666**; the acceptance-weighted fine lab
fraction is **0.7673**. The native eta effect is 23% (11,913 → 9,142); the
partonic CM effect is only 1% (11,876.96 → 11,753.55). The NF/CM with-cut
number is not the lab-frame result.

### No-cut limit

Literal no cuts (`ptj=0`) is divergent. Native returns 1.485e6 ± 43,311 pb,
but the grid copies differ by orders of magnitude; the partonic no-cut probe
has no finite result. There is no finite no-cut total to compare.

### Finite regulator

The smallest full-scan regulator is `ptj=5 GeV`: native **40,116 ± 46 pb**;
corrected partonic convolution **19,509.42 ± 3.7 pb** (coarse 50-point grid).
This is a regulator comparison, not a no-cut result.

### PDF-scale audit

The local LHAPDF shim now exposes:

```python
pdf.xfxQ(pid, x, q)      # q in GeV
pdf.xfxQ2(pid, x, q2)    # q2 in GeV^2
```

All convolution callers in `madgraph_test` use `xfxQ2`. The upstream
`conditional-nf-integration/experiments/integrate_hadronic_process.py` was also
patched. The native `q2fact` is fixed to `91.188^2` and was correct all along.
