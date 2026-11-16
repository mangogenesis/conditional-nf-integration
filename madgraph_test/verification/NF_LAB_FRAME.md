# Applying the lab-frame acceptance to the conditional NF

**Result (clean no-eta event sample):**

```text
σ_lab^NF
=9,155.14 ± 8.92 pb
```

for `p_T>20` GeV, `∣η_lab∣<2.5`, `114`–`2000` GeV plus the
exact `113.3555`–`114` GeV threshold. The error is
`√(0.77_stat^2+6.67_(tail/sample)^2+5.9_(NF accept)^2)`.

Comparisons:

| quantity | value (pb) | difference from NF |
|---|---:|---:|
| NF lab total | **9,155.14 ± 8.92** | — |
| exact partonic lab, per-channel + threshold | 9,142.25 | +12.89 (+0.141%) |
| exact partonic lab, single acceptance + threshold | 9,137.90 | +17.24 (+0.189%) |
| native both-cuts | 9,142.3 ± 6.8 | +12.84 (+0.140%, 1.14σ) |

The NF is 0.14% above native; the exact per-channel result is 0.0005% above
native. The full bug/fix narrative is in
[`../FINDINGS.md`](../FINDINGS.md).

Script: [`apply_lab_frame_nf.py`](apply_lab_frame_nf.py)  
Acceptance table: [`nf_lab_acceptance.csv`](nf_lab_acceptance.csv)  
Summary: [`nf_lab_total.json`](nf_lab_total.json)

---

## 1. Why this is not just multiplying by 0.767

The current NF is trained in the **partonic CM frame** with the CM cut
`∣η_CM∣<2.5`. It cannot sample the no-eta distribution directly.
The lab frame requires the boost `Y` from the PDF luminosity:

```text
y_lab=y_CM+Y.
```

The calculation therefore works event by event:

1. sample phase-space points from the NF at fixed `√ŝ`;
2. map each point to the jet four-momentum in the partonic CM;
3. compute `y_CM`;
4. draw the boost `Y` from the LHAPDF luminosity for the channel;
5. apply `∣y_CM+Y∣<2.5`;
6. include both beam orderings (the NF sample is one ordering; the mirror
   ordering is `y_CM→-y_CM` with the PDF weights swapped).

This gives the lab acceptance of the **CM-cut** NF sample,
`f_(lab∣CM)(√ŝ)`.

The CM-cut NF misses events with `∣y_CM∣>2.5` that still pass the lab
cut. That tail is measured from the native no-eta event sample as

```text
r(√ŝ)=(P(CM fail and lab pass))/(P(CM pass)),
```

and added as a small correction.

---

## 2. Step-by-step checks

### 2.1 NF CM total

| quantity | value |
|---|---:|
| NF CM total, 114–2000 | **11,752.44 ± 0.82 pb** |
| exact partonic CM with-cut | **11,753.90 pb** |
| ratio | **0.99988** (−0.012%) |

### 2.2 NF lab acceptance vs native CM-cut events

The NF acceptance is compared with the clean no-eta event sample restricted
to `∣y_CM∣<2.5`:

| channel | `√ŝ` | native `f_(lab∣CM)` | NF `f_(lab∣CM)` |
|---|---:|---:|---:|
| `gu→Zu` | 150 | 0.7270 | 0.7204 |
| `gu→Zu` | 200 | 0.7755 | 0.7944 |
| `gu→Zu` | 300 | 0.8620 | 0.8611 |
| `gu→Zu` | 500 | 0.9152 | 0.9076 |
| `uū→Zg` | 150 | 0.6645 | 0.6753 |
| `uū→Zg` | 200 | 0.6857 | 0.6812 |
| `uū→Zg` | 300 | 0.7006 | 0.6830 |
| `uū→Zg` | 500 | 0.7611 | 0.7269 |

The agreement is 1–2.5% at most points; the 500 GeV `uū` point has only
66 events and differs by 4.5%. The integrated effect is covered by the 5.9 pb
NF-acceptance model error.

### 2.3 The two pieces

| component | value (pb) |
|---|---:|
| NF lab, CM-cut sample | **9,041.84 ± 0.77** |
| CM-fail but lab-pass tail | **+88.71** |
| NF lab full, 114–2000 | **9,130.55** |
| threshold 113.3555–114 | **+24.59** |
| **NF lab total** | **9,155.14 ± 8.92** |

Per channel:

| channel | NF CM | NF lab (CM-cut) | tail | full lab |
|---|---:|---:|---:|---:|
| `gu→Zu` | 3,196.92 | 2,383.18 | 39.95 | 2,423.13 |
| `gd→Zd` | 2,762.43 | 2,200.09 | 24.01 | 2,224.10 |
| `gū→Zū` | 999.06 | 853.59 | 1.65 | 855.25 |
| `gd̄→Zd̄` | 1,476.79 | 1,262.55 | 8.29 | 1,270.84 |
| `uū→Zg` | 1,683.12 | 1,124.84 | 8.21 | 1,133.04 |
| `dd̄→Zg` | 1,634.12 | 1,217.59 | 6.60 | 1,224.19 |

### 2.4 Independent consistency check

Applying the **NF acceptance to the exact CM integrand** (instead of the NF
CM integrand) gives **9,043.92 pb**, within 2.1 pb of the NF CM-cut result
9,041.84. The acceptance extraction is therefore internally consistent; the
NF CM shape contributes only ~0.02%.

### 2.5 Event-sample dependence of the tail

Two clean no-eta samples give different tails:

| sample | events | weight sum (pb) | CM fraction | tail (pb) |
|---|---:|---:|---:|---:|
| earlier `native_zj_nocuts` subset | 99,611 | 11,891.7 | 0.9899 | 82.53 |
| clean LHAPDF sample | 134,848 | 11,913.95 | 0.9897 | 88.71 |

The clean sample has the correct weight sum (native no-eta 11,913 ± 10.9 pb)
and the correct CM fraction (direct CM scan 0.9896). We adopt 88.71 pb and
quote the 6.18 pb spread as an event-sample systematic. The earlier
9,148.96 pb result used the 82.53 pb tail; it is superseded by 9,155.14 pb,
with the difference inside the quoted uncertainty.

---

## 3. Caveats

1. **Hybrid tail.** The current NF is CM-cut, so the 88.71 pb tail is measured
   from native events, not sampled by the NF. A pure lab-frame NF requires
   retraining without the CM eta cut, or a flow that includes the boost `Y`.
2. **NF acceptance accuracy.** The per-channel NF acceptance differs from the
   native CM-cut acceptance by 1–2.5% at individual energies; the weighted
   total is consistent to 0.14%. The quoted 5.9 pb NF-acceptance model error
   covers this.
3. **Threshold.** The 24.59 pb threshold piece is the exact partonic threshold
   scan, not an NF extrapolation below 114 GeV. The flat-acceptance
   interpolation gives 24.32 pb; the 0.3 pb difference is the method spread.
4. **Native comparison.** The NF 9,155.14 ± 8.92 pb and native
   9,142.3 ± 6.8 pb agree within 1.14σ. The exact per-channel 9,142.25 pb
   agrees with native to 0.0005%.

---

## 4. Reproduce

```bash
cd madgraph_test
export LHAPDF_DATA_PATH="/path/to/lhapdf/share/LHAPDF"
PY="python3"

# Recompute the tail and total from the saved acceptance table:
$PY - <<'PY'
import sys, csv
sys.path.insert(0, "verification")
import apply_lab_frame_nf as nfmod
rows = [
    {"channel": r["channel"], "energy_gev": float(r["energy_gev"]),
     "f_lab_given_cm": float(r["f_lab_given_cm"]), "error": float(r["error"])}
    for r in csv.DictReader(open("verification/nf_lab_acceptance.csv"))
]
nf = nfmod.read_nf()
pdf = nfmod.lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
nfmod.integrate_nf_lab(rows, nf, pdf)
PY

# Or regenerate the acceptance table (slower):
$PY verification/apply_lab_frame_nf.py --n-samples 50000
```
