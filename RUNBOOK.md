# RUNBOOK — reproducing the Z + 1 jet results

All commands are run from this folder unless stated otherwise.  `PY` is a
Python 3.10 interpreter with the packages in `requirements.txt`; `MG5` is the
MadGraph5_aMC@NLO 3.6.7 entry point.

## 0. Prerequisites

- Python 3.10 with numpy, scipy, pandas, matplotlib, torch.
- LHAPDF 6 with `NNPDF23_lo_as_0130_qed` (ID 247000).  Recommended:
  `conda install -c conda-forge lhapdf`, then
  `lhapdf install NNPDF23_lo_as_0130_qed`.
- MadGraph5_aMC@NLO 3.6.7 (for the standalone and the native samples).
- A CUDA GPU is optional for training/integration; `--device cpu` works but is
  much slower.

```bash
export LHAPDF_DATA_PATH=/path/to/lhapdf/share/LHAPDF
export PY=python3
export MG5=/path/to/MG5_aMC_v3_6_7/bin/mg5_aMC
```

Check the PDF set:

```bash
$PY -c "import lhapdf; p=lhapdf.mkPDF('NNPDF23_lo_as_0130_qed',0); print(p.alphasQ(91.188))"
```

Expected `alpha_s ~ 0.130`.

## 1. Generate the Z+1 standalone

The config `configs/processes/zj_hadronic_low.json` describes the six
subprocess groups of `p p > z j`:

```bash
cd conditional-nf-integration
$PY experiments/setup_madgraph_process.py \
  --config configs/processes/zj_hadronic_low.json --mg5 "$MG5"
cd ..
```

This writes `conditional-nf-integration/generated/zj_hadronic_standalone/`
with one `SubProcesses/P1_*` directory per channel and compiles the
`matrix2py` wrappers.  On Windows use `madgraph_windows_build.py
--standalone generated/zj_hadronic_standalone` if the MG5 makefiles do not
produce the `.pyd` modules.

## 2. Train the 12 models

Six channels x two sqrt(s-hat) ranges, balanced profile.  Every training
process must run with the CWD inside the standalone (MadGraph resolves
`Cards/ident_card.dat` relative to it):

```bash
cd conditional-nf-integration/generated/zj_hadronic_standalone
for cfg in zj_hadronic_low zj_hadronic_high; do
  for ch in gu_zu gd_zd gux_zux gdx_zdx uux_zg ddx_zg; do
    $PY -u ../../experiments/run_madgraph_process_conditional_copula.py \
      --config ../../configs/processes/${cfg}.json --channel $ch \
      --speed-profile balanced --device cuda \
      --batch-size 30000 --is-points 128 --me-chunk 512 \
      --output ../../results/${cfg}_${ch}_copula_balanced_gpu
  done
done
cd ../../..
```

Outputs:

```text
conditional-nf-integration/results/zj_hadronic_{low,high}_{channel}_copula_balanced_gpu/
  madgraph_zj_hadronic_{low,high}_{channel}_2_final_model.pt
  madgraph_zj_hadronic_{low,high}_{channel}_2_final_summary.csv
```

A single GPU takes roughly 10-15 h for all 12 models; the four `gq` channels
are the slowest.  The model files are not included in this folder.

## 3. Validate the models

NF vs equal-budget uniform Monte Carlo (read the ESS and relative stderr):

```bash
cd conditional-nf-integration/generated/zj_hadronic_standalone
$PY -u ../../experiments/validate_madgraph_process_integral.py \
  --config ../../configs/processes/zj_hadronic_high.json --channel gu_zu \
  --checkpoint ../../results/zj_hadronic_high_gu_zu_copula_balanced_gpu/madgraph_zj_hadronic_high_gu_zu_2_final_model.pt \
  --nf-evals 200000 --gpu-chunk 20000 --no-sobol --compare-uniform \
  --device cuda --output ../../results/zj_hadronic_high_gu_zu_validation
```

NF vs exact MadEvent at fixed sqrt(s-hat) (one energy or a list):

```bash
$PY ../../experiments/validate_madgraph_process_with_madevent.py \
  --config ../../configs/processes/zj_hadronic_high.json --channel gu_zu \
  --mg5 "$MG5" --energies 250 400 650 1000 1500 1950 \
  --alpha-s 0.13 --ren-scale 91.188 \
  --output ../../results/zj_hadronic_high_gu_zu_madevent
```

Repeat for the six channels.  The stored comparison CSVs are in
`conditional-nf-integration/results/zj_hadronic_madevent_interpolation/`.

## 4. Hadronic convolution (NF)

```bash
cd conditional-nf-integration/generated/zj_hadronic_standalone
$PY -u ../../experiments/integrate_hadronic_process.py \
  --device cuda --output ../../results/zj_hadronic_integral
cd ../../..
```

This evaluates each NF on 60 log-spaced points per range, multiplies by the
LHAPDF luminosity, integrates over sqrt(s-hat), and adds the
0.1389 +/- 0.00026 pb tail.  With the cut configs the output is the
**partonic-CM convolution**, 11752.4 pb, and is *not* the lab-frame result; see
the next section.  The pre-fix 13765.5 pb value (which passed `Q^2` to
`xfxQ`) is deliberately not included.

The per-channel interpolation tables used by the lab-frame analysis are
`results/zj_hadronic_integral/interpolation_{low,high}.csv` (included here, so
the headline can be reproduced without retraining).

## 5. Native and exact reference samples (optional for the headline)

```bash
cd madgraph_test

# Exact partonic no-eta scan (lpp=0)
$MG5 generate_partonic.mg5
$PY prepare_run.py partonic_zj
$PY scan_partonic.py --energy-file results/fine_energy_grid.txt \
  --output results/scan_noeta_fine.csv

# Native no-eta pp sample (lpp=1)
$MG5 generate_native_lhapdf.mg5
$PY prepare_run.py native_zj_lhapdf
$PY run_native_nocuts.py --me-dir native_zj_lhapdf --pdlabel lhapdf \
  --lhaid 247000 --lhapdf-data-path "$LHAPDF_DATA_PATH" \
  --ptj 20 --etaj -1 --points 10000 --iterations 8 --accuracy 0.002 \
  --run-name noeta
$PY analyze_eta_frame_events.py
$PY verification/parse_events.py
cd ..
```

The packaged `results/scan_noeta_fine.csv`, `results/eta_frame_events.csv`,
and `verification/cm_fail_lab_ratio.csv` make these steps unnecessary unless
the native sample itself is being re-derived.

## 6. Analysis chain

```bash
cd madgraph_test

# Exact convolution and the single-acceptance lab correction
$PY integrate_and_plot.py --scan results/scan_noeta_fine.csv \
  --output-dir results/noeta_fine
$PY apply_lab_eta_correction.py --scan results/scan_noeta_fine.csv \
  --output-dir results/noeta_fine
# The script prints the 114-2000 GeV integral (9113.31 pb); adding the
# 113.4-114 GeV threshold piece (24.59 pb) gives the quoted 9137.90 pb.

# Per-channel acceptance (needs verification/events_flat.csv, see §5)
$PY verification/channel_acceptance.py
$PY verification/bootstrap_channel.py

# Numerical checks
$PY verification/check_luminosity.py
$PY verification/check_fine_grid.py
$PY verification/check_integration.py

# NF lab-frame total (uses the cached cm_fail_lab_ratio.csv)
cd ..
$PY reproduce_headline.py
```

The last command prints the per-channel NF contributions and the final
`NF lab + threshold` value.  Expected output:

```text
NF CM total:  11752.44 pb (reference)
NF lab(CM-cut) 114-2000: ~9033 +/- ... pb
CM-fail but lab-pass tail: +~88 pb
NF lab full 114-2000: ~9121 pb
NF lab + threshold: 9155.14 +/- 8.92 pb
exact partonic lab (per-channel + threshold): 9,142.25 pb
```

## 7. Expected results

| quantity | value |
|---|---:|
| native `p p > Z j` (fixed scale, `nn23lo1`) | 9142.3 +/- 6.8 pb |
| exact partonic + LHAPDF, single acceptance | 9137.90 pb |
| exact partonic + LHAPDF, per-channel acceptance | 9142.25 pb |
| NF + LHAPDF, lab frame | 9155.14 +/- 8.92 pb |
| NF / native | 1.00140 |

## 8. Troubleshooting

- **`lha_read` / `ident_card.dat`**: run every integrand/training process with
  the CWD inside `generated/zj_hadronic_standalone`.  Running from the repo
  root hangs or fails.
- **Wrong PDF**: `nn23lo1` is LHAPDF ID 247000 (LO, alpha_s = 0.130).  ID
  230000 is a different NLO set.
- **Frame mismatch**: `etaj` is applied in the lab frame by MadGraph and in
  the partonic CM by the NF.  Do not compare the two without the acceptance
  factor of `apply_lab_eta_correction.py` / `apply_lab_frame_nf.py`.
- **No `events_flat.csv`**: the event-level checks need the native sample from
  §5; the headline runs from the cached ratio table.
- **Windows**: the local LHAPDF shim needs its DLL (not shipped); use a conda
  LHAPDF install, or build `lhapdf-local/src/lhapdf_ctypes.cpp` as described in
  `lhapdf-local/README.md`.
