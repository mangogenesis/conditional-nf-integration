# Validation record

## Scope

The target is a 20-dimensional resonance-aware integral for fully decayed
partonic `gg -> ttH`, conditional on `sqrt(s_hat)` over 600--2000 GeV.  Results
are in femtobarns unless otherwise stated.

## Validation ladder

### 1. Phase-space and integrand checks

The phase-space implementation verifies four-momentum conservation, final-state
mass shells, finite Jacobians, and the expected 20-dimensional interface.

### 2. Pointwise generated matrix element

The production standalone evaluator and an independently generated standalone
export from the same MG5 installation were evaluated on identical momenta.

| Energies | Points per energy | Maximum relative difference | Result |
|---|---:|---:|---|
| 800, 1400, 1990, 2000 GeV | 100 | 0 | PASS |

This rules out a mismatch in process generation, external-particle ordering,
parameter initialization, and generated matrix-element normalization.

### 3. Direct randomized Sobol integration

At fixed energy, independently scrambled Sobol replicates integrate the same
matrix element and phase-space Jacobian without using the trained proposal.

| Energy | Sobol estimate | Standard error | Conditional NF | NF standard error | NF ESS/N |
|---:|---:|---:|---:|---:|---:|
| 1400 | 0.396273 | 0.000212 | 0.396512 | 0.000173 | 0.9126 |
| 1990 | 0.383847 | 0.000129 | 0.383925* | 0.000197* | 0.8839 |
| 2000 | 0.383419 | 0.000196 | 0.383487* | 0.000197* | 0.8835 |

`*` Single 500,000-evaluation validation draw.  The repeated NF means used in
the final comparison are listed below.

The NF--Sobol pulls were approximately 0.87, 0.22, and 0.25 standard deviations
at 1400, 1990, and 2000 GeV respectively.

### 4. Explicitly refined MadEvent integration

Early MadEvent runs used `generate_events`; their repeated empirical scatter was
mistaken for a high-precision integration reference.  The process-specific run
card contains no `req_acc`.  The corrected validation uses MadEvent's explicit
`survey --points --iterations --accuracy` and `refine` commands and reads the
fresh `SubProcesses/results.dat` uncertainty.

| Energy | Repeated NF mean | NF empirical SEM | Sobol | Sobol error | Refined MadEvent | MadEvent error |
|---:|---:|---:|---:|---:|---:|---:|
| 1990 | 0.383909 | 0.000057 | 0.383847 | 0.000129 | 0.383900 | 0.000227 |
| 2000 | 0.383434 | 0.000056 | 0.383419 | 0.000196 | 0.383050 | 0.000253 |

At 1990 GeV all central values essentially coincide.  At 2000 GeV, NF versus
refined MadEvent differs by about 1.48 combined standard deviations and Sobol
versus MadEvent by about 1.15.  There is no statistically significant mismatch.

## Performance evidence

For 500,000 NF evaluations, repeated high-statistics runs took about 18 seconds
per condition and yielded per-run internal errors around 0.00020 fb.  The
refined MadEvent runs took about 179 seconds at 1990 GeV and 114 seconds at
2000 GeV, with errors of 0.00023 and 0.00025 fb.

This supports a substantial post-training online advantage.  It is not yet an
end-to-end speedup claim because NF training cost, hardware utilization, and
different output products must be reported separately.

## Defensible conclusion

The conditional NF importance estimator agrees with independent randomized
Sobol integration and explicitly refined MadEvent integration.  It achieves
high ESS and comparable or better online precision in substantially less time
for this case study.  Generality across scattering processes remains to be
demonstrated.
