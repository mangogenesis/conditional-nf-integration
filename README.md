# Conditional Normalizing Flows for Amortized Monte Carlo Integration

This repository studies conditional normalizing-flow proposals for families of
expensive multidimensional integrals.  A model is trained over a continuous
condition, then reused at held-out conditions without retraining.  Importance
weights preserve the integral estimator while the learned proposal reduces its
variance.

The current physics case study is the fully decayed partonic process

```text
g g > t t~ h,
(t > b w+, w+ > e+ ve),
(t~ > b~ w-, w- > mu- vm~),
(h > b b~)
```

with partonic center-of-mass energy `sqrt(s_hat)` as the condition.  The
resonance-aware phase-space map has 20 integration dimensions.  The proposal
combines learned one-dimensional cubic-spline marginals with an autoregressive
copula and a defensive uniform component.

## Current validated result

The generated matrix element was compared point by point against an independent
MadGraph standalone export at 400 phase-space points spanning 800--2000 GeV.
The maximum relative difference was exactly zero at double precision.

At the difficult high-energy conditions, three independent integration paths
are statistically consistent:

| Energy | Conditional NF [fb] | Randomized Sobol [fb] | Refined MadEvent [fb] |
|---:|---:|---:|---:|
| 1990 GeV | 0.383909 ± 0.000057 | 0.383847 ± 0.000129 | 0.383900 ± 0.000227 |
| 2000 GeV | 0.383434 ± 0.000056 | 0.383419 ± 0.000196 | 0.383050 ± 0.000253 |

The conditional NF has normalized ESS near 0.88 at these conditions.  Its
post-training integration took about 18 seconds for 500,000 evaluations,
whereas the explicitly surveyed/refined MadEvent calculations took roughly
114--179 seconds with comparable uncertainty.  Training cost is excluded from
this online comparison and must be included in amortization and break-even
analyses.

## Repository map

```text
nf/                         flow and spline implementations
madgraph_tth_integrand.py   resonance-aware phase-space map and ME wrapper
experiments/                reproducible training and validation drivers
madgraph_cards/             process-generation cards
tests/                      spline and phase-space tests
docs/                       results and reproduction notes
results_public/             compact CSV/JSON summaries and selected plots
```

The validated one-dimensional cubic inverse should not be modified except for
its existing clamp to `[0,1]`.

## Principal experiment drivers

- `experiments/setup_madgraph_tth_benchmark.py`: generate the standalone ME.
- `experiments/run_madgraph_tth_conditional_copula.py`: train the conditional proposal.
- `experiments/validate_madgraph_tth_integrand.py`: phase-space sanity checks.
- `experiments/validate_madgraph_tth_pointwise.py`: independent pointwise ME comparison.
- `experiments/validate_madgraph_tth_integral.py`: randomized Sobol and NF validation.
- `experiments/validate_madgraph_tth_with_madevent.py`: explicit MadEvent survey/refine validation.
- `experiments/benchmark_nf_vs_madevent_tth.py`: repeated online performance comparison.
- `experiments/reanalyze_nf_vs_madevent_errors.py`: historical error audit.

See `docs/VALIDATION_RESULTS.md` for interpretation and
`docs/REPRODUCIBILITY.md` for the exact validation sequence.

## Status

The ttH case study is validated.  Claims of broad process-level generality are
not yet made.  The next study will apply the same frozen protocol to several
scattering processes, multiple training seeds, and held-out conditions while
reporting training amortization and equal-precision comparisons.
