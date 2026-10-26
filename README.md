# Conditional Normalizing Flows for Monte Carlo Integration

This repository studies a conditional normalizing-flow proposal for families
of multidimensional integrals,

$$
I(c)=\int_{[0,1]^d} f(x\mid c)\,dx.
$$

Instead of training a new sampler for every condition $c$, one model is
trained across a continuous condition domain and reused at held-out conditions.
The network does **not** replace the integrand or directly predict the answer.
It learns where to sample; ordinary importance weights retain the original
integrand in the estimator.

## Main idea: conditional marginals plus a copula

The proposal separates individual coordinate shapes from their dependence:

$$
q_\theta(x\mid c)
=c_\theta(u_1,\ldots,u_d\mid c)
 \prod_{i=1}^d q_{i,\theta}(x_i\mid c),
\qquad
u_i=F_{i,\theta}(x_i\mid c).
$$

- **Conditional marginals** learn how each coordinate changes with the
  condition. They use monotone cubic-Hermite spline maps on $[0,1]$.
- **The autoregressive copula** learns the remaining conditional dependence
  among the marginal rank coordinates $u_i$.
- **A defensive uniform mixture** preserves support throughout the unit
  hypercube and reduces the risk of missing regions assigned very little mass
  by the current flow.

For samples $x_n\sim r_\theta(\cdot\mid c)$, the final estimate is

$$
\widehat I_N(c)=\frac1N\sum_{n=1}^N
\frac{f(x_n\mid c)}{r_\theta(x_n\mid c)}.
$$

Thus the learned model controls variance, while the importance ratio corrects
for proposal mismatch. Training requires only evaluations of the unnormalized,
nonnegative integrand; it does not require exact target samples or known
integrals.

### Why proposal quality matters

The relevant quantity for importance sampling is the second moment of the
weights,

$$
M_2(c)
=\mathbb E_{x\sim q_\theta(\cdot\mid c)}
\left[
\left(\frac{f(x\mid c)}{q_\theta(x\mid c)}\right)^2
\right].
$$

For $N$ independent proposal samples,

$$
\operatorname{Var}\!\left[\widehat I_N(c)\right]
=\frac{M_2(c)-I(c)^2}{N},
\qquad
\frac{\mathrm{ESS}}{N}\approx \frac{I(c)^2}{M_2(c)}.
$$

A proposal can therefore assign approximately the correct total probability
to a region and still perform poorly if it places that probability at the
wrong locations. Rare points with very large $f/q$ then dominate $M_2$ and
the integral estimate. The flow is useful because it moves proposal mass
toward the important geometry while the importance weights keep the estimator
correct.

## Training workflow

The current workflow uses:

1. condition sampling across the training domain;
2. self-normalized importance-weighted target approximations;
3. conditional marginal learning;
4. autoregressive copula learning in marginal-rank coordinates;
5. repeated proposal and training-sample refreshes;
6. full-target importance-weighted KL refinement; and
7. optional chi-squared/weight-variance refinement, accepted only when
   held-out effective sample size improves.

Annealed intermediate targets help move from broad initial sampling toward a
sharp final integrand. Fresh evaluation samples are used for reported integral
estimates.

See [Methods and applications](docs/METHODS_AND_APPLICATIONS.md) for the full
mathematical and implementation description.

## Geometry and topology repair

The 32-dimensional topology studies separate two failure modes:

- **Topology ladder:** the target has two disconnected branches with the same
  internal structure. The global conditional NF learned approximately the
  correct probability in each branch, but its geometry inside the branches was
  inaccurate.
- **Four-structure benchmark:** one condition contains a sharp peak, curved
  ridge, curved sheet, and ring. The global NF collapsed almost entirely onto
  the curved sheet and assigned negligible probability to the other three
  structures.

### Topology ladder

The repair procedure freezes the global NF, finds conditions with low ESS or
large $M_2$, and trains conditional specialists in small neighborhoods around
those conditions. At evaluation time, the proposal combines the nearby
specialist with the protected global proposal.

![Topology target, global NF, and specialist-repaired NF](results_public/repair_comparisons/conditional_topology_global_vs_repaired.png)

The specialist-repaired NF reduced pairwise total variation from
$0.84$--$0.87$ to $0.07$--$0.09$. In repeated frozen importance-sampling
audits, $M_2$ fell from roughly $10^4$ to order one and normalized ESS rose
from roughly $10^{-4}$ to $0.4$--$0.7$.

![Topology importance-sampling comparison](results_public/repair_comparisons/conditional_topology_integration_weights.png)

### Four different structures

Independent coverage samples detected four separated regions without supplying
structure labels or the number of regions. The global curved-sheet component
was kept frozen, while three specialists were trained on the missing regions.
The final proposal mixes the protected global component, the specialists, and
a small uniform safety component.

![Four-structure target, global NF, and specialist-repaired NF](results_public/repair_comparisons/four_structure_protected_comparison.png)

The global NF placed $99.4\%$ of its probability on the curved sheet. The
specialist-repaired NF assigned approximately $24$--$25\%$ to each structure,
close to the exact $25\%$ masses:

![Probability assigned to each structure](results_public/repair_comparisons/four_structure_protected_masses.png)

Coverage is no longer the main failure. The formal audit gives median
$M_2\approx2.21\times10^3$; the largest contributions come from the ring
($1.68\times10^3$) and curved ridge ($1.13\times10^2$). The current work
therefore focuses on improving their local geometry without damaging the
already successful sharp-peak and curved-sheet components.

The original plotting scripts use the label "repaired atlas." In this README,
the more descriptive name **specialist-repaired NF** refers to the same
proposal.

## Applications and validation

### Scattering-process integration

The conditional variable is the partonic center-of-mass energy
$c=\sqrt{\hat s}$. MadGraph supplies generated matrix elements, while
process-specific maps transform unit-hypercube coordinates into physical
phase-space momenta and Jacobians.

| Process | Dimension | Held-out ESS/N | Validation |
|---|---:|---:|---|
| $gg\to t\bar t$ | 2 | 0.981--0.998 | randomized Sobol |
| $gg\to t\bar tH$ | 5 | 0.959--0.987 | two independent NF/Sobol evaluations |
| $u\bar u\to W^+W^-\to e^+\nu_e\mu^-\bar\nu_\mu$ | 8 | 0.927--0.941 | randomized Sobol plus a higher-statistics check |
| fully decayed $gg\to t\bar tH$ | 20 | 0.885--0.941 | pointwise matrix element, Sobol, and refined MadEvent |

For the 8D benchmark, a provisional $2.30\sigma$ difference at 300 GeV was
retested with higher statistics. The result became a $0.18\sigma$ difference:

$$
\sigma_{\rm NF}=0.0529874\pm0.000013\ \mathrm{pb},\qquad
\sigma_{\rm Sobol}=0.0529849\pm0.0000031\ \mathrm{pb}.
$$

### Synthetic 32-dimensional stress test

The `sparse_wave_32` benchmark has a known unit integral and a strongly
dependent conditional autoregressive structure. One frozen conditional NF and
one fixed VEGAS setup were tested on the same 160 held-out conditions with
approximately one million evaluations per method and condition.

| Method | Mean estimate | Mean absolute error | Mean ESS/N | 95% coverage |
|---|---:|---:|---:|---:|
| conditional NF | 0.999254 | 0.001367 | 0.5328 | 0.7875 |
| tested VEGAS configuration | 0.0000135 | 0.999987 | -- | 0.0000 |

The experiment shows that the learned autoregressive proposal represents the
narrow correlated target far better than this particular axis-adaptive VEGAS
configuration under the assigned budget. It is not a universal claim about
all VEGAS implementations. The NF coverage is also below the nominal 0.95,
exposing residual uncertainty calibration and rare-weight-tail issues.

## Repository map

```text
nf/                         conditional spline implementations
madgraph_process_integrand.py
                            configurable matrix element and phase-space layer
madgraph_tth_integrand.py   resonance-aware 20D ttH phase-space implementation
experiments/                training, validation, and benchmark drivers
configs/processes/          process definitions for the generic campaign
tests/                      spline and phase-space tests
docs/                       methods, validation, and reproduction notes
results_public/             curated CSV/JSON summaries and diagnostic plots
```

Important documents:

- [Methods and applications](docs/METHODS_AND_APPLICATIONS.md)
- [Validation results](docs/VALIDATION_RESULTS.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)

## Scope and limitations

The current results establish numerical correctness and conditional reuse for
the tested benchmarks. They do not establish a universal event generator or an
end-to-end speedup for arbitrary calculations.

- Training cost is excluded from quoted online-only timing comparisons.
- Most evaluations reuse one trained checkpoint; they are not yet a systematic
  study over independently trained models.
- Scattering calculations are parton-level and leading order, with specialized
  phase-space maps and no hadronic PDFs in the generic process sweep.
- The present formulation assumes a nonnegative integrand and does not yet
  handle signed subtraction contributions at higher perturbative orders.
- The 32D experiment reveals undercoverage even though its aggregate point
  estimates are accurate.

The validated one-dimensional cubic inverse should not be modified except for
its existing safeguard on the physical interval $[0,1]$.
