#!/usr/bin/env python3
"""Accelerated, checkpoint-compatible entry point for the copula benchmark.

The original benchmark remains the source of all architecture, training, and
evaluation choices.  This entry point only batches independent cubic inverse
evaluations during teacher-forced likelihood computation.
"""

from __future__ import annotations

import run_bayesian_selected_copula_benchmark as baseline

from nf.vectorized_cubic import (
    vectorized_copula_log_prob,
    vectorized_marginal_inverse,
)


def _fast_log_prob_u(self, u, cond):
    return vectorized_copula_log_prob(self, u, cond)


def _fast_x_to_u_and_log_marginal(self, x, cond):
    return vectorized_marginal_inverse(self.marginals, x, cond)


def install_fast_path() -> None:
    baseline.StructuredCopula.log_prob_u = _fast_log_prob_u
    baseline.MarginalCopulaModel.x_to_u_and_log_marginal = (
        _fast_x_to_u_and_log_marginal
    )


if __name__ == "__main__":
    install_fast_path()
    baseline.run(baseline.parse_args())
