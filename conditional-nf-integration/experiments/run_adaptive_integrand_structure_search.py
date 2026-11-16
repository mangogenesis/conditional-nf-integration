"""
run_adaptive_integrand_structure_search.py

Adaptive conditional importance-sampling structure search.

KEY PRINCIPLE
=============

The learning algorithm knows ONLY the unnormalized integrand

    f(x | c)

and the integration domain

    x in [0,1]^d.

It NEVER calls the exact target sampler.

The benchmark target sampler is therefore completely hidden from training.

Training proceeds:

    uniform proposal
        ->
    self-normalized weighted target approximation
        ->
    train marginal flows
        ->
    train full copula
        ->
    refresh weighted samples using learned NF proposal
        ->
    KL refinement
        ->
    chi^2 / weight-variance fine-tuning
        ->
    masked edge evidence
        ->
    Bayesian weak-edge screening
        ->
    candidate pruning
        ->
    ESS-only acceptance

Candidate decision tree:

    all_weak
        ->
    first_half
        ->
    second_half only when informative

Every candidate is warm-started from the SAME current accepted model.

No candidate-to-candidate warm start.

The benchmark used here is:

    regime_switch

but its exact target sampler is NOT implemented or used.

The only benchmark-specific information exposed to training is:

    regime_switch_log_integrand(x, c)

For this synthetic benchmark the true integral happens to equal 1, but
the training algorithm never uses that fact.

Run:

    python experiments/run_adaptive_integrand_structure_search.py \
        --benchmarks regime_switch
"""

import argparse
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_adam(parameters, lr):
    """Use CUDA's fused Adam kernel when supported, with a safe fallback."""
    parameters = list(parameters)
    if parameters and parameters[0].is_cuda:
        try:
            return torch.optim.Adam(parameters, lr=lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(parameters, lr=lr)


COMPILE_TRAINING_KERNELS = False


def compile_training_callable(function):
    """Compile a hot CUDA callable; Dynamo falls back to eager if unsupported."""
    if not COMPILE_TRAINING_KERNELS or not hasattr(torch, "compile"):
        return function
    return torch.compile(
        function,
        mode="reduce-overhead",
        dynamic=True,
    )


# =============================================================================
# Existing project components
# =============================================================================

from run_copula_bin_sweep import seed_everything

from run_bayesian_selected_copula_benchmark import (
    DIM,
    MarginalBank,
    StructuredCopula,
    MarginalCopulaModel,
)

from run_iterative_bayesian_edge_pruning_frozen_null import (
    DEFAULT_SEEDS,
    KEEP_PROB,
    MCMC_ITER,
    MCMC_BURN,
    MCMC_THIN,
    RHO_A,
    RHO_B,
    BAYES_RANDOM_SEED,
    PIP_REMOVE_THRESHOLD,
    IterativeMaskedChild,
    bayesian_round,
    full_ar_graph,
    copy_graph,
    count_edges,
    active_mask_vector,
    sample_training_mask,
    LR_MASKED,
    GRAD_CLIP,
)


# =============================================================================
# Global configuration
# =============================================================================

DEFAULT_OUTPUT = (
    "results/adaptive_integrand_structure_search"
)

SUPPORTED_BENCHMARKS = {
    "regime_switch",
    "sparse_wave_32",
}

# Optional post-training information.  Real matrix-element integrands generally
# do not have an analytic oracle; absence from this mapping is therefore valid.
# This mapping is never consulted during training or model selection.
ORACLE_INTEGRALS = {
    "regime_switch": 1.0,
    "sparse_wave_32": 1.0,
}

BENCHMARK_DIMENSIONS = {
    "regime_switch": 8,
    "sparse_wave_32": 32,
}

# -----------------------------------------------------------------------------
# Conditions
# -----------------------------------------------------------------------------

COND_DIM = 3

# -----------------------------------------------------------------------------
# Weighted target approximation
#
# We keep condition anchors fixed within each small importance-sampling group.
#
# For each c:
#
#     x_1,...,x_K ~ r(x|c)
#
#     w_i = f(x_i|c) / r(x_i|c)
#
# Then normalize w only WITHIN that c.
#
# This removes the unknown I(c).
# -----------------------------------------------------------------------------

# =============================================================================
# Adaptive importance-sampling training
# =============================================================================

# Initial uniform bootstrap needs many points because uniform ESS is ~0.01.
UNIFORM_BOOTSTRAP_POINTS = 512

# Tempering prevents the first uniform cache from collapsing onto one point.
MARGINAL_TEMPERATURES = (
    0.08,
    0.15,
    0.25,
    0.40,
    0.60,
    0.80,
    1.00,
)

HIGH_DIM_MARGINAL_TEMPERATURES = (
    0.01, 0.025, 0.05, 0.08, 0.12, 0.18,
    0.26, 0.38, 0.52, 0.68, 0.84, 1.00,
)
MARGINAL_RESAMPLES_PER_CONDITION = 64
# Marginal-only proposals cannot represent the target copula, so their
# normalized ESS can remain modest even after the marginals improve.  Keep
# enough points per condition to obtain a useful *raw* ESS at later bridge
# stages (about 40 at a 2% normalized ESS).
MARGINAL_ADAPTIVE_POINTS = 512
MARGINAL_TRAIN_CONDITIONS = 1024
MARGINAL_EARLY_STOP_PATIENCE = 3
MARGINAL_MIN_REL_IMPROVEMENT = 1e-3

# The next annealing temperature is selected from a frozen proposal using a
# small pilot cache.  These thresholds are in effective samples, not ratios.
TEMPERATURE_PROBE_CONDITIONS = 256
TEMPERATURE_PROBE_POINTS = 512
TEMPERATURE_TARGET_MEDIAN_RAW_ESS = 80.0
TEMPERATURE_TARGET_Q05_RAW_ESS = 25.0
TEMPERATURE_MIN_INCREMENT = 0.005
TEMPERATURE_MAX_INCREMENT = 0.20
TEMPERATURE_BISECTION_STEPS = 16

# A short full-copula fit at every annealing stage prevents the bridge proposal
# from losing all joint ESS as beta increases.  This model is only a proposal;
# it is not the sparse graph selected by the later structure search.
BRIDGE_COPULA_MAX_STEPS = 1800
BRIDGE_COPULA_CHECK_INTERVAL = 300
BRIDGE_COPULA_PATIENCE = 3
BRIDGE_COPULA_LR = 3e-4

# Every learned proposal retains a uniform component.  This guarantees support
# over the whole integration domain and lets importance weights correct the
# defensive mixture exactly.
PROPOSAL_UNIFORM_FRACTION = 0.10
PROPOSAL_MIN_UNIFORM_FRACTION = 0.02

# Bisection steps in the spline inverse. The final Newton correction is
# separate, so 8-12 steps already agree with the 40-step reference to ~1e-5
# in the log-Jacobian in float32, while costing 2-3x less per dimension.
# Evaluation keeps the 40-step default; this knob is for training.
TRAINING_INVERSE_ITERATIONS = 40

# Once an NF exists, the proposal should be much better.
NF_ADAPTIVE_POINTS = 128

# Default used by the CLI.  The first (uniform) stage overrides this with the
# larger bootstrap value below.
IS_POINTS_PER_CONDITION = NF_ADAPTIVE_POINTS

# Number of distinct conditions in one weighted training step.
#
# Initial:
#   64 * 4096 = 262,144 integrand evaluations / optimization step
#
# Later:
#   256 * 128 = 32,768 evaluations / optimization step
BOOTSTRAP_CONDITIONS_PER_STEP = 64
ADAPTIVE_CONDITIONS_PER_STEP = 256

# Fixed validation caches.
BOOTSTRAP_VAL_CONDITIONS = 256
ADAPTIVE_VAL_CONDITIONS = 1000

# Fixed cache sizes used by marginal fitting and legacy cached routines.
TRAIN_CONDITIONS = BOOTSTRAP_CONDITIONS_PER_STEP
VAL_CONDITIONS = BOOTSTRAP_VAL_CONDITIONS

# We no longer need to resample 10 target points per condition.
RESAMPLES_PER_CONDITION = None

N_ADAPTIVE_KL_ROUNDS = 3

# -----------------------------------------------------------------------------
# Standard training
# -----------------------------------------------------------------------------

TRAIN_BATCH_SIZE = 10_000

# -----------------------------------------------------------------------------
# Marginals
# -----------------------------------------------------------------------------

MARGINAL_LR = 1e-4

MIN_MARGINAL_STEPS = 1200
MAX_MARGINAL_STEPS = 6000

MARGINAL_CHECK_INTERVAL = 300

MARGINAL_REL_TOL = 0.01
MARGINAL_CONVERGENCE_PATIENCE = 2

# -----------------------------------------------------------------------------
# Adaptive KL optimization
# -----------------------------------------------------------------------------

KL_LR_LEVELS = [
    1e-2,
    3e-3,
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
]

KL_MAX_STEPS = 30_000
KL_MIN_STEPS = 1800

KL_CHECK_INTERVAL = 600

KL_SMOOTH_WINDOW = 2

KL_LR_DECAY_TOL = 0.03
KL_LR_PATIENCE = 2

KL_CONVERGENCE_TOL = 0.01
KL_CONVERGENCE_PATIENCE = 3

KL_MAX_LR_FOR_CONVERGENCE = 1e-4

# -----------------------------------------------------------------------------
# Number of proposal-adaptation stages.
#
# Stage 0:
#     uniform -> weighted target samples
#
# Stage 1:
#     NF -> weighted target samples
#
# Stage 2:
#     updated NF -> weighted target samples
# -----------------------------------------------------------------------------

N_ADAPTIVE_KL_ROUNDS = 3

# -----------------------------------------------------------------------------
# chi^2 / importance-weight fine tuning
# -----------------------------------------------------------------------------

CHI2_ENABLED = True

CHI2_LR_LEVELS = [
    3e-4,
    1e-4,
    3e-5,
    1e-5,
]

CHI2_MAX_STEPS = 6000

CHI2_CHECK_INTERVAL = 300

CHI2_CONDITIONS_PER_STEP = 512
CHI2_POINTS_PER_CONDITION = 32

CHI2_GRAD_CLIP = 10.0

# ESS validation during chi2 training
CHI2_VAL_CONDITIONS = 2000
CHI2_VAL_POINTS_PER_CONDITION = 32

# -----------------------------------------------------------------------------
# Masked evidence
# -----------------------------------------------------------------------------

MASKED_STEPS = 1000
REFINE_STEPS = 250

MASKED_KEEP_PROB = KEEP_PROB

DELTA_EVAL_SIZE = 100_000

# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------

MAX_ROUNDS = 10

ESS_TOL = 0.0

PERFORMANCE_SEED = 0
EVAL_SEED = 123456

FINAL_ESS_SIZE = 100_000


# =============================================================================
# Conditions
# =============================================================================

def sample_conditions(
    n,
    device,
):
    return torch.rand(
        n,
        COND_DIM,
        device=device,
    )


# =============================================================================
# Normal helpers
# =============================================================================

def standard_normal_cdf(
    x,
):
    return (
        0.5
        *
        (
            1.0
            +
            torch.erf(
                x
                /
                math.sqrt(2.0)
            )
        )
    )


# =============================================================================
# Truncated normal density
# =============================================================================

def truncated_normal_logpdf(
    x,
    mu,
    sigma,
):
    z = (
        x
        -
        mu
    ) / sigma

    log_base = (
        -0.5
        *
        z.square()
        -
        torch.log(
            sigma
        )
        -
        0.5
        *
        math.log(
            2.0
            *
            math.pi
        )
    )

    a = (
        0.0
        -
        mu
    ) / sigma

    b = (
        1.0
        -
        mu
    ) / sigma

    normalization = (
        standard_normal_cdf(
            b
        )
        -
        standard_normal_cdf(
            a
        )
    ).clamp_min(
        1e-12
    )

    return (
        log_base
        -
        torch.log(
            normalization
        )
    )


# =============================================================================
# regime_switch integrand
#
# IMPORTANT:
#
# This defines f(x|c), NOT an exact sampler.
#
# The training algorithm is allowed to evaluate this function.
# =============================================================================

def regime_switch_mu_sigma(
    child,
    x,
    cond,
):
    c0 = cond[
        :,
        0:1
    ]

    c1 = cond[
        :,
        1:2
    ]

    c2 = cond[
        :,
        2:3
    ]

    parent_map = {
        2: (0, 1),
        3: (2, 0),
        4: (1, 3),
        5: (4, 2),
        6: (5, 3),
        7: (6, 1),
    }

    a_idx, b_idx = (
        parent_map[
            child
        ]
    )

    a = x[
        :,
        a_idx:a_idx + 1
    ]

    b = x[
        :,
        b_idx:b_idx + 1
    ]

    gate = torch.sigmoid(
        8.0
        *
        (
            c0
            -
            0.5
        )
    )

    branch_a = torch.sin(
        2.0
        *
        math.pi
        *
        (
            a
            +
            0.08
            *
            c1
            +
            0.027
            *
            child
        )
    )

    branch_b = torch.cos(
        2.0
        *
        math.pi
        *
        (
            b
            -
            0.07
            *
            c1
            +
            0.031
            *
            child
        )
    )

    interaction = torch.sin(
        2.0
        *
        math.pi
        *
        (
            a
            -
            b
        )
    )

    signal = (
        gate
        *
        branch_a
        +
        (
            1.0
            -
            gate
        )
        *
        branch_b
        +
        0.20
        *
        interaction
    )

    mu = (
        0.50
        +
        0.28
        *
        torch.tanh(
            1.5
            *
            signal
        )
    )

    sigma = (
        0.040
        +
        0.012
        *
        c2
    )

    return (
        mu,
        sigma,
    )


def regime_switch_log_integrand(
    x,
    cond,
):
    """
    Unnormalized log f(x | c).

    Coordinates x0 and x1 are uniform roots.

    Their density contribution is 1, so log contribution = 0.

    No target normalization is used by the algorithm.
    """

    logf = torch.zeros(
        x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )

    inside = (
        (x >= 0.0)
        &
        (x <= 1.0)
    ).all(
        dim=1
    )

    for child in range(
        2,
        DIM,
    ):

        mu, sigma = (
            regime_switch_mu_sigma(
                child,
                x,
                cond,
            )
        )

        logf += (
            truncated_normal_logpdf(
                x[
                    :,
                    child:child + 1
                ],
                mu,
                sigma,
            )
            .reshape(-1)
        )

    logf = torch.where(
        inside,
        logf,
        torch.full_like(
            logf,
            -torch.inf,
        ),
    )

    return logf


def sparse_wave_32_mu_sigma(
    child,
    x,
    cond,
):
    """Two-parent nonlinear conditional used by the 32-D benchmark."""

    parent = child - 1
    skip = max(0, child // 2 - 1)
    a = x[:, parent:parent + 1]
    b = x[:, skip:skip + 1]
    c0 = cond[:, 0:1]
    c1 = cond[:, 1:2]
    c2 = cond[:, 2:3]

    phase = 0.031 * child + 0.10 * c1
    local_wave = torch.sin(2.0 * math.pi * (a + phase))
    skip_wave = torch.cos(2.0 * math.pi * (b - 0.07 * c0))
    interaction = torch.sin(2.0 * math.pi * (a - b))
    gate = torch.sigmoid(6.0 * (c0 - 0.5))
    signal = (
        gate * local_wave
        + (1.0 - gate) * skip_wave
        + 0.15 * interaction
    )
    mu = 0.50 + 0.25 * torch.tanh(1.35 * signal)
    sigma = 0.060 + 0.018 * c2 + 0.003 * float(child % 3)
    return mu, sigma


def sparse_wave_32_log_integrand(
    x,
    cond,
):
    """Normalized-by-construction 32-D density, exposed only as log f."""

    if x.shape[1] != 32:
        raise ValueError("sparse_wave_32 requires dimension 32")
    logf = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
    for child in range(2, 32):
        mu, sigma = sparse_wave_32_mu_sigma(child, x, cond)
        logf += truncated_normal_logpdf(
            x[:, child:child + 1], mu, sigma
        ).reshape(-1)
    return torch.where(
        inside,
        logf,
        torch.full_like(logf, -torch.inf),
    )


def target_log_integrand(
    benchmark,
    x,
    cond,
):
    if benchmark == "regime_switch":

        return regime_switch_log_integrand(
            x,
            cond,
        )

    if benchmark == "sparse_wave_32":
        return sparse_wave_32_log_integrand(x, cond)

    raise ValueError(
        f"Unknown benchmark: "
        f"{benchmark}"
    )


# =============================================================================
# State utilities
# =============================================================================

def cpu_state_dict(
    model,
):
    return {
        key:
            value
            .detach()
            .cpu()
            .clone()

        for key, value
        in model.state_dict().items()
    }


def load_cpu_state(
    model,
    state,
):
    current = (
        model.state_dict()
    )

    transferred = {}

    for key, value in state.items():

        if key not in current:

            continue

        if (
            tuple(
                current[key].shape
            )
            !=
            tuple(
                value.shape
            )
        ):

            continue

        transferred[
            key
        ] = value.to(
            device=
                current[key].device,

            dtype=
                current[key].dtype,
        )

    model.load_state_dict(
        transferred,
        strict=False,
    )


def warm_start_matching_shapes(
    new_model,
    old_model,
):
    old_state = (
        old_model.state_dict()
    )

    new_state = (
        new_model.state_dict()
    )

    copied = 0
    skipped = 0

    copied_numel = 0
    total_numel = 0

    result = {}

    for key, dst in new_state.items():

        total_numel += (
            dst.numel()
        )

        src = old_state.get(
            key,
            None,
        )

        if (
            src is not None
            and
            tuple(
                src.shape
            )
            ==
            tuple(
                dst.shape
            )
        ):

            result[
                key
            ] = (
                src.detach()
                .clone()
            )

            copied += 1

            copied_numel += (
                dst.numel()
            )

        else:

            result[
                key
            ] = dst

            skipped += 1

    new_model.load_state_dict(
        result,
        strict=True,
    )

    fraction = (
        copied_numel
        /
        max(
            total_numel,
            1,
        )
    )

    return (
        copied,
        skipped,
        fraction,
    )


# =============================================================================
# Graph helper
# =============================================================================

def remove_edges(
    graph,
    removals,
):
    new_graph = (
        copy_graph(
            graph
        )
    )

    for parent, child in removals:

        if parent in new_graph[
            child
        ]:

            new_graph[
                child
            ].remove(
                parent
            )

    return new_graph


# =============================================================================
# Real NF
# =============================================================================

def make_real_model(
    graph,
    marginal_state,
    device,
):
    bank = (
        MarginalBank(
            DIM
        )
        .to(
            device
        )
    )

    if marginal_state is not None:

        bank.load_state_dict(
            marginal_state
        )

    copula = StructuredCopula(
        DIM,
        graph,
    )

    model = (
        MarginalCopulaModel(
            bank,
            copula,
            DIM,
        )
        .to(
            device
        )
    )

    for marginal in model.marginals:
        marginal.inverse_iterations = TRAINING_INVERSE_ITERATIONS
    for conditional in model.copula.conditionals.values():
        conditional.inverse_iterations = TRAINING_INVERSE_ITERATIONS

    return model


# =============================================================================
# Robust model sampling
# =============================================================================

@torch.no_grad()
def sample_from_model(
    model,
    n,
    cond,
):
    """
    Same robust sampling strategy used in the previous experiment family.
    """

    model.eval()

    errors = []

    attempts = [
        lambda:
            model.sample(
                n,
                cond,
            ),

        lambda:
            model.sample(
                cond,
            ),

        lambda:
            model.sample(
                cond,
                n,
            ),
    ]

    for attempt in attempts:

        try:

            output = attempt()

            if isinstance(
                output,
                tuple,
            ):

                x = output[
                    0
                ]

            else:

                x = output

            if (
                torch.is_tensor(
                    x
                )
                and
                x.ndim == 2
                and
                x.shape[0] == n
                and
                x.shape[1] == DIM
            ):

                return (
                    x.detach()
                )

        except Exception as exc:

            errors.append(
                repr(
                    exc
                )
            )

    raise RuntimeError(
        "Could not determine "
        "model.sample() interface:\n"
        +
        "\n".join(
            errors
        )
    )


@torch.no_grad()
def sample_from_proposal(
    proposal_model,
    n,
    cond,
    device,
    uniform_fraction=PROPOSAL_UNIFORM_FRACTION,
):
    """Draw from and evaluate a defensive uniform/model mixture."""

    if proposal_model is None:
        x = torch.rand(n, DIM, device=device, dtype=cond.dtype)
        return x, torch.zeros(n, device=device, dtype=cond.dtype)

    if not 0.0 < uniform_fraction < 1.0:
        raise ValueError("uniform_fraction must lie strictly between 0 and 1")

    use_uniform = torch.rand(n, device=device) < uniform_fraction
    x = sample_from_model(proposal_model, n, cond)
    if use_uniform.any():
        x[use_uniform] = torch.rand(
            int(use_uniform.sum().item()),
            DIM,
            device=device,
            dtype=x.dtype,
        )

    log_q = proposal_model.log_prob(x, cond).reshape(-1)
    log_r = torch.logaddexp(
        log_q + math.log1p(-uniform_fraction),
        torch.full_like(log_q, math.log(uniform_fraction)),
    )
    return x.detach(), log_r.detach()


def annealing_uniform_fraction(
    beta,
):
    """Decrease defensive exploration as the annealed proposal improves."""

    return max(
        PROPOSAL_MIN_UNIFORM_FRACTION,
        PROPOSAL_UNIFORM_FRACTION * (1.0 - float(beta)),
    )


@torch.no_grad()
def select_next_temperature(
    benchmark,
    proposal_model,
    current_beta,
    device,
    seed,
):
    """Choose the largest safe next beta using one reusable pilot sample."""

    seed_everything(seed)
    n_conditions = TEMPERATURE_PROBE_CONDITIONS
    points = TEMPERATURE_PROBE_POINTS
    anchors = sample_conditions(n_conditions, device)
    cond = anchors[:, None, :].expand(
        n_conditions, points, COND_DIM
    ).reshape(n_conditions * points, COND_DIM)
    uniform_fraction = annealing_uniform_fraction(current_beta)
    x, log_r = sample_from_proposal(
        proposal_model=proposal_model,
        n=n_conditions * points,
        cond=cond,
        device=device,
        uniform_fraction=uniform_fraction,
    )
    log_f = target_log_integrand(benchmark, x, cond).reshape(
        n_conditions, points
    ).double()
    log_r = log_r.reshape(n_conditions, points).double()

    def diagnostics(beta):
        alpha = torch.softmax(beta * log_f - log_r, dim=1)
        raw_ess = 1.0 / alpha.square().sum(dim=1)
        return (
            float(raw_ess.median().item()),
            float(torch.quantile(raw_ess, 0.05).item()),
        )

    upper = min(1.0, current_beta + TEMPERATURE_MAX_INCREMENT)
    median_ess, q05_ess = diagnostics(upper)
    if (
        median_ess >= TEMPERATURE_TARGET_MEDIAN_RAW_ESS
        and q05_ess >= TEMPERATURE_TARGET_Q05_RAW_ESS
    ):
        chosen = upper
    else:
        low = current_beta
        high = upper
        for _ in range(TEMPERATURE_BISECTION_STEPS):
            middle = 0.5 * (low + high)
            middle_median, middle_q05 = diagnostics(middle)
            if (
                middle_median >= TEMPERATURE_TARGET_MEDIAN_RAW_ESS
                and middle_q05 >= TEMPERATURE_TARGET_Q05_RAW_ESS
            ):
                low = middle
            else:
                high = middle
        chosen = max(
            low,
            min(1.0, current_beta + TEMPERATURE_MIN_INCREMENT),
        )
        median_ess, q05_ess = diagnostics(chosen)

    print()
    print(
        f"TEMPERATURE SELECT: {current_beta:.4f} -> {chosen:.4f} | "
        f"pilot median raw ESS={median_ess:.1f} | q05={q05_ess:.1f} | "
        f"uniform={uniform_fraction:.3f}"
    )
    if (
        median_ess < TEMPERATURE_TARGET_MEDIAN_RAW_ESS
        or q05_ess < TEMPERATURE_TARGET_Q05_RAW_ESS
    ):
        if median_ess < 10.0 or q05_ess < 2.0:
            raise RuntimeError(
                "Annealing would be numerically unsafe even at the minimum "
                "temperature step (median raw ESS < 10 or q05 < 2). "
                "Improve the bridge proposal or increase probe points."
            )
        print(
            "WARNING: preferred temperature ESS target was not reached, but "
            "the hard safety floor passed; continuing with the minimum step."
        )
    return chosen, uniform_fraction


# =============================================================================
# Conditional importance resampling
# =============================================================================

# =============================================================================
# Fixed weighted importance-sampling cache
# =============================================================================

@torch.no_grad()
def build_weighted_cache(
    benchmark,
    proposal_model,
    n_conditions,
    points_per_condition,
    device,
    seed,
    label,
    target_temperature=1.0,
    uniform_fraction=PROPOSAL_UNIFORM_FRACTION,
):
    """
    Build a FIXED importance-sampling validation/evidence cache.

    For every condition:

        x_k ~ r(x | c)

        w_k = f(x_k | c) / r(x_k | c)

    and normalize weights WITHIN the condition.

    proposal_model=None:
        r(x|c) = Uniform([0,1]^d)

    No exact target samples are ever used.
    """

    seed_everything(seed)

    c_anchor = sample_conditions(
        n_conditions,
        device,
    )

    cond = (
        c_anchor[:, None, :]
        .expand(
            n_conditions,
            points_per_condition,
            COND_DIM,
        )
        .reshape(
            n_conditions * points_per_condition,
            COND_DIM,
        )
    )

    # -------------------------------------------------------------------------
    # Draw proposal samples
    # -------------------------------------------------------------------------

    x, log_r = sample_from_proposal(
        proposal_model=proposal_model,
        n=n_conditions * points_per_condition,
        cond=cond,
        device=device,
        uniform_fraction=uniform_fraction,
    )

    # -------------------------------------------------------------------------
    # Evaluate known integrand
    # -------------------------------------------------------------------------

    log_f = target_log_integrand(
        benchmark,
        x,
        cond,
    ).reshape(
        n_conditions,
        points_per_condition,
    )

    log_r = log_r.reshape(
        n_conditions,
        points_per_condition,
    )

    logw = (
        target_temperature * log_f
        -
        log_r
    )

    # -------------------------------------------------------------------------
    # Normalize PER CONDITION.
    #
    # Any unknown I(c) cancels.
    # -------------------------------------------------------------------------

    alpha = torch.softmax(
        logw.double(),
        dim=1,
    ).float()

    # -------------------------------------------------------------------------
    # Conditional normalized ESS
    # -------------------------------------------------------------------------

    conditional_ess = (
        1.0
        /
        (
            points_per_condition
            *
            alpha.square().sum(dim=1)
        )
    )

    x = x.reshape(
        n_conditions,
        points_per_condition,
        DIM,
    )

    cond = cond.reshape(
        n_conditions,
        points_per_condition,
        COND_DIM,
    )

    print()
    print(f"WEIGHTED CACHE: {label}")
    print(
        f"proposal             = "
        f"{'uniform' if proposal_model is None else 'NF/uniform mixture'}"
    )
    print(f"target temperature   = {target_temperature:.3f}")
    print(
        f"conditions           = {n_conditions:,}"
    )
    print(
        f"points / condition   = {points_per_condition:,}"
    )
    print(
        f"total points         = "
        f"{n_conditions * points_per_condition:,}"
    )
    print(
        f"mean conditional ESS = "
        f"{conditional_ess.mean().item():.6f}"
    )
    print(
        f"median cond ESS      = "
        f"{conditional_ess.median().item():.6f}"
    )
    print(
        f"min conditional ESS  = "
        f"{conditional_ess.min().item():.6f}"
    )
    raw_ess = points_per_condition * conditional_ess
    print(f"mean raw ESS         = {raw_ess.mean().item():.2f}")
    print(f"median raw ESS       = {raw_ess.median().item():.2f}")
    print(f"min raw ESS          = {raw_ess.min().item():.2f}")
    if raw_ess.median().item() < 20.0:
        print(
            "WARNING: median raw ESS is below 20; "
            "this bridge stage may be too aggressive."
        )

    return {
        "x": x.detach(),
        "cond": cond.detach(),
        "alpha": alpha.detach(),

        "mean_conditional_ess":
            float(
                conditional_ess.mean().item()
            ),

        "mean_raw_ess": float(raw_ess.mean().item()),
        "median_raw_ess": float(raw_ess.median().item()),
        "min_raw_ess": float(raw_ess.min().item()),
    }


@torch.no_grad()
def build_weighted_target_dataset(
    benchmark,
    proposal_model,
    n_conditions,
    points_per_condition,
    resamples_per_condition,
    device,
    seed,
    label,
    target_temperature=1.0,
    uniform_fraction=PROPOSAL_UNIFORM_FRACTION,
):
    """Compatibility helper for code that needs unweighted pseudo-samples.

    Sampling and weighting use only ``f(x|c)``.  Resampling is performed
    independently within each condition, so the unknown normalizer ``I(c)``
    cancels.  KL copula training uses :func:`weighted_kl_loss` directly and
    does not pass through this lossy resampling step.
    """

    cache = build_weighted_cache(
        benchmark=benchmark,
        proposal_model=proposal_model,
        n_conditions=n_conditions,
        points_per_condition=points_per_condition,
        device=device,
        seed=seed,
        label=label,
        target_temperature=target_temperature,
        uniform_fraction=uniform_fraction,
    )
    draws = (
        points_per_condition
        if resamples_per_condition is None
        else resamples_per_condition
    )
    indices = torch.multinomial(
        cache["alpha"],
        num_samples=draws,
        replacement=True,
    )
    gather_x = indices[..., None].expand(-1, -1, DIM)
    gather_c = indices[..., None].expand(-1, -1, COND_DIM)
    x = torch.gather(cache["x"], 1, gather_x).reshape(-1, DIM)
    cond = torch.gather(cache["cond"], 1, gather_c).reshape(-1, COND_DIM)
    return {
        "x": x.detach(),
        "cond": cond.detach(),
        "mean_conditional_ess": cache["mean_conditional_ess"],
    }

# =============================================================================
# Direct importance-weighted KL loss
# =============================================================================

def weighted_kl_loss(
    model,
    benchmark,
    proposal_model,
    device,
    n_conditions,
    points_per_condition,
    uniform_fraction=PROPOSAL_UNIFORM_FRACTION,
    log_prob_fn=None,
):
    """
    Estimate

        -E_c E_{p(x|c)} log q_theta(x|c)

    using self-normalized importance sampling.

    Proposal r is FIXED during one adaptation stage.

    No exact target sampler.
    """

    # -------------------------------------------------------------------------
    # Conditions
    # -------------------------------------------------------------------------

    c_anchor = sample_conditions(
        n_conditions,
        device,
    )

    cond = (
        c_anchor[:, None, :]
        .expand(
            n_conditions,
            points_per_condition,
            COND_DIM,
        )
        .reshape(
            n_conditions * points_per_condition,
            COND_DIM,
        )
    )

    # -------------------------------------------------------------------------
    # Draw from FIXED proposal.
    #
    # We do NOT differentiate through proposal generation.
    # -------------------------------------------------------------------------

    # no_grad keeps the resulting tensors compatible with the subsequent
    # backward pass; inference_mode tensors cannot safely be saved for it.
    with torch.no_grad():

        x, log_r = sample_from_proposal(
            proposal_model=proposal_model,
            n=n_conditions * points_per_condition,
            cond=cond,
            device=device,
            uniform_fraction=uniform_fraction,
        )

        log_f = target_log_integrand(
            benchmark,
            x,
            cond,
        ).reshape(
            n_conditions,
            points_per_condition,
        )

        log_r = log_r.reshape(
            n_conditions,
            points_per_condition,
        )

        logw = (
            log_f
            -
            log_r
        )

        # -------------------------------------------------------------
        # Normalize per condition.
        # -------------------------------------------------------------

        alpha = torch.softmax(
            logw.double(),
            dim=1,
        ).float()

    # -------------------------------------------------------------------------
    # Evaluate TRAINED model with gradients.
    # -------------------------------------------------------------------------

    if log_prob_fn is None:
        log_prob_fn = model.log_prob
    logq = (
        log_prob_fn(
            x.detach(),
            cond,
        )
        .reshape(
            n_conditions,
            points_per_condition,
        )
    )

    loss_per_condition = (
        -(
            alpha
            *
            logq
        )
        .sum(dim=1)
    )

    loss = (
        loss_per_condition.mean()
    )

    # -------------------------------------------------------------------------
    # Diagnostic source ESS
    # -------------------------------------------------------------------------

    with torch.no_grad():

        source_ess = (
            1.0
            /
            (
                points_per_condition
                *
                alpha.square().sum(dim=1)
            )
        )

    return (
        loss,
        source_ess.mean().detach(),
    )


# =============================================================================
# Dataset minibatch
# =============================================================================

def sample_dataset_batch(
    dataset,
    batch_size,
):
    n = (
        dataset[
            "x"
        ]
        .shape[0]
    )

    idx = torch.randint(
        0,
        n,
        (
            batch_size,
        ),
        device=
            dataset[
                "x"
            ].device,
    )

    return (
        dataset[
            "x"
        ][
            idx
        ],

        dataset[
            "cond"
        ][
            idx
        ],
    )


# =============================================================================
# Marginal likelihood
# =============================================================================

def marginal_log_prob(
    bank,
    x,
    cond,
):
    total_logq = torch.zeros(
        x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )

    for j in range(
        DIM
    ):

        _, log_jac = (
            bank
            .marginals[
                j
            ]
            .inverse(
                x[
                    :,
                    j:j + 1
                ],
                cond,
            )
        )

        total_logq += (
            -log_jac
            .reshape(-1)
        )

    return total_logq


@torch.no_grad()
def cached_marginal_nll(
    bank,
    dataset,
    chunk_size=10_000,
):
    bank.eval()

    total = 0.0
    count = 0

    x = dataset["x"]
    cond = dataset["cond"]
    alpha = dataset.get("alpha")

    if alpha is not None:
        n, points_per_condition, _ = x.shape
        condition_chunk = max(1, chunk_size // points_per_condition)
        for start in range(0, n, condition_chunk):
            end = min(start + condition_chunk, n)
            b = end - start
            logq = marginal_log_prob(
                bank,
                x[start:end].reshape(b * points_per_condition, DIM),
                cond[start:end].reshape(
                    b * points_per_condition, COND_DIM
                ),
            ).reshape(b, points_per_condition)
            values = -(alpha[start:end] * logq).sum(dim=1)
            total += values.double().sum().item()
            count += b
        return total / count

    n = x.shape[0]

    for start in range(
        0,
        n,
        chunk_size,
    ):

        end = min(
            start
            +
            chunk_size,
            n,
        )

        logq = marginal_log_prob(
            bank,
            dataset[
                "x"
            ][
                start:end
            ],
            dataset[
                "cond"
            ][
                start:end
            ],
        )

        total += (
            -logq
            .sum()
            .item()
        )

        count += (
            end
            -
            start
        )

    return (
        total
        /
        count
    )


# =============================================================================
# Marginal training
# =============================================================================

def train_marginals_on_dataset(
    bank,
    train_dataset,
    val_dataset,
    device,
    seed,
):
    seed_everything(
        seed
    )

    optimizer = make_adam(
        bank.parameters(),
        lr=MARGINAL_LR,
    )
    training_marginal_log_prob = compile_training_callable(
        lambda x, cond: marginal_log_prob(bank, x, cond)
    )

    initial_nll = cached_marginal_nll(bank, val_dataset)
    best_nll = initial_nll
    best_state = cpu_state_dict(bank)
    best_step = 0
    no_improve = 0

    history = []

    total_steps = 0

    while (
        total_steps
        <
        MAX_MARGINAL_STEPS
    ):

        block = min(
            MARGINAL_CHECK_INTERVAL,
            MAX_MARGINAL_STEPS
            -
            total_steps,
        )

        losses = []

        bank.train()

        for _ in range(
            block
        ):

            if "alpha" in train_dataset:
                n_conditions, points_per_condition, _ = train_dataset["x"].shape
                condition_batch = min(
                    n_conditions,
                    max(8, TRAIN_BATCH_SIZE // points_per_condition),
                )
                idx = torch.randint(
                    0,
                    n_conditions,
                    (condition_batch,),
                    device=device,
                )
                x = train_dataset["x"][idx]
                cond = train_dataset["cond"][idx]
                alpha = train_dataset["alpha"][idx]
                logq = training_marginal_log_prob(
                    x.reshape(condition_batch * points_per_condition, DIM),
                    cond.reshape(
                        condition_batch * points_per_condition, COND_DIM
                    ),
                ).reshape(condition_batch, points_per_condition)
                loss = -(alpha * logq).sum(dim=1).mean()
            else:
                x, cond = sample_dataset_batch(
                    train_dataset, TRAIN_BATCH_SIZE
                )
                loss = -training_marginal_log_prob(x, cond).mean()

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                bank.parameters(),
                10.0,
            )

            optimizer.step()

            # Keep diagnostics on the GPU; converting every step with .item()
            # serializes the CUDA stream and makes optimization stutter.
            losses.append(loss.detach())

        total_steps += (
            block
        )

        val_nll = cached_marginal_nll(
            bank,
            val_dataset,
        )

        train_nll = float(
            torch.stack(losses[-100:]).mean().item()
        )

        rel = (best_nll - val_nll) / max(abs(best_nll), 1.0)
        improved = rel > MARGINAL_MIN_REL_IMPROVEMENT
        if improved:
            best_nll = val_nll
            best_state = cpu_state_dict(bank)
            best_step = total_steps
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"marginal step {total_steps:5d} "
            f"train NLL={train_nll:.6f} val NLL={val_nll:.6f} "
            f"best={best_nll:.6f} patience="
            f"{no_improve}/{MARGINAL_EARLY_STOP_PATIENCE}"
        )

        history.append(
            {
                "step":
                    total_steps,

                "train_nll":
                    train_nll,

                "val_nll":
                    val_nll,

                "relative_improvement":
                    rel,
            }
        )

        if (
            total_steps >= MIN_MARGINAL_STEPS
            and no_improve >= MARGINAL_EARLY_STOP_PATIENCE
        ):
            print(f"  MARGINAL EARLY STOP at {total_steps}")
            break

    load_cpu_state(bank, best_state)
    restored_nll = cached_marginal_nll(bank, val_dataset)

    return {
        "steps":
            total_steps,

        "val_nll": restored_nll,

        "best_step": best_step,

        "history":
            history,
    }


# =============================================================================
# Fixed weighted validation objective
# =============================================================================

@torch.no_grad()
def cached_weighted_nll(
    model,
    cache,
    condition_chunk=64,
):
    """
    Fixed SNIS estimate of

        -E_p log q.

    Uses the SAME weighted samples at every validation check,
    which makes the adaptive LR criterion meaningful.
    """

    model.eval()

    x = cache["x"]
    cond = cache["cond"]
    alpha = cache["alpha"]

    n_conditions = x.shape[0]

    total = 0.0
    count = 0

    for start in range(
        0,
        n_conditions,
        condition_chunk,
    ):

        end = min(
            start + condition_chunk,
            n_conditions,
        )

        xb = x[start:end]

        cb = cond[start:end]

        ab = alpha[start:end]

        b = xb.shape[0]
        k = xb.shape[1]

        logq = (
            model.log_prob(
                xb.reshape(
                    b * k,
                    DIM,
                ),
                cb.reshape(
                    b * k,
                    COND_DIM,
                ),
            )
            .reshape(
                b,
                k,
            )
        )

        value = (
            -(
                ab
                *
                logq
            )
            .sum(dim=1)
        )

        total += (
            value.double().sum().item()
        )

        count += b

    return (
        total / count
    )


def train_bridge_copula_on_cache(
    model,
    train_cache,
    validation_cache,
    label,
    optimize_joint=False,
):
    """Fit a full bridge, optionally updating marginals and copula together."""

    optimized_module = model if optimize_joint else model.copula
    optimizer = make_adam(
        optimized_module.parameters(),
        lr=BRIDGE_COPULA_LR,
    )
    training_log_prob = compile_training_callable(model.log_prob)
    best_nll = cached_weighted_nll(model, validation_cache)
    best_state = cpu_state_dict(optimized_module)
    best_step = 0
    no_improve = 0
    history = []

    n_conditions, points_per_condition, _ = train_cache["x"].shape
    condition_batch = min(
        n_conditions,
        max(8, TRAIN_BATCH_SIZE // points_per_condition),
    )

    for start_step in range(
        0,
        BRIDGE_COPULA_MAX_STEPS,
        BRIDGE_COPULA_CHECK_INTERVAL,
    ):
        block = min(
            BRIDGE_COPULA_CHECK_INTERVAL,
            BRIDGE_COPULA_MAX_STEPS - start_step,
        )
        losses = []
        model.train()

        for _ in range(block):
            idx = torch.randint(
                0,
                n_conditions,
                (condition_batch,),
                device=train_cache["x"].device,
            )
            x = train_cache["x"][idx]
            cond = train_cache["cond"][idx]
            alpha = train_cache["alpha"][idx]
            logq = training_log_prob(
                x.reshape(condition_batch * points_per_condition, DIM),
                cond.reshape(
                    condition_batch * points_per_condition, COND_DIM
                ),
            ).reshape(condition_batch, points_per_condition)
            loss = -(alpha * logq).sum(dim=1).mean()

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimized_module.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.detach())

        step = start_step + block
        val_nll = cached_weighted_nll(model, validation_cache)
        improved = val_nll < best_nll - 1e-3 * max(abs(best_nll), 1.0)
        if improved:
            best_nll = val_nll
            best_state = cpu_state_dict(model.copula)
            best_step = step
            no_improve = 0
        else:
            no_improve += 1
        train_nll = (
            float(torch.stack(losses).mean().item())
            if losses else float("nan")
        )
        history.append({
            "step": step,
            "train_nll": train_nll,
            "val_nll": val_nll,
            "best_nll": best_nll,
        })
        print(
            f"bridge copula step {step:5d} train NLL={train_nll:.6f} "
            f"val NLL={val_nll:.6f} best={best_nll:.6f} "
            f"patience={no_improve}/{BRIDGE_COPULA_PATIENCE}"
        )
        if no_improve >= BRIDGE_COPULA_PATIENCE:
            print(f"  BRIDGE COPULA EARLY STOP at {step}")
            break

    load_cpu_state(optimized_module, best_state)
    return {
        "best_step": best_step,
        "val_nll": cached_weighted_nll(model, validation_cache),
        "history": history,
        "label": label,
        "optimize_joint": optimize_joint,
    }
# =============================================================================
# Adaptive Adam KL / MLE training
# =============================================================================

def train_copula_weighted(
    model,
    benchmark,
    proposal_model,
    validation_cache,
    device,
    n_conditions_per_step,
    points_per_condition,
    label,
):
    """Train the copula directly from fresh self-normalized IS samples.

    The proposal is frozen inside a round, but every gradient step uses fresh
    conditions and samples.  This preserves the original statistical budget.
    """

    if proposal_model is model:
        raise ValueError(
            "proposal_model must be a frozen snapshot, not model itself"
        )
    if n_conditions_per_step <= 0 or points_per_condition <= 0:
        raise ValueError(
            "n_conditions_per_step and points_per_condition must be positive"
        )

    if proposal_model is not None:
        proposal_model.eval()
        for parameter in proposal_model.parameters():
            parameter.requires_grad_(False)

    lr_levels = list(KL_LR_LEVELS)
    lr_index = 0
    lr = lr_levels[lr_index]
    optimizer = make_adam(model.copula.parameters(), lr=lr)
    training_log_prob = compile_training_callable(model.log_prob)

    initial_nll = cached_weighted_nll(model, validation_cache)
    best_nll = initial_nll
    best_step = 0
    best_lr = lr
    best_state = cpu_state_dict(model)
    values = [initial_nll]
    history = []
    total_steps = 0
    plateau_count = 0
    convergence_count = 0
    converged = False

    print()
    print("=" * 90)
    print(f"IMPORTANCE-WEIGHTED KL TRAINING: {label}")
    print(f"step {0:6d} val NLL={initial_nll:.6f} lr={lr:.1e}")

    while total_steps < KL_MAX_STEPS:
        block = min(KL_CHECK_INTERVAL, KL_MAX_STEPS - total_steps)
        losses = []
        source_esses = []
        model.train()

        for _ in range(block):
            loss, source_ess = weighted_kl_loss(
                model=model,
                benchmark=benchmark,
                proposal_model=proposal_model,
                device=device,
                n_conditions=n_conditions_per_step,
                points_per_condition=points_per_condition,
                log_prob_fn=training_log_prob,
            )

            if not torch.isfinite(loss):
                load_cpu_state(model, best_state)
                if lr_index == len(lr_levels) - 1:
                    raise RuntimeError(
                        "Non-finite weighted KL loss at minimum LR."
                    )
                lr_index += 1
                lr = lr_levels[lr_index]
                optimizer = make_adam(
                    model.copula.parameters(), lr=lr
                )
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.copula.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.detach())
            source_esses.append(source_ess)

        total_steps += block
        val_nll = cached_weighted_nll(model, validation_cache)
        values.append(val_nll)

        if val_nll < best_nll:
            best_nll = val_nll
            best_step = total_steps
            best_lr = lr
            best_state = cpu_state_dict(model)

        rel = float("nan")
        window = KL_SMOOTH_WINDOW
        if len(values) >= 2 * window:
            old_mean = float(np.mean(values[-2 * window:-window]))
            new_mean = float(np.mean(values[-window:]))
            rel = (old_mean - new_mean) / max(abs(old_mean), 1e-8)
            plateau_count = (
                plateau_count + 1 if rel < KL_LR_DECAY_TOL else 0
            )

        if (
            plateau_count >= KL_LR_PATIENCE
            and lr_index < len(lr_levels) - 1
        ):
            old_lr = lr
            lr_index += 1
            lr = lr_levels[lr_index]
            for group in optimizer.param_groups:
                group["lr"] = lr
            plateau_count = 0
            print(f"  LR DECAY {old_lr:.1e} -> {lr:.1e}")

        can_converge = (
            total_steps >= KL_MIN_STEPS
            and lr <= KL_MAX_LR_FOR_CONVERGENCE
        )
        if can_converge and np.isfinite(rel):
            convergence_count = (
                convergence_count + 1
                if abs(rel) < KL_CONVERGENCE_TOL
                else 0
            )
        else:
            convergence_count = 0

        mean_loss = (
            float(torch.stack(losses).mean().item())
            if losses else float("nan")
        )
        mean_source_ess = (
            float(torch.stack(source_esses).mean().item())
            if source_esses else float("nan")
        )
        history.append({
            "step": total_steps,
            "train_nll": mean_loss,
            "source_ess": mean_source_ess,
            "val_nll": val_nll,
            "best_nll": best_nll,
            "relative_improvement": rel,
            "lr": lr,
        })
        print(
            f"step {total_steps:6d} NLL={mean_loss:.6f} "
            f"source ESS={mean_source_ess:.4f} val={val_nll:.6f} "
            f"best={best_nll:.6f} rel={100.0 * rel:+.3f}% lr={lr:.1e}"
        )

        if convergence_count >= KL_CONVERGENCE_PATIENCE:
            converged = True
            print(f"  CONVERGED at {total_steps}")
            break

    load_cpu_state(model, best_state)
    restored = cached_weighted_nll(model, validation_cache)
    return {
        "steps": total_steps,
        "best_step": best_step,
        "best_lr": best_lr,
        "val_nll": restored,
        "converged": converged,
        "history": history,
    }


def train_copula_on_dataset(
    model,
    train_dataset,
    val_dataset,
    label,
):
    """
    Same adaptive Adam idea as the successful 1e-2 experiment,
    except convergence is measured using fixed held-out NLL rather
    than oracle KL.
    """

    lr_levels = list(
        KL_LR_LEVELS
    )

    lr_index = 0

    lr = (
        lr_levels[
            lr_index
        ]
    )

    optimizer = make_adam(
        model.copula.parameters(),
        lr=lr,
    )

    initial_nll = cached_weighted_nll(
        model,
        val_dataset,
    )

    best_nll = initial_nll
    best_step = 0
    best_lr = lr

    best_state = cpu_state_dict(
        model
    )

    values = [
        initial_nll
    ]

    total_steps = 0

    plateau_count = 0
    convergence_count = 0

    converged = False

    history = []

    print()
    print(
        "=" * 90
    )

    print(
        f"KL / MLE TRAINING: {label}"
    )

    print(
        f"step {0:6d} "
        f"val NLL="
        f"{initial_nll:.6f} "
        f"lr="
        f"{lr:.1e}"
    )

    while (
        total_steps
        <
        KL_MAX_STEPS
    ):

        block = min(
            KL_CHECK_INTERVAL,
            KL_MAX_STEPS
            -
            total_steps,
        )

        losses = []

        model.train()

        for _ in range(
            block
        ):

            x, cond = (
                sample_dataset_batch(
                    train_dataset,
                    TRAIN_BATCH_SIZE,
                )
            )

            logq = (
                model.log_prob(
                    x,
                    cond,
                )
                .reshape(-1)
            )

            loss = (
                -logq.mean()
            )

            if not torch.isfinite(
                loss
            ):

                load_cpu_state(
                    model,
                    best_state,
                )

                if (
                    lr_index
                    <
                    len(
                        lr_levels
                    )
                    -
                    1
                ):

                    lr_index += 1

                    lr = (
                        lr_levels[
                            lr_index
                        ]
                    )

                    optimizer = (
                        make_adam(
                            model
                            .copula
                            .parameters(),
                            lr=lr,
                        )
                    )

                    continue

                raise RuntimeError(
                    "Non-finite KL/MLE loss "
                    "at minimum LR."
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model
                .copula
                .parameters(),
                10.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        total_steps += (
            block
        )

        val_nll = cached_weighted_nll(
            model,
            val_dataset,
        )

        values.append(
            val_nll
        )

        if (
            val_nll
            <
            best_nll
        ):

            best_nll = (
                val_nll
            )

            best_step = (
                total_steps
            )

            best_lr = (
                lr
            )

            best_state = (
                cpu_state_dict(
                    model
                )
            )

        old_mean = float(
            "nan"
        )

        new_mean = float(
            "nan"
        )

        rel = float(
            "nan"
        )

        w = (
            KL_SMOOTH_WINDOW
        )

        if (
            len(
                values
            )
            >=
            2
            *
            w
        ):

            old_mean = float(
                np.mean(
                    values[
                        -2 * w:
                        -w
                    ]
                )
            )

            new_mean = float(
                np.mean(
                    values[
                        -w:
                    ]
                )
            )

            rel = (
                old_mean
                -
                new_mean
            ) / max(
                abs(
                    old_mean
                ),
                1e-8,
            )

            if (
                rel
                <
                KL_LR_DECAY_TOL
            ):

                plateau_count += 1

            else:

                plateau_count = 0

        if (
            plateau_count
            >=
            KL_LR_PATIENCE
            and
            lr_index
            <
            len(
                lr_levels
            )
            -
            1
        ):

            old_lr = lr

            lr_index += 1

            lr = (
                lr_levels[
                    lr_index
                ]
            )

            for group in (
                optimizer
                .param_groups
            ):

                group[
                    "lr"
                ] = lr

            plateau_count = 0

            print()
            print(
                f"  LR DECAY "
                f"{old_lr:.1e} "
                f"-> "
                f"{lr:.1e}"
            )

        can_converge = (
            total_steps
            >=
            KL_MIN_STEPS
            and
            lr
            <=
            KL_MAX_LR_FOR_CONVERGENCE
        )

        if (
            can_converge
            and
            np.isfinite(
                rel
            )
        ):

            stable = (
                abs(
                    rel
                )
                <
                KL_CONVERGENCE_TOL
            )

            if stable:

                convergence_count += 1

            else:

                convergence_count = 0

        else:

            convergence_count = 0

        mean_loss = float(
            np.mean(
                losses[-100:]
            )
        )

        history.append(
            {
                "step":
                    total_steps,

                "train_nll":
                    mean_loss,

                "val_nll":
                    val_nll,

                "best_nll":
                    best_nll,

                "relative_improvement":
                    rel,

                "lr":
                    lr,
            }
        )

        if np.isfinite(
            rel
        ):

            print(
                f"step "
                f"{total_steps:6d} "
                f"NLL="
                f"{mean_loss:.6f} "
                f"val="
                f"{val_nll:.6f} "
                f"best="
                f"{best_nll:.6f} "
                f"rel="
                f"{100.0 * rel:+.3f}% "
                f"lr="
                f"{lr:.1e}"
            )

        else:

            print(
                f"step "
                f"{total_steps:6d} "
                f"NLL="
                f"{mean_loss:.6f} "
                f"val="
                f"{val_nll:.6f} "
                f"lr="
                f"{lr:.1e}"
            )

        if can_converge:

            print(
                f"  convergence "
                f"{convergence_count}/"
                f"{KL_CONVERGENCE_PATIENCE}"
            )

        if (
            convergence_count
            >=
            KL_CONVERGENCE_PATIENCE
        ):

            converged = True

            print(
                f"  CONVERGED at "
                f"{total_steps}"
            )

            break

    load_cpu_state(
        model,
        best_state,
    )

    restored = cached_weighted_nll(
        model,
        val_dataset,
    )

    print()
    print(
        "OPTIMIZATION SUMMARY"
    )

    print(
        f"total steps = "
        f"{total_steps}"
    )

    print(
        f"best step   = "
        f"{best_step}"
    )

    print(
        f"best LR     = "
        f"{best_lr:.1e}"
    )

    print(
        f"best NLL    = "
        f"{best_nll:.6f}"
    )

    print(
        f"restored    = "
        f"{restored:.6f}"
    )

    return {
        "steps":
            total_steps,

        "best_step":
            best_step,

        "best_lr":
            best_lr,

        "val_nll":
            restored,

        "converged":
            converged,

        "history":
            history,
    }


# =============================================================================
# ESS from actual proposal
#
# No target sampler.
# =============================================================================

@torch.no_grad()
def evaluate_ess(
    model,
    benchmark,
    device,
    n=FINAL_ESS_SIZE,
    seed=EVAL_SEED,
    chunk_size=10_000,
    uniform_fraction=PROPOSAL_UNIFORM_FRACTION,
):
    """Evaluate the actual defensive proposal used for integration.

    Reporting ESS for the bare NF while training and validating an
    NF/uniform mixture gives contradictory results when the NF misses a rare
    region.  The density in the denominator must match the sampler exactly.
    """
    seed_everything(
        seed
    )

    model.eval()

    logw_parts = []

    generated = 0

    while (
        generated
        <
        n
    ):

        m = min(
            chunk_size,
            n
            -
            generated,
        )

        cond = sample_conditions(
            m,
            device,
        )

        x, log_proposal = sample_from_proposal(
            proposal_model=model,
            n=m,
            cond=cond,
            device=device,
            uniform_fraction=uniform_fraction,
        )

        logf = target_log_integrand(
            benchmark,
            x,
            cond,
        ).reshape(-1)

        logw_parts.append(
            (
                logf
                -
                log_proposal
            )
            .detach()
            .double()
        )

        generated += m

    logw = torch.cat(
        logw_parts
    )

    log_sum_w = torch.logsumexp(
        logw,
        dim=0,
    )

    log_sum_w2 = torch.logsumexp(
        2.0
        *
        logw,
        dim=0,
    )

    log_ess = (
        2.0
        *
        log_sum_w
        -
        math.log(
            logw.numel()
        )
        -
        log_sum_w2
    )

    ess = float(
        torch.exp(
            log_ess
        )
        .item()
    )

    # Integral estimate
    log_mean_w = (
        log_sum_w
        -
        math.log(
            logw.numel()
        )
    )

    log_estimate = float(log_mean_w.item())
    # Convert in Python double precision.  Keeping log_estimate as a first-class
    # diagnostic also preserves useful output if the integral is below even the
    # float64 dynamic range.
    estimate = math.exp(log_estimate) if log_estimate > -745.0 else 0.0

    return {
        "ess":
            ess,

        "integral_estimate":
            estimate,

        "log_integral_estimate":
            log_estimate,

        "max_logw":
            float(
                logw
                .max()
                .item()
            ),

        "std_logw":
            # Cut-rejected points carry log f = -inf and therefore log w = -inf.
            # They contribute zero weight, but a plain .std() over them returns
            # NaN; report the dispersion of the finite weights instead.
            float(
                logw[torch.isfinite(logw)]
                .std()
                .item()
            )
            if bool(torch.isfinite(logw).any())
            else float("nan"),

        "uniform_fraction": float(uniform_fraction),
    }


# =============================================================================
# Conditional chi^2 score-function fine tuning
#
# This is intentionally NOT a differentiable target sampler.
#
# x is sampled from q and detached.
#
# For each fixed c:
#
#     J_c(q) = ∫ f(x|c)^2 / q(x|c) dx
#
# and
#
#     grad J_c
#       =
#     - E_q[
#           (f/q)^2
#           grad log q
#       ].
#
# We normalize squared weights WITHIN each c. This removes unknown I(c)^2.
# =============================================================================

def chi2_score_loss(
    model,
    benchmark,
    device,
    n_conditions,
    points_per_condition,
):
    c_anchor = sample_conditions(
        n_conditions,
        device,
    )

    cond = (
        c_anchor[
            :,
            None,
            :
        ]
        .expand(
            n_conditions,
            points_per_condition,
            COND_DIM,
        )
        .reshape(
            n_conditions
            *
            points_per_condition,
            COND_DIM,
        )
    )

    # Sample from q WITHOUT differentiating through the sample.
    with torch.no_grad():

        x = sample_from_model(
            model,
            n_conditions
            *
            points_per_condition,
            cond,
        )

        logf = (
            target_log_integrand(
                benchmark,
                x,
                cond,
            )
            .reshape(
                n_conditions,
                points_per_condition,
            )
        )

    # Re-evaluate log q WITH gradients.
    logq = (
        model.log_prob(
            x.detach(),
            cond,
        )
        .reshape(
            n_conditions,
            points_per_condition,
        )
    )

    logw = (
        logf
        -
        logq
    )

    # Squared importance weights.
    #
    # Normalize within each condition.
    #
    # alpha_jk ∝ (f/q)^2
    #
    # This removes unknown conditional normalization.
    alpha = torch.softmax(
        2.0
        *
        logw.detach(),
        dim=1,
    )

    # Gradient:
    #
    #   -alpha * grad log q
    #
    # moves q toward high squared-weight regions.
    loss = (
        -(
            alpha
            *
            logq
        )
        .sum(
            dim=1
        )
        .mean()
    )

    # Conditional ESS diagnostic using ordinary weights.
    with torch.no_grad():

        p = torch.softmax(
            logw.double(),
            dim=1,
        )

        ess = (
            1.0
            /
            p
            .square()
            .sum(
                dim=1
            )
            /
            points_per_condition
        )

    return (
        loss,
        float(
            ess
            .mean()
            .item()
        ),
    )


def finetune_chi2(
    model,
    benchmark,
    device,
):
    if not CHI2_ENABLED:

        return {
            "steps":
                0,

            "ess":
                evaluate_ess(
                    model,
                    benchmark,
                    device,
                )[
                    "ess"
                ],
        }

    print()
    print(
        "=" * 90
    )

    print(
        "CHI^2 / WEIGHT-VARIANCE FINE TUNING"
    )

    print(
        "=" * 90
    )

    lr_levels = list(
        CHI2_LR_LEVELS
    )

    lr_index = 0

    lr = (
        lr_levels[
            lr_index
        ]
    )

    optimizer = make_adam(
        model
        .copula
        .parameters(),
        lr=lr,
    )

    before = evaluate_ess(
        model,
        benchmark,
        device,
        n=50_000,
        seed=
            EVAL_SEED
            +
            777,
    )

    best_ess = (
        before[
            "ess"
        ]
    )

    best_state = (
        cpu_state_dict(
            model
        )
    )

    best_step = 0

    print(
        f"step      0 "
        f"global ESS="
        f"{best_ess:.6f} "
        f"lr="
        f"{lr:.1e}"
    )

    total_steps = 0

    no_improve = 0

    while (
        total_steps
        <
        CHI2_MAX_STEPS
    ):

        block = min(
            CHI2_CHECK_INTERVAL,
            CHI2_MAX_STEPS
            -
            total_steps,
        )

        losses = []
        local_esses = []

        model.train()

        for _ in range(
            block
        ):

            loss, local_ess = (
                chi2_score_loss(
                    model=
                        model,

                    benchmark=
                        benchmark,

                    device=
                        device,

                    n_conditions=
                        CHI2_CONDITIONS_PER_STEP,

                    points_per_condition=
                        CHI2_POINTS_PER_CONDITION,
                )
            )

            if not torch.isfinite(
                loss
            ):

                continue

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model
                .copula
                .parameters(),
                CHI2_GRAD_CLIP,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

            local_esses.append(
                local_ess
            )

        total_steps += (
            block
        )

        metrics = evaluate_ess(
            model,
            benchmark,
            device,
            n=50_000,
            seed=
                EVAL_SEED
                +
                777,
        )

        ess = (
            metrics[
                "ess"
            ]
        )

        if (
            ess
            >
            best_ess
        ):

            best_ess = ess

            best_step = (
                total_steps
            )

            best_state = (
                cpu_state_dict(
                    model
                )
            )

            no_improve = 0

        else:

            no_improve += 1

        print(
            f"chi2 step "
            f"{total_steps:5d} "
            f"loss="
            f"{np.mean(losses):.6f} "
            f"local ESS="
            f"{np.mean(local_esses):.6f} "
            f"global ESS="
            f"{ess:.6f} "
            f"best="
            f"{best_ess:.6f} "
            f"lr="
            f"{lr:.1e}"
        )

        # ---------------------------------------------------------------------
        # Simple ESS-controlled LR decay.
        #
        # Two validations without a new best -> lower LR.
        # ---------------------------------------------------------------------

        if (
            no_improve
            >=
            2
            and
            lr_index
            <
            len(
                lr_levels
            )
            -
            1
        ):

            old_lr = lr

            lr_index += 1

            lr = (
                lr_levels[
                    lr_index
                ]
            )

            for group in (
                optimizer.param_groups
            ):

                group[
                    "lr"
                ] = lr

            no_improve = 0

            print(
                f"  CHI2 LR DECAY "
                f"{old_lr:.1e}"
                f" -> "
                f"{lr:.1e}"
            )

        # At minimum LR, three checks without improvement is enough.
        if (
            lr_index
            ==
            len(
                lr_levels
            )
            -
            1
            and
            no_improve
            >=
            3
        ):

            break

    load_cpu_state(
        model,
        best_state,
    )

    final = evaluate_ess(
        model,
        benchmark,
        device,
        n=FINAL_ESS_SIZE,
        seed=EVAL_SEED,
    )

    print()
    print(
        "CHI2 SUMMARY"
    )

    print(
        f"best step = "
        f"{best_step}"
    )

    print(
        f"ESS before= "
        f"{before['ess']:.6f}"
    )

    print(
        f"ESS final = "
        f"{final['ess']:.6f}"
    )

    return {
        "steps":
            total_steps,

        "best_step":
            best_step,

        "ess":
            final[
                "ess"
            ],
    }


# =============================================================================
# x -> marginal u
# =============================================================================

@torch.no_grad()
def marginal_x_to_u(
    bank,
    x,
    cond,
):
    marginals = (
        bank.marginals
        if hasattr(bank, "marginals")
        else bank
    )
    pieces = []

    for j in range(
        DIM
    ):

        uj, _ = (
            marginals[
                j
            ]
            .inverse(
                x[
                    :,
                    j:j + 1
                ],
                cond,
            )
        )

        pieces.append(
            uj.clamp(
                0.0,
                1.0,
            )
        )

    return torch.cat(
        pieces,
        dim=1,
    )


# =============================================================================
# Masked child
# =============================================================================

def train_masked_child_cached(
    child,
    active_parents,
    u_cache,
    cond_cache,
    previous_state,
    steps,
    seed,
    device,
):
    seed_everything(
        seed
    )

    model = (
        IterativeMaskedChild(
            child
        )
        .to(
            device
        )
    )

    if previous_state is not None:

        model.load_state_dict(
            previous_state
        )

    optimizer = make_adam(
        model.parameters(),
        lr=LR_MASKED,
    )

    n_cache = (
        u_cache.shape[0]
    )

    losses = []

    for _ in range(
        steps
    ):

        idx = torch.randint(
            0,
            n_cache,
            (
                TRAIN_BATCH_SIZE,
            ),
            device=device,
        )

        u = u_cache[
            idx
        ]

        cond = cond_cache[
            idx
        ]

        mask = sample_training_mask(
            batch_size=
                TRAIN_BATCH_SIZE,

            child=
                child,

            active_parents=
                active_parents,

            keep_prob=
                MASKED_KEEP_PROB,

            device=
                device,
        )

        logq = model.log_prob(
            u[
                :,
                child:child + 1
            ],
            u,
            cond,
            mask,
        )

        loss = (
            -logq.mean()
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        losses.append(
            float(
                loss.item()
            )
        )

    return (
        model,
        float(
            np.mean(
                losses[-100:]
            )
        ),
    )


# =============================================================================
# Masked delta
# =============================================================================

@torch.no_grad()
def compute_deltas_cached(
    model,
    child,
    active_parents,
    u,
    cond,
    device,
    chunk_size=10_000,
):
    if not active_parents:

        return []

    base_vector = (
        active_mask_vector(
            child,
            active_parents,
            device,
        )
    )

    sums = {
        p: 0.0
        for p in active_parents
    }

    sums2 = {
        p: 0.0
        for p in active_parents
    }

    count = 0

    for start in range(
        0,
        u.shape[0],
        chunk_size,
    ):

        end = min(
            start
            +
            chunk_size,
            u.shape[0],
        )

        ub = u[
            start:end
        ]

        cb = cond[
            start:end
        ]

        n = (
            ub.shape[0]
        )

        base_mask = (
            base_vector[
                None,
                :
            ]
            .expand(
                n,
                -1,
            )
            .clone()
        )

        child_u = ub[
            :,
            child:child + 1
        ]

        log_on = model.log_prob(
            child_u,
            ub,
            cb,
            base_mask,
        )

        for parent in (
            active_parents
        ):

            off = (
                base_mask
                .clone()
            )

            off[
                :,
                parent
            ] = 0.0

            log_off = model.log_prob(
                child_u,
                ub,
                cb,
                off,
            )

            delta = (
                log_on
                -
                log_off
            )

            sums[
                parent
            ] += (
                delta
                .sum()
                .item()
            )

            sums2[
                parent
            ] += (
                delta
                .square()
                .sum()
                .item()
            )

        count += n

    rows = []

    for parent in (
        active_parents
    ):

        mean = (
            sums[
                parent
            ]
            /
            count
        )

        second = (
            sums2[
                parent
            ]
            /
            count
        )

        variance = max(
            second
            -
            mean
            *
            mean,
            0.0,
        )

        rows.append(
            {
                "parent":
                    parent,

                "child":
                    child,

                "delta_loglik":
                    mean,

                "delta_std":
                    math.sqrt(
                        variance
                    ),
            }
        )

    return rows


# =============================================================================
# Build evidence from the CURRENT proposal
#
# Again: no target sampler.
# =============================================================================

def collect_evidence(
    benchmark,
    round_id,
    graph,
    current_model,
    model_cache,
    evidence_cache,
    args,
    device,
):
    rows = []

    # -------------------------------------------------------------------------
    # For every evidence seed, independently importance-resample pseudo-target
    # samples from the CURRENT accepted proposal.
    # -------------------------------------------------------------------------

    for seed in (
        args.seeds
    ):

        print()
        print(
            f"Evidence seed "
            f"{seed}"
        )

        target_data = (
            build_weighted_target_dataset(
                benchmark=
                    benchmark,

                proposal_model=
                    current_model,

                n_conditions=
                    args.evidence_conditions,

                points_per_condition=
                    args.is_points,

                resamples_per_condition=
                    args.evidence_resamples,

                device=
                    device,

                seed=
                    (
                        500_000
                        +
                        seed
                        +
                        1000
                        *
                        round_id
                    ),

                label=
                    f"evidence "
                    f"round={round_id} "
                    f"seed={seed}",
            )
        )

        bank = get_marginal_bank(
            current_model
        )

        with torch.no_grad():

            u = marginal_x_to_u(
                bank,
                target_data[
                    "x"
                ],
                target_data[
                    "cond"
                ],
            )

        # Use first ~100k for evaluation.
        n_eval = min(
            DELTA_EVAL_SIZE,
            u.shape[0],
        )

        u_eval = u[
            :n_eval
        ]

        c_eval = target_data[
            "cond"
        ][
            :n_eval
        ]

        for child in range(
            1,
            DIM,
        ):

            parents = list(
                graph[
                    child
                ]
            )

            if not parents:

                continue

            key = (
                seed,
                child,
                tuple(
                    parents
                ),
            )

            previous_state = (
                model_cache.get(
                    key,
                    None,
                )
            )

            steps = (
                MASKED_STEPS
                if
                previous_state
                is None
                else
                REFINE_STEPS
            )

            mode = (
                "FULL"
                if
                previous_state
                is None
                else
                "REFINE"
            )

            print(
                f"child {child}: "
                f"{parents} "
                f"[{mode} {steps}]"
            )

            masked_model, loss = (
                train_masked_child_cached(
                    child=
                        child,

                    active_parents=
                        parents,

                    u_cache=
                        u,

                    cond_cache=
                        target_data[
                            "cond"
                        ],

                    previous_state=
                        previous_state,

                    steps=
                        steps,

                    seed=
                        (
                            seed
                            *
                            100000
                            +
                            round_id
                            *
                            1000
                            +
                            child
                        ),

                    device=
                        device,
                )
            )

            print(
                f"NLL ~ "
                f"{loss:.6f}"
            )

            child_rows = (
                compute_deltas_cached(
                    model=
                        masked_model,

                    child=
                        child,

                    active_parents=
                        parents,

                    u=
                        u_eval,

                    cond=
                        c_eval,

                    device=
                        device,
                )
            )

            state = cpu_state_dict(
                masked_model
            )

            model_cache[
                key
            ] = state

            evidence_cache[
                key
            ] = copy.deepcopy(
                child_rows
            )

            for row in (
                child_rows
            ):

                row = dict(
                    row
                )

                row.update(
                    {
                        "benchmark":
                            benchmark,

                        "round":
                            round_id,

                        "seed":
                            seed,

                        "source":
                            mode.lower(),
                    }
                )

                rows.append(
                    row
                )

            del masked_model

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Find the marginal bank inside MarginalCopulaModel
# =============================================================================

def get_marginal_bank(
    model,
):
    possible = [
        "bank",
        "marginal_bank",
        "marginals",
    ]

    for name in (
        possible
    ):

        if hasattr(
            model,
            name,
        ):

            obj = getattr(
                model,
                name,
            )

            if isinstance(
                obj,
                MarginalBank,
            ):

                return obj

            # Some MarginalCopulaModel implementations retain only the
            # MarginalBank's ModuleList instead of the bank container itself.
            if name == "marginals" and len(obj) == DIM:
                if all(hasattr(marginal, "inverse") for marginal in obj):
                    return obj

    # Search modules as fallback.
    for module in (
        model.modules()
    ):

        if isinstance(
            module,
            MarginalBank,
        ):

            return module

    raise RuntimeError(
        "Could not locate MarginalBank "
        "inside MarginalCopulaModel."
    )


# =============================================================================
# Train a graph using adaptive importance sampling
# =============================================================================

def train_graph_adaptively(
    benchmark,
    graph,
    marginal_state,
    current_model,
    args,
    device,
    label,
    update_marginals=False,
):
    """
    Train graph using only f(x|c).

    If current_model is None:
        first sampling proposal = uniform.

    Otherwise:
        first sampling proposal = current accepted NF.

    Candidate graphs all receive exactly the SAME current_model proposal.
    """

    seed_everything(
        PERFORMANCE_SEED
    )

    model = make_real_model(
        graph,
        marginal_state,
        device,
    )

    warm_fraction = 0.0

    if current_model is not None:

        copied, skipped, warm_fraction = (
            warm_start_matching_shapes(
                model,
                current_model,
            )
        )

        print()
        print(
            "WARM START"
        )

        print(
            f"copied tensors = "
            f"{copied}"
        )

        print(
            f"skipped tensors = "
            f"{skipped}"
        )

        print(
            f"copied parameter fraction = "
            f"{100.0 * warm_fraction:.2f}%"
        )

    proposal = (
        current_model
    )

    latest_train = None
    latest_val = None

    all_training = []

    for adaptive_round in range(
        N_ADAPTIVE_KL_ROUNDS
    ):

        if (
            proposal is None
            and
            adaptive_round == 0
        ):

            proposal_name = (
                "uniform"
            )

        else:

            proposal_name = (
                "NF"
            )

        print()
        print(
            "#" * 90
        )

        print(
            f"ADAPTIVE KL ROUND "
            f"{adaptive_round}"
        )

        print(
            f"graph      = "
            f"{label}"
        )

        print(
            f"proposal   = "
            f"{proposal_name}"
        )

        print(
            "#" * 90
        )

        points_this_round = (
            UNIFORM_BOOTSTRAP_POINTS
            if proposal is None
            else args.is_points
        )
        conditions_this_round = (
            BOOTSTRAP_CONDITIONS_PER_STEP
            if proposal is None
            else ADAPTIVE_CONDITIONS_PER_STEP
        )
        validation_conditions = (
            BOOTSTRAP_VAL_CONDITIONS
            if proposal is None
            else ADAPTIVE_VAL_CONDITIONS
        )

        val_data = build_weighted_cache(
            benchmark=benchmark,
            proposal_model=proposal,
            n_conditions=validation_conditions,
            points_per_condition=points_this_round,
            device=device,
            seed=20000 + adaptive_round * 100 + PERFORMANCE_SEED,
            label=f"{label} validation adapt={adaptive_round}",
        )

        # Marginals are updated only for the initial full-graph warmup.
        if (
            update_marginals
            and
            adaptive_round
            >
            0
        ):

            bank = get_marginal_bank(
                model
            )

            marginal_train = build_weighted_target_dataset(
                benchmark=benchmark,
                proposal_model=proposal,
                n_conditions=TRAIN_CONDITIONS,
                points_per_condition=points_this_round,
                resamples_per_condition=RESAMPLES_PER_CONDITION,
                device=device,
                seed=10000 + adaptive_round * 100 + PERFORMANCE_SEED,
                label=f"{label} marginal adapt={adaptive_round}",
            )
            marginal_val = build_weighted_target_dataset(
                benchmark=benchmark,
                proposal_model=proposal,
                n_conditions=VAL_CONDITIONS,
                points_per_condition=points_this_round,
                resamples_per_condition=RESAMPLES_PER_CONDITION,
                device=device,
                seed=20000 + adaptive_round * 100 + PERFORMANCE_SEED,
                label=f"{label} marginal validation adapt={adaptive_round}",
            )
            train_marginals_on_dataset(
                bank=
                    bank,

                train_dataset=
                    marginal_train,

                val_dataset=
                    marginal_val,

                device=
                    device,

                seed=
                    PERFORMANCE_SEED
                    +
                    adaptive_round,
            )

        training = train_copula_weighted(
            model=
                model,

            benchmark=
                benchmark,

            proposal_model=
                proposal,

            validation_cache=
                val_data,

            device=
                device,

            n_conditions_per_step=
                conditions_this_round,

            points_per_condition=
                points_this_round,

            label=
                f"{label} "
                f"adapt={adaptive_round}",
        )

        all_training.append(
            training
        )

        latest_train = None

        latest_val = (
            val_data
        )

        # New proposal for the next adaptation round is THIS trained NF.
        proposal = copy.deepcopy(model).to(device)
        proposal.eval()
        for parameter in proposal.parameters():
            parameter.requires_grad_(False)

        intermediate = evaluate_ess(
            model,
            benchmark,
            device,
            n=50_000,
            seed=
                EVAL_SEED
                +
                adaptive_round,
        )

        print()
        print(
            f"ADAPTIVE ROUND "
            f"{adaptive_round} ESS="
            f"{intermediate['ess']:.6f}"
        )

    # -------------------------------------------------------------------------
    # Final chi2 / ESS fine tuning
    # -------------------------------------------------------------------------

    chi2_result = finetune_chi2(
        model=
            model,

        benchmark=
            benchmark,

        device=
            device,
    )

    final_metrics = evaluate_ess(
        model,
        benchmark,
        device,
        n=FINAL_ESS_SIZE,
        seed=EVAL_SEED,
    )

    return (
        model,
        final_metrics,
        {
            "warm_fraction":
                warm_fraction,

            "training":
                all_training,

            "chi2":
                chi2_result,

            "train_dataset":
                latest_train,

            "val_dataset":
                latest_val,
        },
    )


# =============================================================================
# Initial marginal warmup
# =============================================================================

def train_initial_marginals(
    benchmark,
    args,
    device,
):
    """Fit marginals through an annealed, integrand-only bridge."""

    bank = (
        MarginalBank(
            DIM
        )
        .to(
            device
        )
    )
    for marginal in bank.marginals:
        marginal.inverse_iterations = TRAINING_INVERSE_ITERATIONS

    proposal = None
    stage_results = []

    current_beta = 0.0
    stage = 0

    while current_beta < 1.0 - 1e-12:
        if args.temperatures is not None or args.speed_profile in ("smoke", "thorough"):
            if args.temperatures is not None:
                schedule = tuple(args.temperatures)
                schedule_label = "CLI temperature override"
            elif args.speed_profile == "smoke":
                # The coarse five-point schedule is fine in 2-D/5-D/8-D but
                # collapses the weighted cache in 32-D: at beta=0.08 the
                # target's log-density spread is already ~O(10) per point, so
                # the effective sample size falls to ~3 and the marginal fit
                # early-stops on a degenerate cache. Use the same fine grid
                # the thorough profile uses for high dimensions.
                if DIM >= 32:
                    schedule = HIGH_DIM_MARGINAL_TEMPERATURES
                    schedule_label = "high-dimensional smoke schedule"
                else:
                    schedule = (0.02, 0.08, 0.25, 0.55, 1.00)
                    schedule_label = "short smoke-test schedule"
            else:
                schedule = (
                    HIGH_DIM_MARGINAL_TEMPERATURES
                    if DIM >= 32 else MARGINAL_TEMPERATURES
                )
                schedule_label = "original fixed high-quality schedule"
            temperature = next(
                beta for beta in schedule if beta > current_beta + 1e-12
            )
            uniform_fraction = annealing_uniform_fraction(current_beta)
            print()
            print(
                f"TEMPERATURE SCHEDULE: {current_beta:.4f} -> "
                f"{temperature:.4f} | {schedule_label} | "
                f"uniform={uniform_fraction:.3f}"
            )
        else:
            temperature, uniform_fraction = select_next_temperature(
                benchmark=benchmark,
                proposal_model=proposal,
                current_beta=current_beta,
                device=device,
                seed=70000 + 100 * stage,
            )
        points = (
            UNIFORM_BOOTSTRAP_POINTS if proposal is None
            else MARGINAL_ADAPTIVE_POINTS
        )
        train_data = build_weighted_cache(
            benchmark=benchmark,
            proposal_model=proposal,
            n_conditions=MARGINAL_TRAIN_CONDITIONS,
            points_per_condition=points,
            device=device,
            seed=10101 + 100 * stage,
            label=f"marginal train beta={temperature:.4f}",
            target_temperature=temperature,
            uniform_fraction=uniform_fraction,
        )
        val_data = build_weighted_cache(
            benchmark=benchmark,
            proposal_model=proposal,
            n_conditions=VAL_CONDITIONS,
            points_per_condition=points,
            device=device,
            seed=20202 + 100 * stage,
            label=f"marginal val beta={temperature:.4f}",
            target_temperature=temperature,
            uniform_fraction=uniform_fraction,
        )
        if not args.joint_marginal_copula:
            result = train_marginals_on_dataset(
                bank=bank,
                train_dataset=train_data,
                val_dataset=val_data,
                device=device,
                seed=PERFORMANCE_SEED + stage,
            )
            result["temperature"] = temperature

        bridge_model = make_real_model(
            full_ar_graph(DIM),
            cpu_state_dict(bank),
            device,
        )

        # Preserve dependence learned at the preceding temperature while
        # retaining the newly fitted marginal bank.
        if proposal is not None:
            try:
                bridge_model.copula.load_state_dict(
                    proposal.copula.state_dict(),
                    strict=True,
                )
            except (AttributeError, RuntimeError):
                pass

        bridge_result = train_bridge_copula_on_cache(
            model=bridge_model,
            train_cache=train_data,
            validation_cache=val_data,
            label=(
                f"joint marginal+copula beta={temperature:.4f}"
                if args.joint_marginal_copula
                else f"marginal bridge beta={temperature:.4f}"
            ),
            optimize_joint=args.joint_marginal_copula,
        )

        if args.joint_marginal_copula:
            # Carry the jointly updated marginal transport into the next
            # annealing stage and into the later sparse-graph models.
            load_cpu_state(
                bank,
                cpu_state_dict(get_marginal_bank(bridge_model)),
            )
            last_step = (
                bridge_result["history"][-1]["step"]
                if bridge_result["history"] else 0
            )
            result = {
                "temperature": temperature,
                "steps": last_step,
                "best_step": bridge_result["best_step"],
                "val_nll": cached_marginal_nll(bank, val_data),
                "history": bridge_result["history"],
                "joint_fit": True,
            }
        stage_results.append({
            "temperature": temperature,
            "marginal": result,
            "bridge_copula": bridge_result,
        })

        proposal = copy.deepcopy(bridge_model).to(device)
        proposal.eval()
        for parameter in proposal.parameters():
            parameter.requires_grad_(False)

        current_beta = temperature
        stage += 1

    return (
        cpu_state_dict(
            bank
        ),
        {
            "stages": stage_results,
            "val_nll": stage_results[-1]["marginal"]["val_nll"],
        },
        proposal,
    )


# =============================================================================
# Candidate decision tree
# =============================================================================

def split_weak_edges(
    ranked_edges,
):
    n = len(
        ranked_edges
    )

    midpoint = int(
        math.ceil(
            n
            /
            2
        )
    )

    first = list(
        ranked_edges[
            :midpoint
        ]
    )

    second = list(
        ranked_edges[
            midpoint:
        ]
    )

    return (
        first,
        second,
    )


def train_candidate(
    benchmark,
    name,
    removals,
    current_graph,
    current_model,
    current_metrics,
    marginal_state,
    args,
    device,
    round_id,
):
    candidate_graph = remove_edges(
        current_graph,
        removals,
    )

    print()
    print(
        "=" * 100
    )

    print(
        f"TRAIN CANDIDATE: "
        f"{name}"
    )

    print(
        f"round      = "
        f"{round_id}"
    )

    print(
        f"remove     = "
        f"{removals}"
    )

    print(
        f"edges      = "
        f"{count_edges(candidate_graph)}"
    )

    print(
        "sampling   = adaptive "
        "importance sampling"
    )

    print(
        "target sampler = NEVER USED"
    )

    print(
        "=" * 100
    )

    model, metrics, train_info = (
        train_graph_adaptively(
            benchmark=
                benchmark,

            graph=
                candidate_graph,

            marginal_state=
                marginal_state,

            current_model=
                current_model,

            args=
                args,

            device=
                device,

            label=
                f"{name} "
                f"round={round_id}",

            update_marginals=
                False,
        )
    )

    delta_ess = (
        metrics[
            "ess"
        ]
        -
        current_metrics[
            "ess"
        ]
    )

    ess_pass = (
        delta_ess
        >=
        ESS_TOL
    )

    print()
    print(
        f"RESULT: {name}"
    )

    print(
        f"current ESS   = "
        f"{current_metrics['ess']:.6f}"
    )

    print(
        f"candidate ESS = "
        f"{metrics['ess']:.6f}"
    )

    print(
        f"delta ESS     = "
        f"{delta_ess:+.6f}"
    )

    print(
        f"ESS pass      = "
        f"{ess_pass}"
    )

    return {
        "name":
            name,

        "graph":
            candidate_graph,

        "model":
            model,

        "metrics":
            metrics,

        "delta_ess":
            delta_ess,

        "pass":
            ess_pass,

        "train_info":
            train_info,

        "removals":
            removals,
    }


def search_candidates(
    benchmark,
    weak_edges,
    current_graph,
    current_model,
    current_metrics,
    marginal_state,
    args,
    device,
    round_id,
):
    """
    Decision tree:

    1. all weak
    2. first half

    Then:
      - if all_weak failed -> test second half
      - if all_weak passed AND first_half > all_weak -> test second half
      - otherwise stop

    Final winner = highest ESS among passing tested candidates.
    """

    if not weak_edges:

        return None

    first_half, second_half = (
        split_weak_edges(
            weak_edges
        )
    )

    tested = []

    # -------------------------------------------------------------------------
    # 1. All weak
    # -------------------------------------------------------------------------

    all_candidate = train_candidate(
        benchmark=
            benchmark,

        name=
            "all_weak",

        removals=
            weak_edges,

        current_graph=
            current_graph,

        current_model=
            current_model,

        current_metrics=
            current_metrics,

        marginal_state=
            marginal_state,

        args=
            args,

        device=
            device,

        round_id=
            round_id,
    )

    tested.append(
        all_candidate
    )

    # -------------------------------------------------------------------------
    # 2. First half
    # -------------------------------------------------------------------------

    first_candidate = None

    if first_half:

        first_candidate = train_candidate(
            benchmark=
                benchmark,

            name=
                "rank_first_half",

            removals=
                first_half,

            current_graph=
                current_graph,

            current_model=
                current_model,

            current_metrics=
                current_metrics,

            marginal_state=
                marginal_state,

            args=
                args,

            device=
                device,

            round_id=
                round_id,
        )

        tested.append(
            first_candidate
        )

    # -------------------------------------------------------------------------
    # Decide whether second half is informative.
    # -------------------------------------------------------------------------

    test_second = False

    if second_half:

        if not all_candidate[
            "pass"
        ]:

            test_second = True

        elif (
            first_candidate
            is not None
            and
            first_candidate[
                "metrics"
            ][
                "ess"
            ]
            >
            all_candidate[
                "metrics"
            ][
                "ess"
            ]
        ):

            test_second = True

    # -------------------------------------------------------------------------
    # 3. Second half only when needed
    # -------------------------------------------------------------------------

    if test_second:

        second_candidate = (
            train_candidate(
                benchmark=
                    benchmark,

                name=
                    "rank_second_half",

                removals=
                    second_half,

                current_graph=
                    current_graph,

                current_model=
                    current_model,

                current_metrics=
                    current_metrics,

                marginal_state=
                    marginal_state,

                args=
                    args,

                device=
                    device,

                round_id=
                    round_id,
            )
        )

        tested.append(
            second_candidate
        )

    else:

        print()
        print(
            "Second-half candidate skipped "
            "by decision tree."
        )

    passing = [
        c
        for c in tested
        if c[
            "pass"
        ]
    ]

    if not passing:

        for candidate in (
            tested
        ):

            del candidate[
                "model"
            ]

        return None

    passing = sorted(
        passing,
        key=lambda c:
            (
                -c[
                    "metrics"
                ][
                    "ess"
                ],

                count_edges(
                    c[
                        "graph"
                    ]
                ),
            ),
    )

    best = (
        passing[
            0
        ]
    )

    for candidate in (
        tested
    ):

        if candidate is best:

            continue

        del candidate[
            "model"
        ]

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return best


# =============================================================================
# Main benchmark
# =============================================================================

def run_benchmark(
    benchmark,
    args,
    device,
    output_dir,
):
    print()
    print(
        "#" * 120
    )

    print(
        f"ADAPTIVE INTEGRAND BENCHMARK: "
        f"{benchmark}"
    )

    print(
        "Exact target sampler = HIDDEN"
    )

    print(
        "Training access      = f(x|c) only"
    )

    print(
        "#" * 120
    )

    # -------------------------------------------------------------------------
    # Initial marginal warmup from weighted uniform samples.
    # -------------------------------------------------------------------------

    marginal_state, marginal_result, annealed_proposal = (
        train_initial_marginals(
            benchmark,
            args,
            device,
        )
    )

    annealed_metrics = evaluate_ess(
        annealed_proposal,
        benchmark,
        device,
        n=FINAL_ESS_SIZE,
        seed=EVAL_SEED + 500,
    )
    print()
    print("ANNEALED BRIDGE PERFORMANCE (BEFORE FORMAL KL / CHI^2)")
    print(f"ESS      = {annealed_metrics['ess']:.6f}")
    print(f"estimate = {annealed_metrics['integral_estimate']:.12e}")
    print(f"log estimate = {annealed_metrics['log_integral_estimate']:.12g}")

    # -------------------------------------------------------------------------
    # Full graph
    # -------------------------------------------------------------------------

    graph = full_ar_graph(
        DIM
    )

    print()
    print(
        "=" * 100
    )

    print(
        "TRAIN INITIAL FULL GRAPH"
    )

    print(
        f"edges = "
        f"{count_edges(graph)}"
    )

    print(
        "=" * 100
    )

    (
        current_model,
        current_metrics,
        initial_info,
    ) = train_graph_adaptively(
        benchmark=
            benchmark,

        graph=
            graph,

        marginal_state=
            marginal_state,

        current_model=
            annealed_proposal,

        args=
            args,

        device=
            device,

        label=
            "initial_full",

        update_marginals=
            False,
    )

    print()
    print(
        "INITIAL PERFORMANCE"
    )

    print(
        f"ESS      = "
        f"{current_metrics['ess']:.6f}"
    )

    print(f"estimate = {current_metrics['integral_estimate']:.12e}")
    print(f"log estimate = {current_metrics['log_integral_estimate']:.12g}")

    # -------------------------------------------------------------------------
    # Search state
    # -------------------------------------------------------------------------

    masked_model_cache = {}
    evidence_cache = {}

    frozen_sigma0 = None

    history_rows = []

    # -------------------------------------------------------------------------
    # Rounds
    # -------------------------------------------------------------------------

    for round_id in range(
        args.max_rounds
    ):

        print()
        print(
            "#" * 120
        )

        print(
            f"ROUND {round_id}"
        )

        print(
            f"edges = "
            f"{count_edges(graph)}"
        )

        print(
            f"ESS   = "
            f"{current_metrics['ess']:.6f}"
        )

        print(
            graph
        )

        print(
            "#" * 120
        )

        evidence_df = collect_evidence(
            benchmark=
                benchmark,

            round_id=
                round_id,

            graph=
                graph,

            current_model=
                current_model,

            model_cache=
                masked_model_cache,

            evidence_cache=
                evidence_cache,

            args=
                args,

            device=
                device,
        )

        if len(
            evidence_df
        ) == 0:

            print(
                "No active edges."
            )

            break

        posterior_df, posterior = (
            bayesian_round(
                evidence_df,
                args,
                fixed_sigma0=
                    frozen_sigma0,
            )
        )

        if round_id == 0:

            frozen_sigma0 = float(
                posterior[
                    "sigma0_mean"
                ]
            )

            print()
            print(
                f"Frozen sigma0 = "
                f"{frozen_sigma0:.8f}"
            )

        weak = []
        evidence_delta = (
            evidence_df.groupby(["parent", "child"])["delta_loglik"]
            .mean()
            .to_dict()
        )

        required_columns = {"parent", "child", "pip"}
        missing_columns = required_columns - set(posterior_df.columns)
        if missing_columns:
            raise RuntimeError(
                "Bayesian posterior is missing columns: "
                f"{sorted(missing_columns)}; got {list(posterior_df.columns)}"
            )

        for row in posterior_df.to_dict(orient="records"):
            parent = int(row["parent"])
            child = int(row["child"])
            pip = float(row["pip"])
            if pip < args.pip_threshold:
                weak.append(
                    (
                        parent,
                        child,
                        pip,
                        float(evidence_delta[(parent, child)]),
                    )
                )

        # Lowest PIP / weakest delta first.
        weak = sorted(
            weak,
            key=lambda z:
                (
                    z[
                        2
                    ],
                    z[
                        3
                    ],
                ),
        )

        print()
        print(
            "WEAK EDGE RANKING"
        )

        for i, (
            parent,
            child,
            pip,
            delta,
        ) in enumerate(
            weak,
            start=1,
        ):

            print(
                f"{i}. "
                f"{parent}->{child} "
                f"PIP="
                f"{pip:.6f} "
                f"delta="
                f"{delta:.6f}"
            )

        weak_edges = [
            (
                parent,
                child,
            )

            for (
                parent,
                child,
                _,
                _,
            )
            in weak
        ]

        if not weak_edges:

            print(
                "No weak edges."
            )

            break

        best = search_candidates(
            benchmark=
                benchmark,

            weak_edges=
                weak_edges,

            current_graph=
                graph,

            current_model=
                current_model,

            current_metrics=
                current_metrics,

            marginal_state=
                marginal_state,

            args=
                args,

            device=
                device,

            round_id=
                round_id,
        )

        if best is None:

            print()
            print(
                "NO CANDIDATE IMPROVED ESS"
            )

            break

        old_model = (
            current_model
        )

        graph = (
            best[
                "graph"
            ]
        )

        current_model = (
            best[
                "model"
            ]
        )

        current_metrics = (
            best[
                "metrics"
            ]
        )

        history_rows.append(
            {
                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "candidate":
                    best[
                        "name"
                    ],

                "edges":
                    count_edges(
                        graph
                    ),

                "ess":
                    current_metrics[
                        "ess"
                    ],

                "integral_estimate":
                    current_metrics[
                        "integral_estimate"
                    ],

                "removed_edges":
                    str(
                        best[
                            "removals"
                        ]
                    ),
            }
        )

        del old_model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------

    final_metrics = evaluate_ess(
        current_model,
        benchmark,
        device,
        n=FINAL_ESS_SIZE,
        seed=
            EVAL_SEED
            +
            999,
    )

    # Oracle information is introduced only after all training, pruning, and
    # candidate selection decisions are complete.
    true_integral = ORACLE_INTEGRALS.get(benchmark)
    final_metrics["true_integral"] = true_integral
    if true_integral is None:
        final_metrics["integral_error"] = None
        final_metrics["absolute_integral_error"] = None
        final_metrics["relative_integral_error"] = None
    else:
        integral_error = final_metrics["integral_estimate"] - true_integral
        final_metrics["integral_error"] = integral_error
        final_metrics["absolute_integral_error"] = abs(integral_error)
        final_metrics["relative_integral_error"] = (
            abs(integral_error) / abs(true_integral)
        )

    print()
    print(
        "=" * 120
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 120
    )

    print(
        f"graph = "
        f"{graph}"
    )

    print(
        f"edges = "
        f"{count_edges(graph)}"
    )

    print(
        f"ESS   = "
        f"{final_metrics['ess']:.6f}"
    )

    print(f"Ihat  = {final_metrics['integral_estimate']:.12e}")
    print(f"log(Ihat) = {final_metrics['log_integral_estimate']:.12g}")
    if final_metrics["true_integral"] is None:
        print("Itrue = unavailable (no oracle registered)")
        print("|error| = unavailable")
    else:
        print(f"Itrue = {final_metrics['true_integral']:.12e}")
        print(f"|error| = {final_metrics['absolute_integral_error']:.6e}")

    print(
        f"logw std = "
        f"{final_metrics['std_logw']:.6f}"
    )

    print(
        f"logw max = "
        f"{final_metrics['max_logw']:.6f}"
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "benchmark":
            benchmark,

        "graph":
            graph,

        "model_state_dict":
            cpu_state_dict(
                current_model
            ),

        "marginal_state_dict":
            marginal_state,

        "final_metrics":
            final_metrics,

        "training_mode":
            (
                "adaptive importance sampling; "
                "integrand only; "
                "no exact target sampler"
            ),

        "is_points_per_condition":
            args.is_points,

        "adaptive_rounds":
            N_ADAPTIVE_KL_ROUNDS,

        "speed_profile":
            args.speed_profile,

        "joint_marginal_copula":
            args.joint_marginal_copula,

        "chi2_enabled":
            CHI2_ENABLED,

        "annealed_bridge_metrics":
            annealed_metrics,
    }

    torch.save(
        checkpoint,
        output_dir
        /
        f"{benchmark}_final_model.pt",
    )

    pd.DataFrame(
        history_rows
    ).to_csv(
        output_dir
        /
        f"{benchmark}_history.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "benchmark":
                    benchmark,

                "edges":
                    count_edges(
                        graph
                    ),

                "ess":
                    final_metrics[
                        "ess"
                    ],

                "integral_estimate":
                    final_metrics[
                        "integral_estimate"
                    ],

                "log_integral_estimate":
                    final_metrics[
                        "log_integral_estimate"
                    ],

                "true_integral":
                    final_metrics[
                        "true_integral"
                    ],

                "absolute_integral_error":
                    final_metrics[
                        "absolute_integral_error"
                    ],

                "relative_integral_error":
                    final_metrics[
                        "relative_integral_error"
                    ],

                "std_logw":
                    final_metrics[
                        "std_logw"
                    ],

                "graph":
                    str(
                        graph
                    ),
            }
        ]
    ).to_csv(
        output_dir
        /
        f"{benchmark}_final_summary.csv",
        index=False,
    )


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speed-profile",
        choices=("smoke", "fast", "balanced", "thorough"),
        default="thorough",
        help=(
            "Compute allocation. 'thorough' is the default and preserves the "
            "original sample and optimization budgets."
        ),
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "regime_switch"
        ],
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=
            (
                "cuda"
                if
                torch.cuda.is_available()
                else
                "cpu"
            ),
    )

    parser.add_argument(
        "--inverse-iterations",
        type=int,
        default=40,
        help=(
            "Bisection steps in the spline inverse during training. The final "
            "Newton correction is unchanged; 8-12 steps agree with the "
            "40-step reference to ~1e-5 in the log-Jacobian in float32 and "
            "cost 2-3x less per dimension. Evaluation keeps 40."
        ),
    )

    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Override the fixed annealing schedule (for example "
            "--temperatures 0.01 0.025 0.05 0.1 0.25 0.5 1.0). Must be "
            "strictly increasing in (0, 1] and end at 1.0."
        ),
    )

    parser.add_argument(
        "--compile-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compile the deterministic integrand kernel on CUDA. Disable with "
            "--no-compile-kernels if the installed PyTorch lacks compiler support."
        ),
    )

    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimentally compile model.log_prob. Disabled by default because "
            "the current variable-width autoregressive contexts trigger many "
            "Dynamo recompilations."
        ),
    )

    parser.add_argument(
        "--joint-marginal-copula",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental joint marginal/copula fitting. Disabled by default "
            "so the original stable two-pass marginal-then-copula fit is used."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Points per step for the marginal-fit stages (and the chi2/masked "
            "steps). NOTE: this does NOT set the KL batch -- that is "
            "--is-points x ADAPTIVE_CONDITIONS_PER_STEP (256). Defaults to "
            "10 000, or 2 048 under --speed-profile smoke. Passing it "
            "explicitly overrides the smoke cap."
        ),
    )

    parser.add_argument(
        "--is-points",
        type=int,
        default=None,
        help=(
            "Points per condition in each importance-sampling draw, and "
            "therefore half of the KL batch: the KL batch is "
            "--is-points x ADAPTIVE_CONDITIONS_PER_STEP (256). Defaults to "
            f"{IS_POINTS_PER_CONDITION}, or 32 under --speed-profile smoke. "
            "Passing it explicitly overrides the smoke cap."
        ),
    )

    parser.add_argument(
        "--evidence-conditions",
        type=int,
        default=
            10_000,
    )

    parser.add_argument(
        "--evidence-resamples",
        type=int,
        default=
            10,
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=
            MAX_ROUNDS,
    )

    parser.add_argument(
        "--pip-threshold",
        type=float,
        default=
            PIP_REMOVE_THRESHOLD,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=
            list(
                DEFAULT_SEEDS
            ),
    )

    # Bayesian fields expected by bayesian_round().
    parser.add_argument(
        "--mcmc-iter",
        type=int,
        default=
            MCMC_ITER,
    )

    parser.add_argument(
        "--mcmc-burn",
        type=int,
        default=
            MCMC_BURN,
    )

    parser.add_argument(
        "--mcmc-thin",
        type=int,
        default=
            MCMC_THIN,
    )

    parser.add_argument(
        "--rho-a",
        type=float,
        default=
            RHO_A,
    )

    parser.add_argument(
        "--rho-b",
        type=float,
        default=
            RHO_B,
    )

    parser.add_argument(
        "--bayes-seed",
        "--bayes-random-seed",
        dest="bayes_seed",
        type=int,
        default=
            BAYES_RANDOM_SEED,
    )

    parser.add_argument(
        "--keep-prob",
        type=float,
        default=
            MASKED_KEEP_PROB,
    )

    parser.add_argument(
        "--masked-steps",
        type=int,
        default=
            MASKED_STEPS,
    )

    parser.add_argument(
        "--refine-steps",
        type=int,
        default=
            REFINE_STEPS,
    )

    return parser.parse_args()


def configure_speed_profile(profile):
    """Set compute budgets without changing the estimator or target.

    All profiles retain defensive-mixture sampling, independent validation,
    adaptive temperatures, and final importance correction.  They differ only
    in cache sizes and optimization budgets.
    """

    global MARGINAL_TRAIN_CONDITIONS
    global UNIFORM_BOOTSTRAP_POINTS, MARGINAL_ADAPTIVE_POINTS
    global BOOTSTRAP_VAL_CONDITIONS, VAL_CONDITIONS
    global ADAPTIVE_VAL_CONDITIONS
    global TEMPERATURE_PROBE_CONDITIONS, TEMPERATURE_PROBE_POINTS
    global TEMPERATURE_TARGET_MEDIAN_RAW_ESS
    global TEMPERATURE_TARGET_Q05_RAW_ESS
    global TEMPERATURE_MIN_INCREMENT, TEMPERATURE_MAX_INCREMENT
    global BRIDGE_COPULA_MAX_STEPS, BRIDGE_COPULA_CHECK_INTERVAL
    global BRIDGE_COPULA_PATIENCE
    global MIN_MARGINAL_STEPS, MAX_MARGINAL_STEPS
    global MARGINAL_CHECK_INTERVAL, MARGINAL_EARLY_STOP_PATIENCE
    global KL_MAX_STEPS, KL_MIN_STEPS, KL_CHECK_INTERVAL
    global KL_CONVERGENCE_PATIENCE, N_ADAPTIVE_KL_ROUNDS
    global CHI2_MAX_STEPS, CHI2_CHECK_INTERVAL
    global CHI2_CONDITIONS_PER_STEP, CHI2_POINTS_PER_CONDITION
    global CHI2_VAL_CONDITIONS, CHI2_VAL_POINTS_PER_CONDITION
    global FINAL_ESS_SIZE, DELTA_EVAL_SIZE

    profiles = {
        "smoke": {
            "uniform_bootstrap_points": 128,
            "marginal_adaptive_points": 128,
            "bootstrap_val_conditions": 64,
            "marginal_train_conditions": 32,
            "adaptive_val_conditions": 64,
            "temperature_probe_conditions": 32,
            "temperature_probe_points": 128,
            "temperature_median_ess": 16.0,
            "temperature_q05_ess": 3.0,
            "temperature_min_increment": 0.02,
            "temperature_max_increment": 0.40,
            "bridge_max_steps": 100,
            "bridge_check_interval": 50,
            "bridge_patience": 1,
            "marginal_min_steps": 50,
            "marginal_max_steps": 150,
            "marginal_check_interval": 50,
            "marginal_patience": 1,
            "kl_max_steps": 300,
            "kl_min_steps": 100,
            "kl_check_interval": 100,
            "kl_convergence_patience": 1,
            "adaptive_rounds": 1,
            "chi2_max_steps": 300,
            "chi2_check_interval": 100,
            "chi2_conditions_per_step": 64,
            "chi2_points_per_condition": 16,
            "chi2_val_conditions": 128,
            "chi2_val_points": 16,
            "final_ess_size": 5_000,
            "delta_eval_size": 5_000,
        },
        "fast": {
            "uniform_bootstrap_points": 512,
            "marginal_adaptive_points": 512,
            "bootstrap_val_conditions": 256,
            "marginal_train_conditions": 128,
            "adaptive_val_conditions": 256,
            "temperature_probe_conditions": 96,
            "temperature_probe_points": 256,
            "temperature_median_ess": 48.0,
            "temperature_q05_ess": 8.0,
            "temperature_min_increment": 0.015,
            "temperature_max_increment": 0.30,
            "bridge_max_steps": 400,
            "bridge_check_interval": 100,
            "bridge_patience": 1,
            "marginal_min_steps": 200,
            "marginal_max_steps": 800,
            "marginal_check_interval": 100,
            "marginal_patience": 1,
            "kl_max_steps": 6000,
            "kl_min_steps": 900,
            "kl_check_interval": 300,
            "kl_convergence_patience": 2,
            "adaptive_rounds": 2,
            "chi2_max_steps": 2400,
            "chi2_check_interval": 300,
            "chi2_conditions_per_step": 512,
            "chi2_points_per_condition": 32,
            "chi2_val_conditions": 512,
            "chi2_val_points": 32,
            "final_ess_size": 50_000,
            "delta_eval_size": 100_000,
        },
        "balanced": {
            "uniform_bootstrap_points": 512,
            "marginal_adaptive_points": 512,
            "bootstrap_val_conditions": 256,
            "marginal_train_conditions": 256,
            "adaptive_val_conditions": 384,
            "temperature_probe_conditions": 128,
            "temperature_probe_points": 384,
            "temperature_median_ess": 64.0,
            "temperature_q05_ess": 12.0,
            "temperature_min_increment": 0.01,
            "temperature_max_increment": 0.25,
            "bridge_max_steps": 600,
            "bridge_check_interval": 100,
            "bridge_patience": 1,
            "marginal_min_steps": 300,
            "marginal_max_steps": 1200,
            "marginal_check_interval": 100,
            "marginal_patience": 2,
            "kl_max_steps": 12_000,
            "kl_min_steps": 1200,
            "kl_check_interval": 400,
            "kl_convergence_patience": 2,
            "adaptive_rounds": 2,
            "chi2_max_steps": 3600,
            "chi2_check_interval": 300,
            "chi2_conditions_per_step": 512,
            "chi2_points_per_condition": 32,
            "chi2_val_conditions": 1000,
            "chi2_val_points": 32,
            "final_ess_size": 100_000,
            "delta_eval_size": 100_000,
        },
        "thorough": {
            "uniform_bootstrap_points": 512,
            "marginal_adaptive_points": 512,
            "bootstrap_val_conditions": 256,
            "marginal_train_conditions": 1024,
            "adaptive_val_conditions": 1000,
            "temperature_probe_conditions": 256,
            "temperature_probe_points": 512,
            "temperature_median_ess": 80.0,
            "temperature_q05_ess": 25.0,
            "temperature_min_increment": 0.005,
            "temperature_max_increment": 0.20,
            "bridge_max_steps": 1800,
            "bridge_check_interval": 300,
            "bridge_patience": 3,
            "marginal_min_steps": 1200,
            "marginal_max_steps": 6000,
            "marginal_check_interval": 300,
            "marginal_patience": 3,
            "kl_max_steps": 30_000,
            "kl_min_steps": 1800,
            "kl_check_interval": 600,
            "kl_convergence_patience": 3,
            "adaptive_rounds": 3,
            "chi2_max_steps": 6000,
            "chi2_check_interval": 300,
            "chi2_conditions_per_step": 512,
            "chi2_points_per_condition": 32,
            "chi2_val_conditions": 2000,
            "chi2_val_points": 32,
            "final_ess_size": 100_000,
            "delta_eval_size": 100_000,
        },
    }
    config = profiles[profile]
    UNIFORM_BOOTSTRAP_POINTS = config["uniform_bootstrap_points"]
    MARGINAL_ADAPTIVE_POINTS = config["marginal_adaptive_points"]
    BOOTSTRAP_VAL_CONDITIONS = config["bootstrap_val_conditions"]
    VAL_CONDITIONS = config["bootstrap_val_conditions"]
    MARGINAL_TRAIN_CONDITIONS = config["marginal_train_conditions"]
    ADAPTIVE_VAL_CONDITIONS = config["adaptive_val_conditions"]
    TEMPERATURE_PROBE_CONDITIONS = config["temperature_probe_conditions"]
    TEMPERATURE_PROBE_POINTS = config["temperature_probe_points"]
    TEMPERATURE_TARGET_MEDIAN_RAW_ESS = config["temperature_median_ess"]
    TEMPERATURE_TARGET_Q05_RAW_ESS = config["temperature_q05_ess"]
    TEMPERATURE_MIN_INCREMENT = config["temperature_min_increment"]
    TEMPERATURE_MAX_INCREMENT = config["temperature_max_increment"]
    BRIDGE_COPULA_MAX_STEPS = config["bridge_max_steps"]
    BRIDGE_COPULA_CHECK_INTERVAL = config["bridge_check_interval"]
    BRIDGE_COPULA_PATIENCE = config["bridge_patience"]
    MIN_MARGINAL_STEPS = config["marginal_min_steps"]
    MAX_MARGINAL_STEPS = config["marginal_max_steps"]
    MARGINAL_CHECK_INTERVAL = config["marginal_check_interval"]
    MARGINAL_EARLY_STOP_PATIENCE = config["marginal_patience"]
    KL_MAX_STEPS = config["kl_max_steps"]
    KL_MIN_STEPS = config["kl_min_steps"]
    KL_CHECK_INTERVAL = config["kl_check_interval"]
    KL_CONVERGENCE_PATIENCE = config["kl_convergence_patience"]
    N_ADAPTIVE_KL_ROUNDS = config["adaptive_rounds"]
    CHI2_MAX_STEPS = config["chi2_max_steps"]
    CHI2_CHECK_INTERVAL = config["chi2_check_interval"]
    CHI2_CONDITIONS_PER_STEP = config["chi2_conditions_per_step"]
    CHI2_POINTS_PER_CONDITION = config["chi2_points_per_condition"]
    CHI2_VAL_CONDITIONS = config["chi2_val_conditions"]
    CHI2_VAL_POINTS_PER_CONDITION = config["chi2_val_points"]
    FINAL_ESS_SIZE = config["final_ess_size"]
    DELTA_EVAL_SIZE = config["delta_eval_size"]


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    configure_speed_profile(args.speed_profile)

    global TRAINING_INVERSE_ITERATIONS
    TRAINING_INVERSE_ITERATIONS = max(1, int(args.inverse_iterations))

    if args.temperatures is not None:
        temperatures = [float(value) for value in args.temperatures]
        if (
            any(not (0.0 < value <= 1.0) for value in temperatures)
            or abs(temperatures[-1] - 1.0) > 1e-9
            or any(right <= left for left, right in zip(temperatures, temperatures[1:]))
        ):
            raise SystemExit(
                "--temperatures must be strictly increasing in (0, 1] "
                "and end at 1.0"
            )

    if args.speed_profile == "smoke":
        # Exercise every major code path quickly. These settings are deliberately
        # unsuitable for accuracy claims or comparisons.
        # An explicit --is-points or --batch-size survives the smoke cap; only
        # the defaults are capped. See their help text.
        if args.is_points is None:
            args.is_points = 32
        args.max_rounds = min(args.max_rounds, 1)
        args.evidence_conditions = min(args.evidence_conditions, 256)
        args.evidence_resamples = min(args.evidence_resamples, 2)
        args.masked_steps = min(args.masked_steps, 50)
        args.refine_steps = min(args.refine_steps, 25)
        # An explicit --batch-size survives the smoke cap; only the default is
        # capped. See the --batch-size help text.
        if args.batch_size is None:
            args.batch_size = 2048
        args.mcmc_iter = min(args.mcmc_iter, 100)
        args.mcmc_burn = min(args.mcmc_burn, 20)
        args.mcmc_thin = min(args.mcmc_thin, 2)

    global TRAIN_BATCH_SIZE, DIM, target_log_integrand
    global COMPILE_TRAINING_KERNELS

    if args.batch_size is None:
        args.batch_size = TRAIN_BATCH_SIZE

    if args.is_points is None:
        args.is_points = IS_POINTS_PER_CONDITION

    TRAIN_BATCH_SIZE = (
        args.batch_size
    )

    device = torch.device(
        args.device
    )
    COMPILE_TRAINING_KERNELS = bool(
        args.compile_model and device.type == "cuda"
    )
    if COMPILE_TRAINING_KERNELS:
        from torch import _dynamo
        _dynamo.config.suppress_errors = True

    if (
        args.compile_kernels
        and device.type == "cuda"
        and hasattr(torch, "compile")
    ):
        try:
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            # sparse_wave_32 contains many small elementwise operations. The
            # compiler fuses them, while dynamic=True permits changing batch
            # sizes without compiling a new graph at every training phase.
            target_log_integrand = torch.compile(
                target_log_integrand,
                mode="reduce-overhead",
                dynamic=True,
            )
            print("CUDA integrand compilation = enabled")
            print(
                "CUDA training compilation  = "
                f"{'enabled (experimental)' if COMPILE_TRAINING_KERNELS else 'disabled'}"
            )
        except Exception as error:
            print(
                "WARNING: torch.compile setup failed; using eager kernels: "
                f"{error}"
            )

    for benchmark in (
        args.benchmarks
    ):

        if (
            benchmark
            not in
            SUPPORTED_BENCHMARKS
        ):

            raise ValueError(
                f"Unsupported benchmark "
                f"{benchmark}. "
                f"Supported: "
                f"{SUPPORTED_BENCHMARKS}"
            )

    requested_dimensions = {
        BENCHMARK_DIMENSIONS[benchmark]
        for benchmark in args.benchmarks
    }
    if len(requested_dimensions) != 1:
        raise ValueError(
            "Benchmarks with different dimensions must be run separately: "
            f"{[(b, BENCHMARK_DIMENSIONS[b]) for b in args.benchmarks]}"
        )
    DIM = requested_dimensions.pop()

    print()
    print(
        "=" * 120
    )

    print(
        "ADAPTIVE INTEGRAND-ONLY "
        "CONDITIONAL NF"
    )

    print(
        "=" * 120
    )

    print(
        f"device                  = "
        f"{device}"
    )

    print(
        f"dimension               = "
        f"{DIM}"
    )

    print(
        f"speed profile           = "
        f"{args.speed_profile}"
    )

    print(
        f"training batch size     = "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        f"benchmarks              = "
        f"{args.benchmarks}"
    )

    print(
        "target sampler          = NEVER USED"
    )

    print(
        "available target info   = f(x|c)"
    )

    print(
        f"IS points / condition   = "
        f"{args.is_points}"
    )

    print(
        f"resamples / condition   = "
        f"{RESAMPLES_PER_CONDITION}"
    )

    print(
        f"adaptive KL rounds      = "
        f"{N_ADAPTIVE_KL_ROUNDS}"
    )

    print(
        "compute budgets          = "
        f"marginal conditions {MARGINAL_TRAIN_CONDITIONS}, "
        f"marginal steps {MIN_MARGINAL_STEPS}-{MAX_MARGINAL_STEPS}, "
        f"adaptive val {ADAPTIVE_VAL_CONDITIONS}, "
        f"bridge steps {BRIDGE_COPULA_MAX_STEPS}, "
        f"KL max {KL_MAX_STEPS}, chi2 max {CHI2_MAX_STEPS}"
    )

    print(
        "marginal temperatures   = adaptive (conditional ESS controlled)"
    )

    print(
        "temperature ESS targets = "
        f"median {TEMPERATURE_TARGET_MEDIAN_RAW_ESS:.0f}, "
        f"q05 {TEMPERATURE_TARGET_Q05_RAW_ESS:.0f}"
    )

    print(
        f"proposal uniform share  = {PROPOSAL_UNIFORM_FRACTION:.2f}"
    )

    print(
        "KL LR ladder           = "
        +
        " -> ".join(
            f"{lr:.0e}"
            for lr
            in KL_LR_LEVELS
        )
    )

    print(
        f"chi2 fine tuning        = "
        f"{CHI2_ENABLED}"
    )

    print(
        f"marginal/copula fitting = "
        f"{'joint weighted objective' if args.joint_marginal_copula else 'legacy two-pass'}"
    )

    print(
        "candidate search        = "
        "all weak -> first half "
        "-> second half if needed"
    )

    print(
        "graph acceptance        = ESS only"
    )

    print(
        "=" * 120
    )

    output_dir = Path(
        args.output
    )

    for benchmark in (
        args.benchmarks
    ):

        run_benchmark(
            benchmark=
                benchmark,

            args=
                args,

            device=
                device,

            output_dir=
                output_dir,
        )


if __name__ == "__main__":
    main()
