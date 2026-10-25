import argparse
import copy
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import vegas

# ============================================================
# Frozen successful baseline.
#
# DO NOT MODIFY:
#     run_fast_multijump_pruning.py
#
# We reuse its tested model classes, marginal training,
# masked edge evidence, and Bayesian screening.
# ============================================================

import run_fast_multijump_pruning as base


# ============================================================
# Global configuration
# ============================================================

DIM = base.DIM

NEW_BENCHMARKS = {
    "regime_switch",
    "cross_coupled",
}

DEFAULT_BENCHMARKS = [
    "regime_switch",
    "cross_coupled",
]

DEFAULT_OUTPUT = (
    "results/optimized_unknown_benchmarks"
)


# ============================================================
# Training configuration
# ============================================================

TRAIN_BATCH_SIZE = 10_000

# Marginals stay exactly as before.
MARGINAL_LR = 1e-4

# NEW: faster copula optimizer.
COPULA_LR = 3e-4

MIN_CANDIDATE_STEPS = 1800

# Was 12,000 in successful baseline.
MAX_CANDIDATE_STEPS = 30_000

KL_CHECK_INTERVAL = 600

REL_KL_TOL = 0.01
CONVERGENCE_PATIENCE = 2


# ============================================================
# Integration comparison
# ============================================================

FIXED_CONDITION = [
    0.5,
    0.5,
    0.5,
]

NF_SAMPLE_SIZES = [
    100_000,
    1_000_000,
]

NF_REPEATS = 5

CPU_CHUNK_SIZE = 100_000


# ============================================================
# VEGAS
# ============================================================

VEGAS_ADAPT_NITN = [
    10,
    20,
    50,
    100,
    200,
]

VEGAS_ADAPT_NEVAL = 100_000

VEGAS_PROD_NITN = 10
VEGAS_PROD_NEVAL = 100_000

VEGAS_REPEATS = 5


# ============================================================
# Visualization only.
#
# Search algorithm NEVER reads these.
# ============================================================

DIAGNOSTIC_PAIRS = {

    "regime_switch": [
        (0, 2),
        (2, 3),
        (4, 5),
        (5, 6),
    ],

    "cross_coupled": [
        (0, 2),
        (1, 4),
        (2, 5),
        (4, 7),
    ],
}


# ============================================================
# General helpers
# ============================================================

def cpu_state_dict(model):

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in model.state_dict().items()
    }


def graph_key(graph):

    return tuple(
        (
            int(child),

            tuple(
                sorted(
                    int(parent)
                    for parent
                    in graph[child]
                )
            ),
        )

        for child
        in sorted(graph)
    )


def edge_count(graph):

    return sum(
        len(
            graph[child]
        )

        for child
        in graph
    )


# ============================================================
# Standard-normal utilities
# ============================================================

def standard_normal_cdf(x):

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


def standard_normal_icdf(p):

    p = p.clamp(
        1e-7,
        1.0 - 1e-7,
    )

    return (
        math.sqrt(2.0)
        *
        torch.erfinv(
            2.0 * p - 1.0
        )
    )


# ============================================================
# Truncated normal [0,1]
# ============================================================

def sample_truncated_normal(
    mu,
    sigma,
):

    a = (
        0.0 - mu
    ) / sigma

    b = (
        1.0 - mu
    ) / sigma

    cdf_a = (
        standard_normal_cdf(
            a
        )
    )

    cdf_b = (
        standard_normal_cdf(
            b
        )
    )

    r = torch.rand_like(
        mu
    )

    p = (
        cdf_a
        +
        r
        *
        (
            cdf_b
            -
            cdf_a
        )
    )

    z = (
        standard_normal_icdf(
            p
        )
    )

    return (
        mu
        +
        sigma
        *
        z
    ).clamp(
        0.0,
        1.0,
    )


def truncated_normal_logpdf(
    x,
    mu,
    sigma,
):

    z = (
        x - mu
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
        0.0 - mu
    ) / sigma

    b = (
        1.0 - mu
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


# ============================================================
# UNKNOWN BENCHMARK 1
#
# regime_switch
#
# Dependency usefulness varies with condition c0.
#
# IMPORTANT:
# The graph search never reads this mapping.
# ============================================================

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

    parent_pairs = {

        2:
            (0, 1),

        3:
            (2, 0),

        4:
            (1, 3),

        5:
            (4, 2),

        6:
            (5, 3),

        7:
            (6, 1),
    }

    p_a, p_b = (
        parent_pairs[
            child
        ]
    )

    a = x[
        :,
        p_a:p_a + 1
    ]

    b = x[
        :,
        p_b:p_b + 1
    ]

    # Smoothly switch which parent is more important.
    gate = torch.sigmoid(
        8.0
        *
        (
            c0 - 0.5
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
            a - b
        )
    )

    signal = (
        gate
        *
        branch_a
        +
        (
            1.0 - gate
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


# ============================================================
# UNKNOWN BENCHMARK 2
#
# cross_coupled
#
# Sparse nonlinear multi-parent interactions.
#
# Search never receives the true parent structure.
# ============================================================

def cross_coupled_mu_sigma(
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

        2:
            (0, 1),

        3:
            (0, 2),

        4:
            (1, 3),

        5:
            (2, 4),

        6:
            (0, 3, 5),

        7:
            (1, 4, 6),
    }

    parents = (
        parent_map[
            child
        ]
    )

    if len(
        parents
    ) == 2:

        p1, p2 = (
            parents
        )

        a = x[
            :,
            p1:p1 + 1
        ]

        b = x[
            :,
            p2:p2 + 1
        ]

        signal = (
            0.70
            *
            torch.sin(
                2.0
                *
                math.pi
                *
                (
                    a
                    +
                    0.10
                    *
                    c0
                )
            )
            +
            0.50
            *
            torch.cos(
                2.0
                *
                math.pi
                *
                (
                    b
                    -
                    0.08
                    *
                    c1
                )
            )
            +
            0.35
            *
            torch.sin(
                2.0
                *
                math.pi
                *
                (
                    a - b
                )
            )
        )

    else:

        p1, p2, p3 = (
            parents
        )

        a = x[
            :,
            p1:p1 + 1
        ]

        b = x[
            :,
            p2:p2 + 1
        ]

        d = x[
            :,
            p3:p3 + 1
        ]

        signal = (
            0.52
            *
            torch.sin(
                2.0
                *
                math.pi
                *
                (
                    a + b
                )
            )
            +
            0.48
            *
            torch.cos(
                2.0
                *
                math.pi
                *
                (
                    d - b
                )
            )
            +
            1.20
            *
            (
                a - 0.5
            )
            *
            (
                d - 0.5
            )
            +
            0.16
            *
            torch.sin(
                2.0
                *
                math.pi
                *
                c0
            )
        )

    mu = (
        0.50
        +
        0.29
        *
        torch.tanh(
            1.35
            *
            signal
        )
    )

    sigma = (
        0.041
        +
        0.010
        *
        c2
    )

    return (
        mu,
        sigma,
    )


# ============================================================
# Unified conditional law
# ============================================================

def benchmark_mu_sigma(
    benchmark,
    child,
    x,
    cond,
):

    if benchmark == "regime_switch":

        return (
            regime_switch_mu_sigma(
                child,
                x,
                cond,
            )
        )

    if benchmark == "cross_coupled":

        return (
            cross_coupled_mu_sigma(
                child,
                x,
                cond,
            )
        )

    raise ValueError(
        f"Unknown benchmark: "
        f"{benchmark}"
    )


# ============================================================
# Exact target sampler
#
# x0 and x1 are uniform roots.
# ============================================================

def sample_target(
    benchmark,
    cond,
    dim,
):

    n = (
        cond.shape[0]
    )

    x = torch.empty(
        n,
        dim,
        device=cond.device,
        dtype=cond.dtype,
    )

    x[
        :,
        0:2
    ] = torch.rand(
        n,
        2,
        device=cond.device,
        dtype=cond.dtype,
    )

    for child in range(
        2,
        dim,
    ):

        mu, sigma = (
            benchmark_mu_sigma(
                benchmark,
                child,
                x,
                cond,
            )
        )

        x[
            :,
            child:child + 1
        ] = (
            sample_truncated_normal(
                mu,
                sigma,
            )
        )

    return x


# ============================================================
# Exact normalized target log density
#
# Integral is exactly 1.
# ============================================================

def target_logpdf(
    benchmark,
    x,
    cond,
):

    logp = torch.zeros(
        x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )

    for child in range(
        2,
        DIM,
    ):

        mu, sigma = (
            benchmark_mu_sigma(
                benchmark,
                child,
                x,
                cond,
            )
        )

        logp += (
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

    return logp


# ============================================================
# Search-facing benchmark description.
#
# Deliberately does not reveal graph truth.
# ============================================================

def benchmark_description(
    benchmark,
):

    if benchmark == "regime_switch":

        return (
            "UNKNOWN HOLDOUT: "
            "condition-dependent nonlinear target"
        )

    if benchmark == "cross_coupled":

        return (
            "UNKNOWN HOLDOUT: "
            "nonlinear multi-parent target"
        )

    return benchmark


# ============================================================
# Patch the successful machinery so its marginal/evidence/KL
# helpers operate on the new holdout targets.
# ============================================================

base.sample_target = (
    sample_target
)

base.target_logpdf = (
    target_logpdf
)

base.true_graph_description = (
    benchmark_description
)


# ============================================================
# NEW COPULA TRAINER
#
# This is deliberately local to THIS script.
#
# Marginals remain unchanged.
#
# Differences from baseline:
#
#   lr = args.copula_lr
#   max steps default = 30,000
#
# Same:
#
#   forward-KL objective
#   fresh p samples
#   gradient clipping = 10
#   fixed KL cache
#   relative improvement stopping
# ============================================================

def train_copula_until_converged(
    model,
    benchmark,
    kl_cache,
    args,
    device,
):

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=args.copula_lr,
    )

    model.train()

    history = []

    total_steps = 0

    previous_kl = None

    convergence_count = 0

    while (
        total_steps
        <
        args.max_candidate_steps
    ):

        block_steps = min(
            args.kl_check_interval,

            args.max_candidate_steps
            -
            total_steps,
        )

        recent_losses = []

        for _ in range(
            block_steps
        ):

            cond = (
                base.sample_conditions(
                    args.batch_size,
                    device,
                )
            )

            with torch.no_grad():

                x = sample_target(
                    benchmark,
                    cond,
                    DIM,
                )

            logq = model.log_prob(
                x,
                cond,
            )

            loss = (
                -logq.mean()
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.copula.parameters(),
                10.0,
            )

            optimizer.step()

            recent_losses.append(
                loss.item()
            )

        total_steps += (
            block_steps
        )

        kl = (
            base.cached_forward_kl(
                model,
                kl_cache,
            )
        )

        mean_loss = float(
            np.mean(
                recent_losses[
                    -100:
                ]
            )
        )

        if previous_kl is None:

            absolute_improvement = float(
                "nan"
            )

            relative_improvement = float(
                "nan"
            )

        else:

            absolute_improvement = (
                previous_kl
                -
                kl
            )

            relative_improvement = (
                absolute_improvement
                /
                max(
                    abs(
                        previous_kl
                    ),
                    1e-8,
                )
            )

        history.append(
            {

                "step":
                    total_steps,

                "kl":
                    kl,

                "nll":
                    mean_loss,

                "absolute_improvement":
                    absolute_improvement,

                "relative_improvement":
                    relative_improvement,

                "learning_rate":
                    args.copula_lr,
            }
        )

        if previous_kl is None:

            print(
                f"    step "
                f"{total_steps:6d} "
                f"NLL="
                f"{mean_loss:.6f} "
                f"KL="
                f"{kl:.6f} "
                f"lr="
                f"{args.copula_lr:.1e}"
            )

        else:

            print(
                f"    step "
                f"{total_steps:6d} "
                f"NLL="
                f"{mean_loss:.6f} "
                f"KL="
                f"{kl:.6f} "
                f"dKL="
                f"{absolute_improvement:+.6f} "
                f"rel="
                f"{100.0 * relative_improvement:+.3f}% "
                f"lr="
                f"{args.copula_lr:.1e}"
            )

        # ====================================================
        # Same convergence rule as successful baseline
        # ====================================================

        if (
            previous_kl is not None
            and
            total_steps
            >=
            args.min_candidate_steps
        ):

            if (
                relative_improvement
                <
                args.rel_kl_tol
            ):

                convergence_count += 1

            else:

                convergence_count = 0

            print(
                f"      copula convergence "
                f"{convergence_count}/"
                f"{args.convergence_patience}"
            )

            if (
                convergence_count
                >=
                args.convergence_patience
            ):

                print(
                    "      COPULA CONVERGED "
                    f"at {total_steps} steps"
                )

                break

        previous_kl = (
            kl
        )

        model.train()

    converged = (
        convergence_count
        >=
        args.convergence_patience
    )

    if (
        not converged
        and
        total_steps
        >=
        args.max_candidate_steps
    ):

        print(
            "      copula reached max steps "
            f"{args.max_candidate_steps}"
        )

    return {

        "history":
            history,

        "final_steps":
            total_steps,

        "final_kl":
            history[
                -1
            ][
                "kl"
            ],

        "converged":
            converged,
    }


# ============================================================
# Use NEW trainer everywhere inside this script.
#
# This does NOT edit the baseline .py file on disk.
# ============================================================

base.train_model_until_converged = (
    train_copula_until_converged
)


# ============================================================
# Initial full graph
# ============================================================

def train_initial_graph(
    benchmark,
    graph,
    marginal_state,
    kl_cache,
    args,
    device,
):

    base.seed_everything(
        args.performance_seed
    )

    model = (
        base.make_real_model(
            graph,
            marginal_state,
            device,
        )
    )

    print()
    print(
        "=" * 90
    )

    print(
        "TRAIN INITIAL FULL GRAPH"
    )

    print(
        f"copula lr  = "
        f"{args.copula_lr:.1e}"
    )

    print(
        f"max steps  = "
        f"{args.max_candidate_steps:,}"
    )

    print(
        "=" * 90
    )

    training_result = (
        train_copula_until_converged(
            model=
                model,

            benchmark=
                benchmark,

            kl_cache=
                kl_cache,

            args=
                args,

            device=
                device,
        )
    )

    convergence = (
        base.summarize_convergence(
            training_result
        )
    )

    metrics = (
        base.evaluate_ess(
            model=
                model,

            benchmark=
                benchmark,

            device=
                device,

            args=
                args,
        )
    )

    metrics[
        "kl"
    ] = (
        base.cached_forward_kl(
            model,
            kl_cache,
        )
    )

    print()
    print(
        "INITIAL PERFORMANCE"
    )

    print(
        f"  steps = "
        f"{convergence['training_steps']}"
    )

    print(
        f"  conv  = "
        f"{convergence['converged']}"
    )

    print(
        f"  ESS   = "
        f"{metrics['ess']:.6f}"
    )

    print(
        f"  KL    = "
        f"{metrics['kl']:.6f}"
    )

    return (
        model,
        metrics,
        convergence,
    )


# ============================================================
# Candidate graph cache
# ============================================================

def restore_cached_model(
    cached,
    graph,
    marginal_state,
    device,
):

    model = (
        base.make_real_model(
            graph,
            marginal_state,
            device,
        )
    )

    model.load_state_dict(
        cached[
            "model_state"
        ]
    )

    model.eval()

    return model


# ============================================================
# Train or reuse candidate graph
# ============================================================

def train_or_reuse_candidate(
    benchmark,
    name,
    removals,
    current_graph,
    current_metrics,
    marginal_state,
    kl_cache,
    candidate_cache,
    args,
    device,
    round_id,
):

    candidate_graph = (
        base.remove_edges(
            current_graph,
            removals,
        )
    )

    key = (
        graph_key(
            candidate_graph
        )
    )

    if key in candidate_cache:

        print()
        print(
            f"[CACHE HIT] {name}"
        )

        cached = (
            candidate_cache[
                key
            ]
        )

        model = (
            restore_cached_model(
                cached=
                    cached,

                graph=
                    candidate_graph,

                marginal_state=
                    marginal_state,

                device=
                    device,
            )
        )

        metrics = copy.deepcopy(
            cached[
                "metrics"
            ]
        )

        convergence = copy.deepcopy(
            cached[
                "convergence"
            ]
        )

        reused = True

    else:

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
            f"{edge_count(candidate_graph)}"
        )

        print(
            f"copula lr  = "
            f"{args.copula_lr:.1e}"
        )

        print(
            "=" * 100
        )

        base.seed_everything(
            args.performance_seed
        )

        model = (
            base.make_real_model(
                candidate_graph,
                marginal_state,
                device,
            )
        )

        training_result = (
            train_copula_until_converged(
                model=
                    model,

                benchmark=
                    benchmark,

                kl_cache=
                    kl_cache,

                args=
                    args,

                device=
                    device,
            )
        )

        convergence = (
            base.summarize_convergence(
                training_result
            )
        )

        metrics = (
            base.evaluate_ess(
                model=
                    model,

                benchmark=
                    benchmark,

                device=
                    device,

                args=
                    args,
            )
        )

        metrics[
            "kl"
        ] = (
            base.cached_forward_kl(
                model,
                kl_cache,
            )
        )

        candidate_cache[
            key
        ] = {

            "model_state":
                cpu_state_dict(
                    model
                ),

            "metrics":
                copy.deepcopy(
                    metrics
                ),

            "convergence":
                copy.deepcopy(
                    convergence
                ),
        }

        reused = False

    delta_ess = (
        metrics[
            "ess"
        ]
        -
        current_metrics[
            "ess"
        ]
    )

    delta_kl = (
        metrics[
            "kl"
        ]
        -
        current_metrics[
            "kl"
        ]
    )

    # ========================================================
    # ESS ONLY.
    # ========================================================

    accept = (
        delta_ess
        >=
        -args.ess_tol
    )

    result = {

        "name":
            name,

        "removals":
            list(
                removals
            ),

        "graph":
            candidate_graph,

        "model":
            model,

        "ess":
            metrics[
                "ess"
            ],

        "kl":
            metrics[
                "kl"
            ],

        "variance":
            metrics[
                "variance"
            ],

        "relative_error":
            metrics[
                "relative_error"
            ],

        "delta_ess":
            delta_ess,

        "delta_kl":
            delta_kl,

        "accept":
            accept,

        "reused":
            reused,

        **convergence,
    }

    print()
    print(
        f"RESULT: {name}"
    )

    print(
        f"  current ESS   = "
        f"{current_metrics['ess']:.6f}"
    )

    print(
        f"  candidate ESS = "
        f"{metrics['ess']:.6f}"
    )

    print(
        f"  delta ESS     = "
        f"{delta_ess:+.6f}"
    )

    print(
        f"  candidate KL  = "
        f"{metrics['kl']:.6f}"
    )

    print(
        f"  ESS pass      = "
        f"{accept}"
    )

    return result


# ============================================================
# NEW DECISION-TREE CANDIDATE SEARCH
#
# Usually two real trainings.
# Maximum three.
#
#
# 1. all weak
# 2. first half
#
# second half only if:
#
#   - all weak fails, OR
#   - first half beats all weak
#
# ============================================================

def optimized_candidate_search(
    benchmark,
    current_graph,
    current_metrics,
    ranked_edges,
    marginal_state,
    kl_cache,
    candidate_cache,
    args,
    device,
    round_id,
):

    edges = list(
        ranked_edges
    )

    if len(
        edges
    ) == 0:

        return (
            None,
            [],
        )

    midpoint = int(
        math.ceil(
            len(edges)
            /
            2
        )
    )

    first_half = (
        edges[
            :midpoint
        ]
    )

    second_half = (
        edges[
            midpoint:
        ]
    )

    tested = []

    trial_rows = []

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def test(
        name,
        removals,
    ):

        if len(
            removals
        ) == 0:

            return None

        result = (
            train_or_reuse_candidate(
                benchmark=
                    benchmark,

                name=
                    name,

                removals=
                    removals,

                current_graph=
                    current_graph,

                current_metrics=
                    current_metrics,

                marginal_state=
                    marginal_state,

                kl_cache=
                    kl_cache,

                candidate_cache=
                    candidate_cache,

                args=
                    args,

                device=
                    device,

                round_id=
                    round_id,
            )
        )

        tested.append(
            result
        )

        trial_rows.append(
            {

                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "candidate":
                    name,

                "reused":
                    int(
                        result[
                            "reused"
                        ]
                    ),

                "n_removed":
                    len(
                        removals
                    ),

                "removed_edges":
                    str(
                        removals
                    ),

                "edges_before":
                    edge_count(
                        current_graph
                    ),

                "edges_after":
                    edge_count(
                        result[
                            "graph"
                        ]
                    ),

                "training_steps":
                    result[
                        "training_steps"
                    ],

                "converged":
                    int(
                        result[
                            "converged"
                        ]
                    ),

                "last_dkl":
                    result[
                        "last_dkl"
                    ],

                "last_relative_improvement":
                    result[
                        "last_relative_improvement"
                    ],

                "ess_current":
                    current_metrics[
                        "ess"
                    ],

                "ess_candidate":
                    result[
                        "ess"
                    ],

                "delta_ess":
                    result[
                        "delta_ess"
                    ],

                "kl_current":
                    current_metrics[
                        "kl"
                    ],

                "kl_candidate":
                    result[
                        "kl"
                    ],

                "delta_kl":
                    result[
                        "delta_kl"
                    ],

                "ess_pass":
                    int(
                        result[
                            "accept"
                        ]
                    ),
            }
        )

        return result

    # ========================================================
    # 1. Remove all weak
    # ========================================================

    result_all = test(
        "all_weak",
        edges,
    )

    if len(
        edges
    ) == 1:

        candidates = [
            r
            for r
            in tested
            if r[
                "accept"
            ]
        ]

        if not candidates:

            return (
                None,
                trial_rows,
            )

        return (
            candidates[0],
            trial_rows,
        )

    # ========================================================
    # 2. First half
    # ========================================================

    result_first = test(
        "rank_first_half",
        first_half,
    )

    # ========================================================
    # 3. Decide whether second half is worth the cost
    # ========================================================

    test_second = False

    # If aggressive removal fails, a half may still work.
    if not result_all[
        "accept"
    ]:

        test_second = True

    # If first half beats all weak, subset structure matters.
    elif (
        result_first is not None
        and
        result_first[
            "ess"
        ]
        >
        result_all[
            "ess"
        ]
        +
        args.ess_tol
    ):

        test_second = True

    if (
        test_second
        and
        len(
            second_half
        )
        >
        0
    ):

        test(
            "rank_second_half",
            second_half,
        )

    # ========================================================
    # Pick highest ESS among passing candidates.
    #
    # ESS primary.
    # KL then sparsity only break exact-ish ties.
    # ========================================================

    acceptable = [
        candidate
        for candidate
        in tested
        if candidate[
            "accept"
        ]
    ]

    if len(
        acceptable
    ) == 0:

        for candidate in tested:

            del candidate[
                "model"
            ]

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return (
            None,
            trial_rows,
        )

    acceptable.sort(
        key=lambda candidate: (

            -
            candidate[
                "ess"
            ],

            candidate[
                "kl"
            ],

            edge_count(
                candidate[
                    "graph"
                ]
            ),
        )
    )

    winner = (
        acceptable[
            0
        ]
    )

    for candidate in tested:

        if candidate is not winner:

            del candidate[
                "model"
            ]

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print()
    print(
        "=" * 100
    )

    print(
        "CANDIDATE WINNER"
    )

    print(
        f"name      = "
        f"{winner['name']}"
    )

    print(
        f"edges     = "
        f"{edge_count(winner['graph'])}"
    )

    print(
        f"ESS       = "
        f"{winner['ess']:.6f}"
    )

    print(
        f"delta ESS = "
        f"{winner['delta_ess']:+.6f}"
    )

    print(
        "=" * 100
    )

    return (
        winner,
        trial_rows,
    )


# ============================================================
# Build evidence caches
# ============================================================

def build_evidence_caches(
    benchmark,
    marginal_states,
    args,
    device,
):

    cache_by_seed = {}

    for seed in args.seeds:

        print()
        print(
            f"Build u cache seed={seed}"
        )

        bank = (
            base.MarginalBank(
                DIM
            )
            .to(device)
        )

        bank.load_state_dict(
            marginal_states[
                seed
            ]
        )

        bank.eval()

        base.seed_everything(
            seed
            +
            500_000
        )

        u_train, c_train = (
            base.build_u_cache(
                benchmark=
                    benchmark,

                bank=
                    bank,

                n=
                    args.cache_size,

                device=
                    device,
            )
        )

        base.seed_everything(
            seed
            +
            700_000
        )

        u_eval, c_eval = (
            base.build_u_cache(
                benchmark=
                    benchmark,

                bank=
                    bank,

                n=
                    args.delta_eval_size,

                device=
                    device,
            )
        )

        cache_by_seed[
            seed
        ] = {

            "u_train":
                u_train,

            "c_train":
                c_train,

            "u_eval":
                u_eval,

            "c_eval":
                c_eval,
        }

        del bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return cache_by_seed


# ============================================================
# Save marginal training history
# ============================================================

def save_marginal_histories(
    benchmark,
    diagnostics,
    output_dir,
):

    rows = []

    for seed, result in (
        diagnostics.items()
    ):

        for row in (
            result[
                "history"
            ]
        ):

            rows.append(
                {

                    "benchmark":
                        benchmark,

                    "seed":
                        seed,

                    **row,
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        output_dir
        /
        f"{benchmark}_marginal_training.csv",

        index=False,
    )


# ============================================================
# CPU model reconstruction
# ============================================================

def reconstruct_cpu_model(
    checkpoint,
):

    device = torch.device(
        "cpu"
    )

    model = (
        base.make_real_model(
            checkpoint[
                "graph"
            ],

            checkpoint[
                "marginal_state_dict"
            ],

            device,
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


# ============================================================
# Fixed-condition NF CPU integration
#
# Log-domain accumulation to avoid weight overflow.
# ============================================================

@torch.no_grad()
def evaluate_nf_cpu(
    model,
    benchmark,
    condition,
    n_samples,
    seed,
    chunk_size,
):

    model.eval()

    base.seed_everything(
        seed
    )

    logw_parts = []

    generated = 0

    start_time = (
        time.perf_counter()
    )

    while generated < n_samples:

        n = min(
            chunk_size,
            n_samples
            -
            generated,
        )

        cond = (
            torch.tensor(
                condition,
                dtype=torch.float32,
            )
            .reshape(
                1,
                3,
            )
            .expand(
                n,
                -1,
            )
        )

        x = (
            base.sample_from_model(
                model,
                n,
                cond,
            )
        )

        logp = (
            target_logpdf(
                benchmark,
                x,
                cond,
            )
            .reshape(-1)
            .double()
        )

        logq = (
            model.log_prob(
                x,
                cond,
            )
            .reshape(-1)
            .double()
        )

        logw_parts.append(
            logp - logq
        )

        generated += n

    logw = torch.cat(
        logw_parts,
        dim=0,
    )

    n_total = (
        logw.numel()
    )

    log_n = math.log(
        n_total
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

    estimate = float(
        torch.exp(
            log_sum_w
            -
            log_n
        )
        .item()
    )

    ess = float(
        torch.exp(
            2.0
            *
            log_sum_w
            -
            log_n
            -
            log_sum_w2
        )
        .item()
    )

    relative_variance = (
        1.0
        /
        ess
        -
        1.0
    )

    estimated_error = math.sqrt(
        max(
            relative_variance,
            0.0,
        )
        /
        n_total
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    return {

        "benchmark":
            benchmark,

        "method":
            "selected_copula",

        "seed":
            seed,

        "production_evals":
            n_total,

        "estimate":
            estimate,

        "actual_error":
            abs(
                estimate - 1.0
            ),

        "ess":
            ess,

        "estimated_error":
            estimated_error,

        "production_wall_time":
            elapsed,
    }


# ============================================================
# VEGAS
# ============================================================

def make_vegas_integrand(
    benchmark,
    condition,
):

    cond_template = (
        torch.tensor(
            condition,
            dtype=torch.float64,
        )
        .reshape(
            1,
            3,
        )
    )

    @vegas.batchintegrand
    def integrand(x):

        x_torch = (
            torch.from_numpy(
                np.asarray(
                    x,
                    dtype=np.float64,
                )
            )
        )

        n = (
            x_torch.shape[
                0
            ]
        )

        cond = (
            cond_template
            .expand(
                n,
                -1,
            )
        )

        with torch.no_grad():

            logp = (
                target_logpdf(
                    benchmark,
                    x_torch,
                    cond,
                )
            )

            value = (
                torch.exp(
                    logp
                )
            )

        return (
            value
            .cpu()
            .numpy()
        )

    return integrand


def evaluate_vegas(
    benchmark,
    condition,
    adapt_nitn,
    adapt_neval,
    prod_nitn,
    prod_neval,
    seed,
):

    np.random.seed(
        seed
    )

    integrator = (
        vegas.Integrator(
            [
                [
                    0.0,
                    1.0,
                ]

                for _
                in range(
                    DIM
                )
            ]
        )
    )

    integrand = (
        make_vegas_integrand(
            benchmark,
            condition,
        )
    )

    adapt_start = (
        time.perf_counter()
    )

    adapt_result = (
        integrator(
            integrand,
            nitn=
                adapt_nitn,
            neval=
                adapt_neval,
        )
    )

    adapt_time = (
        time.perf_counter()
        -
        adapt_start
    )

    prod_start = (
        time.perf_counter()
    )

    result = (
        integrator(
            integrand,
            nitn=
                prod_nitn,
            neval=
                prod_neval,
        )
    )

    prod_time = (
        time.perf_counter()
        -
        prod_start
    )

    estimate = float(
        result.mean
    )

    reported_std = float(
        result.sdev
    )

    return {

        "benchmark":
            benchmark,

        "method":
            (
                f"VEGAS_"
                f"{adapt_nitn}"
                f"x"
                f"{adapt_neval}"
            ),

        "seed":
            seed,

        "adapt_nitn":
            adapt_nitn,

        "adapt_neval":
            adapt_neval,

        "adaptation_evals":
            adapt_nitn
            *
            adapt_neval,

        "production_evals":
            prod_nitn
            *
            prod_neval,

        "estimate":
            estimate,

        "actual_error":
            abs(
                estimate - 1.0
            ),

        "reported_error":
            (
                reported_std
                /
                abs(
                    estimate
                )
                if estimate != 0.0
                else
                float(
                    "inf"
                )
            ),

        "adaptation_Q":
            (
                float(
                    adapt_result.Q
                )
                if adapt_result.Q
                is not None
                else np.nan
            ),

        "production_Q":
            (
                float(
                    result.Q
                )
                if result.Q
                is not None
                else np.nan
            ),

        "adaptation_wall_time":
            adapt_time,

        "production_wall_time":
            prod_time,

        "total_wall_time":
            adapt_time
            +
            prod_time,
    }


# ============================================================
# True vs learned plots
# ============================================================

@torch.no_grad()
def make_true_vs_learned_plots(
    benchmark,
    model,
    condition,
    output_dir,
    n_samples,
    seed,
):

    plot_dir = (
        output_dir
        /
        f"{benchmark}_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cond = (
        torch.tensor(
            condition,
            dtype=torch.float32,
        )
        .reshape(
            1,
            3,
        )
        .expand(
            n_samples,
            -1,
        )
    )

    base.seed_everything(
        seed
    )

    true_samples = (
        sample_target(
            benchmark,
            cond,
            DIM,
        )
        .cpu()
        .numpy()
    )

    base.seed_everything(
        seed + 1
    )

    learned_samples = (
        base.sample_from_model(
            model,
            n_samples,
            cond,
        )
        .cpu()
        .numpy()
    )

    for i, j in (
        DIAGNOSTIC_PAIRS[
            benchmark
        ]
    ):

        # ----------------------------------------------------
        # True
        # ----------------------------------------------------

        plt.figure(
            figsize=(
                6,
                5,
            )
        )

        plt.hexbin(
            true_samples[
                :,
                i
            ],

            true_samples[
                :,
                j
            ],

            gridsize=
                70,

            mincnt=
                1,
        )

        plt.xlim(
            0,
            1,
        )

        plt.ylim(
            0,
            1,
        )

        plt.xlabel(
            f"$x_{i}$"
        )

        plt.ylabel(
            f"$x_{j}$"
        )

        plt.title(
            f"{benchmark}: true"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir
            /
            f"true_x{i}_x{j}.png",

            dpi=
                180,
        )

        plt.close()

        # ----------------------------------------------------
        # Learned
        # ----------------------------------------------------

        plt.figure(
            figsize=(
                6,
                5,
            )
        )

        plt.hexbin(
            learned_samples[
                :,
                i
            ],

            learned_samples[
                :,
                j
            ],

            gridsize=
                70,

            mincnt=
                1,
        )

        plt.xlim(
            0,
            1,
        )

        plt.ylim(
            0,
            1,
        )

        plt.xlabel(
            f"$x_{i}$"
        )

        plt.ylabel(
            f"$x_{j}$"
        )

        plt.title(
            f"{benchmark}: learned copula"
        )

        plt.tight_layout()

        plt.savefig(
            plot_dir
            /
            f"copula_x{i}_x{j}.png",

            dpi=
                180,
        )

        plt.close()


# ============================================================
# NF-vs-VEGAS plot
# ============================================================

def make_error_plot(
    benchmark,
    comparison_df,
    output_dir,
):

    plot_dir = (
        output_dir
        /
        f"{benchmark}_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = comparison_df[
        comparison_df[
            "production_evals"
        ]
        ==
        1_000_000
    ].copy()

    df[
        "error_percent"
    ] = (
        100.0
        *
        df[
            "actual_error_mean"
        ]
    )

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        df[
            "method"
        ],

        df[
            "error_percent"
        ],
    )

    plt.yscale(
        "log"
    )

    plt.ylabel(
        "Mean actual integration error (%)"
    )

    plt.title(
        f"{benchmark}: NF vs VEGAS"
    )

    plt.xticks(
        rotation=
            30,

        ha=
            "right",
    )

    plt.tight_layout()

    plt.savefig(
        plot_dir
        /
        "nf_vs_vegas_error.png",

        dpi=
            180,
    )

    plt.close()


# ============================================================
# Final fixed-condition comparison
# ============================================================

def run_final_comparison(
    benchmark,
    checkpoint,
    args,
    output_dir,
):

    print()
    print(
        "=" * 110
    )

    print(
        f"CPU NF vs VEGAS: "
        f"{benchmark}"
    )

    print(
        f"condition = "
        f"{FIXED_CONDITION}"
    )

    print(
        "true integral = 1"
    )

    print(
        "=" * 110
    )

    model = (
        reconstruct_cpu_model(
            checkpoint
        )
    )

    make_true_vs_learned_plots(
        benchmark=
            benchmark,

        model=
            model,

        condition=
            FIXED_CONDITION,

        output_dir=
            output_dir,

        n_samples=
            args.plot_samples,

        seed=
            args.eval_seed
            +
            9000,
    )

    # ========================================================
    # NF
    # ========================================================

    nf_rows = []

    for n_samples in (
        args.nf_samples
    ):

        for repeat in range(
            args.nf_repeats
        ):

            seed = (
                args.eval_seed
                +
                n_samples
                +
                1_000_000
                *
                repeat
            )

            result = (
                evaluate_nf_cpu(
                    model=
                        model,

                    benchmark=
                        benchmark,

                    condition=
                        FIXED_CONDITION,

                    n_samples=
                        n_samples,

                    seed=
                        seed,

                    chunk_size=
                        args.cpu_chunk_size,
                )
            )

            nf_rows.append(
                result
            )

            print(
                f"NF N={n_samples:,} "
                f"repeat={repeat}: "
                f"I={result['estimate']:.8f} "
                f"error="
                f"{100 * result['actual_error']:.5f}% "
                f"ESS="
                f"{result['ess']:.5f}"
            )

    nf_df = pd.DataFrame(
        nf_rows
    )

    nf_df.to_csv(
        output_dir
        /
        f"{benchmark}_nf_raw.csv",

        index=False,
    )

    nf_summary = (
        nf_df
        .groupby(
            "production_evals",
            as_index=False,
        )
        .agg(

            estimate_mean=(
                "estimate",
                "mean",
            ),

            estimate_std=(
                "estimate",
                "std",
            ),

            actual_error_mean=(
                "actual_error",
                "mean",
            ),

            actual_error_std=(
                "actual_error",
                "std",
            ),

            ess_mean=(
                "ess",
                "mean",
            ),

            ess_std=(
                "ess",
                "std",
            ),

            estimated_error_mean=(
                "estimated_error",
                "mean",
            ),

            production_time_mean=(
                "production_wall_time",
                "mean",
            ),
        )
    )

    nf_summary.to_csv(
        output_dir
        /
        f"{benchmark}_nf_summary.csv",

        index=False,
    )

    # ========================================================
    # VEGAS
    # ========================================================

    vegas_rows = []

    for adapt_nitn in (
        args.vegas_adapt_nitn
    ):

        for repeat in range(
            args.vegas_repeats
        ):

            result = (
                evaluate_vegas(
                    benchmark=
                        benchmark,

                    condition=
                        FIXED_CONDITION,

                    adapt_nitn=
                        adapt_nitn,

                    adapt_neval=
                        args.vegas_adapt_neval,

                    prod_nitn=
                        args.vegas_prod_nitn,

                    prod_neval=
                        args.vegas_prod_neval,

                    seed=
                        (
                            args.eval_seed
                            +
                            20_000_000
                            +
                            100_000
                            *
                            repeat
                            +
                            adapt_nitn
                        ),
                )
            )

            vegas_rows.append(
                result
            )

            print(
                f"{result['method']} "
                f"repeat={repeat}: "
                f"I={result['estimate']:.8f} "
                f"error="
                f"{100 * result['actual_error']:.5f}%"
            )

    vegas_df = pd.DataFrame(
        vegas_rows
    )

    vegas_df.to_csv(
        output_dir
        /
        f"{benchmark}_vegas_raw.csv",

        index=False,
    )

    vegas_summary = (
        vegas_df
        .groupby(
            [
                "method",
                "adapt_nitn",
                "adapt_neval",
                "adaptation_evals",
                "production_evals",
            ],
            as_index=False,
        )
        .agg(

            estimate_mean=(
                "estimate",
                "mean",
            ),

            estimate_std=(
                "estimate",
                "std",
            ),

            actual_error_mean=(
                "actual_error",
                "mean",
            ),

            actual_error_std=(
                "actual_error",
                "std",
            ),

            reported_error_mean=(
                "reported_error",
                "mean",
            ),

            production_time_mean=(
                "production_wall_time",
                "mean",
            ),

            total_time_mean=(
                "total_wall_time",
                "mean",
            ),
        )
    )

    vegas_summary.to_csv(
        output_dir
        /
        f"{benchmark}_vegas_summary.csv",

        index=False,
    )

    # ========================================================
    # Direct comparison
    # ========================================================

    comparison_rows = []

    for row in (
        nf_summary.itertuples()
    ):

        comparison_rows.append(
            {

                "benchmark":
                    benchmark,

                "method":
                    "selected_copula",

                "adaptation_evals":
                    np.nan,

                "production_evals":
                    row.production_evals,

                "estimate_mean":
                    row.estimate_mean,

                "actual_error_mean":
                    row.actual_error_mean,

                "actual_error_std":
                    row.actual_error_std,

                "ess_mean":
                    row.ess_mean,

                "reported_or_estimated_error":
                    row.estimated_error_mean,

                "production_time_mean":
                    row.production_time_mean,
            }
        )

    for row in (
        vegas_summary.itertuples()
    ):

        comparison_rows.append(
            {

                "benchmark":
                    benchmark,

                "method":
                    row.method,

                "adaptation_evals":
                    row.adaptation_evals,

                "production_evals":
                    row.production_evals,

                "estimate_mean":
                    row.estimate_mean,

                "actual_error_mean":
                    row.actual_error_mean,

                "actual_error_std":
                    row.actual_error_std,

                "ess_mean":
                    np.nan,

                "reported_or_estimated_error":
                    row.reported_error_mean,

                "production_time_mean":
                    row.production_time_mean,
            }
        )

    comparison_df = (
        pd.DataFrame(
            comparison_rows
        )
    )

    comparison_df.to_csv(
        output_dir
        /
        f"{benchmark}_comparison.csv",

        index=False,
    )

    make_error_plot(
        benchmark=
            benchmark,

        comparison_df=
            comparison_df,

        output_dir=
            output_dir,
    )

    print()
    print(
        "DIRECT COMPARISON:"
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    return comparison_df


# ============================================================
# One benchmark
# ============================================================

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
        f"UNKNOWN BENCHMARK: "
        f"{benchmark}"
    )

    print(
        benchmark_description(
            benchmark
        )
    )

    print(
        f"Copula LR       = "
        f"{args.copula_lr:.1e}"
    )

    print(
        f"Max steps       = "
        f"{args.max_candidate_steps:,}"
    )

    print(
        "Candidate search = "
        "decision tree, max 3"
    )

    print(
        "Acceptance       = "
        "ESS only"
    )

    print(
        "#" * 120
    )

    # ========================================================
    # 1. Marginals
    # ========================================================

    (
        marginal_states,
        marginal_diagnostics,
    ) = (
        base.train_all_marginal_states_local(
            benchmark,
            args,
            device,
        )
    )

    save_marginal_histories(
        benchmark=
            benchmark,

        diagnostics=
            marginal_diagnostics,

        output_dir=
            output_dir,
    )

    # ========================================================
    # 2. Evidence u caches
    # ========================================================

    cache_by_seed = (
        build_evidence_caches(
            benchmark=
                benchmark,

            marginal_states=
                marginal_states,

            args=
                args,

            device=
                device,
        )
    )

    # ========================================================
    # 3. Fixed KL cache
    # ========================================================

    base.seed_everything(
        args.eval_seed
    )

    kl_cache = (
        base.build_kl_cache(
            benchmark=
                benchmark,

            n=
                args.kl_cache_size,

            device=
                device,
        )
    )

    # ========================================================
    # 4. Full graph
    # ========================================================

    graph = (
        base.full_ar_graph(
            DIM
        )
    )

    previous_graph = None

    model_cache = {}
    evidence_cache = {}

    candidate_cache = {}

    frozen_sigma0 = None

    history = []
    all_trial_rows = []

    (
        current_model,
        current_metrics,
        initial_convergence,
    ) = (
        train_initial_graph(
            benchmark=
                benchmark,

            graph=
                graph,

            marginal_state=
                marginal_states[
                    args.performance_seed
                ],

            kl_cache=
                kl_cache,

            args=
                args,

            device=
                device,
        )
    )

    initial_graph = (
        base.copy_graph(
            graph
        )
    )

    initial_metrics = (
        copy.deepcopy(
            current_metrics
        )
    )

    candidate_cache[
        graph_key(
            graph
        )
    ] = {

        "model_state":
            cpu_state_dict(
                current_model
            ),

        "metrics":
            copy.deepcopy(
                current_metrics
            ),

        "convergence":
            copy.deepcopy(
                initial_convergence
            ),
    }

    # ========================================================
    # 5. Structure-learning rounds
    # ========================================================

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
            f"{edge_count(graph)}"
        )

        print(
            f"ESS   = "
            f"{current_metrics['ess']:.6f}"
        )

        print(
            f"KL    = "
            f"{current_metrics['kl']:.6f}"
        )

        print(
            "#" * 120
        )

        # ====================================================
        # Masked evidence
        # ========================================================

        evidence_df = (
            base.collect_fast_evidence(
                benchmark=
                    benchmark,

                round_id=
                    round_id,

                graph=
                    graph,

                previous_graph=
                    previous_graph,

                cache_by_seed=
                    cache_by_seed,

                model_cache=
                    model_cache,

                evidence_cache=
                    evidence_cache,

                args=
                    args,

                device=
                    device,
            )
        )

        evidence_df.to_csv(
            output_dir
            /
            (
                f"{benchmark}_"
                f"round_{round_id}_"
                f"evidence.csv"
            ),

            index=False,
        )

        if len(
            evidence_df
        ) == 0:

            print(
                "STOP: no active edges."
            )

            break

        # ====================================================
        # Bayesian screening
        # ========================================================

        posterior_df, posterior = (
            base.bayesian_round(
                evidence_df,
                args,
                fixed_sigma0=
                    frozen_sigma0,
            )
        )

        posterior_df.to_csv(
            output_dir
            /
            (
                f"{benchmark}_"
                f"round_{round_id}_"
                f"posterior.csv"
            ),

            index=False,
        )

        if round_id == 0:

            frozen_sigma0 = float(
                posterior[
                    "sigma0_mean"
                ]
            )

            print(
                f"Frozen sigma0 = "
                f"{frozen_sigma0:.8f}"
            )

        # ====================================================
        # Weak edges
        # ========================================================

        weak = []

        for row in (
            posterior_df.itertuples()
        ):

            pip = float(
                row.pip
            )

            if pip < args.pip_threshold:

                weak.append(
                    (

                        int(
                            row.parent
                        ),

                        int(
                            row.child
                        ),

                        pip,

                        float(
                            row.delta_mean
                        ),
                    )
                )

        weak.sort(
            key=lambda item: (
                item[
                    2
                ],

                item[
                    3
                ],
            )
        )

        if len(
            weak
        ) == 0:

            print(
                "STOP: no low-PIP edges."
            )

            break

        print()
        print(
            "WEAK EDGE RANKING"
        )

        for rank, (
            parent,
            child,
            pip,
            delta,
        ) in enumerate(
            weak,
            start=1,
        ):

            print(
                f"{rank:2d}. "
                f"{parent}->{child} "
                f"PIP={pip:.6f} "
                f"delta={delta:.6f}"
            )

        ranked_edges = [
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

        # ====================================================
        # Real copula validation
        # ========================================================

        best, trial_rows = (
            optimized_candidate_search(
                benchmark=
                    benchmark,

                current_graph=
                    graph,

                current_metrics=
                    current_metrics,

                ranked_edges=
                    ranked_edges,

                marginal_state=
                    marginal_states[
                        args.performance_seed
                    ],

                kl_cache=
                    kl_cache,

                candidate_cache=
                    candidate_cache,

                args=
                    args,

                device=
                    device,

                round_id=
                    round_id,
            )
        )

        all_trial_rows.extend(
            trial_rows
        )

        pd.DataFrame(
            all_trial_rows
        ).to_csv(
            output_dir
            /
            f"{benchmark}_candidate_trials.csv",

            index=False,
        )

        if best is None:

            print(
                "STOP: no candidate "
                "improved/preserved ESS."
            )

            break

        # ====================================================
        # Accept
        # ========================================================

        old_graph = (
            base.copy_graph(
                graph
            )
        )

        old_model = (
            current_model
        )

        previous_graph = (
            old_graph
        )

        graph = (
            base.copy_graph(
                best[
                    "graph"
                ]
            )
        )

        current_model = (
            best[
                "model"
            ]
        )

        current_metrics = {

            "ess":
                best[
                    "ess"
                ],

            "kl":
                best[
                    "kl"
                ],

            "variance":
                best[
                    "variance"
                ],

            "relative_error":
                best[
                    "relative_error"
                ],
        }

        history.append(
            {

                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "candidate":
                    best[
                        "name"
                    ],

                "edges_before":
                    edge_count(
                        old_graph
                    ),

                "edges_after":
                    edge_count(
                        graph
                    ),

                "ess_before":
                    (
                        best[
                            "ess"
                        ]
                        -
                        best[
                            "delta_ess"
                        ]
                    ),

                "ess_after":
                    best[
                        "ess"
                    ],

                "delta_ess":
                    best[
                        "delta_ess"
                    ],

                "kl_after":
                    best[
                        "kl"
                    ],

                "training_steps":
                    best[
                        "training_steps"
                    ],

                "converged":
                    int(
                        best[
                            "converged"
                        ]
                    ),

                "graph_before":
                    base.graph_as_string(
                        old_graph
                    ),

                "graph_after":
                    base.graph_as_string(
                        graph
                    ),
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            output_dir
            /
            f"{benchmark}_history.csv",

            index=False,
        )

        del old_model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        f"FINAL GRAPH: {benchmark}"
    )

    print(
        graph
    )

    print(
        f"edges = "
        f"{edge_count(graph)}"
    )

    print(
        f"ESS   = "
        f"{current_metrics['ess']:.6f}"
    )

    print(
        f"KL    = "
        f"{current_metrics['kl']:.6f}"
    )

    print(
        "=" * 120
    )

    # ========================================================
    # Save checkpoint
    # ========================================================

    checkpoint = {

        "benchmark":
            benchmark,

        "dimension":
            DIM,

        "graph":
            copy.deepcopy(
                graph
            ),

        "model_state_dict":
            cpu_state_dict(
                current_model
            ),

        "marginal_state_dict":
            copy.deepcopy(
                marginal_states[
                    args.performance_seed
                ]
            ),

        "performance_seed":
            args.performance_seed,

        "evidence_seeds":
            list(
                args.seeds
            ),

        "copula_lr":
            args.copula_lr,

        "max_candidate_steps":
            args.max_candidate_steps,

        "final_ess":
            current_metrics[
                "ess"
            ],

        "final_kl":
            current_metrics[
                "kl"
            ],

        "initial_ess":
            initial_metrics[
                "ess"
            ],

        "initial_kl":
            initial_metrics[
                "kl"
            ],

        "accepted_rounds":
            len(
                history
            ),
    }

    torch.save(
        checkpoint,

        output_dir
        /
        f"{benchmark}_final_model.pt",
    )

    # ========================================================
    # Final edges
    # ========================================================

    final_edges = []

    for child in range(
        DIM
    ):

        for parent in (
            graph[
                child
            ]
        ):

            final_edges.append(
                {

                    "benchmark":
                        benchmark,

                    "parent":
                        parent,

                    "child":
                        child,
                }
            )

    pd.DataFrame(
        final_edges
    ).to_csv(
        output_dir
        /
        f"{benchmark}_final_edges.csv",

        index=False,
    )

    # ========================================================
    # Summary before comparison
    # ========================================================

    performance_marginal = (
        marginal_diagnostics[
            args.performance_seed
        ]
    )

    summary = {

        "benchmark":
            benchmark,

        "structure_description":
            benchmark_description(
                benchmark
            ),

        "training_batch_size":
            args.batch_size,

        "copula_lr":
            args.copula_lr,

        "max_candidate_steps":
            args.max_candidate_steps,

        "initial_edge_count":
            edge_count(
                initial_graph
            ),

        "final_edge_count":
            edge_count(
                graph
            ),

        "marginal_training_steps":
            performance_marginal[
                "final_steps"
            ],

        "marginal_converged":
            performance_marginal[
                "converged"
            ],

        "initial_training_steps":
            initial_convergence[
                "training_steps"
            ],

        "initial_converged":
            initial_convergence[
                "converged"
            ],

        "initial_ess":
            initial_metrics[
                "ess"
            ],

        "final_ess":
            current_metrics[
                "ess"
            ],

        "delta_ess":
            (
                current_metrics[
                    "ess"
                ]
                -
                initial_metrics[
                    "ess"
                ]
            ),

        "initial_kl":
            initial_metrics[
                "kl"
            ],

        "final_kl":
            current_metrics[
                "kl"
            ],

        "accepted_rounds":
            len(
                history
            ),

        "candidate_cache_size":
            len(
                candidate_cache
            ),

        "final_graph":
            base.graph_as_string(
                graph
            ),
    }

    # ========================================================
    # Release GPU memory before CPU integration comparison
    # ========================================================

    del current_model
    del cache_by_seed
    del kl_cache
    del model_cache
    del evidence_cache
    del candidate_cache
    del marginal_states
    del marginal_diagnostics

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    comparison_df = (
        run_final_comparison(
            benchmark=
                benchmark,

            checkpoint=
                checkpoint,

            args=
                args,

            output_dir=
                output_dir,
        )
    )

    # ========================================================
    # Pull 1M NF result + best VEGAS result into summary
    # ========================================================

    nf_1m = comparison_df[
        (
            comparison_df[
                "method"
            ]
            ==
            "selected_copula"
        )
        &
        (
            comparison_df[
                "production_evals"
            ]
            ==
            1_000_000
        )
    ]

    vegas_1m = comparison_df[
        (
            comparison_df[
                "method"
            ]
            !=
            "selected_copula"
        )
        &
        (
            comparison_df[
                "production_evals"
            ]
            ==
            1_000_000
        )
    ]

    if len(
        nf_1m
    ):

        row = (
            nf_1m.iloc[
                0
            ]
        )

        summary[
            "nf_1m_actual_error"
        ] = float(
            row[
                "actual_error_mean"
            ]
        )

        summary[
            "nf_1m_ess"
        ] = float(
            row[
                "ess_mean"
            ]
        )

    if len(
        vegas_1m
    ):

        best_vegas = (
            vegas_1m
            .sort_values(
                "actual_error_mean"
            )
            .iloc[
                0
            ]
        )

        summary[
            "best_vegas_1m_method"
        ] = (
            best_vegas[
                "method"
            ]
        )

        summary[
            "best_vegas_1m_actual_error"
        ] = float(
            best_vegas[
                "actual_error_mean"
            ]
        )

    return (
        summary,
        final_edges,
        comparison_df,
    )


# ============================================================
# Main driver
# ============================================================

def run(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "OPTIMIZED UNKNOWN BENCHMARK EXPERIMENT"
    )

    print(
        "=" * 120
    )

    print(
        f"device           = "
        f"{device}"
    )

    print(
        f"batch size       = "
        f"{args.batch_size:,}"
    )

    print(
        f"marginal LR      = "
        f"{MARGINAL_LR:.1e}"
    )

    print(
        f"copula LR        = "
        f"{args.copula_lr:.1e}"
    )

    print(
        f"copula max steps = "
        f"{args.max_candidate_steps:,}"
    )

    print(
        f"benchmarks       = "
        f"{args.benchmarks}"
    )

    print(
        "search           = "
        "all weak -> first half -> second half if needed"
    )

    print(
        "acceptance       = ESS only"
    )

    print(
        "=" * 120
    )

    summaries = []

    all_edges = []

    comparisons = []

    for benchmark in (
        args.benchmarks
    ):

        if benchmark not in (
            NEW_BENCHMARKS
        ):

            raise ValueError(
                f"Unknown benchmark "
                f"{benchmark}. "
                f"Available: "
                f"{sorted(NEW_BENCHMARKS)}"
            )

        (
            summary,
            final_edges,
            comparison_df,
        ) = (
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
        )

        summaries.append(
            summary
        )

        all_edges.extend(
            final_edges
        )

        comparisons.append(
            comparison_df
        )

        pd.DataFrame(
            summaries
        ).to_csv(
            output_dir
            /
            "summary.csv",

            index=False,
        )

        pd.DataFrame(
            all_edges
        ).to_csv(
            output_dir
            /
            "final_graph_edges.csv",

            index=False,
        )

    if comparisons:

        pd.concat(
            comparisons,
            ignore_index=True,
        ).to_csv(
            output_dir
            /
            "combined_nf_vs_vegas.csv",

            index=False,
        )

    print()
    print(
        "=" * 130
    )

    print(
        "ALL UNKNOWN BENCHMARKS COMPLETE"
    )

    print(
        "=" * 130
    )

    print()

    print(
        pd.DataFrame(
            summaries
        ).to_string(
            index=False
        )
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser()
    )

    # --------------------------------------------------------
    # Benchmarks
    # --------------------------------------------------------

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=
            DEFAULT_BENCHMARKS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=
            base.DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--performance-seed",
        type=int,
        default=
            base.PERFORMANCE_SEED,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            TRAIN_BATCH_SIZE,
    )

    parser.add_argument(
        "--copula-lr",
        type=float,
        default=
            COPULA_LR,
    )

    # --------------------------------------------------------
    # Marginals
    # --------------------------------------------------------

    parser.add_argument(
        "--min-marginal-steps",
        type=int,
        default=
            base.MIN_MARGINAL_STEPS,
    )

    parser.add_argument(
        "--max-marginal-steps",
        type=int,
        default=
            base.MAX_MARGINAL_STEPS,
    )

    parser.add_argument(
        "--marginal-check-interval",
        type=int,
        default=
            base.MARGINAL_CHECK_INTERVAL,
    )

    parser.add_argument(
        "--marginal-rel-tol",
        type=float,
        default=
            base.MARGINAL_REL_TOL,
    )

    parser.add_argument(
        "--marginal-convergence-patience",
        type=int,
        default=
            base.MARGINAL_CONVERGENCE_PATIENCE,
    )

    parser.add_argument(
        "--marginal-val-size",
        type=int,
        default=
            base.MARGINAL_VAL_SIZE,
    )

    # --------------------------------------------------------
    # Masked evidence
    # --------------------------------------------------------

    parser.add_argument(
        "--masked-steps",
        type=int,
        default=
            base.MASKED_STEPS,
    )

    parser.add_argument(
        "--refine-steps",
        type=int,
        default=
            base.REFINE_STEPS,
    )

    parser.add_argument(
        "--cache-size",
        type=int,
        default=
            base.CACHE_SIZE,
    )

    parser.add_argument(
        "--delta-eval-size",
        type=int,
        default=
            base.DELTA_EVAL_SIZE,
    )

    parser.add_argument(
        "--keep-prob",
        type=float,
        default=
            base.KEEP_PROB,
    )

    # --------------------------------------------------------
    # Copula convergence
    # --------------------------------------------------------

    parser.add_argument(
        "--min-candidate-steps",
        type=int,
        default=
            MIN_CANDIDATE_STEPS,
    )

    parser.add_argument(
        "--max-candidate-steps",
        type=int,
        default=
            MAX_CANDIDATE_STEPS,
    )

    parser.add_argument(
        "--kl-check-interval",
        type=int,
        default=
            KL_CHECK_INTERVAL,
    )

    parser.add_argument(
        "--rel-kl-tol",
        type=float,
        default=
            REL_KL_TOL,
    )

    parser.add_argument(
        "--convergence-patience",
        type=int,
        default=
            CONVERGENCE_PATIENCE,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    parser.add_argument(
        "--kl-cache-size",
        type=int,
        default=
            base.KL_CACHE_SIZE,
    )

    parser.add_argument(
        "--final-ess",
        type=int,
        default=
            base.FINAL_ESS,
    )

    parser.add_argument(
        "--final-kl",
        type=int,
        default=
            base.FINAL_KL,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=
            base.EVAL_SEED,
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=
            base.MAX_ROUNDS,
    )

    parser.add_argument(
        "--pip-threshold",
        type=float,
        default=
            base.PIP_REMOVE_THRESHOLD,
    )

    parser.add_argument(
        "--ess-tol",
        type=float,
        default=
            0.0,
    )

    # --------------------------------------------------------
    # Bayesian
    # --------------------------------------------------------

    parser.add_argument(
        "--mcmc-iter",
        type=int,
        default=
            base.MCMC_ITER,
    )

    parser.add_argument(
        "--mcmc-burn",
        type=int,
        default=
            base.MCMC_BURN,
    )

    parser.add_argument(
        "--mcmc-thin",
        type=int,
        default=
            base.MCMC_THIN,
    )

    parser.add_argument(
        "--rho-a",
        type=float,
        default=
            base.RHO_A,
    )

    parser.add_argument(
        "--rho-b",
        type=float,
        default=
            base.RHO_B,
    )

    parser.add_argument(
        "--bayes-seed",
        type=int,
        default=
            base.BAYES_RANDOM_SEED,
    )

    # --------------------------------------------------------
    # NF production
    # --------------------------------------------------------

    parser.add_argument(
        "--nf-samples",
        nargs="+",
        type=int,
        default=
            NF_SAMPLE_SIZES,
    )

    parser.add_argument(
        "--nf-repeats",
        type=int,
        default=
            NF_REPEATS,
    )

    parser.add_argument(
        "--cpu-chunk-size",
        type=int,
        default=
            CPU_CHUNK_SIZE,
    )

    # --------------------------------------------------------
    # VEGAS
    # --------------------------------------------------------

    parser.add_argument(
        "--vegas-adapt-nitn",
        nargs="+",
        type=int,
        default=
            VEGAS_ADAPT_NITN,
    )

    parser.add_argument(
        "--vegas-adapt-neval",
        type=int,
        default=
            VEGAS_ADAPT_NEVAL,
    )

    parser.add_argument(
        "--vegas-prod-nitn",
        type=int,
        default=
            VEGAS_PROD_NITN,
    )

    parser.add_argument(
        "--vegas-prod-neval",
        type=int,
        default=
            VEGAS_PROD_NEVAL,
    )

    parser.add_argument(
        "--vegas-repeats",
        type=int,
        default=
            VEGAS_REPEATS,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    parser.add_argument(
        "--plot-samples",
        type=int,
        default=
            50_000,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default=
            DEFAULT_OUTPUT,
    )

    args = (
        parser.parse_args()
    )

    if (
        args.performance_seed
        not in
        args.seeds
    ):

        raise ValueError(
            "--performance-seed must "
            "also appear in --seeds"
        )

    return args


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    run(
        parse_args()
    )