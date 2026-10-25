import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# Existing project imports
# ============================================================

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
    graph_as_string,
    active_mask_vector,
    sample_training_mask,
    LR_MASKED,
    GRAD_CLIP,
)

from run_performance_guarded_structure_benchmark import (
    sample_target as existing_sample_target,
    target_logpdf as existing_target_logpdf,
    true_graph_description as existing_true_graph_description,
)


# ============================================================
# Global defaults
# ============================================================

DEFAULT_OUTPUT = "results/ess_only_2k_train_100k_eval"

# ============================================================
# Training / evaluation sizes
# ============================================================

TRAIN_BATCH_SIZE = 10_000

EVAL_SIZE = 100_000

MARGINAL_VAL_SIZE = EVAL_SIZE
DELTA_EVAL_SIZE = EVAL_SIZE
KL_CACHE_SIZE = EVAL_SIZE
FINAL_ESS = EVAL_SIZE
FINAL_KL = EVAL_SIZE


# ============================================================
# Masked evidence
# ============================================================

MASKED_STEPS = 1000
REFINE_STEPS = 250

CACHE_SIZE = 300_000


# ============================================================
# Marginal convergence
# ============================================================

MIN_MARGINAL_STEPS = 1200
MAX_MARGINAL_STEPS = 6000

MARGINAL_CHECK_INTERVAL = 300

MARGINAL_REL_TOL = 0.01
MARGINAL_CONVERGENCE_PATIENCE = 2


# ============================================================
# Copula convergence
# ============================================================

MIN_CANDIDATE_STEPS = 1800
MAX_CANDIDATE_STEPS = 12_000

KL_CHECK_INTERVAL = 600

REL_KL_TOL = 0.01
CONVERGENCE_PATIENCE = 2


# ============================================================
# Search
# ============================================================

MAX_ROUNDS = 10
MAX_CANDIDATE_TRIALS = 5


# ============================================================
# Performance gate
# ============================================================

ESS_TOL = 0.0

PERFORMANCE_SEED = 0
EVAL_SEED = 123456


# ============================================================
# New holdout benchmarks
# ============================================================

NEW_BENCHMARKS = {
    "branching_tree",
    "skip_lag",
    "diamond_sparse",
    "mixed_sparse_dense",
}


# ============================================================
# Conditions
# ============================================================

def sample_conditions(
    n,
    device,
):
    return torch.rand(
        n,
        3,
        device=device,
    )


# ============================================================
# Standard normal helpers
# ============================================================

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


def standard_normal_icdf(
    p,
):
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

    cdf_a = standard_normal_cdf(
        a
    )

    cdf_b = standard_normal_cdf(
        b
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

    z = standard_normal_icdf(
        p
    )

    x = (
        mu
        +
        sigma
        *
        z
    )

    return x.clamp(
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
# Holdout benchmark conditional laws
# ============================================================

def new_benchmark_mu_sigma(
    benchmark,
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

    # ========================================================
    # Branching tree
    # ========================================================

    if benchmark == "branching_tree":

        parent_map = {
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 2,
            6: 2,
            7: 6,
        }

        parent = (
            parent_map[
                child
            ]
        )

        p = x[
            :,
            parent:parent + 1
        ]

        mu = (
            0.50
            +
            0.27
            *
            torch.sin(
                2.0
                *
                math.pi
                *
                (
                    p
                    +
                    0.10
                    *
                    c0
                    +
                    0.037
                    *
                    child
                )
            )
        )

        sigma = (
            0.045
            +
            0.010
            *
            c2
        )

        return (
            mu,
            sigma,
        )

    # ========================================================
    # Skip lag
    # ========================================================

    if benchmark == "skip_lag":

        parent = (
            child - 2
        )

        p = x[
            :,
            parent:parent + 1
        ]

        mu = (
            0.50
            +
            0.28
            *
            torch.cos(
                2.0
                *
                math.pi
                *
                (
                    p
                    +
                    0.11
                    *
                    c1
                    +
                    0.041
                    *
                    child
                )
            )
        )

        sigma = (
            0.042
            +
            0.010
            *
            c2
        )

        return (
            mu,
            sigma,
        )

    # ========================================================
    # Diamond sparse
    # ========================================================

    if benchmark == "diamond_sparse":

        parent_map = {
            2: (0, 1),
            3: (0, 1),
            4: (2, 3),
            5: (2, 3),
            6: (4, 5),
            7: (4, 5),
        }

        p1, p2 = (
            parent_map[
                child
            ]
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
            0.65
            *
            (
                a - 0.5
            )
            +
            0.45
            *
            (
                b - 0.5
            )
            +
            0.20
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
            +
            0.15
            *
            (
                c0 - 0.5
            )
        )

        mu = (
            0.50
            +
            0.31
            *
            torch.tanh(
                2.2
                *
                signal
            )
        )

        sigma = (
            0.045
            +
            0.008
            *
            c2
        )

        return (
            mu,
            sigma,
        )

    # ========================================================
    # Mixed sparse / dense
    # ========================================================

    if benchmark == "mixed_sparse_dense":

        if child <= 3:

            parent = (
                child - 1
            )

            p = x[
                :,
                parent:parent + 1
            ]

            mu = (
                0.50
                +
                0.27
                *
                torch.sin(
                    2.0
                    *
                    math.pi
                    *
                    (
                        p
                        +
                        0.08
                        *
                        c0
                        +
                        0.025
                        *
                        child
                    )
                )
            )

        else:

            parents = x[
                :,
                :child
            ]

            index = torch.arange(
                1,
                child + 1,
                device=x.device,
                dtype=x.dtype,
            ).reshape(
                1,
                -1,
            )

            weights = (
                torch.sin(
                    0.67
                    *
                    index
                    *
                    (
                        child + 1
                    )
                )
                +
                1.35
            )

            weights = (
                weights
                /
                torch.sqrt(
                    weights
                    .square()
                    .sum()
                )
            )

            signal = (
                (
                    (
                        parents
                        -
                        0.5
                    )
                    *
                    weights
                )
                .sum(
                    dim=1,
                    keepdim=True,
                )
                +
                0.35
                *
                (
                    c0 - 0.5
                )
                +
                0.15
                *
                torch.sin(
                    2.0
                    *
                    math.pi
                    *
                    c1
                )
            )

            mu = (
                0.50
                +
                0.30
                *
                torch.tanh(
                    signal
                )
            )

        sigma = (
            0.043
            +
            0.009
            *
            c2
        )

        return (
            mu,
            sigma,
        )

    raise ValueError(
        f"Unknown benchmark: "
        f"{benchmark}"
    )


# ============================================================
# New target sampler
# ============================================================

def sample_new_target(
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

    if benchmark in {
        "skip_lag",
        "diamond_sparse",
    }:

        x[
            :,
            0:2
        ] = torch.rand(
            n,
            2,
            device=cond.device,
            dtype=cond.dtype,
        )

        start_child = 2

    else:

        x[
            :,
            0:1
        ] = torch.rand(
            n,
            1,
            device=cond.device,
            dtype=cond.dtype,
        )

        start_child = 1

    for child in range(
        start_child,
        dim,
    ):

        mu, sigma = (
            new_benchmark_mu_sigma(
                benchmark,
                child,
                x,
                cond,
            )
        )

        x[
            :,
            child:child + 1
        ] = sample_truncated_normal(
            mu,
            sigma,
        )

    return x


# ============================================================
# New target logpdf
# ============================================================

def new_target_logpdf(
    benchmark,
    x,
    cond,
):
    logp = torch.zeros(
        x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )

    if benchmark in {
        "skip_lag",
        "diamond_sparse",
    }:

        start_child = 2

    else:

        start_child = 1

    for child in range(
        start_child,
        DIM,
    ):

        mu, sigma = (
            new_benchmark_mu_sigma(
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
# Unified target interface
# ============================================================

def sample_target(
    benchmark,
    cond,
    dim,
):
    if benchmark in NEW_BENCHMARKS:

        return sample_new_target(
            benchmark,
            cond,
            dim,
        )

    return existing_sample_target(
        benchmark,
        cond,
        dim,
    )


def target_logpdf(
    benchmark,
    x,
    cond,
):
    if benchmark in NEW_BENCHMARKS:

        return new_target_logpdf(
            benchmark,
            x,
            cond,
        )

    return existing_target_logpdf(
        benchmark,
        x,
        cond,
    )


def true_graph_description(
    benchmark,
):
    descriptions = {
        "branching_tree":
            (
                "HOLDOUT: branching tree; "
                "7 generating edges"
            ),

        "skip_lag":
            (
                "HOLDOUT: lag-2 structure; "
                "6 generating edges"
            ),

        "diamond_sparse":
            (
                "HOLDOUT: layered diamond; "
                "12 generating edges"
            ),

        "mixed_sparse_dense":
            (
                "HOLDOUT: sparse early "
                "+ dense later"
            ),
    }

    if benchmark in descriptions:

        return descriptions[
            benchmark
        ]

    return (
        existing_true_graph_description(
            benchmark
        )
    )


# ============================================================
# Helpers
# ============================================================

def cpu_state_dict(
    model,
):
    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in model.state_dict().items()
    }


def remove_edges(
    graph,
    removals,
):
    new_graph = copy_graph(
        graph
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


# ============================================================
# Fixed marginal validation NLL
# ============================================================

@torch.no_grad()
def cached_marginal_nll(
    bank,
    x_cache,
    cond_cache,
    chunk_size=10_000,
):
    bank.eval()

    total_nll = 0.0
    count = 0

    n = (
        x_cache.shape[0]
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            n,
        )

        x = x_cache[
            start:end
        ]

        cond = cond_cache[
            start:end
        ]

        batch_logq = torch.zeros(
            end - start,
            device=x.device,
            dtype=x.dtype,
        )

        for j in range(DIM):

            _, log_jac = (
                bank.marginals[j]
                .inverse(
                    x[
                        :,
                        j:j + 1
                    ],
                    cond,
                )
            )

            batch_logq += (
                -log_jac.reshape(-1)
            )

        total_nll += (
            -batch_logq.sum().item()
        )

        count += (
            end - start
        )

    return (
        total_nll
        /
        count
    )


# ============================================================
# Marginal training until relative improvement plateaus
# ============================================================

def train_marginals_until_converged(
    benchmark,
    bank,
    args,
    device,
    seed,
):
    # ========================================================
    # Fixed 100k validation cache
    # ========================================================

    seed_everything(
        args.eval_seed
        +
        1000
        +
        seed
    )

    with torch.no_grad():

        cond_val = sample_conditions(
            args.marginal_val_size,
            device,
        )

        x_val = sample_target(
            benchmark,
            cond_val,
            DIM,
        )

    # ========================================================
    # Training RNG
    # ========================================================

    seed_everything(
        seed
    )

    optimizer = torch.optim.Adam(
        bank.parameters(),
        lr=1e-4,
    )

    bank.train()

    history = []

    total_steps = 0
    previous_val_nll = None
    convergence_count = 0

    while (
        total_steps
        <
        args.max_marginal_steps
    ):

        block_steps = min(
            args.marginal_check_interval,
            (
                args.max_marginal_steps
                -
                total_steps
            ),
        )

        recent_losses = []

        # ====================================================
        # 2k training batch
        # ====================================================

        for _ in range(
            block_steps
        ):

            cond = sample_conditions(
                args.batch_size,
                device,
            )

            with torch.no_grad():

                x = sample_target(
                    benchmark,
                    cond,
                    DIM,
                )

            total_logq = torch.zeros(
                x.shape[0],
                device=device,
                dtype=x.dtype,
            )

            for j in range(DIM):

                _, log_jac = (
                    bank.marginals[j]
                    .inverse(
                        x[
                            :,
                            j:j + 1
                        ],
                        cond,
                    )
                )

                total_logq += (
                    -log_jac.reshape(-1)
                )

            loss = (
                -total_logq.mean()
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                bank.parameters(),
                10.0,
            )

            optimizer.step()

            recent_losses.append(
                loss.item()
            )

        total_steps += (
            block_steps
        )

        # ====================================================
        # Fixed 100k validation loss
        # ====================================================

        val_nll = cached_marginal_nll(
            bank=
                bank,

            x_cache=
                x_val,

            cond_cache=
                cond_val,
        )

        train_nll = float(
            np.mean(
                recent_losses[-100:]
            )
        )

        if previous_val_nll is None:

            absolute_improvement = float(
                "nan"
            )

            relative_improvement = float(
                "nan"
            )

        else:

            absolute_improvement = (
                previous_val_nll
                -
                val_nll
            )

            relative_improvement = (
                absolute_improvement
                /
                max(
                    abs(
                        previous_val_nll
                    ),
                    1e-8,
                )
            )

        history.append(
            {
                "step":
                    total_steps,

                "train_nll":
                    train_nll,

                "val_nll":
                    val_nll,

                "absolute_improvement":
                    absolute_improvement,

                "relative_improvement":
                    relative_improvement,
            }
        )

        if previous_val_nll is None:

            print(
                f"    marginal "
                f"step {total_steps:5d} "
                f"train NLL="
                f"{train_nll:.6f} "
                f"val NLL="
                f"{val_nll:.6f}"
            )

        else:

            print(
                f"    marginal "
                f"step {total_steps:5d} "
                f"train NLL="
                f"{train_nll:.6f} "
                f"val NLL="
                f"{val_nll:.6f} "
                f"dNLL="
                f"{absolute_improvement:+.6f} "
                f"rel="
                f"{100.0 * relative_improvement:+.3f}%"
            )

        # ====================================================
        # Same relative convergence rule as before
        # ====================================================

        if (
            previous_val_nll is not None
            and
            total_steps
            >=
            args.min_marginal_steps
        ):

            if (
                relative_improvement
                <
                args.marginal_rel_tol
            ):

                convergence_count += 1

            else:

                convergence_count = 0

            print(
                f"      marginal convergence "
                f"{convergence_count}/"
                f"{args.marginal_convergence_patience}"
            )

            if (
                convergence_count
                >=
                args.marginal_convergence_patience
            ):

                print(
                    f"      MARGINAL CONVERGED "
                    f"at "
                    f"{total_steps} steps"
                )

                break

        previous_val_nll = (
            val_nll
        )

        bank.train()

    converged = (
        convergence_count
        >=
        args.marginal_convergence_patience
    )

    if (
        not converged
        and
        total_steps
        >=
        args.max_marginal_steps
    ):

        print(
            f"      marginal reached max steps "
            f"{args.max_marginal_steps}"
        )

    return {
        "history":
            history,

        "final_steps":
            total_steps,

        "final_val_nll":
            history[-1][
                "val_nll"
            ],

        "converged":
            converged,
    }


# ============================================================
# Train marginal states
# ============================================================

def train_all_marginal_states_local(
    benchmark,
    args,
    device,
):
    states = {}
    diagnostics = {}

    for seed in args.seeds:

        print()
        print(
            "=" * 75
        )

        print(
            f"Training marginals "
            f"seed={seed}"
        )

        print(
            "=" * 75
        )

        seed_everything(
            seed
        )

        bank = (
            MarginalBank(
                DIM
            )
            .to(device)
        )

        result = (
            train_marginals_until_converged(
                benchmark=
                    benchmark,

                bank=
                    bank,

                args=
                    args,

                device=
                    device,

                seed=
                    seed,
            )
        )

        print()
        print(
            f"Marginal seed={seed}: "
            f"steps="
            f"{result['final_steps']} "
            f"val NLL="
            f"{result['final_val_nll']:.6f} "
            f"converged="
            f"{result['converged']}"
        )

        states[
            seed
        ] = cpu_state_dict(
            bank
        )

        diagnostics[
            seed
        ] = result

        del bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return (
        states,
        diagnostics,
    )


# ============================================================
# Marginal x -> u
# ============================================================

@torch.no_grad()
def marginal_x_to_u(
    bank,
    x,
    cond,
):
    pieces = []

    for j in range(DIM):

        uj, _ = (
            bank.marginals[j]
            .inverse(
                x[
                    :,
                    j:j + 1
                ],
                cond,
            )
        )

        uj = uj.clamp(
            0.0,
            1.0,
        )

        pieces.append(
            uj
        )

    return torch.cat(
        pieces,
        dim=1,
    )


# ============================================================
# Evidence cache
# ============================================================

@torch.no_grad()
def build_u_cache(
    benchmark,
    bank,
    n,
    device,
    chunk_size=20_000,
):
    u_parts = []
    c_parts = []

    generated = 0

    while generated < n:

        m = min(
            chunk_size,
            n - generated,
        )

        cond = sample_conditions(
            m,
            device,
        )

        x = sample_target(
            benchmark,
            cond,
            DIM,
        )

        u = marginal_x_to_u(
            bank,
            x,
            cond,
        )

        u_parts.append(
            u.detach()
        )

        c_parts.append(
            cond.detach()
        )

        generated += m

    return (
        torch.cat(
            u_parts,
            dim=0,
        ),
        torch.cat(
            c_parts,
            dim=0,
        ),
    )


# ============================================================
# Fixed 100k target KL cache
# ============================================================

@torch.no_grad()
def build_kl_cache(
    benchmark,
    n,
    device,
):
    cond = sample_conditions(
        n,
        device,
    )

    x = sample_target(
        benchmark,
        cond,
        DIM,
    )

    logp = target_logpdf(
        benchmark,
        x,
        cond,
    ).reshape(-1)

    return {
        "x":
            x.detach(),

        "cond":
            cond.detach(),

        "logp":
            logp.detach(),
    }


# ============================================================
# Model proposal sampling
# ============================================================

@torch.no_grad()
def sample_from_model(
    model,
    n,
    cond,
):
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

            out = attempt()

            if isinstance(
                out,
                tuple,
            ):

                x = out[0]

            else:

                x = out

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

                return x

        except Exception as exc:

            errors.append(
                repr(exc)
            )

    raise RuntimeError(
        "Could not determine "
        "model.sample() interface:\n"
        +
        "\n".join(
            errors
        )
    )


# ============================================================
# Generic evaluation
# ============================================================

@torch.no_grad()
def evaluate_generic(
    model,
    benchmark,
    device,
    n_eval,
    n_kl,
    eval_seed,
    chunk_size=20_000,
):
    model.eval()

    # ========================================================
    # Forward KL diagnostic
    # ========================================================

    seed_everything(
        eval_seed
    )

    kl_sum = 0.0
    kl_count = 0

    generated = 0

    while generated < n_kl:

        m = min(
            chunk_size,
            n_kl - generated,
        )

        cond = sample_conditions(
            m,
            device,
        )

        x = sample_target(
            benchmark,
            cond,
            DIM,
        )

        logp = target_logpdf(
            benchmark,
            x,
            cond,
        ).reshape(-1)

        logq = model.log_prob(
            x,
            cond,
        ).reshape(-1)

        kl_sum += (
            (
                logp
                -
                logq
            )
            .sum()
            .item()
        )

        kl_count += m

        generated += m

    forward_kl = (
        kl_sum
        /
        kl_count
    )

    # ========================================================
    # ESS
    # ========================================================

    seed_everything(
        eval_seed
    )

    logw_parts = []

    generated = 0

    while generated < n_eval:

        m = min(
            chunk_size,
            n_eval - generated,
        )

        cond = sample_conditions(
            m,
            device,
        )

        x = sample_from_model(
            model,
            m,
            cond,
        )

        logq = model.log_prob(
            x,
            cond,
        ).reshape(-1)

        logp = target_logpdf(
            benchmark,
            x,
            cond,
        ).reshape(-1)

        logw_parts.append(
            (
                logp
                -
                logq
            )
            .detach()
        )

        generated += m

    logw = torch.cat(
        logw_parts,
        dim=0,
    )

    max_logw = (
        logw.max()
    )

    w = torch.exp(
        logw
        -
        max_logw
    )

    n_total = (
        w.numel()
    )

    sum_w = (
        w.sum()
    )

    sum_w2 = (
        w.square()
        .sum()
    )

    ess = (
        sum_w.square()
        /
        (
            n_total
            *
            sum_w2
        )
    )

    ess = float(
        ess.item()
    )

    mean_w = (
        w.mean()
    )

    normalized_w = (
        w
        /
        mean_w
    )

    relative_variance = float(
        normalized_w.var(
            unbiased=False
        ).item()
    )

    relative_error = math.sqrt(
        relative_variance
        /
        n_total
    )

    return {
        "ess":
            ess,

        "forward_kl":
            forward_kl,

        "variance":
            relative_variance,

        "relative_error":
            relative_error,
    }


# ============================================================
# Masked child training
# ============================================================

def train_masked_child_cached(
    child,
    active_parents,
    u_cache,
    cond_cache,
    previous_state,
    steps,
    batch_size,
    keep_prob,
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
        .to(device)
    )

    if previous_state is not None:

        model.load_state_dict(
            previous_state
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_MASKED,
    )

    model.train()

    n_cache = (
        u_cache.shape[0]
    )

    recent = []

    for _ in range(
        steps
    ):

        idx = torch.randint(
            0,
            n_cache,
            (
                batch_size,
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
                batch_size,

            child=
                child,

            active_parents=
                active_parents,

            keep_prob=
                keep_prob,

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

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        recent.append(
            loss.item()
        )

    return (
        model,
        float(
            np.mean(
                recent[-100:]
            )
        ),
    )


# ============================================================
# Edge deltas
# ============================================================

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
    if len(
        active_parents
    ) == 0:

        return []

    sums = {
        parent: 0.0
        for parent
        in active_parents
    }

    sums2 = {
        parent: 0.0
        for parent
        in active_parents
    }

    count = 0

    base_vector = active_mask_vector(
        child,
        active_parents,
        device,
    )

    n = (
        u.shape[0]
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            n,
        )

        ub = u[
            start:end
        ]

        cb = cond[
            start:end
        ]

        batch = (
            ub.shape[0]
        )

        base_mask = (
            base_vector
            .unsqueeze(0)
            .expand(
                batch,
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

        for parent in active_parents:

            mask_off = (
                base_mask.clone()
            )

            mask_off[
                :,
                parent
            ] = 0.0

            log_off = model.log_prob(
                child_u,
                ub,
                cb,
                mask_off,
            )

            delta = (
                log_on
                -
                log_off
            )

            sums[
                parent
            ] += (
                delta.sum()
                .item()
            )

            sums2[
                parent
            ] += (
                delta.square()
                .sum()
                .item()
            )

        count += (
            batch
        )

    rows = []

    for parent in active_parents:

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
            mean * mean,
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

                "is_markov1_edge":
                    int(
                        parent
                        ==
                        child - 1
                    ),
            }
        )

    return rows


# ============================================================
# Evidence collection
# ============================================================

def collect_fast_evidence(
    benchmark,
    round_id,
    graph,
    previous_graph,
    cache_by_seed,
    model_cache,
    evidence_cache,
    args,
    device,
):
    rows = []

    trained = 0
    reused = 0

    for seed in args.seeds:

        print()
        print(
            f"Evidence seed {seed}"
        )

        cache = (
            cache_by_seed[
                seed
            ]
        )

        for child in range(
            1,
            DIM,
        ):

            parents = list(
                graph[
                    child
                ]
            )

            if len(
                parents
            ) == 0:

                continue

            key = (
                seed,
                child,
            )

            if previous_graph is None:

                changed = True

            else:

                changed = (
                    list(
                        previous_graph[
                            child
                        ]
                    )
                    !=
                    parents
                )

            if (
                not changed
                and
                key in evidence_cache
            ):

                print(
                    f"  child {child}: "
                    f"{parents} "
                    "[REUSE]"
                )

                for cached_row in (
                    evidence_cache[
                        key
                    ]
                ):

                    row = dict(
                        cached_row
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
                                "reused",
                        }
                    )

                    rows.append(
                        row
                    )

                reused += 1

                continue

            previous_state = (
                model_cache.get(
                    key,
                    None,
                )
            )

            if previous_state is None:

                steps = (
                    args.masked_steps
                )

                mode = "FULL"

            else:

                steps = (
                    args.refine_steps
                )

                mode = "REFINE"

            print(
                f"  child {child}: "
                f"{parents} "
                f"[{mode} {steps}]"
            )

            model, loss = (
                train_masked_child_cached(
                    child=
                        child,

                    active_parents=
                        parents,

                    u_cache=
                        cache[
                            "u_train"
                        ],

                    cond_cache=
                        cache[
                            "c_train"
                        ],

                    previous_state=
                        previous_state,

                    steps=
                        steps,

                    batch_size=
                        args.batch_size,

                    keep_prob=
                        args.keep_prob,

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
                f"    NLL ~ "
                f"{loss:.6f}"
            )

            model.eval()

            child_rows = (
                compute_deltas_cached(
                    model=
                        model,

                    child=
                        child,

                    active_parents=
                        parents,

                    u=
                        cache[
                            "u_eval"
                        ],

                    cond=
                        cache[
                            "c_eval"
                        ],

                    device=
                        device,
                )
            )

            model_cache[
                key
            ] = cpu_state_dict(
                model
            )

            evidence_cache[
                key
            ] = [
                dict(row)
                for row
                in child_rows
            ]

            for child_row in child_rows:

                row = dict(
                    child_row
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

            trained += 1

            del model

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    print()
    print(
        "Evidence summary:"
    )

    print(
        f"  trained/refined = "
        f"{trained}"
    )

    print(
        f"  reused          = "
        f"{reused}"
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Real model
# ============================================================

def make_real_model(
    graph,
    marginal_state,
    device,
):
    bank = (
        MarginalBank(
            DIM
        )
        .to(device)
    )

    bank.load_state_dict(
        marginal_state
    )

    bank.eval()

    for parameter in (
        bank.parameters()
    ):

        parameter.requires_grad = False

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
        .to(device)
    )

    return model


# ============================================================
# Fixed 100k forward KL diagnostic
# ============================================================

@torch.no_grad()
def cached_forward_kl(
    model,
    kl_cache,
    chunk_size=10_000,
):
    model.eval()

    total = 0.0
    count = 0

    n = (
        kl_cache[
            "x"
        ].shape[0]
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            n,
        )

        x = kl_cache[
            "x"
        ][
            start:end
        ]

        cond = kl_cache[
            "cond"
        ][
            start:end
        ]

        logp = kl_cache[
            "logp"
        ][
            start:end
        ].reshape(-1)

        logq = model.log_prob(
            x,
            cond,
        ).reshape(-1)

        total += (
            (
                logp
                -
                logq
            )
            .sum()
            .item()
        )

        count += (
            end - start
        )

    return (
        total
        /
        count
    )


# ============================================================
# Copula training until relative KL improvement plateaus
# ============================================================

def train_model_until_converged(
    model,
    benchmark,
    kl_cache,
    args,
    device,
):
    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=1e-4,
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
            (
                args.max_candidate_steps
                -
                total_steps
            ),
        )

        recent_losses = []

        # ====================================================
        # 2k training batch
        # ====================================================

        for _ in range(
            block_steps
        ):

            cond = sample_conditions(
                args.batch_size,
                device,
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

        # ====================================================
        # 100k fixed KL
        # ====================================================

        kl = cached_forward_kl(
            model,
            kl_cache,
        )

        mean_loss = float(
            np.mean(
                recent_losses[-100:]
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
            }
        )

        if previous_kl is None:

            print(
                f"    step "
                f"{total_steps:5d} "
                f"NLL="
                f"{mean_loss:.6f} "
                f"KL="
                f"{kl:.6f}"
            )

        else:

            print(
                f"    step "
                f"{total_steps:5d} "
                f"NLL="
                f"{mean_loss:.6f} "
                f"KL="
                f"{kl:.6f} "
                f"dKL="
                f"{absolute_improvement:+.6f} "
                f"rel="
                f"{100.0 * relative_improvement:+.3f}%"
            )

        # ====================================================
        # Previous convergence rule
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
                f"      convergence "
                f"{convergence_count}/"
                f"{args.convergence_patience}"
            )

            if (
                convergence_count
                >=
                args.convergence_patience
            ):

                print(
                    f"      CONVERGED "
                    f"at "
                    f"{total_steps} steps"
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
            f"      reached max steps "
            f"{args.max_candidate_steps}"
        )

    return {
        "history":
            history,

        "final_steps":
            total_steps,

        "final_kl":
            history[-1][
                "kl"
            ],

        "converged":
            converged,
    }


# ============================================================
# Convergence summary
# ============================================================

def summarize_convergence(
    training_result,
):
    history = (
        training_result[
            "history"
        ]
    )

    final = (
        history[-1]
    )

    if len(
        history
    ) >= 2:

        last_relative_improvement = (
            final[
                "relative_improvement"
            ]
        )

        last_dkl = (
            final[
                "absolute_improvement"
            ]
        )

    else:

        last_relative_improvement = float(
            "nan"
        )

        last_dkl = float(
            "nan"
        )

    return {
        "training_steps":
            training_result[
                "final_steps"
            ],

        "converged":
            training_result[
                "converged"
            ],

        "final_training_kl":
            training_result[
                "final_kl"
            ],

        "last_dkl":
            last_dkl,

        "last_relative_improvement":
            last_relative_improvement,
    }


# ============================================================
# ESS evaluation
# ============================================================

def evaluate_ess(
    model,
    benchmark,
    device,
    args,
):
    result = evaluate_generic(
        model=
            model,

        benchmark=
            benchmark,

        device=
            device,

        n_eval=
            args.final_ess,

        n_kl=
            args.final_kl,

        eval_seed=
            args.eval_seed,
    )

    return {
        "ess":
            result[
                "ess"
            ],

        "variance":
            result[
                "variance"
            ],

        "relative_error":
            result[
                "relative_error"
            ],

        "kl":
            result[
                "forward_kl"
            ],
    }


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
    seed_everything(
        args.performance_seed
    )

    model = make_real_model(
        graph,
        marginal_state,
        device,
    )

    print()
    print(
        "Training initial REAL graph "
        "until convergence..."
    )

    training_result = (
        train_model_until_converged(
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
        summarize_convergence(
            training_result
        )
    )

    metrics = evaluate_ess(
        model=
            model,

        benchmark=
            benchmark,

        device=
            device,

        args=
            args,
    )

    metrics[
        "kl"
    ] = cached_forward_kl(
        model,
        kl_cache,
    )

    print()
    print(
        "INITIAL MODEL:"
    )

    print(
        f"  steps     = "
        f"{convergence['training_steps']}"
    )

    print(
        f"  converged = "
        f"{convergence['converged']}"
    )

    print(
        f"  ESS       = "
        f"{metrics['ess']:.6f}"
    )

    print(
        f"  KL diag   = "
        f"{metrics['kl']:.6f}"
    )

    return (
        model,
        metrics,
        convergence,
    )


# ============================================================
# Five candidate subsets
# ============================================================

def canonical_edge_subset(
    edges,
):
    return tuple(
        sorted(
            tuple(edge)
            for edge
            in edges
        )
    )


def make_five_candidate_subsets(
    ranked_edges,
):
    edges = list(
        ranked_edges
    )

    n = len(
        edges
    )

    if n == 0:

        return []

    candidates = []

    # 1. Remove all weak edges.
    candidates.append(
        (
            "all_weak",
            edges,
        )
    )

    # 2-3. Ranking halves.
    if n >= 2:

        midpoint = int(
            math.ceil(
                n / 2
            )
        )

        first_half = edges[
            :midpoint
        ]

        second_half = edges[
            midpoint:
        ]

        if first_half:

            candidates.append(
                (
                    "rank_first_half",
                    first_half,
                )
            )

        if second_half:

            candidates.append(
                (
                    "rank_second_half",
                    second_half,
                )
            )

    # 4-5. Alternating partitions.
    alternating_a = (
        edges[
            ::2
        ]
    )

    alternating_b = (
        edges[
            1::2
        ]
    )

    if alternating_a:

        candidates.append(
            (
                "alternating_A",
                alternating_a,
            )
        )

    if alternating_b:

        candidates.append(
            (
                "alternating_B",
                alternating_b,
            )
        )

    seen = set()
    unique = []

    for name, subset in candidates:

        key = canonical_edge_subset(
            subset
        )

        if not key:

            continue

        if key in seen:

            continue

        seen.add(
            key
        )

        unique.append(
            (
                name,
                list(key),
            )
        )

    return unique[
        :5
    ]


# ============================================================
# Candidate search
#
# ESS is the only gate.
# ============================================================

def search_candidates(
    benchmark,
    current_graph,
    current_model,
    current_metrics,
    ranked_edges,
    marginal_state,
    kl_cache,
    args,
    device,
    round_id,
):
    candidate_sets = (
        make_five_candidate_subsets(
            ranked_edges
        )
    )

    if len(
        candidate_sets
    ) == 0:

        return (
            None,
            [],
        )

    accepted = []
    all_candidates = []
    trial_rows = []

    current_ess = (
        current_metrics[
            "ess"
        ]
    )

    current_kl = (
        cached_forward_kl(
            current_model,
            kl_cache,
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ESS-ONLY STRUCTURE SEARCH"
    )

    print(
        f"Weak edge pool = "
        f"{len(ranked_edges)}"
    )

    print(
        f"Candidates     = "
        f"{len(candidate_sets)}"
    )

    print(
        f"Current ESS    = "
        f"{current_ess:.6f}"
    )

    print(
        "=" * 100
    )

    for trial_index, (
        name,
        removals,
    ) in enumerate(
        candidate_sets,
        start=1,
    ):

        candidate_graph = (
            remove_edges(
                current_graph,
                removals,
            )
        )

        print()
        print(
            "=" * 100
        )

        print(
            f"CANDIDATE "
            f"{trial_index}/"
            f"{len(candidate_sets)}"
        )

        print(
            f"name      = "
            f"{name}"
        )

        print(
            f"removals  = "
            f"{removals}"
        )

        print(
            f"n_removed = "
            f"{len(removals)}"
        )

        print(
            f"edges     = "
            f"{count_edges(current_graph)} "
            f"-> "
            f"{count_edges(candidate_graph)}"
        )

        print(
            "=" * 100
        )

        seed_everything(
            args.performance_seed
        )

        candidate_model = (
            make_real_model(
                candidate_graph,
                marginal_state,
                device,
            )
        )

        training_result = (
            train_model_until_converged(
                model=
                    candidate_model,

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
            summarize_convergence(
                training_result
            )
        )

        candidate_metrics = (
            evaluate_ess(
                model=
                    candidate_model,

                benchmark=
                    benchmark,

                device=
                    device,

                args=
                    args,
            )
        )

        candidate_ess = (
            candidate_metrics[
                "ess"
            ]
        )

        delta_ess = (
            candidate_ess
            -
            current_ess
        )

        candidate_kl = (
            cached_forward_kl(
                candidate_model,
                kl_cache,
            )
        )

        delta_kl = (
            candidate_kl
            -
            current_kl
        )

        # ====================================================
        # ESS ONLY
        # ====================================================

        ess_pass = (
            delta_ess
            >=
            -args.ess_tol
        )

        acceptable = (
            ess_pass
        )

        print()
        print(
            "FINAL PERFORMANCE:"
        )

        print(
            f"  ESS current   = "
            f"{current_ess:.6f}"
        )

        print(
            f"  ESS candidate = "
            f"{candidate_ess:.6f}"
        )

        print(
            f"  delta ESS     = "
            f"{delta_ess:+.6f}"
        )

        print()

        print(
            f"  KL current    = "
            f"{current_kl:.6f}"
        )

        print(
            f"  KL candidate  = "
            f"{candidate_kl:.6f}"
        )

        print(
            f"  delta KL      = "
            f"{delta_kl:+.6f} "
            f"[diagnostic]"
        )

        print()

        print(
            f"  ESS gate      = "
            f"{ess_pass}"
        )

        print(
            f"  DECISION      = "
            f"{'ACCEPTABLE' if acceptable else 'FAILED'}"
        )

        candidate = {
            "name":
                name,

            "removals":
                removals,

            "graph":
                candidate_graph,

            "model":
                candidate_model,

            "ess":
                candidate_ess,

            "delta_ess":
                delta_ess,

            "kl":
                candidate_kl,

            "delta_kl":
                delta_kl,

            "variance":
                candidate_metrics[
                    "variance"
                ],

            "relative_error":
                candidate_metrics[
                    "relative_error"
                ],

            "accept":
                acceptable,

            **convergence,
        }

        all_candidates.append(
            candidate
        )

        trial_rows.append(
            {
                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "trial":
                    trial_index,

                "candidate":
                    name,

                "n_removed":
                    len(
                        removals
                    ),

                "removed_edges":
                    str(
                        removals
                    ),

                "edges_before":
                    count_edges(
                        current_graph
                    ),

                "edges_after":
                    count_edges(
                        candidate_graph
                    ),

                "training_steps":
                    convergence[
                        "training_steps"
                    ],

                "converged":
                    int(
                        convergence[
                            "converged"
                        ]
                    ),

                "last_dkl":
                    convergence[
                        "last_dkl"
                    ],

                "last_relative_improvement":
                    convergence[
                        "last_relative_improvement"
                    ],

                "ess_current":
                    current_ess,

                "ess_candidate":
                    candidate_ess,

                "delta_ess":
                    delta_ess,

                "kl_current":
                    current_kl,

                "kl_candidate":
                    candidate_kl,

                "delta_kl":
                    delta_kl,

                "ess_pass":
                    int(
                        ess_pass
                    ),

                "accepted":
                    int(
                        acceptable
                    ),
            }
        )

        if acceptable:

            accepted.append(
                candidate
            )

    # ========================================================
    # Nothing improved ESS
    # ========================================================

    if len(
        accepted
    ) == 0:

        print()
        print(
            "NO CANDIDATE IMPROVED ESS"
        )

        for candidate in (
            all_candidates
        ):

            del candidate[
                "model"
            ]

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return (
            None,
            trial_rows,
        )

    # ========================================================
    # Choose highest ESS.
    #
    # KL and sparsity are only deterministic tie-breaks.
    # ========================================================

    accepted = sorted(
        accepted,
        key=lambda candidate: (
            -
            candidate[
                "ess"
            ],

            candidate[
                "kl"
            ],

            count_edges(
                candidate[
                    "graph"
                ]
            ),
        ),
    )

    best = (
        accepted[0]
    )

    print()
    print(
        "=" * 100
    )

    print(
        "SELECTED CANDIDATE"
    )

    print(
        f"name      = "
        f"{best['name']}"
    )

    print(
        f"edges     = "
        f"{count_edges(best['graph'])}"
    )

    print(
        f"ESS       = "
        f"{best['ess']:.6f}"
    )

    print(
        f"delta ESS = "
        f"{best['delta_ess']:+.6f}"
    )

    print(
        f"KL diag   = "
        f"{best['kl']:.6f}"
    )

    print(
        "=" * 100
    )

    for candidate in (
        all_candidates
    ):

        if candidate is not best:

            del candidate[
                "model"
            ]

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return (
        best,
        trial_rows,
    )


# ============================================================
# Run one benchmark
# ============================================================

def run_benchmark(
    benchmark,
    args,
    device,
    output_dir,
):
    print()
    print(
        "=" * 110
    )

    print(
        "ESS-ONLY ADAPTIVE PRUNING: "
        f"{benchmark}"
    )

    print(
        true_graph_description(
            benchmark
        )
    )

    print(
        "=" * 110
    )

    # ========================================================
    # Marginal training
    # ========================================================

    (
        marginal_states,
        marginal_diagnostics,
    ) = (
        train_all_marginal_states_local(
            benchmark,
            args,
            device,
        )
    )

    if (
        args.performance_seed
        not in
        marginal_states
    ):

        raise RuntimeError(
            "--performance-seed must "
            "appear in --seeds."
        )

    # ========================================================
    # Save marginal training curves
    # ========================================================

    marginal_rows = []

    for seed, result in (
        marginal_diagnostics.items()
    ):

        for row in (
            result[
                "history"
            ]
        ):

            marginal_rows.append(
                {
                    "benchmark":
                        benchmark,

                    "seed":
                        seed,

                    **row,
                }
            )

    pd.DataFrame(
        marginal_rows
    ).to_csv(
        output_dir
        /
        (
            f"{benchmark}_"
            "marginal_training.csv"
        ),
        index=False,
    )

    # ========================================================
    # Build U caches
    # ========================================================

    cache_by_seed = {}

    for seed in args.seeds:

        print()
        print(
            "=" * 70
        )

        print(
            f"BUILDING EVIDENCE CACHE "
            f"seed={seed}"
        )

        print(
            "=" * 70
        )

        bank = (
            MarginalBank(
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

        for parameter in (
            bank.parameters()
        ):

            parameter.requires_grad = False

        print(
            f"Training cache: "
            f"{args.cache_size}"
        )

        u_train, c_train = (
            build_u_cache(
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

        print(
            f"Delta/eval cache: "
            f"{args.delta_eval_size}"
        )

        u_eval, c_eval = (
            build_u_cache(
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

    # ========================================================
    # Fixed 100k KL cache
    # ========================================================

    print()
    print(
        "Building fixed 100k KL cache..."
    )

    seed_everything(
        args.eval_seed
    )

    kl_cache = build_kl_cache(
        benchmark=
            benchmark,

        n=
            args.kl_cache_size,

        device=
            device,
    )

    # ========================================================
    # Initial full graph
    # ========================================================

    graph = full_ar_graph(
        DIM
    )

    previous_graph = None

    model_cache = {}
    evidence_cache = {}

    frozen_sigma0 = None

    history = []
    all_trial_rows = []

    (
        current_model,
        current_metrics,
        initial_convergence,
    ) = train_initial_graph(
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

    initial_graph = copy_graph(
        graph
    )

    initial_metrics = dict(
        current_metrics
    )

    # ========================================================
    # Iterative pruning rounds
    # ========================================================

    for round_id in range(
        args.max_rounds
    ):

        print()
        print(
            "#" * 110
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
            f"KL    = "
            f"{current_metrics['kl']:.6f} "
            f"[diagnostic]"
        )

        print(
            graph
        )

        print(
            "#" * 110
        )

        # ====================================================
        # Evidence
        # ====================================================

        evidence_df = (
            collect_fast_evidence(
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

        if len(
            evidence_df
        ) == 0:

            print(
                "No active edges remain."
            )

            break

        # ====================================================
        # Bayesian weak-edge screening
        # ====================================================

        (
            posterior_df,
            posterior,
        ) = bayesian_round(
            evidence_df,
            args,
            fixed_sigma0=
                frozen_sigma0,
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

        weak_edges = []

        for row in (
            posterior_df.itertuples()
        ):

            pip = float(
                row.pip
            )

            if (
                pip
                <
                args.pip_threshold
            ):

                weak_edges.append(
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

        weak_edges = sorted(
            weak_edges,
            key=lambda item: (
                item[2],
                item[3],
            ),
        )

        if not weak_edges:

            print()
            print(
                "STOP: no low-PIP edges."
            )

            break

        print()
        print(
            "Weak-edge ranking:"
        )

        for rank, (
            parent,
            child,
            pip,
            delta,
        ) in enumerate(
            weak_edges,
            start=1,
        ):

            print(
                f"  {rank:2d}. "
                f"{parent}->{child} "
                f"PIP="
                f"{pip:.6f} "
                f"delta="
                f"{delta:.6f}"
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
            in weak_edges
        ]

        # ====================================================
        # Five candidate tests
        # ====================================================

        best, trial_rows = (
            search_candidates(
                benchmark=
                    benchmark,

                current_graph=
                    graph,

                current_model=
                    current_model,

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
            (
                f"{benchmark}_"
                "candidate_trials.csv"
            ),
            index=False,
        )

        evidence_df.to_csv(
            output_dir
            /
            (
                f"{benchmark}_"
                f"round_{round_id}_"
                "evidence.csv"
            ),
            index=False,
        )

        posterior_df.to_csv(
            output_dir
            /
            (
                f"{benchmark}_"
                f"round_{round_id}_"
                "posterior.csv"
            ),
            index=False,
        )

        if best is None:

            print()
            print(
                "=" * 100
            )

            print(
                "STOP: no candidate "
                "improved ESS."
            )

            print(
                "=" * 100
            )

            break

        # ====================================================
        # Accept highest ESS candidate
        # ====================================================

        old_graph = copy_graph(
            graph
        )

        old_model = (
            current_model
        )

        old_metrics = dict(
            current_metrics
        )

        previous_graph = (
            old_graph
        )

        graph = copy_graph(
            best[
                "graph"
            ]
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

            "variance":
                best[
                    "variance"
                ],

            "relative_error":
                best[
                    "relative_error"
                ],

            "kl":
                best[
                    "kl"
                ],
        }

        print()
        print(
            "=" * 100
        )

        print(
            "ACCEPTED ESS IMPROVEMENT"
        )

        print(
            f"candidate = "
            f"{best['name']}"
        )

        print(
            f"removals  = "
            f"{best['removals']}"
        )

        print(
            f"edges     = "
            f"{count_edges(old_graph)} "
            f"-> "
            f"{count_edges(graph)}"
        )

        print(
            f"ESS       = "
            f"{old_metrics['ess']:.6f} "
            f"-> "
            f"{current_metrics['ess']:.6f}"
        )

        print(
            f"KL diag   = "
            f"{old_metrics['kl']:.6f} "
            f"-> "
            f"{current_metrics['kl']:.6f}"
        )

        print(
            "=" * 100
        )

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

                "removals":
                    str(
                        best[
                            "removals"
                        ]
                    ),

                "edges_before":
                    count_edges(
                        old_graph
                    ),

                "edges_after":
                    count_edges(
                        graph
                    ),

                "ess_before":
                    old_metrics[
                        "ess"
                    ],

                "ess_after":
                    current_metrics[
                        "ess"
                    ],

                "delta_ess":
                    (
                        current_metrics[
                            "ess"
                        ]
                        -
                        old_metrics[
                            "ess"
                        ]
                    ),

                "kl_before":
                    old_metrics[
                        "kl"
                    ],

                "kl_after":
                    current_metrics[
                        "kl"
                    ],

                "delta_kl":
                    (
                        current_metrics[
                            "kl"
                        ]
                        -
                        old_metrics[
                            "kl"
                        ]
                    ),

                "training_steps":
                    best[
                        "training_steps"
                    ],

                "converged":
                    best[
                        "converged"
                    ],

                "frozen_sigma0":
                    frozen_sigma0,

                "graph_before":
                    graph_as_string(
                        old_graph
                    ),

                "graph_after":
                    graph_as_string(
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
    # Final result
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        f"FINAL GRAPH: "
        f"{benchmark}"
    )

    print(
        graph
    )

    print(
        f"Edges = "
        f"{count_edges(graph)}"
    )

    print(
        f"ESS   = "
        f"{current_metrics['ess']:.6f}"
    )

    print(
        f"KL    = "
        f"{current_metrics['kl']:.6f} "
        f"[diagnostic]"
    )

    print(
        "=" * 110
    )

    final_edges = []

    for child in range(
        DIM
    ):

        for parent in graph[
            child
        ]:

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

    performance_marginal = (
        marginal_diagnostics[
            args.performance_seed
        ]
    )

    summary = {
        "benchmark":
            benchmark,

        "structure_description":
            true_graph_description(
                benchmark
            ),

        "training_batch_size":
            args.batch_size,

        "eval_size":
            args.final_ess,

        "initial_edge_count":
            count_edges(
                initial_graph
            ),

        "final_edge_count":
            count_edges(
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

        "marginal_final_val_nll":
            performance_marginal[
                "final_val_nll"
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

        "initial_kl_diagnostic":
            initial_metrics[
                "kl"
            ],

        "final_kl_diagnostic":
            current_metrics[
                "kl"
            ],

        "accepted_rounds":
            len(
                history
            ),

        "final_graph":
            graph_as_string(
                graph
            ),
    }

    del current_model
    del cache_by_seed
    del kl_cache
    del marginal_states
    del marginal_diagnostics

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return (
        summary,
        final_edges,
    )


# ============================================================
# Driver
# ============================================================

def run(
    args,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    print(
        f"Training batch size: "
        f"{args.batch_size}"
    )

    print(
        f"Evaluation size: "
        f"{args.final_ess}"
    )

    print(
        f"Candidates per round: "
        f"{args.max_candidate_trials}"
    )

    print(
        "Acceptance gate: ESS only"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    all_final_edges = []

    for benchmark in (
        args.benchmarks
    ):

        summary, final_edges = (
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

        all_final_edges.extend(
            final_edges
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
            all_final_edges
        ).to_csv(
            output_dir
            /
            "final_graph_edges.csv",
            index=False,
        )

    print()
    print(
        "=" * 120
    )

    print(
        "ESS-ONLY BENCHMARKING COMPLETE"
    )

    print(
        "=" * 120
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
    parser = argparse.ArgumentParser()

    # ========================================================
    # Benchmarks
    # ========================================================

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "branching_tree",
            "skip_lag",
            "diamond_sparse",
            "mixed_sparse_dense",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=
            DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--performance-seed",
        type=int,
        default=
            PERFORMANCE_SEED,
    )

    # ========================================================
    # Training batch
    # ========================================================

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            TRAIN_BATCH_SIZE,
    )

    # ========================================================
    # Marginal convergence
    # ========================================================

    parser.add_argument(
        "--min-marginal-steps",
        type=int,
        default=
            MIN_MARGINAL_STEPS,
    )

    parser.add_argument(
        "--max-marginal-steps",
        type=int,
        default=
            MAX_MARGINAL_STEPS,
    )

    parser.add_argument(
        "--marginal-check-interval",
        type=int,
        default=
            MARGINAL_CHECK_INTERVAL,
    )

    parser.add_argument(
        "--marginal-rel-tol",
        type=float,
        default=
            MARGINAL_REL_TOL,
    )

    parser.add_argument(
        "--marginal-convergence-patience",
        type=int,
        default=
            MARGINAL_CONVERGENCE_PATIENCE,
    )

    parser.add_argument(
        "--marginal-val-size",
        type=int,
        default=
            MARGINAL_VAL_SIZE,
    )

    # ========================================================
    # Evidence
    # ========================================================

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

    parser.add_argument(
        "--cache-size",
        type=int,
        default=
            CACHE_SIZE,
    )

    parser.add_argument(
        "--delta-eval-size",
        type=int,
        default=
            DELTA_EVAL_SIZE,
    )

    parser.add_argument(
        "--keep-prob",
        type=float,
        default=
            KEEP_PROB,
    )

    # ========================================================
    # Copula convergence
    # ========================================================

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

    # ========================================================
    # Evaluation
    # ========================================================

    parser.add_argument(
        "--kl-cache-size",
        type=int,
        default=
            KL_CACHE_SIZE,
    )

    parser.add_argument(
        "--final-ess",
        type=int,
        default=
            FINAL_ESS,
    )

    parser.add_argument(
        "--final-kl",
        type=int,
        default=
            FINAL_KL,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=
            EVAL_SEED,
    )

    # ========================================================
    # Search
    # ========================================================

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=
            MAX_ROUNDS,
    )

    parser.add_argument(
        "--max-candidate-trials",
        type=int,
        default=
            MAX_CANDIDATE_TRIALS,
    )

    parser.add_argument(
        "--pip-threshold",
        type=float,
        default=
            PIP_REMOVE_THRESHOLD,
    )

    # ========================================================
    # ESS gate
    # ========================================================

    parser.add_argument(
        "--ess-tol",
        type=float,
        default=
            ESS_TOL,
    )

    # ========================================================
    # Bayesian parameters
    # ========================================================

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
        type=int,
        default=
            BAYES_RANDOM_SEED,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            DEFAULT_OUTPUT,
    )

    args = (
        parser.parse_args()
    )

    # ========================================================
    # Validation
    # ========================================================

    if (
        args.min_marginal_steps
        >
        args.max_marginal_steps
    ):

        raise ValueError(
            "--min-marginal-steps cannot "
            "exceed --max-marginal-steps"
        )

    if (
        args.min_candidate_steps
        >
        args.max_candidate_steps
    ):

        raise ValueError(
            "--min-candidate-steps cannot "
            "exceed --max-candidate-steps"
        )

    if (
        args.batch_size
        <=
        0
    ):

        raise ValueError(
            "--batch-size must be positive"
        )

    if (
        args.max_candidate_trials
        >
        5
    ):

        print(
            "Warning: this version only generates "
            "five structural candidates per round."
        )

    return args


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    run(
        parse_args()
    )