import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_copula_bin_sweep import (
    sample_target as sample_target_existing,
    target_logpdf as target_logpdf_existing,
    seed_everything,
)

from run_iterative_bayesian_edge_pruning_frozen_null import (
    DIM,
    COND_DIM,
    DEFAULT_SEEDS,
    KEEP_PROB,
    MARGINAL_STEPS,
    MASKED_STEPS,
    BATCH_SIZE,
    N_EVAL,
    MCMC_ITER,
    MCMC_BURN,
    MCMC_THIN,
    RHO_A,
    RHO_B,
    BAYES_RANDOM_SEED,
    PIP_REMOVE_THRESHOLD,

    MarginalBank,
    IterativeMaskedChild,

    bayesian_round,
    choose_removals,

    full_ar_graph,
    copy_graph,
    count_edges,
    graph_as_string,

    active_mask_vector,
    sample_training_mask,

    GRAD_CLIP,
    LR_MASKED,
)

from nf.conditional_cubic_1d import (
    ConditionalCubicFlow1D,
)


# ============================================================
# Configuration
# ============================================================

EXISTING_BENCHMARKS = {
    "single_gaussian",
    "curved_ridge",
    "sharp_broad",
}

NEW_BENCHMARKS = {
    "star_sparse",
    "block_sparse",
    "dense_ar",
    "condition_switch",
}

ALL_BENCHMARKS = [
    "single_gaussian",
    "curved_ridge",
    "sharp_broad",
    "star_sparse",
    "block_sparse",
    "dense_ar",
    "condition_switch",
]

REFINE_STEPS = 400

SCREEN_N_EVAL = 20_000
SCREEN_N_KL = 20_000

MAX_ROUNDS = 10

# Strict performance criterion.
#
# candidate ESS must not be lower
# candidate KL must not be higher
#
# These default to exactly zero.
ESS_TOL = 0.0
KL_TOL = 0.0

DEFAULT_OUTPUT = (
    "results/"
    "performance_guarded_structure_benchmark"
)


# ============================================================
# Standard normal helpers
# ============================================================

def standard_normal(
    device,
):
    return torch.distributions.Normal(
        torch.tensor(
            0.0,
            device=device,
        ),
        torch.tensor(
            1.0,
            device=device,
        ),
    )


# ============================================================
# Truncated Normal [0,1]
# ============================================================

def truncated_normal_sample(
    mu,
    sigma,
):

    device = mu.device

    normal = standard_normal(
        device
    )

    sigma = torch.clamp(
        sigma,
        min=1e-4,
    )

    lower = (
        -mu
        / sigma
    )

    upper = (
        (1.0 - mu)
        / sigma
    )

    cdf_lower = (
        normal.cdf(
            lower
        )
    )

    cdf_upper = (
        normal.cdf(
            upper
        )
    )

    u = torch.rand_like(
        mu
    )

    p = (
        cdf_lower
        +
        u
        * (
            cdf_upper
            - cdf_lower
        )
    )

    p = torch.clamp(
        p,
        1e-7,
        1.0 - 1e-7,
    )

    z = (
        normal.icdf(
            p
        )
    )

    x = (
        mu
        + sigma * z
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

    device = x.device

    normal = standard_normal(
        device
    )

    sigma = torch.clamp(
        sigma,
        min=1e-4,
    )

    z = (
        (x - mu)
        / sigma
    )

    lower = (
        -mu
        / sigma
    )

    upper = (
        (1.0 - mu)
        / sigma
    )

    normalization = (
        normal.cdf(
            upper
        )
        -
        normal.cdf(
            lower
        )
    )

    normalization = torch.clamp(
        normalization,
        min=1e-12,
    )

    log_base = (
        -0.5
        * z.square()
        - math.log(
            math.sqrt(
                2.0 * math.pi
            )
        )
        - torch.log(
            sigma
        )
    )

    return (
        log_base
        -
        torch.log(
            normalization
        )
    )


# ============================================================
# New benchmark 1:
# Sparse star graph
#
#       0
#      /|\
#     1 2 ... 7
#
# True edges:
#
# 0 -> j  for j = 1,...,7
#
# ============================================================

def sample_star_sparse(
    cond,
    dim,
):

    n = cond.shape[0]

    device = cond.device

    x = torch.zeros(
        n,
        dim,
        device=device,
    )

    x[:, 0] = torch.rand(
        n,
        device=device,
    )

    sigma = (
        0.045
        +
        0.015
        * cond[:, 1]
    )

    for j in range(
        1,
        dim,
    ):

        phase = (
            0.13 * j
        )

        mu = (
            0.5
            +
            0.27
            * torch.sin(
                2.0
                * math.pi
                * (
                    x[:, 0]
                    +
                    0.15
                    * cond[:, 0]
                    +
                    phase
                )
            )
        )

        x[:, j] = (
            truncated_normal_sample(
                mu,
                sigma,
            )
        )

    return x


def logpdf_star_sparse(
    x,
    cond,
):

    logp = torch.zeros(
        x.shape[0],
        device=x.device,
    )

    sigma = (
        0.045
        +
        0.015
        * cond[:, 1]
    )

    # x0 is Uniform(0,1), log density = 0.

    for j in range(
        1,
        DIM,
    ):

        phase = (
            0.13 * j
        )

        mu = (
            0.5
            +
            0.27
            * torch.sin(
                2.0
                * math.pi
                * (
                    x[:, 0]
                    +
                    0.15
                    * cond[:, 0]
                    +
                    phase
                )
            )
        )

        logp += (
            truncated_normal_logpdf(
                x[:, j],
                mu,
                sigma,
            )
        )

    return logp


# ============================================================
# New benchmark 2:
# Two independent sparse chains
#
# 0 -> 1 -> 2 -> 3
#
# 4 -> 5 -> 6 -> 7
#
# True edges = 6
# ============================================================

def sample_block_sparse(
    cond,
    dim,
):

    n = cond.shape[0]

    device = cond.device

    x = torch.zeros(
        n,
        dim,
        device=device,
    )

    # Two independent roots
    x[:, 0] = torch.rand(
        n,
        device=device,
    )

    x[:, 4] = torch.rand(
        n,
        device=device,
    )

    sigma = (
        0.045
        +
        0.01
        * cond[:, 2]
    )

    # first block
    for j in [
        1,
        2,
        3,
    ]:

        mu = (
            0.5
            +
            0.28
            * torch.sin(
                2.0
                * math.pi
                * (
                    x[:, j - 1]
                    +
                    0.12
                    * cond[:, 0]
                    +
                    0.07 * j
                )
            )
        )

        x[:, j] = (
            truncated_normal_sample(
                mu,
                sigma,
            )
        )

    # second block
    for j in [
        5,
        6,
        7,
    ]:

        mu = (
            0.5
            +
            0.28
            * torch.cos(
                2.0
                * math.pi
                * (
                    x[:, j - 1]
                    +
                    0.12
                    * cond[:, 1]
                    +
                    0.05 * j
                )
            )
        )

        x[:, j] = (
            truncated_normal_sample(
                mu,
                sigma,
            )
        )

    return x


def logpdf_block_sparse(
    x,
    cond,
):

    logp = torch.zeros(
        x.shape[0],
        device=x.device,
    )

    sigma = (
        0.045
        +
        0.01
        * cond[:, 2]
    )

    for j in [
        1,
        2,
        3,
    ]:

        mu = (
            0.5
            +
            0.28
            * torch.sin(
                2.0
                * math.pi
                * (
                    x[:, j - 1]
                    +
                    0.12
                    * cond[:, 0]
                    +
                    0.07 * j
                )
            )
        )

        logp += (
            truncated_normal_logpdf(
                x[:, j],
                mu,
                sigma,
            )
        )

    for j in [
        5,
        6,
        7,
    ]:

        mu = (
            0.5
            +
            0.28
            * torch.cos(
                2.0
                * math.pi
                * (
                    x[:, j - 1]
                    +
                    0.12
                    * cond[:, 1]
                    +
                    0.05 * j
                )
            )
        )

        logp += (
            truncated_normal_logpdf(
                x[:, j],
                mu,
                sigma,
            )
        )

    return logp


# ============================================================
# New benchmark 3:
# Dense autoregressive target
#
# Every previous x_i contributes to x_j.
#
# True graph = full AR.
#
# Unlike sharp_broad, this has no shared latent variable
# intentionally creating obvious redundancy.
# ============================================================

def dense_mu(
    x,
    cond,
    child,
):

    parents = (
        x[
            :,
            :child
        ]
        - 0.5
    )

    # deterministic nonidentical weights
    indices = torch.arange(
        1,
        child + 1,
        device=x.device,
        dtype=x.dtype,
    )

    weights = (
        torch.sin(
            0.73
            * indices
            * (
                child + 1
            )
        )
        +
        1.4
    )

    weights = (
        weights
        /
        torch.sqrt(
            torch.tensor(
                float(child),
                device=x.device,
            )
        )
    )

    signal = (
        parents
        * weights.unsqueeze(0)
    ).sum(
        dim=1
    )

    signal = (
        signal
        +
        0.40
        * (
            cond[:, 0]
            - 0.5
        )
        +
        0.20
        * torch.sin(
            2.0
            * math.pi
            * cond[:, 1]
        )
    )

    return (
        0.5
        +
        0.30
        * torch.tanh(
            signal
        )
    )


def sample_dense_ar(
    cond,
    dim,
):

    n = cond.shape[0]

    device = cond.device

    x = torch.zeros(
        n,
        dim,
        device=device,
    )

    x[:, 0] = torch.rand(
        n,
        device=device,
    )

    sigma = (
        0.040
        +
        0.008
        * cond[:, 2]
    )

    for j in range(
        1,
        dim,
    ):

        mu = dense_mu(
            x,
            cond,
            j,
        )

        x[:, j] = (
            truncated_normal_sample(
                mu,
                sigma,
            )
        )

    return x


def logpdf_dense_ar(
    x,
    cond,
):

    logp = torch.zeros(
        x.shape[0],
        device=x.device,
    )

    sigma = (
        0.040
        +
        0.008
        * cond[:, 2]
    )

    for j in range(
        1,
        DIM,
    ):

        mu = dense_mu(
            x,
            cond,
            j,
        )

        logp += (
            truncated_normal_logpdf(
                x[:, j],
                mu,
                sigma,
            )
        )

    return logp


# ============================================================
# New benchmark 4:
# Condition-switching topology
#
# c0 < 0.5:
#
#     chain:
#     0 -> 1 -> 2 -> ... -> 7
#
# c0 >= 0.5:
#
#     star:
#     0 -> all others
#
# This benchmark is intentionally designed to expose whether
# one global graph is enough.
# ============================================================

def sample_condition_switch(
    cond,
    dim,
):

    n = cond.shape[0]

    device = cond.device

    x = torch.zeros(
        n,
        dim,
        device=device,
    )

    x[:, 0] = torch.rand(
        n,
        device=device,
    )

    sigma = (
        0.045
        +
        0.010
        * cond[:, 2]
    )

    chain_mode = (
        cond[:, 0]
        < 0.5
    )

    for j in range(
        1,
        dim,
    ):

        chain_parent = (
            x[:, j - 1]
        )

        star_parent = (
            x[:, 0]
        )

        parent = torch.where(
            chain_mode,
            chain_parent,
            star_parent,
        )

        mu = (
            0.5
            +
            0.28
            * torch.sin(
                2.0
                * math.pi
                * (
                    parent
                    +
                    0.12
                    * cond[:, 1]
                    +
                    0.08 * j
                )
            )
        )

        x[:, j] = (
            truncated_normal_sample(
                mu,
                sigma,
            )
        )

    return x


def logpdf_condition_switch(
    x,
    cond,
):

    logp = torch.zeros(
        x.shape[0],
        device=x.device,
    )

    sigma = (
        0.045
        +
        0.010
        * cond[:, 2]
    )

    chain_mode = (
        cond[:, 0]
        < 0.5
    )

    for j in range(
        1,
        DIM,
    ):

        chain_parent = (
            x[:, j - 1]
        )

        star_parent = (
            x[:, 0]
        )

        parent = torch.where(
            chain_mode,
            chain_parent,
            star_parent,
        )

        mu = (
            0.5
            +
            0.28
            * torch.sin(
                2.0
                * math.pi
                * (
                    parent
                    +
                    0.12
                    * cond[:, 1]
                    +
                    0.08 * j
                )
            )
        )

        logp += (
            truncated_normal_logpdf(
                x[:, j],
                mu,
                sigma,
            )
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

    if benchmark in (
        EXISTING_BENCHMARKS
    ):

        return (
            sample_target_existing(
                benchmark,
                cond,
                dim,
            )
        )

    if benchmark == (
        "star_sparse"
    ):

        return sample_star_sparse(
            cond,
            dim,
        )

    if benchmark == (
        "block_sparse"
    ):

        return sample_block_sparse(
            cond,
            dim,
        )

    if benchmark == (
        "dense_ar"
    ):

        return sample_dense_ar(
            cond,
            dim,
        )

    if benchmark == (
        "condition_switch"
    ):

        return (
            sample_condition_switch(
                cond,
                dim,
            )
        )

    raise ValueError(
        f"Unknown benchmark: "
        f"{benchmark}"
    )


def target_logpdf(
    benchmark,
    x,
    cond,
):

    if benchmark in (
        EXISTING_BENCHMARKS
    ):

        return (
            target_logpdf_existing(
                benchmark,
                x,
                cond,
            )
        )

    if benchmark == (
        "star_sparse"
    ):

        return (
            logpdf_star_sparse(
                x,
                cond,
            )
        )

    if benchmark == (
        "block_sparse"
    ):

        return (
            logpdf_block_sparse(
                x,
                cond,
            )
        )

    if benchmark == (
        "dense_ar"
    ):

        return (
            logpdf_dense_ar(
                x,
                cond,
            )
        )

    if benchmark == (
        "condition_switch"
    ):

        return (
            logpdf_condition_switch(
                x,
                cond,
            )
        )

    raise ValueError(
        f"Unknown benchmark: "
        f"{benchmark}"
    )


# ============================================================
# True graph information
# ============================================================

def true_graph_description(
    benchmark,
):

    if benchmark == (
        "single_gaussian"
    ):

        return (
            "independent: 0 true edges"
        )

    if benchmark == (
        "curved_ridge"
    ):

        return (
            "Markov-1 chain: 7 true edges"
        )

    if benchmark == (
        "sharp_broad"
    ):

        return (
            "global latent mixture; "
            "distributed/redundant dependence"
        )

    if benchmark == (
        "star_sparse"
    ):

        return (
            "star: 0->j for j=1..7; "
            "7 true edges"
        )

    if benchmark == (
        "block_sparse"
    ):

        return (
            "two chains: "
            "0->1->2->3 and "
            "4->5->6->7; "
            "6 true edges"
        )

    if benchmark == (
        "dense_ar"
    ):

        return (
            "dense AR: all 28 "
            "previous-coordinate edges"
        )

    if benchmark == (
        "condition_switch"
    ):

        return (
            "c0<0.5 chain; "
            "c0>=0.5 star"
        )

    return "unknown"


# ============================================================
# Sampling conditions
# ============================================================

def sample_conditions(
    n,
    device,
):

    return torch.rand(
        n,
        COND_DIM,
        device=device,
    )


# ============================================================
# Train marginals with extended targets
# ============================================================

def train_marginals_extended(
    bank,
    benchmark,
    device,
    steps,
    batch_size,
):

    optimizer = torch.optim.Adam(
        bank.parameters(),
        lr=1e-4,
    )

    bank.train()

    recent = []

    for step in range(
        steps
    ):

        cond = sample_conditions(
            batch_size,
            device,
        )

        x = sample_target(
            benchmark,
            cond,
            DIM,
        )

        loss = 0.0

        for j in range(
            DIM
        ):

            _, forward_logdet = (
                bank.marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            loss += (
                forward_logdet.mean()
            )

        loss /= DIM

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            bank.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        recent.append(
            loss.item()
        )

        if (
            step + 1
        ) % 300 == 0:

            print(
                f"    marginal "
                f"{step+1:4d}/{steps} "
                f"loss="
                f"{np.mean(recent[-100:]):.6f}"
            )


# ============================================================
# Marginal state banks
# ============================================================

def train_all_marginal_states(
    benchmark,
    args,
    device,
):

    states = {}

    for seed in (
        args.seeds
    ):

        print()
        print(
            f"Training marginals "
            f"seed={seed}"
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

        train_marginals_extended(
            bank,
            benchmark,
            device,
            args.marginal_steps,
            args.batch_size,
        )

        states[
            seed
        ] = {
            key:
                value.detach()
                .cpu()
                .clone()

            for key, value
            in bank.state_dict().items()
        }

        del bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return states


# ============================================================
# Warm-start masked model
# ============================================================

def clone_state_dict(
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


def train_masked_child(
    bank,
    benchmark,
    child,
    active_parents,
    device,
    steps,
    batch_size,
    keep_prob,
    seed,
    previous_state=None,
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

    for _ in range(
        steps
    ):

        cond = sample_conditions(
            batch_size,
            device,
        )

        with torch.no_grad():

            x = sample_target(
                benchmark,
                cond,
                DIM,
            )

            u = bank.x_to_u(
                x,
                cond,
            )

        child_u = (
            u[
                :,
                child:child+1
            ]
        )

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
            child_u,
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

    return model


# ============================================================
# Delta computation
# ============================================================

@torch.no_grad()
def compute_child_deltas(
    model,
    bank,
    benchmark,
    child,
    active_parents,
    device,
    n_eval,
):

    if len(
        active_parents
    ) == 0:

        return []

    cond = sample_conditions(
        n_eval,
        device,
    )

    x = sample_target(
        benchmark,
        cond,
        DIM,
    )

    u = bank.x_to_u(
        x,
        cond,
    )

    child_u = (
        u[
            :,
            child:child+1
        ]
    )

    base_mask = (
        active_mask_vector(
            child,
            active_parents,
            device,
        )
        .unsqueeze(0)
        .expand(
            n_eval,
            -1,
        )
        .clone()
    )

    logq_on = model.log_prob(
        child_u,
        u,
        cond,
        base_mask,
    )

    rows = []

    for parent in (
        active_parents
    ):

        mask_off = (
            base_mask.clone()
        )

        mask_off[
            :,
            parent
        ] = 0.0

        logq_off = (
            model.log_prob(
                child_u,
                u,
                cond,
                mask_off,
            )
        )

        delta = (
            logq_on
            -
            logq_off
        )

        rows.append(
            {
                "parent":
                    parent,

                "child":
                    child,

                "delta_loglik":
                    delta.mean()
                    .item(),

                "delta_std":
                    delta.std(
                        unbiased=True
                    )
                    .item(),

                "is_markov1_edge":
                    int(
                        parent
                        == child - 1
                    ),
            }
        )

    return rows


# ============================================================
# Cached evidence collection
# ============================================================

def collect_evidence(
    benchmark,
    round_id,
    graph,
    previous_graph,
    marginal_states,
    model_cache,
    evidence_cache,
    args,
    device,
):

    rows = []

    for seed in (
        args.seeds
    ):

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

        print()
        print(
            f"Evidence seed {seed}"
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
                    previous_graph[
                        child
                    ]
                    != parents
                )

            if (
                not changed
                and
                key in evidence_cache
            ):

                print(
                    f"  child {child}: "
                    f"{parents} [REUSE]"
                )

                for cached in (
                    evidence_cache[
                        key
                    ]
                ):

                    row = dict(
                        cached
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

                mode = (
                    "FULL"
                )

            else:

                steps = (
                    args.refine_steps
                )

                mode = (
                    "REFINE"
                )

            print(
                f"  child {child}: "
                f"{parents} "
                f"[{mode} {steps}]"
            )

            model_seed = (
                seed * 100000
                +
                round_id * 1000
                +
                child
            )

            model = train_masked_child(
                bank=
                    bank,

                benchmark=
                    benchmark,

                child=
                    child,

                active_parents=
                    parents,

                device=
                    device,

                steps=
                    steps,

                batch_size=
                    args.batch_size,

                keep_prob=
                    args.keep_prob,

                seed=
                    model_seed,

                previous_state=
                    previous_state,
            )

            model.eval()

            child_rows = (
                compute_child_deltas(
                    model=
                        model,

                    bank=
                        bank,

                    benchmark=
                        benchmark,

                    child=
                        child,

                    active_parents=
                        parents,

                    device=
                        device,

                    n_eval=
                        args.n_eval,
                )
            )

            model_cache[
                key
            ] = (
                clone_state_dict(
                    model
                )
            )

            evidence_cache[
                key
            ] = [
                dict(row)
                for row
                in child_rows
            ]

            for child_row in (
                child_rows
            ):

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
                            "trained",
                    }
                )

                rows.append(
                    row
                )

            del model

        del bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return pd.DataFrame(
        rows
    )


# ============================================================
# Restore cached child models
# ============================================================

def restore_models(
    seed,
    model_cache,
    device,
):

    models = {}

    for child in range(
        1,
        DIM,
    ):

        key = (
            seed,
            child,
        )

        model = (
            IterativeMaskedChild(
                child
            )
            .to(device)
        )

        model.load_state_dict(
            model_cache[
                key
            ]
        )

        model.eval()

        for p in (
            model.parameters()
        ):

            p.requires_grad = False

        models[
            child
        ] = model

    return models


# ============================================================
# log q from masked models
# ============================================================

@torch.no_grad()
def log_prob_candidate(
    bank,
    models,
    graph,
    x,
    cond,
    device,
):

    u_list = []

    log_marginal = torch.zeros(
        x.shape[0],
        device=device,
    )

    for j in range(
        DIM
    ):

        uj, forward_ld = (
            bank.marginals[j]
            .inverse(
                x[:, j:j+1],
                cond,
            )
        )

        uj = uj.clamp(
            0.0,
            1.0,
        )

        u_list.append(
            uj
        )

        log_marginal -= (
            forward_ld.squeeze(-1)
        )

    u = torch.cat(
        u_list,
        dim=1,
    )

    log_copula = torch.zeros(
        x.shape[0],
        device=device,
    )

    for child in range(
        1,
        DIM,
    ):

        mask = (
            active_mask_vector(
                child,
                graph[
                    child
                ],
                device,
            )
            .unsqueeze(0)
            .expand(
                x.shape[0],
                -1,
            )
        )

        log_copula += (
            models[
                child
            ]
            .log_prob(
                u[
                    :,
                    child:child+1
                ],
                u,
                cond,
                mask,
            )
        )

    return (
        log_marginal
        +
        log_copula
    )


# ============================================================
# Sample candidate q
# ============================================================

@torch.no_grad()
def sample_candidate(
    bank,
    models,
    graph,
    cond,
    n,
    device,
):

    cond_batch = cond.expand(
        n,
        -1,
    )

    u = torch.zeros(
        n,
        DIM,
        device=device,
    )

    u[:, 0] = torch.rand(
        n,
        device=device,
    )

    for child in range(
        1,
        DIM,
    ):

        mask = (
            active_mask_vector(
                child,
                graph[
                    child
                ],
                device,
            )
            .unsqueeze(0)
            .expand(
                n,
                -1,
            )
        )

        model = models[
            child
        ]

        context = (
            model.make_context(
                u,
                cond_batch,
                mask,
            )
        )

        base = torch.rand(
            n,
            1,
            device=device,
        )

        uj, _ = (
            model.flow.forward(
                base,
                context,
            )
        )

        u[
            :,
            child:child+1
        ] = uj.clamp(
            0.0,
            1.0,
        )

    x_list = []

    for j in range(
        DIM
    ):

        xj, _ = (
            bank.marginals[j]
            .forward(
                u[:, j:j+1],
                cond_batch,
            )
        )

        x_list.append(
            xj.clamp(
                0.0,
                1.0,
            )
        )

    return torch.cat(
        x_list,
        dim=1,
    )


# ============================================================
# Stable ESS
# ============================================================

def ess_fraction(
    logw,
):

    n = logw.numel()

    log_sum_w = torch.logsumexp(
        logw,
        dim=0,
    )

    log_sum_w2 = torch.logsumexp(
        2.0 * logw,
        dim=0,
    )

    return torch.exp(
        2.0 * log_sum_w
        -
        log_sum_w2
        -
        math.log(
            n
        )
    )


# ============================================================
# Fixed screening conditions
# ============================================================

SCREEN_CONDITIONS = torch.tensor(
    [
        [0.10, 0.10, 0.10],
        [0.10, 0.90, 0.90],
        [0.25, 0.50, 0.75],
        [0.40, 0.25, 0.50],

        [0.60, 0.25, 0.50],
        [0.75, 0.50, 0.25],
        [0.90, 0.10, 0.90],
        [0.90, 0.90, 0.10],
    ],
    dtype=torch.float32,
)


# ============================================================
# Performance evaluation
#
# IMPORTANT:
# The same random seed is reset before every graph evaluation.
# This gives common random numbers as much as possible.
# ============================================================

@torch.no_grad()
def evaluate_graph(
    benchmark,
    graph,
    marginal_states,
    model_cache,
    args,
    device,
):

    all_ess = []
    all_kl = []

    for seed in (
        args.seeds
    ):

        seed_everything(
            args.screen_seed
            + seed
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

        models = restore_models(
            seed,
            model_cache,
            device,
        )

        for cond in (
            SCREEN_CONDITIONS
        ):

            cond = (
                cond.to(device)
                .reshape(
                    1,
                    -1,
                )
            )

            # ================================================
            # ESS
            # ================================================

            xq = sample_candidate(
                bank=
                    bank,

                models=
                    models,

                graph=
                    graph,

                cond=
                    cond,

                n=
                    args.screen_n_eval,

                device=
                    device,
            )

            cq = cond.expand(
                args.screen_n_eval,
                -1,
            )

            logq = (
                log_prob_candidate(
                    bank,
                    models,
                    graph,
                    xq,
                    cq,
                    device,
                )
            )

            logp = target_logpdf(
                benchmark,
                xq,
                cq,
            )

            ess = ess_fraction(
                logp - logq
            )

            all_ess.append(
                ess.item()
            )

            # ================================================
            # KL
            #
            # Same target samples because RNG is reset
            # deterministically for candidate/current graph.
            # ================================================

            cp = cond.expand(
                args.screen_n_kl,
                -1,
            )

            xp = sample_target(
                benchmark,
                cp,
                DIM,
            )

            logp_p = target_logpdf(
                benchmark,
                xp,
                cp,
            )

            logq_p = (
                log_prob_candidate(
                    bank,
                    models,
                    graph,
                    xp,
                    cp,
                    device,
                )
            )

            kl = (
                logp_p
                - logq_p
            ).mean()

            all_kl.append(
                kl.item()
            )

        del bank
        del models

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return {
        "ess_mean":
            float(
                np.mean(
                    all_ess
                )
            ),

        "ess_std":
            float(
                np.std(
                    all_ess
                )
            ),

        "kl_mean":
            float(
                np.mean(
                    all_kl
                )
            ),

        "kl_std":
            float(
                np.std(
                    all_kl
                )
            ),
    }


# ============================================================
# Performance acceptance criterion
# ============================================================

def performance_accept(
    current,
    candidate,
    ess_tol,
    kl_tol,
):

    delta_ess = (
        candidate[
            "ess_mean"
        ]
        -
        current[
            "ess_mean"
        ]
    )

    delta_kl = (
        candidate[
            "kl_mean"
        ]
        -
        current[
            "kl_mean"
        ]
    )

    accept = (
        delta_ess
        >=
        -ess_tol
        and
        delta_kl
        <=
        kl_tol
    )

    return (
        accept,
        delta_ess,
        delta_kl,
    )


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
        "=" * 110
    )

    print(
        f"PERFORMANCE-GUARDED STRUCTURE: "
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

    marginal_states = (
        train_all_marginal_states(
            benchmark,
            args,
            device,
        )
    )

    graph = full_ar_graph(
        DIM
    )

    previous_graph = None

    model_cache = {}
    evidence_cache = {}

    frozen_sigma0 = None

    history_rows = []
    edge_rows = []

    current_metrics = None

    for round_id in range(
        args.max_rounds
    ):

        print()
        print(
            "#" * 100
        )

        print(
            f"ROUND {round_id} "
            f"| edges="
            f"{count_edges(graph)}"
        )

        print(
            graph
        )

        print(
            "#" * 100
        )

        # ====================================================
        # Train / update masked intervention models
        # ====================================================

        evidence_df = collect_evidence(
            benchmark=
                benchmark,

            round_id=
                round_id,

            graph=
                graph,

            previous_graph=
                previous_graph,

            marginal_states=
                marginal_states,

            model_cache=
                model_cache,

            evidence_cache=
                evidence_cache,

            args=
                args,

            device=
                device,
        )

        # ====================================================
        # Current graph performance after current models have
        # been brought up to date.
        # ====================================================

        current_metrics = evaluate_graph(
            benchmark=
                benchmark,

            graph=
                graph,

            marginal_states=
                marginal_states,

            model_cache=
                model_cache,

            args=
                args,

            device=
                device,
        )

        print()
        print(
            "Current performance:"
        )

        print(
            f"  ESS = "
            f"{current_metrics['ess_mean']:.6f}"
        )

        print(
            f"  KL  = "
            f"{current_metrics['kl_mean']:.6f}"
        )

        # ====================================================
        # Bayesian inference
        # ====================================================

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

            print(
                f"Frozen sigma0 = "
                f"{frozen_sigma0:.8f}"
            )

        # ====================================================
        # Bayesian proposals
        #
        # one candidate per child maximum
        # ====================================================

        proposals = choose_removals(
            posterior_df,
            args.pip_threshold,
        )

        if len(
            proposals
        ) == 0:

            print(
                "STOP: no Bayesian "
                "removal proposals."
            )

            break

        pip_lookup = {
            (
                int(row.parent),
                int(row.child),
            ):
                float(row.pip)

            for row in (
                posterior_df.itertuples()
            )
        }

        # weakest first
        proposals = sorted(
            proposals,
            key=lambda e:
                pip_lookup[e],
        )

        print()
        print(
            "Proposals:"
        )

        print(
            proposals
        )

        accepted = []

        candidate_graph = (
            copy_graph(
                graph
            )
        )

        candidate_metrics = dict(
            current_metrics
        )

        # ====================================================
        # Sequential ESS/KL guard
        # ====================================================

        for parent, child in (
            proposals
        ):

            trial_graph = (
                copy_graph(
                    candidate_graph
                )
            )

            trial_graph[
                child
            ].remove(
                parent
            )

            # Reset RNG inside evaluate_graph,
            # so comparisons use common random numbers.
            trial_metrics = evaluate_graph(
                benchmark=
                    benchmark,

                graph=
                    trial_graph,

                marginal_states=
                    marginal_states,

                model_cache=
                    model_cache,

                args=
                    args,

                device=
                    device,
            )

            (
                accept,
                delta_ess,
                delta_kl,
            ) = performance_accept(
                current=
                    candidate_metrics,

                candidate=
                    trial_metrics,

                ess_tol=
                    args.ess_tol,

                kl_tol=
                    args.kl_tol,
            )

            print()
            print(
                f"Test edge "
                f"{parent}->{child}"
            )

            print(
                f"  PIP        = "
                f"{pip_lookup[(parent, child)]:.6f}"
            )

            print(
                f"  ESS old    = "
                f"{candidate_metrics['ess_mean']:.6f}"
            )

            print(
                f"  ESS new    = "
                f"{trial_metrics['ess_mean']:.6f}"
            )

            print(
                f"  delta ESS  = "
                f"{delta_ess:+.6f}"
            )

            print(
                f"  KL old     = "
                f"{candidate_metrics['kl_mean']:.6f}"
            )

            print(
                f"  KL new     = "
                f"{trial_metrics['kl_mean']:.6f}"
            )

            print(
                f"  delta KL   = "
                f"{delta_kl:+.6f}"
            )

            print(
                f"  decision   = "
                f"{'ACCEPT' if accept else 'REJECT'}"
            )

            edge_rows.append(
                {
                    "benchmark":
                        benchmark,

                    "round":
                        round_id,

                    "parent":
                        parent,

                    "child":
                        child,

                    "pip":
                        pip_lookup[
                            (
                                parent,
                                child,
                            )
                        ],

                    "ess_before":
                        candidate_metrics[
                            "ess_mean"
                        ],

                    "ess_after":
                        trial_metrics[
                            "ess_mean"
                        ],

                    "delta_ess":
                        delta_ess,

                    "kl_before":
                        candidate_metrics[
                            "kl_mean"
                        ],

                    "kl_after":
                        trial_metrics[
                            "kl_mean"
                        ],

                    "delta_kl":
                        delta_kl,

                    "accepted":
                        int(
                            accept
                        ),
                }
            )

            if accept:

                candidate_graph = (
                    trial_graph
                )

                candidate_metrics = (
                    trial_metrics
                )

                accepted.append(
                    (
                        parent,
                        child,
                    )
                )

        # ====================================================
        # Round history
        # ====================================================

        history_rows.append(
            {
                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "edges_before":
                    count_edges(
                        graph
                    ),

                "ess_before":
                    current_metrics[
                        "ess_mean"
                    ],

                "kl_before":
                    current_metrics[
                        "kl_mean"
                    ],

                "proposed":
                    str(
                        proposals
                    ),

                "accepted":
                    str(
                        accepted
                    ),

                "n_proposed":
                    len(
                        proposals
                    ),

                "n_accepted":
                    len(
                        accepted
                    ),

                "edges_after":
                    count_edges(
                        candidate_graph
                    ),

                "ess_after":
                    candidate_metrics[
                        "ess_mean"
                    ],

                "kl_after":
                    candidate_metrics[
                        "kl_mean"
                    ],

                "frozen_sigma0":
                    frozen_sigma0,

                "graph_before":
                    graph_as_string(
                        graph
                    ),

                "graph_after":
                    graph_as_string(
                        candidate_graph
                    ),
            }
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
            edge_rows
        ).to_csv(
            output_dir
            /
            f"{benchmark}_edge_trials.csv",
            index=False,
        )

        # ====================================================
        # Stop if performance vetoed everything
        # ====================================================

        if len(
            accepted
        ) == 0:

            print()
            print(
                "STOP: every Bayesian "
                "proposal hurt ESS and/or KL."
            )

            break

        previous_graph = (
            copy_graph(
                graph
            )
        )

        graph = (
            candidate_graph
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        f"FINAL GRAPH: {benchmark}"
    )

    print(
        graph
    )

    print(
        f"edges = "
        f"{count_edges(graph)}"
    )

    print(
        "=" * 90
    )

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

    return {
        "benchmark":
            benchmark,

        "true_structure":
            true_graph_description(
                benchmark
            ),

        "final_edge_count":
            count_edges(
                graph
            ),

        "final_graph":
            graph_as_string(
                graph
            ),

        "final_ess_screen":
            current_metrics[
                "ess_mean"
            ]
            if current_metrics
            is not None
            else np.nan,

        "final_kl_screen":
            current_metrics[
                "kl_mean"
            ]
            if current_metrics
            is not None
            else np.nan,

        "final_edges":
            final_edges,
    }


# ============================================================
# Main
# ============================================================

def run(
    args,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    all_edges = []

    for benchmark in (
        args.benchmarks
    ):

        result = run_benchmark(
            benchmark,
            args,
            device,
            output_dir,
        )

        all_edges.extend(
            result.pop(
                "final_edges"
            )
        )

        summaries.append(
            result
        )

        pd.DataFrame(
            summaries
        ).to_csv(
            output_dir
            / "summary.csv",
            index=False,
        )

        pd.DataFrame(
            all_edges
        ).to_csv(
            output_dir
            / "final_graph_edges.csv",
            index=False,
        )

    print()
    print(
        "=" * 120
    )

    print(
        "PERFORMANCE-GUARDED "
        "STRUCTURE BENCHMARK COMPLETE"
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

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=
            ALL_BENCHMARKS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=
            DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--marginal-steps",
        type=int,
        default=
            MARGINAL_STEPS,
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            BATCH_SIZE,
    )

    parser.add_argument(
        "--n-eval",
        type=int,
        default=
            N_EVAL,
    )

    parser.add_argument(
        "--screen-n-eval",
        type=int,
        default=
            SCREEN_N_EVAL,
    )

    parser.add_argument(
        "--screen-n-kl",
        type=int,
        default=
            SCREEN_N_KL,
    )

    parser.add_argument(
        "--screen-seed",
        type=int,
        default=777,
    )

    parser.add_argument(
        "--keep-prob",
        type=float,
        default=
            KEEP_PROB,
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
        "--ess-tol",
        type=float,
        default=
            ESS_TOL,
    )

    parser.add_argument(
        "--kl-tol",
        type=float,
        default=
            KL_TOL,
    )

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

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if args.smoke:

        args.seeds = [
            0,
        ]

        args.marginal_steps = 100
        args.masked_steps = 300
        args.refine_steps = 100

        args.batch_size = 1024

        args.n_eval = 10_000

        args.screen_n_eval = 2_000
        args.screen_n_kl = 2_000

        args.max_rounds = 3

        args.mcmc_iter = 4_000
        args.mcmc_burn = 1_000
        args.mcmc_thin = 2

        args.output = (
            "results/"
            "performance_guarded_"
            "structure_benchmark_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )