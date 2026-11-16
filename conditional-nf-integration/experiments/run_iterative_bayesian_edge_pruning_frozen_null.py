import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import invgamma
from scipy.special import logsumexp

from nf.conditional_cubic_1d import ConditionalCubicFlow1D

from run_copula_bin_sweep import (
    sample_target,
    seed_everything,
)


# ============================================================
# CONFIG
# ============================================================

COND_DIM = 3
DIM = 8

N_BINS = 32
HIDDEN = 128

MARGINAL_STEPS = 1200
MASKED_STEPS = 1800

BATCH_SIZE = 4096
N_EVAL = 100_000

LR_MARGINAL = 1e-4
LR_MASKED = 1e-4

GRAD_CLIP = 10.0

KEEP_PROB = 0.70

DEFAULT_BENCHMARKS = [
    "single_gaussian",
    "curved_ridge",
    "sharp_broad",
]

DEFAULT_SEEDS = [
    0,
    1,
    2,
]

MAX_ROUNDS = 10

PIP_REMOVE_THRESHOLD = 0.5

# ------------------------------------------------------------
# Bayesian MCMC
# ------------------------------------------------------------

MCMC_ITER = 12_000
MCMC_BURN = 3_000
MCMC_THIN = 3

RHO_A = 1.0
RHO_B = 1.0

BAYES_RANDOM_SEED = 12345

# ------------------------------------------------------------
# Priors
# ------------------------------------------------------------

SIGMA0_ALPHA = 2.0
SIGMA0_BETA = 1e-6

SIGMA1_ALPHA = 2.0
SIGMA1_BETA = 1e-4

MU_PRIOR_MEAN = 0.0
MU_PRIOR_SD = 1.0

OUTPUT_DIR = Path(
    "results/"
    "iterative_bayesian_edge_pruning_frozen_null"
)


# ============================================================
# Generic utilities
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


def full_ar_graph(
    dim,
):
    return {
        j: list(range(j))
        for j in range(dim)
    }


def copy_graph(
    graph,
):
    return {
        j: list(parents)
        for j, parents
        in graph.items()
    }


def count_edges(
    graph,
):
    return sum(
        len(parents)
        for parents
        in graph.values()
    )


def graph_as_string(
    graph,
):
    return json.dumps(
        {
            str(j): graph[j]
            for j in range(DIM)
        }
    )


# ============================================================
# Bayesian helpers
# ============================================================

def normal_logpdf(
    x,
    mean,
    variance,
):

    variance = max(
        float(variance),
        1e-12,
    )

    return (
        -0.5
        * (
            math.log(
                2.0
                * math.pi
                * variance
            )
            +
            (
                (x - mean) ** 2
                / variance
            )
        )
    )


def sample_truncated_positive_normal(
    mean,
    sd,
    rng,
):

    sd = max(
        float(sd),
        1e-12,
    )

    for _ in range(
        100_000
    ):

        value = rng.normal(
            mean,
            sd,
        )

        if value > 0.0:
            return value

    return max(
        mean,
        1e-8,
    )


# ============================================================
# Bayesian MCMC
#
# IMPORTANT:
#
# fixed_sigma0=None:
#     infer null scale normally.
#
# fixed_sigma0=float:
#     hold null standard deviation fixed.
#
# ============================================================

def run_mcmc(
    Y,
    rng,
    n_iter,
    burn_in,
    thin,
    rho_a,
    rho_b,
    fixed_sigma0=None,
):

    n_edges, n_rep = (
        Y.shape
    )

    # ========================================================
    # Initialization
    # ========================================================

    edge_means = (
        Y.mean(
            axis=1
        )
    )

    positive = (
        edge_means[
            edge_means > 0
        ]
    )

    if len(
        positive
    ) > 0:

        median_positive = (
            np.median(
                positive
            )
        )

        z = (
            edge_means
            > median_positive
        ).astype(int)

    else:

        z = np.zeros(
            n_edges,
            dtype=int,
        )

    if (
        z.sum() == 0
        and
        n_edges > 0
    ):

        z[
            np.argmax(
                edge_means
            )
        ] = 1

    rho = 0.5

    active_means = (
        edge_means[
            z == 1
        ]
    )

    if len(
        active_means
    ) > 0:

        mu1 = max(
            np.mean(
                active_means
            ),
            1e-3,
        )

    else:

        mu1 = 1e-3

    if fixed_sigma0 is None:

        if np.any(
            z == 0
        ):

            sigma0_sq = max(
                np.var(
                    Y[
                        z == 0
                    ]
                ),
                1e-6,
            )

        else:

            sigma0_sq = max(
                np.var(Y),
                1e-6,
            )

    else:

        sigma0_sq = max(
            float(
                fixed_sigma0
            ) ** 2,
            1e-12,
        )

    if np.any(
        z == 1
    ):

        sigma1_sq = max(
            np.var(
                Y[
                    z == 1
                ]
            ),
            1e-6,
        )

    else:

        sigma1_sq = max(
            np.var(Y),
            1e-6,
        )

    # ========================================================
    # Posterior storage
    # ========================================================

    z_sum = np.zeros(
        n_edges,
        dtype=float,
    )

    rho_samples = []
    mu1_samples = []

    sigma0_samples = []
    sigma1_samples = []

    edge_count_samples = []

    kept = 0

    # ========================================================
    # MCMC
    # ========================================================

    for iteration in range(
        n_iter
    ):

        # ====================================================
        # 1. Sample Z_e
        # ====================================================

        for e in range(
            n_edges
        ):

            observations = (
                Y[e]
            )

            log_lik_0 = 0.0
            log_lik_1 = 0.0

            for value in (
                observations
            ):

                log_lik_0 += (
                    normal_logpdf(
                        value,
                        0.0,
                        sigma0_sq,
                    )
                )

                log_lik_1 += (
                    normal_logpdf(
                        value,
                        mu1,
                        sigma1_sq,
                    )
                )

            rho_safe = min(
                max(
                    rho,
                    1e-10,
                ),
                1.0 - 1e-10,
            )

            log_p0 = (
                math.log(
                    1.0
                    - rho_safe
                )
                +
                log_lik_0
            )

            log_p1 = (
                math.log(
                    rho_safe
                )
                +
                log_lik_1
            )

            denom = logsumexp(
                [
                    log_p0,
                    log_p1,
                ]
            )

            inclusion_prob = (
                math.exp(
                    log_p1
                    - denom
                )
            )

            z[e] = int(
                rng.random()
                < inclusion_prob
            )

        # ====================================================
        # 2. Sample rho
        # ====================================================

        n_active = int(
            z.sum()
        )

        rho = rng.beta(
            rho_a
            + n_active,

            rho_b
            + n_edges
            - n_active,
        )

        # ====================================================
        # 3. sigma0
        #
        # Either infer or keep frozen.
        # ====================================================

        if fixed_sigma0 is None:

            null_values = (
                Y[
                    z == 0
                ]
                .reshape(-1)
            )

            ss0 = float(
                np.sum(
                    null_values ** 2
                )
            )

            alpha0_post = (
                SIGMA0_ALPHA
                + 0.5
                * len(
                    null_values
                )
            )

            beta0_post = (
                SIGMA0_BETA
                + 0.5
                * ss0
            )

            sigma0_sq = invgamma.rvs(
                a=alpha0_post,
                scale=beta0_post,
                random_state=rng,
            )

            sigma0_sq = max(
                float(
                    sigma0_sq
                ),
                1e-12,
            )

        else:

            sigma0_sq = (
                float(
                    fixed_sigma0
                )
                ** 2
            )

        # ====================================================
        # 4. Sample mu1
        # ====================================================

        active_values = (
            Y[
                z == 1
            ]
            .reshape(-1)
        )

        prior_precision = (
            1.0
            /
            (
                MU_PRIOR_SD
                ** 2
            )
        )

        if len(
            active_values
        ) > 0:

            likelihood_precision = (
                len(
                    active_values
                )
                /
                sigma1_sq
            )

            posterior_precision = (
                prior_precision
                +
                likelihood_precision
            )

            posterior_variance = (
                1.0
                /
                posterior_precision
            )

            posterior_mean = (
                (
                    MU_PRIOR_MEAN
                    * prior_precision
                )
                +
                (
                    active_values.sum()
                    /
                    sigma1_sq
                )
            ) / posterior_precision

        else:

            posterior_mean = (
                MU_PRIOR_MEAN
            )

            posterior_variance = (
                MU_PRIOR_SD ** 2
            )

        mu1 = (
            sample_truncated_positive_normal(
                posterior_mean,
                math.sqrt(
                    posterior_variance
                ),
                rng,
            )
        )

        # ====================================================
        # 5. Sample sigma1
        # ====================================================

        if len(
            active_values
        ) > 0:

            residuals = (
                active_values
                - mu1
            )

            ss1 = float(
                np.sum(
                    residuals ** 2
                )
            )

        else:

            ss1 = 0.0

        alpha1_post = (
            SIGMA1_ALPHA
            + 0.5
            * len(
                active_values
            )
        )

        beta1_post = (
            SIGMA1_BETA
            + 0.5
            * ss1
        )

        sigma1_sq = invgamma.rvs(
            a=alpha1_post,
            scale=beta1_post,
            random_state=rng,
        )

        sigma1_sq = max(
            float(
                sigma1_sq
            ),
            1e-12,
        )

        # ====================================================
        # Store
        # ====================================================

        if (
            iteration >= burn_in
            and
            (
                iteration
                - burn_in
            ) % thin == 0
        ):

            z_sum += z

            rho_samples.append(
                rho
            )

            mu1_samples.append(
                mu1
            )

            sigma0_samples.append(
                math.sqrt(
                    sigma0_sq
                )
            )

            sigma1_samples.append(
                math.sqrt(
                    sigma1_sq
                )
            )

            edge_count_samples.append(
                n_active
            )

            kept += 1

        if (
            iteration + 1
        ) % 5000 == 0:

            print(
                f"    iteration "
                f"{iteration+1:6d}/{n_iter} "
                f"| active={n_active:2d} "
                f"| rho={rho:.3f} "
                f"| mu1={mu1:.5f} "
                f"| sigma0="
                f"{math.sqrt(sigma0_sq):.6f} "
                f"| sigma1="
                f"{math.sqrt(sigma1_sq):.6f}"
            )

    # ========================================================
    # Posterior summary
    # ========================================================

    pip = (
        z_sum
        / kept
    )

    return {
        "pip":
            pip,

        "rho_mean":
            float(
                np.mean(
                    rho_samples
                )
            ),

        "rho_sd":
            float(
                np.std(
                    rho_samples
                )
            ),

        "mu1_mean":
            float(
                np.mean(
                    mu1_samples
                )
            ),

        "mu1_sd":
            float(
                np.std(
                    mu1_samples
                )
            ),

        "sigma0_mean":
            float(
                np.mean(
                    sigma0_samples
                )
            ),

        "sigma1_mean":
            float(
                np.mean(
                    sigma1_samples
                )
            ),

        "expected_edges":
            float(
                np.mean(
                    edge_count_samples
                )
            ),

        "edge_count_sd":
            float(
                np.std(
                    edge_count_samples
                )
            ),

        "n_posterior_samples":
            kept,
    }


# ============================================================
# Marginal bank
# ============================================================

class MarginalBank(
    nn.Module
):

    def __init__(
        self,
        dim,
    ):

        super().__init__()

        self.dim = dim

        self.marginals = nn.ModuleList(
            [
                ConditionalCubicFlow1D(
                    cond_dim=COND_DIM,
                    hidden=HIDDEN,
                    n_bins=N_BINS,
                )
                for _ in range(dim)
            ]
        )

    @torch.no_grad()
    def x_to_u(
        self,
        x,
        cond,
    ):

        values = []

        for j in range(
            self.dim
        ):

            uj, _ = (
                self.marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            values.append(
                uj.clamp(
                    0.0,
                    1.0,
                )
            )

        return torch.cat(
            values,
            dim=1,
        )


def train_marginals(
    bank,
    benchmark,
    device,
    steps,
    batch_size,
):

    optimizer = torch.optim.Adam(
        bank.parameters(),
        lr=LR_MARGINAL,
    )

    recent = []

    bank.train()

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
# Masked child model
# ============================================================

class IterativeMaskedChild(
    nn.Module
):

    def __init__(
        self,
        child,
    ):

        super().__init__()

        self.child = child

        context_dim = (
            2 * child
            + COND_DIM
        )

        self.flow = (
            ConditionalCubicFlow1D(
                cond_dim=context_dim,
                hidden=HIDDEN,
                n_bins=N_BINS,
            )
        )

    def make_context(
        self,
        u,
        cond,
        mask,
    ):

        parent_values = (
            u[
                :,
                :self.child
            ]
            - 0.5
        )

        masked_values = (
            parent_values
            * mask
        )

        return torch.cat(
            [
                masked_values,
                mask,
                cond,
            ],
            dim=1,
        )

    def log_prob(
        self,
        child_u,
        u,
        cond,
        mask,
    ):

        context = self.make_context(
            u,
            cond,
            mask,
        )

        _, forward_logdet = (
            self.flow.inverse(
                child_u,
                context,
            )
        )

        return (
            -forward_logdet
            .squeeze(-1)
        )


# ============================================================
# Masks
# ============================================================

def active_mask_vector(
    child,
    active_parents,
    device,
):

    mask = torch.zeros(
        child,
        device=device,
    )

    for parent in (
        active_parents
    ):

        mask[
            parent
        ] = 1.0

    return mask


def sample_training_mask(
    batch_size,
    child,
    active_parents,
    keep_prob,
    device,
):

    active = (
        active_mask_vector(
            child,
            active_parents,
            device,
        )
        .unsqueeze(0)
        .expand(
            batch_size,
            -1,
        )
    )

    dropout = torch.bernoulli(
        torch.full(
            (
                batch_size,
                child,
            ),
            keep_prob,
            device=device,
        )
    )

    return (
        active
        * dropout
    )


# ============================================================
# Train masked child
# ============================================================

def train_masked_child(
    marginal_bank,
    benchmark,
    child,
    active_parents,
    device,
    steps,
    batch_size,
    keep_prob,
    seed,
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_MASKED,
    )

    model.train()

    for step in range(
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

            u = (
                marginal_bank
                .x_to_u(
                    x,
                    cond,
                )
            )

        child_u = (
            u[
                :,
                child:child+1
            ]
        )

        mask = sample_training_mask(
            batch_size,
            child,
            active_parents,
            keep_prob,
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
    marginal_bank,
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

    u = (
        marginal_bank
        .x_to_u(
            x,
            cond,
        )
    )

    child_u = (
        u[
            :,
            child:child+1
        ]
    )

    # ========================================================
    # Baseline:
    # all CURRENT active parents are ON.
    # ========================================================

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

    logq_on = (
        model.log_prob(
            child_u,
            u,
            cond,
            base_mask,
        )
    )

    rows = []

    for parent in (
        active_parents
    ):

        edge_off_mask = (
            base_mask.clone()
        )

        edge_off_mask[
            :,
            parent
        ] = 0.0

        logq_off = (
            model.log_prob(
                child_u,
                u,
                cond,
                edge_off_mask,
            )
        )

        delta = (
            logq_on
            - logq_off
        )

        delta_mean = (
            delta.mean()
            .item()
        )

        delta_std = (
            delta.std(
                unbiased=True
            )
            .item()
        )

        delta_se = (
            delta_std
            /
            math.sqrt(
                n_eval
            )
        )

        rows.append(
            {
                "parent":
                    parent,

                "child":
                    child,

                "delta_loglik":
                    delta_mean,

                "delta_std":
                    delta_std,

                "delta_se":
                    delta_se,

                "is_markov1_edge":
                    int(
                        parent
                        == child - 1
                    ),
            }
        )

    return rows


# ============================================================
# Collect evidence over seeds
# ============================================================

def collect_round_evidence(
    benchmark,
    round_id,
    graph,
    marginal_states,
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
            f"    Evidence seed {seed}"
        )

        for child in range(
            1,
            DIM,
        ):

            active_parents = (
                graph[
                    child
                ]
            )

            if len(
                active_parents
            ) == 0:

                continue

            print(
                f"      child {child}: "
                f"{active_parents}"
            )

            model_seed = (
                seed * 100000
                + round_id * 1000
                + child
            )

            model = train_masked_child(
                marginal_bank=
                    bank,

                benchmark=
                    benchmark,

                child=
                    child,

                active_parents=
                    active_parents,

                device=
                    device,

                steps=
                    args.masked_steps,

                batch_size=
                    args.batch_size,

                keep_prob=
                    args.keep_prob,

                seed=
                    model_seed,
            )

            model.eval()

            child_rows = (
                compute_child_deltas(
                    model=
                        model,

                    marginal_bank=
                        bank,

                    benchmark=
                        benchmark,

                    child=
                        child,

                    active_parents=
                        active_parents,

                    device=
                        device,

                    n_eval=
                        args.n_eval,
                )
            )

            for row in (
                child_rows
            ):

                row.update(
                    {
                        "benchmark":
                            benchmark,

                        "round":
                            round_id,

                        "seed":
                            seed,
                    }
                )

                rows.append(
                    row
                )

            del model

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        del bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return pd.DataFrame(
        rows
    )


# ============================================================
# Evidence matrix
# ============================================================

def evidence_to_matrix(
    evidence_df,
):

    edges_df = (
        evidence_df[
            [
                "parent",
                "child",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "child",
                "parent",
            ]
        )
    )

    edges = [
        (
            int(row.parent),
            int(row.child),
        )
        for row
        in edges_df.itertuples()
    ]

    seeds = sorted(
        evidence_df[
            "seed"
        ]
        .unique()
    )

    edge_index = {
        edge: i
        for i, edge
        in enumerate(
            edges
        )
    }

    seed_index = {
        seed: i
        for i, seed
        in enumerate(
            seeds
        )
    }

    Y = np.full(
        (
            len(edges),
            len(seeds),
        ),
        np.nan,
    )

    for row in (
        evidence_df
        .itertuples()
    ):

        e = edge_index[
            (
                int(
                    row.parent
                ),
                int(
                    row.child
                ),
            )
        ]

        s = seed_index[
            row.seed
        ]

        Y[
            e,
            s
        ] = (
            row.delta_loglik
        )

    if np.isnan(
        Y
    ).any():

        raise RuntimeError(
            "Missing delta observation "
            "for an active edge."
        )

    return (
        edges,
        Y,
    )


# ============================================================
# Bayesian round
# ============================================================

def bayesian_round(
    evidence_df,
    args,
    fixed_sigma0=None,
):

    edges, Y = (
        evidence_to_matrix(
            evidence_df
        )
    )

    round_id = int(
        evidence_df[
            "round"
        ]
        .iloc[0]
    )

    rng = np.random.default_rng(
        args.bayes_seed
        + round_id
    )

    posterior = run_mcmc(
        Y=Y,

        rng=rng,

        n_iter=
            args.mcmc_iter,

        burn_in=
            args.mcmc_burn,

        thin=
            args.mcmc_thin,

        rho_a=
            args.rho_a,

        rho_b=
            args.rho_b,

        fixed_sigma0=
            fixed_sigma0,
    )

    means = (
        Y.mean(
            axis=1
        )
    )

    if Y.shape[
        1
    ] > 1:

        sds = (
            Y.std(
                axis=1,
                ddof=1,
            )
        )

    else:

        sds = np.zeros(
            Y.shape[0]
        )

    rows = []

    for edge_id, (
        parent,
        child,
    ) in enumerate(
        edges
    ):

        rows.append(
            {
                "parent":
                    parent,

                "child":
                    child,

                "delta_mean":
                    float(
                        means[
                            edge_id
                        ]
                    ),

                "delta_sd_across_seeds":
                    float(
                        sds[
                            edge_id
                        ]
                    ),

                "pip":
                    float(
                        posterior[
                            "pip"
                        ][
                            edge_id
                        ]
                    ),

                "is_markov1_edge":
                    int(
                        parent
                        == child - 1
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        ),
        posterior,
    )


# ============================================================
# Conservative pruning
#
# Same rule as before:
#
# AT MOST one removal per child per round.
# ============================================================

def choose_removals(
    posterior_df,
    pip_threshold,
):

    removals = []

    children = sorted(
        posterior_df[
            "child"
        ]
        .unique()
    )

    for child in (
        children
    ):

        current = (
            posterior_df[
                posterior_df[
                    "child"
                ] == child
            ]
            .copy()
        )

        candidates = current[
            current[
                "pip"
            ] < pip_threshold
        ]

        if len(
            candidates
        ) == 0:

            continue

        candidates = (
            candidates
            .sort_values(
                [
                    "pip",
                    "delta_mean",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
        )

        chosen = (
            candidates
            .iloc[0]
        )

        removals.append(
            (
                int(
                    chosen[
                        "parent"
                    ]
                ),
                int(
                    child
                ),
            )
        )

    return removals


def apply_removals(
    graph,
    removals,
):

    new_graph = (
        copy_graph(
            graph
        )
    )

    for parent, child in (
        removals
    ):

        new_graph[
            child
        ].remove(
            parent
        )

    return new_graph


# ============================================================
# Train marginal states once
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
            f"for seed {seed}..."
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

        train_marginals(
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
            value
            .detach()
            .cpu()
            .clone()

            for key, value
            in bank
            .state_dict()
            .items()
        }

        del bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return states


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
        "=" * 100
    )

    print(
        "ITERATIVE BAYESIAN PRUNING "
        f"WITH FROZEN NULL: {benchmark}"
    )

    print(
        "=" * 100
    )

    marginal_states = (
        train_all_marginal_states(
            benchmark,
            args,
            device,
        )
    )

    graph = (
        full_ar_graph(
            DIM
        )
    )

    evidence_history = []
    posterior_history = []
    graph_history = []

    # ========================================================
    # Critical modification:
    #
    # round 0 learns sigma0.
    # round >= 1 reuses it.
    # ========================================================

    frozen_sigma0 = None

    for round_id in range(
        args.max_rounds
    ):

        print()
        print(
            "#" * 100
        )

        print(
            f"ROUND {round_id} "
            f"| active edges="
            f"{count_edges(graph)}"
        )

        print(
            graph
        )

        if frozen_sigma0 is None:

            print(
                "Null scale: "
                "will be inferred this round"
            )

        else:

            print(
                "Frozen null sigma0: "
                f"{frozen_sigma0:.8f}"
            )

        print(
            "#" * 100
        )

        graph_before = (
            copy_graph(
                graph
            )
        )

        # ====================================================
        # 1. Recompute Delta on current graph
        # ====================================================

        evidence_df = (
            collect_round_evidence(
                benchmark=
                    benchmark,

                round_id=
                    round_id,

                graph=
                    graph,

                marginal_states=
                    marginal_states,

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

        evidence_history.append(
            evidence_df
        )

        # ====================================================
        # 2. Bayesian inference
        # ====================================================

        posterior_df, posterior = (
            bayesian_round(
                evidence_df,
                args,
                fixed_sigma0=
                    frozen_sigma0,
            )
        )

        # ====================================================
        # 3. Freeze round-0 null
        # ====================================================

        if round_id == 0:

            frozen_sigma0 = float(
                posterior[
                    "sigma0_mean"
                ]
            )

            print()
            print(
                "=" * 70
            )

            print(
                "FREEZING ROUND-0 NULL SCALE"
            )

            print(
                f"sigma0^(0) = "
                f"{frozen_sigma0:.8f}"
            )

            print(
                "=" * 70
            )

        posterior_df[
            "benchmark"
        ] = benchmark

        posterior_df[
            "round"
        ] = round_id

        posterior_df[
            "sigma0_reference"
        ] = frozen_sigma0

        posterior_history.append(
            posterior_df
        )

        # ====================================================
        # 4. Choose conservative removals
        # ====================================================

        removals = (
            choose_removals(
                posterior_df,
                args.pip_threshold,
            )
        )

        print()
        print(
            "Bayesian round summary:"
        )

        print(
            f"  rho mean             = "
            f"{posterior['rho_mean']:.4f}"
        )

        print(
            f"  inferred E[edges]    = "
            f"{posterior['expected_edges']:.2f}"
        )

        print(
            f"  current graph edges  = "
            f"{count_edges(graph)}"
        )

        print(
            f"  sigma0 used          = "
            f"{posterior['sigma0_mean']:.8f}"
        )

        print(
            f"  frozen sigma0        = "
            f"{frozen_sigma0:.8f}"
        )

        print(
            f"  removals             = "
            f"{removals}"
        )

        print()

        print(
            posterior_df
            .sort_values(
                [
                    "child",
                    "pip",
                ]
            )
            [
                [
                    "parent",
                    "child",
                    "delta_mean",
                    "delta_sd_across_seeds",
                    "pip",
                    "is_markov1_edge",
                ]
            ]
            .to_string(
                index=False
            )
        )

        # ====================================================
        # Save history
        # ====================================================

        graph_history.append(
            {
                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "edges_before":
                    count_edges(
                        graph_before
                    ),

                "posterior_expected_edges":
                    posterior[
                        "expected_edges"
                    ],

                "rho_mean":
                    posterior[
                        "rho_mean"
                    ],

                "sigma0_used":
                    posterior[
                        "sigma0_mean"
                    ],

                "frozen_sigma0":
                    frozen_sigma0,

                "n_removed":
                    len(
                        removals
                    ),

                "removed_edges":
                    str(
                        removals
                    ),

                "graph_before":
                    graph_as_string(
                        graph_before
                    ),
            }
        )

        pd.concat(
            evidence_history,
            ignore_index=True,
        ).to_csv(
            output_dir
            / (
                f"{benchmark}"
                "_round_evidence.csv"
            ),
            index=False,
        )

        pd.concat(
            posterior_history,
            ignore_index=True,
        ).to_csv(
            output_dir
            / (
                f"{benchmark}"
                "_round_posteriors.csv"
            ),
            index=False,
        )

        pd.DataFrame(
            graph_history
        ).to_csv(
            output_dir
            / (
                f"{benchmark}"
                "_graph_history.csv"
            ),
            index=False,
        )

        # ====================================================
        # 5. Stop if no removals
        # ====================================================

        if len(
            removals
        ) == 0:

            print()
            print(
                "STOP: Bayesian selector "
                "does not recommend "
                "further pruning."
            )

            break

        # ====================================================
        # 6. Update graph
        # ====================================================

        graph = apply_removals(
            graph,
            removals,
        )

    # ========================================================
    # Final graph
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        f"FINAL GRAPH: "
        f"{benchmark}"
    )

    print(
        "=" * 90
    )

    print(
        graph
    )

    print(
        f"Final edge count: "
        f"{count_edges(graph)}"
    )

    print(
        f"Frozen null sigma0: "
        f"{frozen_sigma0:.8f}"
    )

    final_rows = []

    for child in range(
        DIM
    ):

        for parent in (
            graph[
                child
            ]
        ):

            final_rows.append(
                {
                    "benchmark":
                        benchmark,

                    "parent":
                        parent,

                    "child":
                        child,

                    "is_markov1_edge":
                        int(
                            parent
                            == child - 1
                        ),
                }
            )

    return (
        graph,
        final_rows,
        graph_history,
        frozen_sigma0,
    )


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

    print()

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_final_edges = []
    all_summaries = []

    for benchmark in (
        args.benchmarks
    ):

        (
            final_graph,
            final_rows,
            history,
            frozen_sigma0,
        ) = run_benchmark(
            benchmark,
            args,
            device,
            output_dir,
        )

        all_final_edges.extend(
            final_rows
        )

        all_summaries.append(
            {
                "benchmark":
                    benchmark,

                "final_edge_count":
                    count_edges(
                        final_graph
                    ),

                "frozen_sigma0":
                    frozen_sigma0,

                "final_graph":
                    graph_as_string(
                        final_graph
                    ),

                "n_rounds":
                    len(
                        history
                    ),
            }
        )

        pd.DataFrame(
            all_final_edges
        ).to_csv(
            output_dir
            / "final_graph_edges.csv",
            index=False,
        )

        pd.DataFrame(
            all_summaries
        ).to_csv(
            output_dir
            / "summary.csv",
            index=False,
        )

    print()
    print(
        "=" * 120
    )

    print(
        "ITERATIVE BAYESIAN EDGE PRUNING "
        "WITH FROZEN NULL COMPLETE"
    )

    print(
        "=" * 120
    )

    print()

    print(
        pd.DataFrame(
            all_summaries
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        f"  "
        f"{output_dir / 'final_graph_edges.csv'}"
    )

    print(
        f"  "
        f"{output_dir / 'summary.csv'}"
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
            DEFAULT_BENCHMARKS,
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
            str(
                OUTPUT_DIR
            ),
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
            1,
            2,
        ]

        args.marginal_steps = 100
        args.masked_steps = 300

        args.batch_size = 1024
        args.n_eval = 10_000

        args.max_rounds = 3

        args.mcmc_iter = 4_000
        args.mcmc_burn = 1_000
        args.mcmc_thin = 2

        args.output = (
            "results/"
            "iterative_bayesian_edge_pruning_"
            "frozen_null_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )