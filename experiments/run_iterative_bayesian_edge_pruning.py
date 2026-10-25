import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nf.conditional_cubic_1d import ConditionalCubicFlow1D

from run_copula_bin_sweep import (
    sample_target,
    seed_everything,
)

# Reuse the Bayesian mixture inference we already validated.
from run_bayesian_delta_edge_selection import run_mcmc


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

# Posterior inclusion probability decision.
# This is NOT a raw-delta threshold.
PIP_REMOVE_THRESHOLD = 0.5

# Bayesian MCMC
MCMC_ITER = 12_000
MCMC_BURN = 3_000
MCMC_THIN = 3

RHO_A = 1.0
RHO_B = 1.0

BAYES_RANDOM_SEED = 12345

OUTPUT_DIR = Path(
    "results/iterative_bayesian_edge_pruning"
)


# ============================================================
# Utilities
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
# Marginals
# ============================================================

class MarginalBank(nn.Module):

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
                f"  marginal "
                f"{step+1:4d}/{steps} "
                f"loss="
                f"{np.mean(recent[-100:]):.6f}"
            )


# ============================================================
# Masked child model
#
# IMPORTANT:
# Architecture always has slots for all previous coordinates.
#
# A separate active_mask determines which graph edges
# currently exist.
#
# Removed edges can therefore NEVER accidentally reactivate.
# ============================================================

class IterativeMaskedChild(nn.Module):

    def __init__(
        self,
        child,
    ):
        super().__init__()

        self.child = child

        # context:
        #
        # masked parent values : child
        # mask indicators      : child
        # external c           : COND_DIM
        #
        context_dim = (
            2 * child
            + COND_DIM
        )

        self.flow = ConditionalCubicFlow1D(
            cond_dim=context_dim,
            hidden=HIDDEN,
            n_bins=N_BINS,
        )

    def make_context(
        self,
        u,
        cond,
        mask,
    ):

        parents = (
            u[:, :self.child]
            - 0.5
        )

        masked_values = (
            parents * mask
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
# Current graph mask
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
        mask[parent] = 1.0

    return mask


def sample_training_mask(
    batch_size,
    child,
    active_parents,
    keep_prob,
    device,
):
    """
    Random dropout ONLY among currently active parents.

    Already-pruned parents stay zero permanently.
    """

    active = active_mask_vector(
        child,
        active_parents,
        device,
    )

    active = active.unsqueeze(
        0
    ).expand(
        batch_size,
        -1,
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
        active * dropout
    )


# ============================================================
# Train masked child for CURRENT graph
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

    recent = []

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

            u = marginal_bank.x_to_u(
                x,
                cond,
            )

        child_u = (
            u[:, child:child+1]
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

        recent.append(
            loss.item()
        )

    return model


# ============================================================
# Recompute Delta on CURRENT graph
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

    u = marginal_bank.x_to_u(
        x,
        cond,
    )

    child_u = (
        u[:, child:child+1]
    )

    # --------------------------------------------
    # Baseline = all CURRENT parents ON.
    # Removed graph edges remain zero.
    # --------------------------------------------

    base_mask = active_mask_vector(
        child,
        active_parents,
        device,
    )

    base_mask = (
        base_mask
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

        edge_off_mask = (
            base_mask.clone()
        )

        edge_off_mask[
            :,
            parent
        ] = 0.0

        logq_off = model.log_prob(
            child_u,
            u,
            cond,
            edge_off_mask,
        )

        delta = (
            logq_on
            - logq_off
        )

        delta_mean = (
            delta.mean().item()
        )

        delta_std = (
            delta.std(
                unbiased=True
            ).item()
        )

        delta_se = (
            delta_std
            / math.sqrt(
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
# Collect one round of evidence across seeds
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

        # Restore corresponding marginal bank
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
            f"\n  Evidence seed {seed}"
        )

        for child in range(
            1,
            DIM,
        ):

            active_parents = (
                graph[child]
            )

            if len(
                active_parents
            ) == 0:
                continue

            print(
                f"    child {child}: "
                f"{active_parents}"
            )

            model_seed = (
                seed * 100000
                + round_id * 1000
                + child
            )

            model = (
                train_masked_child(
                    marginal_bank=bank,
                    benchmark=benchmark,
                    child=child,
                    active_parents=
                        active_parents,
                    device=device,
                    steps=
                        args.masked_steps,
                    batch_size=
                        args.batch_size,
                    keep_prob=
                        args.keep_prob,
                    seed=model_seed,
                )
            )

            model.eval()

            child_rows = (
                compute_child_deltas(
                    model=model,
                    marginal_bank=bank,
                    benchmark=benchmark,
                    child=child,
                    active_parents=
                        active_parents,
                    device=device,
                    n_eval=args.n_eval,
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
# Convert round evidence to matrix
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
        for row in (
            edges_df.itertuples()
        )
    ]

    seeds = sorted(
        evidence_df[
            "seed"
        ].unique()
    )

    edge_index = {
        edge: i
        for i, edge
        in enumerate(edges)
    }

    seed_index = {
        seed: i
        for i, seed
        in enumerate(seeds)
    }

    Y = np.full(
        (
            len(edges),
            len(seeds),
        ),
        np.nan,
    )

    for row in (
        evidence_df.itertuples()
    ):

        e = edge_index[
            (
                int(row.parent),
                int(row.child),
            )
        ]

        s = seed_index[
            row.seed
        ]

        Y[e, s] = (
            row.delta_loglik
        )

    if np.isnan(
        Y
    ).any():

        raise RuntimeError(
            "Missing delta observations "
            "for active edge."
        )

    return (
        edges,
        Y,
    )


# ============================================================
# Bayesian selection for CURRENT round
# ============================================================

def bayesian_round(
    evidence_df,
    args,
):

    edges, Y = (
        evidence_to_matrix(
            evidence_df
        )
    )

    rng = np.random.default_rng(
        args.bayes_seed
        + int(
            evidence_df[
                "round"
            ].iloc[0]
        )
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
    )

    means = (
        Y.mean(
            axis=1
        )
    )

    sds = (
        Y.std(
            axis=1,
            ddof=1,
        )
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
                    means[
                        edge_id
                    ],

                "delta_sd_across_seeds":
                    sds[
                        edge_id
                    ],

                "pip":
                    posterior[
                        "pip"
                    ][edge_id],

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
# ============================================================

def choose_removals(
    posterior_df,
    pip_threshold,
):
    """
    Remove AT MOST ONE edge per child.

    Among PIP < threshold edges for that child,
    remove the edge with the smallest PIP.

    Tie-break using delta_mean.
    """

    removals = []

    children = sorted(
        posterior_df[
            "child"
        ].unique()
    )

    for child in children:

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
            candidates.iloc[0]
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


# ============================================================
# Apply pruning
# ============================================================

def apply_removals(
    graph,
    removals,
):

    new_graph = copy_graph(
        graph
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
# Marginals once per benchmark/seed
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
            value.detach()
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
        f"ITERATIVE BAYESIAN PRUNING: "
        f"{benchmark}"
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

    graph = full_ar_graph(
        DIM
    )

    evidence_history = []
    posterior_history = []
    graph_history = []

    for round_id in range(
        args.max_rounds
    ):

        print()
        print(
            "#" * 90
        )

        print(
            f"ROUND {round_id} "
            f"| active edges="
            f"{count_edges(graph)}"
        )

        print(
            graph
        )

        print(
            "#" * 90
        )

        graph_before = copy_graph(
            graph
        )

        # --------------------------------------------
        # Delta for CURRENT graph
        # --------------------------------------------

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

        # --------------------------------------------
        # Bayesian inference
        # --------------------------------------------

        posterior_df, posterior = (
            bayesian_round(
                evidence_df,
                args,
            )
        )

        posterior_df[
            "benchmark"
        ] = benchmark

        posterior_df[
            "round"
        ] = round_id

        posterior_history.append(
            posterior_df
        )

        # --------------------------------------------
        # Conservative pruning
        # --------------------------------------------

        removals = choose_removals(
            posterior_df,
            args.pip_threshold,
        )

        print()
        print(
            "Bayesian round summary:"
        )

        print(
            f"  rho mean       = "
            f"{posterior['rho_mean']:.4f}"
        )

        print(
            f"  E[active edges]= "
            f"{posterior['expected_edges']:.2f}"
        )

        print(
            f"  current edges  = "
            f"{count_edges(graph)}"
        )

        print(
            f"  removals       = "
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
                    "pip",
                    "is_markov1_edge",
                ]
            ]
            .to_string(
                index=False
            )
        )

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

        # Save every round.
        pd.concat(
            evidence_history,
            ignore_index=True,
        ).to_csv(
            output_dir
            / f"{benchmark}_round_evidence.csv",
            index=False,
        )

        pd.concat(
            posterior_history,
            ignore_index=True,
        ).to_csv(
            output_dir
            / f"{benchmark}_round_posteriors.csv",
            index=False,
        )

        pd.DataFrame(
            graph_history
        ).to_csv(
            output_dir
            / f"{benchmark}_graph_history.csv",
            index=False,
        )

        if len(
            removals
        ) == 0:

            print()
            print(
                "STOP: Bayesian selector "
                "does not recommend any "
                "further removal."
            )

            break

        graph = apply_removals(
            graph,
            removals,
        )

    # --------------------------------------------------------
    # Final graph
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        f"FINAL GRAPH: {benchmark}"
    )

    print(
        "=" * 80
    )

    print(
        graph
    )

    print(
        f"Final edge count: "
        f"{count_edges(graph)}"
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

    print(
        f"\nDevice: {device}\n"
    )

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
        "=" * 110
    )

    print(
        "ITERATIVE BAYESIAN EDGE PRUNING COMPLETE"
    )

    print(
        "=" * 110
    )

    print()

    print(
        pd.DataFrame(
            all_summaries
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
        default=str(
            OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

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
            "iterative_bayesian_edge_pruning_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )