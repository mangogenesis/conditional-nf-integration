import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nf.conditional_cubic_1d import ConditionalCubicFlow1D

from run_copula_bin_sweep import (
    sample_target,
    target_logpdf,
    seed_everything,
)


# ============================================================
# Configuration
# ============================================================

COND_DIM = 3
DIM = 8

N_BINS = 32
HIDDEN = 128

MARGINAL_STEPS = 1200
COPULA_STEPS = 1800

BATCH_SIZE = 4096

LR_MARGINAL = 1e-4
LR_COPULA = 1e-4

GRAD_CLIP = 10.0

N_EVAL = 100_000
N_KL = 50_000

DEFAULT_BENCHMARKS = [
    "single_gaussian",
    "curved_ridge",
    "sharp_broad",
]

DEFAULT_GRAPH_FILE = (
    "results/"
    "bayesian_delta_edge_selection/"
    "median_probability_graph.csv"
)

OUTPUT_DIR = Path(
    "results/bayesian_selected_copula_benchmark"
)


TEST_CONDITIONS = torch.tensor(
    [
        [0.10, 0.10, 0.10],
        [0.10, 0.10, 0.90],
        [0.10, 0.90, 0.10],
        [0.10, 0.90, 0.90],

        [0.90, 0.10, 0.10],
        [0.90, 0.10, 0.90],
        [0.90, 0.90, 0.10],
        [0.90, 0.90, 0.90],

        [0.25, 0.25, 0.50],
        [0.25, 0.75, 0.50],
        [0.75, 0.25, 0.50],
        [0.75, 0.75, 0.50],

        [0.50, 0.25, 0.25],
        [0.50, 0.25, 0.75],
        [0.50, 0.75, 0.25],
        [0.50, 0.75, 0.75],
    ],
    dtype=torch.float32,
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


def count_parameters(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_edges(
    parents,
):
    return sum(
        len(v)
        for v in parents.values()
    )


# ============================================================
# Graphs
# ============================================================

def independent_parents(
    dim,
):
    return {
        j: []
        for j in range(dim)
    }


def markov1_parents(
    dim,
):
    parents = {
        0: []
    }

    for j in range(
        1,
        dim,
    ):
        parents[j] = [
            j - 1
        ]

    return parents


def full_ar_parents(
    dim,
):
    return {
        j: list(range(j))
        for j in range(dim)
    }


def load_bayesian_graphs(
    path,
    benchmarks,
):
    df = pd.read_csv(
        path
    )

    required = {
        "benchmark",
        "parent",
        "child",
        "posterior_inclusion_probability",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Bayesian graph file is missing columns: "
            f"{sorted(missing)}"
        )

    graphs = {}

    for benchmark in benchmarks:

        parents = {
            j: []
            for j in range(DIM)
        }

        current = df[
            df[
                "benchmark"
            ] == benchmark
        ]

        # median_probability_graph.csv should already
        # contain only PIP > 0.5 edges, but checking again
        # makes the script robust.
        current = current[
            current[
                "posterior_inclusion_probability"
            ] > 0.5
        ]

        for row in (
            current.itertuples()
        ):
            parent = int(
                row.parent
            )

            child = int(
                row.child
            )

            parents[
                child
            ].append(
                parent
            )

        for child in parents:
            parents[
                child
            ] = sorted(
                parents[
                    child
                ]
            )

        graphs[
            benchmark
        ] = parents

    return graphs


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

            # log q_j = -log |dx/du|
            # NLL = +log |dx/du|
            loss = (
                loss
                + forward_logdet.mean()
            )

        loss = (
            loss / DIM
        )

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
# Structured copula
# ============================================================

class StructuredCopula(nn.Module):

    def __init__(
        self,
        dim,
        parents,
    ):
        super().__init__()

        self.dim = dim
        self.parents = parents

        self.conditionals = (
            nn.ModuleDict()
        )

        for j in range(
            1,
            dim,
        ):

            context_dim = (
                COND_DIM
                + len(
                    parents[j]
                )
            )

            self.conditionals[
                str(j)
            ] = ConditionalCubicFlow1D(
                cond_dim=context_dim,
                hidden=HIDDEN,
                n_bins=N_BINS,
            )

    def context(
        self,
        u,
        cond,
        j,
    ):

        parents = (
            self.parents[j]
        )

        if len(
            parents
        ) == 0:

            return cond

        return torch.cat(
            [
                u[
                    :,
                    parents
                ],
                cond,
            ],
            dim=1,
        )

    def log_prob_u(
        self,
        u,
        cond,
    ):

        logq = torch.zeros(
            u.shape[0],
            device=u.device,
        )

        for j in range(
            1,
            self.dim,
        ):

            context = self.context(
                u,
                cond,
                j,
            )

            _, forward_logdet = (
                self.conditionals[
                    str(j)
                ]
                .inverse(
                    u[:, j:j+1],
                    context,
                )
            )

            logq = (
                logq
                - forward_logdet
                .squeeze(-1)
            )

        return logq

    def sample_u(
        self,
        n,
        cond,
    ):

        if cond.shape[0] == 1:

            cond = cond.expand(
                n,
                -1,
            )

        u = torch.zeros(
            n,
            self.dim,
            device=cond.device,
        )

        # first copula coordinate uniform
        u[:, 0] = torch.rand(
            n,
            device=cond.device,
        )

        for j in range(
            1,
            self.dim,
        ):

            context = self.context(
                u,
                cond,
                j,
            )

            base = torch.rand(
                n,
                1,
                device=cond.device,
            )

            uj, _ = (
                self.conditionals[
                    str(j)
                ]
                .forward(
                    base,
                    context,
                )
            )

            u[:, j:j+1] = (
                uj.clamp(
                    0.0,
                    1.0,
                )
            )

        return u


# ============================================================
# Combined model
# ============================================================

class MarginalCopulaModel(nn.Module):

    def __init__(
        self,
        marginal_bank,
        copula,
        dim,
    ):
        super().__init__()

        self.dim = dim

        self.marginals = (
            marginal_bank.marginals
        )

        self.copula = copula

    def x_to_u_and_log_marginal(
        self,
        x,
        cond,
    ):

        u_list = []

        log_marginal = torch.zeros(
            x.shape[0],
            device=x.device,
        )

        for j in range(
            self.dim
        ):

            uj, forward_logdet = (
                self.marginals[j]
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

            log_marginal = (
                log_marginal
                - forward_logdet
                .squeeze(-1)
            )

        return (
            torch.cat(
                u_list,
                dim=1,
            ),
            log_marginal,
        )

    def log_prob(
        self,
        x,
        cond,
    ):

        u, log_marginal = (
            self.x_to_u_and_log_marginal(
                x,
                cond,
            )
        )

        log_copula = (
            self.copula
            .log_prob_u(
                u,
                cond,
            )
        )

        return (
            log_marginal
            + log_copula
        )

    def sample(
        self,
        n,
        cond,
    ):

        cond_batch = (
            cond.expand(
                n,
                -1,
            )
        )

        u = (
            self.copula
            .sample_u(
                n,
                cond_batch,
            )
        )

        x_list = []

        for j in range(
            self.dim
        ):

            xj, _ = (
                self.marginals[j]
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
# Copula training
# ============================================================

def train_copula(
    model,
    benchmark,
    device,
    steps,
    batch_size,
):

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_COPULA,
    )

    recent = []

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

            u, _ = (
                model
                .x_to_u_and_log_marginal(
                    x,
                    cond,
                )
            )

        logq = (
            model.copula
            .log_prob_u(
                u,
                cond,
            )
        )

        loss = (
            -logq.mean()
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.copula.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        recent.append(
            loss.item()
        )

        if (
            step + 1
        ) % 450 == 0:

            print(
                f"    copula "
                f"{step+1:4d}/{steps} "
                f"NLL="
                f"{np.mean(recent[-100:]):.6f}"
            )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    benchmark,
    device,
    n_eval,
    n_kl,
):

    rows = []

    for condition_id, cond in enumerate(
        TEST_CONDITIONS
    ):

        cond = (
            cond
            .to(device)
            .reshape(
                1,
                -1,
            )
        )

        # ====================================================
        # Importance-sampling proposal evaluation
        # ====================================================

        cond_eval = (
            cond.expand(
                n_eval,
                -1,
            )
        )

        x = model.sample(
            n_eval,
            cond,
        )

        logq = model.log_prob(
            x,
            cond_eval,
        )

        logp = target_logpdf(
            benchmark,
            x,
            cond_eval,
        )

        logw = (
            logp - logq
        )

        weights = torch.exp(
            logw
        )

        estimate = (
            weights.mean()
        )

        normalized_ess = (
            weights.sum() ** 2
            /
            (
                n_eval
                * weights
                .square()
                .sum()
            )
        )

        variance = (
            weights.var(
                unbiased=True
            )
            / n_eval
        )

        # ====================================================
        # Forward KL
        # ====================================================

        cond_kl = (
            cond.expand(
                n_kl,
                -1,
            )
        )

        x_target = sample_target(
            benchmark,
            cond_kl,
            DIM,
        )

        logp_target = target_logpdf(
            benchmark,
            x_target,
            cond_kl,
        )

        logq_target = model.log_prob(
            x_target,
            cond_kl,
        )

        forward_kl = (
            logp_target
            - logq_target
        ).mean()

        rows.append(
            {
                "condition_id":
                    condition_id,

                "ess":
                    normalized_ess.item(),

                "forward_kl":
                    forward_kl.item(),

                "variance":
                    variance.item(),

                "estimate":
                    estimate.item(),

                "relative_error":
                    abs(
                        estimate.item()
                        - 1.0
                    ),
            }
        )

    return rows


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

    bayesian_graphs = (
        load_bayesian_graphs(
            args.graph_file,
            args.benchmarks,
        )
    )

    all_results = []
    graph_rows = []

    for benchmark in (
        args.benchmarks
    ):

        print()
        print(
            "=" * 100
        )

        print(
            f"{benchmark} | d={DIM}"
        )

        print(
            "=" * 100
        )

        # ====================================================
        # Show Bayesian graph
        # ====================================================

        bayes_parents = (
            bayesian_graphs[
                benchmark
            ]
        )

        print(
            "\nBayesian selected graph:"
        )

        print(
            bayes_parents
        )

        print(
            "Bayesian edge count:",
            count_edges(
                bayes_parents
            ),
        )

        # ====================================================
        # Shared marginals
        # ====================================================

        seed_everything(
            args.seed
        )

        print(
            "\nTraining shared marginals..."
        )

        marginal_bank = (
            MarginalBank(
                DIM
            )
            .to(device)
        )

        train_marginals(
            marginal_bank,
            benchmark,
            device,
            args.marginal_steps,
            args.batch_size,
        )

        marginal_state = {
            key:
            value
            .detach()
            .cpu()
            .clone()

            for key, value
            in marginal_bank
            .state_dict()
            .items()
        }

        def make_bank():

            bank = (
                MarginalBank(
                    DIM
                )
                .to(device)
            )

            bank.load_state_dict(
                marginal_state
            )

            for marginal in (
                bank.marginals
            ):

                for parameter in (
                    marginal.parameters()
                ):

                    parameter.requires_grad = False

            return bank

        # ====================================================
        # Candidate models
        # ====================================================

        graphs = {
            "independent":
                independent_parents(
                    DIM
                ),

            "markov1":
                markov1_parents(
                    DIM
                ),

            "full_ar":
                full_ar_parents(
                    DIM
                ),

            "bayesian_selected":
                bayes_parents,
        }

        # ====================================================
        # Train/evaluate each graph
        # ====================================================

        for graph_name, parents in (
            graphs.items()
        ):

            edge_count = count_edges(
                parents
            )

            print()
            print(
                "-" * 90
            )

            print(
                f"MODEL: {graph_name} "
                f"| edges={edge_count}"
            )

            # IMPORTANT:
            # same seed across graph families.
            seed_everything(
                args.seed
            )

            model = (
                MarginalCopulaModel(
                    make_bank(),
                    StructuredCopula(
                        DIM,
                        parents,
                    ),
                    DIM,
                )
                .to(device)
            )

            train_copula(
                model,
                benchmark,
                device,
                args.copula_steps,
                args.batch_size,
            )

            results = evaluate(
                model,
                benchmark,
                device,
                args.n_eval,
                args.n_kl,
            )

            for row in results:

                row.update(
                    {
                        "benchmark":
                            benchmark,

                        "model":
                            graph_name,

                        "edge_count":
                            edge_count,

                        "parameter_count":
                            count_parameters(
                                model
                            ),
                    }
                )

                all_results.append(
                    row
                )

            for child in range(
                DIM
            ):

                for parent in (
                    parents[
                        child
                    ]
                ):

                    graph_rows.append(
                        {
                            "benchmark":
                                benchmark,

                            "model":
                                graph_name,

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

            pd.DataFrame(
                all_results
            ).to_csv(
                output_dir
                / "raw_results.csv",
                index=False,
            )

            pd.DataFrame(
                graph_rows
            ).to_csv(
                output_dir
                / "graphs.csv",
                index=False,
            )

            del model

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        del marginal_bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # Summary
    # ========================================================

    raw_df = pd.DataFrame(
        all_results
    )

    summary = (
        raw_df
        .groupby(
            [
                "benchmark",
                "model",
                "edge_count",
                "parameter_count",
            ],
            as_index=False,
        )
        .agg(
            ess_mean=(
                "ess",
                "mean",
            ),

            ess_std=(
                "ess",
                "std",
            ),

            forward_kl_mean=(
                "forward_kl",
                "mean",
            ),

            variance_mean=(
                "variance",
                "mean",
            ),

            relative_error_mean=(
                "relative_error",
                "mean",
            ),

            estimate_mean=(
                "estimate",
                "mean",
            ),
        )
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "BAYESIAN-SELECTED COPULA BENCHMARK COMPLETE"
    )

    print(
        "=" * 120
    )

    print()

    print(
        summary
        .sort_values(
            [
                "benchmark",
                "ess_mean",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-file",
        type=str,
        default=
            DEFAULT_GRAPH_FILE,
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=
            DEFAULT_BENCHMARKS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--marginal-steps",
        type=int,
        default=
            MARGINAL_STEPS,
    )

    parser.add_argument(
        "--copula-steps",
        type=int,
        default=
            COPULA_STEPS,
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
        "--n-kl",
        type=int,
        default=
            N_KL,
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

        args.marginal_steps = 100
        args.copula_steps = 300

        args.batch_size = 1024

        args.n_eval = 10_000
        args.n_kl = 5_000

        args.output = (
            "results/"
            "bayesian_selected_copula_benchmark_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )