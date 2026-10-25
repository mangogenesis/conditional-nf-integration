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
# CONFIGURATION
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

DEFAULT_THRESHOLDS = [
    0.0,
    0.01,
    0.02,
]

DEFAULT_TOPKS = [
    1,
    2,
]

DEFAULT_EVIDENCE_PATH = (
    "results/"
    "masked_edge_intervention_smoke/"
    "edge_interventions.csv"
)

OUTPUT_DIR = Path(
    "results/intervention_selected_copula"
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

def sample_conditions(n, device):

    return torch.rand(
        n,
        COND_DIM,
        device=device,
    )


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
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

    for step in range(steps):

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

        for j in range(DIM):

            _, forward_logdet = (
                bank.marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            loss = (
                loss
                + forward_logdet.mean()
            )

        loss = loss / DIM

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

        if (step + 1) % 300 == 0:

            print(
                f"  marginal "
                f"{step+1:4d}/{steps} "
                f"loss="
                f"{np.mean(recent[-100:]):.6f}"
            )


# ============================================================
# Graph construction
# ============================================================

def independent_parents(dim):

    return {
        j: []
        for j in range(dim)
    }


def markov1_parents(dim):

    result = {
        0: []
    }

    for j in range(
        1,
        dim,
    ):

        result[j] = [
            j - 1
        ]

    return result


def full_ar_parents(dim):

    return {
        j: list(range(j))
        for j in range(dim)
    }


def threshold_parents(
    evidence_df,
    threshold,
):

    parents = {
        j: []
        for j in range(DIM)
    }

    selected = evidence_df[
        evidence_df[
            "delta_loglik"
        ] > threshold
    ]

    for _, row in (
        selected.iterrows()
    ):

        child = int(
            row["child"]
        )

        parent = int(
            row["parent"]
        )

        parents[
            child
        ].append(
            parent
        )

    for j in parents:

        parents[j] = sorted(
            parents[j]
        )

    return parents


def topk_parents(
    evidence_df,
    k,
):

    parents = {
        j: []
        for j in range(DIM)
    }

    for child in range(
        1,
        DIM,
    ):

        child_df = (
            evidence_df[
                evidence_df[
                    "child"
                ] == child
            ]
            .sort_values(
                "delta_loglik",
                ascending=False,
            )
        )

        k_use = min(
            k,
            child,
        )

        selected = (
            child_df
            .head(
                k_use
            )
        )

        parents[
            child
        ] = sorted(
            selected[
                "parent"
            ]
            .astype(int)
            .tolist()
        )

    return parents


def count_edges(
    parents,
):

    return sum(
        len(v)
        for v in parents.values()
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

            # Even if there are no parents,
            # q(u_j | c) is still learned.
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

        parent_list = (
            self.parents[j]
        )

        if len(
            parent_list
        ) == 0:

            return cond

        return torch.cat(
            [
                u[
                    :,
                    parent_list
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

        # u_0 is uniform in copula space.
        for j in range(
            1,
            self.dim,
        ):

            ctx = self.context(
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
                    ctx,
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

        u[:, 0] = torch.rand(
            n,
            device=cond.device,
        )

        for j in range(
            1,
            self.dim,
        ):

            ctx = self.context(
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
                    ctx,
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
# Marginal + copula model
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

        values = []

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

            values.append(
                uj
            )

            log_marginal = (
                log_marginal
                - forward_logdet
                .squeeze(-1)
            )

        return (
            torch.cat(
                values,
                dim=1,
            ),
            log_marginal,
        )

    def log_prob(
        self,
        x,
        cond,
    ):

        (
            u,
            log_marginal,
        ) = (
            self
            .x_to_u_and_log_marginal(
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

        values = []

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

            values.append(
                xj.clamp(
                    0.0,
                    1.0,
                )
            )

        return torch.cat(
            values,
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
            cond.to(device)
            .reshape(
                1,
                -1,
            )
        )

        # ----------------------------------------------------
        # Importance sampling from q
        # ----------------------------------------------------

        cond_eval = cond.expand(
            n_eval,
            -1,
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

        ess = (
            weights.sum() ** 2
            /
            (
                n_eval
                * weights.square().sum()
            )
        )

        variance = (
            weights.var(
                unbiased=True
            )
            / n_eval
        )

        # ----------------------------------------------------
        # Forward KL
        # ----------------------------------------------------

        cond_kl = cond.expand(
            n_kl,
            -1,
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
                    ess.item(),

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
# Evidence loading
# ============================================================

def load_evidence(
    path,
):

    df = pd.read_csv(
        path
    )

    required = {
        "benchmark",
        "parent",
        "child",
        "delta_loglik",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "Evidence file missing columns: "
            f"{sorted(missing)}"
        )

    # If multiple seeds exist, use mean intervention evidence.
    grouped = (
        df.groupby(
            [
                "benchmark",
                "parent",
                "child",
            ],
            as_index=False,
        )
        .agg(
            delta_loglik=(
                "delta_loglik",
                "mean",
            ),
            delta_std_across_runs=(
                "delta_loglik",
                "std",
            ),
        )
    )

    return grouped


# ============================================================
# Graph diagnostics
# ============================================================

def graph_rows(
    benchmark,
    graph_name,
    parents,
    evidence_df,
):

    rows = []

    for child in range(
        1,
        DIM,
    ):

        for parent in (
            parents[child]
        ):

            match = evidence_df[
                (
                    evidence_df[
                        "parent"
                    ] == parent
                )
                &
                (
                    evidence_df[
                        "child"
                    ] == child
                )
            ]

            delta = (
                float(
                    match[
                        "delta_loglik"
                    ].iloc[0]
                )
                if len(match) > 0
                else np.nan
            )

            rows.append(
                {
                    "benchmark":
                        benchmark,

                    "graph":
                        graph_name,

                    "parent":
                        parent,

                    "child":
                        child,

                    "delta_loglik":
                        delta,

                    "is_markov1_edge":
                        int(
                            parent
                            == child - 1
                        ),
                }
            )

    return rows


# ============================================================
# Main
# ============================================================

def run(args):

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

    evidence = load_evidence(
        args.evidence
    )

    all_results = []
    all_selected_edges = []

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

        benchmark_evidence = (
            evidence[
                evidence[
                    "benchmark"
                ] == benchmark
            ]
            .copy()
        )

        if len(
            benchmark_evidence
        ) == 0:

            raise ValueError(
                "No intervention evidence "
                f"found for {benchmark}"
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
            k:
            v.detach()
            .cpu()
            .clone()

            for k, v
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

                for p in (
                    marginal.parameters()
                ):

                    p.requires_grad = False

            return bank

        # ====================================================
        # Build graph collection
        # ====================================================

        graph_collection = {
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
        }

        for threshold in (
            args.thresholds
        ):

            name = (
                "delta_gt_"
                + str(threshold)
                .replace(
                    ".",
                    "p",
                )
            )

            graph_collection[
                name
            ] = threshold_parents(
                benchmark_evidence,
                threshold,
            )

        for k in (
            args.topks
        ):

            graph_collection[
                f"top{k}"
            ] = topk_parents(
                benchmark_evidence,
                k,
            )

        # ====================================================
        # Display learned structures
        # ====================================================

        print(
            "\nSelected graph sizes:"
        )

        for name, parents in (
            graph_collection.items()
        ):

            n_edges = count_edges(
                parents
            )

            print(
                f"  {name:18s}: "
                f"{n_edges:2d} edges"
            )

            if (
                name.startswith(
                    "delta_"
                )
                or name.startswith(
                    "top"
                )
            ):

                print(
                    f"    {parents}"
                )

        # ====================================================
        # Train each graph from scratch
        # ====================================================

        for graph_id, (
            graph_name,
            parents,
        ) in enumerate(
            graph_collection.items()
        ):

            print()
            print(
                "-" * 90
            )

            print(
                f"MODEL: {graph_name} "
                f"| edges="
                f"{count_edges(parents)}"
            )

            # Same initialization seed scheme for reproducibility
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

            edge_count = count_edges(
                parents
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

            all_selected_edges.extend(
                graph_rows(
                    benchmark,
                    graph_name,
                    parents,
                    benchmark_evidence,
                )
            )

            pd.DataFrame(
                all_results
            ).to_csv(
                output_dir
                / "raw_results.csv",
                index=False,
            )

            pd.DataFrame(
                all_selected_edges
            ).to_csv(
                output_dir
                / "selected_edges.csv",
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
        "=" * 110
    )

    print(
        "INTERVENTION-SELECTED COPULA TEST COMPLETE"
    )

    print(
        "=" * 110
    )

    print()

    print(
        summary.sort_values(
            [
                "benchmark",
                "ess_mean",
            ],
            ascending=[
                True,
                False,
            ],
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
        "--evidence",
        type=str,
        default=
            DEFAULT_EVIDENCE_PATH,
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=
            DEFAULT_BENCHMARKS,
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=
            DEFAULT_THRESHOLDS,
    )

    parser.add_argument(
        "--topks",
        nargs="+",
        type=int,
        default=
            DEFAULT_TOPKS,
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

    args = (
        parser.parse_args()
    )

    if args.smoke:

        args.marginal_steps = 100
        args.copula_steps = 300

        args.batch_size = 1024

        args.n_eval = 10_000
        args.n_kl = 5_000

        args.output = (
            "results/"
            "intervention_selected_copula_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )