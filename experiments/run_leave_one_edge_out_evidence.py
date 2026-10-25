import argparse
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


# ============================================================
# Configuration
# ============================================================

COND_DIM = 3
DIM = 8

N_BINS = 32
HIDDEN = 128

MARGINAL_STEPS = 1200

# Each child conditional / ablated conditional gets this many steps
EDGE_MODEL_STEPS = 1200

BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 100_000

LR_MARGINAL = 1e-4
LR_EDGE_MODEL = 1e-4

GRAD_CLIP = 10.0

DEFAULT_BENCHMARKS = [
    "single_gaussian",
    "curved_ridge",
    "sharp_broad",
]

DEFAULT_SEEDS = [0]

OUTPUT_DIR = Path(
    "results/leave_one_edge_out_evidence"
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


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# Marginal model
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

    def x_to_u(
        self,
        x,
        cond,
    ):

        u_list = []

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

            u_list.append(
                uj.clamp(
                    0.0,
                    1.0,
                )
            )

        return torch.cat(
            u_list,
            dim=1,
        )


# ============================================================
# Train marginals
# ============================================================

def train_marginals(
    marginal_bank,
    benchmark,
    device,
    steps,
    batch_size,
):

    optimizer = torch.optim.Adam(
        marginal_bank.parameters(),
        lr=LR_MARGINAL,
    )

    recent = []

    marginal_bank.train()

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
                marginal_bank
                .marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            # log q_j(x_j | c)
            # = -log |dx/du|
            loss += (
                forward_logdet.mean()
            )

        loss = (
            loss / DIM
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            marginal_bank.parameters(),
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
# One child conditional model
# ============================================================

class ChildConditional(nn.Module):

    def __init__(
        self,
        parent_indices,
    ):
        super().__init__()

        self.parent_indices = list(
            parent_indices
        )

        context_dim = (
            len(
                self.parent_indices
            )
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
    ):

        if len(
            self.parent_indices
        ) == 0:

            return cond

        return torch.cat(
            [
                u[
                    :,
                    self.parent_indices
                ],
                cond,
            ],
            dim=1,
        )

    def log_prob(
        self,
        child_u,
        u,
        cond,
    ):

        context = self.make_context(
            u,
            cond,
        )

        _, forward_logdet = (
            self.flow.inverse(
                child_u,
                context,
            )
        )

        return (
            -forward_logdet.squeeze(-1)
        )


# ============================================================
# Train one child model
# ============================================================

def train_child_model(
    marginal_bank,
    benchmark,
    child,
    parents,
    device,
    steps,
    batch_size,
    seed,
):

    # Different deterministic seed for each graph,
    # but reproducible across runs.
    seed_everything(
        seed
    )

    model = (
        ChildConditional(
            parent_indices=parents,
        )
        .to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_EDGE_MODEL,
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

        child_u = u[
            :,
            child:child+1
        ]

        logq = model.log_prob(
            child_u,
            u,
            cond,
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

        if (
            step + 1
        ) % 400 == 0:

            print(
                f"      step "
                f"{step+1:4d}/{steps} "
                f"NLL="
                f"{np.mean(recent[-100:]):.6f}"
            )

    return model


# ============================================================
# Held-out evaluation
# ============================================================

@torch.no_grad()
def evaluate_child_models(
    full_model,
    ablated_model,
    marginal_bank,
    benchmark,
    child,
    device,
    n_eval,
):

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

    child_u = u[
        :,
        child:child+1
    ]

    logq_full = full_model.log_prob(
        child_u,
        u,
        cond,
    )

    logq_ablated = (
        ablated_model.log_prob(
            child_u,
            u,
            cond,
        )
    )

    delta = (
        logq_full
        - logq_ablated
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
        /
        math.sqrt(
            n_eval
        )
    )

    # Simple signal-to-noise diagnostic.
    z_score = (
        delta_mean
        / delta_se
        if delta_se > 0
        else np.nan
    )

    return {
        "delta_loglik":
            delta_mean,

        "delta_std":
            delta_std,

        "delta_se":
            delta_se,

        "z_score":
            z_score,

        "full_loglik":
            logq_full
            .mean()
            .item(),

        "ablated_loglik":
            logq_ablated
            .mean()
            .item(),
    }


# ============================================================
# Evaluate full child model alone
# ============================================================

@torch.no_grad()
def evaluate_single_model(
    model,
    marginal_bank,
    benchmark,
    child,
    device,
    n_eval,
):

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

    child_u = u[
        :,
        child:child+1
    ]

    logq = model.log_prob(
        child_u,
        u,
        cond,
    )

    return (
        logq.mean().item()
    )


# ============================================================
# Main benchmark
# ============================================================

def run_one(
    benchmark,
    seed,
    args,
    device,
    output_dir,
):

    print()
    print(
        "=" * 90
    )

    print(
        f"{benchmark} | d={DIM} | seed={seed}"
    )

    print(
        "=" * 90
    )

    seed_everything(
        seed
    )

    # --------------------------------------------------------
    # Shared marginals
    # --------------------------------------------------------

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

    marginal_bank.eval()

    for p in (
        marginal_bank.parameters()
    ):
        p.requires_grad = False

    rows = []
    full_rows = []

    # ========================================================
    # Children 1,...,7
    # ========================================================

    for child in range(
        1,
        DIM,
    ):

        full_parents = list(
            range(child)
        )

        print()
        print(
            "-" * 80
        )

        print(
            f"CHILD u_{child}"
        )

        print(
            f"Full parents: "
            f"{full_parents}"
        )

        # ----------------------------------------------------
        # Full-parent conditional
        # ----------------------------------------------------

        print(
            "  Training FULL child model..."
        )

        full_seed = (
            seed * 10000
            + child * 100
        )

        full_model = train_child_model(
            marginal_bank=
                marginal_bank,

            benchmark=
                benchmark,

            child=
                child,

            parents=
                full_parents,

            device=
                device,

            steps=
                args.edge_steps,

            batch_size=
                args.batch_size,

            seed=
                full_seed,
        )

        full_model.eval()

        full_test_loglik = (
            evaluate_single_model(
                full_model,
                marginal_bank,
                benchmark,
                child,
                device,
                args.n_eval,
            )
        )

        full_rows.append(
            {
                "benchmark":
                    benchmark,

                "seed":
                    seed,

                "child":
                    child,

                "n_parents":
                    len(
                        full_parents
                    ),

                "full_test_loglik":
                    full_test_loglik,

                "parameter_count":
                    count_parameters(
                        full_model
                    ),
            }
        )

        # ----------------------------------------------------
        # Leave each parent out
        # ----------------------------------------------------

        for parent in (
            full_parents
        ):

            ablated_parents = [
                p
                for p
                in full_parents
                if p != parent
            ]

            print()
            print(
                f"  EDGE {parent} -> {child}"
            )

            print(
                f"    ablated parents: "
                f"{ablated_parents}"
            )

            ablation_seed = (
                seed * 10000
                + child * 100
                + parent
                + 1
            )

            ablated_model = (
                train_child_model(
                    marginal_bank=
                        marginal_bank,

                    benchmark=
                        benchmark,

                    child=
                        child,

                    parents=
                        ablated_parents,

                    device=
                        device,

                    steps=
                        args.edge_steps,

                    batch_size=
                        args.batch_size,

                    seed=
                        ablation_seed,
                )
            )

            ablated_model.eval()

            evidence = (
                evaluate_child_models(
                    full_model=
                        full_model,

                    ablated_model=
                        ablated_model,

                    marginal_bank=
                        marginal_bank,

                    benchmark=
                        benchmark,

                    child=
                        child,

                    device=
                        device,

                    n_eval=
                        args.n_eval,
                )
            )

            is_markov1 = int(
                parent
                == child - 1
            )

            rows.append(
                {
                    "benchmark":
                        benchmark,

                    "seed":
                        seed,

                    "parent":
                        parent,

                    "child":
                        child,

                    "is_markov1_edge":
                        is_markov1,

                    "full_parent_count":
                        len(
                            full_parents
                        ),

                    "ablated_parent_count":
                        len(
                            ablated_parents
                        ),

                    "delta_loglik":
                        evidence[
                            "delta_loglik"
                        ],

                    "delta_std":
                        evidence[
                            "delta_std"
                        ],

                    "delta_se":
                        evidence[
                            "delta_se"
                        ],

                    "z_score":
                        evidence[
                            "z_score"
                        ],

                    "full_loglik":
                        evidence[
                            "full_loglik"
                        ],

                    "ablated_loglik":
                        evidence[
                            "ablated_loglik"
                        ],

                    "full_parameter_count":
                        count_parameters(
                            full_model
                        ),

                    "ablated_parameter_count":
                        count_parameters(
                            ablated_model
                        ),
                }
            )

            print(
                f"    ΔlogL = "
                f"{evidence['delta_loglik']:+.6f} "
                f"± "
                f"{evidence['delta_se']:.6f} "
                f"(z={evidence['z_score']:+.2f})"
            )

            del ablated_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del full_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        rows,
        full_rows,
    )


# ============================================================
# Aggregate diagnostics
# ============================================================

def make_summary(
    evidence_df,
):

    summaries = []

    for benchmark in (
        evidence_df[
            "benchmark"
        ].unique()
    ):

        df = evidence_df[
            evidence_df[
                "benchmark"
            ] == benchmark
        ]

        local = df[
            df[
                "is_markov1_edge"
            ] == 1
        ]

        nonlocal_df = df[
            df[
                "is_markov1_edge"
            ] == 0
        ]

        positive = (
            df[
                "delta_loglik"
            ] > 0
        )

        strong_positive = (
            df[
                "z_score"
            ] > 2.0
        )

        summaries.append(
            {
                "benchmark":
                    benchmark,

                "n_edges":
                    len(df),

                "mean_delta":
                    df[
                        "delta_loglik"
                    ].mean(),

                "median_delta":
                    df[
                        "delta_loglik"
                    ].median(),

                "max_delta":
                    df[
                        "delta_loglik"
                    ].max(),

                "min_delta":
                    df[
                        "delta_loglik"
                    ].min(),

                "positive_edge_fraction":
                    positive.mean(),

                "z_gt_2_fraction":
                    strong_positive.mean(),

                "mean_markov1_delta":
                    local[
                        "delta_loglik"
                    ].mean(),

                "mean_nonmarkov_delta":
                    nonlocal_df[
                        "delta_loglik"
                    ].mean(),

                "markov_minus_nonmarkov":
                    (
                        local[
                            "delta_loglik"
                        ].mean()
                        -
                        nonlocal_df[
                            "delta_loglik"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# Ranking
# ============================================================

def make_ranked_edges(
    evidence_df,
):

    ranked = evidence_df.copy()

    ranked[
        "rank_global"
    ] = (
        ranked
        .groupby(
            [
                "benchmark",
                "seed",
            ]
        )[
            "delta_loglik"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    ranked[
        "rank_within_child"
    ] = (
        ranked
        .groupby(
            [
                "benchmark",
                "seed",
                "child",
            ]
        )[
            "delta_loglik"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    return (
        ranked.sort_values(
            [
                "benchmark",
                "seed",
                "rank_global",
            ]
        )
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

    all_rows = []
    all_full_rows = []

    for benchmark in (
        args.benchmarks
    ):

        for seed in (
            args.seeds
        ):

            rows, full_rows = (
                run_one(
                    benchmark=
                        benchmark,

                    seed=
                        seed,

                    args=
                        args,

                    device=
                        device,

                    output_dir=
                        output_dir,
                )
            )

            all_rows.extend(
                rows
            )

            all_full_rows.extend(
                full_rows
            )

            # intermediate save

            pd.DataFrame(
                all_rows
            ).to_csv(
                output_dir
                / "edge_evidence.csv",
                index=False,
            )

            pd.DataFrame(
                all_full_rows
            ).to_csv(
                output_dir
                / "full_child_models.csv",
                index=False,
            )

    evidence_df = pd.DataFrame(
        all_rows
    )

    ranked_df = make_ranked_edges(
        evidence_df
    )

    ranked_df.to_csv(
        output_dir
        / "ranked_edges.csv",
        index=False,
    )

    summary_df = make_summary(
        evidence_df
    )

    summary_df.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "LEAVE-ONE-EDGE-OUT EVIDENCE TEST COMPLETE"
    )

    print(
        "=" * 100
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Top edges by ΔlogL:"
    )

    print()

    for benchmark in (
        args.benchmarks
    ):

        temp = ranked_df[
            ranked_df[
                "benchmark"
            ] == benchmark
        ].head(12)

        print(
            f"\n{benchmark}"
        )

        print(
            temp[
                [
                    "parent",
                    "child",
                    "is_markov1_edge",
                    "delta_loglik",
                    "delta_se",
                    "z_score",
                    "rank_global",
                ]
            ].to_string(
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
        "--edge-steps",
        type=int,
        default=
            EDGE_MODEL_STEPS,
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
            EVAL_BATCH_SIZE,
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
        args.edge_steps = 200

        args.batch_size = 1024
        args.n_eval = 10_000

        args.output = (
            "results/"
            "leave_one_edge_out_evidence_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )