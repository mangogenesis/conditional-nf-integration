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

DEFAULT_SEEDS = [0]

OUTPUT_DIR = Path(
    "results/masked_edge_intervention"
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

    def __init__(self, dim):
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

        for j in range(self.dim):

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


# ============================================================
# Marginal training
# ============================================================

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

            # log q = -log |dx/du|
            # NLL = +log |dx/du|
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
# Masked child conditioner
# ============================================================

class MaskedChildConditional(nn.Module):

    def __init__(
        self,
        child,
    ):
        super().__init__()

        self.child = child
        self.n_parents = child

        # Context:
        #
        #   masked centered parent values : j dims
        #   mask itself                  : j dims
        #   external condition c         : COND_DIM
        #
        context_dim = (
            2 * self.n_parents
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

        parents = u[
            :,
            :self.child
        ]

        centered = (
            parents - 0.5
        )

        masked_values = (
            mask * centered
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
            -forward_logdet.squeeze(-1)
        )


# ============================================================
# Mask sampling
# ============================================================

def sample_training_mask(
    batch_size,
    n_parents,
    device,
    keep_prob,
):

    if n_parents == 0:
        return torch.empty(
            batch_size,
            0,
            device=device,
        )

    probs = torch.full(
        (
            batch_size,
            n_parents,
        ),
        keep_prob,
        device=device,
    )

    mask = torch.bernoulli(
        probs
    )

    return mask


# ============================================================
# Train one child with random parent dropout
# ============================================================

def train_masked_child(
    marginal_bank,
    benchmark,
    child,
    device,
    steps,
    batch_size,
    keep_prob,
    seed,
):

    seed_everything(seed)

    model = (
        MaskedChildConditional(
            child=child,
        )
        .to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_MASKED,
    )

    recent = []

    model.train()

    for step in range(steps):

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

        mask = sample_training_mask(
            batch_size,
            child,
            device,
            keep_prob,
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

        if (step + 1) % 400 == 0:
            print(
                f"      child={child} "
                f"step "
                f"{step+1:4d}/{steps} "
                f"NLL="
                f"{np.mean(recent[-100:]):.6f}"
            )

    return model


# ============================================================
# Edge intervention
# ============================================================

@torch.no_grad()
def evaluate_edge_interventions(
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

    # --------------------------------------------------------
    # Full-parent state
    # --------------------------------------------------------

    mask_all_on = torch.ones(
        n_eval,
        child,
        device=device,
    )

    logq_all_on = model.log_prob(
        child_u,
        u,
        cond,
        mask_all_on,
    )

    rows = []

    for parent in range(child):

        # ----------------------------------------------------
        # Same model, same data, exactly one parent removed
        # ----------------------------------------------------

        mask_off = (
            mask_all_on.clone()
        )

        mask_off[
            :,
            parent
        ] = 0.0

        logq_off = model.log_prob(
            child_u,
            u,
            cond,
            mask_off,
        )

        delta = (
            logq_all_on
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
            / math.sqrt(n_eval)
        )

        z = (
            delta_mean
            / delta_se
            if delta_se > 0
            else np.nan
        )

        rows.append(
            {
                "parent":
                    parent,

                "child":
                    child,

                "is_markov1_edge":
                    int(
                        parent
                        == child - 1
                    ),

                "delta_loglik":
                    delta_mean,

                "delta_std":
                    delta_std,

                "delta_se":
                    delta_se,

                "z_score":
                    z,

                "all_on_loglik":
                    logq_all_on
                    .mean()
                    .item(),

                "edge_off_loglik":
                    logq_off
                    .mean()
                    .item(),
            }
        )

    return rows


# ============================================================
# Optional all-off diagnostic
# ============================================================

@torch.no_grad()
def evaluate_all_on_vs_all_off(
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

    all_on = torch.ones(
        n_eval,
        child,
        device=device,
    )

    all_off = torch.zeros(
        n_eval,
        child,
        device=device,
    )

    ll_on = model.log_prob(
        child_u,
        u,
        cond,
        all_on,
    )

    ll_off = model.log_prob(
        child_u,
        u,
        cond,
        all_off,
    )

    delta = (
        ll_on - ll_off
    )

    return {
        "child":
            child,

        "all_on_loglik":
            ll_on.mean().item(),

        "all_off_loglik":
            ll_off.mean().item(),

        "all_on_minus_all_off":
            delta.mean().item(),
    }


# ============================================================
# Summary
# ============================================================

def make_summary(df):

    rows = []

    for benchmark in (
        df["benchmark"].unique()
    ):

        current = df[
            df["benchmark"]
            == benchmark
        ]

        markov = current[
            current[
                "is_markov1_edge"
            ] == 1
        ]

        other = current[
            current[
                "is_markov1_edge"
            ] == 0
        ]

        rows.append(
            {
                "benchmark":
                    benchmark,

                "n_edges":
                    len(current),

                "mean_delta":
                    current[
                        "delta_loglik"
                    ].mean(),

                "median_delta":
                    current[
                        "delta_loglik"
                    ].median(),

                "max_delta":
                    current[
                        "delta_loglik"
                    ].max(),

                "min_delta":
                    current[
                        "delta_loglik"
                    ].min(),

                "mean_markov1_delta":
                    markov[
                        "delta_loglik"
                    ].mean(),

                "mean_nonmarkov_delta":
                    other[
                        "delta_loglik"
                    ].mean(),

                "markov_minus_nonmarkov":
                    (
                        markov[
                            "delta_loglik"
                        ].mean()
                        -
                        other[
                            "delta_loglik"
                        ].mean()
                    ),

                "positive_fraction":
                    (
                        current[
                            "delta_loglik"
                        ] > 0
                    ).mean(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Run benchmark
# ============================================================

def run_one(
    benchmark,
    seed,
    args,
    device,
):

    print()
    print(
        "=" * 90
    )

    print(
        f"{benchmark} "
        f"| d={DIM} "
        f"| seed={seed}"
    )

    print(
        "=" * 90
    )

    seed_everything(seed)

    # --------------------------------------------------------
    # Train shared marginals
    # --------------------------------------------------------

    print(
        "\nTraining shared marginals..."
    )

    marginal_bank = (
        MarginalBank(DIM)
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

    edge_rows = []
    child_rows = []

    # --------------------------------------------------------
    # Train one masked network per child
    # --------------------------------------------------------

    for child in range(
        1,
        DIM,
    ):

        print()
        print(
            "-" * 80
        )

        print(
            f"CHILD u_{child} "
            f"| candidate parents "
            f"{list(range(child))}"
        )

        child_seed = (
            seed * 1000
            + child
        )

        model = train_masked_child(
            marginal_bank=
                marginal_bank,

            benchmark=
                benchmark,

            child=
                child,

            device=
                device,

            steps=
                args.masked_steps,

            batch_size=
                args.batch_size,

            keep_prob=
                args.keep_prob,

            seed=
                child_seed,
        )

        model.eval()

        current_edges = (
            evaluate_edge_interventions(
                model=
                    model,

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

        alloff = (
            evaluate_all_on_vs_all_off(
                model=
                    model,

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

        child_rows.append(
            {
                "benchmark":
                    benchmark,

                "seed":
                    seed,

                "child":
                    child,

                "keep_prob":
                    args.keep_prob,

                "all_on_loglik":
                    alloff[
                        "all_on_loglik"
                    ],

                "all_off_loglik":
                    alloff[
                        "all_off_loglik"
                    ],

                "all_on_minus_all_off":
                    alloff[
                        "all_on_minus_all_off"
                    ],

                "parameter_count":
                    count_parameters(
                        model
                    ),
            }
        )

        for row in current_edges:

            row.update(
                {
                    "benchmark":
                        benchmark,

                    "seed":
                        seed,

                    "keep_prob":
                        args.keep_prob,
                }
            )

            edge_rows.append(row)

            print(
                f"  edge "
                f"{row['parent']} -> "
                f"{row['child']} "
                f"ΔlogL="
                f"{row['delta_loglik']:+.6f} "
                f"±"
                f"{row['delta_se']:.6f}"
            )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        edge_rows,
        child_rows,
    )


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

    all_edges = []
    all_children = []

    for benchmark in (
        args.benchmarks
    ):

        for seed in (
            args.seeds
        ):

            (
                edge_rows,
                child_rows,
            ) = run_one(
                benchmark=
                    benchmark,

                seed=
                    seed,

                args=
                    args,

                device=
                    device,
            )

            all_edges.extend(
                edge_rows
            )

            all_children.extend(
                child_rows
            )

            pd.DataFrame(
                all_edges
            ).to_csv(
                output_dir
                / "edge_interventions.csv",
                index=False,
            )

            pd.DataFrame(
                all_children
            ).to_csv(
                output_dir
                / "child_diagnostics.csv",
                index=False,
            )

    edge_df = pd.DataFrame(
        all_edges
    )

    edge_df[
        "rank_global"
    ] = (
        edge_df
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

    edge_df[
        "rank_within_child"
    ] = (
        edge_df
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

    edge_df.sort_values(
        [
            "benchmark",
            "seed",
            "rank_global",
        ]
    ).to_csv(
        output_dir
        / "ranked_edges.csv",
        index=False,
    )

    summary = make_summary(
        edge_df
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "MASKED EDGE INTERVENTION TEST COMPLETE"
    )

    print(
        "=" * 100
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    for benchmark in (
        args.benchmarks
    ):

        print(
            f"\nTOP EDGES: {benchmark}"
        )

        temp = (
            edge_df[
                edge_df[
                    "benchmark"
                ] == benchmark
            ]
            .sort_values(
                "delta_loglik",
                ascending=False,
            )
            .head(12)
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
        args.masked_steps = 300

        args.batch_size = 1024
        args.n_eval = 10_000

        args.output = (
            "results/"
            "masked_edge_intervention_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )