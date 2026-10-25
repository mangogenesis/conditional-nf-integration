import argparse
import copy
import json
import time
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

DEFAULT_BENCHMARKS = [
    "single_gaussian",
    "sharp_broad",
    "curved_ridge",
    "xshape",
]

DEFAULT_DIMS = [4, 8]
DEFAULT_SEEDS = [0]

MODELS = [
    "independent",
    "markov1",
    "markov2",
    "full_ar",
]

COND_DIM = 3

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

OUTPUT_DIR = Path(
    "results/copula_structure_benchmark"
)


# ============================================================
# Held-out conditions
# ============================================================

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

def parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


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
# Structured copula flow
# ============================================================

class StructuredCopulaFlow(nn.Module):

    def __init__(
        self,
        dim,
        structure,
        cond_dim=3,
        hidden=128,
        n_bins=32,
    ):
        super().__init__()

        self.dim = dim
        self.structure = structure
        self.cond_dim = cond_dim

        # ----------------------------------------------------
        # Marginals
        # ----------------------------------------------------

        self.marginals = nn.ModuleList(
            [
                ConditionalCubicFlow1D(
                    cond_dim=cond_dim,
                    hidden=hidden,
                    n_bins=n_bins,
                )
                for _ in range(dim)
            ]
        )

        # ----------------------------------------------------
        # Copula conditionals
        # ----------------------------------------------------

        self.copulas = nn.ModuleList()

        if structure == "independent":
            return

        for j in range(
            1,
            dim,
        ):

            if structure == "markov1":

                # U_j | U_{j-1}, c
                context_dim = (
                    cond_dim + 1
                )

            elif structure == "markov2":

                # U_j | U_{j-2}, U_{j-1}, c
                context_dim = (
                    cond_dim
                    + min(2, j)
                )

            elif structure == "full_ar":

                # U_j | U_1,...,U_{j-1},c
                context_dim = (
                    cond_dim + j
                )

            else:

                raise ValueError(
                    f"Unknown structure: "
                    f"{structure}"
                )

            self.copulas.append(
                ConditionalCubicFlow1D(
                    cond_dim=context_dim,
                    hidden=hidden,
                    n_bins=n_bins,
                )
            )

    # ========================================================
    # Context
    # ========================================================

    def copula_context(
        self,
        u,
        cond,
        j,
    ):

        if self.structure == "markov1":

            dependence = (
                u[:, j-1:j]
            )

        elif self.structure == "markov2":

            start = max(
                0,
                j - 2,
            )

            dependence = (
                u[:, start:j]
            )

        elif self.structure == "full_ar":

            dependence = (
                u[:, :j]
            )

        else:

            raise ValueError(
                self.structure
            )

        return torch.cat(
            [
                dependence,
                cond,
            ],
            dim=1,
        )

    # ========================================================
    # x -> u
    # ========================================================

    def x_to_u(
        self,
        x,
        cond,
    ):

        u_list = []

        marginal_logq = torch.zeros(
            x.shape[0],
            device=x.device,
        )

        for j in range(
            self.dim
        ):

            uj, forward_logdet = (
                self.marginals[
                    j
                ].inverse(
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

            marginal_logq -= (
                forward_logdet.reshape(
                    -1
                )
            )

        u = torch.cat(
            u_list,
            dim=1,
        )

        return (
            u,
            marginal_logq,
        )

    # ========================================================
    # log q
    # ========================================================

    def log_prob(
        self,
        x,
        cond,
    ):

        u, logq = self.x_to_u(
            x,
            cond,
        )

        if (
            self.structure
            == "independent"
        ):
            return logq

        for j in range(
            1,
            self.dim,
        ):

            context = (
                self.copula_context(
                    u,
                    cond,
                    j,
                )
            )

            _, forward_logdet = (
                self.copulas[
                    j - 1
                ].inverse(
                    u[:, j:j+1],
                    context,
                )
            )

            logq -= (
                forward_logdet.reshape(
                    -1
                )
            )

        return logq

    # ========================================================
    # Sample
    # ========================================================

    @torch.no_grad()
    def sample(
        self,
        n,
        cond,
    ):

        if cond.shape[0] == 1:
            cond = cond.expand(
                n,
                -1,
            )

        if cond.shape[0] != n:
            raise ValueError(
                "cond must have 1 row "
                "or n rows"
            )

        device = cond.device

        v = torch.rand(
            n,
            self.dim,
            device=device,
            dtype=cond.dtype,
        )

        if (
            self.structure
            == "independent"
        ):

            u = v

        else:

            u = torch.zeros_like(
                v
            )

            u[:, 0] = v[:, 0]

            for j in range(
                1,
                self.dim,
            ):

                # Previous coordinates are guaranteed
                # to remain inside the copula cube.
                u[:, :j] = (
                    u[:, :j]
                    .clamp(
                        0.0,
                        1.0,
                    )
                )

                context = (
                    self.copula_context(
                        u,
                        cond,
                        j,
                    )
                )

                uj, _ = (
                    self.copulas[
                        j - 1
                    ].forward(
                        v[:, j:j+1],
                        context,
                    )
                )

                uj = uj.clamp(
                    0.0,
                    1.0,
                )

                u[:, j] = (
                    uj.reshape(-1)
                )

        u = u.clamp(
            0.0,
            1.0,
        )

        x_list = []

        for j in range(
            self.dim
        ):

            xj, _ = (
                self.marginals[
                    j
                ].forward(
                    u[:, j:j+1],
                    cond,
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
# Marginal training
# ============================================================

def train_marginals(
    model,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
):

    model.train()

    optimizer = torch.optim.Adam(
        model.marginals.parameters(),
        lr=LR_MARGINAL,
    )

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
            dim,
        )

        loss = 0.0

        for j in range(
            dim
        ):

            _, forward_logdet = (
                model.marginals[
                    j
                ].inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            loss += (
                forward_logdet.mean()
            )

        loss = (
            loss / dim
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.marginals.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        recent.append(
            loss.item()
        )

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"    marginals "
                f"{step+1:5d}/{steps} "
                f"loss="
                f"{np.mean(recent[-100:]):.6f}"
            )


# ============================================================
# Copula training
# ============================================================

def train_copula(
    model,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
):

    if (
        model.structure
        == "independent"
    ):
        return

    for p in (
        model.marginals.parameters()
    ):
        p.requires_grad_(
            False
        )

    for p in (
        model.copulas.parameters()
    ):
        p.requires_grad_(
            True
        )

    optimizer = torch.optim.Adam(
        model.copulas.parameters(),
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

        x = sample_target(
            benchmark,
            cond,
            dim,
        )

        with torch.no_grad():

            u, _ = model.x_to_u(
                x,
                cond,
            )

        loss = 0.0

        for j in range(
            1,
            dim,
        ):

            context = (
                model.copula_context(
                    u,
                    cond,
                    j,
                )
            )

            _, forward_logdet = (
                model.copulas[
                    j - 1
                ].inverse(
                    u[:, j:j+1],
                    context,
                )
            )

            loss += (
                forward_logdet.mean()
            )

        loss = (
            loss
            / (dim - 1)
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.copulas.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        recent.append(
            loss.item()
        )

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"    copula "
                f"{step+1:5d}/{steps} "
                f"loss="
                f"{np.mean(recent[-100:]):.6f}"
            )


# ============================================================
# Forward KL
# ============================================================

@torch.no_grad()
def estimate_forward_kl(
    model,
    benchmark,
    cond,
    dim,
    device,
    n,
):

    cond_batch = (
        cond
        .to(device)
        .reshape(
            1,
            -1,
        )
        .expand(
            n,
            -1,
        )
    )

    x = sample_target(
        benchmark,
        cond_batch,
        dim,
    )

    logp = target_logpdf(
        benchmark,
        x,
        cond_batch,
    )

    logq = model.log_prob(
        x,
        cond_batch,
    )

    values = (
        logp - logq
    )

    return {
        "forward_kl":
            values.mean().item(),

        "forward_kl_mc_se":
            (
                values.std(
                    unbiased=True
                )
                / np.sqrt(n)
            ).item(),
    }


# ============================================================
# IS evaluation
# ============================================================

@torch.no_grad()
def evaluate_is(
    model,
    benchmark,
    cond,
    dim,
    device,
    n_eval,
):

    model.eval()

    cond = (
        cond
        .to(device)
        .reshape(
            1,
            -1,
        )
    )

    cond_batch = cond.expand(
        n_eval,
        -1,
    )

    synchronize(
        device
    )

    start = time.perf_counter()

    x = model.sample(
        n_eval,
        cond,
    )

    logq = model.log_prob(
        x,
        cond_batch,
    )

    logp = target_logpdf(
        benchmark,
        x,
        cond_batch,
    )

    synchronize(
        device
    )

    eval_time = (
        time.perf_counter()
        - start
    )

    logw = (
        logp - logq
    )

    if not torch.isfinite(
        logw
    ).all():

        raise RuntimeError(
            "Non-finite importance weights"
        )

    shifted = (
        logw
        - torch.max(
            logw
        )
    )

    scaled = torch.exp(
        shifted
    )

    ess = (
        scaled.sum().pow(2)
        /
        scaled.pow(2).sum()
    )

    ess_fraction = (
        ess.item()
        / n_eval
    )

    weights = torch.exp(
        logw.double()
    )

    estimate = (
        weights.mean()
    )

    weight_variance = (
        weights.var(
            unbiased=True
        )
    )

    variance = (
        weight_variance
        / n_eval
    )

    return {
        "integral_estimate":
            estimate.item(),

        "relative_error":
            abs(
                estimate.item()
                - 1.0
            ),

        "variance":
            variance.item(),

        "ess_fraction":
            ess_fraction,

        "eval_time":
            eval_time,
    }


# ============================================================
# Aggregation
# ============================================================

def build_summary(df):

    return (
        df
        .groupby(
            [
                "benchmark",
                "model",
                "dimension",
            ]
        )
        .agg(
            ess_mean=(
                "ess_fraction",
                "mean",
            ),

            ess_std=(
                "ess_fraction",
                "std",
            ),

            forward_kl_mean=(
                "forward_kl",
                "mean",
            ),

            forward_kl_std=(
                "forward_kl",
                "std",
            ),

            variance_mean=(
                "variance",
                "mean",
            ),

            variance_median=(
                "variance",
                "median",
            ),

            relative_error_mean=(
                "relative_error",
                "mean",
            ),

            relative_error_median=(
                "relative_error",
                "median",
            ),

            parameter_count_mean=(
                "parameter_count",
                "mean",
            ),

            marginal_train_time_mean=(
                "marginal_train_time",
                "mean",
            ),

            copula_train_time_mean=(
                "copula_train_time",
                "mean",
            ),

            total_train_time_mean=(
                "total_train_time",
                "mean",
            ),
        )
        .reset_index()
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

    config = {
        "benchmarks":
            args.benchmarks,

        "dimensions":
            args.dims,

        "seeds":
            args.seeds,

        "models":
            args.models,

        "n_bins":
            N_BINS,

        "hidden":
            HIDDEN,

        "marginal_steps":
            args.marginal_steps,

        "copula_steps":
            args.copula_steps,

        "batch_size":
            args.batch_size,

        "n_eval":
            args.n_eval,

        "n_kl":
            args.n_kl,

        "test_conditions":
            TEST_CONDITIONS.tolist(),
    }

    with open(
        output_dir
        / "config.json",
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    rows = []

    total_groups = (
        len(args.benchmarks)
        * len(args.dims)
        * len(args.seeds)
    )

    group_index = 0

    for benchmark in (
        args.benchmarks
    ):

        for dim in (
            args.dims
        ):

            for seed in (
                args.seeds
            ):

                group_index += 1

                print()
                print(
                    "=" * 78
                )

                print(
                    f"[{group_index}/"
                    f"{total_groups}] "
                    f"{benchmark} "
                    f"| d={dim} "
                    f"| seed={seed}"
                )

                print(
                    "=" * 78
                )

                seed_everything(
                    seed
                )

                # =================================================
                # Shared marginals
                #
                # All dependence models receive exactly the same
                # fitted marginals.
                # =================================================

                shared_model = (
                    StructuredCopulaFlow(
                        dim=dim,
                        structure=
                            "independent",
                        cond_dim=COND_DIM,
                        hidden=HIDDEN,
                        n_bins=N_BINS,
                    )
                    .to(device)
                )

                print(
                    "\nTraining shared marginals..."
                )

                synchronize(
                    device
                )

                start = (
                    time.perf_counter()
                )

                train_marginals(
                    model=shared_model,
                    benchmark=benchmark,
                    dim=dim,
                    device=device,
                    steps=
                        args.marginal_steps,
                    batch_size=
                        args.batch_size,
                )

                synchronize(
                    device
                )

                marginal_train_time = (
                    time.perf_counter()
                    - start
                )

                marginal_state = (
                    copy.deepcopy(
                        shared_model
                        .marginals
                        .state_dict()
                    )
                )

                del shared_model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # =================================================
                # Dependence structures
                # =================================================

                for model_name in (
                    args.models
                ):

                    print()
                    print(
                        "-" * 70
                    )

                    print(
                        f"Model: "
                        f"{model_name}"
                    )

                    print(
                        "-" * 70
                    )

                    seed_everything(
                        seed
                    )

                    model = (
                        StructuredCopulaFlow(
                            dim=dim,
                            structure=
                                model_name,
                            cond_dim=COND_DIM,
                            hidden=HIDDEN,
                            n_bins=N_BINS,
                        )
                        .to(device)
                    )

                    model.marginals.load_state_dict(
                        marginal_state
                    )

                    copula_train_time = 0.0

                    if (
                        model_name
                        != "independent"
                    ):

                        print(
                            "Training copula..."
                        )

                        synchronize(
                            device
                        )

                        start = (
                            time.perf_counter()
                        )

                        train_copula(
                            model=model,
                            benchmark=benchmark,
                            dim=dim,
                            device=device,
                            steps=
                                args.copula_steps,
                            batch_size=
                                args.batch_size,
                        )

                        synchronize(
                            device
                        )

                        copula_train_time = (
                            time.perf_counter()
                            - start
                        )

                    total_train_time = (
                        marginal_train_time
                        +
                        copula_train_time
                    )

                    n_params = (
                        parameter_count(
                            model
                        )
                    )

                    # =============================================
                    # Held-out conditions
                    # =============================================

                    for (
                        condition_id,
                        cond,
                    ) in enumerate(
                        TEST_CONDITIONS
                    ):

                        kl = (
                            estimate_forward_kl(
                                model=model,
                                benchmark=
                                    benchmark,
                                cond=cond,
                                dim=dim,
                                device=device,
                                n=args.n_kl,
                            )
                        )

                        metrics = (
                            evaluate_is(
                                model=model,
                                benchmark=
                                    benchmark,
                                cond=cond,
                                dim=dim,
                                device=device,
                                n_eval=
                                    args.n_eval,
                            )
                        )

                        row = {
                            "benchmark":
                                benchmark,

                            "model":
                                model_name,

                            "dimension":
                                dim,

                            "seed":
                                seed,

                            "condition_id":
                                condition_id,

                            "condition_0":
                                float(
                                    cond[0]
                                ),

                            "condition_1":
                                float(
                                    cond[1]
                                ),

                            "condition_2":
                                float(
                                    cond[2]
                                ),

                            "forward_kl":
                                kl[
                                    "forward_kl"
                                ],

                            "forward_kl_mc_se":
                                kl[
                                    "forward_kl_mc_se"
                                ],

                            "integral_estimate":
                                metrics[
                                    "integral_estimate"
                                ],

                            "relative_error":
                                metrics[
                                    "relative_error"
                                ],

                            "variance":
                                metrics[
                                    "variance"
                                ],

                            "ess_fraction":
                                metrics[
                                    "ess_fraction"
                                ],

                            "parameter_count":
                                n_params,

                            "marginal_train_time":
                                marginal_train_time,

                            "copula_train_time":
                                copula_train_time,

                            "total_train_time":
                                total_train_time,

                            "eval_time":
                                metrics[
                                    "eval_time"
                                ],
                        }

                        rows.append(
                            row
                        )

                        print(
                            f"  c{condition_id:02d} "
                            f"KL="
                            f"{row['forward_kl']:.4f} "
                            f"ESS/N="
                            f"{row['ess_fraction']:.3f} "
                            f"rel="
                            f"{row['relative_error']:.3e}"
                        )

                    # Save continuously.
                    pd.DataFrame(
                        rows
                    ).to_csv(
                        output_dir
                        / "raw_results.csv",
                        index=False,
                    )

                    del model

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # ========================================================
    # Summary
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    summary = build_summary(
        raw_df
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "STRUCTURE BENCHMARK COMPLETE"
    )

    print(
        "=" * 78
    )

    print()

    columns = [
        "benchmark",
        "model",
        "dimension",
        "ess_mean",
        "forward_kl_mean",
        "variance_mean",
        "relative_error_mean",
        "parameter_count_mean",
        "total_train_time_mean",
    ]

    print(
        summary[
            columns
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
        "--dims",
        nargs="+",
        type=int,
        default=
            DEFAULT_DIMS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=
            DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=
            MODELS,
        choices=
            MODELS,
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

        args.benchmarks = [
            "single_gaussian",
            "curved_ridge",
        ]

        args.dims = [4]

        args.seeds = [0]

        args.models = [
            "independent",
            "markov1",
            "full_ar",
        ]

        args.marginal_steps = 100
        args.copula_steps = 150

        args.batch_size = 1024

        args.n_eval = 10_000
        args.n_kl = 5_000

        args.output = (
            "results/"
            "copula_structure_benchmark_smoke"
        )

    return args


if __name__ == "__main__":

    args = parse_args()

    run(args)