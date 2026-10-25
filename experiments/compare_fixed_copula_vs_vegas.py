import argparse
import copy
import math
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import vegas


# ============================================================
# Import SAME benchmark / NF definitions used in pruning
# ============================================================

from run_fast_multijump_pruning import (
    DIM,
    MarginalBank,
    make_real_model,
    sample_target,
    target_logpdf,
    sample_from_model,
    build_kl_cache,
    train_marginals_until_converged,
    train_model_until_converged,
    seed_everything,
)


# ============================================================
# Fixed selected graphs from previous pruning experiment
# ============================================================

FINAL_GRAPHS = {

    "branching_tree": {
        0: [],
        1: [0],
        2: [0, 1],
        3: [0, 1, 2],
        4: [1, 3],
        5: [1, 2, 4],
        6: [2, 5],
        7: [5, 6],
    },

    "skip_lag": {
        0: [],
        1: [0],
        2: [0, 1],
        3: [1],
        4: [2],
        5: [3, 4],
        6: [4],
        7: [2, 4, 5],
    },
}


# ============================================================
# Diagnostic pairs
#
# These are NOT used by training or graph selection.
# They are only for visualization after training.
# ============================================================

PLOT_PAIRS = {

    "branching_tree": [
        (0, 1),
        (1, 3),
        (2, 6),
        (6, 7),
    ],

    "skip_lag": [
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 7),
    ],
}


# ============================================================
# Configuration
# ============================================================

TRAIN_BATCH_SIZE = 100_000

FIXED_CONDITION = [
    0.5,
    0.5,
    0.5,
]


# ------------------------------------------------------------
# NF production budgets
# ------------------------------------------------------------

NF_SAMPLE_SIZES = [
    100_000,
    1_000_000,
]

NF_REPEATS = 5


# ------------------------------------------------------------
# VEGAS adaptation sweep
# ------------------------------------------------------------

VEGAS_ADAPT_NITN = [
    10,
    20,
    50,
    100,
    200,
]

VEGAS_ADAPT_NEVAL = 100_000


# ------------------------------------------------------------
# VEGAS fixed production budget
#
# 10 * 100,000 = 1 million
# ------------------------------------------------------------

VEGAS_PROD_NITN = 10
VEGAS_PROD_NEVAL = 100_000

VEGAS_REPEATS = 5


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

PLOT_SAMPLES = 50_000
WEIGHT_PLOT_SAMPLES = 100_000


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    "results/fixed_copula_vs_vegas_d8"
)


# ============================================================
# Utility
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


# ============================================================
# Training argument object
#
# Keeps previous convergence rules:
#
# Marginal:
# relative validation NLL improvement < 1%
# for 2 consecutive checks.
#
# Copula:
# relative KL improvement < 1%
# for 2 consecutive checks.
# ============================================================

def make_training_args():

    return SimpleNamespace(

        # ----------------------------------------------------
        # General
        # ----------------------------------------------------

        batch_size=
            TRAIN_BATCH_SIZE,

        eval_seed=
            123456,

        performance_seed=
            0,

        # ----------------------------------------------------
        # Marginal convergence
        # ----------------------------------------------------

        marginal_val_size=
            100_000,

        min_marginal_steps=
            1200,

        max_marginal_steps=
            6000,

        marginal_check_interval=
            300,

        marginal_rel_tol=
            0.01,

        marginal_convergence_patience=
            2,

        # ----------------------------------------------------
        # Copula convergence
        # ----------------------------------------------------

        min_candidate_steps=
            1800,

        max_candidate_steps=
            12_000,

        kl_check_interval=
            600,

        rel_kl_tol=
            0.01,

        convergence_patience=
            2,
    )


# ============================================================
# Train one fixed selected graph
#
# No Bayesian model.
# No masked model.
# No PIP.
# No pruning search.
#
# We already know which graph we want to compare.
# ============================================================

def train_fixed_copula(
    benchmark,
    graph,
    device,
):

    args = (
        make_training_args()
    )

    seed = (
        args.performance_seed
    )

    print()
    print(
        "=" * 110
    )

    print(
        f"TRAINING FIXED COPULA: "
        f"{benchmark}"
    )

    print(
        f"device       = "
        f"{device}"
    )

    print(
        f"batch size   = "
        f"{args.batch_size:,}"
    )

    print(
        f"graph        = "
        f"{graph}"
    )

    print(
        "=" * 110
    )

    # ========================================================
    # 1. Train marginals
    # ========================================================

    seed_everything(
        seed
    )

    bank = (
        MarginalBank(
            DIM
        )
        .to(device)
    )

    print()
    print(
        "Training marginals..."
    )

    marginal_result = (
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

    marginal_state = (
        cpu_state_dict(
            bank
        )
    )

    print()
    print(
        f"Marginal training finished:"
    )

    print(
        f"  steps      = "
        f"{marginal_result['final_steps']}"
    )

    print(
        f"  converged  = "
        f"{marginal_result['converged']}"
    )

    print(
        f"  final NLL  = "
        f"{marginal_result['final_val_nll']:.6f}"
    )

    del bank

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # 2. Fixed KL cache
    # ========================================================

    print()
    print(
        "Building fixed 100k KL monitor cache..."
    )

    seed_everything(
        args.eval_seed
    )

    kl_cache = (
        build_kl_cache(
            benchmark=
                benchmark,

            n=
                100_000,

            device=
                device,
        )
    )

    # ========================================================
    # 3. Build fixed selected graph model
    # ========================================================

    seed_everything(
        seed
    )

    model = (
        make_real_model(
            graph=
                graph,

            marginal_state=
                marginal_state,

            device=
                device,
        )
    )

    # ========================================================
    # 4. Train copula
    # ========================================================

    print()
    print(
        "Training selected copula..."
    )

    copula_result = (
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

    print()
    print(
        "Copula training finished:"
    )

    print(
        f"  steps      = "
        f"{copula_result['final_steps']}"
    )

    print(
        f"  converged  = "
        f"{copula_result['converged']}"
    )

    print(
        f"  final KL   = "
        f"{copula_result['final_kl']:.6f}"
    )

    # ========================================================
    # Save everything needed to reconstruct on CPU
    # ========================================================

    checkpoint = {

        "benchmark":
            benchmark,

        "graph":
            copy.deepcopy(
                graph
            ),

        "marginal_state_dict":
            marginal_state,

        "model_state_dict":
            cpu_state_dict(
                model
            ),

        "marginal_steps":
            marginal_result[
                "final_steps"
            ],

        "marginal_converged":
            marginal_result[
                "converged"
            ],

        "copula_steps":
            copula_result[
                "final_steps"
            ],

        "copula_converged":
            copula_result[
                "converged"
            ],
    }

    del kl_cache

    return (
        model,
        checkpoint,
    )


# ============================================================
# Reconstruct saved model purely on CPU
# ============================================================

def reconstruct_cpu_model(
    checkpoint,
):

    device = torch.device(
        "cpu"
    )

    model = (
        make_real_model(
            graph=
                checkpoint[
                    "graph"
                ],

            marginal_state=
                checkpoint[
                    "marginal_state_dict"
                ],

            device=
                device,
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# True target samples at fixed condition
# ============================================================

@torch.no_grad()
def sample_true_fixed_condition(
    benchmark,
    condition,
    n_samples,
    seed,
):

    seed_everything(
        seed
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

    x = (
        sample_target(
            benchmark,
            cond,
            DIM,
        )
    )

    return (
        x.detach()
        .cpu()
        .numpy()
    )


# ============================================================
# NF samples at fixed condition
# ============================================================

@torch.no_grad()
def sample_nf_fixed_condition(
    model,
    condition,
    n_samples,
    seed,
):

    model.eval()

    seed_everything(
        seed
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

    x = (
        sample_from_model(
            model,
            n_samples,
            cond,
        )
    )

    return (
        x.detach()
        .cpu()
        .numpy()
    )


# ============================================================
# Plot target vs learned copula
# ============================================================

def make_copula_comparison_plots(
    benchmark,
    model,
    condition,
    output_dir,
    n_samples,
    seed,
):

    print()
    print(
        "Generating target vs learned copula plots..."
    )

    true_samples = (
        sample_true_fixed_condition(
            benchmark=
                benchmark,

            condition=
                condition,

            n_samples=
                n_samples,

            seed=
                seed,
        )
    )

    nf_samples = (
        sample_nf_fixed_condition(
            model=
                model,

            condition=
                condition,

            n_samples=
                n_samples,

            seed=
                seed + 1,
        )
    )

    plot_dir = (
        output_dir
        /
        f"{benchmark}_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for i, j in (
        PLOT_PAIRS[
            benchmark
        ]
    ):

        # ====================================================
        # True target plot
        # ====================================================

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

        plt.xlabel(
            f"$x_{i}$"
        )

        plt.ylabel(
            f"$x_{j}$"
        )

        plt.title(
            f"{benchmark}: true target\n"
            f"$x_{i}$ vs $x_{j}$, "
            f"c={condition}"
        )

        plt.xlim(
            0.0,
            1.0,
        )

        plt.ylim(
            0.0,
            1.0,
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

        # ====================================================
        # Learned copula plot
        # ====================================================

        plt.figure(
            figsize=(
                6,
                5,
            )
        )

        plt.hexbin(
            nf_samples[
                :,
                i
            ],

            nf_samples[
                :,
                j
            ],

            gridsize=
                70,

            mincnt=
                1,
        )

        plt.xlabel(
            f"$x_{i}$"
        )

        plt.ylabel(
            f"$x_{j}$"
        )

        plt.title(
            f"{benchmark}: learned copula\n"
            f"$x_{i}$ vs $x_{j}$, "
            f"c={condition}"
        )

        plt.xlim(
            0.0,
            1.0,
        )

        plt.ylim(
            0.0,
            1.0,
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

    print(
        f"Saved geometry plots to:"
    )

    print(
        plot_dir
    )


# ============================================================
# Importance-weight diagnostic
# ============================================================

@torch.no_grad()
def plot_log_weight_distribution(
    benchmark,
    model,
    condition,
    output_dir,
    n_samples,
    seed,
):

    model.eval()

    seed_everything(
        seed
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

    x = (
        sample_from_model(
            model,
            n_samples,
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
    )

    logq = (
        model.log_prob(
            x,
            cond,
        )
        .reshape(-1)
    )

    logw = (
        logp
        -
        logq
    )

    logw_np = (
        logw.detach()
        .cpu()
        .numpy()
    )

    plot_dir = (
        output_dir
        /
        f"{benchmark}_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.hist(
        logw_np,
        bins=
            100,

        density=
            True,
    )

    plt.axvline(
        0.0,
        linestyle=
            "--",
    )

    plt.xlabel(
        r"$\log p(x|c)-\log q_\theta(x|c)$"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        f"{benchmark}: importance-weight distribution"
    )

    plt.tight_layout()

    plt.savefig(
        plot_dir
        /
        "log_importance_weights.png",

        dpi=
            180,
    )

    plt.close()

    # Save values too.
    pd.DataFrame(
        {
            "log_weight":
                logw_np
        }
    ).to_csv(
        plot_dir
        /
        "log_importance_weights.csv",

        index=False,
    )


# ============================================================
# NF integration on CPU
# ============================================================

@torch.no_grad()
def evaluate_nf_fixed_condition(
    model,
    benchmark,
    condition,
    n_samples,
    seed,
    chunk_size,
):

    device = torch.device(
        "cpu"
    )

    model.to(
        device
    )

    model.eval()

    seed_everything(
        seed
    )

    sum_w = 0.0
    sum_w2 = 0.0

    count = 0

    start_time = (
        time.perf_counter()
    )

    while count < n_samples:

        n = min(
            chunk_size,
            n_samples
            -
            count,
        )

        cond = (
            torch.tensor(
                condition,
                dtype=torch.float32,
                device=device,
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

        # ====================================================
        # x ~ q(x|c)
        # ====================================================

        x = (
            sample_from_model(
                model,
                n,
                cond,
            )
        )

        # ====================================================
        # Target density
        # ====================================================

        logp = (
            target_logpdf(
                benchmark,
                x,
                cond,
            )
            .reshape(-1)
        )

        # ====================================================
        # Proposal density
        # ====================================================

        logq = (
            model.log_prob(
                x,
                cond,
            )
            .reshape(-1)
        )

        logw = (
            logp
            -
            logq
        )

        # ----------------------------------------------------
        # float64 weight accumulation
        # ----------------------------------------------------

        w = (
            torch.exp(
                logw
            )
            .double()
        )

        sum_w += (
            w.sum()
            .item()
        )

        sum_w2 += (
            w.square()
            .sum()
            .item()
        )

        count += n

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    # ========================================================
    # Integral estimate
    #
    # Exact answer is 1.
    # ========================================================

    estimate = (
        sum_w
        /
        count
    )

    actual_rel_error = abs(
        estimate
        -
        1.0
    )

    # ========================================================
    # Relative ESS
    # ========================================================

    ess = (
        sum_w
        *
        sum_w
        /
        (
            count
            *
            sum_w2
        )
    )

    # ========================================================
    # Relative weight variance
    # ========================================================

    relative_variance = (
        1.0
        /
        ess
        -
        1.0
    )

    estimated_rel_error = math.sqrt(
        relative_variance
        /
        count
    )

    return {

        "benchmark":
            benchmark,

        "method":
            "selected_copula",

        "seed":
            seed,

        "production_evals":
            count,

        "estimate":
            estimate,

        "actual_rel_error":
            actual_rel_error,

        "ess":
            ess,

        "relative_weight_variance":
            relative_variance,

        "estimated_rel_error":
            estimated_rel_error,

        "production_wall_time":
            elapsed,
    }


# ============================================================
# VEGAS integrand
# ============================================================

def make_vegas_integrand(
    benchmark,
    condition,
):

    condition_tensor = (
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

        x_np = np.asarray(
            x,
            dtype=np.float64,
        )

        x_torch = (
            torch.from_numpy(
                x_np
            )
            .to(
                dtype=torch.float64
            )
        )

        n = (
            x_torch.shape[0]
        )

        cond = (
            condition_tensor
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

            values = (
                torch.exp(
                    logp
                )
            )

        return (
            values
            .cpu()
            .numpy()
        )

    return integrand


# ============================================================
# VEGAS adaptation + production
# ============================================================

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

    integ = (
        vegas.Integrator(
            [
                [
                    0.0,
                    1.0,
                ]

                for _ in range(
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

    # ========================================================
    # Adaptation
    # ========================================================

    adaptation_start = (
        time.perf_counter()
    )

    adaptation_result = (
        integ(
            integrand,

            nitn=
                adapt_nitn,

            neval=
                adapt_neval,
        )
    )

    adaptation_time = (
        time.perf_counter()
        -
        adaptation_start
    )

    # ========================================================
    # Production
    # ========================================================

    production_start = (
        time.perf_counter()
    )

    result = (
        integ(
            integrand,

            nitn=
                prod_nitn,

            neval=
                prod_neval,
        )
    )

    production_time = (
        time.perf_counter()
        -
        production_start
    )

    estimate = float(
        result.mean
    )

    reported_std = float(
        result.sdev
    )

    actual_rel_error = abs(
        estimate
        -
        1.0
    )

    if estimate != 0.0:

        reported_rel_error = (
            reported_std
            /
            abs(
                estimate
            )
        )

    else:

        reported_rel_error = float(
            "inf"
        )

    return {

        "benchmark":
            benchmark,

        "method":
            "VEGAS",

        "seed":
            seed,

        "adapt_nitn":
            adapt_nitn,

        "adapt_neval":
            adapt_neval,

        "adaptation_evals":
            (
                adapt_nitn
                *
                adapt_neval
            ),

        "production_evals":
            (
                prod_nitn
                *
                prod_neval
            ),

        "estimate":
            estimate,

        "actual_rel_error":
            actual_rel_error,

        "reported_rel_error":
            reported_rel_error,

        "adaptation_Q":
            (
                float(
                    adaptation_result.Q
                )

                if adaptation_result.Q
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
            adaptation_time,

        "production_wall_time":
            production_time,

        "total_wall_time":
            (
                adaptation_time
                +
                production_time
            ),
    }


# ============================================================
# Run one benchmark
# ============================================================

def run_benchmark(
    benchmark,
    args,
    output_dir,
):

    if benchmark not in FINAL_GRAPHS:

        raise ValueError(
            f"No fixed graph stored for "
            f"{benchmark}"
        )

    graph = (
        FINAL_GRAPHS[
            benchmark
        ]
    )

    # ========================================================
    # Training device
    # ========================================================

    if torch.cuda.is_available():

        training_device = (
            torch.device(
                "cuda"
            )
        )

    else:

        training_device = (
            torch.device(
                "cpu"
            )
        )

    print()
    print(
        "#" * 120
    )

    print(
        f"BENCHMARK: "
        f"{benchmark}"
    )

    print(
        f"fixed condition = "
        f"{FIXED_CONDITION}"
    )

    print(
        f"selected graph  = "
        f"{graph}"
    )

    print(
        "#" * 120
    )

    # ========================================================
    # TRAIN NF
    # ========================================================

    trained_model, checkpoint = (
        train_fixed_copula(
            benchmark=
                benchmark,

            graph=
                graph,

            device=
                training_device,
        )
    )

    # ========================================================
    # Save checkpoint
    # ========================================================

    checkpoint_path = (
        output_dir
        /
        f"{benchmark}_fixed_copula.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print()
    print(
        f"Saved checkpoint:"
    )

    print(
        checkpoint_path
    )

    # ========================================================
    # Remove GPU model
    # ========================================================

    del trained_model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # Reconstruct NF on CPU
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "SWITCHING NF EVALUATION TO CPU"
    )

    print(
        "=" * 100
    )

    nf_model = (
        reconstruct_cpu_model(
            checkpoint
        )
    )

    # ========================================================
    # Make diagnostic plots
    # ========================================================

    make_copula_comparison_plots(
        benchmark=
            benchmark,

        model=
            nf_model,

        condition=
            FIXED_CONDITION,

        output_dir=
            output_dir,

        n_samples=
            args.plot_samples,

        seed=
            args.seed + 7000,
    )

    plot_log_weight_distribution(
        benchmark=
            benchmark,

        model=
            nf_model,

        condition=
            FIXED_CONDITION,

        output_dir=
            output_dir,

        n_samples=
            args.weight_plot_samples,

        seed=
            args.seed + 8000,
    )

    # ========================================================
    # NF production evaluation
    # ========================================================

    nf_rows = []

    for n_samples in (
        args.nf_samples
    ):

        for repeat in range(
            args.nf_repeats
        ):

            seed = (
                args.seed
                +
                100_000
                *
                repeat
                +
                n_samples
            )

            print()
            print(
                "-" * 100
            )

            print(
                f"NF | "
                f"{benchmark} | "
                f"N={n_samples:,} | "
                f"repeat={repeat}"
            )

            row = (
                evaluate_nf_fixed_condition(
                    model=
                        nf_model,

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
                row
            )

            print(
                f"  estimate          = "
                f"{row['estimate']:.10f}"
            )

            print(
                f"  actual error      = "
                f"{100.0 * row['actual_rel_error']:.6f}%"
            )

            print(
                f"  relative ESS      = "
                f"{row['ess']:.6f}"
            )

            print(
                f"  estimated error   = "
                f"{100.0 * row['estimated_rel_error']:.6f}%"
            )

            print(
                f"  CPU time          = "
                f"{row['production_wall_time']:.4f} s"
            )

    # ========================================================
    # VEGAS evaluation
    # ========================================================

    vegas_rows = []

    for adapt_nitn in (
        args.vegas_adapt_nitn
    ):

        for repeat in range(
            args.vegas_repeats
        ):

            seed = (
                args.seed
                +
                10_000_000
                +
                100_000
                *
                repeat
                +
                adapt_nitn
            )

            print()
            print(
                "-" * 100
            )

            print(
                f"VEGAS | "
                f"{benchmark} | "
                f"adapt="
                f"{adapt_nitn}"
                f"x"
                f"{args.vegas_adapt_neval:,} | "
                f"repeat={repeat}"
            )

            row = (
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
                        seed,
                )
            )

            vegas_rows.append(
                row
            )

            print(
                f"  estimate          = "
                f"{row['estimate']:.10f}"
            )

            print(
                f"  actual error      = "
                f"{100.0 * row['actual_rel_error']:.6f}%"
            )

            print(
                f"  reported error    = "
                f"{100.0 * row['reported_rel_error']:.6f}%"
            )

            print(
                f"  production Q      = "
                f"{row['production_Q']:.6e}"
            )

            print(
                f"  total CPU time    = "
                f"{row['total_wall_time']:.4f} s"
            )

    # ========================================================
    # Save raw results
    # ========================================================

    nf_df = (
        pd.DataFrame(
            nf_rows
        )
    )

    vegas_df = (
        pd.DataFrame(
            vegas_rows
        )
    )

    nf_df.to_csv(
        output_dir
        /
        f"{benchmark}_nf_raw.csv",

        index=False,
    )

    vegas_df.to_csv(
        output_dir
        /
        f"{benchmark}_vegas_raw.csv",

        index=False,
    )

    # ========================================================
    # Aggregate NF
    # ========================================================

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
                "actual_rel_error",
                "mean",
            ),

            actual_error_std=(
                "actual_rel_error",
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
                "estimated_rel_error",
                "mean",
            ),

            production_time_mean=(
                "production_wall_time",
                "mean",
            ),
        )
    )

    # ========================================================
    # Aggregate VEGAS
    # ========================================================

    vegas_summary = (
        vegas_df
        .groupby(
            [
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
                "actual_rel_error",
                "mean",
            ),

            actual_error_std=(
                "actual_rel_error",
                "std",
            ),

            reported_error_mean=(
                "reported_rel_error",
                "mean",
            ),

            production_Q_mean=(
                "production_Q",
                "mean",
            ),

            adaptation_time_mean=(
                "adaptation_wall_time",
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

    nf_summary.to_csv(
        output_dir
        /
        f"{benchmark}_nf_summary.csv",

        index=False,
    )

    vegas_summary.to_csv(
        output_dir
        /
        f"{benchmark}_vegas_summary.csv",

        index=False,
    )

    # ========================================================
    # Build one direct comparison CSV
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
                    (
                        f"VEGAS_"
                        f"{row.adapt_nitn}"
                        f"x"
                        f"{row.adapt_neval}"
                    ),

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

    # ========================================================
    # Print final comparison
    # ========================================================

    print()
    print(
        "=" * 130
    )

    print(
        f"ACTUAL ERROR COMPARISON: "
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
        "=" * 130
    )

    print()
    print(
        "COPULA NF:"
    )

    print()

    print(
        nf_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "VEGAS:"
    )

    print()

    print(
        vegas_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "DIRECT COMPARISON:"
    )

    print()

    print(
        comparison_df.to_string(
            index=False
        )
    )

    del nf_model

    return (
        nf_summary,
        vegas_summary,
        comparison_df,
    )


# ============================================================
# Main
# ============================================================

def run(args):

    output_dir = (
        Path(
            args.output
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_comparisons = []

    for benchmark in (
        args.benchmarks
    ):

        (
            _,
            _,
            comparison_df,
        ) = (
            run_benchmark(
                benchmark=
                    benchmark,

                args=
                    args,

                output_dir=
                    output_dir,
            )
        )

        all_comparisons.append(
            comparison_df
        )

    combined = (
        pd.concat(
            all_comparisons,
            ignore_index=True,
        )
    )

    combined.to_csv(
        output_dir
        /
        "combined_comparison.csv",

        index=False,
    )

    print()
    print(
        "=" * 130
    )

    print(
        "ALL BENCHMARKS COMPLETE"
    )

    print(
        "=" * 130
    )

    print()

    print(
        combined.to_string(
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

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "branching_tree",
            "skip_lag",
        ],
    )

    # ========================================================
    # NF
    # ========================================================

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

    # ========================================================
    # VEGAS
    # ========================================================

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

    # ========================================================
    # CPU NF evaluation
    # ========================================================

    parser.add_argument(
        "--cpu-chunk-size",
        type=int,
        default=
            100_000,
    )

    # ========================================================
    # Plots
    # ========================================================

    parser.add_argument(
        "--plot-samples",
        type=int,
        default=
            PLOT_SAMPLES,
    )

    parser.add_argument(
        "--weight-plot-samples",
        type=int,
        default=
            WEIGHT_PLOT_SAMPLES,
    )

    # ========================================================
    # General
    # ========================================================

    parser.add_argument(
        "--seed",
        type=int,
        default=
            123456,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            OUTPUT_DIR,
    )

    return (
        parser.parse_args()
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    run(
        parse_args()
    )