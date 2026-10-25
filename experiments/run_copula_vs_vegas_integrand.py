import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import vegas

from run_copula_bin_sweep import (
    MarginalCopulaFlow,
    target_logpdf,
    parameter_count,
    seed_everything,
)


# ============================================================
# Frozen architecture
# ============================================================

N_BINS = 32
COND_DIM = 3
HIDDEN = 128

DEFAULT_BENCHMARKS = [
    "single_gaussian",
    "sharp_broad",
    "curved_ridge",
    "xshape",
]

DEFAULT_DIMS = [2, 4, 8, 16]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


# ============================================================
# NF training
# ============================================================

TRAIN_STEPS = 3000
BATCH_SIZE = 4096

LR = 1e-4
GRAD_CLIP = 10.0

EPS_START = 0.50
EPS_END = 0.05


# ============================================================
# Final evaluation
# ============================================================

N_EVAL = 100_000


# ============================================================
# VEGAS
# ============================================================

VEGAS_ADAPT_NITN = 10
VEGAS_ADAPT_NEVAL = 10_000

VEGAS_FINAL_NITN = 1
VEGAS_FINAL_NEVAL = 100_000


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


OUTPUT_DIR = Path(
    "results/copula_vs_vegas_integrand"
)


# ============================================================
# Safe high-dimensional copula flow
# ============================================================

class SafeMarginalCopulaFlow(MarginalCopulaFlow):
    """
    Same model as MarginalCopulaFlow, but with numerical
    protection during sampling.

    Mathematically every copula coordinate belongs to [0,1].
    Float32 cubic evaluation can occasionally produce values
    such as 1.000000119 or -1e-8.

    We therefore clamp intermediate copula coordinates before
    feeding them into later copula layers or marginal flows.
    """

    @torch.no_grad()
    def sample(
        self,
        n,
        cond,
    ):

        device = cond.device

        if cond.shape[0] == 1:
            cond = cond.expand(
                n,
                -1,
            )

        if cond.shape[0] != n:
            raise ValueError(
                "cond must have either 1 row or n rows"
            )

        # ----------------------------------------------------
        # Base variables
        # ----------------------------------------------------

        v = torch.rand(
            n,
            self.dim,
            device=device,
            dtype=cond.dtype,
        )

        u = torch.zeros_like(v)

        # First copula coordinate is identity.
        u[:, 0] = v[:, 0]

        # ----------------------------------------------------
        # Autoregressive copula
        # ----------------------------------------------------

        for j in range(
            1,
            self.dim,
        ):

            # Numerical protection on all previous coordinates
            # before using them as conditioning variables.
            previous_u = u[:, :j].clamp(
                0.0,
                1.0,
            )

            context = torch.cat(
                [
                    previous_u,
                    cond,
                ],
                dim=1,
            )

            uj, _ = self.copulas[
                j - 1
            ].forward(
                v[:, j:j+1],
                context,
            )

            # ------------------------------------------------
            # IMPORTANT FIX
            # ------------------------------------------------
            #
            # Cubic map is mathematically [0,1] -> [0,1],
            # but float32 may overshoot very slightly.
            #
            # Example:
            #
            #   1.000000119
            #
            # This is not a physical/model change; it simply
            # enforces the mathematical codomain.
            # ------------------------------------------------

            uj = uj.clamp(
                0.0,
                1.0,
            )

            u[:, j] = uj.reshape(-1)

        # Final protection before marginal transforms.
        u = u.clamp(
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # Marginal transforms u -> x
        # ----------------------------------------------------

        x_list = []

        for j in range(
            self.dim
        ):

            uj = u[:, j:j+1].clamp(
                0.0,
                1.0,
            )

            xj, _ = self.marginals[
                j
            ].forward(
                uj,
                cond,
            )

            # The physical output is also intended to remain
            # inside the unit cube.
            xj = xj.clamp(
                0.0,
                1.0,
            )

            x_list.append(
                xj
            )

        return torch.cat(
            x_list,
            dim=1,
        )


# ============================================================
# Utilities
# ============================================================

def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize()


def make_condition_dict(cond):

    return {
        "condition_0":
            float(cond[0]),

        "condition_1":
            float(cond[1]),

        "condition_2":
            float(cond[2]),
    }


def defensive_epsilon(
    step,
    total_steps,
):

    if total_steps <= 1:
        return EPS_END

    t = (
        step
        / (total_steps - 1)
    )

    return (
        EPS_START * (1.0 - t)
        + EPS_END * t
    )


# ============================================================
# Integrand-only NF training
# ============================================================

def train_integrand_only(
    model,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
):

    """
    Train using only evaluations of f(x|c).

    Defensive proposal:

        r_theta
          = (1-eps) q_theta
            + eps U

    Sample:

        x ~ r_theta

    Importance weight:

        w = f / r_theta

    Objective:

        L = - sum_i wbar_i log q_theta(x_i|c)

    where the normalized weights are detached.
    """

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    recent_loss = []
    recent_ess = []

    for step in range(
        steps
    ):

        # ----------------------------------------------------
        # Random training conditions
        # ----------------------------------------------------

        cond = torch.rand(
            batch_size,
            COND_DIM,
            device=device,
        )

        eps = defensive_epsilon(
            step,
            steps,
        )

        # ----------------------------------------------------
        # Draw from defensive mixture
        # ----------------------------------------------------

        with torch.no_grad():

            x_q = model.sample(
                batch_size,
                cond,
            )

            x_uniform = torch.rand(
                batch_size,
                dim,
                device=device,
            )

            choose_uniform = (
                torch.rand(
                    batch_size,
                    device=device,
                )
                < eps
            )

            x = torch.where(
                choose_uniform[:, None],
                x_uniform,
                x_q,
            )

            x = x.clamp(
                0.0,
                1.0,
            )

        # ----------------------------------------------------
        # Current proposal density
        # ----------------------------------------------------

        logq = model.log_prob(
            x,
            cond,
        )

        # ----------------------------------------------------
        # Integrand evaluation
        # ----------------------------------------------------

        with torch.no_grad():

            logf = target_logpdf(
                benchmark,
                x,
                cond,
            )

        # ----------------------------------------------------
        # Defensive proposal density
        #
        # Uniform density on [0,1]^d:
        #
        # U(x) = 1
        #
        # therefore log U = 0.
        # ----------------------------------------------------

        log_r = torch.logaddexp(
            logq.detach()
            + math.log(
                1.0 - eps
            ),

            torch.full_like(
                logq,
                math.log(eps),
            ),
        )

        # ----------------------------------------------------
        # Importance weights
        # ----------------------------------------------------

        logw = (
            logf
            - log_r
        )

        if not torch.isfinite(
            logw
        ).all():

            n_bad = (
                ~torch.isfinite(
                    logw
                )
            ).sum().item()

            raise RuntimeError(
                f"Training produced "
                f"{n_bad} non-finite "
                f"log weights."
            )

        normalized_weights = (
            torch.softmax(
                logw,
                dim=0,
            )
            .detach()
        )

        # ----------------------------------------------------
        # Weighted maximum likelihood
        # ----------------------------------------------------

        loss = -torch.sum(
            normalized_weights
            * logq
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        with torch.no_grad():

            ess = (
                1.0
                / normalized_weights
                .pow(2)
                .sum()
            )

            ess_fraction = (
                ess.item()
                / batch_size
            )

        recent_loss.append(
            loss.item()
        )

        recent_ess.append(
            ess_fraction
        )

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"  step "
                f"{step+1:5d}/"
                f"{steps} "
                f"loss="
                f"{np.mean(recent_loss[-100:]):.6f} "
                f"train ESS/N="
                f"{np.mean(recent_ess[-100:]):.3f} "
                f"eps={eps:.3f}"
            )


# ============================================================
# NF evaluation
# ============================================================

@torch.no_grad()
def evaluate_nf(
    model,
    benchmark,
    dim,
    cond,
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

    start = (
        time.perf_counter()
    )

    x = model.sample(
        n_eval,
        cond,
    )

    x = x.clamp(
        0.0,
        1.0,
    )

    logq = model.log_prob(
        x,
        cond_batch,
    )

    logf = target_logpdf(
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
        logf
        - logq
    )

    if not torch.isfinite(
        logw
    ).all():

        n_bad = (
            ~torch.isfinite(
                logw
            )
        ).sum().item()

        raise RuntimeError(
            f"NF evaluation produced "
            f"{n_bad} non-finite "
            f"log weights."
        )

    # --------------------------------------------------------
    # Stable ESS
    # --------------------------------------------------------

    shifted = (
        logw
        - torch.max(logw)
    )

    scaled_w = torch.exp(
        shifted
    )

    ess = (
        scaled_w.sum().pow(2)
        /
        scaled_w.pow(2).sum()
    )

    ess_fraction = (
        ess.item()
        / n_eval
    )

    # --------------------------------------------------------
    # Integral estimate in float64
    # --------------------------------------------------------

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

    estimator_variance = (
        weight_variance
        / n_eval
    )

    standard_error = (
        torch.sqrt(
            estimator_variance
        )
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
            estimator_variance.item(),

        "standard_error":
            standard_error.item(),

        "weight_variance":
            weight_variance.item(),

        "ess_fraction":
            ess_fraction,

        "eval_time":
            eval_time,
    }


# ============================================================
# VEGAS integrand
# ============================================================

def make_vegas_integrand(
    benchmark,
    dim,
    cond,
):

    cond_cpu = (
        cond
        .detach()
        .cpu()
        .reshape(
            1,
            -1,
        )
    )

    @vegas.batchintegrand
    def integrand(x):

        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device="cpu",
        )

        x_tensor = x_tensor.clamp(
            0.0,
            1.0,
        )

        batch_cond = (
            cond_cpu.expand(
                x_tensor.shape[0],
                -1,
            )
        )

        with torch.no_grad():

            logf = target_logpdf(
                benchmark,
                x_tensor,
                batch_cond,
            )

            f = torch.exp(
                logf.double()
            )

        return f.numpy()

    return integrand


# ============================================================
# VEGAS evaluation
# ============================================================

def evaluate_vegas(
    benchmark,
    dim,
    cond,
    seed,
    adapt_nitn,
    adapt_neval,
    final_nitn,
    final_neval,
):

    np.random.seed(
        seed
    )

    random.seed(
        seed
    )

    integrator = vegas.Integrator(
        [[0.0, 1.0]]
        * dim
    )

    integrand = (
        make_vegas_integrand(
            benchmark,
            dim,
            cond,
        )
    )

    # --------------------------------------------------------
    # Adaptation
    # --------------------------------------------------------

    start = (
        time.perf_counter()
    )

    integrator(
        integrand,
        nitn=adapt_nitn,
        neval=adapt_neval,
    )

    adapt_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Final fixed-grid estimate
    # --------------------------------------------------------

    start = (
        time.perf_counter()
    )

    result = integrator(
        integrand,
        nitn=final_nitn,
        neval=final_neval,
        adapt=False,
    )

    eval_time = (
        time.perf_counter()
        - start
    )

    estimate = float(
        result.mean
    )

    standard_error = float(
        result.sdev
    )

    variance = (
        standard_error ** 2
    )

    return {
        "integral_estimate":
            estimate,

        "relative_error":
            abs(
                estimate
                - 1.0
            ),

        "variance":
            variance,

        "standard_error":
            standard_error,

        "adapt_time":
            adapt_time,

        "eval_time":
            eval_time,

        "chi2":
            float(
                result.chi2
            )
            if hasattr(
                result,
                "chi2",
            )
            else np.nan,

        "Q":
            float(
                result.Q
            )
            if hasattr(
                result,
                "Q",
            )
            else np.nan,
    }


# ============================================================
# Persistence
# ============================================================

def save_raw(
    rows,
    output_dir,
):

    pd.DataFrame(
        rows
    ).to_csv(
        output_dir
        / "raw_results.csv",
        index=False,
    )


def load_existing_rows(
    output_dir,
):

    path = (
        output_dir
        / "raw_results.csv"
    )

    if not path.exists():
        return []

    return pd.read_csv(
        path
    ).to_dict(
        orient="records"
    )


def completed_key_set(
    output_dir,
):

    path = (
        output_dir
        / "raw_results.csv"
    )

    if not path.exists():
        return set()

    df = pd.read_csv(
        path
    )

    return {
        (
            str(
                row["benchmark"]
            ),
            str(
                row["method"]
            ),
            int(
                row["dimension"]
            ),
            int(
                row["seed"]
            ),
            int(
                row["condition_id"]
            ),
        )
        for _, row
        in df.iterrows()
    }


# ============================================================
# Aggregation
# ============================================================

def build_aggregate(
    raw_df,
):

    return (
        raw_df
        .groupby(
            [
                "benchmark",
                "method",
                "dimension",
            ],
            dropna=False,
        )
        .agg(
            relative_error_mean=(
                "relative_error",
                "mean",
            ),

            relative_error_std=(
                "relative_error",
                "std",
            ),

            relative_error_median=(
                "relative_error",
                "median",
            ),

            variance_mean=(
                "variance",
                "mean",
            ),

            variance_std=(
                "variance",
                "std",
            ),

            variance_median=(
                "variance",
                "median",
            ),

            standard_error_mean=(
                "standard_error",
                "mean",
            ),

            ess_mean=(
                "ess_fraction",
                "mean",
            ),

            ess_std=(
                "ess_fraction",
                "std",
            ),

            train_time_mean=(
                "train_time",
                "mean",
            ),

            adapt_time_mean=(
                "adapt_time",
                "mean",
            ),

            eval_time_mean=(
                "eval_time",
                "mean",
            ),

            total_time_mean=(
                "total_time",
                "mean",
            ),

            n_total_evals_mean=(
                "n_total_evals",
                "mean",
            ),
        )
        .reset_index()
    )


# ============================================================
# Amortization analysis
# ============================================================

def build_amortization(
    raw_df,
):

    rows = []

    groups = raw_df.groupby(
        [
            "benchmark",
            "dimension",
            "seed",
        ]
    )

    for (
        benchmark,
        dimension,
        seed,
    ), group in groups:

        nf = (
            group[
                group[
                    "method"
                ]
                == "copula_nf"
            ]
            .sort_values(
                "condition_id"
            )
        )

        vg = (
            group[
                group[
                    "method"
                ]
                == "vegas"
            ]
            .sort_values(
                "condition_id"
            )
        )

        if (
            len(nf) == 0
            or len(vg) == 0
        ):
            continue

        max_k = min(
            len(nf),
            len(vg),
        )

        nf_train_time = float(
            nf[
                "train_time"
            ].iloc[0]
        )

        nf_train_evals = int(
            nf[
                "n_train_evals"
            ].iloc[0]
        )

        for k in range(
            1,
            max_k + 1,
        ):

            nf_k = (
                nf.iloc[:k]
            )

            vg_k = (
                vg.iloc[:k]
            )

            rows.append(
                {
                    "benchmark":
                        benchmark,

                    "dimension":
                        dimension,

                    "seed":
                        seed,

                    "n_conditions":
                        k,

                    "nf_cumulative_time":
                        (
                            nf_train_time
                            +
                            nf_k[
                                "eval_time"
                            ].sum()
                        ),

                    "vegas_cumulative_time":
                        (
                            vg_k[
                                "adapt_time"
                            ].sum()
                            +
                            vg_k[
                                "eval_time"
                            ].sum()
                        ),

                    "nf_cumulative_evals":
                        (
                            nf_train_evals
                            +
                            nf_k[
                                "n_eval"
                            ].sum()
                        ),

                    "vegas_cumulative_evals":
                        (
                            vg_k[
                                "n_adapt_evals"
                            ].sum()
                            +
                            vg_k[
                                "n_eval"
                            ].sum()
                        ),

                    "nf_mean_relative_error":
                        nf_k[
                            "relative_error"
                        ].mean(),

                    "vegas_mean_relative_error":
                        vg_k[
                            "relative_error"
                        ].mean(),

                    "nf_mean_variance":
                        nf_k[
                            "variance"
                        ].mean(),

                    "vegas_mean_variance":
                        vg_k[
                            "variance"
                        ].mean(),
                }
            )

    return pd.DataFrame(
        rows
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

    print()
    print(
        f"Device for Copula NF: "
        f"{device}"
    )

    print(
        "VEGAS target evaluation: CPU"
    )

    print()

    output_dir = Path(
        args.output
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "architecture": {
            "model":
                "marginal_plus_autoregressive_copula",

            "numerical_sampling_clamp":
                True,

            "n_bins":
                N_BINS,

            "hidden":
                HIDDEN,

            "cond_dim":
                COND_DIM,
        },

        "benchmarks":
            args.benchmarks,

        "dimensions":
            args.dims,

        "seeds":
            args.seeds,

        "test_conditions":
            TEST_CONDITIONS.tolist(),

        "nf": {
            "training_objective":
                "defensive_importance_weighted_forward_kl",

            "train_steps":
                args.train_steps,

            "batch_size":
                args.batch_size,

            "learning_rate":
                LR,

            "epsilon_start":
                EPS_START,

            "epsilon_end":
                EPS_END,

            "n_eval":
                args.n_eval,
        },

        "vegas": {
            "adapt_nitn":
                args.vegas_adapt_nitn,

            "adapt_neval":
                args.vegas_adapt_neval,

            "final_nitn":
                args.vegas_final_nitn,

            "final_neval":
                args.vegas_final_neval,
        },

        "nf_device":
            str(device),

        "vegas_target_device":
            "cpu",
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

    rows = (
        load_existing_rows(
            output_dir
        )
    )

    completed = (
        completed_key_set(
            output_dir
        )
    )

    n_train_evals = (
        args.train_steps
        * args.batch_size
    )

    n_vegas_adapt = (
        args.vegas_adapt_nitn
        * args.vegas_adapt_neval
    )

    n_vegas_eval = (
        args.vegas_final_nitn
        * args.vegas_final_neval
    )

    total_models = (
        len(
            args.benchmarks
        )
        * len(
            args.dims
        )
        * len(
            args.seeds
        )
    )

    model_index = 0

    for benchmark in (
        args.benchmarks
    ):

        for dim in (
            args.dims
        ):

            for seed in (
                args.seeds
            ):

                model_index += 1

                print()
                print(
                    "=" * 78
                )

                print(
                    f"[{model_index}/"
                    f"{total_models}] "
                    f"{benchmark} | "
                    f"d={dim} | "
                    f"seed={seed}"
                )

                print(
                    "=" * 78
                )

                nf_keys = [
                    (
                        benchmark,
                        "copula_nf",
                        dim,
                        seed,
                        cid,
                    )
                    for cid
                    in range(
                        len(
                            TEST_CONDITIONS
                        )
                    )
                ]

                need_nf = not all(
                    key
                    in completed
                    for key
                    in nf_keys
                )

                checkpoint = (
                    checkpoint_dir
                    /
                    (
                        f"{benchmark}"
                        f"_d{dim}"
                        f"_seed{seed}.pt"
                    )
                )

                model = None
                train_time = np.nan
                n_params = np.nan

                # ============================================
                # NF training / loading
                # ============================================

                if need_nf:

                    seed_everything(
                        seed
                    )

                    model = (
                        SafeMarginalCopulaFlow(
                            dim=dim,
                            cond_dim=COND_DIM,
                            hidden=HIDDEN,
                            n_bins=N_BINS,
                        )
                        .to(device)
                    )

                    n_params = (
                        parameter_count(
                            model
                        )
                    )

                    if (
                        checkpoint.exists()
                        and
                        args.resume_checkpoints
                    ):

                        print(
                            "Loading NF checkpoint:",
                            checkpoint,
                        )

                        payload = torch.load(
                            checkpoint,
                            map_location=device,
                        )

                        model.load_state_dict(
                            payload[
                                "model_state_dict"
                            ]
                        )

                        train_time = (
                            payload[
                                "train_time"
                            ]
                        )

                        n_params = (
                            payload[
                                "parameter_count"
                            ]
                        )

                    else:

                        print()
                        print(
                            "Training NF from "
                            "integrand evaluations..."
                        )
                        print()

                        synchronize(
                            device
                        )

                        start = (
                            time.perf_counter()
                        )

                        train_integrand_only(
                            model=model,
                            benchmark=benchmark,
                            dim=dim,
                            device=device,
                            steps=args.train_steps,
                            batch_size=args.batch_size,
                        )

                        synchronize(
                            device
                        )

                        train_time = (
                            time.perf_counter()
                            - start
                        )

                        print()
                        print(
                            f"NF training time: "
                            f"{train_time:.2f} s"
                        )

                        torch.save(
                            {
                                "model_state_dict":
                                    model.state_dict(),

                                "benchmark":
                                    benchmark,

                                "dimension":
                                    dim,

                                "seed":
                                    seed,

                                "n_bins":
                                    N_BINS,

                                "train_time":
                                    train_time,

                                "parameter_count":
                                    n_params,
                            },
                            checkpoint,
                        )

                # ============================================
                # Held-out conditions
                # ============================================

                for (
                    condition_id,
                    cond,
                ) in enumerate(
                    TEST_CONDITIONS
                ):

                    print()
                    print(
                        f"Condition "
                        f"{condition_id+1}/"
                        f"{len(TEST_CONDITIONS)}: "
                        f"{cond.tolist()}"
                    )

                    # ----------------------------------------
                    # NF
                    # ----------------------------------------

                    nf_key = (
                        benchmark,
                        "copula_nf",
                        dim,
                        seed,
                        condition_id,
                    )

                    if (
                        nf_key
                        not in completed
                    ):

                        nf_metrics = (
                            evaluate_nf(
                                model=model,
                                benchmark=benchmark,
                                dim=dim,
                                cond=cond,
                                device=device,
                                n_eval=args.n_eval,
                            )
                        )

                        nf_row = {
                            "benchmark":
                                benchmark,

                            "method":
                                "copula_nf",

                            "dimension":
                                dim,

                            "bins":
                                N_BINS,

                            "seed":
                                seed,

                            "condition_id":
                                condition_id,

                            **make_condition_dict(
                                cond
                            ),

                            "integral_true":
                                1.0,

                            "integral_estimate":
                                nf_metrics[
                                    "integral_estimate"
                                ],

                            "relative_error":
                                nf_metrics[
                                    "relative_error"
                                ],

                            "variance":
                                nf_metrics[
                                    "variance"
                                ],

                            "standard_error":
                                nf_metrics[
                                    "standard_error"
                                ],

                            "weight_variance":
                                nf_metrics[
                                    "weight_variance"
                                ],

                            "ess_fraction":
                                nf_metrics[
                                    "ess_fraction"
                                ],

                            "train_time":
                                train_time,

                            "adapt_time":
                                0.0,

                            "eval_time":
                                nf_metrics[
                                    "eval_time"
                                ],

                            "total_time":
                                (
                                    train_time
                                    +
                                    nf_metrics[
                                        "eval_time"
                                    ]
                                ),

                            "n_train_evals":
                                n_train_evals,

                            "n_adapt_evals":
                                0,

                            "n_eval":
                                args.n_eval,

                            "n_total_evals":
                                (
                                    n_train_evals
                                    +
                                    args.n_eval
                                ),

                            "parameter_count":
                                n_params,

                            "vegas_chi2":
                                np.nan,

                            "vegas_Q":
                                np.nan,
                        }

                        rows.append(
                            nf_row
                        )

                        completed.add(
                            nf_key
                        )

                        save_raw(
                            rows,
                            output_dir,
                        )

                        print(
                            "  NF    "
                            f"I="
                            f"{nf_row['integral_estimate']:.6f} "
                            f"rel="
                            f"{nf_row['relative_error']:.3e} "
                            f"var="
                            f"{nf_row['variance']:.3e} "
                            f"ESS/N="
                            f"{nf_row['ess_fraction']:.3f}"
                        )

                    else:

                        print(
                            "  NF    already complete"
                        )

                    # ----------------------------------------
                    # VEGAS
                    # ----------------------------------------

                    vegas_key = (
                        benchmark,
                        "vegas",
                        dim,
                        seed,
                        condition_id,
                    )

                    if (
                        vegas_key
                        not in completed
                    ):

                        vegas_metrics = (
                            evaluate_vegas(
                                benchmark=benchmark,
                                dim=dim,
                                cond=cond,
                                seed=(
                                    seed
                                    * 100000
                                    +
                                    condition_id
                                ),
                                adapt_nitn=
                                    args.vegas_adapt_nitn,
                                adapt_neval=
                                    args.vegas_adapt_neval,
                                final_nitn=
                                    args.vegas_final_nitn,
                                final_neval=
                                    args.vegas_final_neval,
                            )
                        )

                        vegas_row = {
                            "benchmark":
                                benchmark,

                            "method":
                                "vegas",

                            "dimension":
                                dim,

                            "bins":
                                np.nan,

                            "seed":
                                seed,

                            "condition_id":
                                condition_id,

                            **make_condition_dict(
                                cond
                            ),

                            "integral_true":
                                1.0,

                            "integral_estimate":
                                vegas_metrics[
                                    "integral_estimate"
                                ],

                            "relative_error":
                                vegas_metrics[
                                    "relative_error"
                                ],

                            "variance":
                                vegas_metrics[
                                    "variance"
                                ],

                            "standard_error":
                                vegas_metrics[
                                    "standard_error"
                                ],

                            "weight_variance":
                                np.nan,

                            "ess_fraction":
                                np.nan,

                            "train_time":
                                0.0,

                            "adapt_time":
                                vegas_metrics[
                                    "adapt_time"
                                ],

                            "eval_time":
                                vegas_metrics[
                                    "eval_time"
                                ],

                            "total_time":
                                (
                                    vegas_metrics[
                                        "adapt_time"
                                    ]
                                    +
                                    vegas_metrics[
                                        "eval_time"
                                    ]
                                ),

                            "n_train_evals":
                                0,

                            "n_adapt_evals":
                                n_vegas_adapt,

                            "n_eval":
                                n_vegas_eval,

                            "n_total_evals":
                                (
                                    n_vegas_adapt
                                    +
                                    n_vegas_eval
                                ),

                            "parameter_count":
                                0,

                            "vegas_chi2":
                                vegas_metrics[
                                    "chi2"
                                ],

                            "vegas_Q":
                                vegas_metrics[
                                    "Q"
                                ],
                        }

                        rows.append(
                            vegas_row
                        )

                        completed.add(
                            vegas_key
                        )

                        save_raw(
                            rows,
                            output_dir,
                        )

                        print(
                            "  VEGAS "
                            f"I="
                            f"{vegas_row['integral_estimate']:.6f} "
                            f"rel="
                            f"{vegas_row['relative_error']:.3e} "
                            f"var="
                            f"{vegas_row['variance']:.3e}"
                        )

                    else:

                        print(
                            "  VEGAS already complete"
                        )

                if model is not None:

                    del model

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # ========================================================
    # Final output
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    aggregate = (
        build_aggregate(
            raw_df
        )
    )

    aggregate.to_csv(
        output_dir
        / "aggregate_results.csv",
        index=False,
    )

    amortization = (
        build_amortization(
            raw_df
        )
    )

    amortization.to_csv(
        output_dir
        / "amortization_results.csv",
        index=False,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "COPULA NF VS VEGAS COMPLETE"
    )

    print(
        "=" * 78
    )

    print()

    columns = [
        "benchmark",
        "method",
        "dimension",
        "relative_error_mean",
        "variance_mean",
        "ess_mean",
        "train_time_mean",
        "adapt_time_mean",
        "eval_time_mean",
    ]

    print(
        aggregate[
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
        default=DEFAULT_BENCHMARKS,
    )

    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=DEFAULT_DIMS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--train-steps",
        type=int,
        default=TRAIN_STEPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--n-eval",
        type=int,
        default=N_EVAL,
    )

    parser.add_argument(
        "--vegas-adapt-nitn",
        type=int,
        default=VEGAS_ADAPT_NITN,
    )

    parser.add_argument(
        "--vegas-adapt-neval",
        type=int,
        default=VEGAS_ADAPT_NEVAL,
    )

    parser.add_argument(
        "--vegas-final-nitn",
        type=int,
        default=VEGAS_FINAL_NITN,
    )

    parser.add_argument(
        "--vegas-final-neval",
        type=int,
        default=VEGAS_FINAL_NEVAL,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--resume-checkpoints",
        action="store_true",
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
            "curved_ridge"
        ]

        args.dims = [2]

        args.seeds = [0]

        args.train_steps = 250
        args.batch_size = 1024

        args.n_eval = 10_000

        args.vegas_adapt_nitn = 3
        args.vegas_adapt_neval = 2_000

        args.vegas_final_nitn = 1
        args.vegas_final_neval = 10_000

        args.output = (
            "results/"
            "copula_vs_vegas_integrand_smoke_fixed"
        )

    return args


if __name__ == "__main__":

    args = parse_args()

    run(args)