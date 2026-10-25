import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import vegas


# ============================================================
# Benchmark configuration
# ============================================================

DIM = 8
BENCHMARK = "branching_tree"

DEFAULT_OUTPUT = (
    "results/vegas_adaptation_sweep_branching_tree_d8"
)

DEFAULT_CONDITION = [
    0.5,
    0.5,
    0.5,
]

# ============================================================
# VEGAS adaptation sweep
# ============================================================

DEFAULT_ADAPT_NITN = [
    10,
    20,
    50,
    100,
    200,
]

DEFAULT_ADAPT_NEVAL = 100_000

# ============================================================
# Fixed production stage
# ============================================================

DEFAULT_PROD_NITN = 10
DEFAULT_PROD_NEVAL = 100_000

DEFAULT_REPEATS = 5
DEFAULT_SEED = 12345


# ============================================================
# Normal helpers
# ============================================================

def standard_normal_cdf(x):
    return (
        0.5
        *
        (
            1.0
            +
            torch.erf(
                x
                /
                math.sqrt(2.0)
            )
        )
    )


# ============================================================
# Truncated normal logpdf on [0,1]
# ============================================================

def truncated_normal_logpdf(
    x,
    mu,
    sigma,
):
    z = (
        x - mu
    ) / sigma

    log_base = (
        -0.5
        *
        z.square()
        -
        torch.log(
            sigma
        )
        -
        0.5
        *
        math.log(
            2.0
            *
            math.pi
        )
    )

    a = (
        0.0 - mu
    ) / sigma

    b = (
        1.0 - mu
    ) / sigma

    normalization = (
        standard_normal_cdf(
            b
        )
        -
        standard_normal_cdf(
            a
        )
    ).clamp_min(
        1e-12
    )

    return (
        log_base
        -
        torch.log(
            normalization
        )
    )


# ============================================================
# Branching-tree benchmark
#
# 0 -> 1
# 0 -> 2
# 1 -> 3
# 1 -> 4
# 2 -> 5
# 2 -> 6
# 6 -> 7
# ============================================================

PARENT_MAP = {
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 6,
}


def branching_tree_mu_sigma(
    child,
    x,
    cond,
):
    c0 = cond[
        :,
        0:1
    ]

    c2 = cond[
        :,
        2:3
    ]

    parent = (
        PARENT_MAP[
            child
        ]
    )

    p = x[
        :,
        parent:parent + 1
    ]

    mu = (
        0.50
        +
        0.27
        *
        torch.sin(
            2.0
            *
            math.pi
            *
            (
                p
                +
                0.10
                *
                c0
                +
                0.037
                *
                child
            )
        )
    )

    sigma = (
        0.045
        +
        0.010
        *
        c2
    )

    return (
        mu,
        sigma,
    )


# ============================================================
# Target log density
# ============================================================

def branching_tree_logpdf(
    x,
    cond,
):
    logp = torch.zeros(
        x.shape[0],
        dtype=x.dtype,
        device=x.device,
    )

    # x0 ~ Uniform(0,1), so log-density contribution = 0.
    for child in range(
        1,
        DIM,
    ):

        mu, sigma = (
            branching_tree_mu_sigma(
                child,
                x,
                cond,
            )
        )

        logp += (
            truncated_normal_logpdf(
                x[
                    :,
                    child:child + 1
                ],
                mu,
                sigma,
            )
            .reshape(-1)
        )

    return logp


# ============================================================
# VEGAS integrand
# ============================================================

def make_vegas_integrand(
    condition,
):
    condition_np = np.asarray(
        condition,
        dtype=np.float64,
    )

    condition_torch = (
        torch.tensor(
            condition_np,
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

        x_torch = torch.from_numpy(
            x_np
        ).to(
            dtype=torch.float64
        )

        batch_size = (
            x_torch.shape[0]
        )

        cond = (
            condition_torch
            .expand(
                batch_size,
                -1,
            )
        )

        with torch.no_grad():
            logp = (
                branching_tree_logpdf(
                    x_torch,
                    cond,
                )
            )

            values = torch.exp(
                logp
            )

        return (
            values
            .cpu()
            .numpy()
        )

    return integrand


# ============================================================
# One full adaptation + production experiment
# ============================================================

def run_single_experiment(
    condition,
    adapt_nitn,
    adapt_neval,
    prod_nitn,
    prod_neval,
    repeat,
    seed,
):
    np.random.seed(
        seed
    )

    # Fresh grid every repeat and every adaptation setting.
    integ = vegas.Integrator(
        [
            [0.0, 1.0]
            for _ in range(
                DIM
            )
        ]
    )

    integrand = (
        make_vegas_integrand(
            condition
        )
    )

    # ========================================================
    # Stage 1: adaptation only
    #
    # We intentionally discard the returned integral estimate.
    # We only want VEGAS to learn/adapt its grid here.
    # ========================================================

    adapt_start = (
        time.perf_counter()
    )

    adapt_result = integ(
        integrand,
        nitn=adapt_nitn,
        neval=adapt_neval,
    )

    adapt_time = (
        time.perf_counter()
        -
        adapt_start
    )

    # ========================================================
    # Stage 2: production
    #
    # Same already-adapted integrator.
    # ========================================================

    prod_start = (
        time.perf_counter()
    )

    prod_result = integ(
        integrand,
        nitn=prod_nitn,
        neval=prod_neval,
    )

    prod_time = (
        time.perf_counter()
        -
        prod_start
    )

    total_time = (
        adapt_time
        +
        prod_time
    )

    estimate = float(
        prod_result.mean
    )

    reported_std = float(
        prod_result.sdev
    )

    exact_integral = 1.0

    actual_abs_error = abs(
        estimate
        -
        exact_integral
    )

    actual_rel_error = (
        actual_abs_error
        /
        exact_integral
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

    adaptation_evals = (
        adapt_nitn
        *
        adapt_neval
    )

    production_evals = (
        prod_nitn
        *
        prod_neval
    )

    total_requested_evals = (
        adaptation_evals
        +
        production_evals
    )

    return {
        "benchmark":
            BENCHMARK,

        "dim":
            DIM,

        "condition_c0":
            condition[0],

        "condition_c1":
            condition[1],

        "condition_c2":
            condition[2],

        "repeat":
            repeat,

        "seed":
            seed,

        # ----------------------------------------------------
        # Adaptation
        # ----------------------------------------------------

        "adapt_nitn":
            adapt_nitn,

        "adapt_neval":
            adapt_neval,

        "adaptation_evals":
            adaptation_evals,

        "adaptation_wall_time":
            adapt_time,

        "adaptation_result_mean":
            float(
                adapt_result.mean
            ),

        "adaptation_result_sdev":
            float(
                adapt_result.sdev
            ),

        "adaptation_Q":
            (
                float(
                    adapt_result.Q
                )
                if adapt_result.Q is not None
                else np.nan
            ),

        # ----------------------------------------------------
        # Production
        # ----------------------------------------------------

        "prod_nitn":
            prod_nitn,

        "prod_neval":
            prod_neval,

        "production_evals":
            production_evals,

        "production_wall_time":
            prod_time,

        "estimate":
            estimate,

        "exact_integral":
            exact_integral,

        "reported_std":
            reported_std,

        "reported_rel_error":
            reported_rel_error,

        "actual_abs_error":
            actual_abs_error,

        "actual_rel_error":
            actual_rel_error,

        "production_Q":
            (
                float(
                    prod_result.Q
                )
                if prod_result.Q is not None
                else np.nan
            ),

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        "total_requested_evals":
            total_requested_evals,

        "total_wall_time":
            total_time,

        "evals_per_second":
            (
                total_requested_evals
                /
                total_time
            ),
    }


# ============================================================
# Aggregate repetitions
# ============================================================

def summarize_results(
    raw_df,
):
    summary_df = (
        raw_df
        .groupby(
            [
                "adapt_nitn",
                "adapt_neval",
                "adaptation_evals",
                "prod_nitn",
                "prod_neval",
                "production_evals",
                "total_requested_evals",
            ],
            as_index=False,
        )
        .agg(
            estimate_mean=(
                "estimate",
                "mean",
            ),

            estimate_std_across_runs=(
                "estimate",
                "std",
            ),

            estimate_min=(
                "estimate",
                "min",
            ),

            estimate_max=(
                "estimate",
                "max",
            ),

            reported_std_mean=(
                "reported_std",
                "mean",
            ),

            reported_rel_error_mean=(
                "reported_rel_error",
                "mean",
            ),

            reported_rel_error_std=(
                "reported_rel_error",
                "std",
            ),

            actual_abs_error_mean=(
                "actual_abs_error",
                "mean",
            ),

            actual_abs_error_std=(
                "actual_abs_error",
                "std",
            ),

            actual_rel_error_mean=(
                "actual_rel_error",
                "mean",
            ),

            actual_rel_error_std=(
                "actual_rel_error",
                "std",
            ),

            production_Q_mean=(
                "production_Q",
                "mean",
            ),

            adaptation_wall_time_mean=(
                "adaptation_wall_time",
                "mean",
            ),

            production_wall_time_mean=(
                "production_wall_time",
                "mean",
            ),

            total_wall_time_mean=(
                "total_wall_time",
                "mean",
            ),

            total_wall_time_std=(
                "total_wall_time",
                "std",
            ),
        )
    )

    return summary_df


# ============================================================
# Main driver
# ============================================================

def run(
    args,
):
    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    condition = [
        args.c0,
        args.c1,
        args.c2,
    ]

    print()
    print(
        "=" * 90
    )

    print(
        "VEGAS ADAPTATION -> PRODUCTION SWEEP"
    )

    print(
        "=" * 90
    )

    print(
        f"Benchmark       : "
        f"{BENCHMARK}"
    )

    print(
        f"Dimension       : "
        f"{DIM}"
    )

    print(
        f"Condition       : "
        f"{condition}"
    )

    print(
        f"Exact integral  : "
        f"1.0"
    )

    print()

    print(
        "ADAPTATION:"
    )

    print(
        f"  nitn grid     : "
        f"{args.adapt_nitn}"
    )

    print(
        f"  neval         : "
        f"{args.adapt_neval}"
    )

    print()

    print(
        "PRODUCTION:"
    )

    print(
        f"  nitn          : "
        f"{args.prod_nitn}"
    )

    print(
        f"  neval         : "
        f"{args.prod_neval}"
    )

    print()

    print(
        f"Repeats         : "
        f"{args.repeats}"
    )

    print(
        "=" * 90
    )

    rows = []

    total_jobs = (
        len(
            args.adapt_nitn
        )
        *
        args.repeats
    )

    job = 0

    for adapt_nitn in (
        args.adapt_nitn
    ):

        for repeat in range(
            args.repeats
        ):

            job += 1

            seed = (
                args.seed
                +
                1_000_000
                *
                repeat
                +
                10_000
                *
                adapt_nitn
            )

            print()
            print(
                "-" * 90
            )

            print(
                f"[{job}/{total_jobs}] "
                f"adapt_nitn="
                f"{adapt_nitn}, "
                f"adapt_neval="
                f"{args.adapt_neval}, "
                f"repeat="
                f"{repeat}"
            )

            row = (
                run_single_experiment(
                    condition=
                        condition,

                    adapt_nitn=
                        adapt_nitn,

                    adapt_neval=
                        args.adapt_neval,

                    prod_nitn=
                        args.prod_nitn,

                    prod_neval=
                        args.prod_neval,

                    repeat=
                        repeat,

                    seed=
                        seed,
                )
            )

            rows.append(
                row
            )

            print(
                f"  adaptation evals = "
                f"{row['adaptation_evals']:,}"
            )

            print(
                f"  production evals = "
                f"{row['production_evals']:,}"
            )

            print(
                f"  estimate         = "
                f"{row['estimate']:.10f}"
            )

            print(
                f"  actual rel err   = "
                f"{100.0 * row['actual_rel_error']:.4f}%"
            )

            print(
                f"  reported rel err = "
                f"{100.0 * row['reported_rel_error']:.4f}%"
            )

            print(
                f"  production Q     = "
                f"{row['production_Q']:.6e}"
            )

            print(
                f"  adapt time       = "
                f"{row['adaptation_wall_time']:.3f} s"
            )

            print(
                f"  production time  = "
                f"{row['production_wall_time']:.3f} s"
            )

            print(
                f"  total time       = "
                f"{row['total_wall_time']:.3f} s"
            )

            # Save after every job.
            pd.DataFrame(
                rows
            ).to_csv(
                output_dir
                /
                "vegas_adaptation_raw.csv",
                index=False,
            )

    raw_df = pd.DataFrame(
        rows
    )

    summary_df = (
        summarize_results(
            raw_df
        )
    )

    raw_df.to_csv(
        output_dir
        /
        "vegas_adaptation_raw.csv",
        index=False,
    )

    summary_df.to_csv(
        output_dir
        /
        "vegas_adaptation_summary.csv",
        index=False,
    )

    # ========================================================
    # Rankings
    # ========================================================

    ranked_actual = (
        summary_df
        .sort_values(
            [
                "actual_rel_error_mean",
                "total_wall_time_mean",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ranked_actual.to_csv(
        output_dir
        /
        "vegas_ranked_by_actual_error.csv",
        index=False,
    )

    ranked_reported = (
        summary_df
        .sort_values(
            [
                "reported_rel_error_mean",
                "total_wall_time_mean",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ranked_reported.to_csv(
        output_dir
        /
        "vegas_ranked_by_reported_error.csv",
        index=False,
    )

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "VEGAS ADAPTATION SWEEP COMPLETE"
    )

    print(
        "=" * 120
    )

    display_columns = [
        "adapt_nitn",
        "adapt_neval",
        "adaptation_evals",
        "production_evals",
        "total_requested_evals",
        "estimate_mean",
        "estimate_std_across_runs",
        "reported_rel_error_mean",
        "actual_rel_error_mean",
        "production_Q_mean",
        "total_wall_time_mean",
    ]

    print()
    print(
        summary_df[
            display_columns
        ]
        .sort_values(
            "adapt_nitn"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Best settings by ACTUAL relative error:"
    )

    print()

    print(
        ranked_actual[
            display_columns
        ]
        .head(
            10
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
        "--adapt-nitn",
        nargs="+",
        type=int,
        default=
            DEFAULT_ADAPT_NITN,
    )

    parser.add_argument(
        "--adapt-neval",
        type=int,
        default=
            DEFAULT_ADAPT_NEVAL,
    )

    parser.add_argument(
        "--prod-nitn",
        type=int,
        default=
            DEFAULT_PROD_NITN,
    )

    parser.add_argument(
        "--prod-neval",
        type=int,
        default=
            DEFAULT_PROD_NEVAL,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=
            DEFAULT_REPEATS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=
            DEFAULT_SEED,
    )

    parser.add_argument(
        "--c0",
        type=float,
        default=
            DEFAULT_CONDITION[
                0
            ],
    )

    parser.add_argument(
        "--c1",
        type=float,
        default=
            DEFAULT_CONDITION[
                1
            ],
    )

    parser.add_argument(
        "--c2",
        type=float,
        default=
            DEFAULT_CONDITION[
                2
            ],
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            DEFAULT_OUTPUT,
    )

    return parser.parse_args()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    run(
        parse_args()
    )