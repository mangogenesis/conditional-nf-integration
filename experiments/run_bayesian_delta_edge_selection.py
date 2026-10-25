import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import invgamma
from scipy.special import logsumexp


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT = (
    "results/"
    "masked_edge_intervention/"
    "edge_interventions.csv"
)

DEFAULT_OUTPUT = (
    "results/"
    "bayesian_delta_edge_selection"
)

# ------------------------------------------------------------
# MCMC
# ------------------------------------------------------------

N_ITER = 20_000
BURN_IN = 5_000
THIN = 5

# ------------------------------------------------------------
# Prior on graph density rho
#
# Beta(1,1) does not force a particular density.
# ------------------------------------------------------------

RHO_A = 1.0
RHO_B = 1.0

# ------------------------------------------------------------
# Variance priors
#
# sigma^2 ~ InvGamma(alpha, beta)
#
# Kept weak because the scale of delta is learned from data.
# ------------------------------------------------------------

SIGMA0_ALPHA = 2.0
SIGMA0_BETA = 1e-6

SIGMA1_ALPHA = 2.0
SIGMA1_BETA = 1e-4

# ------------------------------------------------------------
# Positive-edge mean prior
#
# mu1 ~ N(MU_PRIOR_MEAN, MU_PRIOR_SD^2), truncated to mu1 > 0
#
# Broad relative to the observed delta scale.
# ------------------------------------------------------------

MU_PRIOR_MEAN = 0.0
MU_PRIOR_SD = 1.0

RANDOM_SEED = 12345


# ============================================================
# Helpers
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
                2.0 * math.pi * variance
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
    """
    Draw from N(mean, sd^2) truncated to x > 0.

    Rejection sampling is perfectly adequate here because
    the posterior mean should normally be positive once
    useful edges exist.
    """

    sd = max(
        float(sd),
        1e-12,
    )

    for _ in range(100_000):

        value = rng.normal(
            mean,
            sd,
        )

        if value > 0.0:
            return value

    # Extremely defensive fallback
    return max(
        mean,
        1e-8,
    )


# ============================================================
# Prepare evidence
# ============================================================

def load_delta_data(
    input_path,
):

    df = pd.read_csv(
        input_path
    )

    required = {
        "benchmark",
        "seed",
        "parent",
        "child",
        "delta_loglik",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def make_edge_matrix(
    benchmark_df,
):
    """
    Returns:

        edges:
            list[(parent, child)]

        seeds:
            sorted list of seeds

        Y:
            shape [n_edges, n_seeds]

    Each row corresponds to one AR edge.
    """

    edges = (
        benchmark_df[
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
        in edges.itertuples()
    ]

    seeds = sorted(
        benchmark_df[
            "seed"
        ]
        .unique()
        .tolist()
    )

    edge_to_index = {
        edge: index
        for index, edge
        in enumerate(edges)
    }

    seed_to_index = {
        seed: index
        for index, seed
        in enumerate(seeds)
    }

    Y = np.full(
        (
            len(edges),
            len(seeds),
        ),
        np.nan,
        dtype=float,
    )

    for row in (
        benchmark_df
        .itertuples()
    ):

        edge = (
            int(row.parent),
            int(row.child),
        )

        e = edge_to_index[
            edge
        ]

        s = seed_to_index[
            row.seed
        ]

        Y[e, s] = float(
            row.delta_loglik
        )

    if np.isnan(Y).any():

        missing_indices = (
            np.argwhere(
                np.isnan(Y)
            )
        )

        raise ValueError(
            "Missing seed observations "
            "for some edges: "
            f"{missing_indices.tolist()}"
        )

    return (
        edges,
        seeds,
        Y,
    )


# ============================================================
# Gibbs / MCMC
# ============================================================

def run_mcmc(
    Y,
    rng,
    n_iter,
    burn_in,
    thin,
    rho_a,
    rho_b,
):

    n_edges, n_rep = (
        Y.shape
    )

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    edge_means = (
        Y.mean(
            axis=1
        )
    )

    # Initialize roughly from positive-vs-near-zero behavior,
    # but this is ONLY initialization, not the final rule.
    median_positive = np.median(
        edge_means[
            edge_means > 0
        ]
    ) if np.any(
        edge_means > 0
    ) else 0.0

    if median_positive > 0:

        z = (
            edge_means
            > median_positive
        ).astype(int)

    else:

        z = np.zeros(
            n_edges,
            dtype=int,
        )

    if z.sum() == 0:
        z[
            np.argmax(
                edge_means
            )
        ] = 1

    rho = 0.5

    mu1 = max(
        np.mean(
            edge_means[
                z == 1
            ]
        ),
        1e-3,
    )

    sigma0_sq = max(
        np.var(
            Y[
                z == 0
            ]
        )
        if np.any(
            z == 0
        )
        else np.var(Y),
        1e-6,
    )

    sigma1_sq = max(
        np.var(
            Y[
                z == 1
            ]
        )
        if np.any(
            z == 1
        )
        else np.var(Y),
        1e-6,
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

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
    # MCMC loop
    # ========================================================

    for iteration in range(
        n_iter
    ):

        # ====================================================
        # 1. Sample each edge indicator Z_e
        # ====================================================

        for e in range(
            n_edges
        ):

            observations = (
                Y[e]
            )

            log_lik_0 = 0.0
            log_lik_1 = 0.0

            for value in observations:

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
                    1.0 - rho_safe
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

            inclusion_prob = math.exp(
                log_p1
                - denom
            )

            z[e] = int(
                rng.random()
                < inclusion_prob
            )

        # ====================================================
        # 2. Sample graph density rho
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
        # 3. Sample null variance sigma0^2
        #
        # Null mean fixed at zero.
        # ====================================================

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

        # ====================================================
        # 4. Sample positive-edge mean mu1
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
                MU_PRIOR_SD ** 2
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
        # 5. Sample active variance sigma1^2
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
        # Store posterior samples
        # ====================================================

        if (
            iteration >= burn_in
            and
            (
                iteration - burn_in
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
    # Posterior summaries
    # ========================================================

    pip = (
        z_sum
        / kept
    )

    result = {
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

    return result


# ============================================================
# Run one benchmark
# ============================================================

def run_benchmark(
    benchmark,
    benchmark_df,
    args,
):

    print()
    print(
        "=" * 90
    )

    print(
        f"BAYESIAN DELTA SELECTION: "
        f"{benchmark}"
    )

    print(
        "=" * 90
    )

    (
        edges,
        seeds,
        Y,
    ) = make_edge_matrix(
        benchmark_df
    )

    print(
        f"  edges       : "
        f"{len(edges)}"
    )

    print(
        f"  seeds       : "
        f"{seeds}"
    )

    print(
        f"  delta mean  : "
        f"{Y.mean():.6f}"
    )

    print(
        f"  delta range : "
        f"[{Y.min():.6f}, "
        f"{Y.max():.6f}]"
    )

    rng = np.random.default_rng(
        args.random_seed
    )

    posterior = run_mcmc(
        Y=Y,

        rng=rng,

        n_iter=args.n_iter,

        burn_in=args.burn_in,

        thin=args.thin,

        rho_a=args.rho_a,

        rho_b=args.rho_b,
    )

    print()
    print(
        f"  posterior rho       = "
        f"{posterior['rho_mean']:.4f} "
        f"± "
        f"{posterior['rho_sd']:.4f}"
    )

    print(
        f"  posterior mu1       = "
        f"{posterior['mu1_mean']:.6f} "
        f"± "
        f"{posterior['mu1_sd']:.6f}"
    )

    print(
        f"  posterior sigma0    = "
        f"{posterior['sigma0_mean']:.6f}"
    )

    print(
        f"  posterior sigma1    = "
        f"{posterior['sigma1_mean']:.6f}"
    )

    print(
        f"  posterior E[edges]  = "
        f"{posterior['expected_edges']:.2f} "
        f"± "
        f"{posterior['edge_count_sd']:.2f}"
    )

    # --------------------------------------------------------
    # Edge table
    # --------------------------------------------------------

    rows = []

    edge_means = (
        Y.mean(
            axis=1
        )
    )

    edge_sds = (
        Y.std(
            axis=1,
            ddof=1,
        )
    )

    for edge_id, (
        parent,
        child,
    ) in enumerate(
        edges
    ):

        pip = float(
            posterior[
                "pip"
            ][edge_id]
        )

        rows.append(
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

                "delta_mean":
                    float(
                        edge_means[
                            edge_id
                        ]
                    ),

                "delta_sd_across_seeds":
                    float(
                        edge_sds[
                            edge_id
                        ]
                    ),

                "posterior_inclusion_probability":
                    pip,

                # Median probability model.
                #
                # This is NOT a raw-delta threshold.
                # It means posterior inclusion is
                # more likely than exclusion.
                "included_pip_gt_0p5":
                    int(
                        pip > 0.5
                    ),
            }
        )

    edge_df = pd.DataFrame(
        rows
    )

    edge_df = (
        edge_df
        .sort_values(
            [
                "posterior_inclusion_probability",
                "delta_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    edge_df[
        "posterior_rank"
    ] = (
        np.arange(
            len(edge_df)
        )
        + 1
    )

    summary = {
        "benchmark":
            benchmark,

        "n_edges":
            len(edges),

        "n_seeds":
            len(seeds),

        "rho_mean":
            posterior[
                "rho_mean"
            ],

        "rho_sd":
            posterior[
                "rho_sd"
            ],

        "mu1_mean":
            posterior[
                "mu1_mean"
            ],

        "mu1_sd":
            posterior[
                "mu1_sd"
            ],

        "sigma0_mean":
            posterior[
                "sigma0_mean"
            ],

        "sigma1_mean":
            posterior[
                "sigma1_mean"
            ],

        "expected_edges":
            posterior[
                "expected_edges"
            ],

        "edge_count_sd":
            posterior[
                "edge_count_sd"
            ],

        "pip_gt_0p5_edges":
            int(
                edge_df[
                    "included_pip_gt_0p5"
                ].sum()
            ),

        "mean_markov1_pip":
            float(
                edge_df.loc[
                    edge_df[
                        "is_markov1_edge"
                    ] == 1,
                    "posterior_inclusion_probability",
                ]
                .mean()
            ),

        "mean_nonmarkov_pip":
            float(
                edge_df.loc[
                    edge_df[
                        "is_markov1_edge"
                    ] == 0,
                    "posterior_inclusion_probability",
                ]
                .mean()
            ),
    }

    print()
    print(
        "  Top posterior edges:"
    )

    print()

    print(
        edge_df[
            [
                "parent",
                "child",
                "is_markov1_edge",
                "delta_mean",
                "delta_sd_across_seeds",
                "posterior_inclusion_probability",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    return (
        edge_df,
        summary,
    )


# ============================================================
# Main
# ============================================================

def run(args):

    input_df = load_delta_data(
        args.input
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_edges = []
    summaries = []

    benchmarks = (
        args.benchmarks
        if args.benchmarks
        else sorted(
            input_df[
                "benchmark"
            ].unique()
        )
    )

    for benchmark in benchmarks:

        benchmark_df = (
            input_df[
                input_df[
                    "benchmark"
                ] == benchmark
            ]
            .copy()
        )

        if len(
            benchmark_df
        ) == 0:

            print(
                f"Skipping {benchmark}: "
                "no evidence found."
            )

            continue

        edge_df, summary = (
            run_benchmark(
                benchmark,
                benchmark_df,
                args,
            )
        )

        all_edges.append(
            edge_df
        )

        summaries.append(
            summary
        )

    edge_output = pd.concat(
        all_edges,
        ignore_index=True,
    )

    summary_output = pd.DataFrame(
        summaries
    )

    edge_output.to_csv(
        output_dir
        / "posterior_edge_probabilities.csv",
        index=False,
    )

    summary_output.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Deterministic median-probability graph
    # --------------------------------------------------------

    selected = (
        edge_output[
            edge_output[
                "posterior_inclusion_probability"
            ] > 0.5
        ]
        .copy()
    )

    selected.to_csv(
        output_dir
        / "median_probability_graph.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )

    print(
        "BAYESIAN DELTA EDGE SELECTION COMPLETE"
    )

    print(
        "=" * 110
    )

    print()

    print(
        summary_output.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        "  posterior_edge_probabilities.csv"
    )

    print(
        "  median_probability_graph.csv"
    )

    print(
        "  summary.csv"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=
            DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=
            N_ITER,
    )

    parser.add_argument(
        "--burn-in",
        type=int,
        default=
            BURN_IN,
    )

    parser.add_argument(
        "--thin",
        type=int,
        default=
            THIN,
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
        "--random-seed",
        type=int,
        default=
            RANDOM_SEED,
    )

    args = (
        parser.parse_args()
    )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )