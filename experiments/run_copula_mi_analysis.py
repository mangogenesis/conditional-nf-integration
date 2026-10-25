import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
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

DEFAULT_BENCHMARKS = [
    "single_gaussian",
    "sharp_broad",
    "curved_ridge",
    "xshape",
]

DEFAULT_DIMS = [8]
DEFAULT_SEEDS = [0]

COND_DIM = 3

N_BINS = 32
HIDDEN = 128

MARGINAL_STEPS = 1200
BATCH_SIZE = 4096
LR_MARGINAL = 1e-4
GRAD_CLIP = 10.0

# Number of copula samples used to estimate MI
N_MI = 100_000

# Histogram MI estimator
MI_BINS = 32

# Candidate parents to report per coordinate
TOP_K = 3

OUTPUT_DIR = Path(
    "results/copula_mi_analysis"
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


def parameter_count(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# Marginal model only
# ============================================================

class MarginalFlowND(nn.Module):

    def __init__(
        self,
        dim,
        cond_dim,
        hidden,
        n_bins,
    ):

        super().__init__()

        self.dim = dim

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

        return torch.cat(
            u_list,
            dim=1,
        )


# ============================================================
# Train marginals on target samples
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
# Histogram MI estimator
# ============================================================

def estimate_pairwise_mi(
    x,
    y,
    n_bins=32,
):
    """
    Histogram estimate of

        I(X;Y)
          = sum_ij p_ij
            log(
                p_ij / (p_i p_j)
            )

    x and y are assumed to live approximately
    on [0,1].
    """

    hist_2d, _, _ = np.histogram2d(
        x,
        y,
        bins=n_bins,
        range=[
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    )

    total = hist_2d.sum()

    if total <= 0:
        return np.nan

    pxy = (
        hist_2d
        / total
    )

    px = (
        pxy.sum(
            axis=1,
            keepdims=True,
        )
    )

    py = (
        pxy.sum(
            axis=0,
            keepdims=True,
        )
    )

    expected = (
        px
        * py
    )

    mask = (
        (pxy > 0)
        &
        (expected > 0)
    )

    mi = np.sum(
        pxy[mask]
        *
        np.log(
            pxy[mask]
            / expected[mask]
        )
    )

    return float(
        mi
    )


def estimate_mi_matrix(
    u,
    n_bins,
):

    d = u.shape[1]

    matrix = np.zeros(
        (d, d),
        dtype=np.float64,
    )

    for i in range(d):

        for j in range(
            i + 1,
            d,
        ):

            mi = estimate_pairwise_mi(
                u[:, i],
                u[:, j],
                n_bins=n_bins,
            )

            matrix[i, j] = mi
            matrix[j, i] = mi

    return matrix


# ============================================================
# MI summaries
# ============================================================

def off_diagonal_values(
    matrix,
):

    d = matrix.shape[0]

    values = []

    for i in range(d):

        for j in range(
            i + 1,
            d,
        ):

            values.append(
                matrix[i, j]
            )

    return np.asarray(
        values
    )


def local_mi_mean(
    matrix,
    distance=1,
):

    d = matrix.shape[0]

    values = []

    for i in range(
        d - distance
    ):

        values.append(
            matrix[
                i,
                i + distance,
            ]
        )

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def long_range_mi_mean(
    matrix,
    min_distance=2,
):

    d = matrix.shape[0]

    values = []

    for i in range(d):

        for j in range(
            i + 1,
            d,
        ):

            if (
                j - i
                >= min_distance
            ):

                values.append(
                    matrix[i, j]
                )

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def locality_ratio(
    matrix,
):
    """
    Ratio:

        mean MI nearest-neighbor
        ------------------------
        mean MI long-range

    Large values indicate banded/local structure.
    """

    local = local_mi_mean(
        matrix,
        distance=1,
    )

    long_range = (
        long_range_mi_mean(
            matrix,
            min_distance=2,
        )
    )

    if (
        not np.isfinite(
            long_range
        )
        or long_range <= 0
    ):

        return np.inf

    return float(
        local
        / long_range
    )


# ============================================================
# Candidate parent extraction
# ============================================================

def top_k_previous_parents(
    matrix,
    k,
):
    """
    For autoregressive coordinate j,
    only consider previous coordinates:

        0,...,j-1

    Return top-k according to pairwise MI.
    """

    d = matrix.shape[0]

    results = {}

    for j in range(
        1,
        d,
    ):

        scores = [
            (
                i,
                matrix[i, j],
            )
            for i in range(j)
        ]

        scores.sort(
            key=lambda z: z[1],
            reverse=True,
        )

        results[j] = (
            scores[:k]
        )

    return results


# ============================================================
# Plotting
# ============================================================

def plot_mi_matrix(
    matrix,
    benchmark,
    dimension,
    seed,
    condition_id,
    output_path,
):

    plt.figure(
        figsize=(6, 5)
    )

    im = plt.imshow(
        matrix,
        origin="lower",
        aspect="equal",
    )

    plt.colorbar(
        im,
        label="Mutual information [nats]",
    )

    plt.xlabel(
        "Copula coordinate j"
    )

    plt.ylabel(
        "Copula coordinate i"
    )

    plt.title(
        f"{benchmark} | "
        f"d={dimension} | "
        f"seed={seed} | "
        f"condition={condition_id}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()


def plot_average_matrix(
    matrix,
    benchmark,
    dimension,
    output_path,
):

    plt.figure(
        figsize=(6, 5)
    )

    im = plt.imshow(
        matrix,
        origin="lower",
        aspect="equal",
    )

    plt.colorbar(
        im,
        label="Mean mutual information [nats]",
    )

    plt.xlabel(
        "Copula coordinate j"
    )

    plt.ylabel(
        "Copula coordinate i"
    )

    plt.title(
        f"{benchmark} | "
        f"d={dimension} | "
        f"mean across conditions/seeds"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()


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

    matrix_dir = (
        output_dir
        / "matrices"
    )

    plot_dir = (
        output_dir
        / "plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir.mkdir(
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

        "marginal_steps":
            args.marginal_steps,

        "batch_size":
            args.batch_size,

        "n_mi":
            args.n_mi,

        "mi_bins":
            args.mi_bins,

        "top_k":
            args.top_k,

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

    raw_rows = []
    parent_rows = []

    all_matrices = {}

    total_models = (
        len(args.benchmarks)
        * len(args.dims)
        * len(args.seeds)
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

                model = (
                    MarginalFlowND(
                        dim=dim,
                        cond_dim=COND_DIM,
                        hidden=HIDDEN,
                        n_bins=N_BINS,
                    )
                    .to(device)
                )

                print(
                    "\nTraining marginals..."
                )

                synchronize(
                    device
                )

                start = (
                    time.perf_counter()
                )

                train_marginals(
                    model=model,
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

                train_time = (
                    time.perf_counter()
                    - start
                )

                print(
                    f"\nMarginal training time: "
                    f"{train_time:.2f} s"
                )

                key = (
                    benchmark,
                    dim,
                )

                if key not in all_matrices:

                    all_matrices[key] = []

                # =================================================
                # Held-out conditions
                # =================================================

                for (
                    condition_id,
                    cond,
                ) in enumerate(
                    TEST_CONDITIONS
                ):

                    print(
                        f"  condition "
                        f"{condition_id+1}/"
                        f"{len(TEST_CONDITIONS)}"
                    )

                    cond_batch = (
                        cond
                        .to(device)
                        .reshape(
                            1,
                            -1,
                        )
                        .expand(
                            args.n_mi,
                            -1,
                        )
                    )

                    with torch.no_grad():

                        x = sample_target(
                            benchmark,
                            cond_batch,
                            dim,
                        )

                        u = model.x_to_u(
                            x,
                            cond_batch,
                        )

                    u_np = (
                        u
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    matrix = (
                        estimate_mi_matrix(
                            u_np,
                            n_bins=
                                args.mi_bins,
                        )
                    )

                    all_matrices[
                        key
                    ].append(
                        matrix
                    )

                    matrix_name = (
                        f"{benchmark}"
                        f"_d{dim}"
                        f"_seed{seed}"
                        f"_c{condition_id:02d}"
                        f".npy"
                    )

                    np.save(
                        matrix_dir
                        / matrix_name,
                        matrix,
                    )

                    plot_name = (
                        f"{benchmark}"
                        f"_d{dim}"
                        f"_seed{seed}"
                        f"_c{condition_id:02d}"
                        f"_mi.png"
                    )

                    plot_mi_matrix(
                        matrix=matrix,
                        benchmark=benchmark,
                        dimension=dim,
                        seed=seed,
                        condition_id=
                            condition_id,
                        output_path=
                            plot_dir
                            / plot_name,
                    )

                    offdiag = (
                        off_diagonal_values(
                            matrix
                        )
                    )

                    local1 = (
                        local_mi_mean(
                            matrix,
                            distance=1,
                        )
                    )

                    local2 = (
                        local_mi_mean(
                            matrix,
                            distance=2,
                        )
                    )

                    long_range = (
                        long_range_mi_mean(
                            matrix,
                            min_distance=2,
                        )
                    )

                    ratio = (
                        locality_ratio(
                            matrix
                        )
                    )

                    raw_rows.append(
                        {
                            "benchmark":
                                benchmark,

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

                            "mi_mean":
                                float(
                                    np.mean(
                                        offdiag
                                    )
                                ),

                            "mi_max":
                                float(
                                    np.max(
                                        offdiag
                                    )
                                ),

                            "mi_median":
                                float(
                                    np.median(
                                        offdiag
                                    )
                                ),

                            "nearest_neighbor_mi":
                                local1,

                            "distance2_mi":
                                local2,

                            "long_range_mi":
                                long_range,

                            "locality_ratio":
                                ratio,

                            "marginal_train_time":
                                train_time,
                        }
                    )

                    # =============================================
                    # Parent candidates
                    # =============================================

                    parents = (
                        top_k_previous_parents(
                            matrix,
                            k=args.top_k,
                        )
                    )

                    for (
                        child,
                        candidates,
                    ) in parents.items():

                        for (
                            rank,
                            (
                                parent,
                                score,
                            ),
                        ) in enumerate(
                            candidates,
                            start=1,
                        ):

                            parent_rows.append(
                                {
                                    "benchmark":
                                        benchmark,

                                    "dimension":
                                        dim,

                                    "seed":
                                        seed,

                                    "condition_id":
                                        condition_id,

                                    "child":
                                        child,

                                    "parent":
                                        parent,

                                    "rank":
                                        rank,

                                    "mutual_information":
                                        score,

                                    "distance":
                                        child
                                        - parent,
                                }
                            )

                    pd.DataFrame(
                        raw_rows
                    ).to_csv(
                        output_dir
                        / "raw_mi.csv",
                        index=False,
                    )

                    pd.DataFrame(
                        parent_rows
                    ).to_csv(
                        output_dir
                        / "parent_candidates.csv",
                        index=False,
                    )

                del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ========================================================
    # Average MI matrices
    # ========================================================

    for (
        benchmark,
        dim,
    ), matrices in (
        all_matrices.items()
    ):

        stacked = np.stack(
            matrices,
            axis=0,
        )

        mean_matrix = np.mean(
            stacked,
            axis=0,
        )

        np.save(
            matrix_dir
            /
            (
                f"{benchmark}"
                f"_d{dim}"
                f"_mean.npy"
            ),
            mean_matrix,
        )

        plot_average_matrix(
            matrix=
                mean_matrix,

            benchmark=
                benchmark,

            dimension=
                dim,

            output_path=
                plot_dir
                /
                (
                    f"{benchmark}"
                    f"_d{dim}"
                    f"_mean_mi.png"
                ),
        )

    # ========================================================
    # Summary
    # ========================================================

    raw_df = pd.DataFrame(
        raw_rows
    )

    summary = (
        raw_df
        .groupby(
            [
                "benchmark",
                "dimension",
            ]
        )
        .agg(
            mi_mean=(
                "mi_mean",
                "mean",
            ),

            mi_max=(
                "mi_max",
                "mean",
            ),

            nearest_neighbor_mi=(
                "nearest_neighbor_mi",
                "mean",
            ),

            distance2_mi=(
                "distance2_mi",
                "mean",
            ),

            long_range_mi=(
                "long_range_mi",
                "mean",
            ),

            locality_ratio=(
                "locality_ratio",
                "mean",
            ),
        )
        .reset_index()
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
        "COPULA MI ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print()

    print(
        summary.to_string(
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
        "--marginal-steps",
        type=int,
        default=
            MARGINAL_STEPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            BATCH_SIZE,
    )

    parser.add_argument(
        "--n-mi",
        type=int,
        default=
            N_MI,
    )

    parser.add_argument(
        "--mi-bins",
        type=int,
        default=
            MI_BINS,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=
            TOP_K,
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
            "curved_ridge",
            "sharp_broad",
        ]

        args.dims = [8]

        args.seeds = [0]

        args.marginal_steps = 100
        args.batch_size = 1024

        args.n_mi = 10_000
        args.mi_bins = 16

        args.output = (
            "results/"
            "copula_mi_analysis_smoke"
        )

    return args


if __name__ == "__main__":

    args = parse_args()

    run(args)