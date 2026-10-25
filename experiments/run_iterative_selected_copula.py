import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the already validated training/evaluation machinery.
from run_bayesian_selected_copula_benchmark import (
    DIM,
    MarginalBank,
    StructuredCopula,
    MarginalCopulaModel,
    train_marginals,
    train_copula,
    evaluate,
    count_parameters,
    count_edges,
)

from run_copula_bin_sweep import (
    seed_everything,
)


# ============================================================
# Defaults
# ============================================================

DEFAULT_GRAPH_FILE = (
    "results/"
    "iterative_bayesian_edge_pruning_sharp_serious/"
    "final_graph_edges.csv"
)

DEFAULT_OUTPUT = (
    "results/"
    "iterative_selected_copula"
)

DEFAULT_BENCHMARKS = [
    "sharp_broad",
]

SEED = 0

MARGINAL_STEPS = 1200
COPULA_STEPS = 1800

BATCH_SIZE = 4096

N_EVAL = 100_000
N_KL = 50_000


# ============================================================
# Load iterative graph
# ============================================================

def load_iterative_graph(
    graph_file,
    benchmark,
):
    """
    Read the graph produced by

        run_iterative_bayesian_edge_pruning.py

    Expected columns:

        benchmark
        parent
        child

    Returns
    -------
    parents : dict[int, list[int]]

        Example:

        {
            0: [],
            1: [0],
            2: [0, 1],
            ...
        }
    """

    df = pd.read_csv(
        graph_file
    )

    required = {
        "benchmark",
        "parent",
        "child",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Iterative graph file is missing "
            f"required columns: {sorted(missing)}"
        )

    current = (
        df[
            df["benchmark"]
            == benchmark
        ]
        .copy()
    )

    parents = {
        j: []
        for j in range(DIM)
    }

    for row in (
        current.itertuples()
    ):

        parent = int(
            row.parent
        )

        child = int(
            row.child
        )

        if parent >= child:

            raise ValueError(
                f"Invalid autoregressive edge "
                f"{parent} -> {child}"
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
            set(
                parents[
                    child
                ]
            )
        )

    return parents


# ============================================================
# Pretty printing
# ============================================================

def print_graph(
    parents,
):

    print()

    print(
        "Iterative Bayesian graph:"
    )

    print()

    for child in range(
        DIM
    ):

        print(
            f"  child {child}: "
            f"{parents[child]}"
        )

    print()

    print(
        f"Total edges: "
        f"{count_edges(parents)}"
    )


# ============================================================
# Run one benchmark
# ============================================================

def run_benchmark(
    benchmark,
    args,
    device,
):

    print()
    print(
        "=" * 100
    )

    print(
        f"ITERATIVE SELECTED COPULA: "
        f"{benchmark}"
    )

    print(
        "=" * 100
    )

    # ========================================================
    # 1. Load final iterative graph
    # ========================================================

    parents = load_iterative_graph(
        args.graph_file,
        benchmark,
    )

    print_graph(
        parents
    )

    # ========================================================
    # 2. Train marginals
    # ========================================================

    #
    # Same seed used for all stages.
    #
    seed_everything(
        args.seed
    )

    marginal_bank = (
        MarginalBank(
            DIM
        )
        .to(device)
    )

    print()
    print(
        "Training marginals..."
    )

    train_marginals(
        marginal_bank,
        benchmark,
        device,
        args.marginal_steps,
        args.batch_size,
    )

    # ========================================================
    # 3. Freeze marginals
    # ========================================================

    for parameter in (
        marginal_bank.parameters()
    ):

        parameter.requires_grad = False

    marginal_bank.eval()

    # ========================================================
    # 4. Build TRUE sparse copula from iterative graph
    # ========================================================

    #
    # Important:
    #
    # This is NOT the masked intervention network.
    #
    # We now instantiate a normal StructuredCopula whose
    # physical input dimensions correspond exactly to the
    # selected parent set.
    #
    copula = StructuredCopula(
        DIM,
        parents,
    )

    model = (
        MarginalCopulaModel(
            marginal_bank,
            copula,
            DIM,
        )
        .to(device)
    )

    print()
    print(
        "Training sparse copula..."
    )

    print(
        f"  edges      : "
        f"{count_edges(parents)}"
    )

    print(
        f"  parameters : "
        f"{count_parameters(model)}"
    )

    # Reset RNG before copula training.
    seed_everything(
        args.seed
    )

    train_copula(
        model,
        benchmark,
        device,
        args.copula_steps,
        args.batch_size,
    )

    # ========================================================
    # 5. Evaluation
    # ========================================================

    print()
    print(
        "Evaluating..."
    )

    rows = evaluate(
        model,
        benchmark,
        device,
        args.n_eval,
        args.n_kl,
    )

    edge_count = count_edges(
        parents
    )

    parameter_count = (
        count_parameters(
            model
        )
    )

    for row in rows:

        row.update(
            {
                "benchmark":
                    benchmark,

                "model":
                    "iterative_bayesian",

                "edge_count":
                    edge_count,

                "parameter_count":
                    parameter_count,

                "seed":
                    args.seed,
            }
        )

    return (
        rows,
        parents,
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

    print()
    print(
        f"Device: {device}"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []
    graph_rows = []

    # ========================================================
    # Benchmarks
    # ========================================================

    for benchmark in (
        args.benchmarks
    ):

        rows, parents = (
            run_benchmark(
                benchmark,
                args,
                device,
            )
        )

        all_rows.extend(
            rows
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

        # Save incrementally.
        pd.DataFrame(
            all_rows
        ).to_csv(
            output_dir
            / "raw_results.csv",
            index=False,
        )

        pd.DataFrame(
            graph_rows
        ).to_csv(
            output_dir
            / "graph.csv",
            index=False,
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # Summary
    # ========================================================

    raw_df = pd.DataFrame(
        all_rows
    )

    summary = (
        raw_df
        .groupby(
            [
                "benchmark",
                "model",
                "edge_count",
                "parameter_count",
                "seed",
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

            forward_kl_std=(
                "forward_kl",
                "std",
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

            estimate_std=(
                "estimate",
                "std",
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
        "ITERATIVE SELECTED COPULA COMPLETE"
    )

    print(
        "=" * 120
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        f"  {output_dir / 'raw_results.csv'}"
    )

    print(
        f"  {output_dir / 'graph.csv'}"
    )

    print(
        f"  {output_dir / 'summary.csv'}"
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
        default=
            SEED,
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
            DEFAULT_OUTPUT,
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
            "iterative_selected_copula_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )