import argparse
import json
from pathlib import Path

import pandas as pd
import torch

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

DEFAULT_HISTORY_FILE = (
    "results/"
    "iterative_bayesian_edge_pruning_sharp_fast/"
    "sharp_broad_graph_history.csv"
)

DEFAULT_FINAL_GRAPH_FILE = (
    "results/"
    "iterative_bayesian_edge_pruning_sharp_fast/"
    "final_graph_edges.csv"
)

DEFAULT_OUTPUT = (
    "results/"
    "iterative_pruning_path_benchmark"
)

DEFAULT_BENCHMARK = "sharp_broad"

SEED = 0

MARGINAL_STEPS = 1200
COPULA_STEPS = 1800

BATCH_SIZE = 4096

N_EVAL = 100_000
N_KL = 50_000


# ============================================================
# Graph utilities
# ============================================================

def parse_graph_json(
    graph_string,
):
    raw = json.loads(
        graph_string
    )

    parents = {
        j: []
        for j in range(DIM)
    }

    for child_string, parent_list in (
        raw.items()
    ):

        child = int(
            child_string
        )

        parents[
            child
        ] = sorted(
            int(parent)
            for parent in parent_list
        )

    return parents


def load_final_graph(
    path,
    benchmark,
):

    df = pd.read_csv(
        path
    )

    current = df[
        df[
            "benchmark"
        ] == benchmark
    ]

    parents = {
        j: []
        for j in range(DIM)
    }

    for row in (
        current.itertuples()
    ):

        parents[
            int(
                row.child
            )
        ].append(
            int(
                row.parent
            )
        )

    for child in parents:

        parents[
            child
        ] = sorted(
            parents[
                child
            ]
        )

    return parents


# ============================================================
# Load complete pruning path
# ============================================================

def load_graph_path(
    history_file,
    final_graph_file,
    benchmark,
):

    history = pd.read_csv(
        history_file
    )

    history = (
        history[
            history[
                "benchmark"
            ] == benchmark
        ]
        .sort_values(
            "round"
        )
    )

    graph_entries = []

    # --------------------------------------------------------
    # Graph entering each pruning round
    # --------------------------------------------------------

    for row in (
        history.itertuples()
    ):

        parents = parse_graph_json(
            row.graph_before
        )

        graph_entries.append(
            {
                "stage":
                    f"round_{int(row.round)}",

                "round":
                    int(
                        row.round
                    ),

                "parents":
                    parents,

                "edge_count":
                    count_edges(
                        parents
                    ),
            }
        )

    # --------------------------------------------------------
    # Final graph after final pruning action
    # --------------------------------------------------------

    final_parents = (
        load_final_graph(
            final_graph_file,
            benchmark,
        )
    )

    final_count = count_edges(
        final_parents
    )

    already_present = any(
        entry[
            "edge_count"
        ] == final_count
        and
        entry[
            "parents"
        ] == final_parents

        for entry in graph_entries
    )

    if not already_present:

        graph_entries.append(
            {
                "stage":
                    "final",

                "round":
                    len(
                        graph_entries
                    ),

                "parents":
                    final_parents,

                "edge_count":
                    final_count,
            }
        )

    return graph_entries


# ============================================================
# Print graph
# ============================================================

def print_graph(
    stage,
    parents,
):

    print()
    print(
        "-" * 90
    )

    print(
        f"{stage} | "
        f"edges="
        f"{count_edges(parents)}"
    )

    for child in range(
        DIM
    ):

        print(
            f"  child {child}: "
            f"{parents[child]}"
        )


# ============================================================
# Main benchmark
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

    # ========================================================
    # Load pruning path
    # ========================================================

    graph_path = (
        load_graph_path(
            history_file=
                args.history_file,

            final_graph_file=
                args.final_graph_file,

            benchmark=
                args.benchmark,
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ITERATIVE PRUNING PATH BENCHMARK"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Graphs to evaluate:"
    )

    for entry in graph_path:

        print(
            f"  {entry['stage']:10s} "
            f"| edges="
            f"{entry['edge_count']}"
        )

    # ========================================================
    # Train ONE shared marginal bank
    # ========================================================

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
        "Training shared marginals..."
    )

    train_marginals(
        marginal_bank,
        args.benchmark,
        device,
        args.marginal_steps,
        args.batch_size,
    )

    marginal_state = {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in marginal_bank
        .state_dict()
        .items()
    }

    del marginal_bank

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # Helper: recreate identical frozen marginals
    # ========================================================

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

        bank.eval()

        for parameter in (
            bank.parameters()
        ):

            parameter.requires_grad = False

        return bank


    # ========================================================
    # Evaluate every graph
    # ========================================================

    all_rows = []
    graph_rows = []

    for graph_id, entry in enumerate(
        graph_path
    ):

        stage = (
            entry[
                "stage"
            ]
        )

        parents = (
            entry[
                "parents"
            ]
        )

        edge_count = (
            entry[
                "edge_count"
            ]
        )

        print_graph(
            stage,
            parents,
        )

        # ----------------------------------------------------
        # Same seed for every candidate graph.
        # ----------------------------------------------------

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

        print()
        print(
            f"Training copula for "
            f"{stage}..."
        )

        print(
            f"  edges      : "
            f"{edge_count}"
        )

        print(
            f"  parameters : "
            f"{count_parameters(model)}"
        )

        train_copula(
            model,
            args.benchmark,
            device,
            args.copula_steps,
            args.batch_size,
        )

        print()
        print(
            "Evaluating..."
        )

        rows = evaluate(
            model,
            args.benchmark,
            device,
            args.n_eval,
            args.n_kl,
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
                        args.benchmark,

                    "stage":
                        stage,

                    "round":
                        entry[
                            "round"
                        ],

                    "edge_count":
                        edge_count,

                    "parameter_count":
                        parameter_count,

                    "seed":
                        args.seed,
                }
            )

            all_rows.append(
                row
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
                            args.benchmark,

                        "stage":
                            stage,

                        "edge_count":
                            edge_count,

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

        # Save after every graph
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
            / "graphs.csv",
            index=False,
        )

        del model

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
                "stage",
                "round",
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

    summary = (
        summary
        .sort_values(
            "edge_count",
            ascending=False,
        )
        .reset_index(
            drop=True
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
        "PRUNING PATH BENCHMARK COMPLETE"
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
        f"  "
        f"{output_dir / 'raw_results.csv'}"
    )

    print(
        f"  "
        f"{output_dir / 'graphs.csv'}"
    )

    print(
        f"  "
        f"{output_dir / 'summary.csv'}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history-file",
        type=str,
        default=
            DEFAULT_HISTORY_FILE,
    )

    parser.add_argument(
        "--final-graph-file",
        type=str,
        default=
            DEFAULT_FINAL_GRAPH_FILE,
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=
            DEFAULT_BENCHMARK,
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
            "iterative_pruning_path_benchmark_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )