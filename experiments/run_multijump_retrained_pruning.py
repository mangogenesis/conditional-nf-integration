import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_copula_bin_sweep import (
    seed_everything,
)

from run_iterative_bayesian_edge_pruning_frozen_null import (
    DIM,
    DEFAULT_SEEDS,
    KEEP_PROB,
    MARGINAL_STEPS,
    MASKED_STEPS,
    BATCH_SIZE,
    N_EVAL,
    MCMC_ITER,
    MCMC_BURN,
    MCMC_THIN,
    RHO_A,
    RHO_B,
    BAYES_RANDOM_SEED,
    PIP_REMOVE_THRESHOLD,

    bayesian_round,
    choose_removals,

    full_ar_graph,
    copy_graph,
    count_edges,
    graph_as_string,
)

from run_iterative_bayesian_edge_pruning_fast import (
    REFINE_STEPS,
)

from run_performance_guarded_structure_benchmark import (
    train_all_marginal_states,
    collect_evidence,
    true_graph_description,
)

# Reuse the REAL StructuredCopula machinery from the
# retrained-performance script that just worked.
from run_retrained_performance_guarded_pruning import (
    train_and_score_candidate,
    compare_performance,
)


# ============================================================
# Defaults
# ============================================================

DEFAULT_OUTPUT = (
    "results/"
    "multijump_retrained_pruning"
)

COPULA_STEPS = 1800

PERFORMANCE_SEED = 0

PERF_N_EVAL = 100_000
PERF_N_KL = 50_000

MAX_ROUNDS = 10

ESS_TOL = 0.0
KL_TOL = 0.0


# ============================================================
# Graph manipulation
# ============================================================

def apply_removal_set(
    graph,
    removals,
):

    new_graph = copy_graph(
        graph
    )

    for parent, child in removals:

        if parent in new_graph[
            child
        ]:

            new_graph[
                child
            ].remove(
                parent
            )

    return new_graph


# ============================================================
# Build unique multi-jump candidates
# ============================================================

def make_candidate_removal_sets(
    proposals,
    pip_lookup,
    delta_lookup,
    max_prefix=None,
):
    """
    proposals:
        Bayesian proposed removals.

    We rank them weakest-first and generate:

        every singleton

        cumulative prefixes:
            top-2
            top-3
            top-4
            ...
            full proposal set

    This allows nonlocal moves while keeping the number of
    actual StructuredCopula trainings manageable.
    """

    ranked = sorted(
        proposals,
        key=lambda edge: (
            pip_lookup[
                edge
            ],
            delta_lookup[
                edge
            ],
        ),
    )

    candidates = []

    # --------------------------------------------------------
    # Every singleton.
    #
    # Useful if the joint jump is too aggressive.
    # --------------------------------------------------------

    for edge in ranked:

        candidates.append(
            (
                f"single_{edge[0]}_{edge[1]}",
                [edge],
            )
        )

    # --------------------------------------------------------
    # Cumulative prefixes.
    #
    # With d=8 there are at most 7 proposals anyway, so
    # evaluating all prefixes is still reasonable.
    # --------------------------------------------------------

    max_k = len(
        ranked
    )

    if max_prefix is not None:

        max_k = min(
            max_k,
            max_prefix,
        )

    for k in range(
        2,
        max_k + 1,
    ):

        removal_set = (
            ranked[
                :k
            ]
        )

        candidates.append(
            (
                f"prefix_{k}",
                removal_set,
            )
        )

    # --------------------------------------------------------
    # Deduplicate removal sets
    # --------------------------------------------------------

    unique = []

    seen = set()

    for name, removal_set in candidates:

        key = tuple(
            sorted(
                removal_set
            )
        )

        if key in seen:

            continue

        seen.add(
            key
        )

        unique.append(
            (
                name,
                removal_set,
            )
        )

    return unique


# ============================================================
# Candidate ranking
# ============================================================

def choose_best_accepted(
    accepted_candidates,
):
    """
    Primary objective:
        sparsest accepted graph.

    Tie-break:
        higher ESS.

    Second tie-break:
        lower KL.
    """

    if len(
        accepted_candidates
    ) == 0:

        return None

    accepted_candidates = sorted(
        accepted_candidates,
        key=lambda item: (
            item[
                "edge_count"
            ],
            -item[
                "metrics"
            ][
                "ess_mean"
            ],
            item[
                "metrics"
            ][
                "kl_mean"
            ],
        ),
    )

    return accepted_candidates[
        0
    ]


# ============================================================
# Print performance
# ============================================================

def print_candidate_result(
    current_metrics,
    candidate_metrics,
    comparison,
):

    print(
        f"      ESS current   = "
        f"{current_metrics['ess_mean']:.6f}"
    )

    print(
        f"      ESS candidate = "
        f"{candidate_metrics['ess_mean']:.6f}"
    )

    print(
        f"      delta ESS     = "
        f"{comparison['delta_ess']:+.6f}"
    )

    print()

    print(
        f"      KL current    = "
        f"{current_metrics['kl_mean']:.6f}"
    )

    print(
        f"      KL candidate  = "
        f"{candidate_metrics['kl_mean']:.6f}"
    )

    print(
        f"      delta KL      = "
        f"{comparison['delta_kl']:+.6f}"
    )

    print(
        f"      decision      = "
        f"{'PASS' if comparison['accept'] else 'FAIL'}"
    )


# ============================================================
# Run one benchmark
# ============================================================

def run_benchmark(
    benchmark,
    args,
    device,
    output_dir,
):

    print()
    print(
        "=" * 110
    )

    print(
        "MULTI-JUMP RETRAINED PRUNING: "
        f"{benchmark}"
    )

    print(
        true_graph_description(
            benchmark
        )
    )

    print(
        "=" * 110
    )

    # ========================================================
    # Marginals
    # ========================================================

    marginal_states = (
        train_all_marginal_states(
            benchmark,
            args,
            device,
        )
    )

    performance_marginal_state = (
        marginal_states[
            args.performance_seed
        ]
    )

    # ========================================================
    # Initial full graph
    # ========================================================

    graph = (
        full_ar_graph(
            DIM
        )
    )

    previous_graph = None

    model_cache = {}
    evidence_cache = {}

    frozen_sigma0 = None

    history_rows = []
    trial_rows = []

    # ========================================================
    # Initial REAL graph
    # ========================================================

    print()
    print(
        "Training initial REAL full graph..."
    )

    current_metrics = (
        train_and_score_candidate(
            benchmark=
                benchmark,

            graph=
                graph,

            marginal_state=
                performance_marginal_state,

            args=
                args,

            device=
                device,

            candidate_id=
                "initial_full",
        )
    )

    print()
    print(
        f"Initial edges = "
        f"{count_edges(graph)}"
    )

    print(
        f"Initial ESS   = "
        f"{current_metrics['ess_mean']:.6f}"
    )

    print(
        f"Initial KL    = "
        f"{current_metrics['kl_mean']:.6f}"
    )

    # ========================================================
    # Iterative graph search
    # ========================================================

    for round_id in range(
        args.max_rounds
    ):

        print()
        print(
            "#" * 110
        )

        print(
            f"ROUND {round_id}"
        )

        print(
            f"Current graph: "
            f"{count_edges(graph)} edges"
        )

        print(
            f"ESS = "
            f"{current_metrics['ess_mean']:.6f}"
        )

        print(
            f"KL  = "
            f"{current_metrics['kl_mean']:.6f}"
        )

        print(
            graph
        )

        print(
            "#" * 110
        )

        # ====================================================
        # Masked intervention evidence
        # ====================================================

        evidence_df = (
            collect_evidence(
                benchmark=
                    benchmark,

                round_id=
                    round_id,

                graph=
                    graph,

                previous_graph=
                    previous_graph,

                marginal_states=
                    marginal_states,

                model_cache=
                    model_cache,

                evidence_cache=
                    evidence_cache,

                args=
                    args,

                device=
                    device,
            )
        )

        if len(
            evidence_df
        ) == 0:

            print(
                "No active edges remain."
            )

            break

        # ====================================================
        # Bayesian posterior
        # ====================================================

        posterior_df, posterior = (
            bayesian_round(
                evidence_df,
                args,
                fixed_sigma0=
                    frozen_sigma0,
            )
        )

        if round_id == 0:

            frozen_sigma0 = float(
                posterior[
                    "sigma0_mean"
                ]
            )

            print()
            print(
                f"Frozen sigma0 = "
                f"{frozen_sigma0:.8f}"
            )

        # ====================================================
        # Weak-edge proposals
        # ====================================================

        proposals = (
            choose_removals(
                posterior_df,
                args.pip_threshold,
            )
        )

        if len(
            proposals
        ) == 0:

            print()
            print(
                "STOP: Bayesian model "
                "has no removal proposals."
            )

            break

        pip_lookup = {
            (
                int(row.parent),
                int(row.child),
            ):
                float(
                    row.pip
                )

            for row in (
                posterior_df.itertuples()
            )
        }

        delta_lookup = {
            (
                int(row.parent),
                int(row.child),
            ):
                float(
                    row.delta_mean
                )

            for row in (
                posterior_df.itertuples()
            )
        }

        proposals = sorted(
            proposals,
            key=lambda edge: (
                pip_lookup[
                    edge
                ],
                delta_lookup[
                    edge
                ],
            ),
        )

        print()
        print(
            "Bayesian weak-edge ranking:"
        )

        for rank, edge in enumerate(
            proposals,
            start=1,
        ):

            print(
                f"  {rank:2d}. "
                f"{edge[0]}->{edge[1]} "
                f"PIP="
                f"{pip_lookup[edge]:.5f} "
                f"delta="
                f"{delta_lookup[edge]:.6f}"
            )

        # ====================================================
        # Generate multi-jump candidates
        # ====================================================

        candidate_sets = (
            make_candidate_removal_sets(
                proposals=
                    proposals,

                pip_lookup=
                    pip_lookup,

                delta_lookup=
                    delta_lookup,

                max_prefix=
                    args.max_jump,
            )
        )

        print()
        print(
            f"Testing "
            f"{len(candidate_sets)} "
            f"candidate jumps..."
        )

        accepted_candidates = []

        # ====================================================
        # REAL NF evaluation for every candidate
        # ====================================================

        for candidate_index, (
            candidate_name,
            removals,
        ) in enumerate(
            candidate_sets
        ):

            candidate_graph = (
                apply_removal_set(
                    graph,
                    removals,
                )
            )

            edge_count = (
                count_edges(
                    candidate_graph
                )
            )

            print()
            print(
                "=" * 85
            )

            print(
                f"CANDIDATE "
                f"{candidate_index + 1}/"
                f"{len(candidate_sets)}"
            )

            print(
                f"name     = "
                f"{candidate_name}"
            )

            print(
                f"remove   = "
                f"{removals}"
            )

            print(
                f"edges    = "
                f"{count_edges(graph)} "
                f"-> "
                f"{edge_count}"
            )

            print(
                "=" * 85
            )

            candidate_metrics = (
                train_and_score_candidate(
                    benchmark=
                        benchmark,

                    graph=
                        candidate_graph,

                    marginal_state=
                        performance_marginal_state,

                    args=
                        args,

                    device=
                        device,

                    candidate_id=
                        (
                            f"round_"
                            f"{round_id}_"
                            f"{candidate_name}"
                        ),
                )
            )

            comparison = (
                compare_performance(
                    current=
                        current_metrics,

                    candidate=
                        candidate_metrics,

                    ess_tol=
                        args.ess_tol,

                    kl_tol=
                        args.kl_tol,
                )
            )

            print_candidate_result(
                current_metrics,
                candidate_metrics,
                comparison,
            )

            trial_rows.append(
                {
                    "benchmark":
                        benchmark,

                    "round":
                        round_id,

                    "candidate":
                        candidate_name,

                    "removed_edges":
                        str(
                            removals
                        ),

                    "n_removed":
                        len(
                            removals
                        ),

                    "edge_count_before":
                        count_edges(
                            graph
                        ),

                    "edge_count_after":
                        edge_count,

                    "ess_before":
                        current_metrics[
                            "ess_mean"
                        ],

                    "ess_after":
                        candidate_metrics[
                            "ess_mean"
                        ],

                    "delta_ess":
                        comparison[
                            "delta_ess"
                        ],

                    "kl_before":
                        current_metrics[
                            "kl_mean"
                        ],

                    "kl_after":
                        candidate_metrics[
                            "kl_mean"
                        ],

                    "delta_kl":
                        comparison[
                            "delta_kl"
                        ],

                    "accepted":
                        int(
                            comparison[
                                "accept"
                            ]
                        ),
                }
            )

            if comparison[
                "accept"
            ]:

                accepted_candidates.append(
                    {
                        "name":
                            candidate_name,

                        "removals":
                            removals,

                        "graph":
                            candidate_graph,

                        "edge_count":
                            edge_count,

                        "metrics":
                            candidate_metrics,

                        "comparison":
                            comparison,
                    }
                )

            # Save after every expensive candidate
            pd.DataFrame(
                trial_rows
            ).to_csv(
                output_dir
                /
                f"{benchmark}_candidate_trials.csv",
                index=False,
            )

        # ====================================================
        # Find best accepted jump
        # ====================================================

        best = (
            choose_best_accepted(
                accepted_candidates
            )
        )

        if best is None:

            print()
            print(
                "=" * 90
            )

            print(
                "STOP"
            )

            print(
                "No tested sparse graph "
                "matched or improved BOTH "
                "current ESS and current KL."
            )

            print(
                "=" * 90
            )

            history_rows.append(
                {
                    "benchmark":
                        benchmark,

                    "round":
                        round_id,

                    "edges_before":
                        count_edges(
                            graph
                        ),

                    "edges_after":
                        count_edges(
                            graph
                        ),

                    "accepted_candidate":
                        None,

                    "accepted_removals":
                        "[]",

                    "ess_before":
                        current_metrics[
                            "ess_mean"
                        ],

                    "ess_after":
                        current_metrics[
                            "ess_mean"
                        ],

                    "kl_before":
                        current_metrics[
                            "kl_mean"
                        ],

                    "kl_after":
                        current_metrics[
                            "kl_mean"
                        ],

                    "graph_after":
                        graph_as_string(
                            graph
                        ),
                }
            )

            break

        # ====================================================
        # Accept sparsest performance-safe jump
        # ====================================================

        old_graph = (
            copy_graph(
                graph
            )
        )

        old_metrics = dict(
            current_metrics
        )

        graph = (
            best[
                "graph"
            ]
        )

        current_metrics = (
            best[
                "metrics"
            ]
        )

        previous_graph = (
            old_graph
        )

        print()
        print(
            "=" * 100
        )

        print(
            "BEST ACCEPTED JUMP"
        )

        print(
            f"name     = "
            f"{best['name']}"
        )

        print(
            f"removed  = "
            f"{best['removals']}"
        )

        print(
            f"edges    = "
            f"{count_edges(old_graph)} "
            f"-> "
            f"{count_edges(graph)}"
        )

        print(
            f"ESS      = "
            f"{old_metrics['ess_mean']:.6f} "
            f"-> "
            f"{current_metrics['ess_mean']:.6f}"
        )

        print(
            f"KL       = "
            f"{old_metrics['kl_mean']:.6f} "
            f"-> "
            f"{current_metrics['kl_mean']:.6f}"
        )

        print(
            "=" * 100
        )

        history_rows.append(
            {
                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "edges_before":
                    count_edges(
                        old_graph
                    ),

                "edges_after":
                    count_edges(
                        graph
                    ),

                "accepted_candidate":
                    best[
                        "name"
                    ],

                "accepted_removals":
                    str(
                        best[
                            "removals"
                        ]
                    ),

                "ess_before":
                    old_metrics[
                        "ess_mean"
                    ],

                "ess_after":
                    current_metrics[
                        "ess_mean"
                    ],

                "delta_ess":
                    (
                        current_metrics[
                            "ess_mean"
                        ]
                        -
                        old_metrics[
                            "ess_mean"
                        ]
                    ),

                "kl_before":
                    old_metrics[
                        "kl_mean"
                    ],

                "kl_after":
                    current_metrics[
                        "kl_mean"
                    ],

                "delta_kl":
                    (
                        current_metrics[
                            "kl_mean"
                        ]
                        -
                        old_metrics[
                            "kl_mean"
                        ]
                    ),

                "frozen_sigma0":
                    frozen_sigma0,

                "graph_before":
                    graph_as_string(
                        old_graph
                    ),

                "graph_after":
                    graph_as_string(
                        graph
                    ),
            }
        )

        pd.DataFrame(
            history_rows
        ).to_csv(
            output_dir
            /
            f"{benchmark}_history.csv",
            index=False,
        )

        posterior_df.to_csv(
            output_dir
            /
            (
                f"{benchmark}_"
                f"round_{round_id}_posterior.csv"
            ),
            index=False,
        )

        evidence_df.to_csv(
            output_dir
            /
            (
                f"{benchmark}_"
                f"round_{round_id}_evidence.csv"
            ),
            index=False,
        )

    # ========================================================
    # Final graph
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        f"FINAL MULTI-JUMP GRAPH: "
        f"{benchmark}"
    )

    print(
        graph
    )

    print(
        f"Edges = "
        f"{count_edges(graph)}"
    )

    print(
        f"ESS   = "
        f"{current_metrics['ess_mean']:.6f}"
    )

    print(
        f"KL    = "
        f"{current_metrics['kl_mean']:.6f}"
    )

    print(
        "=" * 110
    )

    final_edges = []

    for child in range(
        DIM
    ):

        for parent in (
            graph[
                child
            ]
        ):

            final_edges.append(
                {
                    "benchmark":
                        benchmark,

                    "parent":
                        parent,

                    "child":
                        child,
                }
            )

    summary = {
        "benchmark":
            benchmark,

        "true_structure":
            true_graph_description(
                benchmark
            ),

        "final_edge_count":
            count_edges(
                graph
            ),

        "final_ess":
            current_metrics[
                "ess_mean"
            ],

        "final_kl":
            current_metrics[
                "kl_mean"
            ],

        "final_variance":
            current_metrics[
                "variance_mean"
            ],

        "final_relative_error":
            current_metrics[
                "relative_error_mean"
            ],

        "n_rounds":
            len(
                history_rows
            ),

        "final_graph":
            graph_as_string(
                graph
            ),
    }

    return (
        summary,
        final_edges,
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

    summaries = []
    all_edges = []

    for benchmark in (
        args.benchmarks
    ):

        summary, edges = (
            run_benchmark(
                benchmark,
                args,
                device,
                output_dir,
            )
        )

        summaries.append(
            summary
        )

        all_edges.extend(
            edges
        )

        pd.DataFrame(
            summaries
        ).to_csv(
            output_dir
            / "summary.csv",
            index=False,
        )

        pd.DataFrame(
            all_edges
        ).to_csv(
            output_dir
            / "final_graph_edges.csv",
            index=False,
        )

    print()
    print(
        "=" * 120
    )

    print(
        "MULTI-JUMP RETRAINED "
        "PRUNING COMPLETE"
    )

    print(
        "=" * 120
    )

    print()

    print(
        pd.DataFrame(
            summaries
        ).to_string(
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
            "single_gaussian",
            "curved_ridge",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=
            DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--performance-seed",
        type=int,
        default=
            PERFORMANCE_SEED,
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
        "--refine-steps",
        type=int,
        default=
            REFINE_STEPS,
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
        "--perf-n-eval",
        type=int,
        default=
            PERF_N_EVAL,
    )

    parser.add_argument(
        "--perf-n-kl",
        type=int,
        default=
            PERF_N_KL,
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=123456,
    )

    parser.add_argument(
        "--keep-prob",
        type=float,
        default=
            KEEP_PROB,
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=
            MAX_ROUNDS,
    )

    # Maximum size of a cumulative jump.
    #
    # None / large value means evaluate all available prefixes.
    parser.add_argument(
        "--max-jump",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--pip-threshold",
        type=float,
        default=
            PIP_REMOVE_THRESHOLD,
    )

    parser.add_argument(
        "--ess-tol",
        type=float,
        default=
            ESS_TOL,
    )

    parser.add_argument(
        "--kl-tol",
        type=float,
        default=
            KL_TOL,
    )

    parser.add_argument(
        "--mcmc-iter",
        type=int,
        default=
            MCMC_ITER,
    )

    parser.add_argument(
        "--mcmc-burn",
        type=int,
        default=
            MCMC_BURN,
    )

    parser.add_argument(
        "--mcmc-thin",
        type=int,
        default=
            MCMC_THIN,
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
        "--bayes-seed",
        type=int,
        default=
            BAYES_RANDOM_SEED,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            DEFAULT_OUTPUT,
    )

    return parser.parse_args()


if __name__ == "__main__":

    run(
        parse_args()
    )