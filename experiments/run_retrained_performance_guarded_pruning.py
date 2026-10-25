import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_copula_bin_sweep import seed_everything

from run_bayesian_selected_copula_benchmark import (
    DIM,
    MarginalBank,
    StructuredCopula,
    MarginalCopulaModel,
    train_copula,
    evaluate,
    count_parameters,
)

from run_iterative_bayesian_edge_pruning_frozen_null import (
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

# We reuse the extended benchmark definitions and
# fast masked-evidence machinery from the script
# you just smoke-tested.
from run_performance_guarded_structure_benchmark import (
    ALL_BENCHMARKS,
    sample_target,
    target_logpdf,

    train_all_marginal_states,
    collect_evidence,
    true_graph_description,
)


# ============================================================
# Defaults
# ============================================================

DEFAULT_OUTPUT = (
    "results/"
    "retrained_performance_guarded_pruning"
)

COPULA_STEPS = 1800

PERFORMANCE_SEED = 0

# Use the same evaluation sizes as the earlier serious
# StructuredCopula benchmark.
PERF_N_EVAL = 100_000
PERF_N_KL = 50_000

MAX_ROUNDS = 10

# Strict by default.
ESS_TOL = 0.0
KL_TOL = 0.0


# ============================================================
# Graph helpers
# ============================================================

def apply_edges(
    graph,
    removals,
):

    new_graph = copy_graph(
        graph
    )

    for parent, child in removals:

        if parent in new_graph[child]:

            new_graph[
                child
            ].remove(
                parent
            )

    return new_graph


def same_graph(
    g1,
    g2,
):

    for child in range(
        DIM
    ):

        if sorted(
            g1[child]
        ) != sorted(
            g2[child]
        ):

            return False

    return True


# ============================================================
# Shared marginal bank for REAL copula performance
# ============================================================

def build_performance_marginal_state(
    benchmark,
    marginal_states,
    performance_seed,
):

    if performance_seed not in marginal_states:

        raise RuntimeError(
            f"Performance seed "
            f"{performance_seed} is not among "
            f"trained marginal seeds "
            f"{list(marginal_states.keys())}"
        )

    return marginal_states[
        performance_seed
    ]


def make_frozen_bank(
    state,
    device,
):

    bank = (
        MarginalBank(
            DIM
        )
        .to(device)
    )

    bank.load_state_dict(
        state
    )

    bank.eval()

    for parameter in (
        bank.parameters()
    ):

        parameter.requires_grad = False

    return bank


# ============================================================
# Train an ACTUAL StructuredCopula for one graph
# ============================================================

def train_real_graph(
    benchmark,
    graph,
    marginal_state,
    args,
    device,
    train_seed,
):

    seed_everything(
        train_seed
    )

    bank = make_frozen_bank(
        marginal_state,
        device,
    )

    copula = StructuredCopula(
        DIM,
        graph,
    )

    model = (
        MarginalCopulaModel(
            bank,
            copula,
            DIM,
        )
        .to(device)
    )

    print(
        f"    graph edges      = "
        f"{count_edges(graph)}"
    )

    print(
        f"    model parameters = "
        f"{count_parameters(model)}"
    )

    train_copula(
        model,
        benchmark,
        device,
        args.copula_steps,
        args.batch_size,
    )

    return model


# ============================================================
# Actual performance evaluation
# ============================================================

def evaluate_real_graph(
    model,
    benchmark,
    args,
    device,
):

    # Resetting the evaluation RNG gives us much more
    # comparable candidate/current estimates.
    seed_everything(
        args.eval_seed
    )

    rows = evaluate(
        model,
        benchmark,
        device,
        args.perf_n_eval,
        args.perf_n_kl,
    )

    df = pd.DataFrame(
        rows
    )

    return {
        "ess_mean":
            float(
                df["ess"].mean()
            ),

        "ess_std":
            float(
                df["ess"].std()
            ),

        "kl_mean":
            float(
                df[
                    "forward_kl"
                ].mean()
            ),

        "kl_std":
            float(
                df[
                    "forward_kl"
                ].std()
            ),

        "variance_mean":
            float(
                df[
                    "variance"
                ].mean()
            ),

        "relative_error_mean":
            float(
                df[
                    "relative_error"
                ].mean()
            ),

        "estimate_mean":
            float(
                df[
                    "estimate"
                ].mean()
            ),

        "estimate_std":
            float(
                df[
                    "estimate"
                ].std()
            ),
    }


# ============================================================
# Strict performance criterion
# ============================================================

def compare_performance(
    current,
    candidate,
    ess_tol,
    kl_tol,
):

    delta_ess = (
        candidate[
            "ess_mean"
        ]
        -
        current[
            "ess_mean"
        ]
    )

    delta_kl = (
        candidate[
            "kl_mean"
        ]
        -
        current[
            "kl_mean"
        ]
    )

    ess_ok = (
        delta_ess
        >=
        -ess_tol
    )

    kl_ok = (
        delta_kl
        <=
        kl_tol
    )

    accept = (
        ess_ok
        and
        kl_ok
    )

    return {
        "accept":
            accept,

        "ess_ok":
            ess_ok,

        "kl_ok":
            kl_ok,

        "delta_ess":
            delta_ess,

        "delta_kl":
            delta_kl,
    }


# ============================================================
# Train + evaluate candidate
# ============================================================

def train_and_score_candidate(
    benchmark,
    graph,
    marginal_state,
    args,
    device,
    candidate_id,
):

    # IMPORTANT:
    #
    # use the SAME training seed for every graph.
    # This removes one unnecessary source of variability.
    train_seed = (
        args.performance_seed
    )

    print()
    print(
        f"  Training REAL candidate "
        f"{candidate_id}"
    )

    model = train_real_graph(
        benchmark=
            benchmark,

        graph=
            graph,

        marginal_state=
            marginal_state,

        args=
            args,

        device=
            device,

        train_seed=
            train_seed,
    )

    metrics = evaluate_real_graph(
        model=
            model,

        benchmark=
            benchmark,

        args=
            args,

        device=
            device,
    )

    print(
        f"    ESS = "
        f"{metrics['ess_mean']:.6f}"
    )

    print(
        f"    KL  = "
        f"{metrics['kl_mean']:.6f}"
    )

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return metrics


# ============================================================
# Pretty comparison
# ============================================================

def print_comparison(
    current_metrics,
    candidate_metrics,
    comparison,
):

    print()

    print(
        f"    ESS current   = "
        f"{current_metrics['ess_mean']:.6f}"
    )

    print(
        f"    ESS candidate = "
        f"{candidate_metrics['ess_mean']:.6f}"
    )

    print(
        f"    delta ESS     = "
        f"{comparison['delta_ess']:+.6f}"
    )

    print()

    print(
        f"    KL current    = "
        f"{current_metrics['kl_mean']:.6f}"
    )

    print(
        f"    KL candidate  = "
        f"{candidate_metrics['kl_mean']:.6f}"
    )

    print(
        f"    delta KL      = "
        f"{comparison['delta_kl']:+.6f}"
    )

    print()

    print(
        f"    decision      = "
        f"{'ACCEPT' if comparison['accept'] else 'REJECT'}"
    )


# ============================================================
# One benchmark
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
        f"RETRAINED PERFORMANCE-GUARDED PRUNING: "
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
    # 1. Train marginals once for the Bayesian evidence seeds
    # ========================================================

    marginal_states = (
        train_all_marginal_states(
            benchmark,
            args,
            device,
        )
    )

    performance_marginal_state = (
        build_performance_marginal_state(
            benchmark,
            marginal_states,
            args.performance_seed,
        )
    )

    # ========================================================
    # 2. Full starting graph
    # ========================================================

    graph = full_ar_graph(
        DIM
    )

    previous_graph = None

    model_cache = {}
    evidence_cache = {}

    frozen_sigma0 = None

    history = []
    trials = []

    # ========================================================
    # 3. Train REAL full graph once
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
                "full",
        )
    )

    print()
    print(
        "Initial performance:"
    )

    print(
        f"  edges = "
        f"{count_edges(graph)}"
    )

    print(
        f"  ESS   = "
        f"{current_metrics['ess_mean']:.6f}"
    )

    print(
        f"  KL    = "
        f"{current_metrics['kl_mean']:.6f}"
    )

    # ========================================================
    # Iterative pruning
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
            f"Current edges = "
            f"{count_edges(graph)}"
        )

        print(
            graph
        )

        print(
            f"Current REAL ESS = "
            f"{current_metrics['ess_mean']:.6f}"
        )

        print(
            f"Current REAL KL  = "
            f"{current_metrics['kl_mean']:.6f}"
        )

        print(
            "#" * 110
        )

        # ====================================================
        # Bayesian evidence
        # ====================================================

        evidence_df = collect_evidence(
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

        if len(
            evidence_df
        ) == 0:

            print(
                "No active edges remain."
            )

            break

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

        proposals = choose_removals(
            posterior_df,
            args.pip_threshold,
        )

        if len(
            proposals
        ) == 0:

            print()
            print(
                "STOP: Bayesian model "
                "has no pruning proposals."
            )

            break

        pip_lookup = {
            (
                int(row.parent),
                int(row.child),
            ):
                float(row.pip)

            for row in (
                posterior_df.itertuples()
            )
        }

        delta_lookup = {
            (
                int(row.parent),
                int(row.child),
            ):
                float(row.delta_mean)

            for row in (
                posterior_df.itertuples()
            )
        }

        proposals = sorted(
            proposals,
            key=lambda edge: (
                pip_lookup[edge],
                delta_lookup[edge],
            ),
        )

        print()
        print(
            "Bayesian proposed removals:"
        )

        for edge in proposals:

            print(
                f"  {edge[0]}->{edge[1]} "
                f"PIP="
                f"{pip_lookup[edge]:.4f} "
                f"delta="
                f"{delta_lookup[edge]:.6f}"
            )

        graph_before = (
            copy_graph(
                graph
            )
        )

        metrics_before = dict(
            current_metrics
        )

        accepted_removals = []

        accepted_graph = None

        accepted_metrics = None

        # ====================================================
        # A. First try the WHOLE proposed pruning round
        # ====================================================

        joint_graph = apply_edges(
            graph,
            proposals,
        )

        print()
        print(
            "=" * 80
        )

        print(
            "TRY JOINT BAYESIAN CANDIDATE"
        )

        print(
            f"{count_edges(graph)} "
            f"-> "
            f"{count_edges(joint_graph)} edges"
        )

        print(
            "=" * 80
        )

        joint_metrics = (
            train_and_score_candidate(
                benchmark=
                    benchmark,

                graph=
                    joint_graph,

                marginal_state=
                    performance_marginal_state,

                args=
                    args,

                device=
                    device,

                candidate_id=
                    f"round_{round_id}_joint",
            )
        )

        joint_cmp = compare_performance(
            current=
                current_metrics,

            candidate=
                joint_metrics,

            ess_tol=
                args.ess_tol,

            kl_tol=
                args.kl_tol,
        )

        print_comparison(
            current_metrics,
            joint_metrics,
            joint_cmp,
        )

        trials.append(
            {
                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "trial_type":
                    "joint",

                "removed_edges":
                    str(
                        proposals
                    ),

                "edges_before":
                    count_edges(
                        graph
                    ),

                "edges_after":
                    count_edges(
                        joint_graph
                    ),

                "ess_before":
                    current_metrics[
                        "ess_mean"
                    ],

                "ess_after":
                    joint_metrics[
                        "ess_mean"
                    ],

                "delta_ess":
                    joint_cmp[
                        "delta_ess"
                    ],

                "kl_before":
                    current_metrics[
                        "kl_mean"
                    ],

                "kl_after":
                    joint_metrics[
                        "kl_mean"
                    ],

                "delta_kl":
                    joint_cmp[
                        "delta_kl"
                    ],

                "accepted":
                    int(
                        joint_cmp[
                            "accept"
                        ]
                    ),
            }
        )

        if joint_cmp[
            "accept"
        ]:

            accepted_removals = list(
                proposals
            )

            accepted_graph = (
                joint_graph
            )

            accepted_metrics = (
                joint_metrics
            )

        # ====================================================
        # B. Joint candidate failed:
        #
        # try ONE removal at a time, weakest PIP first.
        #
        # We accept the first edge that improves BOTH metrics.
        #
        # This is much cheaper than exhaustive combinatorics.
        # ====================================================

        else:

            print()
            print(
                "=" * 80
            )

            print(
                "JOINT CANDIDATE REJECTED"
            )

            print(
                "Trying individual removals..."
            )

            print(
                "=" * 80
            )

            for parent, child in (
                proposals
            ):

                single_graph = apply_edges(
                    graph,
                    [
                        (
                            parent,
                            child,
                        )
                    ],
                )

                print()
                print(
                    f"Testing individual "
                    f"{parent}->{child}"
                )

                single_metrics = (
                    train_and_score_candidate(
                        benchmark=
                            benchmark,

                        graph=
                            single_graph,

                        marginal_state=
                            performance_marginal_state,

                        args=
                            args,

                        device=
                            device,

                        candidate_id=
                            (
                                f"round_{round_id}_"
                                f"{parent}_{child}"
                            ),
                    )
                )

                single_cmp = (
                    compare_performance(
                        current=
                            current_metrics,

                        candidate=
                            single_metrics,

                        ess_tol=
                            args.ess_tol,

                        kl_tol=
                            args.kl_tol,
                    )
                )

                print_comparison(
                    current_metrics,
                    single_metrics,
                    single_cmp,
                )

                trials.append(
                    {
                        "benchmark":
                            benchmark,

                        "round":
                            round_id,

                        "trial_type":
                            "single",

                        "removed_edges":
                            str(
                                [
                                    (
                                        parent,
                                        child,
                                    )
                                ]
                            ),

                        "parent":
                            parent,

                        "child":
                            child,

                        "pip":
                            pip_lookup[
                                (
                                    parent,
                                    child,
                                )
                            ],

                        "edges_before":
                            count_edges(
                                graph
                            ),

                        "edges_after":
                            count_edges(
                                single_graph
                            ),

                        "ess_before":
                            current_metrics[
                                "ess_mean"
                            ],

                        "ess_after":
                            single_metrics[
                                "ess_mean"
                            ],

                        "delta_ess":
                            single_cmp[
                                "delta_ess"
                            ],

                        "kl_before":
                            current_metrics[
                                "kl_mean"
                            ],

                        "kl_after":
                            single_metrics[
                                "kl_mean"
                            ],

                        "delta_kl":
                            single_cmp[
                                "delta_kl"
                            ],

                        "accepted":
                            int(
                                single_cmp[
                                    "accept"
                                ]
                            ),
                    }
                )

                if single_cmp[
                    "accept"
                ]:

                    accepted_removals = [
                        (
                            parent,
                            child,
                        )
                    ]

                    accepted_graph = (
                        single_graph
                    )

                    accepted_metrics = (
                        single_metrics
                    )

                    break

        # ====================================================
        # Nothing was safe to remove
        # ====================================================

        if accepted_graph is None:

            history.append(
                {
                    "benchmark":
                        benchmark,

                    "round":
                        round_id,

                    "edges_before":
                        count_edges(
                            graph_before
                        ),

                    "edges_after":
                        count_edges(
                            graph_before
                        ),

                    "proposed_removals":
                        str(
                            proposals
                        ),

                    "accepted_removals":
                        "[]",

                    "ess_before":
                        metrics_before[
                            "ess_mean"
                        ],

                    "ess_after":
                        metrics_before[
                            "ess_mean"
                        ],

                    "kl_before":
                        metrics_before[
                            "kl_mean"
                        ],

                    "kl_after":
                        metrics_before[
                            "kl_mean"
                        ],

                    "rho_mean":
                        posterior[
                            "rho_mean"
                        ],

                    "expected_edges":
                        posterior[
                            "expected_edges"
                        ],

                    "frozen_sigma0":
                        frozen_sigma0,

                    "graph_before":
                        graph_as_string(
                            graph_before
                        ),

                    "graph_after":
                        graph_as_string(
                            graph_before
                        ),
                }
            )

            print()
            print(
                "STOP: no proposed edge "
                "can be removed without "
                "decreasing ESS and/or "
                "increasing KL."
            )

            pd.DataFrame(
                history
            ).to_csv(
                output_dir
                /
                f"{benchmark}_history.csv",
                index=False,
            )

            pd.DataFrame(
                trials
            ).to_csv(
                output_dir
                /
                f"{benchmark}_performance_trials.csv",
                index=False,
            )

            break

        # ====================================================
        # Accept
        # ====================================================

        previous_graph = (
            copy_graph(
                graph
            )
        )

        graph = (
            accepted_graph
        )

        current_metrics = (
            accepted_metrics
        )

        print()
        print(
            "=" * 80
        )

        print(
            "ACCEPTED"
        )

        print(
            f"Removed: "
            f"{accepted_removals}"
        )

        print(
            f"New graph has "
            f"{count_edges(graph)} edges"
        )

        print(
            f"REAL ESS = "
            f"{current_metrics['ess_mean']:.6f}"
        )

        print(
            f"REAL KL  = "
            f"{current_metrics['kl_mean']:.6f}"
        )

        print(
            "=" * 80
        )

        history.append(
            {
                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "edges_before":
                    count_edges(
                        graph_before
                    ),

                "edges_after":
                    count_edges(
                        graph
                    ),

                "proposed_removals":
                    str(
                        proposals
                    ),

                "accepted_removals":
                    str(
                        accepted_removals
                    ),

                "ess_before":
                    metrics_before[
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
                        metrics_before[
                            "ess_mean"
                        ]
                    ),

                "kl_before":
                    metrics_before[
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
                        metrics_before[
                            "kl_mean"
                        ]
                    ),

                "rho_mean":
                    posterior[
                        "rho_mean"
                    ],

                "expected_edges":
                    posterior[
                        "expected_edges"
                    ],

                "frozen_sigma0":
                    frozen_sigma0,

                "graph_before":
                    graph_as_string(
                        graph_before
                    ),

                "graph_after":
                    graph_as_string(
                        graph
                    ),
            }
        )

        # ====================================================
        # Save continuously
        # ====================================================

        pd.DataFrame(
            history
        ).to_csv(
            output_dir
            /
            f"{benchmark}_history.csv",
            index=False,
        )

        pd.DataFrame(
            trials
        ).to_csv(
            output_dir
            /
            f"{benchmark}_performance_trials.csv",
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

        posterior_df.to_csv(
            output_dir
            /
            (
                f"{benchmark}_"
                f"round_{round_id}_posterior.csv"
            ),
            index=False,
        )

    # ========================================================
    # Final graph
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        f"FINAL GRAPH: "
        f"{benchmark}"
    )

    print(
        graph
    )

    print(
        f"Final edges = "
        f"{count_edges(graph)}"
    )

    print(
        f"Final REAL ESS = "
        f"{current_metrics['ess_mean']:.6f}"
    )

    print(
        f"Final REAL KL  = "
        f"{current_metrics['kl_mean']:.6f}"
    )

    print(
        "=" * 100
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

        "final_graph":
            graph_as_string(
                graph
            ),

        "n_rounds":
            len(
                history
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
    final_edges = []

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

        final_edges.extend(
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
            final_edges
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
        "RETRAINED PERFORMANCE-GUARDED "
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "single_gaussian",
            "curved_ridge",
            "sharp_broad",
            "star_sparse",
            "block_sparse",
            "dense_ar",
            "condition_switch",
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

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if args.smoke:

        args.seeds = [
            0,
        ]

        args.marginal_steps = 100
        args.masked_steps = 300
        args.refine_steps = 100

        args.copula_steps = 300

        args.batch_size = 1024

        args.n_eval = 10_000

        args.perf_n_eval = 10_000
        args.perf_n_kl = 5_000

        args.max_rounds = 3

        args.mcmc_iter = 4_000
        args.mcmc_burn = 1_000
        args.mcmc_thin = 2

        args.output = (
            "results/"
            "retrained_performance_guarded_"
            "pruning_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )