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
    DEFAULT_BENCHMARKS,
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

    MarginalBank,
    IterativeMaskedChild,

    train_marginals,
    sample_conditions,
    sample_training_mask,

    compute_child_deltas,

    bayesian_round,
    choose_removals,
    apply_removals,

    full_ar_graph,
    copy_graph,
    count_edges,
    graph_as_string,

    GRAD_CLIP,
    LR_MASKED,
)


# ============================================================
# New accelerated settings
# ============================================================

REFINE_STEPS = 400

DEFAULT_OUTPUT = (
    "results/"
    "iterative_bayesian_edge_pruning_fast"
)


# ============================================================
# CPU-safe state dictionary
# ============================================================

def clone_state_dict_to_cpu(
    model,
):
    return {
        key: value.detach().cpu().clone()
        for key, value
        in model.state_dict().items()
    }


# ============================================================
# Warm-started masked-child training
# ============================================================

def train_masked_child_fast(
    marginal_bank,
    benchmark,
    child,
    active_parents,
    device,
    steps,
    batch_size,
    keep_prob,
    seed,
    previous_state=None,
):
    """
    If previous_state is supplied, initialize the child model
    from the previous pruning round and fine-tune.

    Architecture is unchanged because IterativeMaskedChild(child)
    always has slots for all previous coordinates.
    """

    seed_everything(
        seed
    )

    model = (
        IterativeMaskedChild(
            child
        )
        .to(device)
    )

    # --------------------------------------------------------
    # Warm start
    # --------------------------------------------------------

    if previous_state is not None:

        model.load_state_dict(
            previous_state
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_MASKED,
    )

    model.train()

    recent = []

    for step in range(
        steps
    ):

        cond = sample_conditions(
            batch_size,
            device,
        )

        with torch.no_grad():

            from run_copula_bin_sweep import sample_target

            x = sample_target(
                benchmark,
                cond,
                DIM,
            )

            u = (
                marginal_bank
                .x_to_u(
                    x,
                    cond,
                )
            )

        child_u = (
            u[
                :,
                child:child + 1
            ]
        )

        mask = sample_training_mask(
            batch_size=
                batch_size,

            child=
                child,

            active_parents=
                active_parents,

            keep_prob=
                keep_prob,

            device=
                device,
        )

        logq = (
            model.log_prob(
                child_u,
                u,
                cond,
                mask,
            )
        )

        loss = (
            -logq.mean()
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        recent.append(
            loss.item()
        )

    return (
        model,
        float(
            np.mean(
                recent[-100:]
            )
        ),
    )


# ============================================================
# Train marginal banks once
# ============================================================

def train_all_marginal_states(
    benchmark,
    args,
    device,
):

    states = {}

    for seed in args.seeds:

        print()
        print(
            f"Training marginals for seed {seed}..."
        )

        seed_everything(
            seed
        )

        bank = (
            MarginalBank(
                DIM
            )
            .to(device)
        )

        train_marginals(
            bank,
            benchmark,
            device,
            args.marginal_steps,
            args.batch_size,
        )

        states[
            seed
        ] = {
            key:
                value.detach()
                .cpu()
                .clone()

            for key, value
            in bank.state_dict().items()
        }

        del bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return states


# ============================================================
# Evidence collection with caching
# ============================================================

def collect_round_evidence_fast(
    benchmark,
    round_id,
    graph,
    previous_graph,
    marginal_states,
    model_cache,
    evidence_cache,
    args,
    device,
):
    """
    model_cache[(seed, child)]
        -> CPU state_dict

    evidence_cache[(seed, child)]
        -> list[dict] containing the most recently computed
           evidence for that child.

    For unchanged children:
        reuse cached evidence completely.

    For changed children:
        warm start from previous model and fine tune.
    """

    rows = []

    trained_children = 0
    reused_children = 0

    for seed in args.seeds:

        # ====================================================
        # Restore marginal bank
        # ====================================================

        bank = (
            MarginalBank(
                DIM
            )
            .to(device)
        )

        bank.load_state_dict(
            marginal_states[
                seed
            ]
        )

        bank.eval()

        for parameter in (
            bank.parameters()
        ):

            parameter.requires_grad = False

        print()
        print(
            f"  Evidence seed {seed}"
        )

        # ====================================================
        # Children
        # ====================================================

        for child in range(
            1,
            DIM,
        ):

            active_parents = list(
                graph[
                    child
                ]
            )

            if len(
                active_parents
            ) == 0:

                continue

            key = (
                seed,
                child,
            )

            # ------------------------------------------------
            # Did this child's graph change?
            # ------------------------------------------------

            if previous_graph is None:

                changed = True

            else:

                old_parents = list(
                    previous_graph[
                        child
                    ]
                )

                changed = (
                    old_parents
                    != active_parents
                )

            # ------------------------------------------------
            # Reuse evidence if graph did not change
            # ------------------------------------------------

            if (
                not changed
                and
                key in evidence_cache
                and
                not args.recompute_unchanged
            ):

                print(
                    f"    child {child}: "
                    f"{active_parents} "
                    "[REUSE]"
                )

                cached_rows = (
                    evidence_cache[
                        key
                    ]
                )

                for cached_row in (
                    cached_rows
                ):

                    row = dict(
                        cached_row
                    )

                    row[
                        "benchmark"
                    ] = benchmark

                    row[
                        "round"
                    ] = round_id

                    row[
                        "seed"
                    ] = seed

                    row[
                        "source"
                    ] = "reused"

                    rows.append(
                        row
                    )

                reused_children += 1

                continue

            # ------------------------------------------------
            # Train / fine tune
            # ------------------------------------------------

            previous_state = (
                model_cache.get(
                    key,
                    None,
                )
            )

            if previous_state is None:

                steps = (
                    args.masked_steps
                )

                mode = (
                    "FULL TRAIN"
                )

            else:

                steps = (
                    args.refine_steps
                )

                mode = (
                    "WARM REFINE"
                )

            print(
                f"    child {child}: "
                f"{active_parents} "
                f"[{mode}, {steps} steps]"
            )

            model_seed = (
                seed * 100000
                + round_id * 1000
                + child
            )

            model, final_loss = (
                train_masked_child_fast(
                    marginal_bank=
                        bank,

                    benchmark=
                        benchmark,

                    child=
                        child,

                    active_parents=
                        active_parents,

                    device=
                        device,

                    steps=
                        steps,

                    batch_size=
                        args.batch_size,

                    keep_prob=
                        args.keep_prob,

                    seed=
                        model_seed,

                    previous_state=
                        previous_state,
                )
            )

            print(
                f"      final NLL ~ "
                f"{final_loss:.6f}"
            )

            model.eval()

            # ------------------------------------------------
            # Compute new Delta
            # ------------------------------------------------

            child_rows = (
                compute_child_deltas(
                    model=
                        model,

                    marginal_bank=
                        bank,

                    benchmark=
                        benchmark,

                    child=
                        child,

                    active_parents=
                        active_parents,

                    device=
                        device,

                    n_eval=
                        args.n_eval,
                )
            )

            # ------------------------------------------------
            # Cache model
            # ------------------------------------------------

            model_cache[
                key
            ] = (
                clone_state_dict_to_cpu(
                    model
                )
            )

            # ------------------------------------------------
            # Cache evidence
            # ------------------------------------------------

            evidence_cache[
                key
            ] = [
                dict(row)
                for row
                in child_rows
            ]

            for child_row in (
                child_rows
            ):

                row = dict(
                    child_row
                )

                row[
                    "benchmark"
                ] = benchmark

                row[
                    "round"
                ] = round_id

                row[
                    "seed"
                ] = seed

                row[
                    "source"
                ] = (
                    "full_train"
                    if previous_state is None
                    else
                    "warm_refine"
                )

                rows.append(
                    row
                )

            trained_children += 1

            del model

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        del bank

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    print()
    print(
        "  Round compute summary:"
    )

    print(
        f"    trained/refined children : "
        f"{trained_children}"
    )

    print(
        f"    reused children          : "
        f"{reused_children}"
    )

    return pd.DataFrame(
        rows
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
        "=" * 100
    )

    print(
        "FAST ITERATIVE BAYESIAN PRUNING: "
        f"{benchmark}"
    )

    print(
        "=" * 100
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

    # ========================================================
    # Start full graph
    # ========================================================

    graph = (
        full_ar_graph(
            DIM
        )
    )

    previous_graph = None

    # ========================================================
    # Caches
    # ========================================================

    model_cache = {}

    evidence_cache = {}

    # ========================================================
    # Histories
    # ========================================================

    evidence_history = []

    posterior_history = []

    graph_history = []

    # ========================================================
    # Frozen Bayesian null
    # ========================================================

    frozen_sigma0 = None

    # ========================================================
    # Rounds
    # ========================================================

    for round_id in range(
        args.max_rounds
    ):

        print()
        print(
            "#" * 100
        )

        print(
            f"ROUND {round_id} "
            f"| active edges="
            f"{count_edges(graph)}"
        )

        print(
            graph
        )

        if frozen_sigma0 is None:

            print(
                "Null scale: infer this round"
            )

        else:

            print(
                f"Frozen sigma0: "
                f"{frozen_sigma0:.8f}"
            )

        print(
            "#" * 100
        )

        graph_before = (
            copy_graph(
                graph
            )
        )

        # ====================================================
        # Evidence
        # ====================================================

        evidence_df = (
            collect_round_evidence_fast(
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

        evidence_history.append(
            evidence_df
        )

        # ====================================================
        # Bayesian inference
        # ====================================================

        posterior_df, posterior = (
            bayesian_round(
                evidence_df,
                args,
                fixed_sigma0=
                    frozen_sigma0,
            )
        )

        # ====================================================
        # Freeze round-0 null
        # ====================================================

        if round_id == 0:

            frozen_sigma0 = float(
                posterior[
                    "sigma0_mean"
                ]
            )

            print()
            print(
                "=" * 70
            )

            print(
                "FREEZING ROUND-0 NULL"
            )

            print(
                f"sigma0 = "
                f"{frozen_sigma0:.8f}"
            )

            print(
                "=" * 70
            )

        posterior_df[
            "benchmark"
        ] = benchmark

        posterior_df[
            "round"
        ] = round_id

        posterior_df[
            "sigma0_reference"
        ] = frozen_sigma0

        posterior_history.append(
            posterior_df
        )

        # ====================================================
        # Pruning
        # ====================================================

        removals = (
            choose_removals(
                posterior_df,
                args.pip_threshold,
            )
        )

        print()
        print(
            "Bayesian round summary:"
        )

        print(
            f"  rho mean      = "
            f"{posterior['rho_mean']:.4f}"
        )

        print(
            f"  E[edges]      = "
            f"{posterior['expected_edges']:.2f}"
        )

        print(
            f"  current edges = "
            f"{count_edges(graph)}"
        )

        print(
            f"  sigma0        = "
            f"{posterior['sigma0_mean']:.8f}"
        )

        print(
            f"  removals      = "
            f"{removals}"
        )

        print()

        print(
            posterior_df
            .sort_values(
                [
                    "child",
                    "pip",
                ]
            )
            [
                [
                    "parent",
                    "child",
                    "delta_mean",
                    "pip",
                    "is_markov1_edge",
                ]
            ]
            .to_string(
                index=False
            )
        )

        # ====================================================
        # History
        # ====================================================

        graph_history.append(
            {
                "benchmark":
                    benchmark,

                "round":
                    round_id,

                "edges_before":
                    count_edges(
                        graph_before
                    ),

                "posterior_expected_edges":
                    posterior[
                        "expected_edges"
                    ],

                "rho_mean":
                    posterior[
                        "rho_mean"
                    ],

                "frozen_sigma0":
                    frozen_sigma0,

                "n_removed":
                    len(
                        removals
                    ),

                "removed_edges":
                    str(
                        removals
                    ),

                "graph_before":
                    graph_as_string(
                        graph_before
                    ),
            }
        )

        # ====================================================
        # Save continuously
        # ====================================================

        pd.concat(
            evidence_history,
            ignore_index=True,
        ).to_csv(
            output_dir
            / (
                f"{benchmark}"
                "_round_evidence.csv"
            ),
            index=False,
        )

        pd.concat(
            posterior_history,
            ignore_index=True,
        ).to_csv(
            output_dir
            / (
                f"{benchmark}"
                "_round_posteriors.csv"
            ),
            index=False,
        )

        pd.DataFrame(
            graph_history
        ).to_csv(
            output_dir
            / (
                f"{benchmark}"
                "_graph_history.csv"
            ),
            index=False,
        )

        # ====================================================
        # Stop
        # ====================================================

        if len(
            removals
        ) == 0:

            print()
            print(
                "STOP: no further "
                "Bayesian pruning."
            )

            break

        # ====================================================
        # Advance graph
        #
        # Save current graph so next round can determine
        # exactly which children changed.
        # ====================================================

        previous_graph = (
            copy_graph(
                graph
            )
        )

        graph = (
            apply_removals(
                graph,
                removals,
            )
        )

    # ========================================================
    # Final graph
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        f"FINAL GRAPH: {benchmark}"
    )

    print(
        "=" * 90
    )

    print(
        graph
    )

    print(
        f"Final edge count: "
        f"{count_edges(graph)}"
    )

    final_rows = []

    for child in range(
        DIM
    ):

        for parent in (
            graph[
                child
            ]
        ):

            final_rows.append(
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

    return (
        graph,
        final_rows,
        graph_history,
        frozen_sigma0,
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

    all_final_edges = []

    all_summaries = []

    for benchmark in (
        args.benchmarks
    ):

        (
            final_graph,
            final_rows,
            history,
            frozen_sigma0,
        ) = (
            run_benchmark(
                benchmark,
                args,
                device,
                output_dir,
            )
        )

        all_final_edges.extend(
            final_rows
        )

        all_summaries.append(
            {
                "benchmark":
                    benchmark,

                "final_edge_count":
                    count_edges(
                        final_graph
                    ),

                "frozen_sigma0":
                    frozen_sigma0,

                "final_graph":
                    graph_as_string(
                        final_graph
                    ),

                "n_rounds":
                    len(
                        history
                    ),
            }
        )

        pd.DataFrame(
            all_final_edges
        ).to_csv(
            output_dir
            / "final_graph_edges.csv",
            index=False,
        )

        pd.DataFrame(
            all_summaries
        ).to_csv(
            output_dir
            / "summary.csv",
            index=False,
        )

    print()
    print(
        "=" * 110
    )

    print(
        "FAST ITERATIVE BAYESIAN "
        "EDGE PRUNING COMPLETE"
    )

    print(
        "=" * 110
    )

    print()

    print(
        pd.DataFrame(
            all_summaries
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
        default=
            DEFAULT_BENCHMARKS,
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
        "--keep-prob",
        type=float,
        default=
            KEEP_PROB,
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--pip-threshold",
        type=float,
        default=
            PIP_REMOVE_THRESHOLD,
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

    # --------------------------------------------------------
    # Debug/validation option.
    #
    # If enabled, even unchanged children recompute evidence.
    # Training is still skipped.
    # --------------------------------------------------------

    parser.add_argument(
        "--recompute-unchanged",
        action="store_true",
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
        args.masked_steps = 300
        args.refine_steps = 100

        args.batch_size = 1024
        args.n_eval = 10_000

        args.max_rounds = 3

        args.mcmc_iter = 4_000
        args.mcmc_burn = 1_000
        args.mcmc_thin = 2

        args.output = (
            "results/"
            "iterative_bayesian_"
            "edge_pruning_fast_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )