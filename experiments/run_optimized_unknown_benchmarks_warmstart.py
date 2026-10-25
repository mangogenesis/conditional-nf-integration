"""
run_optimized_unknown_benchmarks_warmstart.py

Extension of:

    run_optimized_unknown_benchmarks.py

The previous script is intentionally left untouched.

Changes
-------
1. Copula initial LR remains 3e-4.
2. Maximum copula steps remain 30,000.
3. Improved convergence detection:
       - 2-point KL moving averages
       - compare previous window vs current window
       - require |relative change| < 1%
       - require 3 consecutive stable checks
4. ReduceLROnPlateau learning-rate decay.
5. Candidate models warm-start from the CURRENT ACCEPTED graph.
6. Every candidate within the same round starts from the SAME
   accepted model -- never from another candidate.
7. Parameters are copied when state-dict key AND tensor shape match.
   Therefore:
       - unchanged child networks transfer exactly
       - compatible deeper layers of changed children can also transfer
       - incompatible layers are initialized normally
8. ESS remains the ONLY graph-acceptance gate.

Usage
-----

python experiments/run_optimized_unknown_benchmarks_warmstart.py \
    --benchmarks regime_switch

or

python experiments/run_optimized_unknown_benchmarks_warmstart.py \
    --benchmarks regime_switch cross_coupled
"""

import copy

import numpy as np
import torch

# ============================================================
# Import the previous working experiment.
#
# DO NOT MODIFY THAT FILE.
# ============================================================

import run_optimized_unknown_benchmarks as exp


# ============================================================
# New optimization defaults
# ============================================================

# Initial LR.
#
# I would keep this at 3e-4 rather than increasing again.
exp.COPULA_LR = 1e-3

# Keep expanded training budget.
exp.MAX_CANDIDATE_STEPS = 30_000

# Same 1% threshold, but applied to smoothed KL.
exp.REL_KL_TOL = 0.01

# More conservative than the previous 2-check rule.
exp.CONVERGENCE_PATIENCE = 3


# ============================================================
# New convergence configuration
# ============================================================

KL_SMOOTH_WINDOW = 2

# Learning-rate scheduler.
LR_DECAY_FACTOR = 1.0 / 3.0

# Three KL checks without sufficient improvement before lowering LR.
LR_PLATEAU_PATIENCE = 3

# Scheduler considers <0.5% improvement a plateau.
LR_PLATEAU_THRESHOLD = 0.005

# Do not let LR become arbitrarily small.
MIN_COPULA_LR = 3e-5


# ============================================================
# Warm-start utilities
# ============================================================

def warm_start_matching_parameters(
    target_model,
    source_state_dict,
):
    """
    Warm-start target_model from source_state_dict.

    A tensor is copied iff:

        1. the state-dict key exists in both models
        2. its shape is identical

    This is intentionally architecture-safe.

    If an edge removal changes the input dimension of one child
    network, the incompatible tensor is NOT copied.

    However, all other compatible tensors are preserved.

    Returns
    -------
    dictionary containing transfer statistics.
    """

    target_state = (
        target_model.state_dict()
    )

    merged_state = {}

    copied_tensors = 0
    copied_parameters = 0

    skipped_tensors = 0
    skipped_parameters = 0

    copied_names = []
    skipped_names = []

    for key, target_tensor in (
        target_state.items()
    ):

        if key in source_state_dict:

            source_tensor = (
                source_state_dict[
                    key
                ]
            )

            if (
                tuple(
                    source_tensor.shape
                )
                ==
                tuple(
                    target_tensor.shape
                )
            ):

                merged_state[
                    key
                ] = (
                    source_tensor
                    .detach()
                    .to(
                        device=
                            target_tensor.device,

                        dtype=
                            target_tensor.dtype,
                    )
                    .clone()
                )

                copied_tensors += 1

                copied_parameters += (
                    target_tensor.numel()
                )

                copied_names.append(
                    key
                )

                continue

        # ----------------------------------------------------
        # Shape mismatch or missing parameter:
        # retain new model initialization.
        # ----------------------------------------------------

        merged_state[
            key
        ] = target_tensor

        skipped_tensors += 1

        skipped_parameters += (
            target_tensor.numel()
        )

        skipped_names.append(
            key
        )

    target_model.load_state_dict(
        merged_state
    )

    total_parameters = (
        copied_parameters
        +
        skipped_parameters
    )

    copied_fraction = (
        copied_parameters
        /
        max(
            total_parameters,
            1,
        )
    )

    return {

        "copied_tensors":
            copied_tensors,

        "skipped_tensors":
            skipped_tensors,

        "copied_parameters":
            copied_parameters,

        "skipped_parameters":
            skipped_parameters,

        "copied_fraction":
            copied_fraction,

        "copied_names":
            copied_names,

        "skipped_names":
            skipped_names,
    }


# ============================================================
# Improved copula convergence
# ============================================================

def train_copula_until_converged(
    model,
    benchmark,
    kl_cache,
    args,
    device,
):
    """
    Train copula using forward KL.

    Improvements over previous version
    ----------------------------------
    1. Adam starts at args.copula_lr.
    2. ReduceLROnPlateau automatically lowers LR.
    3. KL convergence uses moving-average windows.
    4. Large negative KL changes are NOT considered convergence.

    Let

        K_1, ..., K_t

    denote fixed-validation KL measurements.

    With window w=2:

        old = mean(K_{t-3}, K_{t-2})
        new = mean(K_{t-1}, K_t)

    and

        r = (old - new) / |old|.

    Convergence requires

        |r| < rel_kl_tol

    for convergence_patience consecutive checks.
    """

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=args.copula_lr,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,

            mode="min",

            factor=
                LR_DECAY_FACTOR,

            patience=
                LR_PLATEAU_PATIENCE,

            threshold=
                LR_PLATEAU_THRESHOLD,

            threshold_mode=
                "rel",

            min_lr=
                MIN_COPULA_LR,
        )
    )

    model.train()

    history = []

    kl_values = []

    total_steps = 0

    convergence_count = 0

    converged = False

    previous_lr = (
        optimizer.param_groups[
            0
        ][
            "lr"
        ]
    )

    # ========================================================
    # Main training loop
    # ========================================================

    while (
        total_steps
        <
        args.max_candidate_steps
    ):

        block_steps = min(

            args.kl_check_interval,

            args.max_candidate_steps
            -
            total_steps,
        )

        recent_losses = []

        # ====================================================
        # Train one block
        # ====================================================

        for _ in range(
            block_steps
        ):

            cond = (
                exp.base.sample_conditions(
                    args.batch_size,
                    device,
                )
            )

            # ------------------------------------------------
            # Forward-KL training samples come from target p.
            # ------------------------------------------------

            with torch.no_grad():

                x = (
                    exp.sample_target(
                        benchmark,
                        cond,
                        exp.DIM,
                    )
                )

            logq = (
                model.log_prob(
                    x,
                    cond,
                )
            )

            loss = (
                -logq.mean()
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            # ------------------------------------------------
            # Preserve successful gradient clipping.
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.copula.parameters(),
                10.0,
            )

            optimizer.step()

            recent_losses.append(
                float(
                    loss.item()
                )
            )

        total_steps += (
            block_steps
        )

        # ====================================================
        # Fixed-validation KL
        # ====================================================

        model.eval()

        kl = float(
            exp.base.cached_forward_kl(
                model,
                kl_cache,
            )
        )

        kl_values.append(
            kl
        )

        mean_loss = float(
            np.mean(
                recent_losses[
                    -100:
                ]
            )
        )

        # ====================================================
        # Smoothed KL comparison
        # ====================================================

        old_kl_mean = float(
            "nan"
        )

        new_kl_mean = float(
            "nan"
        )

        relative_improvement = float(
            "nan"
        )

        absolute_improvement = float(
            "nan"
        )

        window = (
            KL_SMOOTH_WINDOW
        )

        if (
            len(
                kl_values
            )
            >=
            2
            *
            window
        ):

            old_window = (
                kl_values[
                    -2 * window:
                    -window
                ]
            )

            new_window = (
                kl_values[
                    -window:
                ]
            )

            old_kl_mean = float(
                np.mean(
                    old_window
                )
            )

            new_kl_mean = float(
                np.mean(
                    new_window
                )
            )

            absolute_improvement = (
                old_kl_mean
                -
                new_kl_mean
            )

            relative_improvement = (
                absolute_improvement
                /
                max(
                    abs(
                        old_kl_mean
                    ),
                    1e-8,
                )
            )

        # ====================================================
        # LR scheduling
        #
        # Use smoothed KL once available.
        # Otherwise use raw KL.
        # ====================================================

        scheduler_metric = (
            new_kl_mean

            if np.isfinite(
                new_kl_mean
            )

            else kl
        )

        scheduler.step(
            scheduler_metric
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        lr_changed = (
            current_lr
            <
            previous_lr
            -
            1e-15
        )

        # ====================================================
        # History
        # ====================================================

        history.append(
            {

                "step":
                    total_steps,

                "kl":
                    kl,

                "nll":
                    mean_loss,

                "smoothed_old_kl":
                    old_kl_mean,

                "smoothed_new_kl":
                    new_kl_mean,

                "absolute_improvement":
                    absolute_improvement,

                # Keep this key because the existing
                # summarize_convergence() expects it.
                "relative_improvement":
                    relative_improvement,

                "learning_rate":
                    current_lr,

                "convergence_count":
                    convergence_count,
            }
        )

        # ====================================================
        # Console logging
        # ====================================================

        if np.isfinite(
            relative_improvement
        ):

            print(
                f"    step "
                f"{total_steps:6d} "
                f"NLL="
                f"{mean_loss:.6f} "
                f"KL="
                f"{kl:.6f} "
                f"smooth="
                f"{new_kl_mean:.6f} "
                f"rel="
                f"{100.0 * relative_improvement:+.3f}% "
                f"lr="
                f"{current_lr:.1e}"
            )

        else:

            print(
                f"    step "
                f"{total_steps:6d} "
                f"NLL="
                f"{mean_loss:.6f} "
                f"KL="
                f"{kl:.6f} "
                f"lr="
                f"{current_lr:.1e}"
            )

        if lr_changed:

            print(
                "      LR DECAY: "
                f"{previous_lr:.1e}"
                f" -> "
                f"{current_lr:.1e}"
            )

        previous_lr = (
            current_lr
        )

        # ====================================================
        # Improved convergence criterion
        #
        # IMPORTANT:
        #
        # Previous:
        #
        #     relative_improvement < 1%
        #
        # meant a -20% worsening counted toward convergence.
        #
        # New:
        #
        #     |relative improvement| < 1%
        #
        # genuinely means the smoothed KL is stable.
        # ====================================================

        if (
            total_steps
            >=
            args.min_candidate_steps
            and
            np.isfinite(
                relative_improvement
            )
        ):

            stable = (
                abs(
                    relative_improvement
                )
                <
                args.rel_kl_tol
            )

            if stable:

                convergence_count += 1

            else:

                convergence_count = 0

            print(
                f"      smoothed convergence "
                f"{convergence_count}/"
                f"{args.convergence_patience}"
            )

            if (
                convergence_count
                >=
                args.convergence_patience
            ):

                converged = True

                print(
                    "      COPULA CONVERGED "
                    f"at {total_steps} steps"
                )

                break

        model.train()

    # ========================================================
    # Max-step message
    # ========================================================

    if (
        not converged
        and
        total_steps
        >=
        args.max_candidate_steps
    ):

        print(
            "      copula reached max steps "
            f"{args.max_candidate_steps}"
        )

    # ========================================================
    # Final evaluation
    # ========================================================

    model.eval()

    final_kl = float(
        exp.base.cached_forward_kl(
            model,
            kl_cache,
        )
    )

    return {

        "history":
            history,

        "final_steps":
            total_steps,

        "final_kl":
            final_kl,

        "converged":
            converged,
    }


# ============================================================
# Candidate training WITH warm-start
# ============================================================

def train_or_reuse_candidate(
    benchmark,
    name,
    removals,
    current_graph,
    current_metrics,
    marginal_state,
    kl_cache,
    candidate_cache,
    args,
    device,
    round_id,
):
    """
    Train one candidate graph.

    Warm-start policy
    -----------------
    Every new candidate starts from the CURRENT ACCEPTED graph.

    It never starts from another candidate evaluated earlier
    in the same round.

    This avoids order dependence:

                        candidate A
                      /
        current model --- candidate B
                      \\
                        candidate C

    rather than:

        current -> A -> B -> C
    """

    candidate_graph = (
        exp.base.remove_edges(
            current_graph,
            removals,
        )
    )

    candidate_key = (
        exp.graph_key(
            candidate_graph
        )
    )

    # ========================================================
    # Exact candidate already trained:
    # restore it directly.
    # ========================================================

    if candidate_key in candidate_cache:

        print()
        print(
            f"[CACHE HIT] {name}"
        )

        cached = (
            candidate_cache[
                candidate_key
            ]
        )

        model = (
            exp.restore_cached_model(
                cached=
                    cached,

                graph=
                    candidate_graph,

                marginal_state=
                    marginal_state,

                device=
                    device,
            )
        )

        metrics = copy.deepcopy(
            cached[
                "metrics"
            ]
        )

        convergence = copy.deepcopy(
            cached[
                "convergence"
            ]
        )

        reused = True

        warm_started = False

        warm_fraction = 1.0

        copied_tensors = 0

        skipped_tensors = 0

    # ========================================================
    # New candidate
    # ========================================================

    else:

        print()
        print(
            "=" * 100
        )

        print(
            f"TRAIN CANDIDATE: "
            f"{name}"
        )

        print(
            f"round      = "
            f"{round_id}"
        )

        print(
            f"remove     = "
            f"{removals}"
        )

        print(
            f"edges      = "
            f"{exp.edge_count(candidate_graph)}"
        )

        print(
            f"copula lr  = "
            f"{args.copula_lr:.1e}"
        )

        print(
            "init       = "
            "warm-start current accepted graph"
        )

        print(
            "=" * 100
        )

        # ----------------------------------------------------
        # Deterministic initialization for tensors that cannot
        # be warm-started.
        # ----------------------------------------------------

        exp.base.seed_everything(
            args.performance_seed
        )

        model = (
            exp.base.make_real_model(
                candidate_graph,
                marginal_state,
                device,
            )
        )

        # ====================================================
        # WARM START
        #
        # Current graph should already exist in candidate_cache:
        #
        # initial full graph was cached immediately;
        # every subsequently accepted winner came from a
        # candidate that was also cached.
        # ====================================================

        current_key = (
            exp.graph_key(
                current_graph
            )
        )

        warm_started = False

        warm_fraction = 0.0

        copied_tensors = 0

        skipped_tensors = 0

        if current_key in candidate_cache:

            source_state = (
                candidate_cache[
                    current_key
                ][
                    "model_state"
                ]
            )

            transfer = (
                warm_start_matching_parameters(
                    target_model=
                        model,

                    source_state_dict=
                        source_state,
                )
            )

            warm_started = True

            warm_fraction = (
                transfer[
                    "copied_fraction"
                ]
            )

            copied_tensors = (
                transfer[
                    "copied_tensors"
                ]
            )

            skipped_tensors = (
                transfer[
                    "skipped_tensors"
                ]
            )

            print()
            print(
                "WARM START"
            )

            print(
                f"  copied tensors     = "
                f"{copied_tensors}"
            )

            print(
                f"  reinitialized      = "
                f"{skipped_tensors}"
            )

            print(
                f"  parameter fraction = "
                f"{100.0 * warm_fraction:.2f}%"
            )

        else:

            # This should rarely happen, but we deliberately
            # fall back safely rather than crash.
            print()
            print(
                "WARNING: current graph was not found "
                "in candidate cache."
            )

            print(
                "Candidate will use deterministic fresh "
                "initialization."
            )

        # ====================================================
        # Train candidate
        # ====================================================

        training_result = (
            train_copula_until_converged(
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

        convergence = (
            exp.base.summarize_convergence(
                training_result
            )
        )

        # ====================================================
        # ESS evaluation
        # ====================================================

        metrics = (
            exp.base.evaluate_ess(
                model=
                    model,

                benchmark=
                    benchmark,

                device=
                    device,

                args=
                    args,
            )
        )

        metrics[
            "kl"
        ] = (
            exp.base.cached_forward_kl(
                model,
                kl_cache,
            )
        )

        # ====================================================
        # Save exact trained candidate in graph cache
        # ====================================================

        candidate_cache[
            candidate_key
        ] = {

            "model_state":
                exp.cpu_state_dict(
                    model
                ),

            "metrics":
                copy.deepcopy(
                    metrics
                ),

            "convergence":
                copy.deepcopy(
                    convergence
                ),

            "warm_started":
                warm_started,

            "warm_fraction":
                warm_fraction,
        }

        reused = False

    # ========================================================
    # Candidate performance relative to current accepted graph
    # ========================================================

    delta_ess = (
        metrics[
            "ess"
        ]
        -
        current_metrics[
            "ess"
        ]
    )

    delta_kl = (
        metrics[
            "kl"
        ]
        -
        current_metrics[
            "kl"
        ]
    )

    # ========================================================
    # ESS remains the ONLY acceptance gate.
    # ========================================================

    accept = (
        delta_ess
        >=
        -args.ess_tol
    )

    result = {

        "name":
            name,

        "removals":
            list(
                removals
            ),

        "graph":
            candidate_graph,

        "model":
            model,

        "ess":
            metrics[
                "ess"
            ],

        "kl":
            metrics[
                "kl"
            ],

        "variance":
            metrics[
                "variance"
            ],

        "relative_error":
            metrics[
                "relative_error"
            ],

        "delta_ess":
            delta_ess,

        "delta_kl":
            delta_kl,

        "accept":
            accept,

        "reused":
            reused,

        "warm_started":
            warm_started,

        "warm_fraction":
            warm_fraction,

        "copied_tensors":
            copied_tensors,

        "skipped_tensors":
            skipped_tensors,

        **convergence,
    }

    print()
    print(
        f"RESULT: {name}"
    )

    print(
        f"  current ESS   = "
        f"{current_metrics['ess']:.6f}"
    )

    print(
        f"  candidate ESS = "
        f"{metrics['ess']:.6f}"
    )

    print(
        f"  delta ESS     = "
        f"{delta_ess:+.6f}"
    )

    print(
        f"  candidate KL  = "
        f"{metrics['kl']:.6f}"
    )

    print(
        f"  warm start    = "
        f"{warm_started}"
    )

    if warm_started:

        print(
            f"  transferred   = "
            f"{100.0 * warm_fraction:.2f}%"
        )

    print(
        f"  ESS pass      = "
        f"{accept}"
    )

    return result


# ============================================================
# Patch ONLY this process.
#
# The imported .py file itself remains unchanged on disk.
#
# Functions in exp.optimized_candidate_search() perform global
# lookup of exp.train_or_reuse_candidate at runtime, so replacing
# this symbol is sufficient for the existing decision-tree search
# to use warm-starting.
# ============================================================

exp.train_copula_until_converged = (
    train_copula_until_converged
)

exp.train_or_reuse_candidate = (
    train_or_reuse_candidate
)


# ============================================================
# Entry point
# ============================================================

def main():

    # --------------------------------------------------------
    # Because exp.parse_args() reads these globals at runtime,
    # setting them above changes the DEFAULTS for this wrapper
    # without modifying the previous script.
    # --------------------------------------------------------

    args = (
        exp.parse_args()
    )

    print()
    print(
        "=" * 100
    )

    print(
        "WARM-START OPTIMIZED VERSION"
    )

    print(
        "=" * 100
    )

    print(
        f"Initial copula LR       = "
        f"{args.copula_lr:.1e}"
    )

    print(
        f"Minimum copula LR       = "
        f"{MIN_COPULA_LR:.1e}"
    )

    print(
        f"Maximum steps           = "
        f"{args.max_candidate_steps:,}"
    )

    print(
        f"KL smoothing window     = "
        f"{KL_SMOOTH_WINDOW}"
    )

    print(
        f"KL stability tolerance  = "
        f"{100 * args.rel_kl_tol:.2f}%"
    )

    print(
        f"Convergence patience    = "
        f"{args.convergence_patience}"
    )

    print(
        f"LR plateau patience     = "
        f"{LR_PLATEAU_PATIENCE}"
    )

    print(
        f"LR decay factor         = "
        f"{LR_DECAY_FACTOR:.3f}"
    )

    print(
        "Candidate initialization = "
        "current accepted model"
    )

    print(
        "Graph acceptance         = "
        "ESS only"
    )

    print(
        "=" * 100
    )

    exp.run(
        args
    )


if __name__ == "__main__":

    main()