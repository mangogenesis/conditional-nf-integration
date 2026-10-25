"""
run_optimized_unknown_benchmarks_adaptive_warmstart.py

This script extends:

    run_optimized_unknown_benchmarks.py

without modifying it.

Main changes
------------
1. Adam stays the optimizer.
2. Initial copula LR = 1e-2.
3. Adaptive LR ladder:

       1e-2
       3e-3
       1e-3
       3e-4
       1e-4
       3e-5
       1e-5

4. LR decays based on smoothed fixed-validation KL.
5. Emergency LR decay if validation KL deteriorates strongly.
6. Best-KL model checkpoint is kept during every training run.
7. Candidate models warm-start from the CURRENT ACCEPTED model.
8. Every candidate in a round starts from the same accepted model.
9. ESS remains the ONLY graph-acceptance criterion.
10. Marginal training is unchanged.

Recommended usage
-----------------

python experiments/run_optimized_unknown_benchmarks_adaptive_warmstart.py \
    --benchmarks regime_switch

Then:

python experiments/run_optimized_unknown_benchmarks_adaptive_warmstart.py \
    --benchmarks regime_switch cross_coupled
"""

import copy

import numpy as np
import torch

# ============================================================
# Previous experiment implementation.
#
# DO NOT MODIFY IT.
# ============================================================

import run_optimized_unknown_benchmarks as exp


# ============================================================
# Optimization defaults
# ============================================================

# Initial learning rate requested.
INITIAL_COPULA_LR = 1e-2

# Explicit ladder gives us predictable decay.
COPULA_LR_LEVELS = [
    1e-2,
    3e-3,
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
]

MIN_COPULA_LR = 1e-5

# Keep large maximum budget.
MAX_CANDIDATE_STEPS = 30_000

# ============================================================
# KL monitoring
# ============================================================

KL_SMOOTH_WINDOW = 2

# If smoothed KL improves by less than 3% repeatedly,
# move down one LR level.
LR_DECAY_REL_TOL = 0.03

LR_DECAY_PATIENCE = 2

# Final convergence is stricter.
CONVERGENCE_REL_TOL = 0.01

CONVERGENCE_PATIENCE = 3

# Do not declare final convergence at a large LR.
MAX_LR_FOR_CONVERGENCE = 1e-4

# ============================================================
# Divergence protection
# ============================================================

# If current KL exceeds best KL by 25%, immediately decay LR.
DIVERGENCE_FACTOR = 1.25

# A catastrophic increase gets an optional rollback.
ROLLBACK_FACTOR = 1.50

# ============================================================
# Gradient clipping
# ============================================================

GRAD_CLIP = 10.0


# ============================================================
# Override defaults in imported experiment
# ============================================================

exp.COPULA_LR = INITIAL_COPULA_LR
exp.MAX_CANDIDATE_STEPS = MAX_CANDIDATE_STEPS
exp.REL_KL_TOL = CONVERGENCE_REL_TOL
exp.CONVERGENCE_PATIENCE = CONVERGENCE_PATIENCE


# ============================================================
# Utility
# ============================================================

def clone_state_dict_cpu(model):
    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in model.state_dict().items()
    }


def load_state_dict_device(
    model,
    state_dict,
):
    current = model.state_dict()

    converted = {}

    for key, value in state_dict.items():

        converted[key] = value.to(
            device=current[key].device,
            dtype=current[key].dtype,
        )

    model.load_state_dict(
        converted
    )


def set_optimizer_lr(
    optimizer,
    lr,
):
    for group in optimizer.param_groups:
        group["lr"] = lr


# ============================================================
# Warm start
# ============================================================

def warm_start_matching_parameters(
    target_model,
    source_state_dict,
):
    """
    Transfer all tensors whose name AND shape match.

    This is safe when pruning edges changes the architecture.

    Unchanged children should generally transfer completely.

    For changed children:
      - tensors whose shape does not change are retained
      - incompatible input layers remain freshly initialized
    """

    target_state = (
        target_model.state_dict()
    )

    merged = {}

    copied_tensors = 0
    skipped_tensors = 0

    copied_parameters = 0
    skipped_parameters = 0

    copied_keys = []
    skipped_keys = []

    for key, target_tensor in (
        target_state.items()
    ):

        can_copy = (
            key in source_state_dict
            and
            tuple(
                source_state_dict[
                    key
                ].shape
            )
            ==
            tuple(
                target_tensor.shape
            )
        )

        if can_copy:

            source_tensor = (
                source_state_dict[
                    key
                ]
            )

            merged[
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

            copied_keys.append(
                key
            )

        else:

            merged[
                key
            ] = (
                target_tensor
                .detach()
                .clone()
            )

            skipped_tensors += 1

            skipped_parameters += (
                target_tensor.numel()
            )

            skipped_keys.append(
                key
            )

    target_model.load_state_dict(
        merged
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

        "copied_keys":
            copied_keys,

        "skipped_keys":
            skipped_keys,
    }


# ============================================================
# Adaptive Adam trainer
# ============================================================

def train_copula_until_converged(
    model,
    benchmark,
    kl_cache,
    args,
    device,
):
    """
    Adam optimization with:

      initial lr = 1e-2
      adaptive LR ladder down to 1e-5
      fixed-cache KL monitoring
      smoothed convergence
      divergence protection
      best-model restoration

    The training objective remains the same forward-KL
    cross-entropy objective used previously.
    """

    # ========================================================
    # Adam
    # ========================================================

    lr_levels = list(
        COPULA_LR_LEVELS
    )

    # If CLI supplied a different initial LR,
    # insert it as the first level.
    if not np.isclose(
        args.copula_lr,
        lr_levels[0],
    ):

        lr_levels = [
            args.copula_lr
        ] + [
            lr
            for lr
            in lr_levels
            if lr < args.copula_lr
        ]

    lr_index = 0

    current_lr = (
        lr_levels[
            lr_index
        ]
    )

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=current_lr,
    )

    model.train()

    # ========================================================
    # Histories
    # ========================================================

    history = []

    kl_values = []

    total_steps = 0

    plateau_count = 0

    convergence_count = 0

    converged = False

    # ========================================================
    # Best checkpoint
    # ========================================================

    best_kl = float(
        "inf"
    )

    best_step = 0

    best_lr = current_lr

    best_state = None

    # ========================================================
    # Evaluate warm/fresh initialization at step 0
    # ========================================================

    model.eval()

    initial_kl = float(
        exp.base.cached_forward_kl(
            model,
            kl_cache,
        )
    )

    best_kl = initial_kl

    best_state = (
        clone_state_dict_cpu(
            model
        )
    )

    kl_values.append(
        initial_kl
    )

    history.append(
        {
            "step":
                0,

            "kl":
                initial_kl,

            "nll":
                np.nan,

            "smoothed_old_kl":
                np.nan,

            "smoothed_new_kl":
                np.nan,

            "absolute_improvement":
                np.nan,

            "relative_improvement":
                np.nan,

            "learning_rate":
                current_lr,

            "lr_level":
                lr_index,

            "best_kl":
                best_kl,

            "best_step":
                best_step,

            "plateau_count":
                plateau_count,

            "convergence_count":
                convergence_count,
        }
    )

    print()
    print(
        f"    step {0:6d} "
        f"KL={initial_kl:.6f} "
        f"lr={current_lr:.1e}"
    )

    model.train()

    # ========================================================
    # Training loop
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
        # Training block
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

            # ------------------------------------------------
            # Fail safely on catastrophic loss.
            # ------------------------------------------------

            if not torch.isfinite(
                loss
            ):

                print()
                print(
                    "      NON-FINITE LOSS"
                )

                print(
                    "      restoring best model "
                    "and lowering LR"
                )

                if best_state is not None:

                    load_state_dict_device(
                        model,
                        best_state,
                    )

                if (
                    lr_index
                    <
                    len(
                        lr_levels
                    )
                    -
                    1
                ):

                    lr_index += 1

                    current_lr = (
                        lr_levels[
                            lr_index
                        ]
                    )

                    set_optimizer_lr(
                        optimizer,
                        current_lr,
                    )

                    print(
                        f"      LR -> "
                        f"{current_lr:.1e}"
                    )

                    continue

                else:

                    print(
                        "      already at minimum LR"
                    )

                    break

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.copula.parameters(),
                GRAD_CLIP,
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
        # Fixed-cache KL
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

        if recent_losses:

            mean_loss = float(
                np.mean(
                    recent_losses[
                        -100:
                    ]
                )
            )

        else:

            mean_loss = float(
                "nan"
            )

        # ====================================================
        # Best model update
        # ====================================================

        if kl < best_kl:

            best_kl = kl

            best_step = (
                total_steps
            )

            best_lr = (
                current_lr
            )

            best_state = (
                clone_state_dict_cpu(
                    model
                )
            )

        # ====================================================
        # Smoothed KL
        # ====================================================

        old_kl_mean = float(
            "nan"
        )

        new_kl_mean = float(
            "nan"
        )

        absolute_improvement = float(
            "nan"
        )

        relative_improvement = float(
            "nan"
        )

        w = KL_SMOOTH_WINDOW

        if (
            len(
                kl_values
            )
            >=
            2 * w
        ):

            old_window = (
                kl_values[
                    -2 * w:
                    -w
                ]
            )

            new_window = (
                kl_values[
                    -w:
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
        # Emergency deterioration detection
        # ====================================================

        emergency_decay = False

        rollback = False

        if (
            np.isfinite(
                best_kl
            )
            and
            best_kl > 0.0
        ):

            if (
                kl
                >
                DIVERGENCE_FACTOR
                *
                best_kl
            ):

                emergency_decay = True

            if (
                kl
                >
                ROLLBACK_FACTOR
                *
                best_kl
            ):

                rollback = True

        # ====================================================
        # Normal plateau detection
        # ====================================================

        if np.isfinite(
            relative_improvement
        ):

            # Positive relative improvement means KL decreased.
            #
            # Anything below 3% is considered insufficient
            # progress for the purpose of LR decay.
            if (
                relative_improvement
                <
                LR_DECAY_REL_TOL
            ):

                plateau_count += 1

            else:

                plateau_count = 0

        # ====================================================
        # LR decay
        # ====================================================

        should_decay = (
            emergency_decay
            or
            plateau_count
            >=
            LR_DECAY_PATIENCE
        )

        if (
            should_decay
            and
            lr_index
            <
            len(
                lr_levels
            )
            -
            1
        ):

            old_lr = (
                current_lr
            )

            lr_index += 1

            current_lr = (
                lr_levels[
                    lr_index
                ]
            )

            set_optimizer_lr(
                optimizer,
                current_lr,
            )

            plateau_count = 0

            print()
            print(
                f"      LR DECAY "
                f"{old_lr:.1e}"
                f" -> "
                f"{current_lr:.1e}"
            )

            if emergency_decay:

                print(
                    f"      reason: KL deterioration "
                    f"(current={kl:.6f}, "
                    f"best={best_kl:.6f})"
                )

        # ====================================================
        # Optional rollback only for severe deterioration
        # ====================================================

        if (
            rollback
            and
            best_state is not None
        ):

            print(
                "      severe deterioration: "
                "restoring best-KL checkpoint"
            )

            load_state_dict_device(
                model,
                best_state,
            )

            # Reset Adam moments because parameters jumped
            # backward to an earlier state.
            optimizer = torch.optim.Adam(
                model.copula.parameters(),
                lr=current_lr,
            )

            # Replace newest KL value with best value so
            # convergence logic does not interpret the
            # catastrophic excursion as useful history.
            kl_values[
                -1
            ] = (
                best_kl
            )

            kl = (
                best_kl
            )

            plateau_count = 0
            convergence_count = 0

        # ====================================================
        # Convergence
        #
        # Only at sufficiently small LR.
        #
        # Use absolute smoothed relative change:
        # worsening is not treated as convergence unless it is
        # genuinely tiny.
        # ====================================================

        can_converge = (
            current_lr
            <=
            MAX_LR_FOR_CONVERGENCE
        )

        if (
            total_steps
            >=
            args.min_candidate_steps
            and
            can_converge
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
                CONVERGENCE_REL_TOL
            )

            if stable:

                convergence_count += 1

            else:

                convergence_count = 0

        else:

            convergence_count = 0

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

                "relative_improvement":
                    relative_improvement,

                "learning_rate":
                    current_lr,

                "lr_level":
                    lr_index,

                "best_kl":
                    best_kl,

                "best_step":
                    best_step,

                "plateau_count":
                    plateau_count,

                "convergence_count":
                    convergence_count,
            }
        )

        # ====================================================
        # Logging
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
                f"best="
                f"{best_kl:.6f} "
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
                f"best="
                f"{best_kl:.6f} "
                f"lr="
                f"{current_lr:.1e}"
            )

        if can_converge:

            print(
                f"      smoothed convergence "
                f"{convergence_count}/"
                f"{CONVERGENCE_PATIENCE}"
            )

        # ====================================================
        # Stop
        # ====================================================

        if (
            convergence_count
            >=
            CONVERGENCE_PATIENCE
        ):

            converged = True

            print()
            print(
                f"      COPULA CONVERGED "
                f"at {total_steps} steps"
            )

            break

        model.train()

    # ========================================================
    # Restore BEST model, not simply final model.
    # ========================================================

    if best_state is not None:

        load_state_dict_device(
            model,
            best_state,
        )

    model.eval()

    final_kl = float(
        exp.base.cached_forward_kl(
            model,
            kl_cache,
        )
    )

    if (
        not converged
        and
        total_steps
        >=
        args.max_candidate_steps
    ):

        print()
        print(
            f"      reached max steps "
            f"{args.max_candidate_steps}"
        )

    print()
    print(
        "      OPTIMIZATION SUMMARY"
    )

    print(
        f"      total steps = "
        f"{total_steps}"
    )

    print(
        f"      best step   = "
        f"{best_step}"
    )

    print(
        f"      best LR     = "
        f"{best_lr:.1e}"
    )

    print(
        f"      final LR    = "
        f"{current_lr:.1e}"
    )

    print(
        f"      best KL     = "
        f"{best_kl:.6f}"
    )

    print(
        f"      restored KL = "
        f"{final_kl:.6f}"
    )

    return {
        "history":
            history,

        "final_steps":
            total_steps,

        "final_kl":
            final_kl,

        "best_kl":
            best_kl,

        "best_step":
            best_step,

        "best_lr":
            best_lr,

        "final_lr":
            current_lr,

        "converged":
            converged,
    }


# ============================================================
# Candidate training with warm start
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
    Every candidate starts from the SAME current accepted graph.

    Candidate A is never used to initialize candidate B.
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
    # Exact graph cache
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

        warm_started = bool(
            cached.get(
                "warm_started",
                False,
            )
        )

        warm_fraction = float(
            cached.get(
                "warm_fraction",
                1.0,
            )
        )

        copied_tensors = int(
            cached.get(
                "copied_tensors",
                0,
            )
        )

        skipped_tensors = int(
            cached.get(
                "skipped_tensors",
                0,
            )
        )

    # ========================================================
    # New candidate
    # ========================================================

    else:

        print()
        print(
            "=" * 100
        )

        print(
            f"TRAIN CANDIDATE: {name}"
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
            "optimizer  = Adam"
        )

        print(
            f"initial lr = "
            f"{args.copula_lr:.1e}"
        )

        print(
            "min lr     = 1.0e-05"
        )

        print(
            "init       = warm-start "
            "current accepted model"
        )

        print(
            "=" * 100
        )

        # ----------------------------------------------------
        # Deterministic fresh initialization for incompatible
        # parameters.
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
        # Warm start from CURRENT ACCEPTED graph
        # ========================================================

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

            warm_fraction = float(
                transfer[
                    "copied_fraction"
                ]
            )

            copied_tensors = int(
                transfer[
                    "copied_tensors"
                ]
            )

            skipped_tensors = int(
                transfer[
                    "skipped_tensors"
                ]
            )

            print()
            print(
                "WARM START"
            )

            print(
                f"  copied tensors = "
                f"{copied_tensors}"
            )

            print(
                f"  skipped tensors = "
                f"{skipped_tensors}"
            )

            print(
                f"  copied parameter fraction = "
                f"{100.0 * warm_fraction:.2f}%"
            )

        else:

            print()
            print(
                "WARNING: current accepted graph "
                "not found in cache."
            )

            print(
                "Using deterministic fresh initialization."
            )

        # ====================================================
        # Train
        # ========================================================

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
        # Final performance
        # ========================================================

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
        ] = float(
            exp.base.cached_forward_kl(
                model,
                kl_cache,
            )
        )

        # ====================================================
        # Cache exact trained graph
        # ========================================================

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

            "copied_tensors":
                copied_tensors,

            "skipped_tensors":
                skipped_tensors,
        }

        reused = False

    # ========================================================
    # ESS-only comparison
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
# Patch optimizer and candidate training in THIS process only.
#
# Original files remain unchanged.
# ============================================================

exp.train_copula_until_converged = (
    train_copula_until_converged
)

exp.train_or_reuse_candidate = (
    train_or_reuse_candidate
)


# ============================================================
# Main
# ============================================================

def main():

    args = (
        exp.parse_args()
    )

    # Force requested adaptive optimizer default.
    #
    # User can still explicitly override with:
    #
    #     --copula-lr ...
    #
    # if the underlying parser provides this flag.
    if not hasattr(
        args,
        "copula_lr",
    ):

        args.copula_lr = (
            INITIAL_COPULA_LR
        )

    # If parser default is inherited from previous file,
    # use 1e-2 unless the user explicitly changed it there.
    if (
        args.copula_lr
        is None
    ):

        args.copula_lr = (
            INITIAL_COPULA_LR
        )

    args.max_candidate_steps = (
        MAX_CANDIDATE_STEPS
    )

    args.rel_kl_tol = (
        CONVERGENCE_REL_TOL
    )

    args.convergence_patience = (
        CONVERGENCE_PATIENCE
    )

    print()
    print(
        "=" * 110
    )

    print(
        "ADAPTIVE ADAM + WARM START"
    )

    print(
        "=" * 110
    )

    print(
        "Optimizer       = Adam"
    )

    print(
        f"Initial LR      = "
        f"{args.copula_lr:.1e}"
    )

    print(
        "LR ladder       = "
        + " -> ".join(
            f"{lr:.0e}"
            for lr
            in COPULA_LR_LEVELS
        )
    )

    print(
        f"Minimum LR      = "
        f"{MIN_COPULA_LR:.1e}"
    )

    print(
        f"Max steps       = "
        f"{args.max_candidate_steps:,}"
    )

    print(
        f"KL window       = "
        f"{KL_SMOOTH_WINDOW}"
    )

    print(
        f"LR decay tol    = "
        f"{100.0 * LR_DECAY_REL_TOL:.1f}%"
    )

    print(
        f"LR patience     = "
        f"{LR_DECAY_PATIENCE}"
    )

    print(
        f"Convergence tol = "
        f"{100.0 * CONVERGENCE_REL_TOL:.1f}%"
    )

    print(
        f"Conv patience   = "
        f"{CONVERGENCE_PATIENCE}"
    )

    print(
        f"Converge only at LR <= "
        f"{MAX_LR_FOR_CONVERGENCE:.1e}"
    )

    print(
        "Warm start      = current accepted model"
    )

    print(
        "Final model     = best-KL checkpoint"
    )

    print(
        "Graph gate      = ESS only"
    )

    print(
        "=" * 110
    )

    exp.run(
        args
    )


if __name__ == "__main__":

    main()