import os
import json
import math
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from nf.conditional_autoregressive_cubic_2d import (
    ConditionalAutoregressiveCubicFlow2D,
)


# ============================================================
# Global configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 1234

OUT_DIR = "results/conditional_sharp_broad_2d_loss_sweep"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# Training configuration
# ============================================================

# Keep this relatively small for the objective diagnostic
N_COND = 64
N_X_PER_COND = 512

STEPS = 2000
LR = 5e-5

N_EVAL = 20000
N_GRID = 100


# ============================================================
# Flow
# ============================================================

N_BINS = 16
HIDDEN = 128


# ============================================================
# Target
# ============================================================

FLOOR = 1e-6

A_SHARP = 1.0

MU_RANGE = (0.20, 0.80)
SHARP_SIGMA_RANGE = (0.04, 0.08)

A_BROAD = 0.20
BROAD_MU_X = 0.50
BROAD_MU_Y = 0.50
BROAD_SIGMA = 0.20


# ============================================================
# Loss sweep
# ============================================================

LOSS_CONFIGS = [
    {
        "name": "forward",
        "mode": "forward",
        "lambda_reverse": 0.0,
    },

    {
        "name": "reverse",
        "mode": "reverse",
        "lambda_reverse": 1.0,
    },

    {
        "name": "hybrid_001",
        "mode": "hybrid",
        "lambda_reverse": 0.01,
    },

    {
        "name": "hybrid_01",
        "mode": "hybrid",
        "lambda_reverse": 0.10,
    },

    {
        "name": "hybrid_1",
        "mode": "hybrid",
        "lambda_reverse": 1.0,
    },
]


# ============================================================
# Test conditions
# ============================================================

TEST_CONDITIONS = [
    (0.50, 0.50, 0.060),
    (0.30, 0.50, 0.060),
    (0.50, 0.70, 0.060),
    (0.30, 0.70, 0.060),

    (0.50, 0.50, 0.040),
    (0.50, 0.50, 0.080),

    (0.75, 0.25, 0.050),
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Target
# ============================================================

def target_sharp_broad_2d(
    xy,
    mu_x,
    mu_y,
    sigma_s,
):
    x = xy[..., 0]
    y = xy[..., 1]

    sharp = A_SHARP * torch.exp(
        -0.5 * ((x - mu_x) / sigma_s) ** 2
        -0.5 * ((y - mu_y) / sigma_s) ** 2
    )

    broad = A_BROAD * torch.exp(
        -0.5 * ((x - BROAD_MU_X) / BROAD_SIGMA) ** 2
        -0.5 * ((y - BROAD_MU_Y) / BROAD_SIGMA) ** 2
    )

    return (
        FLOOR
        + sharp
        + broad
    )


# ============================================================
# Exact integral
# ============================================================

def gaussian_integral_1d(mu, sigma):

    def Phi(z):
        return 0.5 * (
            1.0
            + math.erf(
                z / math.sqrt(2.0)
            )
        )

    lo = (
        0.0 - mu
    ) / sigma

    hi = (
        1.0 - mu
    ) / sigma

    return (
        sigma
        * math.sqrt(2.0 * math.pi)
        * (
            Phi(hi)
            - Phi(lo)
        )
    )


def gaussian_integral_2d(
    mu_x,
    mu_y,
    sigma,
):
    return (
        gaussian_integral_1d(
            mu_x,
            sigma,
        )
        *
        gaussian_integral_1d(
            mu_y,
            sigma,
        )
    )


def exact_integral(
    mu_x,
    mu_y,
    sigma_s,
):
    sharp = (
        A_SHARP
        * gaussian_integral_2d(
            mu_x,
            mu_y,
            sigma_s,
        )
    )

    broad = (
        A_BROAD
        * gaussian_integral_2d(
            BROAD_MU_X,
            BROAD_MU_Y,
            BROAD_SIGMA,
        )
    )

    return (
        FLOOR
        + sharp
        + broad
    )


# ============================================================
# Condition sampling
# ============================================================

def sample_conditions(n):

    mu_x = torch.empty(
        n,
        1,
        device=DEVICE,
    ).uniform_(
        *MU_RANGE
    )

    mu_y = torch.empty(
        n,
        1,
        device=DEVICE,
    ).uniform_(
        *MU_RANGE
    )

    sigma = torch.empty(
        n,
        1,
        device=DEVICE,
    ).uniform_(
        *SHARP_SIGMA_RANGE
    )

    return torch.cat(
        [
            mu_x,
            mu_y,
            sigma,
        ],
        dim=-1,
    )


# ============================================================
# Forward-KL training batch
# ============================================================

def sample_forward_batch():

    cond = sample_conditions(
        N_COND
    )

    xy = torch.rand(
        N_COND,
        N_X_PER_COND,
        2,
        device=DEVICE,
    )

    cond_grouped = (
        cond[:, None, :]
        .expand(
            N_COND,
            N_X_PER_COND,
            3,
        )
    )

    return (
        xy.reshape(-1, 2),
        cond_grouped.reshape(-1, 3),
    )


# ============================================================
# Model
# ============================================================

def build_model():

    return (
        ConditionalAutoregressiveCubicFlow2D(
            cond_dim=3,
            hidden=HIDDEN,
            n_bins=N_BINS,
        )
        .to(DEVICE)
    )


# ============================================================
# Forward KL
# ============================================================

def forward_kl_loss(model):

    xy, cond = sample_forward_batch()

    f_vals = target_sharp_broad_2d(
        xy,
        cond[:, 0],
        cond[:, 1],
        cond[:, 2],
    )

    log_q = model.log_prob(
        xy,
        cond,
    )

    f_vals = f_vals.reshape(
        N_COND,
        N_X_PER_COND,
    )

    log_q = log_q.reshape(
        N_COND,
        N_X_PER_COND,
    )

    weights = f_vals.detach()

    loss_per_cond = -(
        weights
        * log_q
    ).sum(
        dim=1
    ) / (
        weights.sum(dim=1)
        + 1e-12
    )

    return (
        loss_per_cond.mean()
    )


# ============================================================
# Reverse KL
# ============================================================

def reverse_kl_loss(model):

    # --------------------------------------------------------
    # Sample conditions
    # --------------------------------------------------------

    cond = sample_conditions(
        N_COND
    )

    cond_grouped = (
        cond[:, None, :]
        .expand(
            N_COND,
            N_X_PER_COND,
            3,
        )
        .reshape(
            -1,
            3,
        )
    )

    # --------------------------------------------------------
    # Sample from base
    # --------------------------------------------------------

    u = torch.rand(
        N_COND * N_X_PER_COND,
        2,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Forward through flow
    #
    # x = T(u)
    #
    # log q(x) = -log |dx/du|
    # --------------------------------------------------------

    x, logdet = model.forward(
        u,
        cond_grouped,
    )

    log_q = (
        -logdet
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    f_vals = target_sharp_broad_2d(
        x,
        cond_grouped[:, 0],
        cond_grouped[:, 1],
        cond_grouped[:, 2],
    )

    log_f = torch.log(
        f_vals
        + 1e-30
    )

    # --------------------------------------------------------
    # Reverse KL up to +log I(c), which does not depend on theta
    # --------------------------------------------------------

    loss = (
        log_q
        - log_f
    ).mean()

    return loss


# ============================================================
# Training one objective
# ============================================================

def train_one_config(
    config,
):

    name = config[
        "name"
    ]

    mode = config[
        "mode"
    ]

    lambda_reverse = config[
        "lambda_reverse"
    ]

    print()
    print(
        "=" * 100
    )

    print(
        f"TRAINING: {name}"
    )

    print(
        "=" * 100
    )

    set_seed(
        SEED
    )

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    history = []

    for step in range(
        1,
        STEPS + 1,
    ):

        loss_fwd = None
        loss_rev = None

        # ----------------------------------------------------
        # Forward only
        # ----------------------------------------------------

        if mode == "forward":

            loss = forward_kl_loss(
                model
            )

            loss_fwd = loss

        # ----------------------------------------------------
        # Reverse only
        # ----------------------------------------------------

        elif mode == "reverse":

            loss = reverse_kl_loss(
                model
            )

            loss_rev = loss

        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        elif mode == "hybrid":

            loss_fwd = forward_kl_loss(
                model
            )

            loss_rev = reverse_kl_loss(
                model
            )

            loss = (
                loss_fwd
                + lambda_reverse
                * loss_rev
            )

        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss "
                f"for {name} "
                f"at step {step}"
            )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        record = {
            "step": step,
            "loss": float(
                loss.detach().cpu()
            ),
        }

        if loss_fwd is not None:

            record[
                "loss_forward"
            ] = float(
                loss_fwd.detach().cpu()
            )

        if loss_rev is not None:

            record[
                "loss_reverse"
            ] = float(
                loss_rev.detach().cpu()
            )

        history.append(
            record
        )

        if (
            step == 1
            or step % 100 == 0
        ):

            msg = (
                f"{name:12s} | "
                f"step={step:04d} | "
                f"loss={loss.item():.6f}"
            )

            if loss_fwd is not None:
                msg += (
                    f" | fwd="
                    f"{loss_fwd.item():.6f}"
                )

            if loss_rev is not None:
                msg += (
                    f" | rev="
                    f"{loss_rev.item():.6f}"
                )

            print(
                msg
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    ckpt_path = os.path.join(
        CKPT_DIR,
        f"{name}.pt",
    )

    torch.save(
        model.state_dict(),
        ckpt_path,
    )

    with open(
        os.path.join(
            OUT_DIR,
            f"{name}_loss_history.json",
        ),
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    return model


# ============================================================
# Importance sampling evaluation
# ============================================================

@torch.no_grad()
def evaluate_integral(
    model,
    condition,
):

    (
        mu_x,
        mu_y,
        sigma,
    ) = condition

    cond = torch.tensor(
        [[
            mu_x,
            mu_y,
            sigma,
        ]],
        dtype=torch.float32,
        device=DEVICE,
    ).repeat(
        N_EVAL,
        1,
    )

    x = model.sample(
        N_EVAL,
        cond,
    )

    log_q = model.log_prob(
        x,
        cond,
    )

    q = torch.exp(
        log_q
    )

    f_vals = target_sharp_broad_2d(
        x,
        cond[:, 0],
        cond[:, 1],
        cond[:, 2],
    )

    weights = (
        f_vals
        / (
            q
            + 1e-30
        )
    )

    estimate = weights.mean()

    weight_var = weights.var(
        unbiased=True
    )

    estimator_var = (
        weight_var
        / N_EVAL
    )

    ess = (
        weights.sum() ** 2
        /
        (
            torch.sum(
                weights ** 2
            )
            + 1e-30
        )
    )

    reference = exact_integral(
        mu_x,
        mu_y,
        sigma,
    )

    rel_error = (
        abs(
            estimate.item()
            - reference
        )
        /
        abs(reference)
    )

    return {
        "estimate": float(
            estimate.cpu()
        ),

        "reference": float(
            reference
        ),

        "variance": float(
            estimator_var.cpu()
        ),

        "rel_error": float(
            rel_error
        ),

        "ess": float(
            ess.cpu()
        ),
    }


# ============================================================
# Density grid
# ============================================================

@torch.no_grad()
def evaluate_grid(
    model,
    condition,
):

    (
        mu_x,
        mu_y,
        sigma,
    ) = condition

    axis = torch.linspace(
        0.0,
        1.0,
        N_GRID,
        device=DEVICE,
    )

    X, Y = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    xy = torch.stack(
        [
            X.reshape(-1),
            Y.reshape(-1),
        ],
        dim=-1,
    )

    cond = torch.tensor(
        [[
            mu_x,
            mu_y,
            sigma,
        ]],
        dtype=torch.float32,
        device=DEVICE,
    ).repeat(
        xy.shape[0],
        1,
    )

    log_q = model.log_prob(
        xy,
        cond,
    )

    q = torch.exp(
        log_q
    )

    f_vals = target_sharp_broad_2d(
        xy,
        cond[:, 0],
        cond[:, 1],
        cond[:, 2],
    )

    reference = exact_integral(
        mu_x,
        mu_y,
        sigma,
    )

    p = (
        f_vals
        / reference
    )

    log_ratio = (
        torch.log(
            p
            + 1e-30
        )
        -
        torch.log(
            q
            + 1e-30
        )
    )

    ise = torch.mean(
        (p - q) ** 2
    )

    mae = torch.mean(
        torch.abs(
            p - q
        )
    )

    important = (
        p
        >
        0.01
        * p.max()
    )

    maxlog = torch.max(
        torch.abs(
            log_ratio[
                important
            ]
        )
    )

    return {
        "p": (
            p.reshape(
                N_GRID,
                N_GRID,
            )
            .cpu()
            .numpy()
        ),

        "q": (
            q.reshape(
                N_GRID,
                N_GRID,
            )
            .cpu()
            .numpy()
        ),

        "logratio": (
            log_ratio.reshape(
                N_GRID,
                N_GRID,
            )
            .cpu()
            .numpy()
        ),

        "ise": float(
            ise.cpu()
        ),

        "mae": float(
            mae.cpu()
        ),

        "maxlog": float(
            maxlog.cpu()
        ),
    }


# ============================================================
# Plot per config and condition
# ============================================================

def condition_tag(
    condition,
):

    mu_x, mu_y, sigma = condition

    return (
        f"mux{mu_x:.2f}_"
        f"muy{mu_y:.2f}_"
        f"sig{sigma:.3f}"
    )


def plot_condition(
    config_name,
    condition,
    grid,
):

    mu_x, mu_y, sigma = condition

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
    )

    extent = [
        0,
        1,
        0,
        1,
    ]

    # target
    im0 = axes[0].imshow(
        grid["p"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[0].set_title(
        "Target $p=f/I$"
    )

    fig.colorbar(
        im0,
        ax=axes[0],
    )

    # q
    im1 = axes[1].imshow(
        grid["q"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[1].set_title(
        f"{config_name}: $q_\\theta$"
    )

    fig.colorbar(
        im1,
        ax=axes[1],
    )

    # log ratio
    im2 = axes[2].imshow(
        grid["logratio"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[2].set_title(
        "$\\log p-\\log q$"
    )

    fig.colorbar(
        im2,
        ax=axes[2],
    )

    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    fig.suptitle(
        f"{config_name} | "
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma={sigma:.3f}$"
    )

    fig.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        f"{config_name}_"
        f"{condition_tag(condition)}.png",
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(
    model,
    config_name,
):

    results = []

    print()
    print(
        f"EVALUATING {config_name}"
    )

    for condition in TEST_CONDITIONS:

        integral_result = (
            evaluate_integral(
                model,
                condition,
            )
        )

        grid_result = (
            evaluate_grid(
                model,
                condition,
            )
        )

        plot_condition(
            config_name,
            condition,
            grid_result,
        )

        result = {
            "mu_x": condition[0],
            "mu_y": condition[1],
            "sigma": condition[2],

            **integral_result,

            "ise":
                grid_result["ise"],

            "mae":
                grid_result["mae"],

            "maxlog":
                grid_result["maxlog"],
        }

        results.append(
            result
        )

        print(
            f"{config_name:12s} | "
            f"c={condition} | "
            f"var={result['variance']:.3e} | "
            f"rel={result['rel_error']:.3e} | "
            f"ESS={result['ess']:.1f} | "
            f"ISE={result['ise']:.3e} | "
            f"maxlog={result['maxlog']:.3f}"
        )

    return results


# ============================================================
# Summary
# ============================================================

def summarize(
    results,
):

    return {
        "mean_variance": float(
            np.mean(
                [
                    r["variance"]
                    for r in results
                ]
            )
        ),

        "mean_rel_error": float(
            np.mean(
                [
                    r["rel_error"]
                    for r in results
                ]
            )
        ),

        "mean_ess": float(
            np.mean(
                [
                    r["ess"]
                    for r in results
                ]
            )
        ),

        "mean_ise": float(
            np.mean(
                [
                    r["ise"]
                    for r in results
                ]
            )
        ),

        "mean_mae": float(
            np.mean(
                [
                    r["mae"]
                    for r in results
                ]
            )
        ),

        "mean_maxlog": float(
            np.mean(
                [
                    r["maxlog"]
                    for r in results
                ]
            )
        ),
    }


# ============================================================
# Cross-objective contact sheet
# ============================================================

def make_objective_contact_sheet(
    all_grid_results,
):
    """
    One figure per test condition.

    Rows = objective
    Columns = target / q / log ratio
    """

    for condition in TEST_CONDITIONS:

        n_rows = len(
            LOSS_CONFIGS
        )

        fig, axes = plt.subplots(
            n_rows,
            3,
            figsize=(
                12,
                3.3 * n_rows,
            ),
        )

        extent = [
            0,
            1,
            0,
            1,
        ]

        for row, config in enumerate(
            LOSS_CONFIGS
        ):

            name = config[
                "name"
            ]

            grid = (
                all_grid_results[
                    name
                ][
                    condition
                ]
            )

            im0 = axes[row, 0].imshow(
                grid["p"].T,
                origin="lower",
                extent=extent,
                aspect="equal",
            )

            axes[row, 0].set_title(
                f"{name}: target"
            )

            fig.colorbar(
                im0,
                ax=axes[row, 0],
            )

            im1 = axes[row, 1].imshow(
                grid["q"].T,
                origin="lower",
                extent=extent,
                aspect="equal",
            )

            axes[row, 1].set_title(
                f"{name}: q"
            )

            fig.colorbar(
                im1,
                ax=axes[row, 1],
            )

            im2 = axes[row, 2].imshow(
                grid["logratio"].T,
                origin="lower",
                extent=extent,
                aspect="equal",
            )

            axes[row, 2].set_title(
                f"{name}: log p - log q"
            )

            fig.colorbar(
                im2,
                ax=axes[row, 2],
            )

            for col in range(3):

                axes[row, col].set_xlabel(
                    "$x$"
                )

                axes[row, col].set_ylabel(
                    "$y$"
                )

        mu_x, mu_y, sigma = condition

        fig.suptitle(
            f"Loss comparison | "
            f"$\\mu_x={mu_x:.2f}$, "
            f"$\\mu_y={mu_y:.2f}$, "
            f"$\\sigma={sigma:.3f}$",
            fontsize=14,
        )

        fig.tight_layout()

        path = os.path.join(
            PLOT_DIR,
            "objective_comparison_"
            + condition_tag(
                condition
            )
            + ".png",
        )

        fig.savefig(
            path,
            dpi=150,
        )

        plt.close(
            fig
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"device={DEVICE}"
    )

    print(
        f"N_COND={N_COND}, "
        f"N_X_PER_COND={N_X_PER_COND}"
    )

    print(
        f"steps={STEPS}"
    )

    all_results = {}
    all_summaries = {}

    trained_models = {}

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for config in LOSS_CONFIGS:

        model = train_one_config(
            config
        )

        trained_models[
            config["name"]
        ] = model

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    all_grid_results = {}

    for config in LOSS_CONFIGS:

        name = config[
            "name"
        ]

        model = trained_models[
            name
        ]

        results = evaluate_model(
            model,
            name,
        )

        summary = summarize(
            results
        )

        all_results[
            name
        ] = results

        all_summaries[
            name
        ] = summary

        # Needed for cross-objective plots
        all_grid_results[
            name
        ] = {}

        for condition in TEST_CONDITIONS:

            all_grid_results[
                name
            ][
                condition
            ] = evaluate_grid(
                model,
                condition,
            )

    # --------------------------------------------------------
    # Objective comparison plots
    # --------------------------------------------------------

    make_objective_contact_sheet(
        all_grid_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "config": {
            "n_cond": N_COND,
            "n_x_per_cond":
                N_X_PER_COND,
            "steps":
                STEPS,
            "n_bins":
                N_BINS,
            "hidden":
                HIDDEN,
            "sigma_range":
                SHARP_SIGMA_RANGE,
            "loss_configs":
                LOSS_CONFIGS,
        },

        "summaries":
            all_summaries,

        "results":
            all_results,
    }

    output_path = os.path.join(
        OUT_DIR,
        "loss_sweep_results.json",
    )

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print summary table
    # --------------------------------------------------------

    print()
    print(
        "=" * 115
    )

    print(
        "LOSS SWEEP SUMMARY"
    )

    print(
        "=" * 115
    )

    print(
        f"{'config':<14}"
        f"{'variance':>16}"
        f"{'rel err':>16}"
        f"{'ESS':>14}"
        f"{'ISE':>16}"
        f"{'MAE':>16}"
        f"{'maxlog':>14}"
    )

    print(
        "-" * 115
    )

    for config in LOSS_CONFIGS:

        name = config[
            "name"
        ]

        s = all_summaries[
            name
        ]

        print(
            f"{name:<14}"
            f"{s['mean_variance']:>16.3e}"
            f"{s['mean_rel_error']:>16.3e}"
            f"{s['mean_ess']:>14.1f}"
            f"{s['mean_ise']:>16.3e}"
            f"{s['mean_mae']:>16.3e}"
            f"{s['mean_maxlog']:>14.3f}"
        )

    print()
    print(
        f"Saved results: {output_path}"
    )

    print(
        f"Saved plots: {PLOT_DIR}"
    )


if __name__ == "__main__":
    main()