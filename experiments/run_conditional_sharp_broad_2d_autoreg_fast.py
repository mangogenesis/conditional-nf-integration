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
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 1234

OUT_DIR = "results/conditional_sharp_broad_2d_autoreg_fast"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# Fast benchmark settings
# ============================================================

N_COND = 64
N_X_PER_COND = 2048

STEPS = 2000
LR = 5e-5

N_EVAL = 20000
N_GRID = 100


# ============================================================
# Flow settings
# ============================================================

N_BINS = 16
HIDDEN = 128


# ============================================================
# Target settings
# ============================================================

FLOOR = 1e-6

A_SHARP = 1.0

MU_RANGE = (0.20, 0.80)

# Easier diagnostic sharp peak
SHARP_SIGMA_RANGE = (0.04, 0.08)

# Fixed broad Gaussian
A_BROAD = 0.20

BROAD_MU_X = 0.50
BROAD_MU_Y = 0.50
BROAD_SIGMA = 0.20


# ============================================================
# Test conditions
# ============================================================

TEST_CONDITIONS = [
    (0.50, 0.50, 0.060),

    (0.30, 0.50, 0.060),

    (0.50, 0.70, 0.060),

    (0.30, 0.70, 0.060),

    # lower sigma edge
    (0.50, 0.50, 0.040),

    # upper sigma edge
    (0.50, 0.50, 0.080),

    # spatially separated sharp peak
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
    """
    Target on [0,1]^2:

        floor
        + moving sharp Gaussian
        + fixed broad Gaussian
    """

    x = xy[..., 0]
    y = xy[..., 1]

    sharp = A_SHARP * torch.exp(
        -0.5 * ((x - mu_x) / sigma_s) ** 2
        -0.5 * ((y - mu_y) / sigma_s) ** 2
    )

    broad = A_BROAD * torch.exp(
        -0.5
        * ((x - BROAD_MU_X) / BROAD_SIGMA) ** 2
        -0.5
        * ((y - BROAD_MU_Y) / BROAD_SIGMA) ** 2
    )

    return (
        FLOOR
        + sharp
        + broad
    )


# ============================================================
# Exact reference integral
# ============================================================

def gaussian_integral_1d(
    mu,
    sigma,
):
    def Phi(z):

        return (
            0.5
            * (
                1.0
                + math.erf(
                    z
                    / math.sqrt(2.0)
                )
            )
        )

    lower = (
        0.0 - mu
    ) / sigma

    upper = (
        1.0 - mu
    ) / sigma

    return (
        sigma
        * math.sqrt(2.0 * math.pi)
        * (
            Phi(upper)
            - Phi(lower)
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

    sharp_integral = (
        A_SHARP
        * gaussian_integral_2d(
            mu_x,
            mu_y,
            sigma_s,
        )
    )

    broad_integral = (
        A_BROAD
        * gaussian_integral_2d(
            BROAD_MU_X,
            BROAD_MU_Y,
            BROAD_SIGMA,
        )
    )

    # Unit-square area = 1
    floor_integral = FLOOR

    return (
        floor_integral
        + sharp_integral
        + broad_integral
    )


# ============================================================
# Conditional training batch
# ============================================================

def sample_grouped_batch():

    mu_x = torch.empty(
        N_COND,
        1,
        device=DEVICE,
    ).uniform_(
        *MU_RANGE
    )

    mu_y = torch.empty(
        N_COND,
        1,
        device=DEVICE,
    ).uniform_(
        *MU_RANGE
    )

    sigma_s = torch.empty(
        N_COND,
        1,
        device=DEVICE,
    ).uniform_(
        *SHARP_SIGMA_RANGE
    )

    cond = torch.cat(
        [
            mu_x,
            mu_y,
            sigma_s,
        ],
        dim=-1,
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

    xy_flat = (
        xy.reshape(
            -1,
            2,
        )
    )

    cond_flat = (
        cond_grouped.reshape(
            -1,
            3,
        )
    )

    return (
        xy_flat,
        cond_flat,
    )


# ============================================================
# Model
# ============================================================

def build_model():

    model = (
        ConditionalAutoregressiveCubicFlow2D(
            cond_dim=3,
            hidden=HIDDEN,
            n_bins=N_BINS,
        )
    )

    return model.to(
        DEVICE
    )


# ============================================================
# Training
# ============================================================

def train_model():

    set_seed(
        SEED
    )

    model = (
        build_model()
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LR,
        )
    )

    loss_history = []

    print(
        "=" * 90
    )

    print(
        "AUTOREGRESSIVE CUBIC 2D SHARP + BROAD DIAGNOSTIC"
    )

    print(
        "=" * 90
    )

    print(
        f"device={DEVICE}"
    )

    print(
        f"N_COND={N_COND}, "
        f"N_X_PER_COND={N_X_PER_COND}"
    )

    print(
        f"sigma range="
        f"{SHARP_SIGMA_RANGE}"
    )

    print(
        f"bins={N_BINS}, "
        f"hidden={HIDDEN}"
    )

    for step in range(
        1,
        STEPS + 1,
    ):

        xy, cond = (
            sample_grouped_batch()
        )

        f_vals = (
            target_sharp_broad_2d(
                xy,
                cond[:, 0],
                cond[:, 1],
                cond[:, 2],
            )
        )

        log_q = (
            model.log_prob(
                xy,
                cond,
            )
        )

        # ----------------------------------------------------
        # Group samples by condition
        # ----------------------------------------------------

        f_vals = (
            f_vals.reshape(
                N_COND,
                N_X_PER_COND,
            )
        )

        log_q = (
            log_q.reshape(
                N_COND,
                N_X_PER_COND,
            )
        )

        # ----------------------------------------------------
        # Forward-KL-style weighted loss
        #
        # Approximately:
        #
        #   - E_p[log q]
        #
        # with p proportional to f.
        # ----------------------------------------------------

        weights = (
            f_vals.detach()
        )

        numerator = (
            weights
            * log_q
        ).sum(
            dim=1
        )

        denominator = (
            weights.sum(
                dim=1
            )
            + 1e-12
        )

        loss_per_cond = (
            -numerator
            / denominator
        )

        loss = (
            loss_per_cond.mean()
        )

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss "
                f"at step {step}: "
                f"{loss.item()}"
            )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        loss_history.append(
            float(
                loss.detach().cpu()
            )
        )

        if (
            step == 1
            or step % 100 == 0
        ):

            print(
                f"step={step:04d} | "
                f"loss={loss.item():.6f}"
            )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        os.path.join(
            CKPT_DIR,
            "model.pt",
        )
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    with open(
        os.path.join(
            OUT_DIR,
            "loss_history.json",
        ),
        "w",
    ) as f:

        json.dump(
            loss_history,
            f,
            indent=2,
        )

    return model


# ============================================================
# Importance-sampling evaluation
# ============================================================

@torch.no_grad()
def evaluate_integral(
    model,
    condition,
):

    (
        mu_x,
        mu_y,
        sigma_s,
    ) = condition

    cond = torch.tensor(
        [[
            mu_x,
            mu_y,
            sigma_s,
        ]],
        dtype=torch.float32,
        device=DEVICE,
    ).repeat(
        N_EVAL,
        1,
    )

    # --------------------------------------------------------
    # Draw samples from learned q
    # --------------------------------------------------------

    samples = (
        model.sample(
            N_EVAL,
            cond,
        )
    )

    log_q = (
        model.log_prob(
            samples,
            cond,
        )
    )

    q = (
        torch.exp(
            log_q
        )
    )

    f_vals = (
        target_sharp_broad_2d(
            samples,
            cond[:, 0],
            cond[:, 1],
            cond[:, 2],
        )
    )

    weights = (
        f_vals
        / (
            q
            + 1e-30
        )
    )

    estimate = (
        weights.mean()
    )

    weight_variance = (
        weights.var(
            unbiased=True
        )
    )

    estimator_variance = (
        weight_variance
        / N_EVAL
    )

    std_error = (
        torch.sqrt(
            estimator_variance
        )
    )

    reference = (
        exact_integral(
            mu_x,
            mu_y,
            sigma_s,
        )
    )

    rel_error = (
        abs(
            estimate.item()
            - reference
        )
        / abs(reference)
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

    pull = (
        estimate.item()
        - reference
    ) / (
        std_error.item()
        + 1e-30
    )

    return {
        "estimate": float(
            estimate.cpu()
        ),

        "reference": float(
            reference
        ),

        "weight_variance": float(
            weight_variance.cpu()
        ),

        "variance": float(
            estimator_variance.cpu()
        ),

        "std_error": float(
            std_error.cpu()
        ),

        "rel_error": float(
            rel_error
        ),

        "ess": float(
            ess.cpu()
        ),

        "pull": float(
            pull
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
        sigma_s,
    ) = condition

    axis = torch.linspace(
        0.0,
        1.0,
        N_GRID,
        device=DEVICE,
    )

    X, Y = (
        torch.meshgrid(
            axis,
            axis,
            indexing="ij",
        )
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
            sigma_s,
        ]],
        dtype=torch.float32,
        device=DEVICE,
    ).repeat(
        xy.shape[0],
        1,
    )

    log_q = (
        model.log_prob(
            xy,
            cond,
        )
    )

    q = (
        torch.exp(
            log_q
        )
    )

    f_vals = (
        target_sharp_broad_2d(
            xy,
            cond[:, 0],
            cond[:, 1],
            cond[:, 2],
        )
    )

    reference = (
        exact_integral(
            mu_x,
            mu_y,
            sigma_s,
        )
    )

    p = (
        f_vals
        / reference
    )

    # --------------------------------------------------------
    # Density metrics
    # --------------------------------------------------------

    ise = (
        torch.mean(
            (p - q) ** 2
        )
    )

    mae = (
        torch.mean(
            torch.abs(
                p - q
            )
        )
    )

    log_ratio = (
        torch.log(
            p + 1e-30
        )
        -
        torch.log(
            q + 1e-30
        )
    )

    important = (
        p
        >
        0.01
        * torch.max(
            p
        )
    )

    maxlog = (
        torch.max(
            torch.abs(
                log_ratio[
                    important
                ]
            )
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
# Plot one condition
# ============================================================

def condition_tag(
    condition,
):

    (
        mu_x,
        mu_y,
        sigma_s,
    ) = condition

    return (
        f"mux{mu_x:.2f}_"
        f"muy{mu_y:.2f}_"
        f"sig{sigma_s:.3f}"
    )


def plot_condition(
    condition,
    grid,
):

    (
        mu_x,
        mu_y,
        sigma_s,
    ) = condition

    fig, axes = (
        plt.subplots(
            1,
            3,
            figsize=(14, 4),
        )
    )

    extent = [
        0.0,
        1.0,
        0.0,
        1.0,
    ]

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Proposal
    # --------------------------------------------------------

    im1 = axes[1].imshow(
        grid["q"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[1].set_title(
        "Autoregressive $q_\\theta$"
    )

    fig.colorbar(
        im1,
        ax=axes[1],
    )

    # --------------------------------------------------------
    # Log ratio
    # --------------------------------------------------------

    im2 = axes[2].imshow(
        grid["logratio"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[2].set_title(
        "$\\log p-\\log q_\\theta$"
    )

    fig.colorbar(
        im2,
        ax=axes[2],
    )

    for ax in axes:

        ax.set_xlabel(
            "$x$"
        )

        ax.set_ylabel(
            "$y$"
        )

    fig.suptitle(
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma_s={sigma_s:.3f}$"
    )

    fig.tight_layout()

    path = (
        os.path.join(
            PLOT_DIR,
            condition_tag(
                condition
            )
            + "_comparison.png",
        )
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(
        fig
    )


# ============================================================
# Contact sheet
# ============================================================

def make_contact_sheet():

    import matplotlib.image as mpimg

    paths = []

    for condition in TEST_CONDITIONS:

        paths.append(
            os.path.join(
                PLOT_DIR,
                condition_tag(
                    condition
                )
                + "_comparison.png",
            )
        )

    fig, axes = (
        plt.subplots(
            2,
            4,
            figsize=(18, 9),
        )
    )

    axes = (
        axes.flatten()
    )

    for ax, path in zip(
        axes,
        paths,
    ):

        image = (
            mpimg.imread(
                path
            )
        )

        ax.imshow(
            image
        )

        ax.axis(
            "off"
        )

    for ax in axes[
        len(paths):
    ]:

        ax.axis(
            "off"
        )

    fig.suptitle(
        "Autoregressive Cubic Flow: Sharp Peak + Broad Background",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        os.path.join(
            PLOT_DIR,
            "autoregressive_sharp_broad_2d_contact_sheet.png",
        )
    )

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(
        fig
    )

    print(
        f"Saved contact sheet: "
        f"{output_path}"
    )


# ============================================================
# Full evaluation
# ============================================================

def evaluate_model(
    model,
):

    results = []

    print()
    print(
        "=" * 100
    )

    print(
        "EVALUATION"
    )

    print(
        "=" * 100
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
            condition,
            grid_result,
        )

        result = {
            "mu_x": condition[0],
            "mu_y": condition[1],
            "sigma_s": condition[2],

            **integral_result,

            "ise": grid_result[
                "ise"
            ],

            "mae": grid_result[
                "mae"
            ],

            "maxlog": grid_result[
                "maxlog"
            ],
        }

        results.append(
            result
        )

        print(
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

    summary = {

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

    return summary


# ============================================================
# Main
# ============================================================

def main():

    model = (
        train_model()
    )

    results = (
        evaluate_model(
            model
        )
    )

    make_contact_sheet()

    summary = (
        summarize(
            results
        )
    )

    output = {

        "config": {

            "architecture":
                "autoregressive_cubic_2d",

            "n_bins":
                N_BINS,

            "hidden":
                HIDDEN,

            "n_cond":
                N_COND,

            "n_x_per_cond":
                N_X_PER_COND,

            "steps":
                STEPS,

            "sigma_range":
                SHARP_SIGMA_RANGE,
        },

        "summary":
            summary,

        "results":
            results,
    }

    results_path = (
        os.path.join(
            OUT_DIR,
            "results.json",
        )
    )

    with open(
        results_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 100
    )

    print(
        "AUTOREGRESSIVE 2D SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"mean variance : "
        f"{summary['mean_variance']:.3e}"
    )

    print(
        f"mean rel err  : "
        f"{summary['mean_rel_error']:.3e}"
    )

    print(
        f"mean ESS      : "
        f"{summary['mean_ess']:.1f}"
    )

    print(
        f"mean ISE      : "
        f"{summary['mean_ise']:.3e}"
    )

    print(
        f"mean MAE      : "
        f"{summary['mean_mae']:.3e}"
    )

    print(
        f"mean maxlog   : "
        f"{summary['mean_maxlog']:.3f}"
    )

    print()

    print(
        f"Saved results: "
        f"{results_path}"
    )

    print(
        f"Saved plots: "
        f"{PLOT_DIR}"
    )


if __name__ == "__main__":
    main()