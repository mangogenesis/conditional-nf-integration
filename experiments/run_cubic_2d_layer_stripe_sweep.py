import os
import json
import math
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from nf.conditional_cubic_2d import ConditionalCubicFlow2D


# ============================================================
# General
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234

OUT_DIR = "results/cubic_2d_layer_stripe_sweep"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# Flow sweep
# ============================================================

LAYER_COUNTS = [1, 2, 4, 8]

N_BINS = 32
HIDDEN = 128


# ============================================================
# Training
# ============================================================

N_COND = 64
N_X_PER_COND = 1024

STEPS = 3000
LR = 5e-5


# ============================================================
# Target
# ============================================================

FLOOR = 1e-6

MU_MIN = 0.20
MU_MAX = 0.80

SIGMA_MIN = 0.03
SIGMA_MAX = 0.08


# ============================================================
# Evaluation
# ============================================================

N_EVAL = 100000
N_GRID = 200

# Main stripe diagnostic condition
DIAGNOSTIC_CONDITION = (
    0.70,
    0.30,
    0.05,
)

TEST_CONDITIONS = [
    (0.50, 0.50, 0.05),
    (0.30, 0.50, 0.05),
    (0.70, 0.50, 0.05),
    (0.50, 0.30, 0.05),
    (0.50, 0.70, 0.05),
    (0.30, 0.70, 0.05),
    (0.70, 0.30, 0.05),
    (0.75, 0.25, 0.04),
]


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Conditions
# ============================================================

def sample_conditions(n):

    mu_x = torch.empty(
        n,
        1,
        device=DEVICE,
    ).uniform_(
        MU_MIN,
        MU_MAX,
    )

    mu_y = torch.empty(
        n,
        1,
        device=DEVICE,
    ).uniform_(
        MU_MIN,
        MU_MAX,
    )

    sigma = torch.empty(
        n,
        1,
        device=DEVICE,
    ).uniform_(
        SIGMA_MIN,
        SIGMA_MAX,
    )

    return torch.cat(
        [
            mu_x,
            mu_y,
            sigma,
        ],
        dim=-1,
    )


def normalize_condition(cond_raw):

    cond = cond_raw.clone()

    cond[..., 0] = (
        cond[..., 0] - MU_MIN
    ) / (
        MU_MAX - MU_MIN
    )

    cond[..., 1] = (
        cond[..., 1] - MU_MIN
    ) / (
        MU_MAX - MU_MIN
    )

    cond[..., 2] = (
        cond[..., 2] - SIGMA_MIN
    ) / (
        SIGMA_MAX - SIGMA_MIN
    )

    return cond


# ============================================================
# Target
# ============================================================

def target_single_gaussian(
    xy,
    cond_raw,
):

    x = xy[..., 0]
    y = xy[..., 1]

    mu_x = cond_raw[..., 0]
    mu_y = cond_raw[..., 1]
    sigma = cond_raw[..., 2]

    gaussian = torch.exp(
        -0.5
        * (
            (x - mu_x)
            / sigma
        ) ** 2
        -
        0.5
        * (
            (y - mu_y)
            / sigma
        ) ** 2
    )

    return (
        FLOOR
        + gaussian
    )


# ============================================================
# Exact integral
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

    lo = (
        -mu / sigma
    )

    hi = (
        (1.0 - mu)
        / sigma
    )

    return (
        sigma
        * math.sqrt(
            2.0 * math.pi
        )
        * (
            Phi(hi)
            - Phi(lo)
        )
    )


def exact_integral(
    condition,
):

    mu_x, mu_y, sigma = condition

    return (
        FLOOR
        +
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


# ============================================================
# Build model
# ============================================================

def build_model(
    n_layers,
):

    model = ConditionalCubicFlow2D(
        cond_dim=3,
        hidden=HIDDEN,
        n_bins=N_BINS,
        n_layers=n_layers,
    )

    return model.to(
        DEVICE
    )


# ============================================================
# Forward KL
# ============================================================

def forward_kl_loss(
    model,
):

    cond_raw = sample_conditions(
        N_COND
    )

    cond_grouped = (
        cond_raw[:, None, :]
        .expand(
            N_COND,
            N_X_PER_COND,
            3,
        )
    )

    xy = torch.rand(
        N_COND,
        N_X_PER_COND,
        2,
        device=DEVICE,
    )

    xy_flat = xy.reshape(
        -1,
        2,
    )

    cond_raw_flat = cond_grouped.reshape(
        -1,
        3,
    )

    cond_model_flat = normalize_condition(
        cond_raw_flat
    )

    f = target_single_gaussian(
        xy_flat,
        cond_raw_flat,
    )

    log_q = model.log_prob(
        xy_flat,
        cond_model_flat,
    )

    f = f.reshape(
        N_COND,
        N_X_PER_COND,
    )

    log_q = log_q.reshape(
        N_COND,
        N_X_PER_COND,
    )

    weights = f.detach()

    loss_per_cond = -(
        weights
        * log_q
    ).sum(
        dim=1
    ) / (
        weights.sum(
            dim=1
        )
        + 1e-30
    )

    return (
        loss_per_cond.mean()
    )


# ============================================================
# Training
# ============================================================

def train_model(
    n_layers,
):

    name = (
        f"layers_{n_layers}"
    )

    set_seed(
        SEED
    )

    print()
    print("=" * 100)
    print(
        f"TRAINING {name}"
    )
    print("=" * 100)

    model = build_model(
        n_layers
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    history = []

    for step in range(
        1,
        STEPS + 1,
    ):

        loss = forward_kl_loss(
            model
        )

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

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                10.0,
            )
        )

        optimizer.step()

        history.append(
            {
                "step":
                    step,

                "loss":
                    float(
                        loss.detach().cpu()
                    ),

                "grad_norm":
                    float(
                        grad_norm.detach().cpu()
                    ),
            }
        )

        if (
            step == 1
            or step % 100 == 0
        ):

            print(
                f"{name:10s} | "
                f"step={step:04d} | "
                f"loss={loss.item():.6f} | "
                f"grad={float(grad_norm):.3f}"
            )

    torch.save(
        model.state_dict(),
        os.path.join(
            CKPT_DIR,
            f"{name}.pt",
        ),
    )

    with open(
        os.path.join(
            OUT_DIR,
            f"{name}_history.json",
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
# Simple Gaussian blur
#
# Used only for stripe score.
# No scipy dependency.
# ============================================================

def gaussian_kernel_1d(
    sigma=2.0,
    radius=6,
):

    x = np.arange(
        -radius,
        radius + 1,
    )

    kernel = np.exp(
        -0.5
        * (
            x / sigma
        ) ** 2
    )

    kernel /= kernel.sum()

    return kernel


def smooth_2d(
    arr,
):

    kernel = gaussian_kernel_1d(
        sigma=2.0,
        radius=6,
    )

    # horizontal convolution
    tmp = np.apply_along_axis(
        lambda row:
            np.convolve(
                row,
                kernel,
                mode="same",
            ),
        axis=1,
        arr=arr,
    )

    # vertical convolution
    smooth = np.apply_along_axis(
        lambda col:
            np.convolve(
                col,
                kernel,
                mode="same",
            ),
        axis=0,
        arr=tmp,
    )

    return smooth


# ============================================================
# Stripe metrics
# ============================================================

def stripe_metrics(
    q_grid,
):

    # --------------------------------------------------------
    # High-frequency residual score
    # --------------------------------------------------------

    q_smooth = smooth_2d(
        q_grid
    )

    residual = (
        q_grid
        - q_smooth
    )

    stripe_score = (
        np.linalg.norm(
            residual
        )
        /
        (
            np.linalg.norm(
                q_grid
            )
            + 1e-30
        )
    )

    # --------------------------------------------------------
    # Directional roughness
    #
    # Finite differences:
    #   dx -> variation in x direction
    #   dy -> variation in y direction
    # --------------------------------------------------------

    dq_dx = np.diff(
        q_grid,
        axis=0,
    )

    dq_dy = np.diff(
        q_grid,
        axis=1,
    )

    roughness_x = np.mean(
        dq_dx ** 2
    )

    roughness_y = np.mean(
        dq_dy ** 2
    )

    # Normalize by average q^2 so models with larger peaks
    # remain somewhat comparable.
    scale = (
        np.mean(
            q_grid ** 2
        )
        + 1e-30
    )

    roughness_x /= scale
    roughness_y /= scale

    return {
        "stripe_score":
            float(
                stripe_score
            ),

        "roughness_x":
            float(
                roughness_x
            ),

        "roughness_y":
            float(
                roughness_y
            ),

        "q_smooth":
            q_smooth,

        "residual":
            residual,
    }


# ============================================================
# Grid evaluation
# ============================================================

@torch.no_grad()
def evaluate_grid(
    model,
    condition,
):

    mu_x, mu_y, sigma = condition

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

    cond_raw = torch.tensor(
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

    cond_model = normalize_condition(
        cond_raw
    )

    log_q = model.log_prob(
        xy,
        cond_model,
    )

    q = torch.exp(
        log_q
    )

    f = target_single_gaussian(
        xy,
        cond_raw,
    )

    I_ref = exact_integral(
        condition
    )

    p = (
        f / I_ref
    )

    target_peak = xy[
        torch.argmax(
            p
        )
    ]

    proposal_peak = xy[
        torch.argmax(
            q
        )
    ]

    peak_error = torch.linalg.vector_norm(
        proposal_peak
        -
        target_peak
    )

    ise = torch.mean(
        (
            p - q
        ) ** 2
    )

    mae = torch.mean(
        torch.abs(
            p - q
        )
    )

    p_grid = p.reshape(
        N_GRID,
        N_GRID,
    ).cpu().numpy()

    q_grid = q.reshape(
        N_GRID,
        N_GRID,
    ).cpu().numpy()

    logratio = (
        torch.log(
            p + 1e-30
        )
        -
        torch.log(
            q + 1e-30
        )
    )

    logratio_grid = logratio.reshape(
        N_GRID,
        N_GRID,
    ).cpu().numpy()

    stripe = stripe_metrics(
        q_grid
    )

    return {
        "axis":
            axis.cpu().numpy(),

        "p":
            p_grid,

        "q":
            q_grid,

        "logratio":
            logratio_grid,

        "q_smooth":
            stripe["q_smooth"],

        "residual":
            stripe["residual"],

        "stripe_score":
            stripe["stripe_score"],

        "roughness_x":
            stripe["roughness_x"],

        "roughness_y":
            stripe["roughness_y"],

        "peak_error":
            float(
                peak_error.cpu()
            ),

        "target_peak_x":
            float(
                target_peak[0].cpu()
            ),

        "target_peak_y":
            float(
                target_peak[1].cpu()
            ),

        "proposal_peak_x":
            float(
                proposal_peak[0].cpu()
            ),

        "proposal_peak_y":
            float(
                proposal_peak[1].cpu()
            ),

        "ise":
            float(
                ise.cpu()
            ),

        "mae":
            float(
                mae.cpu()
            ),

        "pmax":
            float(
                p.max().cpu()
            ),

        "qmax":
            float(
                q.max().cpu()
            ),
    }


# ============================================================
# MC evaluation
# ============================================================

@torch.no_grad()
def evaluate_mc(
    model,
    condition,
):

    mu_x, mu_y, sigma = condition

    cond_raw = torch.tensor(
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

    cond_model = normalize_condition(
        cond_raw
    )

    samples = model.sample(
        N_EVAL,
        cond_model,
    )

    q = torch.exp(
        model.log_prob(
            samples,
            cond_model,
        )
    )

    f = target_single_gaussian(
        samples,
        cond_raw,
    )

    weights = (
        f
        /
        (
            q + 1e-30
        )
    )

    estimate = weights.mean()

    variance = (
        weights.var(
            unbiased=True
        )
        /
        N_EVAL
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
        condition
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
        "estimate":
            float(
                estimate.cpu()
            ),

        "reference":
            float(
                reference
            ),

        "variance":
            float(
                variance.cpu()
            ),

        "rel_error":
            float(
                rel_error
            ),

        "ess":
            float(
                ess.cpu()
            ),

        "ess_fraction":
            float(
                ess.cpu()
                / N_EVAL
            ),
    }


# ============================================================
# Diagnostic plot
# ============================================================

def plot_diagnostic(
    n_layers,
    condition,
    grid,
):

    mu_x, mu_y, sigma = condition

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    extent = [
        0,
        1,
        0,
        1,
    ]

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    im = axes[0, 0].imshow(
        grid["p"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    fig.colorbar(
        im,
        ax=axes[0, 0],
    )

    axes[0, 0].set_title(
        "Target $p$"
    )

    im = axes[0, 1].imshow(
        grid["q"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    fig.colorbar(
        im,
        ax=axes[0, 1],
    )

    axes[0, 1].set_title(
        f"Flow $q$ — {n_layers} layers"
    )

    axes[0, 1].scatter(
        grid["target_peak_x"],
        grid["target_peak_y"],
        marker="x",
        s=80,
        label="target peak",
    )

    axes[0, 1].scatter(
        grid["proposal_peak_x"],
        grid["proposal_peak_y"],
        marker="+",
        s=100,
        label="q peak",
    )

    axes[0, 1].legend()

    im = axes[0, 2].imshow(
        grid["logratio"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    fig.colorbar(
        im,
        ax=axes[0, 2],
    )

    axes[0, 2].set_title(
        "$\\log p - \\log q$"
    )

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    im = axes[1, 0].imshow(
        grid["residual"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    fig.colorbar(
        im,
        ax=axes[1, 0],
    )

    axes[1, 0].set_title(
        "$q - \\mathrm{smooth}(q)$"
    )

    # --------------------------------------------------------
    # Horizontal cross-section:
    # y closest to target mu_y
    # --------------------------------------------------------

    axis = grid["axis"]

    iy = int(
        np.argmin(
            np.abs(
                axis - mu_y
            )
        )
    )

    axes[1, 1].plot(
        axis,
        grid["p"][:, iy],
        label="target",
    )

    axes[1, 1].plot(
        axis,
        grid["q"][:, iy],
        label="flow",
    )

    axes[1, 1].set_title(
        f"Cross-section at y={axis[iy]:.3f}"
    )

    axes[1, 1].set_xlabel(
        "x"
    )

    axes[1, 1].legend()

    # --------------------------------------------------------
    # Vertical cross-section:
    # x closest to target mu_x
    # --------------------------------------------------------

    ix = int(
        np.argmin(
            np.abs(
                axis - mu_x
            )
        )
    )

    axes[1, 2].plot(
        axis,
        grid["p"][ix, :],
        label="target",
    )

    axes[1, 2].plot(
        axis,
        grid["q"][ix, :],
        label="flow",
    )

    axes[1, 2].set_title(
        f"Cross-section at x={axis[ix]:.3f}"
    )

    axes[1, 2].set_xlabel(
        "y"
    )

    axes[1, 2].legend()

    for ax in axes.flat:

        ax.set_xlabel(
            ax.get_xlabel()
            if ax.get_xlabel()
            else "$x$"
        )

    fig.suptitle(
        f"{n_layers} layers | "
        f"mu=({mu_x:.2f},{mu_y:.2f}), "
        f"sigma={sigma:.3f} | "
        f"stripe={grid['stripe_score']:.4f} | "
        f"Rx={grid['roughness_x']:.4e} | "
        f"Ry={grid['roughness_y']:.4e}"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            f"diagnostic_layers_{n_layers}.png",
        ),
        dpi=170,
    )

    plt.close(
        fig
    )


# ============================================================
# Comparison plot
# ============================================================

def plot_layer_comparison(
    layer_grids,
):

    fig, axes = plt.subplots(
        1,
        len(LAYER_COUNTS),
        figsize=(
            4 * len(LAYER_COUNTS),
            4,
        ),
    )

    for ax, n_layers in zip(
        axes,
        LAYER_COUNTS,
    ):

        grid = layer_grids[
            n_layers
        ]

        im = ax.imshow(
            grid["q"].T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
        )

        fig.colorbar(
            im,
            ax=ax,
        )

        ax.set_title(
            f"{n_layers} layer(s)\n"
            f"S={grid['stripe_score']:.3f}"
        )

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            "y"
        )

    fig.suptitle(
        "Cubic coupling layer sweep"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            "layer_comparison.png",
        ),
        dpi=170,
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
        f"layers={LAYER_COUNTS}"
    )

    print(
        f"bins={N_BINS}"
    )

    models = {}

    summaries = {}

    diagnostic_grids = {}

    # ========================================================
    # Train
    # ========================================================

    for n_layers in LAYER_COUNTS:

        models[
            n_layers
        ] = train_model(
            n_layers
        )

    # ========================================================
    # Evaluate
    # ========================================================

    for n_layers in LAYER_COUNTS:

        print()
        print("=" * 120)
        print(
            f"EVALUATING {n_layers} LAYER(S)"
        )
        print("=" * 120)

        model = models[
            n_layers
        ]

        condition_results = []

        for condition in TEST_CONDITIONS:

            grid = evaluate_grid(
                model,
                condition,
            )

            mc = evaluate_mc(
                model,
                condition,
            )

            result = {
                "condition":
                    list(
                        condition
                    ),

                "peak_error":
                    grid["peak_error"],

                "ise":
                    grid["ise"],

                "mae":
                    grid["mae"],

                "stripe_score":
                    grid["stripe_score"],

                "roughness_x":
                    grid["roughness_x"],

                "roughness_y":
                    grid["roughness_y"],

                "pmax":
                    grid["pmax"],

                "qmax":
                    grid["qmax"],

                **mc,
            }

            condition_results.append(
                result
            )

            print(
                f"c={condition} | "
                f"peak={grid['peak_error']:.4f} | "
                f"stripe={grid['stripe_score']:.4f} | "
                f"Rx={grid['roughness_x']:.3e} | "
                f"Ry={grid['roughness_y']:.3e} | "
                f"ISE={grid['ise']:.3e} | "
                f"ESS={mc['ess_fraction']:.3f}"
            )

        # Main fixed diagnostic
        diagnostic_grid = evaluate_grid(
            model,
            DIAGNOSTIC_CONDITION,
        )

        diagnostic_grids[
            n_layers
        ] = diagnostic_grid

        plot_diagnostic(
            n_layers,
            DIAGNOSTIC_CONDITION,
            diagnostic_grid,
        )

        summaries[
            str(n_layers)
        ] = {
            "mean_peak_error":
                float(
                    np.mean(
                        [
                            r["peak_error"]
                            for r in condition_results
                        ]
                    )
                ),

            "mean_ise":
                float(
                    np.mean(
                        [
                            r["ise"]
                            for r in condition_results
                        ]
                    )
                ),

            "mean_stripe_score":
                float(
                    np.mean(
                        [
                            r["stripe_score"]
                            for r in condition_results
                        ]
                    )
                ),

            "mean_roughness_x":
                float(
                    np.mean(
                        [
                            r["roughness_x"]
                            for r in condition_results
                        ]
                    )
                ),

            "mean_roughness_y":
                float(
                    np.mean(
                        [
                            r["roughness_y"]
                            for r in condition_results
                        ]
                    )
                ),

            "mean_ess_fraction":
                float(
                    np.mean(
                        [
                            r["ess_fraction"]
                            for r in condition_results
                        ]
                    )
                ),

            "diagnostic":
                {
                    "condition":
                        list(
                            DIAGNOSTIC_CONDITION
                        ),

                    "stripe_score":
                        diagnostic_grid[
                            "stripe_score"
                        ],

                    "roughness_x":
                        diagnostic_grid[
                            "roughness_x"
                        ],

                    "roughness_y":
                        diagnostic_grid[
                            "roughness_y"
                        ],

                    "peak_error":
                        diagnostic_grid[
                            "peak_error"
                        ],

                    "ise":
                        diagnostic_grid[
                            "ise"
                        ],
                },

            "conditions":
                condition_results,
        }

    # ========================================================
    # Plot all layer counts
    # ========================================================

    plot_layer_comparison(
        diagnostic_grids
    )

    # ========================================================
    # Save
    # ========================================================

    with open(
        os.path.join(
            OUT_DIR,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            {
                "settings": {
                    "layers":
                        LAYER_COUNTS,

                    "bins":
                        N_BINS,

                    "hidden":
                        HIDDEN,

                    "steps":
                        STEPS,

                    "n_cond":
                        N_COND,

                    "n_x_per_cond":
                        N_X_PER_COND,

                    "diagnostic_condition":
                        list(
                            DIAGNOSTIC_CONDITION
                        ),
                },

                "results":
                    summaries,
            },
            f,
            indent=2,
        )

    # ========================================================
    # Final table
    # ========================================================

    print()
    print("=" * 115)
    print(
        "CUBIC COUPLING LAYER SWEEP"
    )
    print("=" * 115)

    print(
        f"{'layers':>8}"
        f"{'peak err':>14}"
        f"{'stripe':>14}"
        f"{'rough X':>16}"
        f"{'rough Y':>16}"
        f"{'ISE':>16}"
        f"{'ESS frac':>14}"
    )

    print("-" * 115)

    for n_layers in LAYER_COUNTS:

        s = summaries[
            str(
                n_layers
            )
        ]

        print(
            f"{n_layers:>8d}"
            f"{s['mean_peak_error']:>14.4f}"
            f"{s['mean_stripe_score']:>14.4f}"
            f"{s['mean_roughness_x']:>16.3e}"
            f"{s['mean_roughness_y']:>16.3e}"
            f"{s['mean_ise']:>16.3e}"
            f"{s['mean_ess_fraction']:>14.3f}"
        )


if __name__ == "__main__":
    main()