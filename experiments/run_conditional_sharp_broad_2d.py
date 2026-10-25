import os
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from nf.conditional_cubic_2d import ConditionalCubicFlow2D


# ============================================================
# Settings
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "results/conditional_sharp_broad_2d"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
SUMMARY_DIR = os.path.join(OUT_DIR, "summaries")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

SEED = 1234

# ------------------------------------------------------------
# Flow architecture
# ------------------------------------------------------------

N_BINS = 16
N_LAYERS = 6
HIDDEN = 128

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

N_COND = 512
N_X_PER_COND = 1024

STEPS = 10000
LR = 5e-5

# ------------------------------------------------------------
# Conditional parameter ranges
# ------------------------------------------------------------

MU_RANGE = (0.20, 0.80)
SHARP_SIGMA_RANGE = (0.015, 0.050)

# ------------------------------------------------------------
# Target definition
# ------------------------------------------------------------

FLOOR = 1e-6

A_SHARP = 1.0

BROAD_MU_X = 0.50
BROAD_MU_Y = 0.50
BROAD_SIGMA = 0.20
A_BROAD = 0.20

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

N_EVAL = 100000

# Heatmap resolution
N_GRID = 150

TEST_CONDITIONS = [
    # centered
    (0.50, 0.50, 0.030),

    # horizontal shift
    (0.30, 0.50, 0.030),

    # vertical shift
    (0.50, 0.70, 0.030),

    # diagonal shift
    (0.30, 0.70, 0.030),

    # narrowest edge
    (0.50, 0.50, 0.015),

    # broadest sharp component
    (0.50, 0.50, 0.050),

    # sharp peak separated from broad component
    (0.75, 0.25, 0.020),
]


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_sharp_broad_2d(
    xy,
    mu_x,
    mu_y,
    sigma_s,
):
    """
    2D target on [0,1]^2:

        floor
        + sharp conditional Gaussian
        + broad fixed Gaussian

    xy: (..., 2)

    mu_x, mu_y, sigma_s should be broadcastable to xy[...,0].
    """

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

    return FLOOR + sharp + broad


def gaussian_integral_1d(mu, sigma):
    def Phi(z):
        return 0.5 * (
            1.0
            + math.erf(z / math.sqrt(2.0))
        )

    lower = (0.0 - mu) / sigma
    upper = (1.0 - mu) / sigma

    return (
        sigma
        * math.sqrt(2.0 * math.pi)
        * (
            Phi(upper)
            - Phi(lower)
        )
    )


def gaussian_integral_2d(mu_x, mu_y, sigma):
    return (
        gaussian_integral_1d(mu_x, sigma)
        * gaussian_integral_1d(mu_y, sigma)
    )


def exact_integral_sharp_broad_2d(
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

    # Area of [0,1]^2 = 1
    floor_integral = FLOOR

    return (
        floor_integral
        + sharp_integral
        + broad_integral
    )


# ============================================================
# Conditional batch sampling
# ============================================================

def sample_grouped_batch():
    """
    Conditions:
        c = (mu_x, mu_y, sigma_s)

    Returns:
        xy_flat:   (N_COND*N_X_PER_COND, 2)
        cond_flat: (N_COND*N_X_PER_COND, 3)
    """

    mu_x = torch.empty(
        N_COND,
        1,
        device=DEVICE,
    ).uniform_(*MU_RANGE)

    mu_y = torch.empty(
        N_COND,
        1,
        device=DEVICE,
    ).uniform_(*MU_RANGE)

    sigma_s = torch.empty(
        N_COND,
        1,
        device=DEVICE,
    ).uniform_(*SHARP_SIGMA_RANGE)

    cond = torch.cat(
        [
            mu_x,
            mu_y,
            sigma_s,
        ],
        dim=1,
    )

    xy = torch.rand(
        N_COND,
        N_X_PER_COND,
        2,
        device=DEVICE,
    )

    cond_expanded = cond[:, None, :].expand(
        N_COND,
        N_X_PER_COND,
        3,
    )

    xy_flat = xy.reshape(-1, 2)
    cond_flat = cond_expanded.reshape(-1, 3)

    return xy_flat, cond_flat


# ============================================================
# Model
# ============================================================

def build_model():
    return ConditionalCubicFlow2D(
        cond_dim=3,
        hidden=HIDDEN,
        n_bins=N_BINS,
        n_layers=N_LAYERS,
    ).to(DEVICE)


# ============================================================
# Training
# ============================================================

def train_model():
    set_seed(SEED)

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loss_history = []

    print("\n" + "=" * 80)
    print("TRAINING 2D CONDITIONAL SHARP + BROAD BENCHMARK")
    print("=" * 80)

    for step in range(1, STEPS + 1):
        xy_flat, cond_flat = sample_grouped_batch()

        mu_x = cond_flat[:, 0]
        mu_y = cond_flat[:, 1]
        sigma_s = cond_flat[:, 2]

        f_vals = target_sharp_broad_2d(
            xy_flat,
            mu_x,
            mu_y,
            sigma_s,
        )

        log_q = model.log_prob(
            xy_flat,
            cond_flat,
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
            weights * log_q
        ).sum(dim=1) / (
            weights.sum(dim=1)
            + 1e-12
        )

        loss = loss_per_cond.mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        loss_history.append(
            {
                "step": step,
                "loss": float(
                    loss.detach().cpu()
                ),
            }
        )

        if step == 1 or step % 250 == 0:
            print(
                f"step {step:05d} | "
                f"loss={loss.item():.6f}"
            )

    ckpt_path = os.path.join(
        CKPT_DIR,
        "conditional_cubic_sharp_broad_2d.pt",
    )

    torch.save(
        model.state_dict(),
        ckpt_path,
    )

    with open(
        os.path.join(
            SUMMARY_DIR,
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
# Integral evaluation
# ============================================================

@torch.no_grad()
def evaluate_integral(
    model,
    mu_x,
    mu_y,
    sigma_s,
):
    cond = torch.tensor(
        [[mu_x, mu_y, sigma_s]],
        device=DEVICE,
    ).repeat(
        N_EVAL,
        1,
    )

    samples = model.sample(
        N_EVAL,
        cond,
    )

    xy = samples.reshape(
        -1,
        2,
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

    weights = (
        f_vals
        / (q + 1e-30)
    )

    estimate = weights.mean()

    weight_variance = weights.var(
        unbiased=True
    )

    variance = (
        weight_variance
        / N_EVAL
    )

    std_error = torch.sqrt(
        variance
    )

    integral_ref = (
        exact_integral_sharp_broad_2d(
            mu_x,
            mu_y,
            sigma_s,
        )
    )

    abs_error = torch.abs(
        estimate
        - integral_ref
    )

    rel_error = (
        abs_error
        / abs(integral_ref)
    )

    pull = (
        estimate
        - integral_ref
    ) / (
        std_error
        + 1e-30
    )

    ess = (
        weights.sum() ** 2
        / (
            torch.sum(
                weights ** 2
            )
            + 1e-30
        )
    )

    return {
        "mu_x": mu_x,
        "mu_y": mu_y,
        "sigma_s": sigma_s,

        "estimate": float(
            estimate.cpu()
        ),

        "integral_ref": float(
            integral_ref
        ),

        "weight_variance": float(
            weight_variance.cpu()
        ),

        "variance": float(
            variance.cpu()
        ),

        "std_error": float(
            std_error.cpu()
        ),

        "abs_error": float(
            abs_error.cpu()
        ),

        "rel_error": float(
            rel_error.cpu()
        ),

        "pull": float(
            pull.cpu()
        ),

        "ess": float(
            ess.cpu()
        ),

        "events": N_EVAL,
    }


# ============================================================
# Grid evaluation
# ============================================================

def make_grid():
    x = torch.linspace(
        0.0,
        1.0,
        N_GRID,
        device=DEVICE,
    )

    y = torch.linspace(
        0.0,
        1.0,
        N_GRID,
        device=DEVICE,
    )

    X, Y = torch.meshgrid(
        x,
        y,
        indexing="ij",
    )

    xy = torch.stack(
        [
            X.reshape(-1),
            Y.reshape(-1),
        ],
        dim=-1,
    )

    return X, Y, xy


@torch.no_grad()
def evaluate_density_grid(
    model,
    mu_x,
    mu_y,
    sigma_s,
):
    X, Y, xy = make_grid()

    cond = torch.tensor(
        [[mu_x, mu_y, sigma_s]],
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

    f = target_sharp_broad_2d(
        xy,
        cond[:, 0],
        cond[:, 1],
        cond[:, 2],
    )

    integral_ref = (
        exact_integral_sharp_broad_2d(
            mu_x,
            mu_y,
            sigma_s,
        )
    )

    p = (
        f
        / integral_ref
    )

    density_ise = torch.mean(
        (q - p) ** 2
    )

    density_mae = torch.mean(
        torch.abs(
            q - p
        )
    )

    p_max = torch.max(
        p
    )

    important = (
        p
        > 0.01 * p_max
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

    max_abs_log_ratio_important = (
        torch.max(
            torch.abs(
                log_ratio[
                    important
                ]
            )
        )
    )

    p_grid = p.reshape(
        N_GRID,
        N_GRID,
    )

    q_grid = q.reshape(
        N_GRID,
        N_GRID,
    )

    log_ratio_grid = (
        log_ratio.reshape(
            N_GRID,
            N_GRID,
        )
    )

    return {
        "X": X.detach().cpu().numpy(),
        "Y": Y.detach().cpu().numpy(),

        "p_grid": (
            p_grid.detach()
            .cpu()
            .numpy()
        ),

        "q_grid": (
            q_grid.detach()
            .cpu()
            .numpy()
        ),

        "log_ratio_grid": (
            log_ratio_grid
            .detach()
            .cpu()
            .numpy()
        ),

        "density_ise": float(
            density_ise.cpu()
        ),

        "density_mae": float(
            density_mae.cpu()
        ),

        "max_abs_log_ratio_important": float(
            max_abs_log_ratio_important
            .cpu()
        ),
    }


# ============================================================
# Plotting
# ============================================================

def condition_tag(
    mu_x,
    mu_y,
    sigma_s,
):
    return (
        f"mux{mu_x:.2f}_"
        f"muy{mu_y:.2f}_"
        f"sig{sigma_s:.3f}"
    )


def plot_heatmaps(
    grid_result,
    mu_x,
    mu_y,
    sigma_s,
):
    tag = condition_tag(
        mu_x,
        mu_y,
        sigma_s,
    )

    p_grid = grid_result["p_grid"]
    q_grid = grid_result["q_grid"]
    log_ratio_grid = (
        grid_result["log_ratio_grid"]
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

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        p_grid.T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    ax.set_title(
        f"Target $f/I_{{ref}}$\n"
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma_s={sigma_s:.3f}$"
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.colorbar(
        im,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            f"{tag}_target.png",
        ),
        dpi=170,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Learned proposal
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        q_grid.T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    ax.set_title(
        f"Learned $q_\\theta$\n"
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma_s={sigma_s:.3f}$"
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.colorbar(
        im,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            f"{tag}_proposal.png",
        ),
        dpi=170,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Log ratio
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        log_ratio_grid.T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    ax.set_title(
        f"$\\log p - \\log q_\\theta$\n"
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma_s={sigma_s:.3f}$"
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.colorbar(
        im,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            f"{tag}_logratio.png",
        ),
        dpi=170,
    )

    plt.close(fig)


# ============================================================
# One combined comparison figure per condition
# ============================================================

def plot_three_panel(
    grid_result,
    mu_x,
    mu_y,
    sigma_s,
):
    tag = condition_tag(
        mu_x,
        mu_y,
        sigma_s,
    )

    p_grid = grid_result["p_grid"]
    q_grid = grid_result["q_grid"]
    log_ratio_grid = (
        grid_result["log_ratio_grid"]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    extent = [
        0.0,
        1.0,
        0.0,
        1.0,
    ]

    im0 = axes[0].imshow(
        p_grid.T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[0].set_title(
        "Target $f/I_{ref}$"
    )

    fig.colorbar(
        im0,
        ax=axes[0],
    )

    im1 = axes[1].imshow(
        q_grid.T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[1].set_title(
        "Learned $q_\\theta$"
    )

    fig.colorbar(
        im1,
        ax=axes[1],
    )

    im2 = axes[2].imshow(
        log_ratio_grid.T,
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
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    fig.suptitle(
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma_s={sigma_s:.3f}$"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            f"{tag}_comparison.png",
        ),
        dpi=170,
    )

    plt.close(fig)


# ============================================================
# Evaluate all test conditions
# ============================================================

def evaluate_model(
    model,
):
    all_results = []

    for (
        mu_x,
        mu_y,
        sigma_s,
    ) in TEST_CONDITIONS:

        print(
            "\nEvaluating "
            f"mu_x={mu_x:.2f}, "
            f"mu_y={mu_y:.2f}, "
            f"sigma_s={sigma_s:.3f}"
        )

        result = evaluate_integral(
            model,
            mu_x,
            mu_y,
            sigma_s,
        )

        grid_result = (
            evaluate_density_grid(
                model,
                mu_x,
                mu_y,
                sigma_s,
            )
        )

        result[
            "density_ise"
        ] = grid_result[
            "density_ise"
        ]

        result[
            "density_mae"
        ] = grid_result[
            "density_mae"
        ]

        result[
            "max_abs_log_ratio_important"
        ] = grid_result[
            "max_abs_log_ratio_important"
        ]

        all_results.append(
            result
        )

        plot_heatmaps(
            grid_result,
            mu_x,
            mu_y,
            sigma_s,
        )

        plot_three_panel(
            grid_result,
            mu_x,
            mu_y,
            sigma_s,
        )

        print(
            f"estimate={result['estimate']:.6e} | "
            f"I_ref={result['integral_ref']:.6e} | "
            f"var={result['variance']:.3e} | "
            f"rel_err={result['rel_error']:.3e} | "
            f"ESS={result['ess']:.3e} | "
            f"ISE={result['density_ise']:.3e} | "
            f"maxlog="
            f"{result['max_abs_log_ratio_important']:.3e}"
        )

    return all_results


# ============================================================
# Contact sheet
# ============================================================

def make_contact_sheet():
    import matplotlib.image as mpimg

    paths = []

    for (
        mu_x,
        mu_y,
        sigma_s,
    ) in TEST_CONDITIONS:

        tag = condition_tag(
            mu_x,
            mu_y,
            sigma_s,
        )

        paths.append(
            os.path.join(
                PLOT_DIR,
                f"{tag}_comparison.png",
            )
        )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
    )

    axes = axes.flatten()

    for ax, path in zip(
        axes,
        paths,
    ):
        img = mpimg.imread(
            path
        )

        ax.imshow(
            img
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
        "Conditional Cubic Flow: Sharp Peak + Broad Background",
        fontsize=16,
    )

    fig.tight_layout()

    out_path = os.path.join(
        PLOT_DIR,
        "sharp_broad_2d_contact_sheet.png",
    )

    fig.savefig(
        out_path,
        dpi=170,
    )

    plt.close(fig)

    print(
        f"Saved contact sheet: "
        f"{out_path}"
    )


# ============================================================
# Summary
# ============================================================

def summarize_results(
    results,
):
    summary = {
        "n_bins": N_BINS,
        "n_layers": N_LAYERS,
        "hidden": HIDDEN,

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

        "mean_density_ise": float(
            np.mean(
                [
                    r["density_ise"]
                    for r in results
                ]
            )
        ),

        "mean_density_mae": float(
            np.mean(
                [
                    r["density_mae"]
                    for r in results
                ]
            )
        ),

        "mean_max_abs_log_ratio_important": float(
            np.mean(
                [
                    r[
                        "max_abs_log_ratio_important"
                    ]
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
    print(
        f"Using device: {DEVICE}"
    )

    print(
        f"Flow: cubic, "
        f"{N_BINS} bins, "
        f"{N_LAYERS} layers"
    )

    print(
        f"Condition: "
        f"(mu_x, mu_y, sigma_s)"
    )

    model = train_model()

    results = evaluate_model(
        model
    )

    make_contact_sheet()

    summary = summarize_results(
        results
    )

    results_path = os.path.join(
        SUMMARY_DIR,
        "sharp_broad_2d_results.json",
    )

    with open(
        results_path,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    summary_path = os.path.join(
        SUMMARY_DIR,
        "sharp_broad_2d_summary.json",
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("2D SHARP + BROAD SUMMARY")
    print("=" * 90)

    print(
        f"mean variance="
        f"{summary['mean_variance']:.3e}"
    )

    print(
        f"mean rel_err="
        f"{summary['mean_rel_error']:.3e}"
    )

    print(
        f"mean ESS="
        f"{summary['mean_ess']:.3e}"
    )

    print(
        f"mean ISE="
        f"{summary['mean_density_ise']:.3e}"
    )

    print(
        f"mean max log-ratio="
        f"{summary['mean_max_abs_log_ratio_important']:.3e}"
    )

    print("\nSaved:")
    print(
        f"  Results: {results_path}"
    )
    print(
        f"  Summary: {summary_path}"
    )
    print(
        f"  Plots:   {PLOT_DIR}"
    )


if __name__ == "__main__":
    main()