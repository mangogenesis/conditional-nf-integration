import os
import json
import math
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from nf.conditional_cubic_2d import ConditionalCubicFlow2D


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 1234

OUT_DIR = "results/conditional_sharp_broad_2d_fast"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# FAST DEVELOPMENT SETTINGS
# ============================================================

N_COND = 128
N_X_PER_COND = 256

STEPS = 2000
LR = 5e-5

N_EVAL = 20000
N_GRID = 100


# ============================================================
# Flow
# ============================================================

N_BINS = 16
N_LAYERS = 6
HIDDEN = 128


# ============================================================
# Target
# ============================================================

FLOOR = 1e-6

A_SHARP = 1.0

# easier sharp peak
SHARP_SIGMA_RANGE = (0.04, 0.08)

MU_RANGE = (0.20, 0.80)

# fixed broad component
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

    # separated sharp peak
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
# Target function
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

    return FLOOR + sharp + broad


# ============================================================
# Exact integral
# ============================================================

def gaussian_integral_1d(
    mu,
    sigma,
):
    def Phi(z):
        return 0.5 * (
            1.0
            + math.erf(
                z / math.sqrt(2.0)
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
# Training batch
# ============================================================

def sample_grouped_batch():

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

    sigma = torch.empty(
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
            sigma,
        ],
        dim=-1,
    )

    xy = torch.rand(
        N_COND,
        N_X_PER_COND,
        2,
        device=DEVICE,
    )

    cond_grouped = cond[:, None, :].expand(
        N_COND,
        N_X_PER_COND,
        3,
    )

    return (
        xy.reshape(-1, 2),
        cond_grouped.reshape(-1, 3),
    )


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

def train():

    set_seed(SEED)

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    history = []

    print("=" * 80)
    print("FAST 2D SHARP + BROAD DIAGNOSTIC")
    print("=" * 80)

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

    for step in range(
        1,
        STEPS + 1,
    ):

        xy, cond = (
            sample_grouped_batch()
        )

        f = target_sharp_broad_2d(
            xy,
            cond[:, 0],
            cond[:, 1],
            cond[:, 2],
        )

        log_q = model.log_prob(
            xy,
            cond,
        )

        f = f.reshape(
            N_COND,
            N_X_PER_COND,
        )

        log_q = log_q.reshape(
            N_COND,
            N_X_PER_COND,
        )

        # ----------------------------------------------------
        # Forward-KL-style grouped conditional loss
        # ----------------------------------------------------

        weights = f.detach()

        numerator = (
            weights * log_q
        ).sum(
            dim=1
        )

        denominator = (
            weights.sum(dim=1)
            + 1e-12
        )

        loss_per_condition = (
            -numerator / denominator
        )

        loss = (
            loss_per_condition.mean()
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )
        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            10.0,
        )

        optimizer.step()

        history.append(
            float(loss.detach().cpu())
        )

        if (
            step == 1
            or step % 100 == 0
        ):
            print(
                f"step={step:04d} | "
                f"loss={loss.item():.6f}"
            )

    torch.save(
        model.state_dict(),
        os.path.join(
            CKPT_DIR,
            "model.pt",
        ),
    )

    return model, history


# ============================================================
# Importance-sampling evaluation
# ============================================================

@torch.no_grad()
def evaluate_integral(
    model,
    condition,
):

    mu_x, mu_y, sigma = (
        condition
    )

    cond = torch.tensor(
        [[mu_x, mu_y, sigma]],
        dtype=torch.float32,
        device=DEVICE,
    ).repeat(
        N_EVAL,
        1,
    )

    samples = model.sample(
        N_EVAL,
        cond,
    )

    log_q = model.log_prob(
        samples,
        cond,
    )

    q = torch.exp(
        log_q
    )

    f = target_sharp_broad_2d(
        samples,
        cond[:, 0],
        cond[:, 1],
        cond[:, 2],
    )

    w = (
        f
        / (q + 1e-30)
    )

    estimate = (
        w.mean()
    )

    weight_var = w.var(
        unbiased=True
    )

    estimator_var = (
        weight_var
        / N_EVAL
    )

    ess = (
        w.sum() ** 2
        / (
            torch.sum(w ** 2)
            + 1e-30
        )
    )

    ref = exact_integral(
        mu_x,
        mu_y,
        sigma,
    )

    rel_error = (
        abs(
            estimate.item()
            - ref
        )
        / abs(ref)
    )

    return {
        "estimate": float(
            estimate.cpu()
        ),

        "reference": float(
            ref
        ),

        "variance": float(
            estimator_var.cpu()
        ),

        "ess": float(
            ess.cpu()
        ),

        "rel_error": float(
            rel_error
        ),
    }


# ============================================================
# Density grids
# ============================================================

@torch.no_grad()
def evaluate_grid(
    model,
    condition,
):

    mu_x, mu_y, sigma = (
        condition
    )

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
        [[mu_x, mu_y, sigma]],
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

    f = target_sharp_broad_2d(
        xy,
        cond[:, 0],
        cond[:, 1],
        cond[:, 2],
    )

    ref = exact_integral(
        mu_x,
        mu_y,
        sigma,
    )

    p = (
        f / ref
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

    ise = torch.mean(
        (p - q) ** 2
    )

    important = (
        p
        >
        0.01 * p.max()
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

        "maxlog": float(
            maxlog.cpu()
        ),
    }


# ============================================================
# Plot one condition
# ============================================================

def plot_condition(
    condition,
    grid,
):

    mu_x, mu_y, sigma = (
        condition
    )

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

    im = axes[0].imshow(
        grid["p"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[0].set_title(
        "Target $p=f/I$"
    )

    fig.colorbar(
        im,
        ax=axes[0],
    )

    im = axes[1].imshow(
        grid["q"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[1].set_title(
        "Learned $q_\\theta$"
    )

    fig.colorbar(
        im,
        ax=axes[1],
    )

    im = axes[2].imshow(
        grid["logratio"].T,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    axes[2].set_title(
        "$\\log p-\\log q$"
    )

    fig.colorbar(
        im,
        ax=axes[2],
    )

    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    fig.suptitle(
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma={sigma:.3f}$"
    )

    fig.tight_layout()

    filename = (
        f"mux{mu_x:.2f}_"
        f"muy{mu_y:.2f}_"
        f"sig{sigma:.3f}.png"
    )

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            filename,
        ),
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    model, history = train()

    all_results = []

    print("\n")
    print("=" * 100)
    print("EVALUATION")
    print("=" * 100)

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
            "sigma": condition[2],

            **integral_result,

            "ise": grid_result[
                "ise"
            ],

            "maxlog": grid_result[
                "maxlog"
            ],
        }

        all_results.append(
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

    summary = {
        "mean_variance": float(
            np.mean([
                x["variance"]
                for x in all_results
            ])
        ),

        "mean_rel_error": float(
            np.mean([
                x["rel_error"]
                for x in all_results
            ])
        ),

        "mean_ess": float(
            np.mean([
                x["ess"]
                for x in all_results
            ])
        ),

        "mean_ise": float(
            np.mean([
                x["ise"]
                for x in all_results
            ])
        ),

        "mean_maxlog": float(
            np.mean([
                x["maxlog"]
                for x in all_results
            ])
        ),
    }

    output = {
        "config": {
            "n_cond": N_COND,
            "n_x_per_cond": N_X_PER_COND,
            "steps": STEPS,
            "n_bins": N_BINS,
            "n_layers": N_LAYERS,
            "hidden": HIDDEN,
            "sigma_range": (
                SHARP_SIGMA_RANGE
            ),
        },

        "summary": summary,

        "results": all_results,
    }

    with open(
        os.path.join(
            OUT_DIR,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n")
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

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
        f"mean maxlog   : "
        f"{summary['mean_maxlog']:.3f}"
    )


if __name__ == "__main__":
    main()