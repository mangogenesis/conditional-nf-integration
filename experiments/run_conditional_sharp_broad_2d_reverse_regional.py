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
# General
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234

OUT_DIR = "results/reverse_regional_hard_2d"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# Training
# ============================================================

N_COND = 128
N_X_PER_COND = 512

STEPS = 5000
LR = 5e-5


# ============================================================
# Regional loss
# ============================================================

# 10 x 10 coarse regions
REGION_GRID = 20

# Each coarse region gets 8 x 8 quadrature points.
# Total density evaluation grid = 80 x 80.
REGION_SUBGRID = 8

# Regional loss only needs a few conditions per step.
N_REGION_COND = 4

# Starting value.
LAMBDA_REGION = 100.0

REGION_EVERY = 1


# ============================================================
# Flow
# ============================================================

N_BINS = 16
HIDDEN = 128


# ============================================================
# Evaluation
# ============================================================

N_EVAL = 100000
N_GRID = 160


# ============================================================
# Target
# ============================================================

FLOOR = 1e-6

A_SHARP = 1.0

MU_RANGE = (0.20, 0.80)

SHARP_SIGMA_RANGE = (
    0.015,
    0.050,
)

A_BROAD = 0.20

BROAD_MU_X = 0.50
BROAD_MU_Y = 0.50
BROAD_SIGMA = 0.20


TEST_CONDITIONS = [
    (0.50, 0.50, 0.030),
    (0.30, 0.50, 0.030),
    (0.50, 0.70, 0.030),
    (0.30, 0.70, 0.030),

    (0.50, 0.50, 0.015),
    (0.50, 0.50, 0.050),

    (0.75, 0.25, 0.020),
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

    a = (
        0.0 - mu
    ) / sigma

    b = (
        1.0 - mu
    ) / sigma

    return (
        sigma
        * math.sqrt(2.0 * math.pi)
        * (
            Phi(b)
            - Phi(a)
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
# Reverse KL
# ============================================================

def reverse_kl_loss(
    model,
):

    cond = sample_conditions(
        N_COND
    )

    cond = (
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

    u = torch.rand(
        N_COND * N_X_PER_COND,
        2,
        device=DEVICE,
    )

    # Cheap direction:
    #
    # u -> x
    x, logdet = model.forward(
        u,
        cond,
    )

    log_q = (
        -logdet
    )

    f = target_sharp_broad_2d(
        x,
        cond[:, 0],
        cond[:, 1],
        cond[:, 2],
    )

    log_f = torch.log(
        f + 1e-30
    )

    return (
        log_q
        - log_f
    ).mean()


# ============================================================
# Fixed regional quadrature grid
# ============================================================

def make_regional_grid():

    resolution = (
        REGION_GRID
        * REGION_SUBGRID
    )

    # Midpoint quadrature rather than including boundaries.
    axis = (
        torch.arange(
            resolution,
            device=DEVICE,
            dtype=torch.float32,
        )
        + 0.5
    ) / resolution

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

    return xy


REGIONAL_XY = make_regional_grid()


# ============================================================
# Convert fine grid values into coarse-region masses
# ============================================================

def coarse_region_masses(
    density,
    n_cond,
):
    """
    density:
        shape
        (n_cond, resolution, resolution)

    Returns
        (n_cond, REGION_GRID**2)

    Every coarse cell contains
        REGION_SUBGRID x REGION_SUBGRID
    fine-grid points.
    """

    G = REGION_GRID
    S = REGION_SUBGRID

    # C, G, S, G, S
    density = density.reshape(
        n_cond,
        G,
        S,
        G,
        S,
    )

    region_mass = density.sum(
        dim=(2, 4)
    )

    region_mass = region_mass.reshape(
        n_cond,
        G * G,
    )

    # Normalize because only relative regional mass matters.
    region_mass = (
        region_mass
        /
        (
            region_mass.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-30
        )
    )

    return region_mass


# ============================================================
# Regional forward KL
# ============================================================

def regional_loss(
    model,
):
    """
    Compute

        D_KL(P_region || Q_region)

    on a deterministic grid.

    Crucially, q is evaluated at FIXED grid points.

    Therefore even a region that receives no samples from q
    can still produce a gradient.
    """

    C = N_REGION_COND

    cond_small = sample_conditions(
        C
    )

    n_grid_points = (
        REGIONAL_XY.shape[0]
    )

    xy = (
        REGIONAL_XY[None, :, :]
        .expand(
            C,
            n_grid_points,
            2,
        )
    )

    cond = (
        cond_small[:, None, :]
        .expand(
            C,
            n_grid_points,
            3,
        )
    )

    xy_flat = xy.reshape(
        -1,
        2,
    )

    cond_flat = cond.reshape(
        -1,
        3,
    )

    # --------------------------------------------------------
    # Target density
    #
    # No gradient is needed through target.
    # --------------------------------------------------------

    with torch.no_grad():

        f = target_sharp_broad_2d(
            xy_flat,
            cond_flat[:, 0],
            cond_flat[:, 1],
            cond_flat[:, 2],
        )

    # --------------------------------------------------------
    # Proposal density
    #
    # IMPORTANT:
    # This must remain differentiable wrt model parameters.
    # --------------------------------------------------------

    log_q = model.log_prob(
        xy_flat,
        cond_flat,
    )

    q = torch.exp(
        log_q
    )

    resolution = (
        REGION_GRID
        * REGION_SUBGRID
    )

    f = f.reshape(
        C,
        resolution,
        resolution,
    )

    q = q.reshape(
        C,
        resolution,
        resolution,
    )

    P = coarse_region_masses(
        f,
        C,
    )

    Q = coarse_region_masses(
        q,
        C,
    )

    # Target regional probabilities don't need gradients.
    P = P.detach()

    eps = 1e-12

    # --------------------------------------------------------
    # Forward KL at the REGIONAL level
    #
    # sum P log(P/Q)
    # --------------------------------------------------------

    kl_per_condition = (
        P
        * (
            torch.log(
                P + eps
            )
            -
            torch.log(
                Q + eps
            )
        )
    ).sum(
        dim=1
    )

    return (
        kl_per_condition.mean(),
        P,
        Q,
    )


# ============================================================
# Training
# ============================================================

def train_model():

    set_seed(
        SEED
    )

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    history = []

    print(
        "=" * 100
    )
    print(
        "REVERSE KL + REGIONAL MASS MATCHING"
    )
    print(
        "=" * 100
    )

    print(
        f"device={DEVICE}"
    )

    print(
        f"sigma={SHARP_SIGMA_RANGE}"
    )

    print(
        f"regional grid="
        f"{REGION_GRID}x{REGION_GRID}"
    )

    print(
        f"regional quadrature="
        f"{REGION_GRID*REGION_SUBGRID}"
        f"x"
        f"{REGION_GRID*REGION_SUBGRID}"
    )

    print(
        f"lambda_region="
        f"{LAMBDA_REGION}"
    )

    for step in range(
        1,
        STEPS + 1,
    ):

        loss_reverse = (
            reverse_kl_loss(
                model
            )
        )

        loss_region, _, _ = (
            regional_loss(
                model
            )
        )

        loss = (
            loss_reverse
            +
            LAMBDA_REGION
            * loss_region
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss "
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

                "reverse":
                    float(
                        loss_reverse.detach().cpu()
                    ),

                "region":
                    float(
                        loss_region.detach().cpu()
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
                f"step={step:05d} | "
                f"total={loss.item():.6f} | "
                f"reverse={loss_reverse.item():.6f} | "
                f"region={loss_region.item():.6f} | "
                f"grad={float(grad_norm):.3f}"
            )

    torch.save(
        model.state_dict(),
        os.path.join(
            CKPT_DIR,
            "reverse_regional.pt",
        ),
    )

    with open(
        os.path.join(
            OUT_DIR,
            "loss_history.json",
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
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_condition(
    model,
    condition,
):

    mu_x, mu_y, sigma = condition

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

    ref = exact_integral(
        mu_x,
        mu_y,
        sigma,
    )

    rel = (
        abs(
            estimate.item()
            - ref
        )
        /
        abs(ref)
    )

    return {
        "estimate":
            float(
                estimate.cpu()
            ),

        "reference":
            float(
                ref
            ),

        "variance":
            float(
                variance.cpu()
            ),

        "rel_error":
            float(
                rel
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
# Plot density
# ============================================================

@torch.no_grad()
def plot_condition(
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

    q = torch.exp(
        model.log_prob(
            xy,
            cond,
        )
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

    logratio = (
        torch.log(
            p + 1e-30
        )
        -
        torch.log(
            q + 1e-30
        )
    )

    p = p.reshape(
        N_GRID,
        N_GRID,
    ).cpu().numpy()

    q = q.reshape(
        N_GRID,
        N_GRID,
    ).cpu().numpy()

    logratio = logratio.reshape(
        N_GRID,
        N_GRID,
    ).cpu().numpy()

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

    for ax, arr, title in zip(
        axes,
        [
            p,
            q,
            logratio,
        ],
        [
            "Target $p$",
            "Reverse + regional $q$",
            "$\\log p-\\log q$",
        ],
    ):

        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=extent,
            aspect="equal",
        )

        fig.colorbar(
            im,
            ax=ax,
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "$x$"
        )

        ax.set_ylabel(
            "$y$"
        )

    fig.suptitle(
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma={sigma:.3f}$"
    )

    fig.tight_layout()

    tag = (
        f"mux{mu_x:.2f}_"
        f"muy{mu_y:.2f}_"
        f"sig{sigma:.3f}"
    )

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            tag + ".png",
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

    model = train_model()

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

        result = evaluate_condition(
            model,
            condition,
        )

        result.update(
            {
                "mu_x":
                    condition[0],

                "mu_y":
                    condition[1],

                "sigma":
                    condition[2],
            }
        )

        results.append(
            result
        )

        plot_condition(
            model,
            condition,
        )

        print(
            f"c={condition} | "
            f"var={result['variance']:.3e} | "
            f"rel={result['rel_error']:.3e} | "
            f"ESS={result['ess']:.1f} "
            f"({result['ess_fraction']:.3f})"
        )

    with open(
        os.path.join(
            OUT_DIR,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()