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
# Global settings
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234

OUT_DIR = "results/forward_objective_sweep_hard_2d"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# Training
# ============================================================

N_COND = 64
N_X_PER_COND = 512

STEPS = 2000
LR = 5e-5


# ============================================================
# Flow
# ============================================================

N_BINS = 16
HIDDEN = 128


# ============================================================
# Auxiliary spatial diagnostics
#
# 20 x 20 regions
# 4 x 4 quadrature points inside each region
#
# Total fixed grid = 80 x 80.
# ============================================================

REGION_GRID = 20
REGION_SUBGRID = 4

N_AUX_COND = 4

# Computing the auxiliary grid every step is expensive.
# Every 5 steps is enough for the first diagnostic.
AUX_EVERY = 5


# ============================================================
# Initial coefficients
#
# These are NOT sacred.
# The script prints each raw auxiliary loss so we can see
# whether rescaling is needed.
# ============================================================

LAMBDA_REGION = 10.0
LAMBDA_LOCAL = 0.10
LAMBDA_CHI2 = 0.10


# ============================================================
# Objective configurations
# ============================================================

OBJECTIVES = [
    {
        "name": "forward",
        "mode": "forward",
    },

    {
        "name": "forward_region",
        "mode": "region",
    },

    {
        "name": "forward_local",
        "mode": "local",
    },

    {
        "name": "forward_chi2",
        "mode": "chi2",
    },
]


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

# Return to the hard benchmark
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
# Exact reference integral
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
# Conditions
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
# Standard forward KL
# ============================================================

def forward_kl_loss(model):

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

    xy_flat = xy.reshape(
        -1,
        2,
    )

    cond_flat = cond_grouped.reshape(
        -1,
        3,
    )

    f = target_sharp_broad_2d(
        xy_flat,
        cond_flat[:, 0],
        cond_flat[:, 1],
        cond_flat[:, 2],
    )

    log_q = model.log_prob(
        xy_flat,
        cond_flat,
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
# Fixed midpoint grid
# ============================================================

def make_auxiliary_grid():

    resolution = (
        REGION_GRID
        * REGION_SUBGRID
    )

    axis = (
        torch.arange(
            resolution,
            dtype=torch.float32,
            device=DEVICE,
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


AUX_XY = make_auxiliary_grid()


# ============================================================
# Evaluate p and q on fixed grid
# ============================================================

def auxiliary_density_data(
    model,
):

    C = N_AUX_COND

    cond_small = sample_conditions(
        C
    )

    n_grid = AUX_XY.shape[0]

    xy = (
        AUX_XY[None, :, :]
        .expand(
            C,
            n_grid,
            2,
        )
    )

    cond = (
        cond_small[:, None, :]
        .expand(
            C,
            n_grid,
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

    # Target needs no model gradients.
    with torch.no_grad():

        f = target_sharp_broad_2d(
            xy_flat,
            cond_flat[:, 0],
            cond_flat[:, 1],
            cond_flat[:, 2],
        )

    # q MUST remain differentiable.
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

    log_f = torch.log(
        f + 1e-30
    )

    q = q.reshape(
        C,
        resolution,
        resolution,
    )

    log_q = log_q.reshape(
        C,
        resolution,
        resolution,
    )

    # --------------------------------------------------------
    # Numerically normalized target p.
    #
    # Grid covers unit square and has equal-area points,
    # so the mean approximates the integral.
    # --------------------------------------------------------

    f_integral = f.mean(
        dim=(1, 2),
        keepdim=True,
    )

    p = (
        f
        /
        (
            f_integral
            + 1e-30
        )
    )

    return {
        "f": f,
        "log_f": log_f,
        "p": p.detach(),
        "q": q,
        "log_q": log_q,
    }


# ============================================================
# Convert fine grid to coarse regional probability masses
# ============================================================

def coarse_region_mass(
    density,
):

    C = density.shape[0]

    G = REGION_GRID
    S = REGION_SUBGRID

    density = density.reshape(
        C,
        G,
        S,
        G,
        S,
    )

    masses = density.sum(
        dim=(2, 4)
    )

    masses = masses.reshape(
        C,
        G * G,
    )

    masses = (
        masses
        /
        (
            masses.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-30
        )
    )

    return masses


# ============================================================
# Objective B:
# Regional target/proposal mass matching
# ============================================================

def regional_mass_loss(
    data,
):

    P = coarse_region_mass(
        data["p"]
    ).detach()

    Q = coarse_region_mass(
        data["q"]
    )

    eps = 1e-12

    # Regional forward KL
    loss_per_cond = (
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
        loss_per_cond.mean()
    )


# ============================================================
# Objective C:
# Local log-ratio variance
# ============================================================

def local_logratio_variance_loss(
    data,
):
    """
    r = log f - log q

    Since

        log p - log q
        =
        log f - log q - log I,

    subtracting the within-region mean removes the unknown
    constant log I automatically.

    Regions are weighted by target regional mass.
    """

    C = data["q"].shape[0]

    G = REGION_GRID
    S = REGION_SUBGRID

    log_ratio = (
        data["log_f"]
        -
        data["log_q"]
    )

    # Shape:
    #
    # C, region_x, sub_x, region_y, sub_y
    log_ratio = log_ratio.reshape(
        C,
        G,
        S,
        G,
        S,
    )

    # Rearrange:
    #
    # C, region_x, region_y, sub_x, sub_y
    log_ratio = log_ratio.permute(
        0,
        1,
        3,
        2,
        4,
    )

    # Flatten points inside each region
    log_ratio = log_ratio.reshape(
        C,
        G,
        G,
        S * S,
    )

    local_mean = log_ratio.mean(
        dim=-1,
        keepdim=True,
    )

    local_variance = (
        (
            log_ratio
            - local_mean
        ) ** 2
    ).mean(
        dim=-1
    )

    # Target regional mass
    P = coarse_region_mass(
        data["p"]
    ).detach()

    P = P.reshape(
        C,
        G,
        G,
    )

    loss_per_cond = (
        P
        * local_variance
    ).sum(
        dim=(1, 2)
    )

    return (
        loss_per_cond.mean()
    )


# ============================================================
# Objective D:
# Chi-square / importance-variance objective
# ============================================================

def chi2_loss(
    data,
):
    """
    Approximate

        integral p(x)^2 / q(x) dx

    over the unit square.

    This equals

        1 + chi^2(p || q)

    and directly controls normalized IS variance.
    """

    p = data["p"]
    q = data["q"]

    # Avoid catastrophic numerical overflow while still
    # heavily penalizing under-covered target regions.
    q_safe = torch.clamp(
        q,
        min=1e-8,
    )

    integrand = (
        p ** 2
        /
        q_safe
    )

    # Unit square, equally spaced midpoint quadrature:
    #
    # integral ≈ mean(grid values)
    loss_per_cond = integrand.mean(
        dim=(1, 2)
    )

    return (
        loss_per_cond.mean()
    )


# ============================================================
# Train one objective
# ============================================================

def train_one_objective(
    config,
):

    name = config["name"]
    mode = config["mode"]

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

    last_aux = None

    for step in range(
        1,
        STEPS + 1,
    ):

        loss_forward = (
            forward_kl_loss(
                model
            )
        )

        loss_aux = torch.zeros(
            (),
            device=DEVICE,
        )

        # ----------------------------------------------------
        # Only calculate expensive fixed-grid term every
        # AUX_EVERY optimization steps.
        # ----------------------------------------------------

        if (
            mode != "forward"
            and step % AUX_EVERY == 0
        ):

            data = auxiliary_density_data(
                model
            )

            if mode == "region":

                loss_aux = regional_mass_loss(
                    data
                )

                coefficient = (
                    LAMBDA_REGION
                )

            elif mode == "local":

                loss_aux = (
                    local_logratio_variance_loss(
                        data
                    )
                )

                coefficient = (
                    LAMBDA_LOCAL
                )

            elif mode == "chi2":

                loss_aux = chi2_loss(
                    data
                )

                coefficient = (
                    LAMBDA_CHI2
                )

            else:

                raise ValueError(
                    f"Unknown mode: {mode}"
                )

            loss = (
                loss_forward
                +
                coefficient
                * loss_aux
            )

            last_aux = float(
                loss_aux.detach().cpu()
            )

        else:

            # On non-auxiliary steps we train with FKL only.
            loss = loss_forward

            coefficient = 0.0

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
                max_norm=10.0,
            )
        )

        optimizer.step()

        record = {
            "step":
                step,

            "total":
                float(
                    loss.detach().cpu()
                ),

            "forward":
                float(
                    loss_forward.detach().cpu()
                ),

            "aux":
                last_aux,

            "grad_norm":
                float(
                    grad_norm.detach().cpu()
                ),
        }

        history.append(
            record
        )

        if (
            step == 1
            or step % 100 == 0
        ):

            print(
                f"{name:16s} | "
                f"step={step:04d} | "
                f"total={loss.item():.6f} | "
                f"fwd={loss_forward.item():.6f} | "
                f"aux={last_aux} | "
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
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_integral(
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
            q
            + 1e-30
        )
    )

    estimate = weights.mean()

    weight_var = weights.var(
        unbiased=True
    )

    variance = (
        weight_var
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
# Grid diagnostics
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
        0.01 * p.max()
    )

    maxlog = torch.max(
        torch.abs(
            logratio[
                important
            ]
        )
    )

    return {
        "p":
            p.reshape(
                N_GRID,
                N_GRID,
            )
            .cpu()
            .numpy(),

        "q":
            q.reshape(
                N_GRID,
                N_GRID,
            )
            .cpu()
            .numpy(),

        "logratio":
            logratio.reshape(
                N_GRID,
                N_GRID,
            )
            .cpu()
            .numpy(),

        "ise":
            float(
                ise.cpu()
            ),

        "mae":
            float(
                mae.cpu()
            ),

        "maxlog":
            float(
                maxlog.cpu()
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
# Plot
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
    name,
    condition,
    grid,
):

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

    arrays = [
        grid["p"],
        grid["q"],
        grid["logratio"],
    ]

    titles = [
        "Target $p$",
        f"{name}: $q_\\theta$",
        "$\\log p-\\log q$",
    ]

    for ax, arr, title in zip(
        axes,
        arrays,
        titles,
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

    mu_x, mu_y, sigma = condition

    fig.suptitle(
        f"{name} | "
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma={sigma:.3f}$"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            f"{name}_"
            + condition_tag(
                condition
            )
            + ".png",
        ),
        dpi=160,
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
        f"sigma range="
        f"{SHARP_SIGMA_RANGE}"
    )

    print(
        f"regions="
        f"{REGION_GRID}x{REGION_GRID}"
    )

    print(
        f"auxiliary grid="
        f"{REGION_GRID * REGION_SUBGRID}"
        f"x"
        f"{REGION_GRID * REGION_SUBGRID}"
    )

    models = {}

    all_results = {}
    summaries = {}

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for config in OBJECTIVES:

        models[
            config["name"]
        ] = train_one_objective(
            config
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for config in OBJECTIVES:

        name = config[
            "name"
        ]

        model = models[
            name
        ]

        results = []

        print()
        print(
            "=" * 100
        )
        print(
            f"EVALUATING {name}"
        )
        print(
            "=" * 100
        )

        for condition in TEST_CONDITIONS:

            integral_result = evaluate_integral(
                model,
                condition,
            )

            grid_result = evaluate_grid(
                model,
                condition,
            )

            plot_condition(
                name,
                condition,
                grid_result,
            )

            result = {
                "mu_x":
                    condition[0],

                "mu_y":
                    condition[1],

                "sigma":
                    condition[2],

                **integral_result,

                "ise":
                    grid_result["ise"],

                "mae":
                    grid_result["mae"],

                "maxlog":
                    grid_result["maxlog"],

                "pmax":
                    grid_result["pmax"],

                "qmax":
                    grid_result["qmax"],
            }

            results.append(
                result
            )

            print(
                f"c={condition} | "
                f"var={result['variance']:.3e} | "
                f"rel={result['rel_error']:.3e} | "
                f"ESS={result['ess']:.1f} "
                f"({result['ess_fraction']:.3f}) | "
                f"ISE={result['ise']:.3e} | "
                f"maxlog={result['maxlog']:.3f} | "
                f"pmax={result['pmax']:.2f} | "
                f"qmax={result['qmax']:.2f}"
            )

        all_results[
            name
        ] = results

        summaries[
            name
        ] = {
            "mean_variance":
                float(
                    np.mean(
                        [
                            r["variance"]
                            for r in results
                        ]
                    )
                ),

            "mean_rel_error":
                float(
                    np.mean(
                        [
                            r["rel_error"]
                            for r in results
                        ]
                    )
                ),

            "mean_ess":
                float(
                    np.mean(
                        [
                            r["ess"]
                            for r in results
                        ]
                    )
                ),

            "mean_ess_fraction":
                float(
                    np.mean(
                        [
                            r["ess_fraction"]
                            for r in results
                        ]
                    )
                ),

            "mean_ise":
                float(
                    np.mean(
                        [
                            r["ise"]
                            for r in results
                        ]
                    )
                ),

            "mean_maxlog":
                float(
                    np.mean(
                        [
                            r["maxlog"]
                            for r in results
                        ]
                    )
                ),
        }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "config": {
            "n_cond":
                N_COND,

            "n_x_per_cond":
                N_X_PER_COND,

            "steps":
                STEPS,

            "region_grid":
                REGION_GRID,

            "region_subgrid":
                REGION_SUBGRID,

            "lambda_region":
                LAMBDA_REGION,

            "lambda_local":
                LAMBDA_LOCAL,

            "lambda_chi2":
                LAMBDA_CHI2,

            "sigma_range":
                SHARP_SIGMA_RANGE,
        },

        "summaries":
            summaries,

        "results":
            all_results,
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

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    print()
    print(
        "=" * 105
    )

    print(
        "FORWARD OBJECTIVE SWEEP"
    )

    print(
        "=" * 105
    )

    print(
        f"{'objective':<20}"
        f"{'variance':>16}"
        f"{'rel err':>14}"
        f"{'ESS':>14}"
        f"{'ESS frac':>12}"
        f"{'ISE':>14}"
        f"{'maxlog':>12}"
    )

    print(
        "-" * 105
    )

    for config in OBJECTIVES:

        name = config[
            "name"
        ]

        s = summaries[
            name
        ]

        print(
            f"{name:<20}"
            f"{s['mean_variance']:>16.3e}"
            f"{s['mean_rel_error']:>14.3e}"
            f"{s['mean_ess']:>14.1f}"
            f"{s['mean_ess_fraction']:>12.3f}"
            f"{s['mean_ise']:>14.3e}"
            f"{s['mean_maxlog']:>12.3f}"
        )


if __name__ == "__main__":
    main()