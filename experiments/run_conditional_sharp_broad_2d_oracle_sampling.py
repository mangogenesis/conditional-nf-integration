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

OUT_DIR = "results/oracle_sampling_hard_2d"
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
# Oracle sampling
# ============================================================

# r(x|c)
# =
# alpha * Uniform
# +
# (1-alpha) * truncated local Gaussian

ORACLE_UNIFORM_ALPHA = 0.25

# Local proposal standard deviation:
#
# sigma_oracle = ORACLE_SIGMA_MULT * sigma_target
ORACLE_SIGMA_MULT = 2.0


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

MU_MIN = 0.20
MU_MAX = 0.80

SIGMA_MIN = 0.015
SIGMA_MAX = 0.050

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

    # Most important diagnostic
    (0.75, 0.25, 0.020),
]


# ============================================================
# Two controlled configurations
# ============================================================

CONFIGS = [
    {
        "name": "uniform",
        "sampling": "uniform",
    },

    {
        "name": "oracle",
        "sampling": "oracle",
    },
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
# Normal distribution helpers
# ============================================================

SQRT2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def normal_cdf(z):

    return (
        0.5
        * (
            1.0
            + torch.erf(
                z / SQRT2
            )
        )
    )


def normal_icdf(p):

    # protect erfinv near exactly ±1
    p = torch.clamp(
        p,
        min=1e-7,
        max=1.0 - 1e-7,
    )

    return (
        SQRT2
        * torch.erfinv(
            2.0 * p - 1.0
        )
    )


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

    sharp = (
        A_SHARP
        * torch.exp(
            -0.5
            * (
                (x - mu_x)
                / sigma_s
            ) ** 2

            -0.5
            * (
                (y - mu_y)
                / sigma_s
            ) ** 2
        )
    )

    broad = (
        A_BROAD
        * torch.exp(
            -0.5
            * (
                (x - BROAD_MU_X)
                / BROAD_SIGMA
            ) ** 2

            -0.5
            * (
                (y - BROAD_MU_Y)
                / BROAD_SIGMA
            ) ** 2
        )
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
                    z / math.sqrt(2.0)
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


# ============================================================
# Normalize condition for the neural network
# ============================================================

def normalize_condition(cond_raw):

    cond = cond_raw.clone()

    cond[..., 0] = (
        cond[..., 0]
        - MU_MIN
    ) / (
        MU_MAX
        - MU_MIN
    )

    cond[..., 1] = (
        cond[..., 1]
        - MU_MIN
    ) / (
        MU_MAX
        - MU_MIN
    )

    cond[..., 2] = (
        cond[..., 2]
        - SIGMA_MIN
    ) / (
        SIGMA_MAX
        - SIGMA_MIN
    )

    return cond


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
# Truncated Gaussian density in 1D
# ============================================================

def truncated_normal_log_prob(
    x,
    mu,
    sigma,
):
    """
    Density of:

        N(mu, sigma^2)

    truncated to [0, 1].
    """

    z = (
        x - mu
    ) / sigma

    log_base = (
        -0.5 * z ** 2
        - torch.log(
            sigma
        )
        - math.log(
            SQRT_2PI
        )
    )

    lower_z = (
        0.0 - mu
    ) / sigma

    upper_z = (
        1.0 - mu
    ) / sigma

    Z = (
        normal_cdf(
            upper_z
        )
        -
        normal_cdf(
            lower_z
        )
    )

    return (
        log_base
        -
        torch.log(
            Z + 1e-30
        )
    )


# ============================================================
# Exact truncated Gaussian sampling
# ============================================================

def sample_truncated_normal(
    mu,
    sigma,
):
    """
    mu, sigma shape:
        (...,)

    Returns exact samples from truncated normal on [0,1].
    """

    lower_z = (
        -mu / sigma
    )

    upper_z = (
        (1.0 - mu)
        / sigma
    )

    lower_cdf = normal_cdf(
        lower_z
    )

    upper_cdf = normal_cdf(
        upper_z
    )

    u = torch.rand_like(
        mu
    )

    p = (
        lower_cdf
        +
        u
        * (
            upper_cdf
            - lower_cdf
        )
    )

    z = normal_icdf(
        p
    )

    x = (
        mu
        + sigma * z
    )

    return torch.clamp(
        x,
        min=0.0,
        max=1.0,
    )


# ============================================================
# Oracle proposal density
# ============================================================

def oracle_log_density(
    xy,
    cond_raw,
):
    """
    r(x|c)
      =
      alpha * Uniform([0,1]^2)
      +
      (1-alpha) * G_local(x|c)

    Uniform density on unit square is exactly 1.
    """

    mu_x = cond_raw[..., 0]
    mu_y = cond_raw[..., 1]

    sigma_local = (
        ORACLE_SIGMA_MULT
        * cond_raw[..., 2]
    )

    log_gx = truncated_normal_log_prob(
        xy[..., 0],
        mu_x,
        sigma_local,
    )

    log_gy = truncated_normal_log_prob(
        xy[..., 1],
        mu_y,
        sigma_local,
    )

    log_g = (
        log_gx
        + log_gy
    )

    log_uniform_component = (
        math.log(
            ORACLE_UNIFORM_ALPHA
        )
        +
        torch.zeros_like(
            log_g
        )
    )

    log_local_component = (
        math.log(
            1.0
            - ORACLE_UNIFORM_ALPHA
        )
        +
        log_g
    )

    return torch.logsumexp(
        torch.stack(
            [
                log_uniform_component,
                log_local_component,
            ],
            dim=0,
        ),
        dim=0,
    )


# ============================================================
# Draw oracle-mixture training samples
# ============================================================

def sample_oracle_points(
    cond_raw,
):
    """
    cond_raw:
        (N_COND, N_X_PER_COND, 3)

    Returns
        xy
        log r(x|c)
        is_local mask
    """

    mu_x = cond_raw[..., 0]
    mu_y = cond_raw[..., 1]

    sigma_local = (
        ORACLE_SIGMA_MULT
        * cond_raw[..., 2]
    )

    # --------------------------------------------------------
    # Draw both candidate types
    # --------------------------------------------------------

    xy_uniform = torch.rand(
        *cond_raw.shape[:-1],
        2,
        device=DEVICE,
    )

    x_local = sample_truncated_normal(
        mu_x,
        sigma_local,
    )

    y_local = sample_truncated_normal(
        mu_y,
        sigma_local,
    )

    xy_local = torch.stack(
        [
            x_local,
            y_local,
        ],
        dim=-1,
    )

    # --------------------------------------------------------
    # Choose mixture component
    # --------------------------------------------------------

    choose_local = (
        torch.rand(
            *cond_raw.shape[:-1],
            device=DEVICE,
        )
        >
        ORACLE_UNIFORM_ALPHA
    )

    xy = torch.where(
        choose_local[..., None],
        xy_local,
        xy_uniform,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Evaluate FULL mixture density,
    # not only density of selected component.
    # --------------------------------------------------------

    log_r = oracle_log_density(
        xy,
        cond_raw,
    )

    return (
        xy,
        log_r,
        choose_local,
    )


# ============================================================
# Uniform forward-KL baseline
# ============================================================

def uniform_forward_loss(
    model,
):

    cond_raw = sample_conditions(
        N_COND
    )

    cond_raw_grouped = (
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

    cond_raw_flat = (
        cond_raw_grouped.reshape(
            -1,
            3,
        )
    )

    cond_model_flat = (
        normalize_condition(
            cond_raw_flat
        )
    )

    f = target_sharp_broad_2d(
        xy_flat,
        cond_raw_flat[:, 0],
        cond_raw_flat[:, 1],
        cond_raw_flat[:, 2],
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

    # effective number of target-side weighted points
    train_ess = (
        weights.sum(
            dim=1
        ) ** 2
        /
        (
            torch.sum(
                weights ** 2,
                dim=1,
            )
            + 1e-30
        )
    )

    return (
        loss_per_cond.mean(),
        train_ess.mean(),
    )


# ============================================================
# Oracle importance-corrected forward KL
# ============================================================

def oracle_forward_loss(
    model,
):

    cond_raw = sample_conditions(
        N_COND
    )

    cond_raw_grouped = (
        cond_raw[:, None, :]
        .expand(
            N_COND,
            N_X_PER_COND,
            3,
        )
    )

    (
        xy,
        log_r,
        choose_local,
    ) = sample_oracle_points(
        cond_raw_grouped
    )

    xy_flat = xy.reshape(
        -1,
        2,
    )

    cond_raw_flat = (
        cond_raw_grouped.reshape(
            -1,
            3,
        )
    )

    cond_model_flat = (
        normalize_condition(
            cond_raw_flat
        )
    )

    f = target_sharp_broad_2d(
        xy_flat,
        cond_raw_flat[:, 0],
        cond_raw_flat[:, 1],
        cond_raw_flat[:, 2],
    )

    log_f = torch.log(
        f + 1e-30
    )

    log_r_flat = log_r.reshape(
        -1
    )

    # --------------------------------------------------------
    # Importance correction:
    #
    # target-side weight = f / r
    # --------------------------------------------------------

    log_weight = (
        log_f
        -
        log_r_flat
    )

    # stabilization is harmless because normalization occurs
    # separately for each condition below.
    log_weight = log_weight.reshape(
        N_COND,
        N_X_PER_COND,
    )

    max_log_weight = (
        log_weight.max(
            dim=1,
            keepdim=True,
        ).values
    )

    weights = torch.exp(
        log_weight
        -
        max_log_weight
    ).detach()

    log_q = model.log_prob(
        xy_flat,
        cond_model_flat,
    ).reshape(
        N_COND,
        N_X_PER_COND,
    )

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

    train_ess = (
        weights.sum(
            dim=1
        ) ** 2
        /
        (
            torch.sum(
                weights ** 2,
                dim=1,
            )
            + 1e-30
        )
    )

    local_fraction = (
        choose_local.float().mean()
    )

    return (
        loss_per_cond.mean(),
        train_ess.mean(),
        local_fraction,
    )


# ============================================================
# Train
# ============================================================

def train_model(
    config,
):

    name = config["name"]
    sampling = config["sampling"]

    print()
    print("=" * 100)
    print(
        f"TRAINING: {name}"
    )
    print("=" * 100)

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

        if sampling == "uniform":

            (
                loss,
                train_ess,
            ) = uniform_forward_loss(
                model
            )

            local_fraction = None

        elif sampling == "oracle":

            (
                loss,
                train_ess,
                local_fraction,
            ) = oracle_forward_loss(
                model
            )

        else:

            raise ValueError(
                sampling
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

        record = {
            "step":
                step,

            "loss":
                float(
                    loss.detach().cpu()
                ),

            "train_weight_ess":
                float(
                    train_ess.detach().cpu()
                ),

            "grad_norm":
                float(
                    grad_norm.detach().cpu()
                ),

            "local_fraction":
                (
                    None
                    if local_fraction is None
                    else float(
                        local_fraction.detach().cpu()
                    )
                ),
        }

        history.append(
            record
        )

        if (
            step == 1
            or step % 100 == 0
        ):

            message = (
                f"{name:10s} | "
                f"step={step:04d} | "
                f"loss={loss.item():.6f} | "
                f"trainESS={train_ess.item():.1f}"
                f"/{N_X_PER_COND} | "
                f"grad={float(grad_norm):.3f}"
            )

            if local_fraction is not None:

                message += (
                    f" | local="
                    f"{local_fraction.item():.3f}"
                )

            print(
                message
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

    log_q = model.log_prob(
        samples,
        cond_model,
    )

    q = torch.exp(
        log_q
    )

    f = target_sharp_broad_2d(
        samples,
        cond_raw[:, 0],
        cond_raw[:, 1],
        cond_raw[:, 2],
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

    f = target_sharp_broad_2d(
        xy,
        cond_raw[:, 0],
        cond_raw[:, 1],
        cond_raw[:, 2],
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
        0.01
        * p.max()
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

    extent = [
        0,
        1,
        0,
        1,
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
        f"sigma range="
        f"({SIGMA_MIN}, {SIGMA_MAX})"
    )

    print(
        f"oracle uniform alpha="
        f"{ORACLE_UNIFORM_ALPHA}"
    )

    print(
        f"oracle sigma multiplier="
        f"{ORACLE_SIGMA_MULT}"
    )

    models = {}

    all_results = {}
    summaries = {}

    # --------------------------------------------------------
    # Train both models
    # --------------------------------------------------------

    for config in CONFIGS:

        models[
            config["name"]
        ] = train_model(
            config
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for config in CONFIGS:

        name = config["name"]

        model = models[
            name
        ]

        print()
        print("=" * 100)
        print(
            f"EVALUATING: {name}"
        )
        print("=" * 100)

        results = []

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
        "settings": {
            "n_cond":
                N_COND,

            "n_x_per_cond":
                N_X_PER_COND,

            "steps":
                STEPS,

            "oracle_uniform_alpha":
                ORACLE_UNIFORM_ALPHA,

            "oracle_sigma_mult":
                ORACLE_SIGMA_MULT,

            "condition_normalized":
                True,
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
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "ORACLE-SAMPLING CONTROL"
    )
    print("=" * 100)

    print(
        f"{'model':<14}"
        f"{'variance':>16}"
        f"{'rel err':>14}"
        f"{'ESS':>14}"
        f"{'ESS frac':>14}"
        f"{'ISE':>14}"
        f"{'maxlog':>12}"
    )

    print("-" * 100)

    for config in CONFIGS:

        name = config["name"]

        s = summaries[
            name
        ]

        print(
            f"{name:<14}"
            f"{s['mean_variance']:>16.3e}"
            f"{s['mean_rel_error']:>14.3e}"
            f"{s['mean_ess']:>14.1f}"
            f"{s['mean_ess_fraction']:>14.3f}"
            f"{s['mean_ise']:>14.3e}"
            f"{s['mean_maxlog']:>12.3f}"
        )


if __name__ == "__main__":
    main()