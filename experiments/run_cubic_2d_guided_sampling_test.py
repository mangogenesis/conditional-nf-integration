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

OUT_DIR = "results/cubic_2d_guided_sampling_test"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# Flow
# ============================================================

N_BINS = 32
N_LAYERS = 4
HIDDEN = 128


# ============================================================
# Training
# ============================================================

N_COND = 64
N_X_PER_COND = 1024

STEPS = 3000
LR = 5e-5


# ============================================================
# Evaluation
# ============================================================

N_EVAL = 100000
N_GRID = 160


# ============================================================
# Target
# ============================================================

FLOOR = 1e-6

MU_MIN = 0.20
MU_MAX = 0.80

SIGMA_MIN = 0.03
SIGMA_MAX = 0.08

A_SHARP = 1.0

A_BROAD = 0.20
BROAD_MU_X = 0.50
BROAD_MU_Y = 0.50
BROAD_SIGMA = 0.20


# ============================================================
# Guided sampler mixture weights
# ============================================================

# Single Gaussian:
#
# 95% near target peak
# 5% uniform safety/exploration

SINGLE_LOCAL_WEIGHT = 0.95
SINGLE_UNIFORM_WEIGHT = 0.05


# Sharp + broad:
#
# 45% sharp
# 45% broad
# 10% uniform

MIX_SHARP_WEIGHT = 0.45
MIX_BROAD_WEIGHT = 0.45
MIX_UNIFORM_WEIGHT = 0.10


# ============================================================
# Configurations
# ============================================================

CONFIGS = [
    {
        "name": "single_uniform",
        "target": "single",
        "sampling": "uniform",
    },
    {
        "name": "single_guided",
        "target": "single",
        "sampling": "guided",
    },
    {
        "name": "broad_uniform",
        "target": "broad",
        "sampling": "uniform",
    },
    {
        "name": "broad_guided",
        "target": "broad",
        "sampling": "guided",
    },
]


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
# Normal helpers
# ============================================================

SQRT2 = math.sqrt(2.0)
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def normal_cdf(z):

    return 0.5 * (
        1.0
        + torch.erf(
            z / SQRT2
        )
    )


def normal_icdf(p):

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

def target_2d(
    xy,
    cond_raw,
    target_type,
):

    x = xy[..., 0]
    y = xy[..., 1]

    mu_x = cond_raw[..., 0]
    mu_y = cond_raw[..., 1]
    sigma = cond_raw[..., 2]

    sharp = (
        A_SHARP
        * torch.exp(
            -0.5
            * ((x - mu_x) / sigma) ** 2
            -0.5
            * ((y - mu_y) / sigma) ** 2
        )
    )

    result = (
        FLOOR
        + sharp
    )

    if target_type == "broad":

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

        result = (
            result
            + broad
        )

    return result


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
    condition,
    target_type,
):

    mu_x, mu_y, sigma = condition

    value = (
        FLOOR
        +
        A_SHARP
        * gaussian_integral_2d(
            mu_x,
            mu_y,
            sigma,
        )
    )

    if target_type == "broad":

        value += (
            A_BROAD
            * gaussian_integral_2d(
                BROAD_MU_X,
                BROAD_MU_Y,
                BROAD_SIGMA,
            )
        )

    return value


# ============================================================
# Truncated Gaussian
# ============================================================

def sample_truncated_normal(
    mu,
    sigma,
):

    lo = (
        -mu / sigma
    )

    hi = (
        (1.0 - mu)
        / sigma
    )

    cdf_lo = normal_cdf(
        lo
    )

    cdf_hi = normal_cdf(
        hi
    )

    u = torch.rand_like(
        mu
    )

    p = (
        cdf_lo
        +
        u
        * (
            cdf_hi
            - cdf_lo
        )
    )

    z = normal_icdf(
        p
    )

    return torch.clamp(
        mu + sigma * z,
        0.0,
        1.0,
    )


def truncated_normal_log_prob(
    x,
    mu,
    sigma,
):

    z = (
        x - mu
    ) / sigma

    log_base = (
        -0.5 * z ** 2
        - torch.log(
            sigma
        )
        - LOG_SQRT_2PI
    )

    lo = (
        -mu / sigma
    )

    hi = (
        (1.0 - mu)
        / sigma
    )

    Z = (
        normal_cdf(hi)
        -
        normal_cdf(lo)
    )

    return (
        log_base
        -
        torch.log(
            Z + 1e-30
        )
    )


def gaussian2d_log_prob(
    xy,
    mu_x,
    mu_y,
    sigma,
):

    return (
        truncated_normal_log_prob(
            xy[..., 0],
            mu_x,
            sigma,
        )
        +
        truncated_normal_log_prob(
            xy[..., 1],
            mu_y,
            sigma,
        )
    )


# ============================================================
# Guided proposal density
# ============================================================

def guided_log_density(
    xy,
    cond_raw,
    target_type,
):

    mu_x = cond_raw[..., 0]
    mu_y = cond_raw[..., 1]
    sigma = cond_raw[..., 2]

    log_sharp = gaussian2d_log_prob(
        xy,
        mu_x,
        mu_y,
        sigma,
    )

    # Uniform density on [0,1]^2 = 1
    log_uniform = torch.zeros_like(
        log_sharp
    )

    if target_type == "single":

        components = torch.stack(
            [
                math.log(
                    SINGLE_LOCAL_WEIGHT
                )
                + log_sharp,

                math.log(
                    SINGLE_UNIFORM_WEIGHT
                )
                + log_uniform,
            ],
            dim=0,
        )

    elif target_type == "broad":

        broad_mu_x = torch.full_like(
            mu_x,
            BROAD_MU_X,
        )

        broad_mu_y = torch.full_like(
            mu_y,
            BROAD_MU_Y,
        )

        broad_sigma = torch.full_like(
            sigma,
            BROAD_SIGMA,
        )

        log_broad = gaussian2d_log_prob(
            xy,
            broad_mu_x,
            broad_mu_y,
            broad_sigma,
        )

        components = torch.stack(
            [
                math.log(
                    MIX_SHARP_WEIGHT
                )
                + log_sharp,

                math.log(
                    MIX_BROAD_WEIGHT
                )
                + log_broad,

                math.log(
                    MIX_UNIFORM_WEIGHT
                )
                + log_uniform,
            ],
            dim=0,
        )

    else:

        raise ValueError(
            target_type
        )

    return torch.logsumexp(
        components,
        dim=0,
    )


# ============================================================
# Guided sampling
# ============================================================

def sample_guided(
    cond_raw,
    target_type,
):

    mu_x = cond_raw[..., 0]
    mu_y = cond_raw[..., 1]
    sigma = cond_raw[..., 2]

    shape = cond_raw.shape[:-1]

    # --------------------------------------------------------
    # Sharp samples
    # --------------------------------------------------------

    sharp_x = sample_truncated_normal(
        mu_x,
        sigma,
    )

    sharp_y = sample_truncated_normal(
        mu_y,
        sigma,
    )

    sharp_xy = torch.stack(
        [
            sharp_x,
            sharp_y,
        ],
        dim=-1,
    )

    # --------------------------------------------------------
    # Uniform samples
    # --------------------------------------------------------

    uniform_xy = torch.rand(
        *shape,
        2,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Single Gaussian case
    # --------------------------------------------------------

    if target_type == "single":

        r = torch.rand(
            *shape,
            device=DEVICE,
        )

        choose_sharp = (
            r
            < SINGLE_LOCAL_WEIGHT
        )

        xy = torch.where(
            choose_sharp[..., None],
            sharp_xy,
            uniform_xy,
        )

    # --------------------------------------------------------
    # Sharp + broad case
    # --------------------------------------------------------

    elif target_type == "broad":

        broad_mu_x = torch.full_like(
            mu_x,
            BROAD_MU_X,
        )

        broad_mu_y = torch.full_like(
            mu_y,
            BROAD_MU_Y,
        )

        broad_sigma = torch.full_like(
            sigma,
            BROAD_SIGMA,
        )

        broad_x = sample_truncated_normal(
            broad_mu_x,
            broad_sigma,
        )

        broad_y = sample_truncated_normal(
            broad_mu_y,
            broad_sigma,
        )

        broad_xy = torch.stack(
            [
                broad_x,
                broad_y,
            ],
            dim=-1,
        )

        r = torch.rand(
            *shape,
            device=DEVICE,
        )

        sharp_cut = (
            MIX_SHARP_WEIGHT
        )

        broad_cut = (
            MIX_SHARP_WEIGHT
            +
            MIX_BROAD_WEIGHT
        )

        choose_sharp = (
            r < sharp_cut
        )

        choose_broad = (
            (r >= sharp_cut)
            &
            (r < broad_cut)
        )

        xy = uniform_xy

        xy = torch.where(
            choose_broad[..., None],
            broad_xy,
            xy,
        )

        xy = torch.where(
            choose_sharp[..., None],
            sharp_xy,
            xy,
        )

    else:

        raise ValueError(
            target_type
        )

    # Full mixture density
    log_r = guided_log_density(
        xy,
        cond_raw,
        target_type,
    )

    return (
        xy,
        log_r,
    )


# ============================================================
# Build flow
# ============================================================

def build_model():

    return ConditionalCubicFlow2D(
        cond_dim=3,
        hidden=HIDDEN,
        n_bins=N_BINS,
        n_layers=N_LAYERS,
    ).to(
        DEVICE
    )


# ============================================================
# Forward-KL loss
# ============================================================

def forward_loss(
    model,
    target_type,
    sampling,
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

    if sampling == "uniform":

        xy = torch.rand(
            N_COND,
            N_X_PER_COND,
            2,
            device=DEVICE,
        )

        # Uniform density on unit square = 1
        log_r = torch.zeros(
            N_COND,
            N_X_PER_COND,
            device=DEVICE,
        )

    elif sampling == "guided":

        (
            xy,
            log_r,
        ) = sample_guided(
            cond_grouped,
            target_type,
        )

    else:

        raise ValueError(
            sampling
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

    f = target_2d(
        xy_flat,
        cond_raw_flat,
        target_type,
    )

    log_f = torch.log(
        f + 1e-30
    )

    log_weight = (
        log_f
        -
        log_r.reshape(-1)
    )

    log_weight = log_weight.reshape(
        N_COND,
        N_X_PER_COND,
    )

    # Stabilization done independently per condition
    log_weight = (
        log_weight
        -
        log_weight.max(
            dim=1,
            keepdim=True,
        ).values
    )

    weights = torch.exp(
        log_weight
    ).detach()

    log_q = model.log_prob(
        xy_flat,
        cond_model_flat,
    ).reshape(
        N_COND,
        N_X_PER_COND,
    )

    loss_per_condition = -(
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

    return (
        loss_per_condition.mean(),
        train_ess.mean(),
    )


# ============================================================
# Train
# ============================================================

def train_model(
    config,
):

    name = config["name"]

    set_seed(
        SEED
    )

    print()
    print("=" * 100)
    print(
        f"TRAINING {name}"
    )
    print("=" * 100)

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

        loss, train_ess = forward_loss(
            model,
            config["target"],
            config["sampling"],
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

                "train_ess":
                    float(
                        train_ess.detach().cpu()
                    ),

                "grad":
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
                f"{name:20s} | "
                f"step={step:04d} | "
                f"loss={loss.item():.6f} | "
                f"trainESS="
                f"{train_ess.item():.1f}/"
                f"{N_X_PER_COND} | "
                f"grad={float(grad_norm):.3f}"
            )

    torch.save(
        model.state_dict(),
        os.path.join(
            CKPT_DIR,
            f"{name}.pt",
        ),
    )

    return model


# ============================================================
# Grid evaluation
# ============================================================

@torch.no_grad()
def evaluate_grid(
    model,
    condition,
    target_type,
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

    q = torch.exp(
        model.log_prob(
            xy,
            cond_model,
        )
    )

    f = target_2d(
        xy,
        cond_raw,
        target_type,
    )

    I = exact_integral(
        condition,
        target_type,
    )

    p = (
        f / I
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

    target_peak = xy[
        torch.argmax(p)
    ]

    proposal_peak = xy[
        torch.argmax(q)
    ]

    peak_error = torch.linalg.vector_norm(
        proposal_peak
        -
        target_peak
    )

    ise = torch.mean(
        (p - q) ** 2
    )

    return {
        "p":
            p.reshape(
                N_GRID,
                N_GRID,
            ).cpu().numpy(),

        "q":
            q.reshape(
                N_GRID,
                N_GRID,
            ).cpu().numpy(),

        "logratio":
            logratio.reshape(
                N_GRID,
                N_GRID,
            ).cpu().numpy(),

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
# Importance sampling evaluation
# ============================================================

@torch.no_grad()
def evaluate_mc(
    model,
    condition,
    target_type,
):

    mu_x, mu_y, sigma = condition

    cond_raw = torch.tensor(
        [[
            mu_x,
            mu_y,
            sigma,
        ]],
        device=DEVICE,
        dtype=torch.float32,
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

    f = target_2d(
        samples,
        cond_raw,
        target_type,
    )

    w = (
        f
        /
        (
            q + 1e-30
        )
    )

    variance = (
        w.var(
            unbiased=True
        )
        /
        N_EVAL
    )

    ess = (
        w.sum() ** 2
        /
        (
            torch.sum(
                w ** 2
            )
            + 1e-30
        )
    )

    return {
        "variance":
            float(
                variance.cpu()
            ),

        "ess_fraction":
            float(
                ess.cpu()
                / N_EVAL
            ),
    }


# ============================================================
# Plot
# ============================================================

def plot_result(
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
        "Flow $q$",
        "$\\log p - \\log q$",
    ]

    for ax, arr, title in zip(
        axes,
        arrays,
        titles,
    ):

        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=[0, 1, 0, 1],
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

    axes[1].scatter(
        grid["target_peak_x"],
        grid["target_peak_y"],
        marker="x",
        s=80,
        label="target peak",
    )

    axes[1].scatter(
        grid["proposal_peak_x"],
        grid["proposal_peak_y"],
        marker="+",
        s=100,
        label="flow peak",
    )

    axes[1].legend()

    mu_x, mu_y, sigma = condition

    fig.suptitle(
        f"{name} | "
        f"mu=({mu_x:.2f},{mu_y:.2f}), "
        f"sigma={sigma:.3f} | "
        f"peak error={grid['peak_error']:.3f}"
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
            f"{name}_{tag}.png",
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
        f"bins={N_BINS}, "
        f"layers={N_LAYERS}"
    )

    models = {}
    all_results = {}

    # --------------------------------------------------------
    # Train
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

        results = []

        print()
        print("=" * 120)
        print(
            f"EVALUATING {name}"
        )
        print("=" * 120)

        for condition in TEST_CONDITIONS:

            grid = evaluate_grid(
                model,
                condition,
                config["target"],
            )

            mc = evaluate_mc(
                model,
                condition,
                config["target"],
            )

            plot_result(
                name,
                condition,
                grid,
            )

            result = {
                "condition":
                    condition,

                **grid,
                **mc,
            }

            # Don't put images into JSON
            result.pop(
                "p"
            )
            result.pop(
                "q"
            )
            result.pop(
                "logratio"
            )

            results.append(
                result
            )

            print(
                f"c={condition} | "
                f"peak err={grid['peak_error']:.4f} | "
                f"qmax={grid['qmax']:.2f} | "
                f"ISE={grid['ise']:.3e} | "
                f"ESS={mc['ess_fraction']:.3f}"
            )

        all_results[
            name
        ] = results

    with open(
        os.path.join(
            OUT_DIR,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()