import os
import json
import math
import random

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from nf.conditional_cubic_1d import ConditionalCubicFlow1D


# ============================================================
# General
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234

OUT_DIR = "results/product_cubic_2d_control"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# Flow / training
# ============================================================

N_BINS = 32
HIDDEN = 128

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

N_GRID = 200
N_EVAL = 100000

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
# Reproducibility
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
        n, 1, device=DEVICE
    ).uniform_(
        MU_MIN,
        MU_MAX,
    )

    mu_y = torch.empty(
        n, 1, device=DEVICE
    ).uniform_(
        MU_MIN,
        MU_MAX,
    )

    sigma = torch.empty(
        n, 1, device=DEVICE
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


def normalize_mu(mu):

    return (
        mu - MU_MIN
    ) / (
        MU_MAX - MU_MIN
    )


def normalize_sigma(sigma):

    return (
        sigma - SIGMA_MIN
    ) / (
        SIGMA_MAX - SIGMA_MIN
    )


def make_x_condition(cond_raw):

    mu_x = cond_raw[..., 0:1]
    sigma = cond_raw[..., 2:3]

    return torch.cat(
        [
            normalize_mu(mu_x),
            normalize_sigma(sigma),
        ],
        dim=-1,
    )


def make_y_condition(cond_raw):

    mu_y = cond_raw[..., 1:2]
    sigma = cond_raw[..., 2:3]

    return torch.cat(
        [
            normalize_mu(mu_y),
            normalize_sigma(sigma),
        ],
        dim=-1,
    )


# ============================================================
# 2D single Gaussian target
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

    return (
        FLOOR
        +
        torch.exp(
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
                z
                / math.sqrt(2.0)
            )
        )

    lo = -mu / sigma

    hi = (
        1.0 - mu
    ) / sigma

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
# Product flow
# ============================================================

class IndependentProductCubicFlow2D(
    nn.Module
):

    def __init__(
        self,
        hidden=128,
        n_bins=32,
    ):
        super().__init__()

        self.flow_x = ConditionalCubicFlow1D(
            cond_dim=2,
            hidden=hidden,
            n_bins=n_bins,
        )

        self.flow_y = ConditionalCubicFlow1D(
            cond_dim=2,
            hidden=hidden,
            n_bins=n_bins,
        )

    def forward(
        self,
        u,
        cond_raw,
    ):

        cond_x = make_x_condition(
            cond_raw
        )

        cond_y = make_y_condition(
            cond_raw
        )

        ux = u[:, 0]
        uy = u[:, 1]

        x, logdet_x = (
            self.flow_x.forward(
                ux,
                cond_x,
            )
        )

        y, logdet_y = (
            self.flow_y.forward(
                uy,
                cond_y,
            )
        )

        x = x.reshape(-1)
        y = y.reshape(-1)

        logdet_x = logdet_x.reshape(-1)
        logdet_y = logdet_y.reshape(-1)

        xy = torch.stack(
            [
                x,
                y,
            ],
            dim=-1,
        )

        logdet = (
            logdet_x
            +
            logdet_y
        )

        return xy, logdet

    def inverse(
        self,
        xy,
        cond_raw,
    ):

        cond_x = make_x_condition(
            cond_raw
        )

        cond_y = make_y_condition(
            cond_raw
        )

        x = xy[:, 0]
        y = xy[:, 1]

        ux, logdet_x = (
            self.flow_x.inverse(
                x,
                cond_x,
            )
        )

        uy, logdet_y = (
            self.flow_y.inverse(
                y,
                cond_y,
            )
        )

        ux = ux.reshape(-1)
        uy = uy.reshape(-1)

        logdet_x = logdet_x.reshape(-1)
        logdet_y = logdet_y.reshape(-1)

        u = torch.stack(
            [
                ux,
                uy,
            ],
            dim=-1,
        )

        # Your 1D inverse returns
        # log |dx/du|, not log |du/dx|.
        forward_logdet = (
            logdet_x
            +
            logdet_y
        )

        return (
            u,
            forward_logdet,
        )

    def log_prob(
        self,
        xy,
        cond_raw,
    ):

        _, forward_logdet = (
            self.inverse(
                xy,
                cond_raw,
            )
        )

        # Uniform base on [0,1]^2:
        #
        # log q(x,y)
        # =
        # - log |det dx/du|
        return (
            -forward_logdet
        )

    def sample(
        self,
        n,
        cond_raw,
    ):

        u = torch.rand(
            n,
            2,
            device=cond_raw.device,
            dtype=cond_raw.dtype,
        )

        xy, _ = self.forward(
            u,
            cond_raw,
        )

        return xy


# ============================================================
# Build model
# ============================================================

def build_model():

    return (
        IndependentProductCubicFlow2D(
            hidden=HIDDEN,
            n_bins=N_BINS,
        )
        .to(DEVICE)
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

    cond_flat = (
        cond_grouped.reshape(
            -1,
            3,
        )
    )

    f = target_single_gaussian(
        xy_flat,
        cond_flat,
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
# Train
# ============================================================

def train_model():

    set_seed(SEED)

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    print("=" * 100)
    print(
        "TRAINING INDEPENDENT PRODUCT CUBIC FLOW"
    )
    print("=" * 100)

    history = []

    for step in range(
        1,
        STEPS + 1,
    ):

        loss = forward_kl_loss(
            model
        )

        if not torch.isfinite(loss):

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
                f"step={step:04d} | "
                f"loss={loss.item():.6f} | "
                f"grad={float(grad_norm):.3f}"
            )

    torch.save(
        model.state_dict(),
        os.path.join(
            CKPT_DIR,
            "product_cubic_2d.pt",
        ),
    )

    with open(
        os.path.join(
            OUT_DIR,
            "history.json",
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
# Stripe metric
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


def smooth_2d(arr):

    kernel = gaussian_kernel_1d()

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


def stripe_metrics(
    q_grid,
):

    smooth = smooth_2d(
        q_grid
    )

    residual = (
        q_grid
        - smooth
    )

    score = (
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

    dq_dx = np.diff(
        q_grid,
        axis=0,
    )

    dq_dy = np.diff(
        q_grid,
        axis=1,
    )

    scale = (
        np.mean(
            q_grid ** 2
        )
        + 1e-30
    )

    rough_x = (
        np.mean(
            dq_dx ** 2
        )
        / scale
    )

    rough_y = (
        np.mean(
            dq_dy ** 2
        )
        / scale
    )

    return {
        "stripe_score":
            float(score),

        "roughness_x":
            float(rough_x),

        "roughness_y":
            float(rough_y),

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

    log_q = model.log_prob(
        xy,
        cond_raw,
    )

    q = torch.exp(
        log_q
    )

    f = target_single_gaussian(
        xy,
        cond_raw,
    )

    reference = exact_integral(
        condition
    )

    p = (
        f / reference
    )

    target_peak = xy[
        torch.argmax(p)
    ]

    proposal_peak = xy[
        torch.argmax(q)
    ]

    peak_error = (
        torch.linalg.vector_norm(
            target_peak
            -
            proposal_peak
        )
    )

    ise = torch.mean(
        (p - q) ** 2
    )

    p_grid = (
        p.reshape(
            N_GRID,
            N_GRID,
        )
        .cpu()
        .numpy()
    )

    q_grid = (
        q.reshape(
            N_GRID,
            N_GRID,
        )
        .cpu()
        .numpy()
    )

    stripe = stripe_metrics(
        q_grid
    )

    return {
        "p":
            p_grid,

        "q":
            q_grid,

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
# Monte Carlo evaluation
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

    samples = model.sample(
        N_EVAL,
        cond_raw,
    )

    log_q = model.log_prob(
        samples,
        cond_raw,
    )

    q = torch.exp(
        log_q
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
            float(reference),

        "variance":
            float(
                variance.cpu()
            ),

        "rel_error":
            float(rel_error),

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
        grid["residual"],
    ]

    titles = [
        "Target $p$",
        "Independent product $q$",
        "$q - \\mathrm{smooth}(q)$",
    ]

    for ax, arr, title in zip(
        axes,
        arrays,
        titles,
    ):

        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=[
                0,
                1,
                0,
                1,
            ],
            aspect="equal",
        )

        fig.colorbar(
            im,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

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
        label="q peak",
    )

    axes[1].legend()

    mu_x, mu_y, sigma = condition

    fig.suptitle(
        f"mu=({mu_x:.2f},{mu_y:.2f}), "
        f"sigma={sigma:.3f} | "
        f"stripe={grid['stripe_score']:.4f}"
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
            f"{tag}.png",
        ),
        dpi=170,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"device={DEVICE}"
    )

    print(
        f"bins={N_BINS}"
    )

    model = train_model()

    results = []

    print()
    print("=" * 115)
    print(
        "INDEPENDENT PRODUCT CUBIC CONTROL"
    )
    print("=" * 115)

    for condition in TEST_CONDITIONS:

        grid = evaluate_grid(
            model,
            condition,
        )

        mc = evaluate_mc(
            model,
            condition,
        )

        plot_result(
            condition,
            grid,
        )

        result = {
            "condition":
                list(condition),

            "peak_error":
                grid["peak_error"],

            "stripe_score":
                grid["stripe_score"],

            "roughness_x":
                grid["roughness_x"],

            "roughness_y":
                grid["roughness_y"],

            "ise":
                grid["ise"],

            "pmax":
                grid["pmax"],

            "qmax":
                grid["qmax"],

            **mc,
        }

        results.append(
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

    summary = {
        "mean_peak_error":
            float(
                np.mean(
                    [
                        r["peak_error"]
                        for r in results
                    ]
                )
            ),

        "mean_stripe_score":
            float(
                np.mean(
                    [
                        r["stripe_score"]
                        for r in results
                    ]
                )
            ),

        "mean_roughness_x":
            float(
                np.mean(
                    [
                        r["roughness_x"]
                        for r in results
                    ]
                )
            ),

        "mean_roughness_y":
            float(
                np.mean(
                    [
                        r["roughness_y"]
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

        "mean_ess_fraction":
            float(
                np.mean(
                    [
                        r["ess_fraction"]
                        for r in results
                    ]
                )
            ),
    }

    with open(
        os.path.join(
            OUT_DIR,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            {
                "summary":
                    summary,

                "results":
                    results,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 90)
    print(
        "SUMMARY"
    )
    print("=" * 90)

    print(
        f"mean peak error = "
        f"{summary['mean_peak_error']:.4f}"
    )

    print(
        f"mean stripe     = "
        f"{summary['mean_stripe_score']:.4f}"
    )

    print(
        f"mean rough X    = "
        f"{summary['mean_roughness_x']:.3e}"
    )

    print(
        f"mean rough Y    = "
        f"{summary['mean_roughness_y']:.3e}"
    )

    print(
        f"mean ISE        = "
        f"{summary['mean_ise']:.3e}"
    )

    print(
        f"mean ESS frac   = "
        f"{summary['mean_ess_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()