# run_copula_benchmark_sweep.py

import os
import csv
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from nf.conditional_cubic_1d import ConditionalCubicFlow1D
from nf.conditional_cubic_2d import ConditionalCubicFlow2D


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 1234

RESULT_DIR = "results/copula_benchmark_sweep"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# Training configuration
# ============================================================

N_DATA = 100_000
BATCH_SIZE = 4096

HIDDEN = 128
N_BINS = 32

MARGINAL_STEPS = 2500
COPULA_STEPS = 2500
COUPLING_STEPS = 3000

LR_MARGINAL = 1e-4
LR_COPULA = 1e-4
LR_COUPLING = 5e-5

COUPLING_LAYERS = 4

GRAD_CLIP = 10.0


# ============================================================
# Evaluation
# ============================================================

N_GRID = 250
N_EVAL = 100_000

EPS = 1e-30


# ============================================================
# Conditional benchmark ranges
# ============================================================

MU_MIN = 0.20
MU_MAX = 0.80

SIGMA_MIN = 0.03
SIGMA_MAX = 0.08

FLOOR = 1e-6

BROAD_WEIGHT = 0.2
BROAD_MU = 0.5
BROAD_SIGMA = 0.2

RIDGE_SIGMA = 0.035
X_SIGMA = 0.035


# ============================================================
# Fixed test condition
# ============================================================

TEST_COND = (
    0.70,
    0.30,
    0.05,
)


# ============================================================
# Benchmarks
# ============================================================

BENCHMARK_NAMES = [
    "single_gaussian",
    "sharp_broad",
    "curved_ridge",
    "xshape",
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
# Condition helpers
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


def normalize_condition(cond):

    out = cond.clone()

    out[:, 0] = (
        out[:, 0] - MU_MIN
    ) / (
        MU_MAX - MU_MIN
    )

    out[:, 1] = (
        out[:, 1] - MU_MIN
    ) / (
        MU_MAX - MU_MIN
    )

    out[:, 2] = (
        out[:, 2] - SIGMA_MIN
    ) / (
        SIGMA_MAX - SIGMA_MIN
    )

    return out


def dummy_condition(n):

    return torch.zeros(
        n,
        1,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# Gaussian helpers
# ============================================================

def gaussian_integral_1d(
    mu,
    sigma,
):

    normal = torch.distributions.Normal(
        torch.tensor(
            0.0,
            dtype=mu.dtype,
            device=mu.device,
        ),
        torch.tensor(
            1.0,
            dtype=mu.dtype,
            device=mu.device,
        ),
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
            normal.cdf(hi)
            -
            normal.cdf(lo)
        )
    )


def sample_truncated_gaussian_1d(
    mu,
    sigma,
):

    n = mu.shape[0]

    result = torch.empty_like(
        mu
    )

    remaining = torch.ones(
        n,
        dtype=torch.bool,
        device=mu.device,
    )

    while remaining.any():

        idx = torch.where(
            remaining
        )[0]

        proposal = (
            mu[idx]
            +
            sigma[idx]
            * torch.randn_like(
                mu[idx]
            )
        )

        valid = (
            (proposal >= 0.0)
            &
            (proposal <= 1.0)
        )

        accepted_idx = idx[
            valid
        ]

        result[
            accepted_idx
        ] = proposal[
            valid
        ]

        remaining[
            accepted_idx
        ] = False

    return result


def truncated_normal_pdf(
    x,
    mu,
    sigma,
):

    normal = torch.distributions.Normal(
        torch.tensor(
            0.0,
            dtype=x.dtype,
            device=x.device,
        ),
        torch.tensor(
            1.0,
            dtype=x.dtype,
            device=x.device,
        ),
    )

    z = (
        x - mu
    ) / sigma

    numerator = (
        torch.exp(
            -0.5 * z**2
        )
        /
        (
            sigma
            *
            math.sqrt(
                2.0 * math.pi
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

    norm = (
        normal.cdf(hi)
        -
        normal.cdf(lo)
    )

    return (
        numerator
        /
        (
            norm + EPS
        )
    )


# ============================================================
# Benchmark 1
# Single moving Gaussian
# ============================================================

def single_gaussian_unnormalized(
    xy,
    cond_raw,
):

    x = xy[:, 0]
    y = xy[:, 1]

    mu_x = cond_raw[:, 0]
    mu_y = cond_raw[:, 1]
    sigma = cond_raw[:, 2]

    sharp = torch.exp(
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
        +
        sharp
    )


def single_gaussian_integral(
    condition,
):

    mu_x, mu_y, sigma = condition

    mu_x = torch.tensor(
        [mu_x],
        dtype=torch.float32,
        device=DEVICE,
    )

    mu_y = torch.tensor(
        [mu_y],
        dtype=torch.float32,
        device=DEVICE,
    )

    sigma = torch.tensor(
        [sigma],
        dtype=torch.float32,
        device=DEVICE,
    )

    Ix = gaussian_integral_1d(
        mu_x,
        sigma,
    )[0]

    Iy = gaussian_integral_1d(
        mu_y,
        sigma,
    )[0]

    return (
        FLOOR
        +
        Ix * Iy
    ).item()


def sample_single_gaussian(
    cond_raw,
):

    mu_x = cond_raw[:, 0]
    mu_y = cond_raw[:, 1]
    sigma = cond_raw[:, 2]

    x = sample_truncated_gaussian_1d(
        mu_x,
        sigma,
    )

    y = sample_truncated_gaussian_1d(
        mu_y,
        sigma,
    )

    return torch.stack(
        [
            x,
            y,
        ],
        dim=-1,
    )


# ============================================================
# Benchmark 2
# Sharp + broad
# ============================================================

def sharp_broad_unnormalized(
    xy,
    cond_raw,
):

    x = xy[:, 0]
    y = xy[:, 1]

    mu_x = cond_raw[:, 0]
    mu_y = cond_raw[:, 1]
    sigma = cond_raw[:, 2]

    sharp = torch.exp(
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

    broad = torch.exp(
        -0.5
        * (
            (x - BROAD_MU)
            / BROAD_SIGMA
        ) ** 2
        -
        0.5
        * (
            (y - BROAD_MU)
            / BROAD_SIGMA
        ) ** 2
    )

    return (
        FLOOR
        +
        sharp
        +
        BROAD_WEIGHT
        * broad
    )


def sharp_broad_integral(
    condition,
):

    mu_x, mu_y, sigma = condition

    mu_x = torch.tensor(
        [mu_x],
        dtype=torch.float32,
        device=DEVICE,
    )

    mu_y = torch.tensor(
        [mu_y],
        dtype=torch.float32,
        device=DEVICE,
    )

    sigma = torch.tensor(
        [sigma],
        dtype=torch.float32,
        device=DEVICE,
    )

    Ix = gaussian_integral_1d(
        mu_x,
        sigma,
    )[0]

    Iy = gaussian_integral_1d(
        mu_y,
        sigma,
    )[0]

    I_sharp = (
        Ix * Iy
    )

    broad_mu = torch.tensor(
        [BROAD_MU],
        dtype=torch.float32,
        device=DEVICE,
    )

    broad_sigma = torch.tensor(
        [BROAD_SIGMA],
        dtype=torch.float32,
        device=DEVICE,
    )

    Ib = gaussian_integral_1d(
        broad_mu,
        broad_sigma,
    )[0]

    I_broad = (
        BROAD_WEIGHT
        *
        Ib
        *
        Ib
    )

    return (
        FLOOR
        +
        I_sharp
        +
        I_broad
    ).item()


def sample_sharp_broad(
    cond_raw,
):

    n = cond_raw.shape[0]

    mu_x = cond_raw[:, 0]
    mu_y = cond_raw[:, 1]
    sigma = cond_raw[:, 2]

    Ix = gaussian_integral_1d(
        mu_x,
        sigma,
    )

    Iy = gaussian_integral_1d(
        mu_y,
        sigma,
    )

    I_sharp = (
        Ix * Iy
    )

    broad_mu = torch.full_like(
        mu_x,
        BROAD_MU,
    )

    broad_sigma = torch.full_like(
        sigma,
        BROAD_SIGMA,
    )

    Ib = gaussian_integral_1d(
        broad_mu,
        broad_sigma,
    )

    I_broad = (
        BROAD_WEIGHT
        *
        Ib
        *
        Ib
    )

    I_floor = torch.full_like(
        I_sharp,
        FLOOR,
    )

    total = (
        I_floor
        +
        I_sharp
        +
        I_broad
    )

    p_floor = (
        I_floor / total
    )

    p_sharp = (
        I_sharp / total
    )

    r = torch.rand(
        n,
        device=DEVICE,
    )

    floor_mask = (
        r < p_floor
    )

    sharp_mask = (
        (r >= p_floor)
        &
        (
            r
            <
            p_floor
            +
            p_sharp
        )
    )

    broad_mask = ~(
        floor_mask
        |
        sharp_mask
    )

    x = torch.empty(
        n,
        device=DEVICE,
    )

    y = torch.empty(
        n,
        device=DEVICE,
    )

    if floor_mask.any():

        nf = int(
            floor_mask.sum().item()
        )

        x[floor_mask] = torch.rand(
            nf,
            device=DEVICE,
        )

        y[floor_mask] = torch.rand(
            nf,
            device=DEVICE,
        )

    if sharp_mask.any():

        x[sharp_mask] = (
            sample_truncated_gaussian_1d(
                mu_x[sharp_mask],
                sigma[sharp_mask],
            )
        )

        y[sharp_mask] = (
            sample_truncated_gaussian_1d(
                mu_y[sharp_mask],
                sigma[sharp_mask],
            )
        )

    if broad_mask.any():

        nb = int(
            broad_mask.sum().item()
        )

        mu = torch.full(
            (nb,),
            BROAD_MU,
            device=DEVICE,
        )

        sig = torch.full(
            (nb,),
            BROAD_SIGMA,
            device=DEVICE,
        )

        x[broad_mask] = (
            sample_truncated_gaussian_1d(
                mu,
                sig,
            )
        )

        y[broad_mask] = (
            sample_truncated_gaussian_1d(
                mu,
                sig,
            )
        )

    return torch.stack(
        [
            x,
            y,
        ],
        dim=-1,
    )


# ============================================================
# Benchmark 3
# Curved ridge
# ============================================================

def ridge_center(x):

    return (
        0.5
        +
        0.25
        * torch.sin(
            2.0
            * math.pi
            * x
        )
    )


def curved_ridge_density(
    xy,
    cond_raw=None,
):

    x = xy[:, 0]
    y = xy[:, 1]

    mu = ridge_center(
        x
    )

    sigma = torch.full_like(
        y,
        RIDGE_SIGMA,
    )

    return truncated_normal_pdf(
        y,
        mu,
        sigma,
    )


def sample_curved_ridge(
    n,
):

    x = torch.rand(
        n,
        device=DEVICE,
    )

    mu = ridge_center(
        x
    )

    sigma = torch.full_like(
        x,
        RIDGE_SIGMA,
    )

    y = sample_truncated_gaussian_1d(
        mu,
        sigma,
    )

    return torch.stack(
        [
            x,
            y,
        ],
        dim=-1,
    )


# ============================================================
# Benchmark 4
# X-shaped mixture
# ============================================================

def xshape_density(
    xy,
    cond_raw=None,
):

    x = xy[:, 0]
    y = xy[:, 1]

    sigma = torch.full_like(
        x,
        X_SIGMA,
    )

    branch_1 = truncated_normal_pdf(
        y,
        x,
        sigma,
    )

    branch_2 = truncated_normal_pdf(
        y,
        1.0 - x,
        sigma,
    )

    return (
        0.5
        * branch_1
        +
        0.5
        * branch_2
    )


def sample_xshape(
    n,
):

    x = torch.rand(
        n,
        device=DEVICE,
    )

    branch = torch.randint(
        0,
        2,
        (n,),
        device=DEVICE,
    )

    mu = torch.where(
        branch == 0,
        x,
        1.0 - x,
    )

    sigma = torch.full_like(
        x,
        X_SIGMA,
    )

    y = sample_truncated_gaussian_1d(
        mu,
        sigma,
    )

    return torch.stack(
        [
            x,
            y,
        ],
        dim=-1,
    )


# ============================================================
# Copula model
# ============================================================

class ConditionalAutoregressiveCubicCopula2D(
    nn.Module
):

    def __init__(
        self,
        cond_dim,
        hidden=128,
        n_bins=32,
    ):

        super().__init__()

        self.conditional = (
            ConditionalCubicFlow1D(
                cond_dim=(
                    1 + cond_dim
                ),
                hidden=hidden,
                n_bins=n_bins,
            )
        )

    def forward(
        self,
        v,
        cond,
    ):

        v1 = v[:, 0]
        v2 = v[:, 1]

        u1 = v1

        context = torch.cat(
            [
                u1[:, None],
                cond,
            ],
            dim=-1,
        )

        u2, logdet = (
            self.conditional.forward(
                v2,
                context,
            )
        )

        u = torch.stack(
            [
                u1,
                u2.reshape(-1),
            ],
            dim=-1,
        )

        return (
            u,
            logdet.reshape(-1),
        )

    def inverse(
        self,
        u,
        cond,
    ):

        u1 = u[:, 0]
        u2 = u[:, 1]

        context = torch.cat(
            [
                u1[:, None],
                cond,
            ],
            dim=-1,
        )

        v2, forward_logdet = (
            self.conditional.inverse(
                u2,
                context,
            )
        )

        v = torch.stack(
            [
                u1,
                v2.reshape(-1),
            ],
            dim=-1,
        )

        return (
            v,
            forward_logdet.reshape(-1),
        )

    def log_prob(
        self,
        u,
        cond,
    ):

        _, forward_logdet = (
            self.inverse(
                u,
                cond,
            )
        )

        return (
            -forward_logdet
        )

    @torch.no_grad()
    def sample(
        self,
        cond,
    ):

        n = cond.shape[0]

        v = torch.rand(
            n,
            2,
            device=cond.device,
        )

        u, _ = self.forward(
            v,
            cond,
        )

        return u


# ============================================================
# Training: marginal flows
# ============================================================

def marginal_nll(
    flow,
    x,
    cond,
):

    _, forward_logdet = (
        flow.inverse(
            x,
            cond,
        )
    )

    log_q = (
        -forward_logdet.reshape(-1)
    )

    return (
        -log_q.mean()
    )


def train_marginals(
    benchmark_name,
    x_data,
    cond,
):

    cond_dim = cond.shape[1]

    flow_x = ConditionalCubicFlow1D(
        cond_dim=cond_dim,
        hidden=HIDDEN,
        n_bins=N_BINS,
    ).to(DEVICE)

    flow_y = ConditionalCubicFlow1D(
        cond_dim=cond_dim,
        hidden=HIDDEN,
        n_bins=N_BINS,
    ).to(DEVICE)

    opt_x = torch.optim.Adam(
        flow_x.parameters(),
        lr=LR_MARGINAL,
    )

    opt_y = torch.optim.Adam(
        flow_y.parameters(),
        lr=LR_MARGINAL,
    )

    print()
    print(
        f"[{benchmark_name}] "
        "training marginal flows"
    )

    for step in range(
        1,
        MARGINAL_STEPS + 1,
    ):

        idx = torch.randint(
            0,
            x_data.shape[0],
            (BATCH_SIZE,),
            device=DEVICE,
        )

        xb = x_data[idx]
        cb = cond[idx]

        loss_x = marginal_nll(
            flow_x,
            xb[:, 0],
            cb,
        )

        opt_x.zero_grad()
        loss_x.backward()

        torch.nn.utils.clip_grad_norm_(
            flow_x.parameters(),
            GRAD_CLIP,
        )

        opt_x.step()

        loss_y = marginal_nll(
            flow_y,
            xb[:, 1],
            cb,
        )

        opt_y.zero_grad()
        loss_y.backward()

        torch.nn.utils.clip_grad_norm_(
            flow_y.parameters(),
            GRAD_CLIP,
        )

        opt_y.step()

        if (
            step == 1
            or step % 250 == 0
        ):

            print(
                f"marginal step "
                f"{step:04d} | "
                f"Lx={loss_x.item():.6f} | "
                f"Ly={loss_y.item():.6f}"
            )

    torch.save(
        flow_x.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            f"{benchmark_name}_marginal_x.pt",
        ),
    )

    torch.save(
        flow_y.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            f"{benchmark_name}_marginal_y.pt",
        ),
    )

    return (
        flow_x,
        flow_y,
    )


# ============================================================
# Training: copula
# ============================================================

def train_copula(
    benchmark_name,
    u_data,
    cond,
):

    copula = (
        ConditionalAutoregressiveCubicCopula2D(
            cond_dim=cond.shape[1],
            hidden=HIDDEN,
            n_bins=N_BINS,
        )
        .to(DEVICE)
    )

    optimizer = torch.optim.Adam(
        copula.parameters(),
        lr=LR_COPULA,
    )

    print()
    print(
        f"[{benchmark_name}] "
        "training copula"
    )

    for step in range(
        1,
        COPULA_STEPS + 1,
    ):

        idx = torch.randint(
            0,
            u_data.shape[0],
            (BATCH_SIZE,),
            device=DEVICE,
        )

        ub = u_data[idx]
        cb = cond[idx]

        log_c = copula.log_prob(
            ub,
            cb,
        )

        loss = (
            -log_c.mean()
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            copula.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        if (
            step == 1
            or step % 250 == 0
        ):

            print(
                f"copula step "
                f"{step:04d} | "
                f"loss={loss.item():.6f}"
            )

    torch.save(
        copula.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            f"{benchmark_name}_copula.pt",
        ),
    )

    return copula


# ============================================================
# Training: direct cubic coupling
# ============================================================

def train_cubic_coupling(
    benchmark_name,
    x_data,
    cond,
):

    coupling = ConditionalCubicFlow2D(
        cond_dim=cond.shape[1],
        hidden=HIDDEN,
        n_bins=N_BINS,
        n_layers=COUPLING_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        coupling.parameters(),
        lr=LR_COUPLING,
    )

    print()
    print(
        f"[{benchmark_name}] "
        "training direct cubic coupling"
    )

    for step in range(
        1,
        COUPLING_STEPS + 1,
    ):

        idx = torch.randint(
            0,
            x_data.shape[0],
            (BATCH_SIZE,),
            device=DEVICE,
        )

        xb = x_data[idx]
        cb = cond[idx]

        log_q = coupling.log_prob(
            xb,
            cb,
        )

        loss = (
            -log_q.mean()
        )

        optimizer.zero_grad()
        loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                coupling.parameters(),
                GRAD_CLIP,
            )
        )

        optimizer.step()

        if (
            step == 1
            or step % 250 == 0
        ):

            print(
                f"coupling step "
                f"{step:04d} | "
                f"loss={loss.item():.6f} | "
                f"grad={float(grad_norm):.3f}"
            )

    torch.save(
        coupling.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            f"{benchmark_name}_cubic_coupling.pt",
        ),
    )

    return coupling


# ============================================================
# Copula model density
# ============================================================

@torch.no_grad()
def copula_model_log_prob(
    xy,
    cond,
    flow_x,
    flow_y,
    copula,
):

    u1, forward_ld_x = (
        flow_x.inverse(
            xy[:, 0],
            cond,
        )
    )

    u2, forward_ld_y = (
        flow_y.inverse(
            xy[:, 1],
            cond,
        )
    )

    u1 = u1.reshape(-1)
    u2 = u2.reshape(-1)

    # ========================================================
    # Numerical boundary protection
    # ========================================================

    tol = 1e-5

    bad_u1 = (
        (u1 < -tol)
        |
        (u1 > 1.0 + tol)
    )

    bad_u2 = (
        (u2 < -tol)
        |
        (u2 > 1.0 + tol)
    )

    if bad_u1.any() or bad_u2.any():

        raise RuntimeError(
            "Marginal inverse produced a genuinely "
            "out-of-range copula coordinate.\n"
            f"u1 range: "
            f"[{u1.min().item():.8f}, "
            f"{u1.max().item():.8f}]\n"
            f"u2 range: "
            f"[{u2.min().item():.8f}, "
            f"{u2.max().item():.8f}]"
        )

    # Only correct tiny floating-point excursions
    u1 = u1.clamp(
        0.0,
        1.0,
    )

    u2 = u2.clamp(
        0.0,
        1.0,
    )

    # ========================================================
    # Marginal log densities
    # ========================================================

    log_qx = (
        -forward_ld_x.reshape(-1)
    )

    log_qy = (
        -forward_ld_y.reshape(-1)
    )

    # ========================================================
    # Copula density
    # ========================================================

    u = torch.stack(
        [
            u1,
            u2,
        ],
        dim=-1,
    )

    log_c = copula.log_prob(
        u,
        cond,
    )

    # ========================================================
    # Full density
    # ========================================================

    return (
        log_qx
        +
        log_qy
        +
        log_c
    )


@torch.no_grad()
def sample_copula_model(
    cond,
    flow_x,
    flow_y,
    copula,
):

    u = copula.sample(
        cond
    )

    x1, _ = flow_x.forward(
        u[:, 0],
        cond,
    )

    x2, _ = flow_y.forward(
        u[:, 1],
        cond,
    )

    return torch.stack(
        [
            x1.reshape(-1),
            x2.reshape(-1),
        ],
        dim=-1,
    )


# ============================================================
# Coupling wrappers
# ============================================================

@torch.no_grad()
def coupling_log_prob(
    xy,
    cond,
    model,
):

    return model.log_prob(
        xy,
        cond,
    )


@torch.no_grad()
def sample_coupling(
    cond,
    model,
):

    n = cond.shape[0]

    return model.sample(
        n=n,
        cond=cond,
        device=DEVICE,
    )


# ============================================================
# Artifact score
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


def compute_artifact_score(
    p_grid,
    q_grid,
):

    p_residual = (
        p_grid
        -
        smooth_2d(
            p_grid
        )
    )

    q_residual = (
        q_grid
        -
        smooth_2d(
            q_grid
        )
    )

    artifact = (
        q_residual
        -
        p_residual
    )

    return float(
        np.linalg.norm(
            artifact
        )
        /
        (
            np.linalg.norm(
                p_grid
            )
            +
            EPS
        )
    )


# ============================================================
# Benchmark preparation
# ============================================================

def prepare_benchmark(
    benchmark_name,
):

    if benchmark_name == "single_gaussian":

        cond_raw = sample_conditions(
            N_DATA
        )

        x_data = sample_single_gaussian(
            cond_raw
        )

        cond_model = normalize_condition(
            cond_raw
        )

        test_raw = torch.tensor(
            [TEST_COND],
            dtype=torch.float32,
            device=DEVICE,
        )

        test_model = normalize_condition(
            test_raw
        )

        integral = single_gaussian_integral(
            TEST_COND
        )

        return {
            "x_data":
                x_data,

            "cond_model":
                cond_model,

            "test_raw":
                test_raw,

            "test_model":
                test_model,

            "density":
                single_gaussian_unnormalized,

            "integral":
                integral,

            "conditional":
                True,

            "peak_metric":
                True,
        }

    if benchmark_name == "sharp_broad":

        cond_raw = sample_conditions(
            N_DATA
        )

        x_data = sample_sharp_broad(
            cond_raw
        )

        cond_model = normalize_condition(
            cond_raw
        )

        test_raw = torch.tensor(
            [TEST_COND],
            dtype=torch.float32,
            device=DEVICE,
        )

        test_model = normalize_condition(
            test_raw
        )

        integral = sharp_broad_integral(
            TEST_COND
        )

        return {
            "x_data":
                x_data,

            "cond_model":
                cond_model,

            "test_raw":
                test_raw,

            "test_model":
                test_model,

            "density":
                sharp_broad_unnormalized,

            "integral":
                integral,

            "conditional":
                True,

            "peak_metric":
                True,
        }

    if benchmark_name == "curved_ridge":

        x_data = sample_curved_ridge(
            N_DATA
        )

        cond_model = dummy_condition(
            N_DATA
        )

        test_raw = None

        test_model = dummy_condition(
            1
        )

        return {
            "x_data":
                x_data,

            "cond_model":
                cond_model,

            "test_raw":
                test_raw,

            "test_model":
                test_model,

            "density":
                curved_ridge_density,

            "integral":
                1.0,

            "conditional":
                False,

            "peak_metric":
                False,
        }

    if benchmark_name == "xshape":

        x_data = sample_xshape(
            N_DATA
        )

        cond_model = dummy_condition(
            N_DATA
        )

        test_raw = None

        test_model = dummy_condition(
            1
        )

        return {
            "x_data":
                x_data,

            "cond_model":
                cond_model,

            "test_raw":
                test_raw,

            "test_model":
                test_model,

            "density":
                xshape_density,

            "integral":
                1.0,

            "conditional":
                False,

            "peak_metric":
                False,
        }

    raise ValueError(
        benchmark_name
    )


# ============================================================
# Exact target evaluation
# ============================================================

@torch.no_grad()
def evaluate_target_density(
    benchmark,
    xy,
):

    density_fn = benchmark[
        "density"
    ]

    if benchmark[
        "conditional"
    ]:

        cond_raw = (
            benchmark[
                "test_raw"
            ]
            .repeat(
                xy.shape[0],
                1,
            )
        )

        f = density_fn(
            xy,
            cond_raw,
        )

    else:

        f = density_fn(
            xy
        )

    p = (
        f
        /
        benchmark[
            "integral"
        ]
    )

    return (
        f,
        p,
    )


# ============================================================
# Evaluate one model
# ============================================================

@torch.no_grad()
def evaluate_model(
    benchmark_name,
    benchmark,
    model_name,
    coupling=None,
    flow_x=None,
    flow_y=None,
    copula=None,
):

    print()
    print(
        f"Evaluating "
        f"{benchmark_name} | "
        f"{model_name}"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    axis = (
        torch.arange(
            N_GRID,
            device=DEVICE,
            dtype=torch.float32,
        )
        + 0.5
    ) / N_GRID

    X, Y = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    xy_grid = torch.stack(
        [
            X.reshape(-1),
            Y.reshape(-1),
        ],
        dim=-1,
    )

    cond_grid = (
        benchmark[
            "test_model"
        ]
        .repeat(
            xy_grid.shape[0],
            1,
        )
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    _, p = evaluate_target_density(
        benchmark,
        xy_grid,
    )

    # --------------------------------------------------------
    # Model density
    # --------------------------------------------------------

    if model_name == "cubic_coupling":

        log_q = coupling_log_prob(
            xy_grid,
            cond_grid,
            coupling,
        )

    elif model_name == "copula":

        log_q = copula_model_log_prob(
            xy_grid,
            cond_grid,
            flow_x,
            flow_y,
            copula,
        )

    else:

        raise ValueError(
            model_name
        )

    q = torch.exp(
        log_q
    )

    # --------------------------------------------------------
    # Density metrics
    # --------------------------------------------------------

    ise = torch.mean(
        (
            q - p
        ) ** 2
    ).item()

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

    artifact = compute_artifact_score(
        p_grid,
        q_grid,
    )

    p_max = float(
        p.max().item()
    )

    q_max = float(
        q.max().item()
    )

    peak_error = None

    if benchmark[
        "peak_metric"
    ]:

        target_peak = xy_grid[
            torch.argmax(p)
        ]

        model_peak = xy_grid[
            torch.argmax(q)
        ]

        peak_error = float(
            torch.linalg.vector_norm(
                target_peak
                -
                model_peak
            ).item()
        )

    # --------------------------------------------------------
    # MC evaluation
    # --------------------------------------------------------

    cond_mc = (
        benchmark[
            "test_model"
        ]
        .repeat(
            N_EVAL,
            1,
        )
    )

    if model_name == "cubic_coupling":

        samples = sample_coupling(
            cond_mc,
            coupling,
        )

        log_q_samples = coupling_log_prob(
            samples,
            cond_mc,
            coupling,
        )

    else:

        samples = sample_copula_model(
            cond_mc,
            flow_x,
            flow_y,
            copula,
        )

        log_q_samples = copula_model_log_prob(
            samples,
            cond_mc,
            flow_x,
            flow_y,
            copula,
        )

    q_samples = torch.exp(
        log_q_samples
    )

    density_fn = benchmark[
        "density"
    ]

    if benchmark[
        "conditional"
    ]:

        cond_raw_mc = (
            benchmark[
                "test_raw"
            ]
            .repeat(
                N_EVAL,
                1,
            )
        )

        f_samples = density_fn(
            samples,
            cond_raw_mc,
        )

    else:

        f_samples = density_fn(
            samples
        )

    weights = (
        f_samples
        /
        (
            q_samples
            +
            EPS
        )
    )

    estimate = float(
        weights.mean().item()
    )

    variance = float(
        (
            weights.var(
                unbiased=True
            )
            /
            N_EVAL
        ).item()
    )

    ess = (
        weights.sum() ** 2
        /
        (
            torch.sum(
                weights ** 2
            )
            +
            EPS
        )
    )

    ess_fraction = float(
        (
            ess
            /
            N_EVAL
        ).item()
    )

    reference = float(
        benchmark[
            "integral"
        ]
    )

    relative_error = float(
        abs(
            estimate
            -
            reference
        )
        /
        abs(
            reference
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    residual = (
        q_grid
        -
        p_grid
    )

    vmax = max(
        p_grid.max(),
        q_grid.max(),
    )

    rmax = max(
        float(
            np.abs(
                residual
            ).max()
        ),
        1e-12,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    im0 = axes[0].imshow(
        p_grid.T,
        origin="lower",
        extent=[
            0,
            1,
            0,
            1,
        ],
        aspect="equal",
        vmin=0,
        vmax=vmax,
    )

    axes[0].set_title(
        "Target p"
    )

    im1 = axes[1].imshow(
        q_grid.T,
        origin="lower",
        extent=[
            0,
            1,
            0,
            1,
        ],
        aspect="equal",
        vmin=0,
        vmax=vmax,
    )

    axes[1].set_title(
        model_name
    )

    im2 = axes[2].imshow(
        residual.T,
        origin="lower",
        extent=[
            0,
            1,
            0,
            1,
        ],
        aspect="equal",
        vmin=-rmax,
        vmax=rmax,
    )

    axes[2].set_title(
        "q - p"
    )

    for ax in axes:

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            "y"
        )

    fig.colorbar(
        im0,
        ax=axes[0],
    )

    fig.colorbar(
        im1,
        ax=axes[1],
    )

    fig.colorbar(
        im2,
        ax=axes[2],
    )

    fig.suptitle(
        f"{benchmark_name} | "
        f"{model_name}"
    )

    plt.tight_layout()

    plot_path = os.path.join(
        PLOT_DIR,
        (
            f"{benchmark_name}"
            f"_{model_name}.png"
        ),
    )

    plt.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    result = {
        "benchmark":
            benchmark_name,

        "model":
            model_name,

        "ise":
            ise,

        "artifact_score":
            artifact,

        "peak_error":
            peak_error,

        "p_max":
            p_max,

        "q_max":
            q_max,

        "qmax_over_pmax":
            (
                q_max
                /
                p_max
            ),

        "integral_estimate":
            estimate,

        "integral_reference":
            reference,

        "relative_error":
            relative_error,

        "variance":
            variance,

        "ess_fraction":
            ess_fraction,
    }

    return result


# ============================================================
# Save results
# ============================================================

def save_results(
    results,
):

    json_path = os.path.join(
        RESULT_DIR,
        "results.json",
    )

    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    csv_path = os.path.join(
        RESULT_DIR,
        "summary.csv",
    )

    fields = [
        "benchmark",
        "model",
        "ise",
        "artifact_score",
        "peak_error",
        "p_max",
        "q_max",
        "qmax_over_pmax",
        "integral_estimate",
        "integral_reference",
        "relative_error",
        "variance",
        "ess_fraction",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# Pretty summary
# ============================================================

def print_summary(
    results,
):

    print()
    print("=" * 130)
    print(
        "FINAL BENCHMARK SUMMARY"
    )
    print("=" * 130)

    print(
        f"{'Benchmark':<20}"
        f"{'Model':<20}"
        f"{'ISE':>12}"
        f"{'Artifact':>12}"
        f"{'ESS':>10}"
        f"{'Variance':>15}"
        f"{'Rel err':>12}"
        f"{'Peak err':>12}"
    )

    print("-" * 130)

    for r in results:

        peak = (
            "-"
            if r["peak_error"] is None
            else
            f"{r['peak_error']:.4f}"
        )

        print(
            f"{r['benchmark']:<20}"
            f"{r['model']:<20}"
            f"{r['ise']:>12.3e}"
            f"{r['artifact_score']:>12.4f}"
            f"{r['ess_fraction']:>10.3f}"
            f"{r['variance']:>15.3e}"
            f"{r['relative_error']:>12.3e}"
            f"{peak:>12}"
        )

    print()
    print(
        f"Results saved to: "
        f"{RESULT_DIR}"
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Results directory: "
        f"{RESULT_DIR}"
    )

    all_results = []

    for benchmark_name in BENCHMARK_NAMES:

        print()
        print("=" * 100)
        print(
            f"BENCHMARK: "
            f"{benchmark_name}"
        )
        print("=" * 100)

        benchmark = prepare_benchmark(
            benchmark_name
        )

        x_data = benchmark[
            "x_data"
        ]

        cond_model = benchmark[
            "cond_model"
        ]

        # ====================================================
        # MODEL 1
        # Direct cubic coupling
        # ====================================================

        coupling = train_cubic_coupling(
            benchmark_name,
            x_data,
            cond_model,
        )

        coupling.eval()

        result_coupling = evaluate_model(
            benchmark_name=benchmark_name,
            benchmark=benchmark,
            model_name="cubic_coupling",
            coupling=coupling,
        )

        all_results.append(
            result_coupling
        )

        save_results(
            all_results
        )

        print(
            result_coupling
        )

        # ====================================================
        # MODEL 2
        # Marginal cubic flows + copula
        # ====================================================

        flow_x, flow_y = train_marginals(
            benchmark_name,
            x_data,
            cond_model,
        )

        flow_x.eval()
        flow_y.eval()

        # Freeze marginals before copula training

        for parameter in flow_x.parameters():
            parameter.requires_grad_(
                False
            )

        for parameter in flow_y.parameters():
            parameter.requires_grad_(
                False
            )

        # ----------------------------------------------------
        # x -> u
        # ----------------------------------------------------

        with torch.no_grad():

            u1, _ = flow_x.inverse(
                x_data[:, 0],
                cond_model,
            )

            u2, _ = flow_y.inverse(
                x_data[:, 1],
                cond_model,
            )

            u_data = torch.stack(
                [
                    u1.reshape(-1),
                    u2.reshape(-1),
                ],
                dim=-1,
            )

        # ----------------------------------------------------
        # Train dependence model
        # ----------------------------------------------------

        copula = train_copula(
            benchmark_name,
            u_data,
            cond_model,
        )

        copula.eval()

        result_copula = evaluate_model(
            benchmark_name=benchmark_name,
            benchmark=benchmark,
            model_name="copula",
            flow_x=flow_x,
            flow_y=flow_y,
            copula=copula,
        )

        all_results.append(
            result_copula
        )

        save_results(
            all_results
        )

        print(
            result_copula
        )

        # ====================================================
        # Free memory
        # ====================================================

        del coupling
        del flow_x
        del flow_y
        del copula
        del x_data
        del cond_model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # Final outputs
    # ========================================================

    save_results(
        all_results
    )

    print_summary(
        all_results
    )


# ============================================================

if __name__ == "__main__":

    main()