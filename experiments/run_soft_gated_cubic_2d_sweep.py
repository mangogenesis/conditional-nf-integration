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

OUT_DIR = "results/soft_gated_cubic_2d_sweep"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ============================================================
# Sweep
# ============================================================

N_BLOCKS_LIST = [1, 2, 4]

N_BINS = 32
HIDDEN = 128

# soft inverse uses bisection
SOFT_INVERSE_ITERS = 24


# ============================================================
# Training
# ============================================================

N_COND = 64
N_X_PER_COND = 512

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

DIAGNOSTIC_CONDITION = (
    0.70,
    0.30,
    0.05,
)


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
# Condition utilities
# ============================================================

def sample_conditions(n):

    mu_x = torch.empty(
        n, 1, device=DEVICE
    ).uniform_(MU_MIN, MU_MAX)

    mu_y = torch.empty(
        n, 1, device=DEVICE
    ).uniform_(MU_MIN, MU_MAX)

    sigma = torch.empty(
        n, 1, device=DEVICE
    ).uniform_(SIGMA_MIN, SIGMA_MAX)

    return torch.cat(
        [mu_x, mu_y, sigma],
        dim=-1,
    )


def normalize_condition(cond):

    out = cond.clone()

    out[..., 0] = (
        out[..., 0] - MU_MIN
    ) / (
        MU_MAX - MU_MIN
    )

    out[..., 1] = (
        out[..., 1] - MU_MIN
    ) / (
        MU_MAX - MU_MIN
    )

    out[..., 2] = (
        out[..., 2] - SIGMA_MIN
    ) / (
        SIGMA_MAX - SIGMA_MIN
    )

    return out


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

    return (
        FLOOR
        +
        torch.exp(
            -0.5
            * ((x - mu_x) / sigma) ** 2
            -0.5
            * ((y - mu_y) / sigma) ** 2
        )
    )


# ============================================================
# Exact integral
# ============================================================

def gaussian_integral_1d(mu, sigma):

    def Phi(z):

        return 0.5 * (
            1.0
            + math.erf(
                z / math.sqrt(2.0)
            )
        )

    lo = -mu / sigma
    hi = (1.0 - mu) / sigma

    return (
        sigma
        * math.sqrt(2.0 * math.pi)
        * (
            Phi(hi)
            - Phi(lo)
        )
    )


def exact_integral(condition):

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
# Gate network
# ============================================================

class GateNet(nn.Module):

    def __init__(
        self,
        context_dim,
        hidden=64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                context_dim,
                hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                1,
            ),
        )

        # Initial lambda ~ 0.5
        nn.init.zeros_(
            self.net[-1].weight
        )

        nn.init.zeros_(
            self.net[-1].bias
        )

    def forward(self, context):

        return torch.sigmoid(
            self.net(context)
        ).squeeze(-1)


# ============================================================
# One soft 1D transformation
# ============================================================
def safe_unit_interval(
    x,
    tol=1e-6,
    name="tensor",
):
    """
    Enforce the mathematical [0,1] domain while allowing only
    tiny floating-point excursions.

    Tiny numerical drift:
        -1e-8, 1+1e-7
    will be clipped.

    A genuine violation:
        -0.01, 1.02
    will raise an error.
    """

    xmin = x.detach().min().item()
    xmax = x.detach().max().item()

    if (
        xmin < -tol
        or xmax > 1.0 + tol
    ):
        raise RuntimeError(
            f"{name} genuinely left [0,1]: "
            f"min={xmin:.9e}, "
            f"max={xmax:.9e}, "
            f"tol={tol:.1e}"
        )

    return x.clamp(
        0.0,
        1.0,
    )
    
class SoftCubicTransform1D(nn.Module):
    """
    Soft monotone cubic transformation

        z' = (1-lambda) z + lambda T(z | context)

    with

        lambda = sigmoid(g(context))

    Since

        0 <= lambda <= 1

    and both z and T(z) lie in [0,1],

    the output theoretically also lies in [0,1].

    The safe_unit_interval() calls only correct tiny
    floating-point excursions.
    """

    def __init__(
        self,
        context_dim,
        hidden=128,
        n_bins=32,
        inverse_iters=24,
    ):
        super().__init__()

        self.spline = ConditionalCubicFlow1D(
            cond_dim=context_dim,
            hidden=hidden,
            n_bins=n_bins,
        )

        self.gate = GateNet(
            context_dim=context_dim,
            hidden=64,
        )

        self.inverse_iters = (
            inverse_iters
        )


    # ========================================================
    # Gate
    # ========================================================

    def gate_value(
        self,
        context,
    ):

        lam = self.gate(
            context
        )

        # sigmoid already guarantees this,
        # but we explicitly check for debugging
        if torch.any(
            ~torch.isfinite(lam)
        ):
            raise RuntimeError(
                "Non-finite gate value detected"
            )

        return lam


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        z,
        context,
    ):

        z = z.reshape(-1)

        # ----------------------------------------------------
        # z mathematically must be inside [0,1].
        #
        # Correct only tiny floating point excursions.
        # ----------------------------------------------------

        z = safe_unit_interval(
            z,
            name="soft-transform input",
        )

        # ----------------------------------------------------
        # Full cubic map
        #
        # T : [0,1] -> [0,1]
        # ----------------------------------------------------

        (
            t,
            log_t_prime,
        ) = self.spline.forward(
            z,
            context,
        )

        t = t.reshape(-1)

        log_t_prime = (
            log_t_prime.reshape(-1)
        )

        # Cubic output theoretically lies in [0,1]
        t = safe_unit_interval(
            t,
            name="cubic output",
        )

        # ----------------------------------------------------
        # T'(z)
        # ----------------------------------------------------

        t_prime = torch.exp(
            log_t_prime
        )

        if torch.any(
            ~torch.isfinite(t_prime)
        ):
            raise RuntimeError(
                "Non-finite cubic derivative"
            )

        if torch.any(
            t_prime <= 0
        ):
            raise RuntimeError(
                "Non-positive cubic derivative: "
                f"min={t_prime.min().item():.9e}"
            )

        # ----------------------------------------------------
        # Learned soft gate
        # ----------------------------------------------------

        lam = self.gate_value(
            context
        )

        # ----------------------------------------------------
        # Soft interpolation:
        #
        # S(z)
        # =
        # (1-lambda) z
        # +
        # lambda T(z)
        # ----------------------------------------------------

        out = (
            (1.0 - lam) * z
            +
            lam * t
        )

        # The convex combination must mathematically
        # remain in [0,1].
        out = safe_unit_interval(
            out,
            name="soft-transform output",
        )

        # ----------------------------------------------------
        # Derivative:
        #
        # S'(z)
        # =
        # (1-lambda)
        # +
        # lambda T'(z)
        #
        # lambda is independent of z within this scalar
        # transformation because it depends only on context.
        # ----------------------------------------------------

        derivative = (
            (1.0 - lam)
            +
            lam * t_prime
        )

        if torch.any(
            ~torch.isfinite(derivative)
        ):
            raise RuntimeError(
                "Non-finite soft-transform derivative"
            )

        if torch.any(
            derivative <= 0
        ):
            raise RuntimeError(
                "Non-positive soft-transform derivative: "
                f"min={derivative.min().item():.9e}"
            )

        logdet = torch.log(
            derivative
        )

        return (
            out,
            logdet,
            lam,
        )


    # ========================================================
    # Inverse
    # ========================================================

    def inverse(
        self,
        out,
        context,
    ):
        """
        Invert

            out =
            (1-lambda) z
            +
            lambda T(z)

        using bisection.

        Because the soft map is strictly monotone,
        there is a unique solution z in [0,1].
        """

        out = out.reshape(-1)

        out = safe_unit_interval(
            out,
            name="soft-inverse input",
        )

        lam = self.gate_value(
            context
        )

        # ----------------------------------------------------
        # Initial bisection bracket
        # ----------------------------------------------------

        low = torch.zeros_like(
            out
        )

        high = torch.ones_like(
            out
        )

        # ----------------------------------------------------
        # Bisection
        # ----------------------------------------------------

        for _ in range(
            self.inverse_iters
        ):

            mid = (
                low + high
            ) * 0.5

            (
                t_mid,
                _,
            ) = self.spline.forward(
                mid,
                context,
            )

            t_mid = (
                t_mid.reshape(-1)
            )

            t_mid = safe_unit_interval(
                t_mid,
                name="inverse cubic midpoint",
            )

            blended_mid = (
                (1.0 - lam) * mid
                +
                lam * t_mid
            )

            go_right = (
                blended_mid < out
            )

            low = torch.where(
                go_right,
                mid,
                low,
            )

            high = torch.where(
                go_right,
                high,
                mid,
            )

        # ----------------------------------------------------
        # Recovered input
        # ----------------------------------------------------

        z = (
            low + high
        ) * 0.5

        z = safe_unit_interval(
            z,
            name="recovered soft input",
        )

        # ----------------------------------------------------
        # Evaluate derivative at recovered z
        # ----------------------------------------------------

        (
            t,
            log_t_prime,
        ) = self.spline.forward(
            z,
            context,
        )

        t = t.reshape(-1)

        log_t_prime = (
            log_t_prime.reshape(-1)
        )

        t_prime = torch.exp(
            log_t_prime
        )

        if torch.any(
            ~torch.isfinite(t_prime)
        ):
            raise RuntimeError(
                "Non-finite inverse cubic derivative"
            )

        if torch.any(
            t_prime <= 0
        ):
            raise RuntimeError(
                "Non-positive inverse cubic derivative: "
                f"min={t_prime.min().item():.9e}"
            )

        derivative = (
            (1.0 - lam)
            +
            lam * t_prime
        )

        if torch.any(
            ~torch.isfinite(derivative)
        ):
            raise RuntimeError(
                "Non-finite inverse soft derivative"
            )

        if torch.any(
            derivative <= 0
        ):
            raise RuntimeError(
                "Non-positive inverse soft derivative: "
                f"min={derivative.min().item():.9e}"
            )

        # This is still the FORWARD log Jacobian
        #
        # log |d out / dz|
        logdet_forward = torch.log(
            derivative
        )

        return (
            z,
            logdet_forward,
            lam,
        )


# ============================================================
# One 2D soft block
# ============================================================

class SoftCubicBlock2D(nn.Module):
    """
    Sequential soft transformation:

        x1 = soft_x(x | y, c)
        y1 = soft_y(y | x1, c)

    Unlike ordinary coupling, neither coordinate has to stay
    exactly unchanged.
    """

    def __init__(
        self,
        cond_dim=3,
        hidden=128,
        n_bins=32,
        inverse_iters=24,
    ):
        super().__init__()

        context_dim = (
            1 + cond_dim
        )

        self.x_transform = (
            SoftCubicTransform1D(
                context_dim=context_dim,
                hidden=hidden,
                n_bins=n_bins,
                inverse_iters=inverse_iters,
            )
        )

        self.y_transform = (
            SoftCubicTransform1D(
                context_dim=context_dim,
                hidden=hidden,
                n_bins=n_bins,
                inverse_iters=inverse_iters,
            )
        )

    def forward(
        self,
        xy,
        cond,
    ):

        x = xy[:, 0]
        y = xy[:, 1]

        # ------------------------------------------
        # transform x using old y
        # ------------------------------------------

        context_x = torch.cat(
            [
                y[:, None],
                cond,
            ],
            dim=-1,
        )

        (
            x_new,
            ld_x,
            lam_x,
        ) = self.x_transform.forward(
            x,
            context_x,
        )

        # ------------------------------------------
        # transform y using NEW x
        # ------------------------------------------

        context_y = torch.cat(
            [
                x_new[:, None],
                cond,
            ],
            dim=-1,
        )

        (
            y_new,
            ld_y,
            lam_y,
        ) = self.y_transform.forward(
            y,
            context_y,
        )

        out = torch.stack(
            [
                x_new,
                y_new,
            ],
            dim=-1,
        )

        logdet = (
            ld_x + ld_y
        )

        return (
            out,
            logdet,
            lam_x,
            lam_y,
        )

    def inverse(
        self,
        out,
        cond,
    ):

        x_new = out[:, 0]
        y_new = out[:, 1]

        # ------------------------------------------
        # inverse y first because forward y used x_new
        # ------------------------------------------

        context_y = torch.cat(
            [
                x_new[:, None],
                cond,
            ],
            dim=-1,
        )

        (
            y,
            ld_y,
            lam_y,
        ) = self.y_transform.inverse(
            y_new,
            context_y,
        )

        # ------------------------------------------
        # now inverse x using recovered old y
        # ------------------------------------------

        context_x = torch.cat(
            [
                y[:, None],
                cond,
            ],
            dim=-1,
        )

        (
            x,
            ld_x,
            lam_x,
        ) = self.x_transform.inverse(
            x_new,
            context_x,
        )

        xy = torch.stack(
            [
                x,
                y,
            ],
            dim=-1,
        )

        forward_logdet = (
            ld_x + ld_y
        )

        return (
            xy,
            forward_logdet,
            lam_x,
            lam_y,
        )


# ============================================================
# Full soft-gated flow
# ============================================================

class SoftGatedCubicFlow2D(nn.Module):

    def __init__(
        self,
        cond_dim=3,
        hidden=128,
        n_bins=32,
        n_blocks=2,
        inverse_iters=24,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                SoftCubicBlock2D(
                    cond_dim=cond_dim,
                    hidden=hidden,
                    n_bins=n_bins,
                    inverse_iters=inverse_iters,
                )
                for _ in range(
                    n_blocks
                )
            ]
        )

    def forward(
        self,
        u,
        cond,
    ):

        z = u

        total_logdet = torch.zeros(
            u.shape[0],
            device=u.device,
            dtype=u.dtype,
        )

        lambdas = []

        for block in self.blocks:

            (
                z,
                ld,
                lam_x,
                lam_y,
            ) = block.forward(
                z,
                cond,
            )

            total_logdet = (
                total_logdet + ld
            )

            lambdas.append(
                (
                    lam_x,
                    lam_y,
                )
            )

        return (
            z,
            total_logdet,
            lambdas,
        )

    def inverse(
        self,
        x,
        cond,
    ):

        z = x

        total_forward_logdet = (
            torch.zeros(
                x.shape[0],
                device=x.device,
                dtype=x.dtype,
            )
        )

        lambdas = []

        for block in reversed(
            self.blocks
        ):

            (
                z,
                ld,
                lam_x,
                lam_y,
            ) = block.inverse(
                z,
                cond,
            )

            total_forward_logdet = (
                total_forward_logdet
                + ld
            )

            lambdas.append(
                (
                    lam_x,
                    lam_y,
                )
            )

        return (
            z,
            total_forward_logdet,
            lambdas,
        )

    def log_prob(
        self,
        x,
        cond,
    ):

        (
            _,
            forward_logdet,
            _,
        ) = self.inverse(
            x,
            cond,
        )

        return (
            -forward_logdet
        )

    def sample(
        self,
        n,
        cond,
    ):

        u = torch.rand(
            n,
            2,
            device=cond.device,
            dtype=cond.dtype,
        )

        x, _, _ = self.forward(
            u,
            cond,
        )

        return x


# ============================================================
# Model
# ============================================================

def build_model(
    n_blocks,
):

    return SoftGatedCubicFlow2D(
        cond_dim=3,
        hidden=HIDDEN,
        n_bins=N_BINS,
        n_blocks=n_blocks,
        inverse_iters=SOFT_INVERSE_ITERS,
    ).to(DEVICE)

@torch.no_grad()
def sanity_check_soft_model(
    model,
    n=10000,
):

    print()
    print("=" * 80)
    print("SOFT FLOW SANITY CHECK")
    print("=" * 80)

    cond_raw = sample_conditions(
        n
    )

    cond = normalize_condition(
        cond_raw
    )

    u = torch.rand(
        n,
        2,
        device=DEVICE,
    )

    # forward
    (
        x,
        logdet_fwd,
        _,
    ) = model.forward(
        u,
        cond,
    )

    print(
        "forward x min/max:",
        x.min().item(),
        x.max().item(),
    )

    # inverse
    (
        u_rec,
        logdet_inv_point,
        _,
    ) = model.inverse(
        x,
        cond,
    )

    reconstruction_error = (
        torch.abs(
            u - u_rec
        )
    )

    logdet_error = torch.abs(
        logdet_fwd
        -
        logdet_inv_point
    )

    print(
        "max reconstruction error:",
        reconstruction_error.max().item(),
    )

    print(
        "mean reconstruction error:",
        reconstruction_error.mean().item(),
    )

    print(
        "max logdet error:",
        logdet_error.max().item(),
    )

    print(
        "mean logdet error:",
        logdet_error.mean().item(),
    )

    log_q = model.log_prob(
        x,
        cond,
    )

    identity_error = torch.abs(
        log_q
        +
        logdet_fwd
    )

    print(
        "max logq + logdet error:",
        identity_error.max().item(),
    )

    print(
        "mean logq + logdet error:",
        identity_error.mean().item(),
    )

    print("=" * 80)
# ============================================================
# Forward KL
# ============================================================

def forward_kl_loss(
    model,
):

    cond_raw = sample_conditions(
        N_COND
    )

    cond_model = normalize_condition(
        cond_raw
    )

    cond_grouped_raw = (
        cond_raw[:, None, :]
        .expand(
            N_COND,
            N_X_PER_COND,
            3,
        )
    )

    cond_grouped_model = (
        cond_model[:, None, :]
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
        cond_grouped_raw.reshape(
            -1,
            3,
        )
    )

    cond_model_flat = (
        cond_grouped_model.reshape(
            -1,
            3,
        )
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

    return loss_per_condition.mean()


# ============================================================
# Training
# ============================================================

def train_model(
    n_blocks,
):

    set_seed(SEED)

    model = build_model(
        n_blocks
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    name = (
        f"soft_blocks_{n_blocks}"
    )

    print()
    print("=" * 100)
    print(
        f"TRAINING {name}"
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
                f"non-finite loss "
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
                f"{name:16s} | "
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

    sanity_check_soft_model(
        model
    )

    return model


# ============================================================
# Stripe diagnostics
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

    return np.apply_along_axis(
        lambda col:
            np.convolve(
                col,
                kernel,
                mode="same",
            ),
        axis=0,
        arr=tmp,
    )


def stripe_metrics(
    p_grid,
    q_grid,
):

    p_smooth = smooth_2d(
        p_grid
    )

    q_smooth = smooth_2d(
        q_grid
    )

    # Old raw score
    q_residual = (
        q_grid - q_smooth
    )

    raw_score = (
        np.linalg.norm(
            q_residual
        )
        /
        (
            np.linalg.norm(
                q_grid
            )
            + 1e-30
        )
    )

    # Better artifact score:
    #
    # remove legitimate high-frequency structure
    # already present in target
    p_residual = (
        p_grid - p_smooth
    )

    artifact_residual = (
        q_residual
        -
        p_residual
    )

    artifact_score = (
        np.linalg.norm(
            artifact_residual
        )
        /
        (
            np.linalg.norm(
                p_grid
            )
            + 1e-30
        )
    )

    return (
        float(raw_score),
        float(artifact_score),
        artifact_residual,
    )


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

    p_grid = p.reshape(
        N_GRID,
        N_GRID,
    ).cpu().numpy()

    q_grid = q.reshape(
        N_GRID,
        N_GRID,
    ).cpu().numpy()

    (
        stripe_score,
        artifact_score,
        artifact_residual,
    ) = stripe_metrics(
        p_grid,
        q_grid,
    )

    return {
        "p":
            p_grid,

        "q":
            q_grid,

        "artifact_residual":
            artifact_residual,

        "stripe_score":
            stripe_score,

        "artifact_score":
            artifact_score,

        "peak_error":
            float(
                peak_error.cpu()
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
    }


# ============================================================
# Gate diagnostics
# ============================================================

@torch.no_grad()
def evaluate_gates(
    model,
    condition,
):

    mu_x, mu_y, sigma = condition

    N = 10000

    cond_raw = torch.tensor(
        [[
            mu_x,
            mu_y,
            sigma,
        ]],
        dtype=torch.float32,
        device=DEVICE,
    ).repeat(
        N,
        1,
    )

    cond_model = normalize_condition(
        cond_raw
    )

    u = torch.rand(
        N,
        2,
        device=DEVICE,
    )

    _, _, lambdas = model.forward(
        u,
        cond_model,
    )

    results = []

    for i, (
        lam_x,
        lam_y,
    ) in enumerate(
        lambdas
    ):

        results.append(
            {
                "block":
                    i,

                "lambda_x_mean":
                    float(
                        lam_x.mean().cpu()
                    ),

                "lambda_x_std":
                    float(
                        lam_x.std().cpu()
                    ),

                "lambda_y_mean":
                    float(
                        lam_y.mean().cpu()
                    ),

                "lambda_y_std":
                    float(
                        lam_y.std().cpu()
                    ),
            }
        )

    return results


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
    n_blocks,
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
        grid[
            "artifact_residual"
        ],
    ]

    titles = [
        "Target $p$",
        "Soft-gated flow $q$",
        "Artifact residual",
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
        label="target",
    )

    axes[1].scatter(
        grid["proposal_peak_x"],
        grid["proposal_peak_y"],
        marker="+",
        s=100,
        label="q",
    )

    axes[1].legend()

    mu_x, mu_y, sigma = condition

    fig.suptitle(
        f"{n_blocks} soft block(s) | "
        f"mu=({mu_x:.2f},{mu_y:.2f}), "
        f"sigma={sigma:.3f} | "
        f"artifact={grid['artifact_score']:.4f}"
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
            f"blocks_{n_blocks}_{tag}.png",
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

    summaries = {}

    for n_blocks in N_BLOCKS_LIST:

        model = train_model(
            n_blocks
        )

        results = []

        print()
        print("=" * 110)
        print(
            f"EVALUATING {n_blocks} SOFT BLOCK(S)"
        )
        print("=" * 110)

        for condition in TEST_CONDITIONS:

            grid = evaluate_grid(
                model,
                condition,
            )

            mc = evaluate_mc(
                model,
                condition,
            )

            gates = evaluate_gates(
                model,
                condition,
            )

            plot_result(
                n_blocks,
                condition,
                grid,
            )

            results.append(
                {
                    "condition":
                        list(condition),

                    "peak_error":
                        grid[
                            "peak_error"
                        ],

                    "stripe_score":
                        grid[
                            "stripe_score"
                        ],

                    "artifact_score":
                        grid[
                            "artifact_score"
                        ],

                    "ise":
                        grid[
                            "ise"
                        ],

                    "qmax":
                        grid[
                            "qmax"
                        ],

                    **mc,

                    "gates":
                        gates,
                }
            )

            print(
                f"c={condition} | "
                f"peak={grid['peak_error']:.4f} | "
                f"stripe={grid['stripe_score']:.4f} | "
                f"artifact={grid['artifact_score']:.4f} | "
                f"ISE={grid['ise']:.3e} | "
                f"ESS={mc['ess_fraction']:.3f}"
            )

        summaries[
            str(n_blocks)
        ] = {
            "mean_peak_error":
                float(
                    np.mean(
                        [
                            r["peak_error"]
                            for r in results
                        ]
                    )
                ),

            "mean_stripe":
                float(
                    np.mean(
                        [
                            r["stripe_score"]
                            for r in results
                        ]
                    )
                ),

            "mean_artifact":
                float(
                    np.mean(
                        [
                            r["artifact_score"]
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

            "results":
                results,
        }

    with open(
        os.path.join(
            OUT_DIR,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            summaries,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print(
        "SOFT-GATED CUBIC SWEEP"
    )
    print("=" * 100)

    print(
        f"{'blocks':>8}"
        f"{'peak':>12}"
        f"{'stripe':>12}"
        f"{'artifact':>14}"
        f"{'ISE':>16}"
        f"{'ESS':>12}"
    )

    print("-" * 100)

    for n_blocks in N_BLOCKS_LIST:

        s = summaries[
            str(n_blocks)
        ]

        print(
            f"{n_blocks:>8d}"
            f"{s['mean_peak_error']:>12.4f}"
            f"{s['mean_stripe']:>12.4f}"
            f"{s['mean_artifact']:>14.4f}"
            f"{s['mean_ise']:>16.3e}"
            f"{s['mean_ess_fraction']:>12.3f}"
        )


if __name__ == "__main__":
    main()