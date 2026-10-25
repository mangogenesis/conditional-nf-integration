# experiments/run_copula_bin_sweep.py

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nf.conditional_cubic_1d import ConditionalCubicFlow1D


# ============================================================
# Configuration
# ============================================================

DEFAULT_BINS = [8, 16, 32, 64]
DEFAULT_DIMS = [2, 4, 8]
DEFAULT_SEEDS = [0, 1, 2]

DEFAULT_BENCHMARKS = [
    "single_gaussian",
    "sharp_broad",
    "curved_ridge",
    "xshape",
]

# Five held-out conditions.
# Training conditions are sampled continuously from [0,1]^3.
TEST_CONDITIONS = torch.tensor(
    [
        [0.15, 0.20, 0.20],
        [0.25, 0.75, 0.80],
        [0.50, 0.50, 0.50],
        [0.75, 0.25, 0.70],
        [0.85, 0.80, 0.30],
    ],
    dtype=torch.float32,
)

COND_DIM = 3

HIDDEN = 128

MARGINAL_STEPS = 1200
COPULA_STEPS = 1800

BATCH_SIZE = 4096

LR_MARGINAL = 1e-4
LR_COPULA = 1e-4

GRAD_CLIP = 10.0

N_EVAL = 50_000
GRID_SIZE_2D = 120

OUTPUT_DIR = Path("results/copula_bin_sweep")


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def sample_conditions(n, device):
    return torch.rand(n, COND_DIM, device=device)


# ============================================================
# Truncated Normal
# ============================================================

SQRT2 = math.sqrt(2.0)
LOG_2PI = math.log(2.0 * math.pi)


def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / SQRT2))


def normal_icdf(p):
    p = p.clamp(1e-7, 1.0 - 1e-7)
    return SQRT2 * torch.erfinv(2.0 * p - 1.0)


def truncated_normal_sample(mu, sigma):
    """
    Sample TN(mu, sigma^2) restricted to [0,1].
    mu, sigma can have arbitrary matching shapes.
    """

    a = (0.0 - mu) / sigma
    b = (1.0 - mu) / sigma

    cdf_a = normal_cdf(a)
    cdf_b = normal_cdf(b)

    z = cdf_a + torch.rand_like(mu) * (cdf_b - cdf_a)
    x = mu + sigma * normal_icdf(z)

    return x.clamp(0.0, 1.0)


def truncated_normal_logpdf(x, mu, sigma):
    """
    Normalized log-density on [0,1].
    """

    z = (x - mu) / sigma

    log_base = (
        -0.5 * z.pow(2)
        - torch.log(sigma)
        - 0.5 * LOG_2PI
    )

    a = (0.0 - mu) / sigma
    b = (1.0 - mu) / sigma

    norm = normal_cdf(b) - normal_cdf(a)
    norm = norm.clamp_min(1e-12)

    return log_base - torch.log(norm)


# ============================================================
# Benchmark definitions
#
# Every target is normalized:
#
#       integral_[0,1]^d p(x | c) dx = 1
#
# Therefore our importance-sampling integral should equal 1.
# ============================================================

def gaussian_centers(cond, dim):
    """
    Gives a dimension-dependent moving center.

    For d=2:
        center ~= (mu, 1-mu)

    so this naturally reproduces the old (0.7,0.3)-type geometry.
    """

    mu = 0.20 + 0.60 * cond[:, 0:1]

    index = torch.arange(
        dim,
        device=cond.device,
        dtype=cond.dtype,
    )

    phase = 2.0 * math.pi * index / dim

    center = 0.5 + (mu - 0.5) * torch.cos(phase)[None, :]

    return center


def gaussian_sigma(cond):
    return 0.03 + 0.05 * cond[:, 1:2]


# ------------------------------------------------------------
# 1. Single Gaussian
# ------------------------------------------------------------

def sample_single_gaussian(cond, dim):
    center = gaussian_centers(cond, dim)
    sigma = gaussian_sigma(cond).expand(-1, dim)

    return truncated_normal_sample(center, sigma)


def logpdf_single_gaussian(x, cond):
    dim = x.shape[1]

    center = gaussian_centers(cond, dim)
    sigma = gaussian_sigma(cond).expand(-1, dim)

    return truncated_normal_logpdf(
        x,
        center,
        sigma,
    ).sum(dim=1)


# ------------------------------------------------------------
# 2. Sharp + broad mixture
# ------------------------------------------------------------

def sample_sharp_broad(cond, dim):
    n = cond.shape[0]

    sharp_center = gaussian_centers(cond, dim)
    sharp_sigma = gaussian_sigma(cond).expand(-1, dim)

    broad_center = torch.full_like(sharp_center, 0.5)
    broad_sigma = torch.full_like(sharp_center, 0.18)

    # Condition changes mixture composition.
    sharp_weight = 0.60 + 0.35 * cond[:, 2]

    choose_sharp = (
        torch.rand(n, device=cond.device) < sharp_weight
    )

    sharp = truncated_normal_sample(
        sharp_center,
        sharp_sigma,
    )

    broad = truncated_normal_sample(
        broad_center,
        broad_sigma,
    )

    return torch.where(
        choose_sharp[:, None],
        sharp,
        broad,
    )


def logpdf_sharp_broad(x, cond):
    dim = x.shape[1]

    sharp_center = gaussian_centers(cond, dim)
    sharp_sigma = gaussian_sigma(cond).expand(-1, dim)

    broad_center = torch.full_like(sharp_center, 0.5)
    broad_sigma = torch.full_like(sharp_center, 0.18)

    log_sharp = truncated_normal_logpdf(
        x,
        sharp_center,
        sharp_sigma,
    ).sum(dim=1)

    log_broad = truncated_normal_logpdf(
        x,
        broad_center,
        broad_sigma,
    ).sum(dim=1)

    w = 0.60 + 0.35 * cond[:, 2]

    return torch.logaddexp(
        torch.log(w) + log_sharp,
        torch.log1p(-w) + log_broad,
    )


# ------------------------------------------------------------
# 3. Curved ridge
#
# x1 ~ Uniform(0,1)
#
# x_j | x_{j-1}
#   ~ TN(
#       0.5 + A sin(2 pi frequency x_{j-1} + phase_j),
#       sigma
#      )
#
# This gives a nonlinear dependence chain in high dimension.
# ------------------------------------------------------------

def curved_parameters(cond):
    amplitude = 0.15 + 0.15 * cond[:, 0]
    sigma = 0.02 + 0.04 * cond[:, 1]
    frequency = 0.75 + 0.75 * cond[:, 2]

    return amplitude, sigma, frequency


def sample_curved_ridge(cond, dim):
    n = cond.shape[0]

    x = torch.empty(
        n,
        dim,
        device=cond.device,
        dtype=cond.dtype,
    )

    x[:, 0] = torch.rand(
        n,
        device=cond.device,
    )

    amp, sigma, freq = curved_parameters(cond)

    for j in range(1, dim):

        phase = 0.35 * j

        mu = (
            0.5
            + amp
            * torch.sin(
                2.0 * math.pi * freq * x[:, j - 1]
                + phase
            )
        )

        mu = mu.clamp(0.02, 0.98)

        x[:, j] = truncated_normal_sample(
            mu,
            sigma,
        )

    return x


def logpdf_curved_ridge(x, cond):
    dim = x.shape[1]

    amp, sigma, freq = curved_parameters(cond)

    logp = torch.zeros(
        x.shape[0],
        device=x.device,
    )

    # x1 is uniform -> log density = 0

    for j in range(1, dim):

        phase = 0.35 * j

        mu = (
            0.5
            + amp
            * torch.sin(
                2.0 * math.pi * freq * x[:, j - 1]
                + phase
            )
        )

        mu = mu.clamp(0.02, 0.98)

        logp += truncated_normal_logpdf(
            x[:, j],
            mu,
            sigma,
        )

    return logp


# ------------------------------------------------------------
# 4. X-shaped / branching dependence
#
# x1 ~ Uniform
#
# x_j | x_{j-1}
#
# either follows:
#
#       x_j ~= x_{j-1}
#
# or:
#
#       x_j ~= 1 - x_{j-1}
#
# with condition-dependent branch weight and strength.
# ------------------------------------------------------------

def xshape_parameters(cond):
    alpha = 0.75 + 0.24 * cond[:, 0]
    sigma = 0.02 + 0.04 * cond[:, 1]
    branch_weight = 0.25 + 0.50 * cond[:, 2]

    return alpha, sigma, branch_weight


def sample_xshape(cond, dim):
    n = cond.shape[0]

    x = torch.empty(
        n,
        dim,
        device=cond.device,
        dtype=cond.dtype,
    )

    x[:, 0] = torch.rand(
        n,
        device=cond.device,
    )

    alpha, sigma, weight = xshape_parameters(cond)

    for j in range(1, dim):

        previous = x[:, j - 1]

        mean_forward = (
            alpha * previous
            + (1.0 - alpha) * 0.5
        )

        mean_reverse = (
            alpha * (1.0 - previous)
            + (1.0 - alpha) * 0.5
        )

        choose_forward = (
            torch.rand(n, device=cond.device)
            < weight
        )

        mu = torch.where(
            choose_forward,
            mean_forward,
            mean_reverse,
        )

        x[:, j] = truncated_normal_sample(
            mu,
            sigma,
        )

    return x


def logpdf_xshape(x, cond):
    dim = x.shape[1]

    alpha, sigma, weight = xshape_parameters(cond)

    logp = torch.zeros(
        x.shape[0],
        device=x.device,
    )

    for j in range(1, dim):

        previous = x[:, j - 1]

        mean_forward = (
            alpha * previous
            + (1.0 - alpha) * 0.5
        )

        mean_reverse = (
            alpha * (1.0 - previous)
            + (1.0 - alpha) * 0.5
        )

        log_forward = truncated_normal_logpdf(
            x[:, j],
            mean_forward,
            sigma,
        )

        log_reverse = truncated_normal_logpdf(
            x[:, j],
            mean_reverse,
            sigma,
        )

        logp += torch.logaddexp(
            torch.log(weight) + log_forward,
            torch.log1p(-weight) + log_reverse,
        )

    return logp


# ============================================================
# Benchmark dispatcher
# ============================================================

def sample_target(name, cond, dim):

    if name == "single_gaussian":
        return sample_single_gaussian(cond, dim)

    if name == "sharp_broad":
        return sample_sharp_broad(cond, dim)

    if name == "curved_ridge":
        return sample_curved_ridge(cond, dim)

    if name == "xshape":
        return sample_xshape(cond, dim)

    raise ValueError(name)


def target_logpdf(name, x, cond):

    if name == "single_gaussian":
        return logpdf_single_gaussian(x, cond)

    if name == "sharp_broad":
        return logpdf_sharp_broad(x, cond)

    if name == "curved_ridge":
        return logpdf_curved_ridge(x, cond)

    if name == "xshape":
        return logpdf_xshape(x, cond)

    raise ValueError(name)


# ============================================================
# Generic d-dimensional marginal + autoregressive copula NF
# ============================================================

class MarginalCopulaFlow(nn.Module):

    def __init__(
        self,
        dim,
        cond_dim,
        hidden,
        n_bins,
    ):
        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim

        # Independent conditional marginal flows
        self.marginals = nn.ModuleList(
            [
                ConditionalCubicFlow1D(
                    cond_dim=cond_dim,
                    hidden=hidden,
                    n_bins=n_bins,
                )
                for _ in range(dim)
            ]
        )

        # Autoregressive copula:
        #
        # u1 = v1
        #
        # uj = Tj(
        #       vj |
        #       u1,...,u_{j-1}, c
        #      )
        #
        self.copulas = nn.ModuleList()

        for j in range(1, dim):

            context_dim = cond_dim + j

            self.copulas.append(
                ConditionalCubicFlow1D(
                    cond_dim=context_dim,
                    hidden=hidden,
                    n_bins=n_bins,
                )
            )

    # --------------------------------------------------------
    # Marginal inverse x -> u
    # --------------------------------------------------------

    def x_to_u(self, x, cond):

        u_list = []
        marginal_logq = torch.zeros(
            x.shape[0],
            device=x.device,
        )

        for j in range(self.dim):

            uj, forward_logdet = self.marginals[j].inverse(
                x[:, j:j+1],
                cond,
            )

            uj = uj.clamp(0.0, 1.0)

            u_list.append(uj)

            # Your validated 1D convention:
            #
            # inverse() returns forward log|dx/du|
            #
            # therefore
            #
            # log q(x) = - forward_logdet
            marginal_logq -= forward_logdet.reshape(-1)

        u = torch.cat(u_list, dim=1)

        return u, marginal_logq

    # --------------------------------------------------------
    # Full log density
    # --------------------------------------------------------

    def log_prob(self, x, cond):

        u, logq = self.x_to_u(x, cond)

        # u1 = v1 contributes no density term.
        for j in range(1, self.dim):

            context = torch.cat(
                [
                    u[:, :j],
                    cond,
                ],
                dim=1,
            )

            _, forward_logdet = self.copulas[j - 1].inverse(
                u[:, j:j+1],
                context,
            )

            # log copula =
            # - log |du_j / dv_j|
            logq -= forward_logdet.reshape(-1)

        return logq

    # --------------------------------------------------------
    # Sample v -> u -> x
    # --------------------------------------------------------

    @torch.no_grad()
    def sample(self, n, cond):

        device = cond.device

        if cond.shape[0] == 1:
            cond = cond.expand(n, -1)

        assert cond.shape[0] == n

        v = torch.rand(
            n,
            self.dim,
            device=device,
        )

        u = torch.zeros_like(v)

        u[:, 0] = v[:, 0]

        for j in range(1, self.dim):

            context = torch.cat(
                [
                    u[:, :j],
                    cond,
                ],
                dim=1,
            )

            uj, _ = self.copulas[j - 1].forward(
                v[:, j:j+1],
                context,
            )

            u[:, j] = uj.reshape(-1)

        x_list = []

        for j in range(self.dim):

            xj, _ = self.marginals[j].forward(
                u[:, j:j+1],
                cond,
            )

            x_list.append(xj)

        return torch.cat(x_list, dim=1)


# ============================================================
# Training
# ============================================================

def train_marginals(
    model,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
):

    for p in model.copulas.parameters():
        p.requires_grad_(False)

    for p in model.marginals.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        model.marginals.parameters(),
        lr=LR_MARGINAL,
    )

    model.train()

    losses = []

    for step in range(steps):

        cond = sample_conditions(
            batch_size,
            device,
        )

        x = sample_target(
            benchmark,
            cond,
            dim,
        )

        loss = 0.0

        for j in range(dim):

            _, forward_logdet = model.marginals[j].inverse(
                x[:, j:j+1],
                cond,
            )

            # NLL = forward logdet
            loss = loss + forward_logdet.mean()

        loss = loss / dim

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.marginals.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 250 == 0:
            print(
                f"  marginal "
                f"{step+1:5d}/{steps} "
                f"loss={np.mean(losses[-100:]):.6f}"
            )


def train_copula(
    model,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
):

    for p in model.marginals.parameters():
        p.requires_grad_(False)

    for p in model.copulas.parameters():
        p.requires_grad_(True)

    if dim == 1:
        return

    optimizer = torch.optim.Adam(
        model.copulas.parameters(),
        lr=LR_COPULA,
    )

    model.train()

    losses = []

    for step in range(steps):

        cond = sample_conditions(
            batch_size,
            device,
        )

        x = sample_target(
            benchmark,
            cond,
            dim,
        )

        with torch.no_grad():
            u, _ = model.x_to_u(
                x,
                cond,
            )

        loss = 0.0

        for j in range(1, dim):

            context = torch.cat(
                [
                    u[:, :j],
                    cond,
                ],
                dim=1,
            )

            _, forward_logdet = model.copulas[j - 1].inverse(
                u[:, j:j+1],
                context,
            )

            loss = loss + forward_logdet.mean()

        loss = loss / (dim - 1)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.copulas.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 250 == 0:
            print(
                f"  copula   "
                f"{step+1:5d}/{steps} "
                f"loss={np.mean(losses[-100:]):.6f}"
            )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_condition(
    model,
    benchmark,
    dim,
    cond,
    device,
    n_eval,
):

    model.eval()

    cond = cond.to(device).reshape(1, -1)

    cond_batch = cond.expand(n_eval, -1)

    start = time.perf_counter()

    x = model.sample(
        n_eval,
        cond,
    )

    logq = model.log_prob(
        x,
        cond_batch,
    )

    logp = target_logpdf(
        benchmark,
        x,
        cond_batch,
    )

    eval_time = time.perf_counter() - start

    logw = logp - logq

    # Prevent overflow from occasional numerical outliers.
    logw = logw.clamp(
        min=-50.0,
        max=50.0,
    )

    w = torch.exp(logw)

    integral_estimate = w.mean().item()

    relative_error = abs(
        integral_estimate - 1.0
    )

    variance_weights = w.var(
        unbiased=True
    ).item()

    variance_estimator = (
        variance_weights / n_eval
    )

    ess = (
        w.sum().pow(2)
        / w.pow(2).sum()
    ).item()

    ess_fraction = ess / n_eval

    return {
        "integral_estimate": integral_estimate,
        "relative_error": relative_error,
        "variance": variance_estimator,
        "weight_variance": variance_weights,
        "ess_fraction": ess_fraction,
        "eval_time": eval_time,
    }


# ============================================================
# Optional 2D density ISE
# ============================================================

@torch.no_grad()
def compute_2d_ise(
    model,
    benchmark,
    cond,
    device,
    grid_size,
):

    axis = (
        torch.arange(
            grid_size,
            device=device,
            dtype=torch.float32,
        )
        + 0.5
    ) / grid_size

    xx, yy = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    grid = torch.stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
        ],
        dim=1,
    )

    cond = cond.to(device).reshape(1, -1)

    cond_grid = cond.expand(
        grid.shape[0],
        -1,
    )

    logp = target_logpdf(
        benchmark,
        grid,
        cond_grid,
    )

    logq = model.log_prob(
        grid,
        cond_grid,
    )

    p = torch.exp(logp)
    q = torch.exp(logq)

    ise = torch.mean(
        (p - q).pow(2)
    ).item()

    return ise


# ============================================================
# Aggregation
# ============================================================

def build_summary(df):

    grouped = df.groupby(
        [
            "benchmark",
            "dimension",
            "bins",
        ]
    )

    summary = grouped.agg(
        ess_mean=("ess_fraction", "mean"),
        ess_std=("ess_fraction", "std"),

        variance_mean=("variance", "mean"),
        variance_std=("variance", "std"),

        relative_error_mean=("relative_error", "mean"),
        relative_error_std=("relative_error", "std"),

        train_time_mean=("train_time", "mean"),
        eval_time_mean=("eval_time", "mean"),

        parameter_count=("parameter_count", "mean"),

        ise_2d_mean=("ise_2d", "mean"),
    ).reset_index()

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_metric(
    summary,
    metric,
    ylabel,
    output_dir,
    logy=False,
):

    for benchmark in summary["benchmark"].unique():

        subset = summary[
            summary["benchmark"] == benchmark
        ]

        plt.figure(figsize=(7, 5))

        for dim in sorted(
            subset["dimension"].unique()
        ):

            dsub = subset[
                subset["dimension"] == dim
            ].sort_values("bins")

            plt.plot(
                dsub["bins"],
                dsub[metric],
                marker="o",
                label=f"d={dim}",
            )

        plt.xlabel("Number of spline bins")
        plt.ylabel(ylabel)

        if logy:
            plt.yscale("log")

        plt.title(
            benchmark.replace("_", " ").title()
        )

        plt.legend()
        plt.tight_layout()

        path = (
            output_dir
            / f"{benchmark}_{metric}.png"
        )

        plt.savefig(
            path,
            dpi=180,
        )

        plt.close()


def make_plots(summary, plot_dir):

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_metric(
        summary,
        "ess_mean",
        "ESS / N",
        plot_dir,
    )

    plot_metric(
        summary,
        "variance_mean",
        "Variance of integral estimator",
        plot_dir,
        logy=True,
    )

    plot_metric(
        summary,
        "relative_error_mean",
        "Mean relative error",
        plot_dir,
        logy=True,
    )

    plot_metric(
        summary,
        "train_time_mean",
        "Training time [s]",
        plot_dir,
    )


# ============================================================
# Main sweep
# ============================================================

def run(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}\n")

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir = output_dir / "plots"

    config = {
        "benchmarks": args.benchmarks,
        "dimensions": args.dims,
        "bins": args.bins,
        "seeds": args.seeds,
        "n_conditions": len(TEST_CONDITIONS),

        "marginal_steps": args.marginal_steps,
        "copula_steps": args.copula_steps,

        "batch_size": args.batch_size,
        "n_eval": args.n_eval,

        "hidden": HIDDEN,
        "cond_dim": COND_DIM,

        "device": str(device),
    }

    with open(
        output_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    raw_rows = []

    total_models = (
        len(args.benchmarks)
        * len(args.dims)
        * len(args.bins)
        * len(args.seeds)
    )

    model_index = 0

    for benchmark in args.benchmarks:

        for dim in args.dims:

            for bins in args.bins:

                for seed in args.seeds:

                    model_index += 1

                    print("\n" + "=" * 70)
                    print(
                        f"[{model_index}/{total_models}] "
                        f"{benchmark} | "
                        f"d={dim} | "
                        f"bins={bins} | "
                        f"seed={seed}"
                    )
                    print("=" * 70)

                    seed_everything(seed)

                    model = MarginalCopulaFlow(
                        dim=dim,
                        cond_dim=COND_DIM,
                        hidden=HIDDEN,
                        n_bins=bins,
                    ).to(device)

                    n_params = parameter_count(model)

                    train_start = time.perf_counter()

                    train_marginals(
                        model=model,
                        benchmark=benchmark,
                        dim=dim,
                        device=device,
                        steps=args.marginal_steps,
                        batch_size=args.batch_size,
                    )

                    train_copula(
                        model=model,
                        benchmark=benchmark,
                        dim=dim,
                        device=device,
                        steps=args.copula_steps,
                        batch_size=args.batch_size,
                    )

                    train_time = (
                        time.perf_counter()
                        - train_start
                    )

                    print(
                        f"  train time: "
                        f"{train_time:.1f} s"
                    )

                    for condition_id, cond in enumerate(
                        TEST_CONDITIONS
                    ):

                        metrics = evaluate_condition(
                            model=model,
                            benchmark=benchmark,
                            dim=dim,
                            cond=cond,
                            device=device,
                            n_eval=args.n_eval,
                        )

                        if dim == 2:

                            ise = compute_2d_ise(
                                model=model,
                                benchmark=benchmark,
                                cond=cond,
                                device=device,
                                grid_size=GRID_SIZE_2D,
                            )

                        else:
                            ise = np.nan

                        row = {
                            "benchmark": benchmark,
                            "dimension": dim,
                            "bins": bins,
                            "seed": seed,

                            "condition_id": condition_id,

                            "condition_0": float(cond[0]),
                            "condition_1": float(cond[1]),
                            "condition_2": float(cond[2]),

                            "integral_true": 1.0,

                            "integral_estimate":
                                metrics[
                                    "integral_estimate"
                                ],

                            "relative_error":
                                metrics[
                                    "relative_error"
                                ],

                            "variance":
                                metrics[
                                    "variance"
                                ],

                            "weight_variance":
                                metrics[
                                    "weight_variance"
                                ],

                            "ess_fraction":
                                metrics[
                                    "ess_fraction"
                                ],

                            "train_time":
                                train_time,

                            "eval_time":
                                metrics[
                                    "eval_time"
                                ],

                            "parameter_count":
                                n_params,

                            "ise_2d":
                                ise,
                        }

                        raw_rows.append(row)

                        print(
                            f"  c{condition_id}: "
                            f"ESS={row['ess_fraction']:.3f} "
                            f"var={row['variance']:.3e} "
                            f"rel={row['relative_error']:.3e}"
                        )

                    # Save continuously in case a long run stops.
                    pd.DataFrame(
                        raw_rows
                    ).to_csv(
                        output_dir
                        / "raw_results.csv",
                        index=False,
                    )

                    del model

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    raw_df = pd.DataFrame(raw_rows)

    summary = build_summary(
        raw_df
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    make_plots(
        summary,
        plot_dir,
    )

    print("\n")
    print("=" * 70)
    print("BIN SWEEP COMPLETE")
    print("=" * 70)

    display_columns = [
        "benchmark",
        "dimension",
        "bins",
        "ess_mean",
        "variance_mean",
        "relative_error_mean",
        "train_time_mean",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bins",
        nargs="+",
        type=int,
        default=DEFAULT_BINS,
    )

    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=DEFAULT_DIMS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
    )

    parser.add_argument(
        "--marginal-steps",
        type=int,
        default=MARGINAL_STEPS,
    )

    parser.add_argument(
        "--copula-steps",
        type=int,
        default=COPULA_STEPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--n-eval",
        type=int,
        default=N_EVAL,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    if args.smoke:

        args.bins = [16, 32]
        args.dims = [2]
        args.seeds = [0]

        args.benchmarks = [
            "single_gaussian",
            "curved_ridge",
        ]

        args.marginal_steps = 100
        args.copula_steps = 150

        args.batch_size = 1024
        args.n_eval = 5000

        args.output = (
            "results/copula_bin_sweep_smoke"
        )

    return args


if __name__ == "__main__":

    args = parse_args()

    run(args)