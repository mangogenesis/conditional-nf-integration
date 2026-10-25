import os
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from nf.conditional_cubic_1d import ConditionalCubicFlow1D


# ============================================================
# Settings
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "results/conditional_cubic_bin_sweep_1d"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
SUMMARY_DIR = os.path.join(OUT_DIR, "summaries")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

SEED = 1234

N_COND = 1024
N_X_PER_COND = 512
STEPS = 10000
LR = 5e-5
HIDDEN = 128

MU_RANGE = (0.20, 0.80)
SIGMA_RANGE = (0.03, 0.12)

FLOOR = 1e-6
N_EVAL = 100000
N_GRID = 2000

CUBIC_BINS = [4, 6, 8, 16, 32, 64, 128]

TEST_CONDITIONS = [
    (0.30, 0.05),
    (0.50, 0.08),
    (0.75, 0.04),
    (0.25, 0.10),
    (0.65, 0.06),
    (0.50, 0.03),
    (0.50, 0.12),
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


def target_unnormalized(x, mu, sigma, floor=FLOOR):
    return floor + torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def exact_integral_1d(mu, sigma, floor=FLOOR):
    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    a = (0.0 - mu) / sigma
    b = (1.0 - mu) / sigma

    peak = sigma * math.sqrt(2.0 * math.pi) * (Phi(b) - Phi(a))

    return floor + peak


def sample_grouped_batch():
    """
    Samples grouped training data.

    Returns:
        x_flat:    shape (N_COND * N_X_PER_COND, 1)
        cond_flat: shape (N_COND * N_X_PER_COND, 2)
    """
    mu = torch.empty(N_COND, 1, device=DEVICE).uniform_(*MU_RANGE)
    sigma = torch.empty(N_COND, 1, device=DEVICE).uniform_(*SIGMA_RANGE)

    cond = torch.cat([mu, sigma], dim=1)

    x = torch.rand(N_COND, N_X_PER_COND, 1, device=DEVICE)

    cond_expanded = cond[:, None, :].expand(N_COND, N_X_PER_COND, 2)

    x_flat = x.reshape(-1, 1)
    cond_flat = cond_expanded.reshape(-1, 2)

    return x_flat, cond_flat


def unpack_forward(output):
    """
    Some flow implementations return x.
    Some return (x, logdet).
    This handles both.
    """
    if isinstance(output, tuple):
        return output[0]
    return output


def unpack_sample(output):
    """
    Some sample implementations return x.
    Some return (x, log_prob).
    This handles both.
    """
    if isinstance(output, tuple):
        return output[0]
    return output


# ============================================================
# Model
# ============================================================

def build_model(n_bins):
    """
    Change this constructor if your ConditionalCubicFlow1D uses different arguments.
    """
    model = ConditionalCubicFlow1D(
        cond_dim=2,
        hidden=HIDDEN,
        n_bins=n_bins,
    ).to(DEVICE)

    return model


# ============================================================
# Training
# ============================================================

def train_one_model(n_bins):
    name = f"cubic_bins{n_bins}_noreg"

    print("\n" + "=" * 80)
    print(f"Training {name}")
    print("=" * 80)

    set_seed(SEED)

    model = build_model(n_bins)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_history = []

    for step in range(1, STEPS + 1):
        x_flat, cond_flat = sample_grouped_batch()

        mu_flat = cond_flat[:, 0:1]
        sigma_flat = cond_flat[:, 1:2]

        f_vals = target_unnormalized(x_flat, mu_flat, sigma_flat)
        log_q = model.log_prob(x_flat, cond_flat)

        f_vals = f_vals.reshape(N_COND, N_X_PER_COND)
        log_q = log_q.reshape(N_COND, N_X_PER_COND)

        weights = f_vals.detach()

        loss_per_cond = -(
            weights * log_q
        ).sum(dim=1) / (weights.sum(dim=1) + 1e-12)

        loss = loss_per_cond.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
            }
        )

        if step % 500 == 0 or step == 1:
            print(f"step {step:05d} | loss={loss.item():.6f}")

    ckpt_path = os.path.join(CKPT_DIR, f"{name}.pt")
    torch.save(model.state_dict(), ckpt_path)

    loss_path = os.path.join(SUMMARY_DIR, f"{name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)

    return model, name


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_integral(model, name, n_bins, mu, sigma):
    cond = torch.tensor([[mu, sigma]], device=DEVICE).repeat(N_EVAL, 1)

    samples = model.sample(N_EVAL, cond)
    x = unpack_sample(samples).reshape(-1, 1)

    log_q = model.log_prob(x, cond).reshape(-1)
    q = torch.exp(log_q)

    mu_t = torch.full_like(x, mu)
    sigma_t = torch.full_like(x, sigma)

    f_vals = target_unnormalized(x, mu_t, sigma_t).reshape(-1)

    weights = f_vals / (q + 1e-30)

    estimate = weights.mean()
    weight_variance = weights.var(unbiased=True)
    variance = weight_variance / N_EVAL
    std_error = torch.sqrt(variance)

    integral_ref = exact_integral_1d(mu, sigma)

    abs_error = torch.abs(estimate - integral_ref)
    rel_error = abs_error / abs(integral_ref)

    pull = (estimate - integral_ref) / (std_error + 1e-30)

    ess = weights.sum() ** 2 / (torch.sum(weights ** 2) + 1e-30)

    return {
        "config_name": name,
        "model_type": "cubic",
        "n_bins": n_bins,
        "mu": mu,
        "sigma": sigma,
        "estimate": float(estimate.cpu()),
        "integral_ref": float(integral_ref),
        "weight_variance": float(weight_variance.cpu()),
        "variance": float(variance.cpu()),
        "std_error": float(std_error.cpu()),
        "abs_error": float(abs_error.cpu()),
        "rel_error": float(rel_error.cpu()),
        "pull": float(pull.cpu()),
        "ess": float(ess.cpu()),
        "events": N_EVAL,
    }


def evaluate_density_and_plot(model, name, n_bins, mu, sigma):
    x_grid = torch.linspace(0.0, 1.0, N_GRID, device=DEVICE).reshape(-1, 1)
    cond_grid = torch.tensor([[mu, sigma]], device=DEVICE).repeat(N_GRID, 1)

    with torch.no_grad():
        log_q = model.log_prob(x_grid, cond_grid).reshape(-1)
        q = torch.exp(log_q)

        f = target_unnormalized(
            x_grid,
            torch.full_like(x_grid, mu),
            torch.full_like(x_grid, sigma),
        ).reshape(-1)

        integral_ref = exact_integral_1d(mu, sigma)
        p = f / integral_ref

        density_ise = torch.mean((q - p) ** 2)
        density_mae = torch.mean(torch.abs(q - p))

        p_max = torch.max(p)
        important = p > 0.01 * p_max

        log_ratio = torch.log(p + 1e-30) - torch.log(q + 1e-30)

        max_abs_log_ratio_important = torch.max(
            torch.abs(log_ratio[important])
        )

    # --------------------------------------------------------
    # Derivative dT/du by autograd
    # --------------------------------------------------------
    u_grid = torch.linspace(0.0, 1.0, N_GRID, device=DEVICE).reshape(-1, 1)
    u_grid.requires_grad_(True)

    cond_u = torch.tensor([[mu, sigma]], device=DEVICE).repeat(N_GRID, 1)

    x_from_u = unpack_forward(model.forward(u_grid, cond_u))

    dT_du = torch.autograd.grad(
        outputs=x_from_u.sum(),
        inputs=u_grid,
        create_graph=False,
        retain_graph=False,
    )[0]

    # --------------------------------------------------------
    # Move arrays to CPU
    # --------------------------------------------------------
    x_np = x_grid.detach().cpu().numpy().reshape(-1)
    q_np = q.detach().cpu().numpy().reshape(-1)
    p_np = p.detach().cpu().numpy().reshape(-1)
    log_ratio_np = log_ratio.detach().cpu().numpy().reshape(-1)

    u_np = u_grid.detach().cpu().numpy().reshape(-1)
    x_u_np = x_from_u.detach().cpu().numpy().reshape(-1)
    dT_du_np = dT_du.detach().cpu().numpy().reshape(-1)

    # --------------------------------------------------------
    # Density plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_np, p_np, label="target $f/I_{ref}$")
    ax.plot(x_np, q_np, label="$q_\\theta$")
    ax.set_title(
        f"{name}: $\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()

    density_path = os.path.join(
        PLOT_DIR,
        f"{name}_mu{mu:.2f}_sigma{sigma:.2f}_density.png",
    )
    fig.savefig(density_path, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Log-ratio plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_np, log_ratio_np)
    ax.axhline(0.0, linewidth=1)
    ax.set_title(
        f"{name}: log target - log proposal, "
        f"$\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\log(f/I_{ref}) - \\log q_\\theta$")
    fig.tight_layout()

    logratio_path = os.path.join(
        PLOT_DIR,
        f"{name}_mu{mu:.2f}_sigma{sigma:.2f}_logratio.png",
    )
    fig.savefig(logratio_path, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Transform derivative plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(u_np, dT_du_np)
    ax.set_title(
        f"{name}: transform derivative, "
        f"$\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$"
    )
    ax.set_xlabel("$u$")
    ax.set_ylabel("$dT_\\theta/du$")
    fig.tight_layout()

    derivative_path = os.path.join(
        PLOT_DIR,
        f"{name}_mu{mu:.2f}_sigma{sigma:.2f}_derivative.png",
    )
    fig.savefig(derivative_path, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Transform plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(u_np, x_u_np)
    ax.set_title(
        f"{name}: transform $x=T_\\theta(u|c)$, "
        f"$\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$"
    )
    ax.set_xlabel("$u$")
    ax.set_ylabel("$x$")
    fig.tight_layout()

    transform_path = os.path.join(
        PLOT_DIR,
        f"{name}_mu{mu:.2f}_sigma{sigma:.2f}_transform.png",
    )
    fig.savefig(transform_path, dpi=160)
    plt.close(fig)

    return {
        "density_ise": float(density_ise.detach().cpu()),
        "density_mae": float(density_mae.detach().cpu()),
        "max_abs_log_ratio_important": float(
            max_abs_log_ratio_important.detach().cpu()
        ),
    }


def evaluate_model(model, name, n_bins):
    all_results = []

    for mu, sigma in TEST_CONDITIONS:
        result = evaluate_integral(model, name, n_bins, mu, sigma)
        density_result = evaluate_density_and_plot(model, name, n_bins, mu, sigma)

        result.update(density_result)
        result["label"] = f"{name} mu={mu:.2f}, sigma={sigma:.2f}"

        all_results.append(result)

        print(
            f"{name} | mu={mu:.2f}, sigma={sigma:.2f} | "
            f"var={result['variance']:.3e} | "
            f"rel={result['rel_error']:.3e} | "
            f"ESS={result['ess']:.3e} | "
            f"ISE={result['density_ise']:.3e} | "
            f"maxlog={result['max_abs_log_ratio_important']:.3e}"
        )

    return all_results


# ============================================================
# Contact sheets
# ============================================================

def make_contact_sheet_for_model(name):
    """
    Creates one contact sheet containing density plots for all test conditions.
    """
    import matplotlib.image as mpimg

    image_paths = []

    for mu, sigma in TEST_CONDITIONS:
        path = os.path.join(
            PLOT_DIR,
            f"{name}_mu{mu:.2f}_sigma{sigma:.2f}_density.png",
        )
        image_paths.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, path in zip(axes, image_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    fig.suptitle(name, fontsize=16)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{name}_contact_sheet.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved contact sheet: {out_path}")


def make_derivative_contact_sheet_for_model(name):
    """
    Creates one contact sheet containing derivative plots for all test conditions.
    """
    import matplotlib.image as mpimg

    image_paths = []

    for mu, sigma in TEST_CONDITIONS:
        path = os.path.join(
            PLOT_DIR,
            f"{name}_mu{mu:.2f}_sigma{sigma:.2f}_derivative.png",
        )
        image_paths.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, path in zip(axes, image_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    fig.suptitle(f"{name}: transform derivative", fontsize=16)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{name}_derivative_contact_sheet.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved derivative contact sheet: {out_path}")


def make_logratio_contact_sheet_for_model(name):
    """
    Creates one contact sheet containing log-ratio plots for all test conditions.
    """
    import matplotlib.image as mpimg

    image_paths = []

    for mu, sigma in TEST_CONDITIONS:
        path = os.path.join(
            PLOT_DIR,
            f"{name}_mu{mu:.2f}_sigma{sigma:.2f}_logratio.png",
        )
        image_paths.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, path in zip(axes, image_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    fig.suptitle(f"{name}: log-ratio", fontsize=16)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{name}_logratio_contact_sheet.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved log-ratio contact sheet: {out_path}")


# ============================================================
# Summary
# ============================================================

def summarize_results(all_results):
    grouped = defaultdict(list)

    for r in all_results:
        grouped[r["config_name"]].append(r)

    summary = []

    for name, rows in grouped.items():
        summary_row = {
            "config_name": name,
            "model_type": "cubic",
            "n_bins": rows[0]["n_bins"],
            "mean_variance": float(np.mean([r["variance"] for r in rows])),
            "mean_rel_error": float(np.mean([r["rel_error"] for r in rows])),
            "mean_ess": float(np.mean([r["ess"] for r in rows])),
            "mean_density_ise": float(np.mean([r["density_ise"] for r in rows])),
            "mean_density_mae": float(np.mean([r["density_mae"] for r in rows])),
            "mean_max_abs_log_ratio_important": float(
                np.mean([r["max_abs_log_ratio_important"] for r in rows])
            ),
        }

        summary.append(summary_row)

    summary = sorted(summary, key=lambda x: x["n_bins"])

    return summary


# ============================================================
# Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Cubic bins: {CUBIC_BINS}")

    all_results = []

    for n_bins in CUBIC_BINS:
        model, name = train_one_model(n_bins)

        results = evaluate_model(model, name, n_bins)
        all_results.extend(results)

        make_contact_sheet_for_model(name)
        make_derivative_contact_sheet_for_model(name)
        make_logratio_contact_sheet_for_model(name)

        result_path = os.path.join(SUMMARY_DIR, f"{name}_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

    summary = summarize_results(all_results)

    all_results_path = os.path.join(
        SUMMARY_DIR,
        "all_cubic_bin_sweep_results.json",
    )
    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_path = os.path.join(
        SUMMARY_DIR,
        "cubic_bin_sweep_summary.json",
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("CUBIC BIN SWEEP SUMMARY")
    print("=" * 80)

    for row in summary:
        print(
            f"{row['config_name']:24s} | "
            f"bins={row['n_bins']:<4d} | "
            f"mean variance={row['mean_variance']:.3e} | "
            f"mean rel_err={row['mean_rel_error']:.3e} | "
            f"mean ESS={row['mean_ess']:.3e} | "
            f"mean ISE={row['mean_density_ise']:.3e} | "
            f"mean max log-ratio={row['mean_max_abs_log_ratio_important']:.3e}"
        )

    print("\nSaved:")
    print(f"  All results: {all_results_path}")
    print(f"  Summary:     {summary_path}")
    print(f"  Plots:       {PLOT_DIR}")


if __name__ == "__main__":
    main()