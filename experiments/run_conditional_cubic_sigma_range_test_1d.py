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

OUT_DIR = "results/conditional_cubic_sigma_range_test_1d"
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

# ------------------------------------------------------------
# Main comparison:
# same model, same benchmark, only sigma training domain changes
# ------------------------------------------------------------

TRAIN_CONFIGS = [
    {
        "name": "cubic_bins32_sigma_original",
        "sigma_range": (0.03, 0.12),
    },
    {
        "name": "cubic_bins32_sigma_extended",
        "sigma_range": (0.02, 0.14),
    },
]

N_BINS = 32

FLOOR = 1e-6
N_EVAL = 100000
N_GRID = 2000


# Same test conditions as before
TEST_CONDITIONS = [
    (0.30, 0.05),
    (0.50, 0.08),
    (0.75, 0.04),
    (0.25, 0.10),
    (0.65, 0.06),

    # Original sigma boundary cases
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
    return floor + torch.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def exact_integral_1d(mu, sigma, floor=FLOOR):
    def Phi(z):
        return 0.5 * (
            1.0 + math.erf(z / math.sqrt(2.0))
        )

    a = (0.0 - mu) / sigma
    b = (1.0 - mu) / sigma

    peak = (
        sigma
        * math.sqrt(2.0 * math.pi)
        * (Phi(b) - Phi(a))
    )

    return floor + peak


def sample_grouped_batch(sigma_range):
    """
    Sample conditions and x-points for grouped conditional training.

    Conditions:
        mu    ~ Uniform(MU_RANGE)
        sigma ~ Uniform(sigma_range)

    x:
        x ~ Uniform(0,1)

    Returns:
        x_flat:    (N_COND*N_X_PER_COND, 1)
        cond_flat: (N_COND*N_X_PER_COND, 2)
    """

    mu = torch.empty(
        N_COND, 1, device=DEVICE
    ).uniform_(*MU_RANGE)

    sigma = torch.empty(
        N_COND, 1, device=DEVICE
    ).uniform_(*sigma_range)

    cond = torch.cat([mu, sigma], dim=1)

    x = torch.rand(
        N_COND,
        N_X_PER_COND,
        1,
        device=DEVICE,
    )

    cond_expanded = cond[:, None, :].expand(
        N_COND,
        N_X_PER_COND,
        2,
    )

    x_flat = x.reshape(-1, 1)
    cond_flat = cond_expanded.reshape(-1, 2)

    return x_flat, cond_flat


def unpack_forward(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def unpack_sample(output):
    if isinstance(output, tuple):
        return output[0]
    return output


# ============================================================
# Model
# ============================================================

def build_model():
    """
    Modify constructor here if your class uses different names.
    """

    model = ConditionalCubicFlow1D(
        cond_dim=2,
        hidden=HIDDEN,
        n_bins=N_BINS,
    ).to(DEVICE)

    return model


# ============================================================
# Training
# ============================================================

def train_one_model(config):
    name = config["name"]
    sigma_range = config["sigma_range"]

    print("\n" + "=" * 80)
    print(f"Training {name}")
    print(f"sigma training range = {sigma_range}")
    print("=" * 80)

    set_seed(SEED)

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loss_history = []

    for step in range(1, STEPS + 1):

        x_flat, cond_flat = sample_grouped_batch(
            sigma_range
        )

        mu_flat = cond_flat[:, 0:1]
        sigma_flat = cond_flat[:, 1:2]

        f_vals = target_unnormalized(
            x_flat,
            mu_flat,
            sigma_flat,
        )

        log_q = model.log_prob(
            x_flat,
            cond_flat,
        )

        f_vals = f_vals.reshape(
            N_COND,
            N_X_PER_COND,
        )

        log_q = log_q.reshape(
            N_COND,
            N_X_PER_COND,
        )

        weights = f_vals.detach()

        # Normalize each condition separately
        loss_per_cond = -(
            weights * log_q
        ).sum(dim=1) / (
            weights.sum(dim=1) + 1e-12
        )

        loss = loss_per_cond.mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        loss_history.append(
            {
                "step": step,
                "loss": float(
                    loss.detach().cpu()
                ),
            }
        )

        if step == 1 or step % 500 == 0:
            print(
                f"step {step:05d} | "
                f"loss={loss.item():.6f}"
            )

    # Save checkpoint
    ckpt_path = os.path.join(
        CKPT_DIR,
        f"{name}.pt",
    )

    torch.save(
        model.state_dict(),
        ckpt_path,
    )

    # Save loss history
    loss_path = os.path.join(
        SUMMARY_DIR,
        f"{name}_loss_history.json",
    )

    with open(loss_path, "w") as f:
        json.dump(
            loss_history,
            f,
            indent=2,
        )

    return model


# ============================================================
# Integral Evaluation
# ============================================================

@torch.no_grad()
def evaluate_integral(
    model,
    config,
    mu,
    sigma,
):
    name = config["name"]

    cond = torch.tensor(
        [[mu, sigma]],
        device=DEVICE,
    ).repeat(N_EVAL, 1)

    samples = model.sample(
        N_EVAL,
        cond,
    )

    x = unpack_sample(samples).reshape(-1, 1)

    log_q = model.log_prob(
        x,
        cond,
    ).reshape(-1)

    q = torch.exp(log_q)

    mu_t = torch.full_like(
        x,
        mu,
    )

    sigma_t = torch.full_like(
        x,
        sigma,
    )

    f_vals = target_unnormalized(
        x,
        mu_t,
        sigma_t,
    ).reshape(-1)

    weights = f_vals / (
        q + 1e-30
    )

    estimate = weights.mean()

    weight_variance = weights.var(
        unbiased=True
    )

    variance = (
        weight_variance / N_EVAL
    )

    std_error = torch.sqrt(
        variance
    )

    integral_ref = exact_integral_1d(
        mu,
        sigma,
    )

    abs_error = torch.abs(
        estimate - integral_ref
    )

    rel_error = (
        abs_error
        / abs(integral_ref)
    )

    pull = (
        estimate - integral_ref
    ) / (
        std_error + 1e-30
    )

    ess = (
        weights.sum() ** 2
        / (
            torch.sum(weights ** 2)
            + 1e-30
        )
    )

    return {
        "config_name": name,
        "n_bins": N_BINS,
        "sigma_train_min": config["sigma_range"][0],
        "sigma_train_max": config["sigma_range"][1],

        "mu": mu,
        "sigma": sigma,

        "estimate": float(
            estimate.cpu()
        ),

        "integral_ref": float(
            integral_ref
        ),

        "weight_variance": float(
            weight_variance.cpu()
        ),

        "variance": float(
            variance.cpu()
        ),

        "std_error": float(
            std_error.cpu()
        ),

        "abs_error": float(
            abs_error.cpu()
        ),

        "rel_error": float(
            rel_error.cpu()
        ),

        "pull": float(
            pull.cpu()
        ),

        "ess": float(
            ess.cpu()
        ),

        "events": N_EVAL,
    }


# ============================================================
# Density Evaluation + Plots
# ============================================================

def evaluate_density_and_plot(
    model,
    config,
    mu,
    sigma,
):
    name = config["name"]

    x_grid = torch.linspace(
        0.0,
        1.0,
        N_GRID,
        device=DEVICE,
    ).reshape(-1, 1)

    cond_grid = torch.tensor(
        [[mu, sigma]],
        device=DEVICE,
    ).repeat(N_GRID, 1)

    # --------------------------------------------------------
    # Density
    # --------------------------------------------------------

    with torch.no_grad():

        log_q = model.log_prob(
            x_grid,
            cond_grid,
        ).reshape(-1)

        q = torch.exp(log_q)

        f = target_unnormalized(
            x_grid,
            torch.full_like(
                x_grid,
                mu,
            ),
            torch.full_like(
                x_grid,
                sigma,
            ),
        ).reshape(-1)

        integral_ref = exact_integral_1d(
            mu,
            sigma,
        )

        p = f / integral_ref

        density_ise = torch.mean(
            (q - p) ** 2
        )

        density_mae = torch.mean(
            torch.abs(q - p)
        )

        p_max = torch.max(p)

        important = (
            p > 0.01 * p_max
        )

        log_ratio = (
            torch.log(p + 1e-30)
            -
            torch.log(q + 1e-30)
        )

        max_abs_log_ratio_important = (
            torch.max(
                torch.abs(
                    log_ratio[important]
                )
            )
        )

    # --------------------------------------------------------
    # Transform derivative
    # --------------------------------------------------------

    u_grid = torch.linspace(
        0.0,
        1.0,
        N_GRID,
        device=DEVICE,
    ).reshape(-1, 1)

    u_grid.requires_grad_(True)

    cond_u = torch.tensor(
        [[mu, sigma]],
        device=DEVICE,
    ).repeat(N_GRID, 1)

    x_from_u = unpack_forward(
        model.forward(
            u_grid,
            cond_u,
        )
    )

    dT_du = torch.autograd.grad(
        outputs=x_from_u.sum(),
        inputs=u_grid,
        create_graph=False,
        retain_graph=False,
    )[0]

    # --------------------------------------------------------
    # CPU arrays
    # --------------------------------------------------------

    x_np = (
        x_grid.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    q_np = (
        q.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    p_np = (
        p.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    log_ratio_np = (
        log_ratio.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    u_np = (
        u_grid.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    dT_du_np = (
        dT_du.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # --------------------------------------------------------
    # Density plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        x_np,
        p_np,
        label="target $f/I_{ref}$",
    )

    ax.plot(
        x_np,
        q_np,
        label="$q_\\theta$",
    )

    ax.set_title(
        f"{name}: "
        f"$\\mu={mu:.2f}$, "
        f"$\\sigma={sigma:.2f}$"
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.legend()

    fig.tight_layout()

    density_path = os.path.join(
        PLOT_DIR,
        f"{name}_"
        f"mu{mu:.2f}_"
        f"sigma{sigma:.2f}_"
        f"density.png",
    )

    fig.savefig(
        density_path,
        dpi=160,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Log ratio
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        x_np,
        log_ratio_np,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel("$x$")

    ax.set_ylabel(
        "$\\log(f/I_{ref})"
        "-\\log q_\\theta$"
    )

    ax.set_title(
        f"{name}: "
        f"$\\mu={mu:.2f}$, "
        f"$\\sigma={sigma:.2f}$"
    )

    fig.tight_layout()

    logratio_path = os.path.join(
        PLOT_DIR,
        f"{name}_"
        f"mu{mu:.2f}_"
        f"sigma{sigma:.2f}_"
        f"logratio.png",
    )

    fig.savefig(
        logratio_path,
        dpi=160,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Derivative
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        u_np,
        dT_du_np,
    )

    ax.set_xlabel("$u$")
    ax.set_ylabel("$dT_\\theta/du$")

    ax.set_title(
        f"{name}: transform derivative "
        f"$\\mu={mu:.2f}$, "
        f"$\\sigma={sigma:.2f}$"
    )

    fig.tight_layout()

    derivative_path = os.path.join(
        PLOT_DIR,
        f"{name}_"
        f"mu{mu:.2f}_"
        f"sigma{sigma:.2f}_"
        f"derivative.png",
    )

    fig.savefig(
        derivative_path,
        dpi=160,
    )

    plt.close(fig)

    return {
        "density_ise": float(
            density_ise.detach().cpu()
        ),

        "density_mae": float(
            density_mae.detach().cpu()
        ),

        "max_abs_log_ratio_important": float(
            max_abs_log_ratio_important
            .detach()
            .cpu()
        ),
    }


# ============================================================
# Evaluate full model
# ============================================================

def evaluate_model(
    model,
    config,
):
    all_results = []

    for mu, sigma in TEST_CONDITIONS:

        result = evaluate_integral(
            model,
            config,
            mu,
            sigma,
        )

        density_result = (
            evaluate_density_and_plot(
                model,
                config,
                mu,
                sigma,
            )
        )

        result.update(
            density_result
        )

        result["label"] = (
            f"{config['name']} "
            f"mu={mu:.2f}, "
            f"sigma={sigma:.2f}"
        )

        all_results.append(
            result
        )

        print(
            f"{config['name']} | "
            f"mu={mu:.2f}, "
            f"sigma={sigma:.2f} | "
            f"var={result['variance']:.3e} | "
            f"rel={result['rel_error']:.3e} | "
            f"ESS={result['ess']:.3e} | "
            f"ISE={result['density_ise']:.3e} | "
            f"maxlog="
            f"{result['max_abs_log_ratio_important']:.3e}"
        )

    return all_results


# ============================================================
# Contact Sheet
# ============================================================

def make_contact_sheet_for_model(
    config,
):
    import matplotlib.image as mpimg

    name = config["name"]

    image_paths = []

    for mu, sigma in TEST_CONDITIONS:

        path = os.path.join(
            PLOT_DIR,
            f"{name}_"
            f"mu{mu:.2f}_"
            f"sigma{sigma:.2f}_"
            f"density.png",
        )

        image_paths.append(path)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    axes = axes.flatten()

    for ax, path in zip(
        axes,
        image_paths,
    ):
        img = mpimg.imread(path)

        ax.imshow(img)
        ax.axis("off")

    for ax in axes[
        len(image_paths):
    ]:
        ax.axis("off")

    sigma_min, sigma_max = (
        config["sigma_range"]
    )

    fig.suptitle(
        f"{name}\n"
        f"training sigma range = "
        f"[{sigma_min:.2f}, {sigma_max:.2f}]",
        fontsize=16,
    )

    fig.tight_layout()

    out_path = os.path.join(
        PLOT_DIR,
        f"{name}_contact_sheet.png",
    )

    fig.savefig(
        out_path,
        dpi=160,
    )

    plt.close(fig)

    print(
        f"Saved contact sheet: "
        f"{out_path}"
    )


# ============================================================
# Special edge-condition comparison plot
# ============================================================

def make_edge_comparison_plot(
    trained_models,
):
    """
    Compare the original and extended sigma training ranges
    directly for sigma=0.03 and sigma=0.12.
    """

    edge_conditions = [
        (0.50, 0.03),
        (0.50, 0.12),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.5),
    )

    for ax, (mu, sigma) in zip(
        axes,
        edge_conditions,
    ):

        x_grid = torch.linspace(
            0.0,
            1.0,
            N_GRID,
            device=DEVICE,
        ).reshape(-1, 1)

        f = target_unnormalized(
            x_grid,
            torch.full_like(
                x_grid,
                mu,
            ),
            torch.full_like(
                x_grid,
                sigma,
            ),
        ).reshape(-1)

        integral_ref = exact_integral_1d(
            mu,
            sigma,
        )

        p = (
            f / integral_ref
        ).detach().cpu().numpy()

        x_np = (
            x_grid.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        ax.plot(
            x_np,
            p,
            linewidth=2,
            label="target",
        )

        for config, model in trained_models:

            cond_grid = torch.tensor(
                [[mu, sigma]],
                device=DEVICE,
            ).repeat(
                N_GRID,
                1,
            )

            with torch.no_grad():

                log_q = model.log_prob(
                    x_grid,
                    cond_grid,
                ).reshape(-1)

                q = (
                    torch.exp(log_q)
                    .cpu()
                    .numpy()
                )

            sigma_min, sigma_max = (
                config["sigma_range"]
            )

            label = (
                f"train sigma "
                f"[{sigma_min:.2f},"
                f"{sigma_max:.2f}]"
            )

            ax.plot(
                x_np,
                q,
                label=label,
            )

        ax.set_title(
            f"$\\mu={mu:.2f}$, "
            f"$\\sigma={sigma:.2f}$"
        )

        ax.set_xlabel("$x$")
        ax.set_ylabel("density")

        ax.legend()

    fig.tight_layout()

    out_path = os.path.join(
        PLOT_DIR,
        "edge_condition_comparison.png",
    )

    fig.savefig(
        out_path,
        dpi=180,
    )

    plt.close(fig)

    print(
        f"Saved edge comparison: "
        f"{out_path}"
    )


# ============================================================
# Summary
# ============================================================

def summarize_results(
    all_results,
):
    grouped = defaultdict(list)

    for r in all_results:
        grouped[
            r["config_name"]
        ].append(r)

    summary = []

    for name, rows in grouped.items():

        summary_row = {
            "config_name": name,

            "n_bins":
                rows[0]["n_bins"],

            "sigma_train_min":
                rows[0]["sigma_train_min"],

            "sigma_train_max":
                rows[0]["sigma_train_max"],

            "mean_variance":
                float(
                    np.mean(
                        [
                            r["variance"]
                            for r in rows
                        ]
                    )
                ),

            "mean_rel_error":
                float(
                    np.mean(
                        [
                            r["rel_error"]
                            for r in rows
                        ]
                    )
                ),

            "mean_ess":
                float(
                    np.mean(
                        [
                            r["ess"]
                            for r in rows
                        ]
                    )
                ),

            "mean_density_ise":
                float(
                    np.mean(
                        [
                            r["density_ise"]
                            for r in rows
                        ]
                    )
                ),

            "mean_max_abs_log_ratio_important":
                float(
                    np.mean(
                        [
                            r[
                                "max_abs_log_ratio_important"
                            ]
                            for r in rows
                        ]
                    )
                ),
        }

        # ----------------------------------------------------
        # Separate statistics for the two old boundary cases
        # ----------------------------------------------------

        edge_rows = [
            r for r in rows
            if (
                abs(r["mu"] - 0.50) < 1e-8
                and (
                    abs(r["sigma"] - 0.03) < 1e-8
                    or
                    abs(r["sigma"] - 0.12) < 1e-8
                )
            )
        ]

        summary_row[
            "edge_mean_variance"
        ] = float(
            np.mean(
                [
                    r["variance"]
                    for r in edge_rows
                ]
            )
        )

        summary_row[
            "edge_mean_ise"
        ] = float(
            np.mean(
                [
                    r["density_ise"]
                    for r in edge_rows
                ]
            )
        )

        summary_row[
            "edge_mean_max_log_ratio"
        ] = float(
            np.mean(
                [
                    r[
                        "max_abs_log_ratio_important"
                    ]
                    for r in edge_rows
                ]
            )
        )

        summary.append(
            summary_row
        )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"Using device: {DEVICE}"
    )

    print(
        f"Cubic bins: {N_BINS}"
    )

    print(
        "Testing sigma training domains:"
    )

    for config in TRAIN_CONFIGS:
        print(
            f"  {config['name']} -> "
            f"{config['sigma_range']}"
        )

    all_results = []
    trained_models = []

    for config in TRAIN_CONFIGS:

        model = train_one_model(
            config
        )

        trained_models.append(
            (config, model)
        )

        results = evaluate_model(
            model,
            config,
        )

        all_results.extend(
            results
        )

        make_contact_sheet_for_model(
            config
        )

        result_path = os.path.join(
            SUMMARY_DIR,
            f"{config['name']}_results.json",
        )

        with open(
            result_path,
            "w",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
            )

    # --------------------------------------------------------
    # Direct plot of old boundary cases
    # --------------------------------------------------------

    make_edge_comparison_plot(
        trained_models
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize_results(
        all_results
    )

    all_results_path = os.path.join(
        SUMMARY_DIR,
        "all_sigma_range_results.json",
    )

    with open(
        all_results_path,
        "w",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    summary_path = os.path.join(
        SUMMARY_DIR,
        "sigma_range_summary.json",
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("SIGMA TRAINING RANGE COMPARISON")
    print("=" * 100)

    for row in summary:

        print(
            f"{row['config_name']:34s} | "
            f"sigma=["
            f"{row['sigma_train_min']:.2f},"
            f"{row['sigma_train_max']:.2f}] | "
            f"mean var={row['mean_variance']:.3e} | "
            f"mean ESS={row['mean_ess']:.3e} | "
            f"mean ISE={row['mean_density_ise']:.3e} | "
            f"edge ISE={row['edge_mean_ise']:.3e} | "
            f"edge maxlog="
            f"{row['edge_mean_max_log_ratio']:.3e}"
        )

    print("\nSaved:")
    print(f"  Summary: {summary_path}")
    print(f"  Plots:   {PLOT_DIR}")


if __name__ == "__main__":
    main()