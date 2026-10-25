import os
import json
import math
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from nf.conditional_cubic_2d import (
    ConditionalCubicFlow2D,
)


# ============================================================
# General
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234

OUT_DIR = "results/cubic_2d_single_vs_broad"
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
# Conditional sharp Gaussian
# ============================================================

FLOOR = 1e-6

MU_MIN = 0.20
MU_MAX = 0.80

SIGMA_MIN = 0.03
SIGMA_MAX = 0.08


# ============================================================
# Broad component
# ============================================================

A_BROAD = 0.20

BROAD_MU_X = 0.50
BROAD_MU_Y = 0.50
BROAD_SIGMA = 0.20


# ============================================================
# Targets
# ============================================================

TARGET_CONFIGS = [
    {
        "name": "single",
        "use_broad": False,
    },
    {
        "name": "sharp_plus_broad",
        "use_broad": True,
    },
]


# ============================================================
# Diagnostic conditions
#
# Especially important:
#   corners / displaced peaks
# ============================================================

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
# Condition handling
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
    """
    Map each condition coordinate approximately to [0,1].

    The target always receives RAW physical parameters.
    Only the neural network receives these normalized values.
    """

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
    mu_x,
    mu_y,
    sigma,
    use_broad,
):

    x = xy[..., 0]
    y = xy[..., 1]

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

    result = (
        FLOOR
        + sharp
    )

    if use_broad:

        broad = (
            A_BROAD
            * torch.exp(
                -0.5
                * (
                    (x - BROAD_MU_X)
                    / BROAD_SIGMA
                ) ** 2
                -
                0.5
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
                    z
                    / math.sqrt(2.0)
                )
            )
        )

    lo = (
        -mu
        / sigma
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
    use_broad,
):

    mu_x, mu_y, sigma = condition

    value = (
        FLOOR
        +
        gaussian_integral_2d(
            mu_x,
            mu_y,
            sigma,
        )
    )

    if use_broad:

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
# Model
# ============================================================

def build_model():

    model = ConditionalCubicFlow2D(
        cond_dim=3,
        hidden=HIDDEN,
        n_bins=N_BINS,
        n_layers=N_LAYERS,
    )

    return model.to(
        DEVICE
    )


# ============================================================
# Forward KL
# ============================================================

def forward_kl_loss(
    model,
    use_broad,
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

    f = target_2d(
        xy_flat,
        cond_raw_flat[:, 0],
        cond_raw_flat[:, 1],
        cond_raw_flat[:, 2],
        use_broad,
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

    return (
        loss_per_condition.mean()
    )


# ============================================================
# Train
# ============================================================

def train_model(
    name,
    use_broad,
):

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

        loss = forward_kl_loss(
            model,
            use_broad,
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
                max_norm=10.0,
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
                f"{name:20s} | "
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
# Density-grid diagnostics
# ============================================================

@torch.no_grad()
def evaluate_grid(
    model,
    condition,
    use_broad,
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

    f = target_2d(
        xy,
        cond_raw[:, 0],
        cond_raw[:, 1],
        cond_raw[:, 2],
        use_broad,
    )

    I_ref = exact_integral(
        condition,
        use_broad,
    )

    p = (
        f / I_ref
    )

    # --------------------------------------------------------
    # Peak locations
    # --------------------------------------------------------

    target_peak_index = torch.argmax(
        p
    )

    proposal_peak_index = torch.argmax(
        q
    )

    target_peak_xy = xy[
        target_peak_index
    ]

    proposal_peak_xy = xy[
        proposal_peak_index
    ]

    # --------------------------------------------------------
    # Density means / centers of mass
    #
    # Equal-area grid => weighted discrete sums are enough.
    # --------------------------------------------------------

    target_norm = (
        p.sum()
        + 1e-30
    )

    proposal_norm = (
        q.sum()
        + 1e-30
    )

    target_mean_x = (
        xy[:, 0]
        * p
    ).sum() / target_norm

    target_mean_y = (
        xy[:, 1]
        * p
    ).sum() / target_norm

    proposal_mean_x = (
        xy[:, 0]
        * q
    ).sum() / proposal_norm

    proposal_mean_y = (
        xy[:, 1]
        * q
    ).sum() / proposal_norm

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    peak_error = torch.sqrt(
        torch.sum(
            (
                proposal_peak_xy
                - target_peak_xy
            ) ** 2
        )
    )

    mean_error = torch.sqrt(
        (
            proposal_mean_x
            - target_mean_x
        ) ** 2
        +
        (
            proposal_mean_y
            - target_mean_y
        ) ** 2
    )

    ise = torch.mean(
        (
            p - q
        ) ** 2
    )

    mae = torch.mean(
        torch.abs(
            p - q
        )
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

        "target_peak_x":
            float(
                target_peak_xy[0].cpu()
            ),

        "target_peak_y":
            float(
                target_peak_xy[1].cpu()
            ),

        "proposal_peak_x":
            float(
                proposal_peak_xy[0].cpu()
            ),

        "proposal_peak_y":
            float(
                proposal_peak_xy[1].cpu()
            ),

        "peak_error":
            float(
                peak_error.cpu()
            ),

        "target_mean_x":
            float(
                target_mean_x.cpu()
            ),

        "target_mean_y":
            float(
                target_mean_y.cpu()
            ),

        "proposal_mean_x":
            float(
                proposal_mean_x.cpu()
            ),

        "proposal_mean_y":
            float(
                proposal_mean_y.cpu()
            ),

        "mean_error":
            float(
                mean_error.cpu()
            ),

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
# Importance-sampling evaluation
# ============================================================

@torch.no_grad()
def evaluate_integral(
    model,
    condition,
    use_broad,
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

    f = target_2d(
        samples,
        cond_raw[:, 0],
        cond_raw[:, 1],
        cond_raw[:, 2],
        use_broad,
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
        condition,
        use_broad,
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
            float(
                reference
            ),

        "variance":
            float(
                variance.cpu()
            ),

        "rel_error":
            float(
                rel_error
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


def plot_grid(
    target_name,
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
        "Cubic flow $q_\\theta$",
        "$\\log p - \\log q$",
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

        ax.set_xlabel(
            "$x$"
        )

        ax.set_ylabel(
            "$y$"
        )

        ax.set_title(
            title
        )

    # --------------------------------------------------------
    # Mark target/proposal peak positions
    # --------------------------------------------------------

    axes[0].scatter(
        grid["target_peak_x"],
        grid["target_peak_y"],
        marker="x",
        s=80,
        label="target peak",
    )

    axes[0].legend()

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
        f"{target_name} | "
        f"$\\mu_x={mu_x:.2f}$, "
        f"$\\mu_y={mu_y:.2f}$, "
        f"$\\sigma={sigma:.3f}$ | "
        f"peak error={grid['peak_error']:.3f}"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            PLOT_DIR,
            f"{target_name}_"
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
        f"bins={N_BINS}"
    )

    print(
        f"layers={N_LAYERS}"
    )

    print(
        f"sigma range="
        f"({SIGMA_MIN}, {SIGMA_MAX})"
    )

    models = {}
    all_results = {}
    summaries = {}

    # ========================================================
    # Train same architecture on both targets
    # ========================================================

    for config in TARGET_CONFIGS:

        name = config["name"]

        model = train_model(
            name=name,
            use_broad=config["use_broad"],
        )

        models[name] = model

    # ========================================================
    # Evaluate
    # ========================================================

    for config in TARGET_CONFIGS:

        name = config["name"]
        use_broad = config["use_broad"]

        model = models[name]

        print()
        print("=" * 120)
        print(
            f"EVALUATING: {name}"
        )
        print("=" * 120)

        results = []

        for condition in TEST_CONDITIONS:

            grid = evaluate_grid(
                model,
                condition,
                use_broad,
            )

            integral = evaluate_integral(
                model,
                condition,
                use_broad,
            )

            plot_grid(
                name,
                condition,
                grid,
            )

            result = {
                "mu_x":
                    condition[0],

                "mu_y":
                    condition[1],

                "sigma":
                    condition[2],

                **integral,

                "target_peak_x":
                    grid["target_peak_x"],

                "target_peak_y":
                    grid["target_peak_y"],

                "proposal_peak_x":
                    grid["proposal_peak_x"],

                "proposal_peak_y":
                    grid["proposal_peak_y"],

                "peak_error":
                    grid["peak_error"],

                "target_mean_x":
                    grid["target_mean_x"],

                "target_mean_y":
                    grid["target_mean_y"],

                "proposal_mean_x":
                    grid["proposal_mean_x"],

                "proposal_mean_y":
                    grid["proposal_mean_y"],

                "mean_error":
                    grid["mean_error"],

                "ise":
                    grid["ise"],

                "mae":
                    grid["mae"],

                "maxlog":
                    grid["maxlog"],

                "pmax":
                    grid["pmax"],

                "qmax":
                    grid["qmax"],
            }

            results.append(
                result
            )

            print(
                f"c={condition} | "
                f"target peak="
                f"({result['target_peak_x']:.3f}, "
                f"{result['target_peak_y']:.3f}) | "
                f"q peak="
                f"({result['proposal_peak_x']:.3f}, "
                f"{result['proposal_peak_y']:.3f}) | "
                f"peak err={result['peak_error']:.3f} | "
                f"mean err={result['mean_error']:.3f} | "
                f"ESS={result['ess_fraction']:.3f} | "
                f"ISE={result['ise']:.3e}"
            )

        all_results[name] = results

        summaries[name] = {
            "mean_peak_error":
                float(
                    np.mean(
                        [
                            r["peak_error"]
                            for r in results
                        ]
                    )
                ),

            "mean_center_error":
                float(
                    np.mean(
                        [
                            r["mean_error"]
                            for r in results
                        ]
                    )
                ),

            "mean_variance":
                float(
                    np.mean(
                        [
                            r["variance"]
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
        }

    # ========================================================
    # Save
    # ========================================================

    with open(
        os.path.join(
            OUT_DIR,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            {
                "settings": {
                    "bins":
                        N_BINS,

                    "layers":
                        N_LAYERS,

                    "hidden":
                        HIDDEN,

                    "steps":
                        STEPS,

                    "sigma_range":
                        [
                            SIGMA_MIN,
                            SIGMA_MAX,
                        ],

                    "condition_normalization":
                        True,
                },

                "summaries":
                    summaries,

                "results":
                    all_results,
            },
            f,
            indent=2,
        )

    # ========================================================
    # Final comparison
    # ========================================================

    print()
    print("=" * 100)
    print(
        "CENTER-LOCALIZATION COMPARISON"
    )
    print("=" * 100)

    print(
        f"{'target':<24}"
        f"{'peak err':>14}"
        f"{'mean err':>14}"
        f"{'ESS frac':>14}"
        f"{'variance':>16}"
        f"{'ISE':>14}"
    )

    print("-" * 100)

    for config in TARGET_CONFIGS:

        name = config["name"]

        s = summaries[name]

        print(
            f"{name:<24}"
            f"{s['mean_peak_error']:>14.4f}"
            f"{s['mean_center_error']:>14.4f}"
            f"{s['mean_ess_fraction']:>14.3f}"
            f"{s['mean_variance']:>16.3e}"
            f"{s['mean_ise']:>14.3e}"
        )


if __name__ == "__main__":
    main()