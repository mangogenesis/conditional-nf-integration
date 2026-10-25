import os
import json
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from mc_baselines.uniform_mc import uniform_mc_integrate
from mc_baselines.vegas_integrate import vegas_integrate

from nf.cube_spline_flow import CubeSplineFlow
from nf.train import train_nf, save_nf_checkpoint
from nf.evaluate import nf_integrate


# ======================================================
# Global settings
# ======================================================

DIM = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EVENTS = list(range(10000, 100001, 10000))
VEGAS_NITN = 10

N_BINS = 256
SIGMA1 = 0.015
SIGMA2 = 0.025
FLOOR = 1e-6

CENTERS = [
    (0.50, 0.50),
    (0.20, 0.20),
    (0.80, 0.80),
]

ROOT_DIR = "results/center_sweep_256"
PLOT_DIR = f"{ROOT_DIR}/plots"
CKPT_DIR = f"{ROOT_DIR}/checkpoints"
SUMMARY_DIR = f"{ROOT_DIR}/summaries"

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ======================================================
# Reproducibility
# ======================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================
# Benchmark
# ======================================================

def make_sharp_peak_fn(mu1, mu2, sigma1=SIGMA1, sigma2=SIGMA2, floor=FLOOR):
    """
    Sharp 2D Gaussian peak on [0,1]^2.
    """

    def benchmark_fn(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        z1 = ((x1 - mu1) / sigma1) ** 2
        z2 = ((x2 - mu2) / sigma2) ** 2

        return floor + torch.exp(-0.5 * (z1 + z2))

    return benchmark_fn


def sharp_peak_2d_exact_integral(
    mu1,
    mu2,
    sigma1=SIGMA1,
    sigma2=SIGMA2,
    floor=FLOOR,
):
    """
    Exact integral of floor + anisotropic Gaussian peak over [0,1]^2.
    """

    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def one_dim(mu, sigma):
        a = (0.0 - mu) / sigma
        b = (1.0 - mu) / sigma
        return sigma * math.sqrt(2.0 * math.pi) * (Phi(b) - Phi(a))

    return floor + one_dim(mu1, sigma1) * one_dim(mu2, sigma2)


# ======================================================
# Helpers
# ======================================================

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def center_name(mu1, mu2):
    return f"mu{mu1:.2f}_{mu2:.2f}".replace(".", "p")


def build_model():
    return CubeSplineFlow(
        dim=DIM,
        n_blocks=8,
        hidden=128,
        n_bins=N_BINS,
        perm_sequence="01010101",
    ).to(DEVICE)


def add_accuracy_metrics(res, integral_ref):
    estimate = res.get("estimate", np.nan)
    variance = res.get("variance", np.nan)

    std_error = math.sqrt(variance) if np.isfinite(variance) and variance >= 0 else np.nan
    abs_error = abs(estimate - integral_ref)
    rel_error = abs_error / (abs(integral_ref) + 1e-300)

    if np.isfinite(std_error) and std_error > 0:
        pull = (estimate - integral_ref) / std_error
    else:
        pull = np.nan

    res.update(
        {
            "integral_ref": integral_ref,
            "std_error": std_error,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "pull": pull,
        }
    )

    return res


def make_loss_plots(history, name):
    history = np.asarray(history, dtype=float)
    finite_history = history[np.isfinite(history)]

    if len(finite_history) == 0:
        print(f"Skipping loss plot for {name}: no finite values.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.title(f"Loss: {name}")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_loss.png", dpi=300)
    plt.close()


# ======================================================
# Main diagnostic plots
# ======================================================

def make_learned_q_plot(
    model,
    benchmark_fn,
    name,
    mu1,
    mu2,
    n_grid=300,
):
    """
    2D learned proposal density q_theta(x).
    No background grid.
    """

    model.eval()

    x1 = torch.linspace(0, 1, n_grid, device=DEVICE)
    x2 = torch.linspace(0, 1, n_grid, device=DEVICE)

    try:
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    except TypeError:
        X1, X2 = torch.meshgrid(x1, x2)

    X = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

    with torch.no_grad():
        log_q = model.log_prob(X)
        q_vals = torch.exp(log_q)

    Q = q_vals.reshape(n_grid, n_grid).detach().cpu().numpy()
    X1_np = X1.detach().cpu().numpy()
    X2_np = X2.detach().cpu().numpy()

    plt.figure(figsize=(5.5, 4.8))
    c = plt.contourf(X1_np, X2_np, Q, levels=80)
    plt.colorbar(c, label=r"$q_\theta(x)$")

    plt.scatter([mu1], [mu2], marker="x", s=60, linewidths=2, label="Peak center")

    plt.title(fr"Learned proposal, $\mu=({mu1:.2f},{mu2:.2f})$")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(False)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    path = f"{PLOT_DIR}/{name}_learned_q.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


def make_target_and_q_panel(
    model,
    benchmark_fn,
    name,
    mu1,
    mu2,
    integral_ref,
    n_grid=300,
):
    """
    Three-panel plot:
    target density f/I_ref, learned q, and log-ratio.
    """

    model.eval()

    x1 = torch.linspace(0, 1, n_grid, device=DEVICE)
    x2 = torch.linspace(0, 1, n_grid, device=DEVICE)

    try:
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    except TypeError:
        X1, X2 = torch.meshgrid(x1, x2)

    X = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

    with torch.no_grad():
        f_vals = benchmark_fn(X)
        log_q = model.log_prob(X)
        q_vals = torch.exp(log_q)

        target_density = f_vals / integral_ref
        log_ratio = torch.log(target_density + 1e-12) - log_q

    F = target_density.reshape(n_grid, n_grid).detach().cpu().numpy()
    Q = q_vals.reshape(n_grid, n_grid).detach().cpu().numpy()
    LR = log_ratio.reshape(n_grid, n_grid).detach().cpu().numpy()

    X1_np = X1.detach().cpu().numpy()
    X2_np = X2.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    panels = [
        (F, r"Target density $f/I_{\rm ref}$"),
        (Q, r"Learned proposal $q_\theta$"),
        (np.clip(LR, np.percentile(LR, 1), np.percentile(LR, 99)), r"Log ratio, capped"),
    ]

    for ax, (Z, title) in zip(axes, panels):
        c = ax.contourf(X1_np, X2_np, Z, levels=80)
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

        ax.scatter([mu1], [mu2], marker="x", s=45, linewidths=2)
        ax.set_title(title)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(False)

    fig.suptitle(fr"Center sweep diagnostic, $\mu=({mu1:.2f},{mu2:.2f})$")
    fig.tight_layout()

    path = f"{PLOT_DIR}/{name}_target_q_logratio_panel.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print("Saved:", path)


def make_slice_diagnostic(
    model,
    benchmark_fn,
    name,
    mu1,
    mu2,
    x2_values,
    integral_ref,
    n_grid=2000,
):
    """
    Slice diagnostic for one trained model.
    No background grid.
    Uses vertical line for current peak center mu1.
    """

    model.eval()

    for x2_value in x2_values:
        x1 = torch.linspace(0, 1, n_grid, device=DEVICE)
        x2 = torch.full_like(x1, float(x2_value))
        X = torch.stack([x1, x2], dim=1)

        with torch.no_grad():
            f_vals = benchmark_fn(X)
            log_q = model.log_prob(X)
            q_vals = torch.exp(log_q)

        x1_np = x1.detach().cpu().numpy()
        target_np = (f_vals / integral_ref).detach().cpu().numpy()
        q_np = q_vals.detach().cpu().numpy()

        positive = np.concatenate(
            [
                target_np[np.isfinite(target_np) & (target_np > 0)],
                q_np[np.isfinite(q_np) & (q_np > 0)],
            ]
        )

        if len(positive) > 0:
            ymin = max(np.percentile(positive, 1), 1e-12)
            ymax = np.percentile(positive, 99.9)
        else:
            ymin, ymax = None, None

        plt.figure(figsize=(6.8, 4.0))

        plt.semilogy(x1_np, target_np + 1e-12, linewidth=2.0, label=r"Target $f/I_{\rm ref}$")
        plt.semilogy(x1_np, q_np + 1e-12, linewidth=1.6, label=r"Proposal $q_\theta$")

        plt.axvline(mu1, linestyle="--", linewidth=1.0, label=r"Peak center")

        # Optional: keep bin boundaries, but no background grid.
        edges = np.linspace(0.0, 1.0, N_BINS + 1)
        for edge in edges:
            plt.axvline(edge, color="gray", alpha=0.08, linewidth=0.4)

        if ymin is not None and ymax is not None and ymax > ymin:
            plt.ylim(ymin, ymax)

        plt.xlim(0, 1)
        plt.xlabel(r"$x_1$")
        plt.ylabel("Density value")
        plt.title(fr"Slice at $x_2={x2_value:.3f}$, $\mu=({mu1:.2f},{mu2:.2f})$")
        plt.grid(False)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()

        path = f"{PLOT_DIR}/{name}_slice_x2_{x2_value:.3f}.png"
        plt.savefig(path, dpi=300)
        plt.close()

        print("Saved:", path)


# ======================================================
# Run one center
# ======================================================

def run_one_center(mu1, mu2):
    name = center_name(mu1, mu2)
    benchmark_fn = make_sharp_peak_fn(mu1, mu2)
    integral_ref = sharp_peak_2d_exact_integral(mu1, mu2)

    print("\n" + "=" * 100)
    print(f"Running center: mu=({mu1}, {mu2})")
    print(f"Exact integral reference: {integral_ref:.10e}")
    print("=" * 100)

    all_results = {}

    # --------------------------
    # Baselines
    # --------------------------
    uniform_rows = []
    vegas_rows = []

    for i, n in enumerate(N_EVENTS):
        set_seed(i)

        res = uniform_mc_integrate(
            lambda x: benchmark_fn(x),
            dim=DIM,
            N=n,
        )
        res = add_accuracy_metrics(res, integral_ref)
        res.update(
            {
                "events": n,
                "trial": i,
                "method": "uniform",
                "label": "Uniform MC",
                "center": [mu1, mu2],
                "sigma": [SIGMA1, SIGMA2],
                "n_bins": None,
            }
        )
        uniform_rows.append(res)
        print("uniform", name, n, res)

        set_seed(i)

        neval = n // VEGAS_NITN
        res = vegas_integrate(
            lambda x: benchmark_fn(x),
            dim=DIM,
            nitn=VEGAS_NITN,
            neval=neval,
        )
        res = add_accuracy_metrics(res, integral_ref)
        res.update(
            {
                "events": n,
                "trial": i,
                "method": "vegas",
                "label": "VEGAS",
                "center": [mu1, mu2],
                "sigma": [SIGMA1, SIGMA2],
                "nitn": VEGAS_NITN,
                "neval": neval,
                "n_bins": None,
            }
        )
        vegas_rows.append(res)
        print("vegas", name, n, res)

    all_results["Uniform MC"] = uniform_rows
    all_results["VEGAS"] = vegas_rows

    save_json(uniform_rows, f"{SUMMARY_DIR}/{name}_uniform.json")
    save_json(vegas_rows, f"{SUMMARY_DIR}/{name}_vegas.json")

    # --------------------------
    # Train NF
    # --------------------------
    set_seed(0)

    model = build_model()

    out = train_nf(
        model=model,
        benchmark_fn=benchmark_fn,
        dim=DIM,
        device=DEVICE,
        steps=15000,
        batch_size=300000,
        lr=1e-5,
        loss_name="kl",
        tol=5e-4,
        patience=5,
        log_every=200,
    )

    ckpt_path = f"{CKPT_DIR}/{name}_rq_bins{N_BINS}.pt"
    save_nf_checkpoint(out, ckpt_path)

    make_loss_plots(out["history"], f"{name}_rq_bins{N_BINS}")

    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    nf_rows = []

    for i, n in enumerate(N_EVENTS):
        set_seed(i)

        res = nf_integrate(
            model=model,
            benchmark_fn=benchmark_fn,
            n_events=n,
            device=DEVICE,
        )
        res = add_accuracy_metrics(res, integral_ref)
        res.update(
            {
                "events": n,
                "trial": i,
                "method": f"rq_bins{N_BINS}",
                "label": f"RQ spline bins={N_BINS}",
                "center": [mu1, mu2],
                "sigma": [SIGMA1, SIGMA2],
                "n_bins": N_BINS,
            }
        )
        nf_rows.append(res)
        print("nf", name, n, res)

    all_results[f"RQ spline bins={N_BINS}"] = nf_rows

    save_json(nf_rows, f"{SUMMARY_DIR}/{name}_rq_bins{N_BINS}.json")
    save_json(all_results, f"{SUMMARY_DIR}/{name}_all_results.json")

    # --------------------------
    # Diagnostics
    # --------------------------
    make_learned_q_plot(
        model=model,
        benchmark_fn=benchmark_fn,
        name=f"{name}_rq_bins{N_BINS}",
        mu1=mu1,
        mu2=mu2,
    )

    make_target_and_q_panel(
        model=model,
        benchmark_fn=benchmark_fn,
        name=f"{name}_rq_bins{N_BINS}",
        mu1=mu1,
        mu2=mu2,
        integral_ref=integral_ref,
    )

    x2_values = [
        mu2 - SIGMA2,
        mu2 + SIGMA2,
        mu2 - 3.0 * SIGMA2,
        mu2 + 3.0 * SIGMA2,
    ]


    x2_values = [min(max(v, 0.01), 0.99) for v in x2_values]
    x2_values = sorted(set([round(v, 4) for v in x2_values]))

    make_slice_diagnostic(
        model=model,
        benchmark_fn=benchmark_fn,
        name=f"{name}_rq_bins{N_BINS}",
        mu1=mu1,
        mu2=mu2,
        x2_values=x2_values,
        integral_ref=integral_ref,
    )

    return all_results


# ======================================================
# Summary across centers
# ======================================================

def summarize_center_results(center_results):
    summary_rows = []

    for center_key, all_results in center_results.items():
        for label, rows in all_results.items():
            final = rows[-1]

            summary_rows.append(
                {
                    "center": center_key,
                    "label": label,
                    "estimate": final.get("estimate", np.nan),
                    "integral_ref": final.get("integral_ref", np.nan),
                    "variance": final.get("variance", np.nan),
                    "std_error": final.get("std_error", np.nan),
                    "abs_error": final.get("abs_error", np.nan),
                    "rel_error": final.get("rel_error", np.nan),
                    "pull": final.get("pull", np.nan),
                    "ess": final.get("ess", np.nan),
                }
            )

    summary_rows = sorted(summary_rows, key=lambda r: (r["center"], r["label"]))

    print("\n" + "=" * 120)
    print("CENTER SWEEP FINAL SUMMARY")
    print("=" * 120)

    for r in summary_rows:
        print(
            f"{r['center']:15s} "
            f"{r['label']:25s} "
            f"estimate={r['estimate']:.8e} "
            f"I_ref={r['integral_ref']:.8e} "
            f"variance={r['variance']:.3e} "
            f"rel_err={r['rel_error']:.3e} "
            f"pull={r['pull']:.2f} "
            f"ESS={r['ess']:.3e}"
        )

    save_json(summary_rows, f"{SUMMARY_DIR}/center_sweep_256_summary.json")

    return summary_rows


# ======================================================
# Main
# ======================================================

def main():
    print("DEVICE:", DEVICE)
    print("N_BINS:", N_BINS)
    print("SIGMA:", (SIGMA1, SIGMA2))
    print("CENTERS:", CENTERS)

    center_results = {}

    for mu1, mu2 in CENTERS:
        key = f"({mu1:.2f},{mu2:.2f})"
        center_results[key] = run_one_center(mu1, mu2)

    summarize_center_results(center_results)

    print("\nDone.")
    print("Results:", ROOT_DIR)


if __name__ == "__main__":
    main()