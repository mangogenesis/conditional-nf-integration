import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from benchmarks.registry import get_benchmark
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

ROOT_DIR = "results/sharpness_sweep"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ======================================================
# Sharpness sweep benchmarks
# ======================================================

BENCHMARKS_TO_RUN = [
    "sharp_peak_008_010",
    "sharp_peak_005_007",
    "sharp_peak_003_005",
    "sharp_peak_002_003",
    "sharp_peak_0015_0025",
]

SIGMA_INFO = {
    "sharp_peak_008_010": (0.08, 0.10),
    "sharp_peak_005_007": (0.05, 0.07),
    "sharp_peak_003_005": (0.03, 0.05),
    "sharp_peak_002_003": (0.02, 0.03),
    "sharp_peak_0015_0025": (0.015, 0.025),
}


# ======================================================
# Fixed RQ spline config for sharpness sweep
# ======================================================

RQ_CONFIG = {
    "name": "rq_spline_alt01",
    "label": "RQ spline (perm=01010101)",
    "model_type": "rq_spline",
    "seed": 0,
    "model_kwargs": dict(
        dim=DIM,
        n_blocks=8,
        hidden=128,
        n_bins=64,
        perm_sequence="01010101",
    ),
    "train_kwargs": dict(
        steps=15000,
        batch_size=300000,
        lr=1e-5,
        loss_name="kl",
        tol=5e-4,
        patience=5,
        log_every=200,
    ),
}


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
# Helpers
# ======================================================

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_model(cfg):
    if cfg["model_type"] == "rq_spline":
        return CubeSplineFlow(**cfg["model_kwargs"]).to(DEVICE)

    raise ValueError(f"Unknown model_type: {cfg['model_type']}")


def get_permutation_order(model):
    perms = []

    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transforms"):
        layers = model.transforms
    else:
        return perms

    for i, layer in enumerate(layers):
        if hasattr(layer, "perm"):
            perms.append(
                {
                    "layer_index": i,
                    "perm": layer.perm.detach().cpu().tolist(),
                }
            )

    return perms


def make_loss_plots(history, name, label, plot_dir):
    history = np.asarray(history, dtype=float)
    finite_history = history[np.isfinite(history)]

    if len(finite_history) == 0:
        print(f"Skipping loss plots for {name}: no finite values.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.title(f"Loss: {label}")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_loss_raw.png", dpi=300)
    plt.close()

    if len(finite_history) > 2:
        lo = np.percentile(finite_history, 1)
        hi = np.percentile(finite_history, 99)

        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            plt.figure(figsize=(7, 4))
            plt.plot(history)
            plt.ylim(lo, hi)
            plt.title(f"Loss: {label} capped 1--99%")
            plt.xlabel("Training step")
            plt.ylabel("Loss, capped")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/{name}_loss_capped_1_99.png", dpi=300)
            plt.close()


# ======================================================
# Baselines
# ======================================================

def run_uniform(benchmark_fn, benchmark_name, out_dir):
    rows = []

    for i, n in enumerate(N_EVENTS):
        set_seed(i)

        res = uniform_mc_integrate(
            lambda x: benchmark_fn(x),
            dim=DIM,
            N=n,
        )

        res.update(
            {
                "events": n,
                "trial": i,
                "method": "uniform",
                "label": "Uniform MC",
                "benchmark": benchmark_name,
                "dim": DIM,
            }
        )

        rows.append(res)
        print("uniform", benchmark_name, n, res)

    save_json(rows, f"{out_dir}/{benchmark_name}_d{DIM}_uniform.json")
    return rows


def run_vegas(benchmark_fn, benchmark_name, out_dir):
    rows = []

    for i, budget in enumerate(N_EVENTS):
        set_seed(i)

        neval = budget // VEGAS_NITN

        res = vegas_integrate(
            lambda x: benchmark_fn(x),
            dim=DIM,
            nitn=VEGAS_NITN,
            neval=neval,
        )

        res.update(
            {
                "events": budget,
                "trial": i,
                "method": "vegas",
                "label": "VEGAS",
                "benchmark": benchmark_name,
                "dim": DIM,
                "nitn": VEGAS_NITN,
                "neval": neval,
            }
        )

        rows.append(res)
        print("vegas", benchmark_name, budget, res)

    save_json(rows, f"{out_dir}/{benchmark_name}_d{DIM}_vegas.json")
    return rows


# ======================================================
# Diagnostic plots
# ======================================================

def make_diagnostic_plots(model, benchmark_fn, benchmark_name, name, label, plot_dir):
    n_grid = 300

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
        log_w = torch.log(f_vals + 1e-12) - log_q
        w_vals = torch.exp(log_w)

    F = f_vals.reshape(n_grid, n_grid).detach().cpu()
    Q = q_vals.reshape(n_grid, n_grid).detach().cpu()
    LOGW = log_w.reshape(n_grid, n_grid).detach().cpu()
    W = w_vals.reshape(n_grid, n_grid).detach().cpu()

    X1c = X1.detach().cpu()
    X2c = X2.detach().cpu()

    def safe_quantile(Z, q):
        z = Z.flatten()
        z = z[torch.isfinite(z)]
        if z.numel() == 0:
            return None
        return torch.quantile(z, q).item()

    def capped(Z, q_low=None, q_high=None):
        Z_plot = Z.clone()

        if q_low is not None:
            lo = safe_quantile(Z_plot, q_low)
            if lo is not None:
                Z_plot = torch.clamp(Z_plot, min=lo)

        if q_high is not None:
            hi = safe_quantile(Z_plot, q_high)
            if hi is not None:
                Z_plot = torch.clamp(Z_plot, max=hi)

        return Z_plot

    def contour(Z, title, cbar_label, filename, q_low=None, q_high=None, overlay_F=None):
        plt.figure(figsize=(7, 6))

        Z_plot = capped(Z, q_low=q_low, q_high=q_high)

        c = plt.contourf(X1c, X2c, Z_plot, levels=80)
        plt.colorbar(c, label=cbar_label)

        if overlay_F is not None:
            F_overlay = capped(overlay_F, q_high=0.99)
            plt.contour(X1c, X2c, F_overlay, levels=6, linewidths=0.8)

        plt.title(title)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{filename}", dpi=300)
        plt.close()

    contour(
        F,
        f"True benchmark: {benchmark_name}",
        r"$f(x)$",
        f"{benchmark_name}_true_f.png",
        q_high=0.99,
    )

    contour(
        Q,
        f"Learned q(x) with true contours: {label}",
        r"$q_\theta(x)$, capped",
        f"{name}_learned_q_with_true_contours.png",
        q_high=0.99,
        overlay_F=F,
    )

    contour(
        LOGW,
        f"Log weight capped 1--99%: {label}",
        r"$\log f - \log q_\theta$, capped",
        f"{name}_log_weight_capped_1_99.png",
        q_low=0.01,
        q_high=0.99,
    )

    contour(
        W,
        f"Weight f/q capped 99%: {label}",
        r"$f/q_\theta$, capped",
        f"{name}_weight_capped99.png",
        q_high=0.99,
    )


# ======================================================
# Train/evaluate RQ
# ======================================================

def train_and_eval_rq(cfg, benchmark_fn, benchmark_name, out_dir, plot_dir, ckpt_dir, perm_dir):
    name = cfg["name"]
    seed = cfg.get("seed", 0)

    print("\n" + "=" * 80)
    print("Running:", benchmark_name, name)
    print("Seed:", seed)
    print("=" * 80)

    set_seed(seed)

    model = build_model(cfg)
    perm_info = get_permutation_order(model)

    save_json(
        {
            "name": name,
            "label": cfg["label"],
            "seed": seed,
            "model_type": cfg["model_type"],
            "model_kwargs": cfg["model_kwargs"],
            "permutations": perm_info,
        },
        f"{perm_dir}/{name}_permutations.json",
    )

    print("Permutation order:")
    for p in perm_info:
        print(f"  layer {p['layer_index']}: {p['perm']}")

    out = train_nf(
        model=model,
        benchmark_fn=benchmark_fn,
        dim=DIM,
        device=DEVICE,
        **cfg["train_kwargs"],
    )

    ckpt_path = f"{ckpt_dir}/{name}.pt"
    save_nf_checkpoint(out, ckpt_path)

    make_loss_plots(out["history"], name, cfg["label"], plot_dir)

    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []

    for i, n in enumerate(N_EVENTS):
        set_seed(i)

        res = nf_integrate(
            model=model,
            benchmark_fn=benchmark_fn,
            n_events=n,
            device=DEVICE,
        )

        res.update(
            {
                "events": n,
                "trial": i,
                "method": name,
                "label": cfg["label"],
                "benchmark": benchmark_name,
                "dim": DIM,
                "model_type": cfg["model_type"],
                "seed": seed,
                "config": cfg,
                "permutations": perm_info,
            }
        )

        rows.append(res)
        print(name, benchmark_name, n, res)

    save_json(rows, f"{out_dir}/{benchmark_name}_d{DIM}_{name}.json")

    make_diagnostic_plots(model, benchmark_fn, benchmark_name, name, cfg["label"], plot_dir)

    return rows


# ======================================================
# Comparison plot
# ======================================================

def plot_metric(all_results, benchmark_name, metric, ylabel, filename, plot_dir, logy=False):
    plt.figure(figsize=(8, 6))

    for label, rows in all_results.items():
        xs = [r["events"] for r in rows]
        ys = [r[metric] for r in rows]
        plt.plot(xs, ys, marker="o", label=label)

    if logy:
        plt.yscale("log")

    plt.xlabel("Number of events")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}: {benchmark_name}, d={DIM}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{filename}", dpi=300)
    plt.close()


# ======================================================
# Summary
# ======================================================

def summarize_one_benchmark(all_results, benchmark_name, sigma1, sigma2):
    rows = []

    for label, results in all_results.items():
        final = results[-1]
        rows.append(
            {
                "benchmark": benchmark_name,
                "sigma1": sigma1,
                "sigma2": sigma2,
                "label": label,
                "estimate": final.get("estimate", np.nan),
                "variance": final.get("variance", np.nan),
                "error": final.get("error", np.nan),
                "ess": final.get("ess", np.nan),
            }
        )

    return rows


# ======================================================
# Main
# ======================================================

def main():
    print("DEVICE:", DEVICE)

    os.makedirs(ROOT_DIR, exist_ok=True)

    global_summary = []

    for benchmark_name in BENCHMARKS_TO_RUN:
        print("\n" + "#" * 100)
        print("BENCHMARK:", benchmark_name)
        print("#" * 100)

        benchmark_fn = get_benchmark(benchmark_name)
        sigma1, sigma2 = SIGMA_INFO[benchmark_name]

        out_dir = f"{ROOT_DIR}/{benchmark_name}"
        plot_dir = f"{out_dir}/plots"
        ckpt_dir = f"{out_dir}/checkpoints"
        perm_dir = f"{out_dir}/permutations"

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(perm_dir, exist_ok=True)

        all_results = {}

        uniform_rows = run_uniform(benchmark_fn, benchmark_name, out_dir)
        vegas_rows = run_vegas(benchmark_fn, benchmark_name, out_dir)
        rq_rows = train_and_eval_rq(
            RQ_CONFIG,
            benchmark_fn,
            benchmark_name,
            out_dir,
            plot_dir,
            ckpt_dir,
            perm_dir,
        )

        all_results["Uniform MC"] = uniform_rows
        all_results["VEGAS"] = vegas_rows
        all_results[RQ_CONFIG["label"]] = rq_rows

        save_json(all_results, f"{out_dir}/{benchmark_name}_d{DIM}_all_results.json")

        plot_metric(
            all_results,
            benchmark_name,
            metric="estimate",
            ylabel="Integral estimate",
            filename="estimate_comparison.png",
            plot_dir=plot_dir,
            logy=False,
        )

        plot_metric(
            all_results,
            benchmark_name,
            metric="variance",
            ylabel="Estimator variance",
            filename="variance_comparison.png",
            plot_dir=plot_dir,
            logy=True,
        )

        global_summary.extend(
            summarize_one_benchmark(all_results, benchmark_name, sigma1, sigma2)
        )

    save_json(global_summary, f"{ROOT_DIR}/sharpness_sweep_summary.json")

    print("\nDone.")
    print("Results:", ROOT_DIR)


if __name__ == "__main__":
    main()