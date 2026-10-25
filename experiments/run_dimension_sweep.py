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

BENCHMARK_BASE = "sharp_peak_nd_008"

DIMS = [2, 4, 6, 8]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EVENTS = list(range(10000, 100001, 10000))
VEGAS_NITN = 10

N_BINS = 32
N_BLOCKS = 8
HIDDEN = 128

ROOT_DIR = "results/dimension_sweep"
OUT_DIR = f"{ROOT_DIR}/{BENCHMARK_BASE}_bins{N_BINS}"
PLOT_DIR = f"{OUT_DIR}/plots"
CKPT_DIR = f"{OUT_DIR}/checkpoints"
PERM_DIR = f"{OUT_DIR}/permutations"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PERM_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ======================================================
# Benchmark builder
# ======================================================

def build_benchmark(dim):
    fn = get_benchmark(BENCHMARK_BASE)

    x_test = torch.rand(10, dim, device=DEVICE)

    with torch.no_grad():
        y_test = fn(x_test)

    if y_test.shape != (10,):
        raise ValueError(
            f"Benchmark {BENCHMARK_BASE} returned shape {y_test.shape}, expected (10,)."
        )

    print(f"Using benchmark {BENCHMARK_BASE} with dim={dim}")

    return fn, f"{BENCHMARK_BASE}_d{dim}"


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


def make_loss_plots(history, name, label, dim):
    history = np.asarray(history, dtype=float)
    finite_history = history[np.isfinite(history)]

    if len(finite_history) == 0:
        print(f"Skipping loss plots for {name}: no finite values.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.title(f"Loss: {label}, d={dim}")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_d{dim}_loss_raw.png", dpi=300)
    plt.close()

    if len(finite_history) > 2:
        lo = np.percentile(finite_history, 1)
        hi = np.percentile(finite_history, 99)

        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            plt.figure(figsize=(7, 4))
            plt.plot(history)
            plt.ylim(lo, hi)
            plt.title(f"Loss: {label}, d={dim}, capped 1--99%")
            plt.xlabel("Training step")
            plt.ylabel("Loss, capped")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/{name}_d{dim}_loss_capped_1_99.png", dpi=300)
            plt.close()


def make_2d_diagnostic_plots(model, benchmark_fn, benchmark_name, name, label, dim):
    """
    Only for dim=2. Makes learned q, true f, and log-weight plots.
    """

    if dim != 2:
        return

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

    def contour(Z, title, cbar_label, filename, q_low=None, q_high=None):
        plt.figure(figsize=(7, 6))

        Z_plot = capped(Z, q_low=q_low, q_high=q_high)

        c = plt.contourf(X1c, X2c, Z_plot, levels=80)
        plt.colorbar(c, label=cbar_label)

        plt.title(title)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{filename}", dpi=300)
        plt.close()

    contour(
        F,
        f"True benchmark: {benchmark_name}",
        r"$f(x)$",
        f"{name}_d{dim}_true_f.png",
        q_high=0.99,
    )

    contour(
        Q,
        f"Learned q(x): {label}, d={dim}",
        r"$q_\theta(x)$",
        f"{name}_d{dim}_learned_q.png",
        q_high=0.99,
    )

    contour(
        LOGW,
        f"Log weight capped 1--99%: {label}, d={dim}",
        r"$\log f - \log q_\theta$, capped",
        f"{name}_d{dim}_log_weight_capped_1_99.png",
        q_low=0.01,
        q_high=0.99,
    )

    contour(
        W,
        f"Weight f/q capped 99%: {label}, d={dim}",
        r"$f/q_\theta$, capped",
        f"{name}_d{dim}_weight_capped99.png",
        q_high=0.99,
    )


# ======================================================
# Config
# ======================================================

def make_nf_config(dim):
    return {
        "name": f"rq_spline_d{dim}_bins{N_BINS}",
        "label": f"RQ spline d={dim}, bins={N_BINS}",
        "model_type": "rq_spline",
        "seed": 0,
        "model_kwargs": dict(
            dim=dim,
            n_blocks=N_BLOCKS,
            hidden=HIDDEN,
            n_bins=N_BINS,
            permute="reverse",
            use_rotation=False,
            rotation_angle=0.0,
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
# Baselines
# ======================================================

def run_uniform(dim, benchmark_fn, benchmark_name):
    rows = []

    for i, n in enumerate(N_EVENTS):
        set_seed(i)

        res = uniform_mc_integrate(
            lambda x: benchmark_fn(x),
            dim=dim,
            N=n,
        )

        res.update(
            {
                "events": n,
                "trial": i,
                "method": "uniform",
                "label": f"Uniform MC d={dim}",
                "benchmark": benchmark_name,
                "dim": dim,
            }
        )

        rows.append(res)
        print("uniform", "d=", dim, "N=", n, res)

    save_json(rows, f"{OUT_DIR}/{benchmark_name}_uniform.json")
    return rows


def run_vegas(dim, benchmark_fn, benchmark_name):
    rows = []

    for i, budget in enumerate(N_EVENTS):
        set_seed(i)

        neval = budget // VEGAS_NITN

        res = vegas_integrate(
            lambda x: benchmark_fn(x),
            dim=dim,
            nitn=VEGAS_NITN,
            neval=neval,
        )

        res.update(
            {
                "events": budget,
                "trial": i,
                "method": "vegas",
                "label": f"VEGAS d={dim}",
                "benchmark": benchmark_name,
                "dim": dim,
                "nitn": VEGAS_NITN,
                "neval": neval,
            }
        )

        rows.append(res)
        print("vegas", "d=", dim, "budget=", budget, res)

    save_json(rows, f"{OUT_DIR}/{benchmark_name}_vegas.json")
    return rows


# ======================================================
# Train/evaluate NF
# ======================================================

def train_and_eval_nf(dim, benchmark_fn, benchmark_name):
    cfg = make_nf_config(dim)
    name = cfg["name"]
    seed = cfg.get("seed", 0)

    print("\n" + "=" * 100)
    print("Running NF:", name)
    print("Dimension:", dim)
    print("Seed:", seed)
    print("=" * 100)

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
        f"{PERM_DIR}/{name}_permutations.json",
    )

    print("Permutation order:")
    for p in perm_info:
        print(f"  layer {p['layer_index']}: {p['perm']}")

    out = train_nf(
        model=model,
        benchmark_fn=benchmark_fn,
        dim=dim,
        device=DEVICE,
        **cfg["train_kwargs"],
    )

    ckpt_path = f"{CKPT_DIR}/{name}.pt"
    save_nf_checkpoint(out, ckpt_path)

    make_loss_plots(out["history"], name, cfg["label"], dim)

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
                "dim": dim,
                "model_type": cfg["model_type"],
                "seed": seed,
                "config": cfg,
                "permutations": perm_info,
            }
        )

        rows.append(res)
        print(name, "N=", n, res)

    save_json(rows, f"{OUT_DIR}/{benchmark_name}_{name}.json")

    make_2d_diagnostic_plots(
        model=model,
        benchmark_fn=benchmark_fn,
        benchmark_name=benchmark_name,
        name=name,
        label=cfg["label"],
        dim=dim,
    )

    return rows


# ======================================================
# Comparison plots
# ======================================================

def plot_metric_by_events(all_results, metric, ylabel, filename, logy=False):
    plt.figure(figsize=(9, 6))

    for label, rows in all_results.items():
        xs = [r["events"] for r in rows]
        ys = [r.get(metric, np.nan) for r in rows]
        plt.plot(xs, ys, marker="o", label=label)

    if logy:
        plt.yscale("log")

    plt.xlabel("Number of events")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}: {BENCHMARK_BASE}, bins={N_BINS}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{filename}", dpi=300)
    plt.close()


def plot_final_metric_vs_dim(summary_rows, metric, ylabel, filename, logy=False):
    plt.figure(figsize=(7, 5))

    methods = sorted(set(r["method_group"] for r in summary_rows))

    for method in methods:
        rows = [r for r in summary_rows if r["method_group"] == method]
        rows = sorted(rows, key=lambda r: r["dim"])

        xs = [r["dim"] for r in rows]
        ys = [r.get(metric, np.nan) for r in rows]

        plt.plot(xs, ys, marker="o", label=method)

    if logy:
        plt.yscale("log")

    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs dimension: {BENCHMARK_BASE}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{filename}", dpi=300)
    plt.close()


# ======================================================
# Summary
# ======================================================

def summarize_final_results(all_results):
    summary_rows = []

    print("\n" + "=" * 110)
    print("FINAL RESULT SUMMARY")
    print("=" * 110)

    for label, rows in all_results.items():
        final = rows[-1]

        method = final["method"]
        dim = final["dim"]

        if method == "uniform":
            method_group = "Uniform MC"
        elif method == "vegas":
            method_group = "VEGAS"
        else:
            method_group = "RQ spline NF"

        row = {
            "label": label,
            "method": method,
            "method_group": method_group,
            "dim": dim,
            "events": final.get("events", np.nan),
            "estimate": final.get("estimate", np.nan),
            "variance": final.get("variance", np.nan),
            "error": final.get("error", np.nan),
            "ess": final.get("ess", np.nan),
        }

        summary_rows.append(row)

    summary_rows = sorted(summary_rows, key=lambda r: (r["dim"], r["variance"]))

    for r in summary_rows:
        print(
            f"d={r['dim']:<2d}  "
            f"{r['method_group']:12s}  "
            f"estimate = {r['estimate']:.8e}  "
            f"variance = {r['variance']:.3e}  "
            f"error = {r['error']:.3e}  "
            f"ESS = {r['ess']:.3e}"
        )

    return summary_rows


# ======================================================
# Main
# ======================================================

def main():
    print("DEVICE:", DEVICE)
    print("BENCHMARK_BASE:", BENCHMARK_BASE)
    print("DIMS:", DIMS)
    print("N_BINS:", N_BINS)
    print("N_BLOCKS:", N_BLOCKS)
    print("HIDDEN:", HIDDEN)

    all_results = {}

    for dim in DIMS:
        print("\n" + "#" * 120)
        print(f"STARTING DIMENSION d={dim}")
        print("#" * 120)

        benchmark_fn, benchmark_name = build_benchmark(dim)

        uniform_rows = run_uniform(dim, benchmark_fn, benchmark_name)
        vegas_rows = run_vegas(dim, benchmark_fn, benchmark_name)
        nf_rows = train_and_eval_nf(dim, benchmark_fn, benchmark_name)

        all_results[f"Uniform MC d={dim}"] = uniform_rows
        all_results[f"VEGAS d={dim}"] = vegas_rows
        all_results[f"RQ spline NF d={dim}"] = nf_rows

    save_json(all_results, f"{OUT_DIR}/{BENCHMARK_BASE}_dimension_sweep_all_results.json")

    plot_metric_by_events(
        all_results,
        metric="estimate",
        ylabel="Integral estimate",
        filename="estimate_comparison_all_dims.png",
        logy=False,
    )

    plot_metric_by_events(
        all_results,
        metric="variance",
        ylabel="Estimator variance",
        filename="variance_comparison_all_dims.png",
        logy=True,
    )

    summary_rows = summarize_final_results(all_results)

    save_json(summary_rows, f"{OUT_DIR}/dimension_sweep_summary.json")

    plot_final_metric_vs_dim(
        summary_rows,
        metric="variance",
        ylabel="Final estimator variance",
        filename="final_variance_vs_dimension.png",
        logy=True,
    )

    plot_final_metric_vs_dim(
        summary_rows,
        metric="estimate",
        ylabel="Final integral estimate",
        filename="final_estimate_vs_dimension.png",
        logy=False,
    )

    print("\nDone.")
    print("Results:", OUT_DIR)


if __name__ == "__main__":
    main()