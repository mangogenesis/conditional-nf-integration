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
BENCHMARK = "easy_four_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EVENTS = list(range(10000, 100001, 10000))
VEGAS_NITN = 10

ROOT_DIR = "results/depth_sweep"
OUT_DIR = f"{ROOT_DIR}/{BENCHMARK}"
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
# Depth configs
# ======================================================

DEPTHS = [4, 8, 16, 24]

# Run each depth multiple times to test stability.
TRAIN_SEEDS = [0, 1, 2, 3]


def alternating_pattern(n_blocks, start="0"):
    if start == "0":
        return "".join("0" if i % 2 == 0 else "1" for i in range(n_blocks))
    elif start == "1":
        return "".join("1" if i % 2 == 0 else "0" for i in range(n_blocks))
    else:
        raise ValueError("start must be '0' or '1'")


def make_depth_configs():
    cfgs = []

    for n_blocks in DEPTHS:
        pattern = alternating_pattern(n_blocks, start="0")

        for seed in TRAIN_SEEDS:
            cfgs.append(
                {
                    "name": f"rq_spline_depth{n_blocks}_seed{seed}_alt01",
                    "label": f"RQ spline depth={n_blocks}, seed={seed}",
                    "model_type": "rq_spline",
                    "seed": seed,
                    "depth": n_blocks,
                    "model_kwargs": dict(
                        dim=DIM,
                        n_blocks=n_blocks,
                        hidden=128,
                        n_bins=64,
                        perm_sequence=pattern,
                    ),
                    "train_kwargs": dict(
                        steps=30000,
                        batch_size=300000,
                        lr=1e-5,
                        loss_name="kl",
                        tol=5e-4,
                        patience=8,
                        log_every=200,
                    ),
                }
            )

    return cfgs


CONFIGS = make_depth_configs()


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


def make_loss_plots(history, name, label):
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
    plt.savefig(f"{PLOT_DIR}/{name}_loss_raw.png", dpi=300)
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
            plt.savefig(f"{PLOT_DIR}/{name}_loss_capped_1_99.png", dpi=300)
            plt.close()


# ======================================================
# Baselines
# ======================================================

def run_uniform():
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
                "benchmark": BENCHMARK,
                "dim": DIM,
            }
        )

        rows.append(res)
        print("uniform", n, res)

    save_json(rows, f"{OUT_DIR}/{BENCHMARK}_d{DIM}_uniform.json")
    return rows


def run_vegas():
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
                "benchmark": BENCHMARK,
                "dim": DIM,
                "nitn": VEGAS_NITN,
                "neval": neval,
            }
        )

        rows.append(res)
        print("vegas", budget, res)

    save_json(rows, f"{OUT_DIR}/{BENCHMARK}_d{DIM}_vegas.json")
    return rows


# ======================================================
# Diagnostic plots
# ======================================================

def make_diagnostic_plots(model, name, label):
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
        plt.savefig(f"{PLOT_DIR}/{filename}", dpi=300)
        plt.close()

    contour(
        F,
        f"True benchmark: {BENCHMARK}",
        r"$f(x)$",
        f"{BENCHMARK}_true_f.png",
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

def train_and_eval(cfg):
    name = cfg["name"]
    seed = cfg.get("seed", 0)

    print("\n" + "=" * 80)
    print("Running:", name)
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
        f"{PERM_DIR}/{name}_permutations.json",
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

    ckpt_path = f"{CKPT_DIR}/{name}.pt"
    save_nf_checkpoint(out, ckpt_path)

    make_loss_plots(out["history"], name, cfg["label"])

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
                "benchmark": BENCHMARK,
                "dim": DIM,
                "model_type": cfg["model_type"],
                "seed": seed,
                "config": cfg,
                "permutations": perm_info,
            }
        )

        rows.append(res)
        print(name, n, res)

    save_json(rows, f"{OUT_DIR}/{BENCHMARK}_d{DIM}_{name}.json")

    make_diagnostic_plots(model, name, cfg["label"])

    return rows


# ======================================================
# Comparison plot
# ======================================================

def plot_metric(all_results, metric, ylabel, filename, logy=False):
    plt.figure(figsize=(8, 6))

    for label, rows in all_results.items():
        xs = [r["events"] for r in rows]
        ys = [r[metric] for r in rows]
        plt.plot(xs, ys, marker="o", label=label)

    if logy:
        plt.yscale("log")

    plt.xlabel("Number of events")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}: {BENCHMARK}, d={DIM}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{filename}", dpi=300)
    plt.close()


# ======================================================
# Summary
# ======================================================

def summarize_final_results(all_results):
    rows = []

    for label, results in all_results.items():
        final = results[-1]

        rows.append(
            {
                "label": label,
                "estimate": final.get("estimate", np.nan),
                "variance": final.get("variance", np.nan),
                "error": final.get("error", np.nan),
                "ess": final.get("ess", np.nan),
            }
        )

    rows = sorted(rows, key=lambda r: r["variance"])

    print("\n" + "=" * 100)
    print("FINAL RESULT SUMMARY")
    print("=" * 100)

    for r in rows:
        print(
            f"{r['label']:45s} "
            f"estimate = {r['estimate']:.8e}  "
            f"variance = {r['variance']:.3e}  "
            f"error = {r['error']:.3e}  "
            f"ESS = {r['ess']:.3e}"
        )

    return rows


def summarize_by_depth(summary_rows):
    grouped = {}

    for r in summary_rows:
        label = r["label"]

        # Only summarize RQ spline runs.
        if "depth=" not in label:
            continue

        # label format: "RQ spline depth=8, seed=0"
        depth_part = label.split("depth=")[1].split(",")[0]
        depth = int(depth_part)

        if depth not in grouped:
            grouped[depth] = []

        grouped[depth].append(r)

    depth_summary = []

    print("\n" + "=" * 100)
    print("DEPTH SUMMARY ACROSS SEEDS")
    print("=" * 100)

    for depth in sorted(grouped.keys()):
        rows = grouped[depth]

        estimates = np.array([r["estimate"] for r in rows], dtype=float)
        variances = np.array([r["variance"] for r in rows], dtype=float)
        errors = np.array([r["error"] for r in rows], dtype=float)

        out = {
            "depth": depth,
            "n_runs": len(rows),
            "estimate_mean": float(np.mean(estimates)),
            "estimate_std": float(np.std(estimates)),
            "variance_mean": float(np.mean(variances)),
            "variance_std": float(np.std(variances)),
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
        }

        depth_summary.append(out)

        print(
            f"depth={depth:2d}  "
            f"runs={out['n_runs']}  "
            f"estimate = {out['estimate_mean']:.8e} ± {out['estimate_std']:.2e}  "
            f"variance = {out['variance_mean']:.3e} ± {out['variance_std']:.3e}  "
            f"error = {out['error_mean']:.3e} ± {out['error_std']:.3e}"
        )

    return depth_summary
# ======================================================
# Main
# ======================================================

def main():
    print("DEVICE:", DEVICE)
    print("BENCHMARK:", BENCHMARK)
    print("DEPTHS:", DEPTHS)
    print("TRAIN_SEEDS:", TRAIN_SEEDS)
    print("TOTAL CONFIGS:", len(CONFIGS))

    all_results = {}

    uniform_rows = run_uniform()
    vegas_rows = run_vegas()

    all_results["Uniform MC"] = uniform_rows
    all_results["VEGAS"] = vegas_rows

    for cfg in CONFIGS:
        rows = train_and_eval(cfg)
        all_results[cfg["label"]] = rows

    save_json(all_results, f"{OUT_DIR}/{BENCHMARK}_d{DIM}_all_results.json")

    plot_metric(
        all_results,
        metric="estimate",
        ylabel="Integral estimate",
        filename="estimate_comparison.png",
        logy=False,
    )

    plot_metric(
        all_results,
        metric="variance",
        ylabel="Estimator variance",
        filename="variance_comparison.png",
        logy=True,
    )

    summary_rows = summarize_final_results(all_results)
    save_json(summary_rows, f"{OUT_DIR}/depth_sweep_summary.json")

    depth_summary = summarize_by_depth(summary_rows)
    save_json(depth_summary, f"{OUT_DIR}/depth_sweep_grouped_by_depth.json")

    print("\nDone.")
    print("Results:", OUT_DIR)


if __name__ == "__main__":
    main()