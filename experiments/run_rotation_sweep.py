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
BENCHMARK = "sharp_peak_002_003"
benchmark_fn = get_benchmark(BENCHMARK)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EVENTS = list(range(10000, 100001, 10000))
VEGAS_NITN = 10
N_BINS = 128

ROOT_DIR = "results/rotation_sweep"
OUT_DIR = f"{ROOT_DIR}/{BENCHMARK}_bins{N_BINS}"
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
# Rotation sweep configs
# ======================================================


ANGLES = {
    "none": None,
    "pi12": np.pi / 12,
    "pi6": np.pi / 6,
    "pi4": np.pi / 4,
    "pi3": np.pi / 3,
}


def make_rotation_configs():
    cfgs = []

    for angle_name, theta in ANGLES.items():
        use_rotation = theta is not None

        cfgs.append(
            {
                "name": f"rq_spline_bins{N_BINS}_rot_{angle_name}",
                "label": f"RQ spline bins={N_BINS}, rotation={angle_name}",
                "model_type": "rq_spline",
                "seed": 0,
                "model_kwargs": dict(
                    dim=DIM,
                    n_blocks=8,
                    hidden=128,
                    n_bins=N_BINS,
                    perm_sequence="01010101",
                    use_rotation=use_rotation,
                    rotation_angle=0.0 if theta is None else theta,
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
        )

    return cfgs


CONFIGS = make_rotation_configs()

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

    def contour(
        Z,
        title,
        cbar_label,
        filename,
        q_low=None,
        q_high=None,
        overlay_F=None,
    ):
        plt.figure(figsize=(7, 6))

        Z_plot = capped(Z, q_low=q_low, q_high=q_high)

        c = plt.contourf(X1c, X2c, Z_plot, levels=80)
        plt.colorbar(c, label=cbar_label)

        if overlay_F is not None:
            F_overlay = capped(overlay_F, q_high=0.99)
            plt.contour(
                X1c,
                X2c,
                F_overlay,
                levels=6,
                linewidths=0.8,
            )

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
        f"Learned q(x): {label}",
        r"$q_\theta(x)$",
        f"{name}_learned_q.png",
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

def make_slice_diagnostic_plots(
    model,
    name,
    label,
    x2_values=(0.26, 0.29, 0.32, 0.35, 0.38, 0.41, 0.44),
    n_grid=1000,
    integral_ref=None,
):
    """
    Slice diagnostic:
    Fix x2 and scan x1. Compare target f(x) and learned proposal q_theta(x).

    This is useful for checking whether the proposal learns a compact peak
    or spreads density into horizontal/vertical stripes.
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
            log_w = torch.log(f_vals + 1e-12) - log_q

        x1_np = x1.detach().cpu().numpy()
        f_np = f_vals.detach().cpu().numpy()
        q_np = q_vals.detach().cpu().numpy()
        logw_np = log_w.detach().cpu().numpy()

        if integral_ref is not None:
            f_density_np = f_np / integral_ref
        else:
            f_density_np = f_np

        # Normalize f and q separately so we compare shapes.
        f_norm = f_np / (np.max(f_np) + 1e-12)
        q_norm = q_np / (np.max(q_np) + 1e-12)

        # --------------------------------------------------
        # Plot normalized f and q on the same slice
        # --------------------------------------------------
        plt.figure(figsize=(7, 4))
        plt.plot(x1_np, f_norm, label=r"Target $f$, normalized")
        plt.plot(x1_np, q_norm, label=r"Proposal $q_\theta$, normalized")
        plt.axvline(0.65, linestyle="--", linewidth=1, label=r"Peak center $x_1=0.65$")
        plt.title(fr"Slice comparison at $x_2={x2_value:.3f}$: {label}")
        plt.xlabel(r"$x_1$")
        plt.ylabel("Normalized value")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            f"{PLOT_DIR}/{name}_slice_x1_at_x2_{x2_value:.3f}_fq_normalized.png",
            dpi=300,
        )
        plt.close()

        # --------------------------------------------------
        # Plot absolute f and q values
        # Useful to see if q over/under-estimates amplitude.
        # --------------------------------------------------
        plt.figure(figsize=(7, 4))
        if integral_ref is not None:
            plt.plot(x1_np, f_density_np, label=r"Target density $f/I_{\rm ref}$")
            plt.ylabel("Density value")
            filename_suffix = "fq_absolute_scaled"
        else:
            plt.plot(x1_np, f_np, label=r"Target $f$")
            plt.ylabel("Value")
            filename_suffix = "fq_absolute"
        
        plt.plot(x1_np, q_np, label=r"Proposal $q_\theta$")
        plt.axvline(0.65, linestyle="--", linewidth=1, label=r"Peak center $x_1=0.65$")
        plt.title(fr"Absolute slice at $x_2={x2_value:.3f}$: {label}")
        plt.xlabel(r"$x_1$")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            f"{PLOT_DIR}/{name}_slice_x1_at_x2_{x2_value:.3f}_{filename_suffix}.png",
            dpi=300,
        )
        plt.close()



        plt.figure(figsize=(7, 4))
        if integral_ref is not None:
            plt.semilogy(x1_np, f_density_np + 1e-12, label=r"Target density $f/I_{\rm ref}$")
            filename_suffix = "fq_absolute_scaled_log"
        else:
            plt.semilogy(x1_np, f_np + 1e-12, label=r"Target $f$")
            filename_suffix = "fq_absolute_log"

        plt.semilogy(x1_np, q_np + 1e-12, label=r"Proposal $q_\theta$")
        plt.axvline(0.65, linestyle="--", linewidth=1, label=r"Peak center $x_1=0.65$")
        plt.title(fr"Log-scale density slice at $x_2={x2_value:.3f}$: {label}")
        plt.xlabel(r"$x_1$")
        plt.ylabel("Density value, log scale")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            f"{PLOT_DIR}/{name}_slice_x1_at_x2_{x2_value:.3f}_{filename_suffix}.png",
            dpi=300,
        )
        plt.close()
        # --------------------------------------------------
        # Plot log weight on the slice
        # Ideally this should be flat near important regions.
        # --------------------------------------------------
        finite = np.isfinite(logw_np)

        if np.any(finite):
            lo = np.percentile(logw_np[finite], 1)
            hi = np.percentile(logw_np[finite], 99)
            logw_plot = np.clip(logw_np, lo, hi)
        else:
            logw_plot = logw_np

        plt.figure(figsize=(7, 4))
        plt.plot(x1_np, logw_plot, label=r"$\log f-\log q_\theta$")
        plt.axvline(0.65, linestyle="--", linewidth=1, label=r"Peak center $x_1=0.65$")
        plt.title(fr"Log-weight slice at $x_2={x2_value:.3f}$: {label}")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$\log f-\log q_\theta$, capped 1--99%")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            f"{PLOT_DIR}/{name}_slice_x1_at_x2_{x2_value:.3f}_log_weight.png",
            dpi=300,
        )
        plt.close()

def make_density_comparison_panel(model, name, label, integral_ref, n_grid=300):
    if integral_ref is None:
        print(f"Skipping density comparison panel for {name}: integral_ref is None.")
        return

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

        target_density = f_vals / integral_ref
        log_target_density = torch.log(target_density + 1e-12)

        q_vals = torch.exp(log_q)
        log_density_ratio = log_target_density - log_q

    P = target_density.reshape(n_grid, n_grid).detach().cpu().numpy()
    Q = q_vals.reshape(n_grid, n_grid).detach().cpu().numpy()
    R = log_density_ratio.reshape(n_grid, n_grid).detach().cpu().numpy()

    X1c = X1.detach().cpu().numpy()
    X2c = X2.detach().cpu().numpy()

    def cap_np(Z, q_low=None, q_high=None):
        Z_flat = Z[np.isfinite(Z)]
        if Z_flat.size == 0:
            return Z

        Z_plot = Z.copy()

        if q_low is not None:
            lo = np.quantile(Z_flat, q_low)
            Z_plot = np.maximum(Z_plot, lo)

        if q_high is not None:
            hi = np.quantile(Z_flat, q_high)
            Z_plot = np.minimum(Z_plot, hi)

        return Z_plot

    P_plot = cap_np(P, q_high=0.99)
    Q_plot = cap_np(Q, q_high=0.99)
    R_plot = cap_np(R, q_low=0.01, q_high=0.99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    c0 = axes[0].contourf(X1c, X2c, P_plot, levels=80)
    fig.colorbar(c0, ax=axes[0])
    axes[0].set_title(r"Target density $f/I_{\rm ref}$")

    c1 = axes[1].contourf(X1c, X2c, Q_plot, levels=80)
    fig.colorbar(c1, ax=axes[1])
    axes[1].set_title(r"Learned density $q_\theta$")

    c2 = axes[2].contourf(X1c, X2c, R_plot, levels=80)
    fig.colorbar(c2, ax=axes[2])
    axes[2].set_title(r"$\log(f/I_{\rm ref})-\log q_\theta$")

    for ax in axes:
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Density comparison: {label}", fontsize=14)

    plt.savefig(
        f"{PLOT_DIR}/{name}_density_comparison_panel.png",
        dpi=300,
    )
    plt.close()
# ======================================================
# Train/evaluate RQ
# ======================================================

def train_and_eval(cfg,integral_ref=None):
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
    make_density_comparison_panel(
        model,
        name,
        cfg["label"],
        integral_ref = integral_ref,
    )
    
    make_slice_diagnostic_plots(
        model,
        name,
        cfg["label"],
        x2_values=(0.26, 0.29, 0.32, 0.35, 0.38, 0.41, 0.44),
        integral_ref=integral_ref,
    )

    return rows


# ======================================================
# Comparison plots
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


# ======================================================
# Main
# ======================================================

def main():
    print("DEVICE:", DEVICE)
    print("BENCHMARK:", BENCHMARK)
    print("N_BINS:", N_BINS)
    print("ANGLES:", list(ANGLES.keys()))

    all_results = {}

    uniform_rows = run_uniform()
    vegas_rows = run_vegas()

    all_results["Uniform MC"] = uniform_rows
    all_results["VEGAS"] = vegas_rows
    
    I_REF = vegas_rows[-1]["estimate"]
    print("I_REF from final VEGAS estimate:", I_REF)
    
    for cfg in CONFIGS:
        rows = train_and_eval(cfg,integral_ref=I_REF)
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
    save_json(summary_rows, f"{OUT_DIR}/rotation_sweep_summary.json")

    print("\nDone.")
    print("Results:", OUT_DIR)


if __name__ == "__main__":
    main()