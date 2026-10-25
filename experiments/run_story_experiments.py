import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from benchmarks.registry import get_benchmark
from mc_baselines.uniform_mc import uniform_mc_integrate
from mc_baselines.vegas_integrate import vegas_integrate

from nf.cube_flow import CubeRealNVP
from nf.cube_piecewise_spline_flow import CubePiecewiseSplineFlow
from nf.cube_spline_flow import CubeSplineFlow
from nf.cube_affine_flow import CubeAffineFlow

from nf.train import train_nf, save_nf_checkpoint
from nf.evaluate import nf_integrate


# ======================================================
# Global settings
# ======================================================

DIM = 2
# BENCHMARK = "sharp_peak"
BENCHMARK = "asymmetric_multi_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EVENTS = list(range(10000, 100001, 10000))
VEGAS_NITN = 10

OUT_DIR = f"results/{BENCHMARK}_permutation"
PLOT_DIR = f"{OUT_DIR}/plots"
CKPT_DIR = f"{OUT_DIR}/checkpoints"
PERM_DIR = f"{OUT_DIR}/permutations"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PERM_DIR, exist_ok=True)

EDGE_CUT = 0

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

# Configs

# ======================================================

# Deterministic permutation experiments.

# In 2D:

#   0 means [0, 1]

#   1 means [1, 0]

#

# Since n_blocks = 8 for Models 2--4, we use length-8 strings.

PERMUTATION_PATTERNS = {

    "all0": "00000000",

    "all1": "11111111",

    "alt01": "01010101",

    "alt10": "10101010",

}

def make_seeded_configs():

    cfgs = []

    # ------------------------------------------------------

    # Model 1: RealNVP + Gaussian base + sigmoid

    # No cube permutation experiment here.

    # ------------------------------------------------------

    cfgs.append(

        {

            "name": "realnvp_sigmoid",

            "label": "RealNVP + Gaussian base + sigmoid",

            "model_type": "realnvp",

            "seed": 0,

            "model_kwargs": dict(

                dim=DIM,

                n_blocks=12,

                hidden=128,

                eps_mix=0.0,

            ),

            "train_kwargs": dict(

                steps=10000,

                batch_size=300000,

                lr=1e-4,

                loss_name="kl",

                tol=1e-4,

                patience=5,

                log_every=200,

            ),

        }

    )

    # # ------------------------------------------------------

    # # Model 2: Cube affine coupling

    # # Test all-0 and all-1 deterministic permutation patterns.

    # # ------------------------------------------------------

    # for pattern_name, pattern in PERMUTATION_PATTERNS.items():

    #     cfgs.append(

    #         {

    #             "name": f"cube_affine_{pattern_name}",

    #             "label": f"Cube affine (perm={pattern})",

    #             "model_type": "cube_affine",

    #             "seed": 0,

    #             "perm_sequence": pattern,

    #             "model_kwargs": dict(

    #                 dim=DIM,

    #                 n_blocks=8,

    #                 hidden=128,

    #                 # Build normally, then overwrite permutations below.

    #                 permute="random",

    #                 seed=0,

    #             ),

    #             "train_kwargs": dict(

    #                 steps=20000,

    #                 batch_size=300000,

    #                 lr=1e-5,

    #                 loss_name="kl",

    #                 tol=5e-4,

    #                 patience=5,

    #                 log_every=200,

    #             ),

    #         }

    #     )

    # ------------------------------------------------------

    # Model 3: Piecewise linear spline flow

    # ------------------------------------------------------

    for pattern_name, pattern in PERMUTATION_PATTERNS.items():

        cfgs.append(

            {

                "name": f"piecewise_{pattern_name}",

                "label": f"Piecewise spline (perm={pattern})",

                "model_type": "piecewise",

                "seed": 0,

                "perm_sequence": pattern,

                "model_kwargs": dict(

                    dim=DIM,

                    n_blocks=8,

                    hidden=128,

                    n_bins=32,

                    perm_sequence=pattern,

                ),

                "train_kwargs": dict(

                    steps=750,

                    batch_size=300000,

                    lr=1e-5,

                    loss_name="kl",

                    tol=5e-4,

                    patience=5,

                    log_every=200,

                ),

            }

        )

    # ------------------------------------------------------

    # Model 4: Rational quadratic spline flow

    # ------------------------------------------------------

    for pattern_name, pattern in PERMUTATION_PATTERNS.items():

        cfgs.append(

            {

                "name": f"rq_spline_{pattern_name}",

                "label": f"RQ spline (perm={pattern})",

                "model_type": "rq_spline",

                "seed": 0,

                "perm_sequence": pattern,

                "model_kwargs": dict(

                    dim=DIM,

                    n_blocks=8,

                    hidden=128,

                    n_bins=64,

                    perm_sequence=pattern,

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


CONFIGS = make_seeded_configs()


# ======================================================
# Helpers
# ======================================================

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_model(cfg):
    if cfg["model_type"] == "realnvp":
        return CubeRealNVP(**cfg["model_kwargs"]).to(DEVICE)

    if cfg["model_type"] == "cube_affine":
        return CubeAffineFlow(**cfg["model_kwargs"]).to(DEVICE)

    if cfg["model_type"] == "piecewise":
        return CubePiecewiseSplineFlow(**cfg["model_kwargs"]).to(DEVICE)

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


def save_permutation_info(cfg, model):
    name = cfg["name"]
    seed = cfg.get("seed", 0)

    perm_info = get_permutation_order(model)

    out = {
        "name": name,
        "label": cfg["label"],
        "seed": seed,
        "model_type": cfg["model_type"],
        "model_kwargs": cfg["model_kwargs"],
        "permutations": perm_info,
    }

    path = f"{PERM_DIR}/{name}_permutations.json"
    save_json(out, path)

    print("Permutation order:")
    if len(perm_info) == 0:
        print("  no permutation layers found")
    else:
        for p in perm_info:
            print(f"  layer {p['layer_index']}: {p['perm']}")

    return perm_info


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
# Loss plotting
# ======================================================

def make_loss_plots(history, name, label):
    history = np.asarray(history, dtype=float)

    # Remove non-finite values only for plotting limits.
    finite_history = history[np.isfinite(history)]

    if len(finite_history) == 0:
        print(f"Skipping loss plots for {name}: no finite values.")
        return

    # ---------------------------
    # Raw loss plot
    # ---------------------------

    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.title(f"Loss: {label}")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_loss_raw.png", dpi=300)
    plt.close()

    # Keep old filename so existing slides do not break.
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.title(f"Loss: {label}")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_loss.png", dpi=300)
    plt.close()

    # ---------------------------
    # Log-scale loss plot
    # Only valid if all finite values are positive.
    # ---------------------------

    if np.all(finite_history > 0):
        plt.figure(figsize=(7, 4))
        plt.plot(history)
        plt.yscale("log")
        plt.title(f"Loss: {label} log scale")
        plt.xlabel("Training step")
        plt.ylabel("Loss, log scale")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{name}_loss_log.png", dpi=300)
        plt.close()
    else:
        print(f"Skipping log-scale loss plot for {name}: loss contains non-positive values.")

    # ---------------------------
    # Capped y-axis loss plot
    # Useful if raw loss hides late-time behavior.
    # ---------------------------

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
        else:
            print(f"Skipping capped loss plot for {name}: invalid y-limits.")


# ======================================================
# Train/evaluate NF
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

    perm_info = save_permutation_info(cfg, model)

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

    # Reload saved checkpoint and evaluate.
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

    X = torch.stack(
        [X1.reshape(-1), X2.reshape(-1)],
        dim=1,
    )

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

        c = plt.contourf(X1c, X2c, Z_plot, levels=60)
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

    # ---------------------------
    # True benchmark
    # ---------------------------

    contour(
        F,
        "True benchmark: four-Gaussian structure",
        r"$f(x)$",
        f"{BENCHMARK}_true_f.png",
        q_high=0.99,
    )

    # ---------------------------
    # Learned proposal q(x)
    # ---------------------------

    contour(
        Q,
        f"Learned proposal q(x): {label}",
        r"$q_\theta(x)$",
        f"{name}_learned_q.png",
    )

    contour(
        Q,
        f"Learned proposal q(x), capped 99%: {label}",
        r"$q_\theta(x)$, capped",
        f"{name}_learned_q_capped99.png",
        q_high=0.99,
    )

    contour(
        Q,
        f"Learned proposal q(x), capped 1--99%: {label}",
        r"$q_\theta(x)$, capped",
        f"{name}_learned_q_capped_1_99.png",
        q_low=0.01,
        q_high=0.99,
    )

    # Compatibility filenames.
    # These no longer cut the edge; they are full-domain capped plots.
    contour(
        Q,
        f"Learned proposal q(x), edge capped 99%: {label}",
        r"$q_\theta(x)$, capped",
        f"{name}_learned_q_interior_capped99.png",
        q_high=0.99,
    )

    contour(
        Q,
        f"Learned proposal q(x), edge clipped 99%: {label}",
        r"$q_\theta(x)$, capped",
        f"{name}_learned_q_interior_clipped99.png",
        q_high=0.99,
    )

    # ---------------------------
    # Learned q with true f contours overlaid
    # ---------------------------

    contour(
        Q,
        f"Learned q(x) with true f contours: {label}",
        r"$q_\theta(x)$, capped",
        f"{name}_learned_q_with_true_contours.png",
        q_high=0.99,
        overlay_F=F,
    )

    contour(
        Q,
        f"Learned q(x) with true f contours, capped 1--99%: {label}",
        r"$q_\theta(x)$, capped",
        f"{name}_learned_q_with_true_contours_capped_1_99.png",
        q_low=0.01,
        q_high=0.99,
        overlay_F=F,
    )

    # Compatibility filename.
    contour(
        Q,
        f"Full-domain q(x) with true f contours: {label}",
        r"$q_\theta(x)$, capped",
        f"{name}_learned_q_interior_with_true_contours.png",
        q_high=0.99,
        overlay_F=F,
    )

    # ---------------------------
    # Log-weight landscape
    # ---------------------------

    contour(
        LOGW,
        f"Log weight log(f/q): {label}",
        r"$\log f - \log q_\theta$",
        f"{name}_log_weight.png",
    )

    contour(
        LOGW,
        f"Log weight, capped 1--99%: {label}",
        r"$\log f - \log q_\theta$, capped",
        f"{name}_log_weight_capped_1_99.png",
        q_low=0.01,
        q_high=0.99,
    )

    # Compatibility filenames.
    # These no longer cut the edge; they are full-domain capped plots.
    contour(
        LOGW,
        f"Log weight clipped 1--99%: {label}",
        r"$\log f - \log q_\theta$, capped",
        f"{name}_log_weight_clipped99.png",
        q_low=0.01,
        q_high=0.99,
    )

    contour(
        LOGW,
        f"Log weight edge-capped 1--99%: {label}",
        r"$\log f - \log q_\theta$, capped",
        f"{name}_log_weight_interior_capped_1_99.png",
        q_low=0.01,
        q_high=0.99,
    )

    contour(
        LOGW,
        f"Log weight edge-clipped 1--99%: {label}",
        r"$\log f - \log q_\theta$, capped",
        f"{name}_log_weight_interior_clipped99.png",
        q_low=0.01,
        q_high=0.99,
    )

    # ---------------------------
    # Weight landscape f/q
    # ---------------------------

    contour(
        W,
        f"Weight f/q, capped 99%: {label}",
        r"$f/q_\theta$, capped",
        f"{name}_weight_capped99.png",
        q_high=0.99,
    )

    contour(
        W,
        f"Weight f/q, capped 1--99%: {label}",
        r"$f/q_\theta$, capped",
        f"{name}_weight_capped_1_99.png",
        q_low=0.01,
        q_high=0.99,
    )

    # Compatibility filenames.
    contour(
        W,
        f"Weight f/q clipped at 99%: {label}",
        r"$f/q_\theta$, capped",
        f"{name}_weight_clipped99.png",
        q_high=0.99,
    )

    contour(
        W,
        f"Weight f/q edge-capped 99%: {label}",
        r"$f/q_\theta$, capped",
        f"{name}_weight_interior_capped99.png",
        q_high=0.99,
    )

    contour(
        W,
        f"Weight f/q edge-clipped 99%: {label}",
        r"$f/q_\theta$, capped",
        f"{name}_weight_interior_clipped99.png",
        q_high=0.99,
    )

    # ---------------------------
    # Samples
    # ---------------------------

    with torch.no_grad():
        samples = model.sample(20000, device=DEVICE).detach().cpu()

    plt.figure(figsize=(7, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"Samples: {label}")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_samples.png", dpi=300)
    plt.close()


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


def select_best_by_family(all_results):
    families = {}

    for label, rows in all_results.items():
        final_var = rows[-1]["variance"]

        if "RQ spline" in label:
            family = "RQ spline"
        elif "Piecewise spline" in label:
            family = "Piecewise spline"
        elif "Cube affine" in label:
            family = "Cube affine"
        elif "RealNVP" in label:
            family = "RealNVP + sigmoid"
        elif "VEGAS" in label:
            family = "VEGAS"
        elif "Uniform" in label:
            family = "Uniform MC"
        else:
            family = label

        if family not in families or final_var < families[family][1]:
            families[family] = (label, final_var, rows)

    best_results = {}

    for family, (best_label, final_var, rows) in families.items():
        best_results[f"{family}: {best_label}"] = rows

    return best_results


# ======================================================
# Summary
# ======================================================

def summarize_best_variance(all_results):
    print("\n" + "=" * 80)
    print("BEST VARIANCE SUMMARY")
    print("=" * 80)

    rows = []

    for label, results in all_results.items():
        final_var = results[-1]["variance"]
        rows.append((label, final_var))

    rows = sorted(rows, key=lambda x: x[1])

    for label, var in rows:
        print(f"{label:45s} variance = {var:.3e}")


def summarize_by_family(all_results):
    families = {}

    for label, results in all_results.items():
        final_var = results[-1]["variance"]

        if "RQ spline" in label:
            family = "RQ spline"
        elif "Piecewise spline" in label:
            family = "Piecewise spline"
        elif "Cube affine" in label:
            family = "Cube affine"
        elif "RealNVP" in label:
            family = "RealNVP + sigmoid"
        elif "VEGAS" in label:
            family = "VEGAS"
        elif "Uniform" in label:
            family = "Uniform MC"
        else:
            family = label

        if family not in families or final_var < families[family][1]:
            families[family] = (label, final_var)

    print("\n" + "=" * 80)
    print("BEST BY FAMILY")
    print("=" * 80)

    for family, (label, var) in sorted(families.items(), key=lambda x: x[1][1]):
        print(f"{family:30s} best = {label:45s} variance = {var:.3e}")


def summarize_final_results(all_results):
    print("\n" + "=" * 100)
    print("FINAL RESULT SUMMARY")
    print("=" * 100)

    rows = []

    for label, results in all_results.items():
        final = results[-1]

        estimate = final.get("estimate", np.nan)
        variance = final.get("variance", np.nan)
        error = final.get("error", np.nan)
        ess = final.get("ess", np.nan)

        rows.append((label, estimate, variance, error, ess))

    rows = sorted(rows, key=lambda x: x[2])

    for label, estimate, variance, error, ess in rows:
        print(
            f"{label:45s} "
            f"estimate = {estimate:.8e}  "
            f"variance = {variance:.3e}  "
            f"error = {error:.3e}  "
            f"ESS = {ess:.3e}"
        )
# ======================================================
# Main
# ======================================================

def main():
    print("DEVICE:", DEVICE)
    print("BENCHMARK:", BENCHMARK)
    print("CONFIG COUNT:", len(CONFIGS))
    print("EDGE_CUT:", EDGE_CUT)

    all_results = {}

    uniform_rows = run_uniform()
    vegas_rows = run_vegas()

    all_results["Uniform MC"] = uniform_rows
    all_results["VEGAS"] = vegas_rows

    for cfg in CONFIGS:
        rows = train_and_eval(cfg)
        all_results[cfg["label"]] = rows

    save_json(all_results, f"{OUT_DIR}/{BENCHMARK}_d{DIM}_all_results.json")

    # Full comparison: all seeds and all methods.
    plot_metric(
        all_results,
        metric="estimate",
        ylabel="Integral estimate",
        filename="story_estimate_comparison.png",
        logy=False,
    )

    plot_metric(
        all_results,
        metric="variance",
        ylabel="Estimator variance",
        filename="story_variance_comparison.png",
        logy=True,
    )

    plot_metric(
        all_results,
        metric="runtime",
        ylabel="Runtime (s)",
        filename="story_runtime_comparison.png",
        logy=False,
    )

    # Cleaner presentation comparison: best seed per model family.
    best_by_family_results = select_best_by_family(all_results)

    plot_metric(
        best_by_family_results,
        metric="estimate",
        ylabel="Integral estimate",
        filename="story_estimate_best_by_family.png",
        logy=False,
    )

    plot_metric(
        best_by_family_results,
        metric="variance",
        ylabel="Estimator variance",
        filename="story_variance_best_by_family.png",
        logy=True,
    )

    plot_metric(
        best_by_family_results,
        metric="runtime",
        ylabel="Runtime (s)",
        filename="story_runtime_best_by_family.png",
        logy=False,
    )

    print("\nDone.")
    print("Results:", OUT_DIR)
    print("Plots:", PLOT_DIR)
    print("Permutations:", PERM_DIR)
    summarize_final_results(all_results)
    summarize_best_variance(all_results)
    summarize_by_family(all_results)


if __name__ == "__main__":
    main()