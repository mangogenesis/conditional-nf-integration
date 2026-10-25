import os
import json
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from nf.conditional_rq_1d import ConditionalRQFlow1D


# ======================================================
# Global settings
# ======================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "results/conditional_gaussian_1d_sweep"
PLOT_DIR = f"{OUT_DIR}/plots"
CKPT_DIR = f"{OUT_DIR}/checkpoints"
SUMMARY_DIR = f"{OUT_DIR}/summaries"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Grouped conditional training
N_COND = 1024
N_X_PER_COND = 512
BATCH_SIZE = N_COND * N_X_PER_COND

STEPS = 10000
LR = 5e-5
HIDDEN = 128

MU_RANGE = (0.20, 0.80)
SIGMA_RANGE = (0.03, 0.12)
FLOOR = 1e-6

TEST_CONDITIONS = [
    (0.30, 0.05),
    (0.50, 0.08),
    (0.75, 0.04),
    (0.25, 0.10),
    (0.65, 0.06),
    (0.50, 0.03),
    (0.50, 0.12),
]

N_EVAL = 100000

SWEEP_CONFIGS = [
    {"n_bins": 4, "reg_lambda": 0.0,  "name": "bins4_noreg"},
    {"n_bins": 6, "reg_lambda": 0.0,  "name": "bins6_noreg"},
    {"n_bins": 8, "reg_lambda": 0.0,  "name": "bins8_noreg"},
    {"n_bins": 8, "reg_lambda": 1e-3, "name": "bins8_reg1e3"},
    {"n_bins": 16, "reg_lambda": 0.0, "name": "bins16_noreg"},
    {"n_bins": 32, "reg_lambda": 0.0, "name": "bins32_noreg"},
]


# ======================================================
# Reproducibility
# ======================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================
# Benchmark
# ======================================================

def gaussian_1d(x, mu, sigma, floor=FLOOR):
    z = ((x - mu) / sigma) ** 2
    return floor + torch.exp(-0.5 * z)


def exact_integral_1d(mu, sigma, floor=FLOOR):
    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    a = (0.0 - mu) / sigma
    b = (1.0 - mu) / sigma

    peak = sigma * math.sqrt(2.0 * math.pi) * (Phi(b) - Phi(a))

    return floor + peak


def sample_grouped_batch(n_cond, n_x_per_cond, device):
    mu_cond = torch.empty(n_cond, 1, device=device).uniform_(*MU_RANGE)
    sigma_cond = torch.empty(n_cond, 1, device=device).uniform_(*SIGMA_RANGE)

    cond_base = torch.cat([mu_cond, sigma_cond], dim=1)

    x = torch.rand(n_cond, n_x_per_cond, 1, device=device)

    mu = mu_cond[:, None, :].expand(n_cond, n_x_per_cond, 1)
    sigma = sigma_cond[:, None, :].expand(n_cond, n_x_per_cond, 1)
    cond = cond_base[:, None, :].expand(n_cond, n_x_per_cond, 2)

    x_flat = x.reshape(-1, 1)
    mu_flat = mu.reshape(-1, 1)
    sigma_flat = sigma.reshape(-1, 1)
    cond_flat = cond.reshape(-1, 2)

    return x_flat, mu_flat, sigma_flat, cond_flat, cond_base


# ======================================================
# Helpers
# ======================================================

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def condition_name(mu, sigma):
    return f"mu{mu:.2f}_sigma{sigma:.2f}".replace(".", "p")


def add_accuracy_metrics(res, integral_ref):
    estimate = res["estimate"]
    variance = res["variance"]

    std_error = math.sqrt(variance) if variance >= 0 else np.nan
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


def effective_sample_size(weights):
    s1 = weights.sum()
    s2 = (weights ** 2).sum()
    return float((s1 ** 2 / (s2 + 1e-12)).item())


# ======================================================
# Evaluation
# ======================================================

def cnf_integrate(model, mu_value, sigma_value, n_events=N_EVAL):
    model.eval()

    cond = torch.tensor(
        [[mu_value, sigma_value]],
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        x = model.sample(n_events, cond, device=DEVICE)

        mu = torch.full_like(x, mu_value)
        sigma = torch.full_like(x, sigma_value)

        f_vals = gaussian_1d(x, mu, sigma)
        log_q = model.log_prob(x, cond.expand(n_events, -1))

        f_vals = f_vals.reshape(-1)
        log_q = log_q.reshape(-1)

        weights = f_vals * torch.exp(-log_q)

        estimate = weights.mean()
        weight_variance = weights.var(unbiased=True)
        variance = weight_variance / n_events
        ess = effective_sample_size(weights)

    return {
        "estimate": float(estimate.item()),
        "weight_variance": float(weight_variance.item()),
        "variance": float(variance.item()),
        "error": float(torch.sqrt(variance).item()),
        "ess": ess,
        "events": n_events,
    }


def density_error_metrics(model, mu_value, sigma_value, n_grid=2000):
    model.eval()

    x = torch.linspace(0, 1, n_grid, device=DEVICE).reshape(-1, 1)
    mu = torch.full_like(x, mu_value)
    sigma = torch.full_like(x, sigma_value)
    cond = torch.cat([mu, sigma], dim=1)

    with torch.no_grad():
        f_vals = gaussian_1d(x, mu, sigma)
        I_ref = exact_integral_1d(mu_value, sigma_value)
        target = (f_vals / I_ref).reshape(-1)

        log_q = model.log_prob(x, cond).reshape(-1)
        q_vals = torch.exp(log_q)

        log_ratio = torch.log(target + 1e-12) - log_q

        ise = torch.mean((q_vals - target) ** 2)
        mae = torch.mean(torch.abs(q_vals - target))

        # Focus on important target region too
        important = target > 0.01 * target.max()
        if important.any():
            max_abs_log_ratio_important = torch.max(torch.abs(log_ratio[important]))
        else:
            max_abs_log_ratio_important = torch.tensor(float("nan"), device=DEVICE)

    return {
        "density_ise": float(ise.item()),
        "density_mae": float(mae.item()),
        "max_abs_log_ratio_important": float(max_abs_log_ratio_important.item()),
    }


# ======================================================
# Training
# ======================================================

def build_model(n_bins):
    return ConditionalRQFlow1D(
        cond_dim=2,
        hidden=HIDDEN,
        n_bins=n_bins,
    ).to(DEVICE)


def train_one_config(cfg):
    n_bins = cfg["n_bins"]
    reg_lambda = cfg["reg_lambda"]
    name = cfg["name"]

    set_seed(0)

    model = build_model(n_bins)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []

    print("\n" + "=" * 100)
    print(f"Training config: {name}")
    print("=" * 100)
    print("DEVICE:", DEVICE)
    print("N_COND:", N_COND)
    print("N_X_PER_COND:", N_X_PER_COND)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("STEPS:", STEPS)
    print("N_BINS:", n_bins)
    print("LR:", LR)
    print("REG_LAMBDA:", reg_lambda)

    for step in range(STEPS):
        x_flat, mu_flat, sigma_flat, cond_flat, cond_base = sample_grouped_batch(
            n_cond=N_COND,
            n_x_per_cond=N_X_PER_COND,
            device=DEVICE,
        )

        f_vals = gaussian_1d(x_flat, mu_flat, sigma_flat)
        log_q = model.log_prob(x_flat, cond_flat)

        f_vals = f_vals.reshape(N_COND, N_X_PER_COND)
        log_q = log_q.reshape(N_COND, N_X_PER_COND)

        weights = f_vals.detach()

        loss_per_cond = -(
            weights * log_q
        ).sum(dim=1) / (weights.sum(dim=1) + 1e-12)

        main_loss = loss_per_cond.mean()

        if reg_lambda > 0:
            reg = model.spline_regularization(cond_base)
            loss = main_loss + reg_lambda * reg
        else:
            reg = torch.tensor(0.0, device=DEVICE)
            loss = main_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()

        loss_value = float(loss.item())
        main_value = float(main_loss.item())
        reg_value = float(reg.item())

        history.append(
            {
                "step": step,
                "loss": loss_value,
                "main_loss": main_value,
                "reg": reg_value,
            }
        )

        if step % 200 == 0:
            print(
                f"step {step:6d} | "
                f"loss = {loss_value:.6f} | "
                f"main = {main_value:.6f} | "
                f"reg = {reg_value:.6f}"
            )

    ckpt_path = f"{CKPT_DIR}/{name}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "config": {
                "name": name,
                "n_bins": n_bins,
                "hidden": HIDDEN,
                "mu_range": MU_RANGE,
                "sigma_range": SIGMA_RANGE,
                "n_cond": N_COND,
                "n_x_per_cond": N_X_PER_COND,
                "batch_size": BATCH_SIZE,
                "steps": STEPS,
                "lr": LR,
                "reg_lambda": reg_lambda,
            },
        },
        ckpt_path,
    )

    print("Saved checkpoint:", ckpt_path)

    make_loss_plot(history, name)

    return model, history


# ======================================================
# Plots
# ======================================================

def make_loss_plot(history, name):
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    main_losses = [h["main_loss"] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, label="total loss")
    plt.plot(steps, main_losses, label="main loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Training loss: {name}")
    plt.grid(False)
    plt.legend(frameon=False)
    plt.tight_layout()

    path = f"{PLOT_DIR}/{name}_loss.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


def plot_condition(model, cfg, mu_value, sigma_value, n_grid=1200):
    name = cfg["name"]

    model.eval()

    x = torch.linspace(0, 1, n_grid, device=DEVICE).reshape(-1, 1)
    mu = torch.full_like(x, mu_value)
    sigma = torch.full_like(x, sigma_value)
    cond = torch.cat([mu, sigma], dim=1)

    with torch.no_grad():
        f_vals = gaussian_1d(x, mu, sigma)
        I_ref = exact_integral_1d(mu_value, sigma_value)
        target = f_vals / I_ref

        log_q = model.log_prob(x, cond)
        q_vals = torch.exp(log_q)

        target = target.reshape(-1)
        log_q = log_q.reshape(-1)
        q_vals = q_vals.reshape(-1)

        log_ratio = torch.log(target + 1e-12) - log_q

    x_np = x.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)
    q_np = q_vals.detach().cpu().numpy().reshape(-1)
    log_ratio_np = log_ratio.detach().cpu().numpy().reshape(-1)

    cond_name = condition_name(mu_value, sigma_value)

    plt.figure(figsize=(7, 4))
    plt.plot(x_np, target_np, linewidth=2.0, label=r"Target density $f/I$")
    plt.plot(x_np, q_np, linewidth=1.6, label=r"CNF proposal $q_\theta$")
    plt.axvline(mu_value, linestyle="--", linewidth=1.0, label=r"$\mu$")
    plt.xlabel(r"$x$")
    plt.ylabel("Density")
    plt.title(
        fr"{name}: $\mu={mu_value:.2f}$, $\sigma={sigma_value:.2f}$"
    )
    plt.grid(False)
    plt.legend(frameon=False)
    plt.tight_layout()

    path = f"{PLOT_DIR}/{name}_{cond_name}_density.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

    finite = np.isfinite(log_ratio_np)
    if np.any(finite):
        lo = np.percentile(log_ratio_np[finite], 1)
        hi = np.percentile(log_ratio_np[finite], 99)
        log_ratio_plot = np.clip(log_ratio_np, lo, hi)
    else:
        log_ratio_plot = log_ratio_np

    plt.figure(figsize=(7, 3.5))
    plt.plot(x_np, log_ratio_plot, linewidth=1.5)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(mu_value, linestyle="--", linewidth=1.0)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\log(f/I)-\log q_\theta$")
    plt.title(
        fr"Log-ratio {name}: $\mu={mu_value:.2f}$, $\sigma={sigma_value:.2f}$"
    )
    plt.grid(False)
    plt.tight_layout()

    path = f"{PLOT_DIR}/{name}_{cond_name}_log_ratio.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


def make_contact_sheet_for_config(cfg):
    try:
        from PIL import Image, ImageOps
    except ImportError:
        print("PIL not installed; skipping contact sheet.")
        return

    name = cfg["name"]

    image_paths = []

    for mu, sigma in TEST_CONDITIONS:
        cond_name = condition_name(mu, sigma)
        image_paths.append(f"{PLOT_DIR}/{name}_{cond_name}_density.png")

    images = []

    for path in image_paths:
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert("RGB")
        img = ImageOps.expand(img, border=8, fill="white")
        images.append(img)

    if len(images) == 0:
        return

    thumb_w = 520
    thumb_h = 320

    thumbs = []

    for img in images:
        img.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        x0 = (thumb_w - img.width) // 2
        y0 = (thumb_h - img.height) // 2
        canvas.paste(img, (x0, y0))
        thumbs.append(canvas)

    n_cols = 2
    n_rows = int(math.ceil(len(thumbs) / n_cols))

    sheet = Image.new("RGB", (n_cols * thumb_w, n_rows * thumb_h), "white")

    for i, img in enumerate(thumbs):
        r = i // n_cols
        c = i % n_cols
        sheet.paste(img, (c * thumb_w, r * thumb_h))

    path = f"{PLOT_DIR}/{name}_contact_sheet.png"
    sheet.save(path)

    print("Saved:", path)


# ======================================================
# Main
# ======================================================

def main():
    print("DEVICE:", DEVICE)
    print("Sweep configs:")
    for cfg in SWEEP_CONFIGS:
        print(cfg)

    all_summary_rows = []

    for cfg in SWEEP_CONFIGS:
        model, history = train_one_config(cfg)

        cfg_summary_rows = []

        for mu_value, sigma_value in TEST_CONDITIONS:
            plot_condition(model, cfg, mu_value, sigma_value)

            I_ref = exact_integral_1d(mu_value, sigma_value)

            res = cnf_integrate(
                model=model,
                mu_value=mu_value,
                sigma_value=sigma_value,
                n_events=N_EVAL,
            )

            res = add_accuracy_metrics(res, I_ref)

            density_metrics = density_error_metrics(
                model=model,
                mu_value=mu_value,
                sigma_value=sigma_value,
            )

            res.update(density_metrics)

            res.update(
                {
                    "config_name": cfg["name"],
                    "n_bins": cfg["n_bins"],
                    "reg_lambda": cfg["reg_lambda"],
                    "mu": mu_value,
                    "sigma": sigma_value,
                    "label": (
                        f"{cfg['name']} "
                        f"mu={mu_value:.2f}, sigma={sigma_value:.2f}"
                    ),
                }
            )

            cfg_summary_rows.append(res)
            all_summary_rows.append(res)

            print(
                f"{cfg['name']:14s} | "
                f"mu={mu_value:.2f}, sigma={sigma_value:.2f} | "
                f"estimate={res['estimate']:.8e}, "
                f"I_ref={res['integral_ref']:.8e}, "
                f"variance={res['variance']:.3e}, "
                f"rel_err={res['rel_error']:.3e}, "
                f"ESS={res['ess']:.3e}, "
                f"ISE={res['density_ise']:.3e}, "
                f"max_log_ratio_imp={res['max_abs_log_ratio_important']:.3e}"
            )

        save_json(
            cfg_summary_rows,
            f"{SUMMARY_DIR}/{cfg['name']}_summary.json",
        )

        make_contact_sheet_for_config(cfg)

    save_json(
        all_summary_rows,
        f"{SUMMARY_DIR}/all_sweep_summary.json",
    )

    summarize_sweep(all_summary_rows)

    print("\nDone.")
    print("Results:", OUT_DIR)


def summarize_sweep(rows):
    print("\n" + "=" * 120)
    print("SWEEP SUMMARY BY CONFIG")
    print("=" * 120)

    config_names = sorted(set(r["config_name"] for r in rows))

    summary = []

    for name in config_names:
        sub = [r for r in rows if r["config_name"] == name]

        mean_var = np.mean([r["variance"] for r in sub])
        mean_rel_err = np.mean([r["rel_error"] for r in sub])
        mean_ess = np.mean([r["ess"] for r in sub])
        mean_ise = np.mean([r["density_ise"] for r in sub])
        mean_log_ratio = np.mean([r["max_abs_log_ratio_important"] for r in sub])

        row = {
            "config_name": name,
            "mean_variance": float(mean_var),
            "mean_rel_error": float(mean_rel_err),
            "mean_ess": float(mean_ess),
            "mean_density_ise": float(mean_ise),
            "mean_max_abs_log_ratio_important": float(mean_log_ratio),
        }

        summary.append(row)

        print(
            f"{name:14s} | "
            f"mean variance={mean_var:.3e} | "
            f"mean rel_err={mean_rel_err:.3e} | "
            f"mean ESS={mean_ess:.3e} | "
            f"mean ISE={mean_ise:.3e} | "
            f"mean max log-ratio important={mean_log_ratio:.3e}"
        )

    save_json(summary, f"{SUMMARY_DIR}/sweep_summary_by_config.json")


if __name__ == "__main__":
    main()