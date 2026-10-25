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

OUT_DIR = "results/conditional_gaussian_1d"
PLOT_DIR = f"{OUT_DIR}/plots"
CKPT_DIR = f"{OUT_DIR}/checkpoints"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Grouped conditional training:
# N_COND = number of different conditions per step
# N_X_PER_COND = number of x samples for each condition
N_COND = 1024
N_X_PER_COND = 512
BATCH_SIZE = N_COND * N_X_PER_COND

STEPS = 10000
LR = 5e-5
REG_LAMBDA = 1e-3

N_BINS = 8
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
]

N_EVAL = 100000


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
    """
    x:     (N,1)
    mu:    (N,1)
    sigma: (N,1)
    """
    z = ((x - mu) / sigma) ** 2
    return floor + torch.exp(-0.5 * z)


def exact_integral_1d(mu, sigma, floor=FLOOR):
    """
    Exact integral of floor + Gaussian over [0,1].
    """

    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    a = (0.0 - mu) / sigma
    b = (1.0 - mu) / sigma

    peak = sigma * math.sqrt(2.0 * math.pi) * (Phi(b) - Phi(a))

    return floor + peak


def sample_grouped_batch(n_cond, n_x_per_cond, device):
    """
    Sample grouped conditional batch.

    Returns flattened tensors for model evaluation, plus shape info.

    We sample:
        c_b = (mu_b, sigma_b), b = 1,...,N_COND
        x_bj ~ Uniform(0,1), j = 1,...,N_X_PER_COND

    Then flatten:
        x_flat:    (N_COND * N_X_PER_COND, 1)
        mu_flat:   (N_COND * N_X_PER_COND, 1)
        sigma_flat:(N_COND * N_X_PER_COND, 1)
        cond_flat: (N_COND * N_X_PER_COND, 2)
    """

    mu_cond = torch.empty(n_cond, 1, device=device).uniform_(*MU_RANGE)
    sigma_cond = torch.empty(n_cond, 1, device=device).uniform_(*SIGMA_RANGE)

    cond_base = torch.cat([mu_cond, sigma_cond], dim=1)  # (N_COND, 2)

    x = torch.rand(n_cond, n_x_per_cond, 1, device=device)

    mu = mu_cond[:, None, :].expand(n_cond, n_x_per_cond, 1)
    sigma = sigma_cond[:, None, :].expand(n_cond, n_x_per_cond, 1)
    cond = cond_base[:, None, :].expand(n_cond, n_x_per_cond, 2)

    x_flat = x.reshape(-1, 1)
    mu_flat = mu.reshape(-1, 1)
    sigma_flat = sigma.reshape(-1, 1)
    cond_flat = cond.reshape(-1, 2)

    return x_flat, mu_flat, sigma_flat, cond_flat


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


# ======================================================
# Integration evaluation
# ======================================================

def effective_sample_size(weights):
    s1 = weights.sum()
    s2 = (weights ** 2).sum()
    return float((s1 ** 2 / (s2 + 1e-12)).item())


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


# ======================================================
# Training
# ======================================================

def train():
    set_seed(0)

    model = ConditionalRQFlow1D(
        cond_dim=2,
        hidden=HIDDEN,
        n_bins=N_BINS,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []

    print("DEVICE:", DEVICE)
    print("Training conditional Gaussian 1D CNF")
    print("N_COND:", N_COND)
    print("N_X_PER_COND:", N_X_PER_COND)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("STEPS:", STEPS)
    print("N_BINS:", N_BINS)
    print("LR:", LR)

    for step in range(STEPS):
        # ======================================================
        # Grouped conditional batch
        # ======================================================
        # Sample N_COND different conditions:
        # c_b = (mu_b, sigma_b)
        mu_cond = torch.empty(N_COND, 1, device=DEVICE).uniform_(*MU_RANGE)
        sigma_cond = torch.empty(N_COND, 1, device=DEVICE).uniform_(*SIGMA_RANGE)

        cond_base = torch.cat([mu_cond, sigma_cond], dim=1)
        # shape: (N_COND, 2)

        # For each condition, sample N_X_PER_COND uniform x values
        x = torch.rand(N_COND, N_X_PER_COND, 1, device=DEVICE)

        # Broadcast each condition across its x samples
        mu = mu_cond[:, None, :].expand(N_COND, N_X_PER_COND, 1)
        sigma = sigma_cond[:, None, :].expand(N_COND, N_X_PER_COND, 1)
        cond = cond_base[:, None, :].expand(N_COND, N_X_PER_COND, 2)

        # Flatten for model evaluation
        x_flat = x.reshape(-1, 1)
        mu_flat = mu.reshape(-1, 1)
        sigma_flat = sigma.reshape(-1, 1)
        cond_flat = cond.reshape(-1, 2)

        # ======================================================
        # Evaluate f and log q
        # ======================================================
        f_vals = gaussian_1d(x_flat, mu_flat, sigma_flat)
        log_q = model.log_prob(x_flat, cond_flat)

        # Reshape back to grouped form:
        # rows = conditions
        # columns = x samples for that condition
        f_vals = f_vals.reshape(N_COND, N_X_PER_COND)
        log_q = log_q.reshape(N_COND, N_X_PER_COND)

        weights = f_vals.detach()

        # ======================================================
        # Grouped conditional KL-style loss
        # ======================================================
        # For each condition c_b:
        #
        # L_b =
        # - sum_j f(x_bj; c_b) log q(x_bj | c_b)
        #   / sum_j f(x_bj; c_b)
        #
        # Then average over conditions.
        loss_per_cond = -(
            weights * log_q
        ).sum(dim=1) / (weights.sum(dim=1) + 1e-12)

        main_loss = loss_per_cond.mean()
        reg = model.spline_regularization(cond_base)
        loss = main_loss + REG_LAMBDA * reg

        # ======================================================
        # Optimization
        # ======================================================
        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        opt.step()

        loss_value = float(loss.item())
        history.append(loss_value)

        if step % 200 == 0:
            print(
                f"step {step:6d} | "
                f"loss = {float(loss.item()):.6f} | "
                f"main = {float(main_loss.item()):.6f} | "
                f"reg = {float(reg.item()):.6f}"
            )

    ckpt_path = f"{CKPT_DIR}/conditional_gaussian_1d.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "config": {
                "n_bins": N_BINS,
                "hidden": HIDDEN,
                "mu_range": MU_RANGE,
                "sigma_range": SIGMA_RANGE,
                "n_cond": N_COND,
                "n_x_per_cond": N_X_PER_COND,
                "batch_size": BATCH_SIZE,
                "steps": STEPS,
                "lr": LR,
                "reg_lambda": REG_LAMBDA,
            },
        },
        ckpt_path,
    )

    print("Saved checkpoint:", ckpt_path)

    make_loss_plot(history)

    return model, history


# ======================================================
# Plots
# ======================================================

def make_loss_plot(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Conditional Gaussian 1D training loss")
    plt.grid(False)
    plt.tight_layout()

    path = f"{PLOT_DIR}/loss.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


def plot_condition(model, mu_value, sigma_value, n_grid=1200):
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

    name = condition_name(mu_value, sigma_value)

    # Linear density plot
    plt.figure(figsize=(7, 4))
    plt.plot(x_np, target_np, linewidth=2.0, label=r"Target density $f/I$")
    plt.plot(x_np, q_np, linewidth=1.6, label=r"CNF proposal $q_\theta(x|\mu,\sigma)$")
    plt.axvline(mu_value, linestyle="--", linewidth=1.0, label=r"$\mu$")
    plt.xlabel(r"$x$")
    plt.ylabel("Density")
    plt.title(fr"Conditional Gaussian: $\mu={mu_value:.2f}$, $\sigma={sigma_value:.2f}$")
    plt.grid(False)
    plt.legend(frameon=False)
    plt.tight_layout()

    path = f"{PLOT_DIR}/{name}_density.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

    # Log density plot
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

    plt.figure(figsize=(7, 4))
    plt.semilogy(x_np, target_np + 1e-12, linewidth=2.0, label=r"Target density $f/I$")
    plt.semilogy(x_np, q_np + 1e-12, linewidth=1.6, label=r"CNF proposal $q_\theta$")
    plt.axvline(mu_value, linestyle="--", linewidth=1.0, label=r"$\mu$")

    if ymin is not None and ymax is not None and ymax > ymin:
        plt.ylim(ymin, ymax)

    plt.xlabel(r"$x$")
    plt.ylabel("Density, log scale")
    plt.title(fr"Log density: $\mu={mu_value:.2f}$, $\sigma={sigma_value:.2f}$")
    plt.grid(False)
    plt.legend(frameon=False)
    plt.tight_layout()

    path = f"{PLOT_DIR}/{name}_density_log.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

    # Log-ratio plot
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
    plt.title(fr"Log-ratio: $\mu={mu_value:.2f}$, $\sigma={sigma_value:.2f}$")
    plt.grid(False)
    plt.tight_layout()

    path = f"{PLOT_DIR}/{name}_log_ratio.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


def make_contact_sheet():
    """
    Make one contact sheet from the density plots.
    Requires PIL.
    """
    try:
        from PIL import Image, ImageOps, ImageDraw
    except ImportError:
        print("PIL not installed; skipping contact sheet.")
        return

    image_paths = []

    for mu, sigma in TEST_CONDITIONS:
        name = condition_name(mu, sigma)
        image_paths.append(f"{PLOT_DIR}/{name}_density.png")

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

    path = f"{PLOT_DIR}/conditional_gaussian_contact_sheet.png"
    sheet.save(path)

    print("Saved:", path)


# ======================================================
# Main
# ======================================================

def main():
    model, history = train()

    summary_rows = []

    for mu_value, sigma_value in TEST_CONDITIONS:
        plot_condition(model, mu_value, sigma_value)

        I_ref = exact_integral_1d(mu_value, sigma_value)

        res = cnf_integrate(
            model=model,
            mu_value=mu_value,
            sigma_value=sigma_value,
            n_events=N_EVAL,
        )

        res = add_accuracy_metrics(res, I_ref)

        res.update(
            {
                "mu": mu_value,
                "sigma": sigma_value,
                "label": f"CNF mu={mu_value:.2f}, sigma={sigma_value:.2f}",
            }
        )

        summary_rows.append(res)

        print(
            f"mu={mu_value:.2f}, sigma={sigma_value:.2f} | "
            f"estimate={res['estimate']:.8e}, "
            f"I_ref={res['integral_ref']:.8e}, "
            f"variance={res['variance']:.3e}, "
            f"rel_err={res['rel_error']:.3e}, "
            f"pull={res['pull']:.2f}, "
            f"ESS={res['ess']:.3e}"
        )

    save_json(summary_rows, f"{OUT_DIR}/conditional_gaussian_1d_summary.json")

    make_contact_sheet()

    print("\nDone.")
    print("Results:", OUT_DIR)


if __name__ == "__main__":
    main()