#!/usr/bin/env python3
"""Evaluation-only plots comparing a trained NF with the exact sparse_wave_32 law.

The exact sampler is used only here, after training.  It is never passed to the
training or integration algorithms.  The benchmark density is a normalized
autoregressive law: x0 and x1 are uniform and every later coordinate is a
truncated normal conditional on earlier coordinates.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr, ndtri

import run_adaptive_integrand_structure_search as adaptive
from run_final_checkpoint_integration_comparison import COND_DIM, DIM, load_nf
from run_vegas_sparse_wave_32_cpu import SparseWave32


def exact_benchmark_sample(condition: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Draw exact samples from the normalized sparse_wave_32 benchmark."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, DIM), dtype=np.float64)
    x[:, :2] = rng.random((n, 2))
    c0, c1, c2 = map(float, condition)
    gate = 1.0 / (1.0 + math.exp(-6.0 * (c0 - 0.5)))
    for child in range(2, DIM):
        parent = child - 1
        skip = max(0, child // 2 - 1)
        a, b = x[:, parent], x[:, skip]
        signal = (
            gate * np.sin(2.0 * math.pi * (a + 0.031 * child + 0.10 * c1))
            + (1.0 - gate) * np.cos(2.0 * math.pi * (b - 0.07 * c0))
            + 0.15 * np.sin(2.0 * math.pi * (a - b))
        )
        mu = 0.50 + 0.25 * np.tanh(1.35 * signal)
        sigma = 0.060 + 0.018 * c2 + 0.003 * float(child % 3)
        lo = ndtr((0.0 - mu) / sigma)
        hi = ndtr((1.0 - mu) / sigma)
        p = lo + rng.random(n) * (hi - lo)
        p = np.clip(p, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
        x[:, child] = np.clip(mu + sigma * ndtri(p), 0.0, 1.0)
    return x


@torch.no_grad()
def nf_sample(model, condition: np.ndarray, n: int, device: torch.device,
              chunk: int, seed: int):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    pieces, logq_parts = [], []
    remaining = n
    while remaining:
        count = min(chunk, remaining)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(count, COND_DIM)
        sample = adaptive.sample_from_model(model, count, cond)
        pieces.append(sample.cpu().numpy())
        logq_parts.append(model.log_prob(sample, cond).reshape(-1).double().cpu().numpy())
        remaining -= count
    return np.concatenate(pieces), np.concatenate(logq_parts)


@torch.no_grad()
def evaluate_logq(model, samples: np.ndarray, condition: np.ndarray,
                  device: torch.device, chunk: int) -> np.ndarray:
    parts = []
    for start in range(0, len(samples), chunk):
        x = torch.as_tensor(samples[start:start + chunk], dtype=torch.float32, device=device)
        cond = torch.as_tensor(condition, dtype=torch.float32, device=device)
        cond = cond.expand(len(x), COND_DIM)
        parts.append(model.log_prob(x, cond).reshape(-1).double().cpu().numpy())
    return np.concatenate(parts)


def plot_marginals(exact: np.ndarray, learned: np.ndarray, condition: np.ndarray,
                   output: Path, bins: int):
    fig, axes = plt.subplots(4, 8, figsize=(20, 10), sharex=True)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for dim, ax in enumerate(axes.flat):
        target, _ = np.histogram(exact[:, dim], edges, density=True)
        nf, _ = np.histogram(learned[:, dim], edges, density=True)
        ax.plot(centers, target, color="black", lw=1.6, label="exact benchmark")
        ax.plot(centers, nf, color="#1f77b4", lw=1.25, label="conditional NF")
        ax.set_title(f"$x_{{{dim}}}$", fontsize=9)
        ax.grid(alpha=0.15)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        f"One-dimensional marginals | c=({condition[0]:.3f}, "
        f"{condition[1]:.3f}, {condition[2]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_marginal_ratios(exact: np.ndarray, learned: np.ndarray,
                         condition: np.ndarray, output: Path, bins: int):
    fig, axes = plt.subplots(4, 8, figsize=(20, 10), sharex=True, sharey=True)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for dim, ax in enumerate(axes.flat):
        target, _ = np.histogram(exact[:, dim], edges)
        nf, _ = np.histogram(learned[:, dim], edges)
        # Jeffreys smoothing prevents empty tail bins from creating infinities.
        ratio = ((nf + 0.5) / (nf.sum() + 0.5 * bins)) / (
            (target + 0.5) / (target.sum() + 0.5 * bins)
        )
        ax.plot(centers, ratio, color="#7b3294", lw=1.2)
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.set_ylim(0.25, 4.0)
        ax.set_title(f"$x_{{{dim}}}$", fontsize=9)
        ax.grid(alpha=0.15)
    fig.suptitle(
        f"Marginal density ratio: NF / exact | c=({condition[0]:.3f}, "
        f"{condition[1]:.3f}, {condition[2]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_informative_pairs(exact: np.ndarray, learned: np.ndarray, count: int = 4):
    ce = np.corrcoef(exact, rowvar=False)
    cn = np.corrcoef(learned, rowvar=False)
    upper = np.triu_indices(DIM, 1)
    strong = np.argsort(np.abs(ce[upper]))[::-1]
    wrong = np.argsort(np.abs(cn[upper] - ce[upper]))[::-1]
    pairs = []
    for ordering in (wrong, strong):
        for k in ordering:
            pair = (int(upper[0][k]), int(upper[1][k]))
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) >= count:
                return pairs
    return pairs


def plot_dependence(exact: np.ndarray, learned: np.ndarray, condition: np.ndarray,
                    output: Path, max_scatter: int, seed: int):
    pairs = select_informative_pairs(exact, learned)
    rng = np.random.default_rng(seed)
    ie = rng.choice(len(exact), min(max_scatter, len(exact)), replace=False)
    il = rng.choice(len(learned), min(max_scatter, len(learned)), replace=False)
    fig, axes = plt.subplots(2, len(pairs), figsize=(16, 7), sharex=False, sharey=False)
    for col, (i, j) in enumerate(pairs):
        axes[0, col].hexbin(exact[ie, i], exact[ie, j], gridsize=45,
                            extent=(0, 1, 0, 1), mincnt=1, cmap="Greys")
        axes[1, col].hexbin(learned[il, i], learned[il, j], gridsize=45,
                            extent=(0, 1, 0, 1), mincnt=1, cmap="Blues")
        axes[0, col].set_title(f"$x_{{{i}}}$ vs $x_{{{j}}}$")
        axes[1, col].set_xlabel(f"$x_{{{i}}}$")
        axes[0, col].set_ylabel("exact benchmark" if col == 0 else f"$x_{{{j}}}$")
        axes[1, col].set_ylabel("conditional NF" if col == 0 else f"$x_{{{j}}}$")
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    fig.suptitle(
        f"Dependence comparison | c=({condition[0]:.3f}, "
        f"{condition[1]:.3f}, {condition[2]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_weight_diagnostics(exact: np.ndarray, learned: np.ndarray,
                            exact_logq: np.ndarray, learned_logq: np.ndarray,
                            condition: np.ndarray, output: Path):
    integrand = SparseWave32(condition)
    logw_exact = integrand.log_value(exact) - exact_logq
    logw_nf = integrand.log_value(learned) - learned_logq
    combined = np.concatenate([logw_exact, logw_nf])
    lo, hi = np.quantile(combined, [0.001, 0.9995])
    bins = np.linspace(lo, hi, 120)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(logw_nf, bins=bins, density=True, histtype="step", lw=1.5,
                 color="#1f77b4", label=r"$x\sim q_{NF}$")
    axes[0].hist(logw_exact, bins=bins, density=True, histtype="step", lw=1.5,
                 color="black", label=r"$x\sim \pi$ (exact)")
    axes[0].axvline(0.0, color="grey", ls="--", lw=1)
    axes[0].set_xlabel(r"$\log(\pi/q_{NF})$")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    for values, color, label in (
        (logw_nf, "#1f77b4", r"$x\sim q_{NF}$"),
        (logw_exact, "black", r"$x\sim \pi$ (exact)"),
    ):
        ordered = np.sort(values)
        survival = (len(ordered) - np.arange(len(ordered))) / len(ordered)
        axes[1].plot(ordered, survival, color=color, lw=1.35, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"threshold $t$ for $\log(\pi/q_{NF})>t$")
    axes[1].set_ylabel("survival probability")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        f"Importance-weight tails | c=({condition[0]:.3f}, "
        f"{condition[1]:.3f}, {condition[2]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return logw_exact


def plot_tail_anatomy(exact: np.ndarray, logw_exact: np.ndarray,
                      condition: np.ndarray, output: Path, top: int):
    top = min(top, len(exact))
    indices = np.argpartition(logw_exact, -top)[-top:]
    indices = indices[np.argsort(logw_exact[indices])[::-1]]
    mean = exact.mean(axis=0)
    scale = np.maximum(exact.std(axis=0), 1e-8)
    standardized = (exact[indices] - mean) / scale
    fig, ax = plt.subplots(figsize=(15, 6))
    limit = max(2.0, float(np.quantile(np.abs(standardized), 0.995)))
    image = ax.imshow(standardized, aspect="auto", cmap="coolwarm",
                      vmin=-limit, vmax=limit)
    ax.set_xlabel("integration dimension")
    ax.set_ylabel("exact samples ranked by largest $\log(\pi/q)$")
    ax.set_title(
        f"Anatomy of {top} most undercovered exact events | "
        f"c=({condition[0]:.3f}, {condition[1]:.3f}, {condition[2]:.3f})"
    )
    fig.colorbar(image, ax=ax, label="standardized coordinate")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_condition_landscape(results_file: Path, output: Path):
    if not results_file.exists():
        print(f"Condition landscape skipped: {results_file} does not exist", flush=True)
        return
    frame = pd.read_csv(results_file)
    frame = frame[frame["method"].isin(["nf_pure", "nf_mixture"])].copy()
    if frame.empty:
        print("Condition landscape skipped: no NF rows found", flush=True)
        return
    methods = list(dict.fromkeys(frame["method"]))
    fig, axes = plt.subplots(len(methods), 3, figsize=(15, 4.5 * len(methods)), squeeze=False)
    pairs = [("c0", "c1"), ("c0", "c2"), ("c1", "c2")]
    for row, method in enumerate(methods):
        data = frame[frame["method"] == method]
        colors = np.log10(np.maximum(data["normalized_ess"].to_numpy(), 1e-8))
        for col, (a, b) in enumerate(pairs):
            scatter = axes[row, col].scatter(data[a], data[b], c=colors,
                                             cmap="viridis", s=28)
            axes[row, col].set_xlabel(a)
            axes[row, col].set_ylabel(b)
            axes[row, col].set_title(f"{method}: log10 normalized ESS")
            fig.colorbar(scatter, ax=axes[row, col])
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(exact: np.ndarray, learned: np.ndarray, condition: np.ndarray,
                      output: Path):
    target_corr = np.corrcoef(exact, rowvar=False)
    nf_corr = np.corrcoef(learned, rowvar=False)
    difference = nf_corr - target_corr
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, matrix, title, cmap, limit in (
        (axes[0], target_corr, "Exact benchmark", "coolwarm", 1.0),
        (axes[1], nf_corr, "Conditional NF", "coolwarm", 1.0),
        (axes[2], difference, "NF minus exact", "PuOr", max(0.05, np.max(np.abs(difference)))),
    ):
        image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("dimension")
        ax.set_ylabel("dimension")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(
        f"Linear correlation | c=({condition[0]:.3f}, "
        f"{condition[1]:.3f}, {condition[2]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("results/adaptive_integrand_all_weak/sparse_wave_32_final_model.pt"),
    )
    parser.add_argument(
        "--conditions-file", type=Path,
        default=Path("results/final_checkpoint_comparison/conditions.csv"),
    )
    parser.add_argument("--condition-indices", type=int, nargs="+", default=[0, 80, 128])
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--gpu-chunk", type=int, default=16_384)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--max-scatter", type=int, default=100_000)
    parser.add_argument("--tail-events", type=int, default=50)
    parser.add_argument(
        "--results-file", type=Path,
        default=Path("results/final_checkpoint_comparison/condition_results.csv"),
    )
    parser.add_argument("--seed", type=int, default=91027)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path,
                        default=Path("results/nf_vs_exact_benchmark_plots"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.conditions_file)
    missing = {"c0", "c1", "c2"} - set(frame.columns)
    if missing:
        raise ValueError(f"conditions file is missing columns: {sorted(missing)}")
    if min(args.condition_indices) < 0 or max(args.condition_indices) >= len(frame):
        raise IndexError(f"condition indices must be between 0 and {len(frame) - 1}")

    device = torch.device(args.device)
    model, _, checkpoint = load_nf(args.checkpoint, device)
    print(
        f"Loaded conditional NF with {sum(len(v) for v in checkpoint['graph'].values())} edges.\n"
        "Exact target samples are used for post-training visualization only.",
        flush=True,
    )
    for position, index in enumerate(args.condition_indices):
        condition = frame.loc[index, ["c0", "c1", "c2"]].to_numpy(dtype=np.float64)
        exact = exact_benchmark_sample(condition, args.samples, args.seed + 2 * position)
        learned, learned_logq = nf_sample(
            model, condition, args.samples, device, args.gpu_chunk,
            args.seed + 2 * position + 1,
        )
        exact_logq = evaluate_logq(model, exact, condition, device, args.gpu_chunk)
        prefix = args.output / f"condition_{index:04d}"
        plot_marginals(exact, learned, condition, Path(f"{prefix}_marginals.png"), args.bins)
        plot_marginal_ratios(
            exact, learned, condition, Path(f"{prefix}_marginal_ratios.png"), args.bins
        )
        plot_dependence(
            exact, learned, condition, Path(f"{prefix}_dependence.png"),
            args.max_scatter, args.seed + position,
        )
        plot_correlations(exact, learned, condition, Path(f"{prefix}_correlations.png"))
        logw_exact = plot_weight_diagnostics(
            exact, learned, exact_logq, learned_logq, condition,
            Path(f"{prefix}_weight_tails.png"),
        )
        plot_tail_anatomy(
            exact, logw_exact, condition, Path(f"{prefix}_tail_anatomy.png"),
            args.tail_events,
        )
        corr_error = float(np.sqrt(np.mean(
            (np.corrcoef(learned, rowvar=False) - np.corrcoef(exact, rowvar=False)) ** 2
        )))
        print(f"condition {index}: correlation RMSE={corr_error:.5f}", flush=True)
    plot_condition_landscape(
        args.results_file, args.output / "conditional_ess_landscape.png"
    )
    print(f"Plots: {args.output.resolve()}")


if __name__ == "__main__":
    main()
