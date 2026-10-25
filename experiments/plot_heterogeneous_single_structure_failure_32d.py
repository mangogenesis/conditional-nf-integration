#!/usr/bin/env python3
"""Visualize how an old NF misses structures in the heterogeneous benchmark.

This is a diagnostic-only script.  It uses the exact benchmark sampler to
compare a frozen proposal with the four target structures at one condition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_heterogeneous_topology_32d as audit  # noqa: E402
import run_heterogeneous_topology_32d as benchmark  # noqa: E402


DISPLAY_NAMES = ("Sharp peak", "Curved ridge", "Curved sheet", "Ring")
STRUCTURE_PAIRS = ((1, 2), (1, 10), (1, 3), (1, 2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 10_000 or args.bins < 10 or args.chunk <= 0:
        parser.error("increase the sampling/plotting budget")
    if any(value < 0 or value > 1 for value in args.fixed_condition):
        parser.error("condition must lie in [0,1]^3")
    return args


def histogram2d(values, pair, edges):
    counts, _, _ = np.histogram2d(
        values[:, pair[0]], values[:, pair[1]], bins=(edges, edges)
    )
    return counts


def density_image(axis, counts, title, pair, norm):
    image = axis.imshow(
        counts.T, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
        cmap="viridis", norm=norm,
    )
    axis.set_title(title)
    axis.set_xlabel(fr"$x_{{{pair[0]}}}$")
    axis.set_ylabel(fr"$x_{{{pair[1]}}}$")
    return image


@torch.inference_mode()
def sample_proposal(adaptive, model, condition, count, chunk):
    blocks = []
    remaining = count
    while remaining:
        current = min(remaining, chunk)
        cond = condition.reshape(1, -1).expand(current, -1)
        blocks.append(adaptive.sample_from_model(model, current, cond).cpu())
        remaining -= current
    return torch.cat(blocks, dim=0)


@torch.inference_mode()
def exact_log_diagnostics(model, x, condition, chunk):
    logq_blocks = []
    logf_blocks = []
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk]
        cond = condition.reshape(1, -1).expand(len(block), -1)
        logq_blocks.append(model.log_prob(block, cond).reshape(-1).cpu())
        logf_blocks.append(
            benchmark.heterogeneous_log_integrand(block, cond).cpu()
        )
    logq = torch.cat(logq_blocks).double().numpy()
    logf = torch.cat(logf_blocks).double().numpy()
    return logq, logf - logq


def save_overview(target, proposal, target_labels, proposal_labels, args):
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pair = (0, 1)
    target_count = histogram2d(target, pair, edges)
    proposal_count = histogram2d(proposal, pair, edges)
    positive = np.concatenate((
        target_count[target_count > 0], proposal_count[proposal_count > 0]
    ))
    norm = LogNorm(vmin=1.0, vmax=max(1.0, float(positive.max())))
    target_probability = target_count / target_count.sum()
    proposal_probability = proposal_count / proposal_count.sum()
    difference = proposal_probability - target_probability
    difference_scale = max(float(np.abs(difference).max()), 1e-12)

    figure, axes = plt.subplots(
        1, 5, figsize=(21.5, 4.2), constrained_layout=True
    )
    density_image(axes[0], target_count, "Exact target", pair, norm)
    density = density_image(
        axes[1], proposal_count, "Old NF proposal", pair, norm
    )
    residual = axes[2].imshow(
        difference.T, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-difference_scale, vcenter=0.0, vmax=difference_scale
        ),
    )
    axes[2].set_title("NF minus target")
    axes[2].set_xlabel(r"$x_0$")
    axes[2].set_ylabel(r"$x_1$")

    target_x0, _ = np.histogram(target[:, 0], bins=edges, density=True)
    proposal_x0, _ = np.histogram(proposal[:, 0], bins=edges, density=True)
    axes[3].plot(centers, target_x0, color="black", lw=2.2, label="Target")
    axes[3].plot(
        centers, proposal_x0, color="#07879b", lw=2.0, label="Old NF"
    )
    axes[3].set_title(r"Branch coordinate $x_0$")
    axes[3].set_xlabel(r"$x_0$")
    axes[3].set_ylabel("density")
    axes[3].legend(frameon=False)

    locations = np.arange(benchmark.COMPONENTS)
    target_mass = np.array([
        np.mean(target_labels == index) for index in locations
    ])
    proposal_mass = np.array([
        np.mean(proposal_labels == index) for index in locations
    ])
    width = 0.38
    axes[4].bar(
        locations - width / 2, target_mass, width, color="black",
        alpha=0.75, label="Target",
    )
    axes[4].bar(
        locations + width / 2, proposal_mass, width, color="#07879b",
        alpha=0.85, label="Old NF",
    )
    axes[4].set_xticks(locations, ("Peak", "Ridge", "Sheet", "Ring"))
    axes[4].tick_params(axis="x", rotation=25)
    axes[4].set_ylim(0, max(0.3, 1.08 * max(target_mass.max(), proposal_mass.max())))
    axes[4].set_ylabel("probability mass")
    axes[4].set_title("Mass assigned to each structure")
    axes[4].legend(frameon=False)

    figure.colorbar(density, ax=axes[:2], label="samples per bin", shrink=0.78)
    figure.colorbar(
        residual, ax=axes[2], label="probability-mass difference", shrink=0.78
    )
    figure.suptitle(
        "Heterogeneous 32D benchmark: the old NF covers only part of the target",
        fontsize=16,
    )
    path = args.output / "old_nf_vs_target_overview.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return target_mass, proposal_mass


def save_combined_x1_x2_comparison(target, proposal, args):
    """Compare the full target and proposal after marginalizing over x0."""
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    pair = (1, 2)
    target_count = histogram2d(target, pair, edges)
    proposal_count = histogram2d(proposal, pair, edges)
    positive = np.concatenate((
        target_count[target_count > 0], proposal_count[proposal_count > 0]
    ))
    maximum = float(positive.max()) if len(positive) else 1.0
    norm = LogNorm(vmin=1.0, vmax=max(1.0, maximum))

    target_probability = target_count / max(target_count.sum(), 1.0)
    proposal_probability = proposal_count / max(proposal_count.sum(), 1.0)
    difference = proposal_probability - target_probability
    difference_scale = max(float(np.abs(difference).max()), 1e-12)

    figure, axes = plt.subplots(
        1, 3, figsize=(14.6, 4.5), constrained_layout=True
    )
    density_image(
        axes[0], target_count, "Exact target: all structures", pair, norm
    )
    density = density_image(
        axes[1], proposal_count, "Frozen NF proposal", pair, norm
    )
    residual = axes[2].imshow(
        difference.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-difference_scale, vcenter=0.0, vmax=difference_scale
        ),
    )
    axes[2].set_title("NF minus target")
    axes[2].set_xlabel(r"$x_1$")
    axes[2].set_ylabel(r"$x_2$")

    figure.colorbar(
        density, ax=axes[:2], label="samples per bin", shrink=0.82
    )
    figure.colorbar(
        residual,
        ax=axes[2],
        label="probability-mass difference per bin",
        shrink=0.82,
    )
    figure.suptitle(
        r"Full benchmark projected onto $(x_1,x_2)$: target versus NF",
        fontsize=16,
    )
    path = args.output / "combined_x1_x2_target_vs_nf.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def save_structure_panels(target, proposal, target_labels, proposal_labels, args):
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    figure, axes = plt.subplots(
        2, benchmark.COMPONENTS, figsize=(17.2, 7.5),
        constrained_layout=True,
    )
    last_image = None
    for component, (name, pair) in enumerate(
        zip(DISPLAY_NAMES, STRUCTURE_PAIRS)
    ):
        exact = target[target_labels == component]
        learned = proposal[proposal_labels == component]
        exact_count = histogram2d(exact, pair, edges)
        learned_count = histogram2d(learned, pair, edges)
        positive = np.concatenate((
            exact_count[exact_count > 0], learned_count[learned_count > 0]
        ))
        maximum = float(positive.max()) if len(positive) else 1.0
        norm = LogNorm(vmin=1.0, vmax=max(1.0, maximum))
        last_image = density_image(
            axes[0, component], exact_count,
            f"{name}: exact (N={len(exact):,})", pair, norm,
        )
        density_image(
            axes[1, component], learned_count,
            f"{name}: old NF (N={len(learned):,})", pair, norm,
        )
    if last_image is not None:
        figure.colorbar(
            last_image, ax=axes, label="samples per bin", shrink=0.76
        )
    figure.suptitle(
        "Structure-by-structure geometry: exact target versus old NF",
        fontsize=16,
    )
    path = args.output / "old_nf_structure_by_structure.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def save_density_ratio_diagnostics(logq, logw, labels, args):
    rng = np.random.default_rng(args.seed + 7)
    q_values = []
    w_values = []
    for component in range(benchmark.COMPONENTS):
        indices = np.flatnonzero(labels == component)
        if len(indices) > 20_000:
            indices = rng.choice(indices, 20_000, replace=False)
        q_values.append(logq[indices][np.isfinite(logq[indices])])
        w_values.append(logw[indices][np.isfinite(logw[indices])])

    figure, axes = plt.subplots(
        1, 2, figsize=(12.8, 4.8), constrained_layout=True
    )
    for axis, values, ylabel, title in (
        (
            axes[0], q_values, r"$log q_\phi(x)$",
            "NF density evaluated on exact target samples",
        ),
        (
            axes[1], w_values, r"$log[f(x)/q_\phi(x)]$",
            "Importance-weight severity",
        ),
    ):
        axis.boxplot(values, showfliers=False, labels=DISPLAY_NAMES)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    path = args.output / "old_nf_density_ratio_by_structure.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    adaptive, model = audit.load_model(args.checkpoint, device)
    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    )
    target, target_labels = benchmark.sample_exact_target(
        args.samples, condition
    )
    proposal = sample_proposal(
        adaptive, model, condition, args.samples, args.chunk
    )
    proposal_device = proposal.to(device)
    proposal_cond = condition.reshape(1, -1).expand(args.samples, -1)
    proposal_labels = benchmark.structure_labels(
        proposal_device, proposal_cond
    ).cpu()
    logq, logw = exact_log_diagnostics(
        model, target, condition, args.chunk
    )

    target_np = target.cpu().numpy()
    proposal_np = proposal.numpy()
    target_labels_np = target_labels.cpu().numpy()
    proposal_labels_np = proposal_labels.numpy()
    target_mass, proposal_mass = save_overview(
        target_np, proposal_np, target_labels_np, proposal_labels_np, args
    )
    save_combined_x1_x2_comparison(target_np, proposal_np, args)
    save_structure_panels(
        target_np, proposal_np, target_labels_np, proposal_labels_np, args
    )
    save_density_ratio_diagnostics(logq, logw, target_labels_np, args)

    rows = []
    for component, key in enumerate(benchmark.STRUCTURE_NAMES):
        mask = target_labels_np == component
        finite_q = logq[mask & np.isfinite(logq)]
        finite_w = logw[mask & np.isfinite(logw)]
        rows.append({
            "structure": key,
            "target_sample_mass": float(target_mass[component]),
            "proposal_sample_mass": float(proposal_mass[component]),
            "median_log_q_on_target": (
                float(np.median(finite_q)) if len(finite_q) else -math.inf
            ),
            "median_log_weight_on_target": (
                float(np.median(finite_w)) if len(finite_w) else math.inf
            ),
            "q99_log_weight_on_target": (
                float(np.quantile(finite_w, 0.99)) if len(finite_w) else math.inf
            ),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "structure_comparison.csv", index=False)
    summary = {
        "benchmark": benchmark.BENCHMARK,
        "checkpoint": str(args.checkpoint),
        "condition": list(args.fixed_condition),
        "samples_per_distribution": args.samples,
        "figures": [
            "old_nf_vs_target_overview.png",
            "combined_x1_x2_target_vs_nf.png",
            "old_nf_structure_by_structure.png",
            "old_nf_density_ratio_by_structure.png",
        ],
        "structures": rows,
        "exact_target_samples_used_for_training": False,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nSTRUCTURE COMPARISON")
    print(frame.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
