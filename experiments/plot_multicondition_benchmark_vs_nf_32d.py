#!/usr/bin/env python3
"""Visual comparison of exact 32D benchmarks and frozen NF proposals.

Supports the factorized-sharp, autoregressive-sharp, and topology-ladder
benchmarks. Exact target samples are used only for this post-training audit.
"""

from __future__ import annotations

import argparse
import json
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

import plot_geometry_dependence_32d as geometry_plot  # noqa: E402
import plot_topology_target_vs_nf_32d as topology_plot  # noqa: E402
import run_geometry_ladder_32d as geometry  # noqa: E402
import run_topology_ladder_32d as topology  # noqa: E402


DIM = 32
COND_DIM = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--conditions", type=float, nargs="+",
        default=(
            0.50, 0.50, 0.50,
            0.60, 0.55, 0.45,
            0.40, 0.45, 0.55,
        ),
        help="Flat list c0 c1 c2 repeated for each displayed condition.",
    )
    parser.add_argument("--pair", type=int, nargs=2)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--top-pairs", type=int, default=4,
        help="Number of strongest exact-target pairs shown as 2D densities.",
    )
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.conditions) == 0 or len(args.conditions) % COND_DIM:
        parser.error("--conditions must contain groups of three numbers")
    if any(value < 0 or value > 1 for value in args.conditions):
        parser.error("conditions must lie in [0,1]^3")
    if args.samples < 1000:
        parser.error("--samples must be at least 1000")
    if args.top_pairs < 1:
        parser.error("--top-pairs must be positive")
    return args


def load_model(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    benchmark = checkpoint.get("benchmark")
    topology.register_benchmarks()
    adaptive = topology.adaptive
    adaptive.DIM = DIM
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return adaptive, model, benchmark


@torch.inference_mode()
def sample_factorized(condition, samples, sigma, device):
    cond = condition.expand(samples, COND_DIM)
    x = torch.empty(samples, DIM, device=device)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    )
    epsilon = torch.finfo(x.dtype).eps
    scale = torch.full((samples, 1), sigma, device=device)
    for child in range(DIM):
        mu = geometry.factorized_mu(child, cond)
        lower = normal.cdf((0.0 - mu) / scale)
        upper = normal.cdf((1.0 - mu) / scale)
        probability = lower + torch.rand_like(lower) * (upper - lower)
        x[:, child:child + 1] = (
            mu + scale * normal.icdf(probability.clamp(epsilon, 1.0 - epsilon))
        ).clamp(0.0, 1.0)
    return x


@torch.inference_mode()
def sample_exact(benchmark, condition, samples, device):
    if benchmark == geometry.FACTORIZED_SHARP:
        return sample_factorized(condition, samples, 0.025, device)
    if benchmark == geometry.FACTORIZED_BROAD:
        return sample_factorized(condition, samples, 0.15, device)
    if benchmark in {geometry.AUTOREGRESSIVE_SHARP, geometry.AUTOREGRESSIVE_BROAD}:
        return geometry_plot.sample_exact_target(
            benchmark, condition, samples, device
        )
    if benchmark in topology.BENCHMARKS:
        values, _ = topology_plot.sample_exact_target(
            benchmark, condition, samples, device
        )
        return values
    raise ValueError(f"unsupported comparison benchmark {benchmark!r}")


def default_pair(benchmark):
    if benchmark in topology.BENCHMARKS:
        return 0, 1
    if benchmark in {geometry.AUTOREGRESSIVE_SHARP, geometry.AUTOREGRESSIVE_BROAD}:
        return 25, 26
    return 0, 1


def histogram2d(values, pair, edges):
    result, _, _ = np.histogram2d(
        values[:, pair[0]], values[:, pair[1]], bins=(edges, edges)
    )
    return result


def save_pairwise_diagnostics(
    target, learned, condition, condition_index, args, benchmark
):
    """Save all-pair NMI matrices and strongest-pair density comparisons."""
    nmi_bins = min(args.bins, 32)
    target_nmi = geometry_plot.normalized_mutual_information(target, nmi_bins)
    learned_nmi = geometry_plot.normalized_mutual_information(learned, nmi_bins)
    difference = learned_nmi - target_nmi
    off_diagonal = ~np.eye(DIM, dtype=bool)
    nmi_rmse = float(np.sqrt(np.mean(difference[off_diagonal] ** 2)))

    nmi_figure, nmi_axes = plt.subplots(
        1, 3, figsize=(16.0, 5.0), constrained_layout=True
    )
    for axis, matrix, title in (
        (nmi_axes[0], target_nmi, "Exact benchmark"),
        (nmi_axes[1], learned_nmi, "Frozen NF proposal"),
    ):
        image = axis.imshow(
            matrix, origin="lower", vmin=0.0, vmax=1.0, cmap="magma"
        )
        axis.set_title(title)
        axis.set_xlabel("coordinate")
        axis.set_ylabel("coordinate")
    nmi_figure.colorbar(
        image, ax=nmi_axes[:2], label="normalized mutual information"
    )
    scale = max(float(np.abs(difference[off_diagonal]).max()), 1e-8)
    difference_image = nmi_axes[2].imshow(
        difference, origin="lower", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
    )
    nmi_axes[2].set_title(f"NF minus exact (RMSE={nmi_rmse:.4f})")
    nmi_axes[2].set_xlabel("coordinate")
    nmi_axes[2].set_ylabel("coordinate")
    nmi_figure.colorbar(difference_image, ax=nmi_axes[2], label="NMI difference")
    nmi_figure.suptitle(
        f"{benchmark}: all pairwise dependence\n"
        f"condition={tuple(float(x) for x in condition)}",
        fontsize=16,
    )
    nmi_path = args.output / f"pairwise_nmi_condition_{condition_index:02d}.png"
    nmi_figure.savefig(nmi_path, dpi=220, bbox_inches="tight")
    nmi_figure.savefig(nmi_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(nmi_figure)

    lower_i, lower_j = np.tril_indices(DIM, k=-1)
    ordering = np.argsort(target_nmi[lower_i, lower_j])[::-1]
    count = min(args.top_pairs, len(ordering))
    pairs = [
        (int(lower_i[position]), int(lower_j[position]))
        for position in ordering[:count]
    ]
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    pair_figure, pair_axes = plt.subplots(
        2, count, figsize=(4.1 * count, 7.5), squeeze=False,
        constrained_layout=True,
    )
    pair_rows = []
    for column, pair in enumerate(pairs):
        target_count = histogram2d(target, pair, edges)
        learned_count = histogram2d(learned, pair, edges)
        p = target_count / target_count.sum()
        q = learned_count / learned_count.sum()
        pair_tv = 0.5 * float(np.abs(p - q).sum())
        maximum = max(float(target_count.max()), float(learned_count.max()), 1.0)
        norm = LogNorm(vmin=1.0, vmax=maximum)
        for row, density, label in (
            (0, target_count, "Exact"), (1, learned_count, "NF")
        ):
            pair_image = pair_axes[row, column].imshow(
                density.T, origin="lower", extent=(0, 1, 0, 1),
                aspect="auto", cmap="viridis", norm=norm,
            )
            pair_axes[row, column].set_xlabel(fr"$x_{{{pair[0]}}}$")
            pair_axes[row, column].set_ylabel(fr"$x_{{{pair[1]}}}$")
            pair_axes[row, column].set_title(
                f"{label}: pair ({pair[0]}, {pair[1]})\n"
                f"target NMI={target_nmi[pair]:.3f}, TV={pair_tv:.3f}"
            )
        pair_rows.append({
            "condition_index": condition_index,
            "c0": float(condition[0]), "c1": float(condition[1]),
            "c2": float(condition[2]),
            "first_coordinate": pair[0], "second_coordinate": pair[1],
            "target_nmi": float(target_nmi[pair]),
            "nf_nmi": float(learned_nmi[pair]),
            "pair_total_variation": pair_tv,
            "all_pair_nmi_rmse": nmi_rmse,
        })
    pair_figure.colorbar(
        pair_image, ax=pair_axes, label="samples per bin", shrink=0.72
    )
    pair_figure.suptitle(
        f"{benchmark}: strongest exact-target coordinate pairs\n"
        f"condition={tuple(float(x) for x in condition)}",
        fontsize=16,
    )
    pair_path = args.output / f"strongest_pairs_condition_{condition_index:02d}.png"
    pair_figure.savefig(pair_path, dpi=220, bbox_inches="tight")
    pair_figure.savefig(pair_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(pair_figure)
    return pair_rows, nmi_rmse


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    adaptive, model, benchmark = load_model(args.checkpoint, device)
    conditions = np.asarray(args.conditions, dtype=np.float32).reshape(-1, 3)
    pair = tuple(args.pair) if args.pair is not None else default_pair(benchmark)
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = len(conditions)
    figure, axes = plt.subplots(
        rows, 4, figsize=(17.0, 4.3 * rows), squeeze=False,
        constrained_layout=True,
    )
    metrics = []
    pairwise_metrics = []

    for index, condition_np in enumerate(conditions):
        condition = torch.as_tensor(condition_np, device=device)
        cond = condition.expand(args.samples, COND_DIM)
        target = sample_exact(benchmark, condition, args.samples, device)
        with torch.inference_mode():
            learned = adaptive.sample_from_model(model, args.samples, cond)
        target = target.cpu().numpy()
        learned = learned.cpu().numpy()
        current_pairwise, nmi_rmse = save_pairwise_diagnostics(
            target, learned, condition_np, index, args, benchmark
        )
        pairwise_metrics.extend(current_pairwise)
        target_count = histogram2d(target, pair, edges)
        learned_count = histogram2d(learned, pair, edges)
        target_probability = target_count / target_count.sum()
        learned_probability = learned_count / learned_count.sum()
        difference = learned_probability - target_probability
        pair_tv = 0.5 * float(np.abs(difference).sum())

        positive = np.concatenate((
            target_count[target_count > 0], learned_count[learned_count > 0]
        ))
        density_norm = LogNorm(vmin=1.0, vmax=max(1.0, float(positive.max())))
        for axis, density, title in (
            (axes[index, 0], target_count, "Exact benchmark"),
            (axes[index, 1], learned_count, "Frozen NF proposal"),
        ):
            density_image = axis.imshow(
                density.T, origin="lower", extent=(0, 1, 0, 1),
                aspect="auto", cmap="viridis", norm=density_norm,
            )
            axis.set_title(title)
            axis.set_xlabel(fr"$x_{{{pair[0]}}}$")
            axis.set_ylabel(fr"$x_{{{pair[1]}}}$")

        difference_scale = max(float(np.abs(difference).max()), 1e-8)
        difference_image = axes[index, 2].imshow(
            difference.T, origin="lower", extent=(0, 1, 0, 1),
            aspect="auto", cmap="coolwarm",
            norm=TwoSlopeNorm(
                vmin=-difference_scale, vcenter=0.0, vmax=difference_scale
            ),
        )
        axes[index, 2].set_title(f"NF minus exact (pair TV={pair_tv:.4f})")
        axes[index, 2].set_xlabel(fr"$x_{{{pair[0]}}}$")
        axes[index, 2].set_ylabel(fr"$x_{{{pair[1]}}}$")

        target_marginal, _ = np.histogram(
            target[:, pair[0]], bins=edges, density=True
        )
        learned_marginal, _ = np.histogram(
            learned[:, pair[0]], bins=edges, density=True
        )
        marginal_p = target_marginal / max(target_marginal.sum(), 1e-30)
        marginal_q = learned_marginal / max(learned_marginal.sum(), 1e-30)
        marginal_tv = 0.5 * float(np.abs(marginal_p - marginal_q).sum())
        axes[index, 3].plot(
            centers, target_marginal, color="black", lw=2.2,
            label="Exact benchmark",
        )
        axes[index, 3].plot(
            centers, learned_marginal, color="#07879b", lw=2.0,
            label="NF proposal",
        )
        axes[index, 3].set_title(
            fr"Marginal $x_{{{pair[0]}}}$ (TV={marginal_tv:.4f})"
        )
        axes[index, 3].set_xlabel(fr"$x_{{{pair[0]}}}$")
        axes[index, 3].set_ylabel("density")
        axes[index, 3].legend(frameon=False)

        label = "condition = " + np.array2string(
            condition_np, precision=2, separator=", "
        )
        axes[index, 0].text(
            0.0, 1.14, label, transform=axes[index, 0].transAxes,
            fontsize=12, fontweight="bold", ha="left", va="bottom",
        )
        metrics.append({
            "condition_index": index,
            "c0": float(condition_np[0]),
            "c1": float(condition_np[1]),
            "c2": float(condition_np[2]),
            "pair_first": pair[0],
            "pair_second": pair[1],
            "pair_total_variation": pair_tv,
            "first_coordinate_marginal_tv": marginal_tv,
            "all_pair_nmi_rmse": nmi_rmse,
        })

    figure.colorbar(
        density_image, ax=axes[:, :2], label="samples per bin", shrink=0.75
    )
    figure.colorbar(
        difference_image, ax=axes[:, 2], label="probability-mass difference",
        shrink=0.75,
    )
    figure.suptitle(
        f"{benchmark}: exact target versus learned NF across conditions",
        fontsize=18,
    )
    png = args.output / "benchmark_vs_nf_multicondition.png"
    pdf = args.output / "benchmark_vs_nf_multicondition.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    frame = pd.DataFrame(metrics)
    frame.to_csv(args.output / "comparison_metrics.csv", index=False)
    pd.DataFrame(pairwise_metrics).to_csv(
        args.output / "pairwise_comparison_metrics.csv", index=False
    )
    summary = {
        "benchmark": benchmark,
        "checkpoint": str(args.checkpoint),
        "conditions": conditions.tolist(),
        "pair": list(pair),
        "samples_per_distribution_per_condition": args.samples,
        "mean_pair_total_variation": float(frame.pair_total_variation.mean()),
        "maximum_pair_total_variation": float(frame.pair_total_variation.max()),
        "exact_target_samples_used_for_training": False,
    }
    with (args.output / "comparison_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nFigure: {png.resolve()}")


if __name__ == "__main__":
    main()
