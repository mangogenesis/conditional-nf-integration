#!/usr/bin/env python3
"""Compare exact-target and frozen-NF dependence for a 32D benchmark.

The exact autoregressive sampler is used only for post-training diagnostics.
It is never exposed to the training procedure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_geometry_ladder_32d as geometry  # noqa: E402


DIM = 32
COND_DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--condition", type=float, nargs=COND_DIM, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1_000:
        parser.error("--samples must be at least 1000")
    if args.bins < 8:
        parser.error("--bins must be at least 8")
    if any(value < 0.0 or value > 1.0 for value in args.condition):
        parser.error("condition values must lie in [0,1]")
    return args


def load_model(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    benchmark = checkpoint.get("benchmark")
    if benchmark not in {
        geometry.AUTOREGRESSIVE_BROAD,
        geometry.AUTOREGRESSIVE_SHARP,
    }:
        raise ValueError(
            "This dependence plot currently requires an autoregressive "
            "geometry-ladder checkpoint"
        )
    geometry.register_benchmarks()
    adaptive = geometry.adaptive
    adaptive.DIM = DIM
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return adaptive, model, checkpoint, benchmark


@torch.inference_mode()
def sample_exact_target(
    benchmark: str,
    condition: torch.Tensor,
    samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Ancestral sampler for the normalized autoregressive benchmark."""
    sigma = 0.025 if benchmark == geometry.AUTOREGRESSIVE_SHARP else 0.15
    cond = condition.expand(samples, COND_DIM)
    x = torch.zeros(samples, DIM, device=device, dtype=torch.float32)
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    )
    epsilon = torch.finfo(x.dtype).eps
    for child in range(DIM):
        mu, scale = geometry.component_mu_sigma(child, x, cond, sigma)
        lower = standard_normal.cdf((0.0 - mu) / scale)
        upper = standard_normal.cdf((1.0 - mu) / scale)
        probability = lower + torch.rand_like(lower) * (upper - lower)
        probability = probability.clamp(epsilon, 1.0 - epsilon)
        x[:, child:child + 1] = (
            mu + scale * standard_normal.icdf(probability)
        ).clamp(0.0, 1.0)
    return x


def normalized_mutual_information(x: np.ndarray, bins: int) -> np.ndarray:
    """Fixed-bin pairwise NMI, sensitive to nonlinear dependence."""
    dimension = x.shape[1]
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = np.eye(dimension, dtype=np.float64)
    for i in range(dimension):
        for j in range(i):
            counts, _, _ = np.histogram2d(
                x[:, i], x[:, j], bins=(edges, edges)
            )
            probability = counts / counts.sum()
            pi = probability.sum(axis=1)
            pj = probability.sum(axis=0)
            expected = pi[:, None] * pj[None, :]
            mask = probability > 0.0
            mutual_information = float(
                np.sum(probability[mask] * np.log(
                    probability[mask] / expected[mask]
                ))
            )
            hi = -float(np.sum(pi[pi > 0.0] * np.log(pi[pi > 0.0])))
            hj = -float(np.sum(pj[pj > 0.0] * np.log(pj[pj > 0.0])))
            denominator = math.sqrt(max(hi * hj, np.finfo(float).tiny))
            result[i, j] = result[j, i] = mutual_information / denominator
    return result


def strongest_true_edge(target_nmi: np.ndarray) -> tuple[int, int]:
    graph = geometry.ladder_graph(DIM)
    edges = [(parent, child) for child, parents in graph.items()
             for parent in parents]
    return max(edges, key=lambda edge: target_nmi[edge[0], edge[1]])


def conditional_profile(
    values: np.ndarray, parent: int, child: int, bins: int
):
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    assignment = np.clip(np.digitize(values[:, parent], edges) - 1, 0, bins - 1)
    mean = np.full(bins, np.nan)
    low = np.full(bins, np.nan)
    high = np.full(bins, np.nan)
    for index in range(bins):
        selected = values[assignment == index, child]
        if len(selected) >= 10:
            mean[index] = selected.mean()
            low[index], high[index] = np.quantile(selected, [0.16, 0.84])
    return centers, mean, low, high


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    adaptive, model, checkpoint, benchmark = load_model(args.checkpoint, device)
    condition = torch.tensor(args.condition, device=device, dtype=torch.float32)
    cond = condition.expand(args.samples, COND_DIM)
    target = sample_exact_target(benchmark, condition, args.samples, device)
    with torch.inference_mode():
        learned = adaptive.sample_from_model(model, args.samples, cond)
    target = target.cpu().numpy()
    learned = learned.cpu().numpy()

    target_nmi = normalized_mutual_information(target, args.bins)
    learned_nmi = normalized_mutual_information(learned, args.bins)
    difference = learned_nmi - target_nmi
    parent, child = strongest_true_edge(target_nmi)

    density_bins = np.linspace(0.0, 1.0, args.bins + 1)
    target_density, _, _ = np.histogram2d(
        target[:, parent], target[:, child], bins=(density_bins, density_bins)
    )
    learned_density, _, _ = np.histogram2d(
        learned[:, parent], learned[:, child], bins=(density_bins, density_bins)
    )
    positive = np.concatenate([
        target_density[target_density > 0.0],
        learned_density[learned_density > 0.0],
    ])
    density_norm = LogNorm(vmin=1.0, vmax=float(positive.max()))

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    for axis, density, title in (
        (axes[0, 0], target_density, "Exact target dependence"),
        (axes[0, 1], learned_density, "Frozen NF dependence"),
    ):
        image = axis.imshow(
            density.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="auto",
            cmap="viridis",
            norm=density_norm,
        )
        axis.set_title(title)
        axis.set_xlabel(fr"parent $x_{{{parent}}}$")
        axis.set_ylabel(fr"child $x_{{{child}}}$")
    fig.colorbar(image, ax=axes[0, :2], label="samples per bin")

    profile_bins = max(12, args.bins // 2)
    for values, label, color in (
        (target, "Exact target", "black"),
        (learned, "NF", "#0b84a5"),
    ):
        centers, mean, low, high = conditional_profile(
            values, parent, child, profile_bins
        )
        axes[0, 2].plot(centers, mean, color=color, lw=2.2, label=label)
        axes[0, 2].fill_between(
            centers, low, high, color=color, alpha=0.14, linewidth=0
        )
    axes[0, 2].set_title("Learned conditional trend")
    axes[0, 2].set_xlabel(fr"parent $x_{{{parent}}}$")
    axes[0, 2].set_ylabel(fr"$x_{{{child}}}$: mean and 16--84% band")
    axes[0, 2].legend(frameon=False)

    heatmaps = (
        (axes[1, 0], target_nmi, "Exact target: nonlinear dependence", "magma"),
        (axes[1, 1], learned_nmi, "Frozen NF: nonlinear dependence", "magma"),
    )
    for axis, matrix, title, cmap in heatmaps:
        shown = axis.imshow(matrix, origin="lower", vmin=0.0, vmax=1.0, cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("coordinate")
        axis.set_ylabel("coordinate")
    fig.colorbar(shown, ax=axes[1, :2], label="binned normalized mutual information")

    maximum_difference = float(np.max(np.abs(difference)))
    scale = max(maximum_difference, 1e-6)
    shown_difference = axes[1, 2].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
    )
    axes[1, 2].set_title("NF minus exact dependence")
    axes[1, 2].set_xlabel("coordinate")
    axes[1, 2].set_ylabel("coordinate")
    fig.colorbar(shown_difference, ax=axes[1, 2], label="NMI difference")

    off_diagonal = ~np.eye(DIM, dtype=bool)
    nmi_rmse = float(np.sqrt(np.mean(difference[off_diagonal] ** 2)))
    target_pair = target_density / target_density.sum()
    learned_pair = learned_density / learned_density.sum()
    pair_tv = 0.5 * float(np.abs(target_pair - learned_pair).sum())
    fig.suptitle(
        "32D autoregressive benchmark: exact versus learned dependence\n"
        f"condition={tuple(args.condition)}, samples={args.samples:,}, "
        f"NMI RMSE={nmi_rmse:.4f}, selected-pair TV={pair_tv:.4f}",
        fontsize=15,
    )

    png = args.output / "target_vs_nf_dependence.png"
    pdf = args.output / "target_vs_nf_dependence.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "benchmark": benchmark,
        "condition": list(args.condition),
        "samples_per_distribution": args.samples,
        "selected_parent": parent,
        "selected_child": child,
        "target_selected_edge_nmi": float(target_nmi[parent, child]),
        "nf_selected_edge_nmi": float(learned_nmi[parent, child]),
        "nmi_matrix_rmse_off_diagonal": nmi_rmse,
        "nmi_maximum_absolute_difference": maximum_difference,
        "selected_pair_total_variation": pair_tv,
        "exact_target_sampler_used_for_training": False,
        "checkpoint": str(args.checkpoint),
    }
    with (args.output / "dependence_metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nFigure: {png.resolve()}")


if __name__ == "__main__":
    main()
