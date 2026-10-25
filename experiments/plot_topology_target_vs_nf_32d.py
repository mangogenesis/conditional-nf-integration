#!/usr/bin/env python3
"""Plot an exact 32D topology benchmark against a frozen NF proposal.

The exact target sampler is diagnostic only and is never exposed to training.
The figure focuses on disconnected-support coverage: branch shape, branch mass,
proposal leakage into the zero-density gap, and one-dimensional marginal error.
"""

from __future__ import annotations

import argparse
import json
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

import run_topology_ladder_32d as topology  # noqa: E402


COND_DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--condition", type=float, nargs=COND_DIM, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1_000:
        parser.error("--samples must be at least 1000")
    if args.bins < 12:
        parser.error("--bins must be at least 12")
    if any(value < 0.0 or value > 1.0 for value in args.condition):
        parser.error("condition values must lie in [0,1]")
    return args


def load_model(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    benchmark = checkpoint.get("benchmark")
    if benchmark not in topology.BENCHMARKS:
        raise ValueError(f"{benchmark!r} is not a topology-ladder benchmark")
    topology.register_benchmarks()
    adaptive = topology.adaptive
    adaptive.DIM = topology.DIMENSION
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return adaptive, model, benchmark


@torch.inference_mode()
def sample_reference_target(
    condition: torch.Tensor,
    samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Ancestral sample from the normalized reference autoregressive target."""
    cond = condition.expand(samples, COND_DIM)
    x = torch.zeros(
        samples, topology.DIMENSION, device=device, dtype=torch.float32
    )
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    )
    epsilon = torch.finfo(x.dtype).eps
    for child in range(topology.DIMENSION):
        mu, scale = topology.geometry.component_mu_sigma(
            child, x, cond, topology.REFERENCE_SIGMA
        )
        lower = normal.cdf((0.0 - mu) / scale)
        upper = normal.cdf((1.0 - mu) / scale)
        probability = lower + torch.rand_like(lower) * (upper - lower)
        probability = probability.clamp(epsilon, 1.0 - epsilon)
        x[:, child:child + 1] = (
            mu + scale * normal.icdf(probability)
        ).clamp(0.0, 1.0)
    return x


@torch.inference_mode()
def sample_exact_target(
    benchmark: str,
    condition: torch.Tensor,
    samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = sample_reference_target(condition, samples, device)
    cond = condition.expand(samples, COND_DIM)
    p1 = topology.branch_probabilities(benchmark, cond)[:, 1]
    branch = (torch.rand(samples, device=device) < p1).long()
    lower = branch.to(reference.dtype) * (1.0 - topology.BRANCH_WIDTH)
    reference[:, 0] = lower + topology.BRANCH_WIDTH * reference[:, 0]
    return reference, branch


def marginal_total_variation(
    target: np.ndarray, learned: np.ndarray, bins: int
) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = []
    for coordinate in range(target.shape[1]):
        target_count, _ = np.histogram(target[:, coordinate], bins=edges)
        learned_count, _ = np.histogram(learned[:, coordinate], bins=edges)
        p = target_count / target_count.sum()
        q = learned_count / learned_count.sum()
        values.append(0.5 * np.abs(p - q).sum())
    return np.asarray(values)


def normalized_mutual_information(values: np.ndarray, bins: int) -> np.ndarray:
    """Pairwise fixed-bin NMI, including nonlinear dependence."""
    dimension = values.shape[1]
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = np.eye(dimension, dtype=np.float64)
    for first in range(dimension):
        for second in range(first):
            counts, _, _ = np.histogram2d(
                values[:, first], values[:, second], bins=(edges, edges)
            )
            probability = counts / counts.sum()
            marginal_first = probability.sum(axis=1)
            marginal_second = probability.sum(axis=0)
            independent = marginal_first[:, None] * marginal_second[None, :]
            mask = probability > 0.0
            mutual_information = float(np.sum(
                probability[mask]
                * np.log(probability[mask] / independent[mask])
            ))
            entropy_first = -float(np.sum(
                marginal_first[marginal_first > 0.0]
                * np.log(marginal_first[marginal_first > 0.0])
            ))
            entropy_second = -float(np.sum(
                marginal_second[marginal_second > 0.0]
                * np.log(marginal_second[marginal_second > 0.0])
            ))
            denominator = np.sqrt(max(
                entropy_first * entropy_second, np.finfo(float).tiny
            ))
            result[first, second] = result[second, first] = (
                mutual_information / denominator
            )
    return result


def region_masses(values: np.ndarray) -> np.ndarray:
    x0 = values[:, 0]
    return np.asarray([
        np.mean(x0 <= topology.BRANCH_WIDTH),
        np.mean(
            (x0 > topology.BRANCH_WIDTH)
            & (x0 < 1.0 - topology.BRANCH_WIDTH)
        ),
        np.mean(x0 >= 1.0 - topology.BRANCH_WIDTH),
    ])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    adaptive, model, benchmark = load_model(args.checkpoint, device)
    condition = torch.tensor(args.condition, device=device, dtype=torch.float32)
    cond = condition.expand(args.samples, COND_DIM)
    target, _ = sample_exact_target(benchmark, condition, args.samples, device)
    with torch.inference_mode():
        learned = adaptive.sample_from_model(model, args.samples, cond)
    target = target.cpu().numpy()
    learned = learned.cpu().numpy()

    target_mass = region_masses(target)
    learned_mass = region_masses(learned)
    marginal_tv = marginal_total_variation(target, learned, args.bins)
    dependence_bins = min(args.bins, 32)
    target_nmi = normalized_mutual_information(target, dependence_bins)
    learned_nmi = normalized_mutual_information(learned, dependence_bins)
    nmi_difference = learned_nmi - target_nmi
    off_diagonal = ~np.eye(topology.DIMENSION, dtype=bool)
    nmi_rmse = float(np.sqrt(np.mean(nmi_difference[off_diagonal] ** 2)))
    nmi_maximum_error = float(np.max(np.abs(
        nmi_difference[off_diagonal]
    )))
    p1_exact = float(topology.branch_probabilities(
        benchmark, condition.reshape(1, COND_DIM)
    )[0, 1])

    edges = np.linspace(0.0, 1.0, args.bins + 1)
    target_2d, _, _ = np.histogram2d(
        target[:, 0], target[:, 1], bins=(edges, edges)
    )
    learned_2d, _, _ = np.histogram2d(
        learned[:, 0], learned[:, 1], bins=(edges, edges)
    )
    maximum_count = max(float(target_2d.max()), float(learned_2d.max()))
    density_norm = LogNorm(vmin=1.0, vmax=max(1.0, maximum_count))

    dark = "#17233c"
    target_color = "#222222"
    nf_color = "#07879b"
    gap_color = "#d95f59"
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)

    for axis, density, title in (
        (axes[0, 0], target_2d, "Exact benchmark"),
        (axes[0, 1], learned_2d, "Frozen NF proposal"),
    ):
        image = axis.imshow(
            density.T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="auto",
            cmap="viridis",
            norm=density_norm,
        )
        axis.axvspan(
            topology.BRANCH_WIDTH,
            1.0 - topology.BRANCH_WIDTH,
            color=gap_color,
            alpha=0.14,
            label="zero-density gap",
        )
        axis.set_title(title, color=dark)
        axis.set_xlabel(r"topology coordinate $x_0$")
        axis.set_ylabel(r"geometry coordinate $x_1$")
    axes[0, 0].legend(frameon=False, loc="upper center")
    fig.colorbar(image, ax=axes[0, :2], label="samples per bin")

    centers = 0.5 * (edges[:-1] + edges[1:])
    target_hist, _ = np.histogram(target[:, 0], bins=edges, density=True)
    learned_hist, _ = np.histogram(learned[:, 0], bins=edges, density=True)
    axes[0, 2].plot(
        centers, target_hist, color=target_color, lw=2.4, label="Exact benchmark"
    )
    axes[0, 2].plot(
        centers, learned_hist, color=nf_color, lw=2.2, label="NF proposal"
    )
    axes[0, 2].axvspan(
        topology.BRANCH_WIDTH,
        1.0 - topology.BRANCH_WIDTH,
        color=gap_color,
        alpha=0.14,
    )
    axes[0, 2].set_title(r"Marginal topology in $x_0$", color=dark)
    axes[0, 2].set_xlabel(r"$x_0$")
    axes[0, 2].set_ylabel("density")
    axes[0, 2].legend(frameon=False)

    positions = np.arange(3)
    width = 0.36
    axes[1, 0].bar(
        positions - width / 2,
        target_mass,
        width,
        color=target_color,
        label="Exact benchmark",
    )
    axes[1, 0].bar(
        positions + width / 2,
        learned_mass,
        width,
        color=nf_color,
        label="NF proposal",
    )
    axes[1, 0].set_xticks(positions, ("branch 0", "gap", "branch 1"))
    axes[1, 0].set_ylabel("probability mass")
    axes[1, 0].set_title("Disconnected-support coverage", color=dark)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(
        np.arange(topology.DIMENSION), marginal_tv, color=nf_color, width=0.82
    )
    axes[1, 1].set_xlabel("coordinate")
    axes[1, 1].set_ylabel("marginal total-variation distance")
    axes[1, 1].set_title("Coordinate-wise marginal mismatch", color=dark)

    axes[1, 2].axis("off")
    branch_error = abs(learned_mass[2] - p1_exact)
    diagnostic = (
        f"Benchmark: {benchmark}\n"
        f"condition = ({args.condition[0]:.3f}, {args.condition[1]:.3f}, "
        f"{args.condition[2]:.3f})\n\n"
        f"Exact branch-1 mass:  {p1_exact:.4f}\n"
        f"NF branch-1 mass:     {learned_mass[2]:.4f}\n"
        f"absolute mass error:  {branch_error:.4f}\n"
        f"NF gap mass:          {learned_mass[1]:.4f}\n\n"
        f"mean marginal TV:     {marginal_tv.mean():.4f}\n"
        f"maximum marginal TV:  {marginal_tv.max():.4f}\n"
        f"samples/model:        {args.samples:,}"
    )
    axes[1, 2].text(
        0.04,
        0.96,
        diagnostic,
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=12.5,
        family="monospace",
        color=dark,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#f2f5f8", "edgecolor": "none"},
    )

    fig.suptitle(
        "32D disconnected-topology benchmark: exact target versus learned NF proposal",
        fontsize=18,
        color=dark,
    )
    png = args.output / "topology_target_vs_nf.png"
    pdf = args.output / "topology_target_vs_nf.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    dependence_figure, dependence_axes = plt.subplots(
        1, 3, figsize=(16.2, 5.0), constrained_layout=True
    )
    for axis, matrix, title in (
        (dependence_axes[0], target_nmi, "Exact benchmark dependence"),
        (dependence_axes[1], learned_nmi, "Frozen NF dependence"),
    ):
        nmi_image = axis.imshow(
            matrix, origin="lower", vmin=0.0, vmax=1.0, cmap="magma"
        )
        axis.set_title(title, color=dark)
        axis.set_xlabel("coordinate")
        axis.set_ylabel("coordinate")
    dependence_figure.colorbar(
        nmi_image,
        ax=dependence_axes[:2],
        label="normalized mutual information",
        fraction=0.046,
    )
    maximum_difference = max(
        float(np.max(np.abs(nmi_difference[off_diagonal]))), 1e-6
    )
    difference_image = dependence_axes[2].imshow(
        nmi_difference,
        origin="lower",
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-maximum_difference,
            vcenter=0.0,
            vmax=maximum_difference,
        ),
    )
    dependence_axes[2].set_title(
        f"NF minus exact (off-diagonal RMSE={nmi_rmse:.4f})",
        color=dark,
    )
    dependence_axes[2].set_xlabel("coordinate")
    dependence_axes[2].set_ylabel("coordinate")
    dependence_figure.colorbar(
        difference_image,
        ax=dependence_axes[2],
        label="NMI difference",
        fraction=0.046,
    )
    dependence_figure.suptitle(
        "Pairwise nonlinear dependence across all 32 dimensions",
        fontsize=17,
        color=dark,
    )
    dependence_png = args.output / "topology_pairwise_dependence.png"
    dependence_pdf = args.output / "topology_pairwise_dependence.pdf"
    dependence_figure.savefig(dependence_png, dpi=220, bbox_inches="tight")
    dependence_figure.savefig(dependence_pdf, bbox_inches="tight")
    plt.close(dependence_figure)

    summary = {
        "benchmark": benchmark,
        "condition": list(map(float, args.condition)),
        "samples_per_distribution": args.samples,
        "exact_branch_1_mass": p1_exact,
        "nf_branch_0_mass": float(learned_mass[0]),
        "nf_gap_mass": float(learned_mass[1]),
        "nf_branch_1_mass": float(learned_mass[2]),
        "absolute_branch_1_mass_error": float(branch_error),
        "mean_marginal_total_variation": float(marginal_tv.mean()),
        "maximum_marginal_total_variation": float(marginal_tv.max()),
        "pairwise_nmi_bins": dependence_bins,
        "off_diagonal_nmi_rmse": nmi_rmse,
        "maximum_off_diagonal_nmi_error": nmi_maximum_error,
    }
    with (args.output / "topology_target_vs_nf.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nPlot: {png.resolve()}")
    print(f"Dependence plot: {dependence_png.resolve()}")


if __name__ == "__main__":
    main()
