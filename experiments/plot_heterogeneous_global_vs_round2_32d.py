#!/usr/bin/env python3
"""Compare the exact four-structure target, global NF, and Round-2 repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_heterogeneous_discovered_region_mixture_32d as mixture_audit  # noqa: E402
import audit_heterogeneous_topology_32d as global_audit  # noqa: E402
import run_heterogeneous_topology_32d as benchmark  # noqa: E402


DISPLAY_NAMES = ("Sharp peak", "Curved ridge", "Curved sheet", "Ring")
STRUCTURE_PAIRS = ((1, 2), (1, 10), (1, 3), (1, 2))
ROW_COLORS = ("#1b1f24", "#df5b4f", "#07879b")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--repaired-mixture", type=Path, required=True)
    parser.add_argument(
        "--candidate-label", default="Protected specialists",
        help="Label used for the third proposal in figures and tables",
    )
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--samples", type=int, default=400_000)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 10_000 or args.bins < 10 or args.chunk <= 0:
        parser.error("increase the sampling/plotting budget")
    return args


@torch.inference_mode()
def sample_global(adaptive, model, condition, count, chunk):
    blocks = []
    for start in range(0, count, chunk):
        n = min(chunk, count - start)
        cond = condition.reshape(1, -1).expand(n, -1)
        blocks.append(adaptive.sample_from_model(model, n, cond).cpu())
    return torch.cat(blocks)


@torch.inference_mode()
def sample_repaired(
    adaptive, specialists, weights, uniform_fraction, condition, count, chunk
):
    blocks = []
    for start in range(0, count, chunk):
        n = min(chunk, count - start)
        cond = condition.reshape(1, -1).expand(n, -1)
        x, _ = mixture_audit.sample_mixture(
            adaptive, specialists, weights, uniform_fraction, cond
        )
        blocks.append(x.cpu())
    return torch.cat(blocks)


def labels_for(x, condition, device, chunk):
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(x), chunk):
            block = x[start:start + chunk].to(device)
            cond = condition.reshape(1, -1).expand(len(block), -1)
            blocks.append(benchmark.structure_labels(block, cond).cpu())
    return torch.cat(blocks).numpy()


def histogram2d(values, pair, edges):
    counts, _, _ = np.histogram2d(
        values[:, pair[0]], values[:, pair[1]], bins=(edges, edges)
    )
    return counts


def plot_structure_grid(samples, labels, args):
    row_names = ("Exact benchmark", "Global NF", args.candidate_label)
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    figure, axes = plt.subplots(
        3, benchmark.COMPONENTS, figsize=(17.6, 10.2),
        constrained_layout=True,
    )
    images = []
    for component, (name, pair) in enumerate(
        zip(DISPLAY_NAMES, STRUCTURE_PAIRS)
    ):
        counts = []
        subsets = []
        for values, value_labels in zip(samples, labels):
            subset = values[value_labels == component]
            subsets.append(subset)
            counts.append(histogram2d(subset, pair, edges))
        positive = np.concatenate([
            count[count > 0] for count in counts if np.any(count > 0)
        ])
        maximum = float(positive.max()) if len(positive) else 1.0
        norm = LogNorm(vmin=1.0, vmax=max(1.0, maximum))
        for row, (row_name, count, subset) in enumerate(
            zip(row_names, counts, subsets)
        ):
            image = axes[row, component].imshow(
                count.T, origin="lower", extent=(0, 1, 0, 1),
                aspect="auto", cmap="viridis", norm=norm,
            )
            axes[row, component].set_title(
                f"{name}: {row_name} (N={len(subset):,})", fontsize=10
            )
            axes[row, component].set_xlabel(fr"$x_{{{pair[0]}}}$")
            axes[row, component].set_ylabel(fr"$x_{{{pair[1]}}}$")
            images.append(image)
    figure.colorbar(
        images[-1], ax=axes, label="samples per bin", shrink=0.74, pad=0.01
    )
    figure.suptitle(
        f"Four-structure benchmark: exact target, global NF, and "
        f"{args.candidate_label}",
        fontsize=17,
    )
    path = args.output / "target_global_candidate_structure_by_structure.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_mass_comparison(labels, args):
    row_names = ("Exact benchmark", "Global NF", args.candidate_label)
    masses = np.array([
        [np.mean(value_labels == component)
         for component in range(benchmark.COMPONENTS)]
        for value_labels in labels
    ])
    locations = np.arange(benchmark.COMPONENTS)
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    for row, (name, color) in enumerate(zip(row_names, ROW_COLORS)):
        axis.bar(
            locations + (row - 1) * width, masses[row], width,
            color=color, alpha=0.88, label=name,
        )
    axis.set_xticks(locations, DISPLAY_NAMES)
    axis.set_ylim(0.0, max(0.32, 1.12 * float(masses.max())))
    axis.set_ylabel("proposal probability mass")
    axis.set_title("Mass assigned to each structure")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    path = args.output / "target_global_candidate_structure_masses.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return masses


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    condition = torch.tensor(
        args.fixed_condition, dtype=torch.float32, device=device
    )
    print(f"Sampling exact benchmark: {args.samples:,} points", flush=True)
    target, target_labels = benchmark.sample_exact_target(
        args.samples, condition
    )
    print(f"Sampling global NF: {args.samples:,} points", flush=True)
    global_adaptive, global_model = global_audit.load_model(
        args.global_checkpoint, device
    )
    global_samples = sample_global(
        global_adaptive, global_model, condition, args.samples, args.chunk
    )
    repaired_adaptive, specialists, weights, uniform_fraction = (
        mixture_audit.load_mixture(args.repaired_mixture, device)
    )
    print(
        f"Sampling {args.candidate_label}: {args.samples:,} points",
        flush=True,
    )
    repaired_samples = sample_repaired(
        repaired_adaptive, specialists, weights, uniform_fraction,
        condition, args.samples, args.chunk,
    )

    sample_arrays = (
        target.cpu().numpy(), global_samples.numpy(), repaired_samples.numpy()
    )
    label_arrays = (
        target_labels.cpu().numpy(),
        labels_for(global_samples, condition, device, args.chunk),
        labels_for(repaired_samples, condition, device, args.chunk),
    )
    print("Rendering comparison figures", flush=True)
    plot_structure_grid(sample_arrays, label_arrays, args)
    masses = plot_mass_comparison(label_arrays, args)

    rows = []
    row_names = ("Exact benchmark", "Global NF", args.candidate_label)
    for row, proposal in enumerate(row_names):
        gap = float(np.mean(label_arrays[row] < 0))
        for component, structure in enumerate(benchmark.STRUCTURE_NAMES):
            rows.append({
                "proposal": proposal,
                "structure": structure,
                "sample_mass": float(masses[row, component]),
                "gap_mass": gap,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "structure_mass_comparison.csv", index=False)
    summary = {
        "benchmark": benchmark.BENCHMARK,
        "global_checkpoint": str(args.global_checkpoint),
        "repaired_mixture": str(args.repaired_mixture),
        "condition": list(args.fixed_condition),
        "samples_per_distribution": args.samples,
        "figures": [
            "target_global_candidate_structure_by_structure.png",
            "target_global_candidate_structure_masses.png",
        ],
        "rows": rows,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print("\nSTRUCTURE MASS COMPARISON")
    print(frame.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
