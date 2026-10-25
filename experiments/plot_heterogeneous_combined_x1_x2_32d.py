#!/usr/bin/env python3
"""Plot the combined x1-x2 projection of the heterogeneous 32D target.

This lightweight diagnostic reproduces the first three coordinates of the
exact benchmark sampler without importing the training stack.  It shows that
x0 selects a branch while every branch defines a different conditional law
for the same displayed coordinates x1 and x2.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm


COMPONENTS = 4
BRANCH_WIDTH = 0.11
MINIMUM_GAP = 0.03
NAMES = ("Sharp peak", "Curved ridge", "Curved sheet", "Ring")
COLORS = ("#D95F59", "#D99A24", "#07879B", "#6950A1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 0 or value > 1 for value in args.condition):
        parser.error("condition must lie in [0,1]^3")
    if args.samples < 10_000 or args.bins < 20:
        parser.error("increase the plotting budget")
    return args


def branch_probabilities(condition):
    c0, c1, c2 = condition
    logits = torch.tensor(
        (
            1.25 * (c0 - 0.5) + 0.55 * math.sin(2 * math.pi * c1),
            -0.85 * (c0 - 0.5) + 1.10 * (c2 - 0.5),
            0.75 * math.sin(2 * math.pi * c2) - 0.65 * (c1 - 0.5),
            -0.55 * (c2 - 0.5) + 0.70 * (c1 - 0.5),
        ),
        dtype=torch.float64,
    )
    return 0.04 + 0.84 * torch.softmax(logits, dim=0)


def branch_intervals(condition):
    c0, c1, c2 = condition
    gap_logits = torch.tensor(
        (
            0.8 * (c0 - 0.5),
            0.7 * math.sin(2 * math.pi * c1),
            0.8 * (c2 - c0),
            -0.7 * math.sin(2 * math.pi * c2),
            0.8 * (c1 - 0.5),
        ),
        dtype=torch.float64,
    )
    free_gap = 1.0 - COMPONENTS * BRANCH_WIDTH - 5 * MINIMUM_GAP
    gaps = MINIMUM_GAP + free_gap * torch.softmax(gap_logits, dim=0)
    lower = []
    cursor = gaps[0]
    for component in range(COMPONENTS):
        lower.append(cursor)
        cursor = cursor + BRANCH_WIDTH + gaps[component + 1]
    lower = torch.stack(lower)
    return lower, lower + BRANCH_WIDTH


def truncated_normal(mean, scale, count, lower=0.0, upper=1.0):
    mean = torch.as_tensor(mean, dtype=torch.float64).expand(count)
    scale = torch.as_tensor(scale, dtype=torch.float64).expand(count)
    cdf_lower = torch.special.ndtr((lower - mean) / scale)
    cdf_upper = torch.special.ndtr((upper - mean) / scale)
    u = cdf_lower + torch.rand(count, dtype=torch.float64) * (
        cdf_upper - cdf_lower
    )
    eps = torch.finfo(torch.float64).eps
    return mean + scale * torch.special.ndtri(u.clamp(eps, 1 - eps))


def coordinate_base(child, condition, phase_offset=0.0):
    c0, c1, c2 = condition
    phase = 0.37 * child + phase_offset
    signal = (
        0.70 * math.sin(2 * math.pi * c0 + phase)
        + 0.35 * math.cos(2 * math.pi * c1 - 0.6 * phase)
        + 0.20 * math.sin(2 * math.pi * c2 + 0.2 * phase)
    )
    return 0.5 + 0.18 * math.tanh(signal)


def sample_peak(count, condition):
    c0, c1, c2 = condition
    orientation = 0.62 * math.tanh(2.2 * (c1 - 0.5))
    scale_factor = 0.85 + 0.30 * c2
    base1 = coordinate_base(1, condition, phase_offset=0.2)
    x1 = truncated_normal(base1, 0.024 * scale_factor, count)
    base2 = coordinate_base(2, condition, phase_offset=0.2)
    mean2 = (base2 + orientation * (x1 - base1)).clamp(0.06, 0.94)
    x2 = truncated_normal(mean2, 0.026 * scale_factor, count)
    return x1, x2


def sample_ridge(count, condition):
    c0, c1, c2 = condition
    driver_mean = 0.50 + 0.12 * math.sin(2 * math.pi * c0)
    driver_scale = 0.19 + 0.025 * c2
    x1 = truncated_normal(driver_mean, driver_scale, count)
    curvature = 0.16 + 0.055 * c1
    frequency = 0.75 + 0.55 * c2
    thickness = 0.030 + 0.012 * c0
    phase = 0.29 * 2 + 1.3 * (c0 - c2)
    mean2 = (
        0.5
        + curvature * torch.sin(2 * math.pi * frequency * x1 + phase)
        + 0.035 * torch.cos(math.pi * x1 - 0.17 * 2)
    ).clamp(0.06, 0.94)
    x2 = truncated_normal(mean2, thickness * (0.90 + 0.04 * 2), count)
    return x1, x2


def sample_sheet(count, condition):
    c0, c1, c2 = condition
    mean1 = 0.46 + 0.10 * math.sin(2 * math.pi * c0)
    mean2 = 0.54 + 0.10 * math.cos(2 * math.pi * c2)
    x1 = truncated_normal(mean1, 0.20 + 0.02 * c1, count)
    x2 = truncated_normal(mean2, 0.18 + 0.03 * c0, count)
    return x1, x2


def sample_ring(count, condition):
    c0, c1, c2 = condition
    center0 = 0.50 + 0.035 * math.sin(2 * math.pi * c0)
    center1 = 0.50 + 0.035 * math.cos(2 * math.pi * c2)
    rotation = math.pi * (0.10 + 0.80 * c1)
    axis_a = 0.88 + 0.28 * c0
    axis_b = 1.18 - 0.24 * c2
    radius = 0.195 + 0.025 * math.sin(2 * math.pi * c2)
    radial_scale = 0.016 + 0.008 * c0
    radial = truncated_normal(radius, radial_scale, count, 0.10, 0.30)
    angle = 2 * math.pi * torch.rand(count, dtype=torch.float64) - math.pi
    y0 = radial * torch.cos(angle)
    y1 = radial * torch.sin(angle)
    cosine, sine = math.cos(rotation), math.sin(rotation)
    x1 = center0 + cosine * axis_a * y0 - sine * axis_b * y1
    x2 = center1 + sine * axis_a * y0 + cosine * axis_b * y1
    return x1, x2


def sample_target(count, condition):
    probabilities = branch_probabilities(condition)
    labels = torch.multinomial(probabilities, count, replacement=True)
    lower, _ = branch_intervals(condition)
    x = torch.empty(count, 3, dtype=torch.float64)
    x[:, 0] = lower[labels] + BRANCH_WIDTH * torch.rand(
        count, dtype=torch.float64
    )
    samplers = (sample_peak, sample_ridge, sample_sheet, sample_ring)
    for component, sampler in enumerate(samplers):
        mask = labels == component
        current = int(mask.sum())
        if current:
            x1, x2 = sampler(current, condition)
            x[mask, 1] = x1
            x[mask, 2] = x2
    return x.numpy(), labels.numpy(), probabilities.numpy(), lower.numpy()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    condition = tuple(args.condition)
    x, labels, probabilities, lower = sample_target(args.samples, condition)

    figure, axes = plt.subplots(
        1, 3, figsize=(16.5, 4.8), constrained_layout=True
    )
    counts, xedges, yedges = np.histogram2d(
        x[:, 1], x[:, 2], bins=args.bins, range=((0, 1), (0, 1))
    )
    image = axes[0].imshow(
        counts.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal",
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(1, counts.max())),
    )
    axes[0].set_title(r"All branches combined: $p(x_1,x_2\mid c)$")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    figure.colorbar(image, ax=axes[0], label="samples per bin", shrink=0.82)

    rng = np.random.default_rng(args.seed + 1)
    for component, (name, color) in enumerate(zip(NAMES, COLORS)):
        indices = np.flatnonzero(labels == component)
        if len(indices) > 3500:
            indices = rng.choice(indices, 3500, replace=False)
        axes[1].scatter(
            x[indices, 1], x[indices, 2], s=3.5, alpha=0.28,
            color=color, label=name, rasterized=True,
        )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect("equal")
    axes[1].set_title(r"Same projection, colored by the $x_0$ branch")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    axes[1].legend(frameon=False, markerscale=3, loc="upper right")

    for component, (name, color) in enumerate(zip(NAMES, COLORS)):
        indices = np.flatnonzero(labels == component)
        if len(indices) > 3500:
            indices = rng.choice(indices, 3500, replace=False)
        axes[2].scatter(
            x[indices, 0], x[indices, 1], s=3.5, alpha=0.25,
            color=color, rasterized=True,
        )
        axes[2].axvspan(
            lower[component], lower[component] + BRANCH_WIDTH,
            color=color, alpha=0.08,
        )
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_title(r"$x_0$ selects one of four conditional laws")
    axes[2].set_xlabel(r"branch coordinate $x_0$")
    axes[2].set_ylabel(r"$x_1$")

    condition_text = ", ".join(f"{value:.2f}" for value in condition)
    figure.suptitle(
        rf"Heterogeneous target at $c=({condition_text})$: one benchmark, four branches",
        fontsize=16,
    )
    path = args.output / "combined_x1_x2_conditioned_by_x0.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    print("branch probabilities:")
    for name, probability, start in zip(NAMES, probabilities, lower):
        print(
            f"  {name:12s} mass={probability:.6f} "
            f"x0=[{start:.6f}, {start + BRANCH_WIDTH:.6f}]"
        )
    print(f"\nPlot: {path.resolve()}")


if __name__ == "__main__":
    main()
