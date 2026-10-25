#!/usr/bin/env python3
"""A normalized 32D benchmark with four heterogeneous disconnected structures.

The first coordinate assigns four disjoint, condition-dependent slabs.  The
remaining coordinates contain respectively:

1. a sharp tilted peak;
2. a thin curved ridge;
3. a curved two-dimensional sheet;
4. an elliptical ring.

The three conditions jointly change mixture masses, separation, position,
orientation, width and curvature.  Every conditional component is normalized,
so the complete integral is exactly one.  Training is still integrand-only;
``sample_exact_target`` exists solely for external diagnostic plots/audits and
is never registered with the adaptive trainer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_geometry_ladder_32d as geometry  # noqa: E402


adaptive = geometry.adaptive
DIMENSION = 32
CONDITION_DIMENSION = 3
COMPONENTS = 4
BRANCH_WIDTH = 0.11
MINIMUM_GAP = 0.03
BENCHMARK = "heterogeneous_four_structure_32"
STRUCTURE_NAMES = ("sharp_peak", "curved_ridge", "curved_sheet", "ring")


def branch_probabilities(cond):
    """Smooth nonvanishing condition-dependent structure masses."""
    c0, c1, c2 = cond[:, 0], cond[:, 1], cond[:, 2]
    logits = torch.stack(
        (
            1.25 * (c0 - 0.5) + 0.55 * torch.sin(2 * math.pi * c1),
            -0.85 * (c0 - 0.5) + 1.10 * (c2 - 0.5),
            0.75 * torch.sin(2 * math.pi * c2) - 0.65 * (c1 - 0.5),
            -0.55 * (c2 - 0.5) + 0.70 * (c1 - 0.5),
        ),
        dim=1,
    )
    # A small uniform floor keeps every fundamentally different structure
    # observable throughout condition space.
    return 0.04 + 0.84 * torch.softmax(logits, dim=1)


def branch_intervals(cond):
    """Ordered disjoint x0 intervals whose gaps change with condition."""
    c0, c1, c2 = cond[:, 0], cond[:, 1], cond[:, 2]
    gap_logits = torch.stack(
        (
            0.8 * (c0 - 0.5),
            0.7 * torch.sin(2 * math.pi * c1),
            0.8 * (c2 - c0),
            -0.7 * torch.sin(2 * math.pi * c2),
            0.8 * (c1 - 0.5),
        ),
        dim=1,
    )
    free_gap = 1.0 - COMPONENTS * BRANCH_WIDTH - 5 * MINIMUM_GAP
    gaps = MINIMUM_GAP + free_gap * torch.softmax(gap_logits, dim=1)
    lower = []
    cursor = gaps[:, 0]
    for component in range(COMPONENTS):
        lower.append(cursor)
        cursor = cursor + BRANCH_WIDTH + gaps[:, component + 1]
    lower = torch.stack(lower, dim=1)
    return lower, lower + BRANCH_WIDTH


def _normal_logpdf(value, mean, scale):
    z = (value - mean) / scale
    return -0.5 * z.square() - torch.log(scale) - 0.5 * math.log(2 * math.pi)


def _truncated_normal_logpdf(value, mean, scale, lower=0.0, upper=1.0):
    lower_tensor = torch.as_tensor(lower, dtype=value.dtype, device=value.device)
    upper_tensor = torch.as_tensor(upper, dtype=value.dtype, device=value.device)
    cdf_lower = torch.special.ndtr((lower_tensor - mean) / scale)
    cdf_upper = torch.special.ndtr((upper_tensor - mean) / scale)
    normalizer = (cdf_upper - cdf_lower).clamp_min(torch.finfo(value.dtype).tiny)
    result = _normal_logpdf(value, mean, scale) - torch.log(normalizer)
    inside = (value >= lower_tensor) & (value <= upper_tensor)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def _truncated_normal_sample(mean, scale, lower=0.0, upper=1.0):
    lower_tensor = torch.as_tensor(lower, dtype=mean.dtype, device=mean.device)
    upper_tensor = torch.as_tensor(upper, dtype=mean.dtype, device=mean.device)
    cdf_lower = torch.special.ndtr((lower_tensor - mean) / scale)
    cdf_upper = torch.special.ndtr((upper_tensor - mean) / scale)
    u = cdf_lower + torch.rand_like(mean) * (cdf_upper - cdf_lower)
    eps = torch.finfo(mean.dtype).eps
    return mean + scale * torch.special.ndtri(u.clamp(eps, 1 - eps))


def _coordinate_base(child, cond, phase_offset=0.0):
    phase = 0.37 * child + phase_offset
    signal = (
        0.70 * torch.sin(2 * math.pi * cond[:, 0] + phase)
        + 0.35 * torch.cos(2 * math.pi * cond[:, 1] - 0.6 * phase)
        + 0.20 * torch.sin(2 * math.pi * cond[:, 2] + 0.2 * phase)
    )
    return 0.5 + 0.18 * torch.tanh(signal)


def sharp_peak_log_density(x, cond):
    """Narrow autoregressive ellipsoid with condition-dependent tilt."""
    result = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    previous_base = None
    orientation = 0.62 * torch.tanh(2.2 * (cond[:, 1] - 0.5))
    scale_factor = 0.85 + 0.30 * cond[:, 2]
    for child in range(1, DIMENSION):
        base = _coordinate_base(child, cond, phase_offset=0.2)
        mean = base
        if child > 1:
            mean = mean + orientation * (x[:, child - 1] - previous_base)
        mean = mean.clamp(0.06, 0.94)
        scale = (0.022 + 0.002 * (child % 4)) * scale_factor
        result += _truncated_normal_logpdf(
            x[:, child], mean, scale
        )
        previous_base = base
    return result


def curved_ridge_log_density(x, cond):
    """One broad coordinate driving 30 narrow nonlinear coordinates."""
    driver_mean = 0.50 + 0.12 * torch.sin(2 * math.pi * cond[:, 0])
    driver_scale = 0.19 + 0.025 * cond[:, 2]
    result = _truncated_normal_logpdf(x[:, 1], driver_mean, driver_scale)
    curvature = 0.16 + 0.055 * cond[:, 1]
    frequency = 0.75 + 0.55 * cond[:, 2]
    thickness = 0.030 + 0.012 * cond[:, 0]
    for child in range(2, DIMENSION):
        phase = 0.29 * child + 1.3 * (cond[:, 0] - cond[:, 2])
        mean = (
            0.5
            + curvature * torch.sin(
                2 * math.pi * frequency * x[:, 1] + phase
            )
            + 0.035 * torch.cos(math.pi * x[:, 1] - 0.17 * child)
        ).clamp(0.06, 0.94)
        scale = thickness * (0.90 + 0.04 * (child % 5))
        result += _truncated_normal_logpdf(x[:, child], mean, scale)
    return result


def curved_sheet_log_density(x, cond):
    """Two broad intrinsic coordinates embedded as a curved 2D sheet."""
    mean1 = 0.46 + 0.10 * torch.sin(2 * math.pi * cond[:, 0])
    mean2 = 0.54 + 0.10 * torch.cos(2 * math.pi * cond[:, 2])
    result = _truncated_normal_logpdf(
        x[:, 1], mean1, 0.20 + 0.02 * cond[:, 1]
    )
    result += _truncated_normal_logpdf(
        x[:, 2], mean2, 0.18 + 0.03 * cond[:, 0]
    )
    angle = math.pi * (0.20 + 0.60 * cond[:, 1])
    u = torch.cos(angle) * (x[:, 1] - 0.5) + torch.sin(angle) * (x[:, 2] - 0.5)
    v = -torch.sin(angle) * (x[:, 1] - 0.5) + torch.cos(angle) * (x[:, 2] - 0.5)
    thickness = 0.038 + 0.012 * cond[:, 2]
    for child in range(3, DIMENSION):
        phase = 0.23 * child + 0.8 * cond[:, 0]
        mean = (
            0.5
            + 0.19 * torch.sin(2 * math.pi * u + phase)
            + 0.13 * torch.cos(2 * math.pi * v - 0.7 * phase)
            + 0.045 * u * v
        ).clamp(0.06, 0.94)
        scale = thickness * (0.92 + 0.035 * (child % 6))
        result += _truncated_normal_logpdf(x[:, child], mean, scale)
    return result


def _ring_parameters(cond):
    center = torch.stack(
        (
            0.50 + 0.035 * torch.sin(2 * math.pi * cond[:, 0]),
            0.50 + 0.035 * torch.cos(2 * math.pi * cond[:, 2]),
        ),
        dim=1,
    )
    rotation = math.pi * (0.10 + 0.80 * cond[:, 1])
    axis_a = 0.88 + 0.28 * cond[:, 0]
    axis_b = 1.18 - 0.24 * cond[:, 2]
    radius = 0.195 + 0.025 * torch.sin(2 * math.pi * cond[:, 2])
    radial_scale = 0.016 + 0.008 * cond[:, 0]
    return center, rotation, axis_a, axis_b, radius, radial_scale


def _ring_inverse(x12, cond):
    center, rotation, axis_a, axis_b, radius, radial_scale = _ring_parameters(cond)
    delta = x12 - center
    cosine, sine = torch.cos(rotation), torch.sin(rotation)
    rotated0 = cosine * delta[:, 0] + sine * delta[:, 1]
    rotated1 = -sine * delta[:, 0] + cosine * delta[:, 1]
    y0, y1 = rotated0 / axis_a, rotated1 / axis_b
    radial = torch.sqrt(y0.square() + y1.square()).clamp_min(1e-12)
    angle = torch.atan2(y1, y0)
    return radial, angle, axis_a * axis_b, radius, radial_scale


def ring_log_density(x, cond):
    """Thin oriented elliptical annulus with condition-dependent geometry."""
    radial, angle, determinant, radius, radial_scale = _ring_inverse(
        x[:, 1:3], cond
    )
    radial_log = _truncated_normal_logpdf(
        radial, radius, radial_scale, lower=0.10, upper=0.30
    )
    result = (
        radial_log
        - math.log(2 * math.pi)
        - torch.log(radial)
        - torch.log(determinant)
    )
    thickness = 0.033 + 0.010 * cond[:, 1]
    for child in range(3, DIMENSION):
        harmonic = 1 + child % 3
        phase = 0.21 * child + 0.7 * cond[:, 2]
        mean = (
            0.5
            + 0.17 * torch.sin(harmonic * angle + phase)
            + 0.055 * torch.cos(angle - 0.13 * child)
        ).clamp(0.06, 0.94)
        scale = thickness * (0.90 + 0.04 * (child % 5))
        result += _truncated_normal_logpdf(x[:, child], mean, scale)
    inside = ((x[:, 1:3] >= 0) & (x[:, 1:3] <= 1)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def component_log_densities(x, cond):
    """Normalized component densities including their disjoint x0 slabs."""
    lower, upper = branch_intervals(cond)
    geometry_logs = (
        sharp_peak_log_density(x, cond),
        curved_ridge_log_density(x, cond),
        curved_sheet_log_density(x, cond),
        ring_log_density(x, cond),
    )
    rows = []
    for component, geometry_log in enumerate(geometry_logs):
        inside = (x[:, 0] >= lower[:, component]) & (x[:, 0] <= upper[:, component])
        log_density = geometry_log - math.log(BRANCH_WIDTH)
        rows.append(torch.where(
            inside, log_density, torch.full_like(log_density, -torch.inf)
        ))
    return torch.stack(rows, dim=1)


def heterogeneous_log_integrand(x, cond):
    if x.shape[1] != DIMENSION or cond.shape[1] != CONDITION_DIMENSION:
        raise ValueError("heterogeneous benchmark requires x=(N,32), c=(N,3)")
    probabilities = branch_probabilities(cond)
    result = torch.logsumexp(
        component_log_densities(x, cond) + probabilities.log(), dim=1
    )
    inside = ((x >= 0) & (x <= 1)).all(dim=1)
    return torch.where(inside, result, torch.full_like(result, -torch.inf))


def structure_labels(x, cond):
    lower, upper = branch_intervals(cond)
    membership = (
        (x[:, 0, None] >= lower) & (x[:, 0, None] <= upper)
    )
    label = membership.to(torch.int64).argmax(dim=1)
    return torch.where(membership.any(dim=1), label, torch.full_like(label, -1))


def _sample_peak(x, cond):
    previous_base = None
    orientation = 0.62 * torch.tanh(2.2 * (cond[:, 1] - 0.5))
    scale_factor = 0.85 + 0.30 * cond[:, 2]
    for child in range(1, DIMENSION):
        base = _coordinate_base(child, cond, phase_offset=0.2)
        mean = base
        if child > 1:
            mean = mean + orientation * (x[:, child - 1] - previous_base)
        mean = mean.clamp(0.06, 0.94)
        scale = (0.022 + 0.002 * (child % 4)) * scale_factor
        x[:, child] = _truncated_normal_sample(mean, scale)
        previous_base = base


def _sample_ridge(x, cond):
    driver_mean = 0.50 + 0.12 * torch.sin(2 * math.pi * cond[:, 0])
    driver_scale = 0.19 + 0.025 * cond[:, 2]
    x[:, 1] = _truncated_normal_sample(driver_mean, driver_scale)
    curvature = 0.16 + 0.055 * cond[:, 1]
    frequency = 0.75 + 0.55 * cond[:, 2]
    thickness = 0.030 + 0.012 * cond[:, 0]
    for child in range(2, DIMENSION):
        phase = 0.29 * child + 1.3 * (cond[:, 0] - cond[:, 2])
        mean = (
            0.5
            + curvature * torch.sin(2 * math.pi * frequency * x[:, 1] + phase)
            + 0.035 * torch.cos(math.pi * x[:, 1] - 0.17 * child)
        ).clamp(0.06, 0.94)
        scale = thickness * (0.90 + 0.04 * (child % 5))
        x[:, child] = _truncated_normal_sample(mean, scale)


def _sample_sheet(x, cond):
    mean1 = 0.46 + 0.10 * torch.sin(2 * math.pi * cond[:, 0])
    mean2 = 0.54 + 0.10 * torch.cos(2 * math.pi * cond[:, 2])
    x[:, 1] = _truncated_normal_sample(mean1, 0.20 + 0.02 * cond[:, 1])
    x[:, 2] = _truncated_normal_sample(mean2, 0.18 + 0.03 * cond[:, 0])
    angle = math.pi * (0.20 + 0.60 * cond[:, 1])
    u = torch.cos(angle) * (x[:, 1] - 0.5) + torch.sin(angle) * (x[:, 2] - 0.5)
    v = -torch.sin(angle) * (x[:, 1] - 0.5) + torch.cos(angle) * (x[:, 2] - 0.5)
    thickness = 0.038 + 0.012 * cond[:, 2]
    for child in range(3, DIMENSION):
        phase = 0.23 * child + 0.8 * cond[:, 0]
        mean = (
            0.5
            + 0.19 * torch.sin(2 * math.pi * u + phase)
            + 0.13 * torch.cos(2 * math.pi * v - 0.7 * phase)
            + 0.045 * u * v
        ).clamp(0.06, 0.94)
        scale = thickness * (0.92 + 0.035 * (child % 6))
        x[:, child] = _truncated_normal_sample(mean, scale)


def _sample_ring(x, cond):
    center, rotation, axis_a, axis_b, radius, radial_scale = _ring_parameters(cond)
    radial = _truncated_normal_sample(
        radius, radial_scale, lower=0.10, upper=0.30
    )
    angle = 2 * math.pi * torch.rand(len(x), dtype=x.dtype, device=x.device) - math.pi
    y0, y1 = radial * torch.cos(angle), radial * torch.sin(angle)
    cosine, sine = torch.cos(rotation), torch.sin(rotation)
    x[:, 1] = center[:, 0] + cosine * axis_a * y0 - sine * axis_b * y1
    x[:, 2] = center[:, 1] + sine * axis_a * y0 + cosine * axis_b * y1
    thickness = 0.033 + 0.010 * cond[:, 1]
    for child in range(3, DIMENSION):
        harmonic = 1 + child % 3
        phase = 0.21 * child + 0.7 * cond[:, 2]
        mean = (
            0.5
            + 0.17 * torch.sin(harmonic * angle + phase)
            + 0.055 * torch.cos(angle - 0.13 * child)
        ).clamp(0.06, 0.94)
        scale = thickness * (0.90 + 0.04 * (child % 5))
        x[:, child] = _truncated_normal_sample(mean, scale)


@torch.no_grad()
def sample_exact_target(n, cond):
    """Diagnostic-only exact sampler; the NF trainer never calls this."""
    if cond.ndim == 1:
        cond = cond.reshape(1, -1).expand(n, -1)
    if len(cond) != n:
        raise ValueError("cond must contain one row or n rows")
    probabilities = branch_probabilities(cond)
    component = torch.multinomial(probabilities, 1).reshape(-1)
    lower, _ = branch_intervals(cond)
    x = torch.empty(n, DIMENSION, dtype=cond.dtype, device=cond.device)
    x[:, 0] = lower[torch.arange(n, device=cond.device), component]
    x[:, 0] += BRANCH_WIDTH * torch.rand(n, dtype=cond.dtype, device=cond.device)
    samplers = (_sample_peak, _sample_ridge, _sample_sheet, _sample_ring)
    for index, sampler in enumerate(samplers):
        mask = component == index
        if bool(mask.any()):
            block = torch.empty(
                int(mask.sum()), DIMENSION, dtype=cond.dtype, device=cond.device
            )
            block[:, 0] = x[mask, 0]
            sampler(block, cond[mask])
            x[mask] = block
    return x, component


def dependency_graph(dim):
    graph = {child: [] for child in range(dim)}
    if dim > 2:
        graph[2] = [1]
    for child in range(3, dim):
        graph[child] = sorted({1, 2, child - 1})
    return graph


def _pop_fixed_condition():
    if "--fixed-condition" not in sys.argv:
        return None
    index = sys.argv.index("--fixed-condition")
    if len(sys.argv) < index + 4:
        raise SystemExit("--fixed-condition requires C0 C1 C2")
    values = tuple(float(value) for value in sys.argv[index + 1:index + 4])
    del sys.argv[index:index + 4]
    if any(value < 0 or value > 1 for value in values):
        raise SystemExit("fixed condition must lie in [0,1]^3")
    return values


def register_benchmark(fixed_condition=None):
    geometry.register_benchmarks()
    previous_target = adaptive.target_log_integrand

    def target_log_integrand(benchmark, x, cond):
        if benchmark == BENCHMARK:
            return heterogeneous_log_integrand(x, cond)
        return previous_target(benchmark, x, cond)

    adaptive.SUPPORTED_BENCHMARKS.add(BENCHMARK)
    adaptive.BENCHMARK_DIMENSIONS[BENCHMARK] = DIMENSION
    adaptive.ORACLE_INTEGRALS[BENCHMARK] = 1.0
    adaptive.target_log_integrand = target_log_integrand
    adaptive.full_ar_graph = dependency_graph
    if fixed_condition is not None:
        condition = tuple(float(value) for value in fixed_condition)

        def sample_fixed_conditions(n, device):
            return torch.tensor(
                condition, dtype=torch.float32, device=device
            ).reshape(1, CONDITION_DIMENSION).expand(n, -1).clone()

        adaptive.sample_conditions = sample_fixed_conditions
    print(
        "32D heterogeneous topology: sharp peak + curved ridge + curved "
        "sheet + ring on four condition-dependent disconnected branches; "
        "exact I(c)=1"
    )
    if fixed_condition is not None:
        print(f"Training condition frozen at c={fixed_condition}")


def main():
    fixed_condition = _pop_fixed_condition()
    register_benchmark(fixed_condition)
    adaptive.main()


if __name__ == "__main__":
    main()
