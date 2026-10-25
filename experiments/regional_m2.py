"""Utilities for stable regional second-moment objectives.

For samples drawn from a frozen density ``r``, the contribution to

    M2(q) = integral f(x)^2 / q(x) dx

is ``f(x)^2 / (q(x) r(x))``.  The helpers below aggregate those
contributions over a fixed partition without exponentiating their full
dynamic range.
"""

from __future__ import annotations

import math

import torch


def assign_nearest_region(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Assign each row of ``x`` to its nearest Euclidean center."""
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError("x and centers must be compatible rank-two tensors")
    return torch.cdist(x, centers).argmin(dim=1)


def regional_log_moments(
    log_contribution: torch.Tensor,
    region_index: torch.Tensor,
    region_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log global and raw regional Monte Carlo M2 estimates.

    ``log_contribution`` is ``log(f^2 / (q r))`` for samples from ``r``.
    Regional estimates retain the global ``1/N`` normalization, so their
    sum is exactly the global estimate.
    """
    values = log_contribution.reshape(-1)
    labels = region_index.reshape(-1)
    if len(values) != len(labels):
        raise ValueError("contributions and region labels must have equal size")
    if region_count <= 0:
        raise ValueError("region_count must be positive")
    if len(values) == 0:
        raise ValueError("at least one contribution is required")
    if int(labels.min()) < 0 or int(labels.max()) >= region_count:
        raise ValueError("region label lies outside [0, region_count)")

    scale = values.detach().max()
    scaled = torch.exp(values - scale)
    totals = torch.zeros(
        region_count, dtype=scaled.dtype, device=scaled.device
    )
    totals.scatter_add_(0, labels, scaled)
    log_n = math.log(len(values))
    log_regions = torch.where(
        totals > 0,
        torch.log(totals) + scale - log_n,
        torch.full_like(totals, -torch.inf),
    )
    log_global = torch.logsumexp(values, dim=0) - log_n
    return log_global, log_regions


def combined_log_objective(
    log_global: torch.Tensor,
    log_regions: torch.Tensor,
    regional_weight: float,
    temperature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``log(M2 + C * max_region M2_region)`` stably.

    A positive ``temperature`` replaces the hard maximum with a smooth
    maximum in linear M2 units.  The returned second value is the (possibly
    smoothed) log regional maximum.
    """
    if regional_weight < 0:
        raise ValueError("regional_weight must be nonnegative")
    finite = torch.isfinite(log_regions)
    if not finite.any():
        raise ValueError("all regions are empty")
    active = log_regions[finite]
    if temperature > 0:
        # Work relative to global M2. Regional/global values lie in [0,1].
        fractions = torch.exp(active - log_global)
        smooth_fraction = temperature * torch.logsumexp(
            fractions / temperature, dim=0
        )
        # log of a positive smooth maximum.  Numerical clamping affects only
        # extremely small, effectively empty regional fractions.
        log_regional_max = log_global + torch.log(
            smooth_fraction.clamp_min(torch.finfo(fractions.dtype).tiny)
        )
    else:
        log_regional_max = active.max()

    if regional_weight == 0:
        return log_global, log_regional_max
    objective = torch.logaddexp(
        log_global,
        log_regional_max + math.log(regional_weight),
    )
    return objective, log_regional_max


@torch.no_grad()
def diverse_weighted_centers(
    x: torch.Tensor,
    log_priority: torch.Tensor,
    count: int,
    minimum_distance: float = 0.10,
    refinement_steps: int = 8,
) -> torch.Tensor:
    """Cluster high-contribution points into diverse regional centers."""
    if x.ndim != 2 or log_priority.ndim != 1 or len(x) != len(log_priority):
        raise ValueError("x and log_priority have incompatible shapes")
    if count <= 0 or count > len(x):
        raise ValueError("count must lie in [1, len(x)]")

    order = torch.argsort(log_priority, descending=True)
    chosen = [int(order[0])]
    for candidate in order[1:].tolist():
        distance = torch.cdist(
            x[candidate:candidate + 1], x[chosen]
        ).min()
        if float(distance) >= minimum_distance:
            chosen.append(candidate)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        remaining = [int(index) for index in order.tolist() if int(index) not in chosen]
        chosen.extend(remaining[:count - len(chosen)])
    centers = x[chosen].clone()

    for _ in range(refinement_steps):
        labels = assign_nearest_region(x, centers)
        updated = centers.clone()
        for region in range(count):
            mask = labels == region
            if not mask.any():
                continue
            local_log = log_priority[mask]
            # Clipping prevents a single discovered point from defining an
            # entire geometric region while preserving tail emphasis.
            local_log = torch.clamp(
                local_log,
                min=float(local_log.max()) - 20.0,
            )
            weight = torch.softmax(local_log, dim=0)
            updated[region] = (weight[:, None] * x[mask]).sum(dim=0)
        if torch.max(torch.abs(updated - centers)) < 1e-6:
            centers = updated
            break
        centers = updated
    return centers
