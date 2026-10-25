"""Batched evaluation helpers for independent conditional cubic flows.

This module does not change the validated scalar spline primitive.  It applies
the same equations to a stack of independently-parameterized flows, avoiding
one Python/CUDA launch sequence per coordinate during teacher-forced training.
"""

from collections import defaultdict
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .conditional_cubic_1d import ConditionalCubicFlow1D


def _stacked_linear(x: Tensor, layers: Sequence[torch.nn.Linear]) -> Tensor:
    """Apply one different Linear layer to each leading slice of ``x``."""
    weight = torch.stack([layer.weight for layer in layers], dim=0)
    bias = torch.stack([layer.bias for layer in layers], dim=0)
    return torch.einsum("gnc,goc->gno", x, weight) + bias[:, None, :]


def batched_cubic_inverse(
    flows: Sequence[ConditionalCubicFlow1D],
    values: Tensor,
    contexts: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate several compatible cubic inverses in parallel.

    ``values`` has shape ``(G, N)`` and ``contexts`` has shape ``(G, N, C)``.
    Flow parameters remain independent and checkpoint-compatible.
    """
    if not flows:
        raise ValueError("at least one flow is required")
    if values.ndim != 2 or contexts.ndim != 3:
        raise ValueError("values must be (G,N) and contexts must be (G,N,C)")
    if values.shape[:2] != contexts.shape[:2] or values.shape[0] != len(flows):
        raise ValueError("flow, value, and context group sizes must agree")

    reference = flows[0]
    signature = (
        reference.cond_dim,
        reference.hidden,
        reference.n_bins,
        reference.min_bin_width,
        reference.min_bin_height,
        reference.inverse_iterations,
    )
    for flow in flows:
        current = (
            flow.cond_dim, flow.hidden, flow.n_bins,
            flow.min_bin_width, flow.min_bin_height, flow.inverse_iterations,
        )
        if current != signature:
            raise ValueError("all flows in a vectorized group must be compatible")
    if torch.any((values < 0) | (values > 1)):
        raise ValueError("values must lie in [0, 1]")

    linears = [[flow.net[index] for flow in flows] for index in (0, 2, 4)]
    hidden = F.relu(_stacked_linear(contexts, linears[0]))
    hidden = F.relu(_stacked_linear(hidden, linears[1]))
    raw_widths, raw_heights = _stacked_linear(hidden, linears[2]).chunk(2, dim=-1)

    bins = reference.n_bins
    widths = reference.min_bin_width + (1 - bins * reference.min_bin_width) * F.softmax(raw_widths, -1)
    heights = reference.min_bin_height + (1 - bins * reference.min_bin_height) * F.softmax(raw_heights, -1)
    cumwidths = F.pad(widths.cumsum(-1), (1, 0))
    cumheights = F.pad(heights.cumsum(-1), (1, 0))
    cumwidths = torch.cat((cumwidths[..., :-1], torch.ones_like(cumwidths[..., -1:])), -1)
    cumheights = torch.cat((cumheights[..., :-1], torch.ones_like(cumheights[..., -1:])), -1)

    index = (values[..., None] >= cumheights[..., 1:]).sum(-1).clamp_max(bins - 1)
    take = index[..., None]
    width = widths.gather(-1, take).squeeze(-1)
    height = heights.gather(-1, take).squeeze(-1)
    u0 = cumwidths[..., :-1].gather(-1, take).squeeze(-1)
    x0 = cumheights[..., :-1].gather(-1, take).squeeze(-1)

    secants = heights / widths
    if bins == 1:
        derivatives = torch.cat((secants, secants), dim=-1)
    else:
        left_w, right_w = widths[..., :-1], widths[..., 1:]
        weight1 = 2 * right_w + left_w
        weight2 = right_w + 2 * left_w
        interior = (weight1 + weight2) / (
            weight1 / secants[..., :-1] + weight2 / secants[..., 1:]
        )
        derivatives = torch.cat((secants[..., :1], interior, secants[..., -1:]), -1)
    d0 = derivatives[..., :-1].gather(-1, take).squeeze(-1)
    d1 = derivatives[..., 1:].gather(-1, take).squeeze(-1)

    def hermite(t: Tensor) -> tuple[Tensor, Tensor]:
        t2, t3 = t * t, t * t * t
        x = ((2 * t3 - 3 * t2 + 1) * x0
             + (t3 - 2 * t2 + t) * width * d0
             + (-2 * t3 + 3 * t2) * (x0 + height)
             + (t3 - t2) * width * d1)
        derivative = ((6 * t2 - 6 * t) * x0
                      + (3 * t2 - 4 * t + 1) * width * d0
                      + (-6 * t2 + 6 * t) * (x0 + height)
                      + (3 * t2 - 2 * t) * width * d1)
        return x, derivative

    low, high = torch.zeros_like(values), torch.ones_like(values)
    for _ in range(reference.inverse_iterations):
        mid = (low + high) * 0.5
        x_mid, _ = hermite(mid)
        low = torch.where(x_mid < values, mid, low)
        high = torch.where(x_mid < values, high, mid)
    t = (low + high) * 0.5
    x_t, dx_dt = hermite(t)
    tiny = torch.finfo(values.dtype).tiny
    t = (t - (x_t - values) / dx_dt.clamp_min(tiny)).clamp(0, 1)
    _, dx_dt = hermite(t)
    u = u0 + width * t
    logabsdet = torch.log(dx_dt.clamp_min(tiny)) - torch.log(width)
    return u, logabsdet


def vectorized_copula_log_prob(copula, u: Tensor, cond: Tensor) -> Tensor:
    """Teacher-forced copula log density, grouped by context width."""
    groups = defaultdict(list)
    for child in range(1, copula.dim):
        groups[len(copula.parents[child])].append(child)

    logq = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
    for children in groups.values():
        flows = [copula.conditionals[str(child)] for child in children]
        contexts = torch.stack([copula.context(u, cond, child) for child in children], 0)
        values = torch.stack([u[:, child] for child in children], 0)
        _, logdet = batched_cubic_inverse(flows, values, contexts)
        logq = logq - logdet.sum(0)
    return logq


def vectorized_marginal_inverse(marginals, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
    """Transform all equal-shaped marginals in one batched inverse call."""
    flows = list(marginals)
    contexts = cond.unsqueeze(0).expand(len(flows), -1, -1)
    u_by_dim, logdet_by_dim = batched_cubic_inverse(flows, x.transpose(0, 1), contexts)
    return u_by_dim.transpose(0, 1).clamp(0.0, 1.0), -logdet_by_dim.sum(0)
