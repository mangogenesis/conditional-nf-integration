"""Conditional monotone cubic-Hermite flow on the unit interval."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .conditional_linear_1d import ConditionalLinearFlow1D


class ConditionalCubicFlow1D(ConditionalLinearFlow1D):
    """A C1 monotone cubic Hermite map, inverted by per-bin bisection."""

    def __init__(
        self,
        cond_dim: int = 2,
        hidden: int = 128,
        n_bins: int = 8,
        min_bin_width: float = 1e-4,
        min_bin_height: float = 1e-4,
        inverse_iterations: int = 40,
    ) -> None:
        super().__init__(cond_dim, hidden, n_bins, min_bin_width, min_bin_height)
        if inverse_iterations < 1:
            raise ValueError("inverse_iterations must be positive")
        self.inverse_iterations = inverse_iterations

    @staticmethod
    def _derivatives(widths: Tensor, heights: Tensor) -> Tensor:
        """PCHIP knot slopes; these preserve monotonicity for positive secants."""
        secants = heights / widths
        if secants.shape[1] == 1:
            return torch.cat((secants, secants), dim=-1)
        left_w, right_w = widths[:, :-1], widths[:, 1:]
        weight1 = 2 * right_w + left_w
        weight2 = right_w + 2 * left_w
        interior = (weight1 + weight2) / (weight1 / secants[:, :-1] + weight2 / secants[:, 1:])
        return torch.cat((secants[:, :1], interior, secants[:, -1:]), dim=-1)

    @staticmethod
    def _hermite(t: Tensor, y0: Tensor, height: Tensor, width: Tensor, d0: Tensor, d1: Tensor) -> tuple[Tensor, Tensor]:
        t2, t3 = t * t, t * t * t
        x = ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * width * d0
             + (-2 * t3 + 3 * t2) * (y0 + height) + (t3 - t2) * width * d1)
        dx_dt = ((6 * t2 - 6 * t) * y0 + (3 * t2 - 4 * t + 1) * width * d0
                 + (-6 * t2 + 6 * t) * (y0 + height) + (3 * t2 - 2 * t) * width * d1)
        return x, dx_dt

    def _selected(self, values: Tensor, cond: Tensor, inverse: bool):
        widths, heights, cumwidths, cumheights = self._params(cond)
        knots = cumheights if inverse else cumwidths
        idx = (values[:, None] >= knots[:, 1:]).sum(-1).clamp_max(self.n_bins - 1)
        take = idx[:, None]
        width = widths.gather(1, take).squeeze(1)
        height = heights.gather(1, take).squeeze(1)
        u0 = cumwidths[:, :-1].gather(1, take).squeeze(1)
        x0 = cumheights[:, :-1].gather(1, take).squeeze(1)
        derivatives = self._derivatives(widths, heights)
        d0 = derivatives[:, :-1].gather(1, take).squeeze(1)
        d1 = derivatives[:, 1:].gather(1, take).squeeze(1)
        return width, height, u0, x0, d0, d1

    def forward(self, u: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        u, cond, column = self._values_and_cond(u, cond)
        if torch.any((u < 0) | (u > 1)):
            raise ValueError("u must lie in [0, 1]")
        width, height, u0, x0, d0, d1 = self._selected(u, cond, inverse=False)
        t = ((u - u0) / width).clamp(0, 1)
        x, dx_dt = self._hermite(t, x0, height, width, d0, d1)
        logabsdet = torch.log(dx_dt.clamp_min(torch.finfo(dx_dt.dtype).tiny)) - torch.log(width)
        return self._restore(x, column), logabsdet

    def inverse(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        x, cond, column = self._values_and_cond(x, cond)
        if torch.any((x < 0) | (x > 1)):
            raise ValueError("x must lie in [0, 1]")
        width, height, u0, x0, d0, d1 = self._selected(x, cond, inverse=True)
        low, high = torch.zeros_like(x), torch.ones_like(x)
        for _ in range(self.inverse_iterations):
            mid = (low + high) * 0.5
            x_mid, _ = self._hermite(mid, x0, height, width, d0, d1)
            low = torch.where(x_mid < x, mid, low)
            high = torch.where(x_mid < x, high, mid)
        t = (low + high) * 0.5
        # One Newton correction makes autograd follow the implicit root, while
        # bisection supplies the globally safe initial value.
        x_t, dx_dt = self._hermite(t, x0, height, width, d0, d1)
        t = (t - (x_t - x) / dx_dt.clamp_min(torch.finfo(x.dtype).tiny)).clamp(0, 1)
        _, dx_dt = self._hermite(t, x0, height, width, d0, d1)
        u = u0 + width * t
        u = u.clamp(0.0,1.0,)
        logabsdet = torch.log(dx_dt.clamp_min(torch.finfo(dx_dt.dtype).tiny)) - torch.log(width)
        return self._restore(u, column), logabsdet
