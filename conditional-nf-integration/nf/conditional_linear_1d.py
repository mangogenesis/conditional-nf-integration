"""Conditional piecewise-linear flow on the unit interval."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConditionalLinearFlow1D(nn.Module):
    """A conditional monotone piecewise-linear map from ``[0, 1]`` to itself."""

    def __init__(
        self,
        cond_dim: int = 2,
        hidden: int = 128,
        n_bins: int = 8,
        min_bin_width: float = 1e-4,
        min_bin_height: float = 1e-4,
    ) -> None:
        super().__init__()
        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        if n_bins * min_bin_width >= 1 or n_bins * min_bin_height >= 1:
            raise ValueError("minimum bin sizes leave no room for the bins")

        self.cond_dim = cond_dim
        self.hidden = hidden
        self.n_bins = n_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * n_bins),
        )

    def _params(self, cond: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raw_widths, raw_heights = self.net(cond).chunk(2, dim=-1)
        widths = self.min_bin_width + (1 - self.n_bins * self.min_bin_width) * F.softmax(raw_widths, -1)
        heights = self.min_bin_height + (1 - self.n_bins * self.min_bin_height) * F.softmax(raw_heights, -1)
        cumwidths = F.pad(widths.cumsum(-1), (1, 0))
        cumheights = F.pad(heights.cumsum(-1), (1, 0))
        # Avoid accumulated softmax roundoff at the right boundary.
        cumwidths = torch.cat((cumwidths[:, :-1], torch.ones_like(cumwidths[:, -1:])), -1)
        cumheights = torch.cat((cumheights[:, :-1], torch.ones_like(cumheights[:, -1:])), -1)
        return widths, heights, cumwidths, cumheights

    @staticmethod
    def _values_and_cond(values: Tensor, cond: Tensor) -> tuple[Tensor, Tensor, bool]:
        if values.ndim == 1:
            flat, column = values, False
        elif values.ndim == 2 and values.shape[1] == 1:
            flat, column = values[:, 0], True
        else:
            raise ValueError("values must have shape (N,) or (N, 1)")
        if cond.ndim != 2:
            raise ValueError("cond must have shape (N, cond_dim) or (1, cond_dim)")
        if cond.shape[0] == 1 and flat.shape[0] != 1:
            cond = cond.expand(flat.shape[0], -1)
        elif cond.shape[0] != flat.shape[0]:
            raise ValueError("cond batch size must be 1 or match the values batch size")
        return flat, cond, column

    @staticmethod
    def _restore(values: Tensor, column: bool) -> Tensor:
        return values[:, None] if column else values

    def forward(self, u: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        u, cond, column = self._values_and_cond(u, cond)
        if torch.any((u < 0) | (u > 1)):
            raise ValueError("u must lie in [0, 1]")
        widths, heights, cumwidths, cumheights = self._params(cond)
        idx = (u[:, None] >= cumwidths[:, 1:]).sum(-1).clamp_max(self.n_bins - 1)
        take = idx[:, None]
        width = widths.gather(1, take).squeeze(1)
        height = heights.gather(1, take).squeeze(1)
        u0 = cumwidths[:, :-1].gather(1, take).squeeze(1)
        x0 = cumheights[:, :-1].gather(1, take).squeeze(1)
        x = x0 + (u - u0) * height / width
        logabsdet = torch.log(height) - torch.log(width)
        return self._restore(x, column), logabsdet

    def inverse(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        x, cond, column = self._values_and_cond(x, cond)
        if torch.any((x < 0) | (x > 1)):
            raise ValueError("x must lie in [0, 1]")
        widths, heights, cumwidths, cumheights = self._params(cond)
        idx = (x[:, None] >= cumheights[:, 1:]).sum(-1).clamp_max(self.n_bins - 1)
        take = idx[:, None]
        width = widths.gather(1, take).squeeze(1)
        height = heights.gather(1, take).squeeze(1)
        u0 = cumwidths[:, :-1].gather(1, take).squeeze(1)
        x0 = cumheights[:, :-1].gather(1, take).squeeze(1)
        u = u0 + (x - x0) * width / height
        logabsdet = torch.log(height) - torch.log(width)
        return self._restore(u, column), logabsdet

    def log_prob(self, x: Tensor, cond: Tensor) -> Tensor:
        """Return ``log q(x | cond)`` for a uniform base distribution."""
        _, forward_logabsdet = self.inverse(x, cond)
        return -forward_logabsdet

    def sample(self, n: int, cond: Tensor, device=None) -> Tensor:
        if cond.ndim != 2 or cond.shape[0] not in (1, n):
            raise ValueError("cond batch size must be 1 or n")
        if device is None:
            device = cond.device
        cond = cond.to(device=device)
        if cond.shape[0] == 1:
            cond = cond.expand(n, -1)
        u = torch.rand(n, 1, device=device, dtype=cond.dtype)
        return self.forward(u, cond)[0]