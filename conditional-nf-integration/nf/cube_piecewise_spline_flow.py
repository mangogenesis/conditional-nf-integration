import torch
import torch.nn as nn


# =========================================
# MLP
# =========================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

        # identity-like init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# =========================================
# Permutation
# =========================================

class Permute(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.register_buffer("perm", perm.long())

        inv = torch.empty_like(self.perm)
        inv[self.perm] = torch.arange(self.perm.numel(), device=self.perm.device)
        self.register_buffer("invperm", inv.long())

    def forward(self, x):
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def inverse(self, y):
        return y[:, self.invperm], torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)


# =========================================
# Piecewise linear spline coupling
# =========================================

class PiecewiseLinearSplineCoupling(nn.Module):
    """
    Cube-native piecewise linear spline:
    u ∈ [0,1]^d → v ∈ [0,1]^d
    """

    def __init__(self, dim, mask, hidden=128, n_bins=16, min_height=1e-2):
        super().__init__()

        self.dim = dim
        self.n_bins = n_bins
        self.min_height = min_height

        self.register_buffer("mask", mask.float())

        free_idx = (mask == 0).nonzero(as_tuple=False).flatten()
        self.register_buffer("free_idx", free_idx)
        self.n_free = free_idx.numel()

        self.net = MLP(dim, self.n_free * n_bins, hidden)

    def _heights(self, x_masked):
        logits = self.net(x_masked)
        logits = logits.view(-1, self.n_free, self.n_bins)

        heights = torch.softmax(logits, dim=-1)
        heights = self.min_height + (1.0 - self.n_bins * self.min_height) * heights
        heights = heights / heights.sum(dim=-1, keepdim=True)

        return heights

    def forward(self, u):
        eps = 1e-6
        u = u.clamp(eps, 1.0 - eps)

        u_masked = u * self.mask
        heights = self._heights(u_masked)

        u_free = u[:, self.free_idx]

        K = self.n_bins
        width = 1.0 / K

        bin_idx = torch.floor(u_free * K).long().clamp(0, K - 1)

        cum_heights = torch.cat([
            torch.zeros(heights.shape[0], self.n_free, 1, device=u.device, dtype=u.dtype),
            torch.cumsum(heights, dim=-1),
        ], dim=-1)

        h_k = torch.gather(heights, 2, bin_idx.unsqueeze(-1)).squeeze(-1)
        y_k = torch.gather(cum_heights, 2, bin_idx.unsqueeze(-1)).squeeze(-1)

        x_left = bin_idx.to(u.dtype) * width
        slope = h_k / width

        v_free = y_k + slope * (u_free - x_left)

        v = u.clone()
        v[:, self.free_idx] = v_free

        logdet = torch.log(slope + 1e-12).sum(dim=-1)

        return v, logdet

    def inverse(self, v):
        eps = 1e-6
        v = v.clamp(eps, 1.0 - eps)

        v_masked = v * self.mask
        heights = self._heights(v_masked)

        v_free = v[:, self.free_idx]

        K = self.n_bins
        width = 1.0 / K

        cum_heights = torch.cat([
            torch.zeros(heights.shape[0], self.n_free, 1, device=v.device, dtype=v.dtype),
            torch.cumsum(heights, dim=-1),
        ], dim=-1)

        bin_idx = torch.searchsorted(
            cum_heights.contiguous(),
            v_free.unsqueeze(-1).contiguous(),
            right=True,
        ).squeeze(-1) - 1

        bin_idx = bin_idx.clamp(0, K - 1)

        h_k = torch.gather(heights, 2, bin_idx.unsqueeze(-1)).squeeze(-1)
        y_k = torch.gather(cum_heights, 2, bin_idx.unsqueeze(-1)).squeeze(-1)

        x_left = bin_idx.to(v.dtype) * width
        slope = h_k / width

        u_free = x_left + (v_free - y_k) / (slope + 1e-12)

        u = v.clone()
        u[:, self.free_idx] = u_free

        logdet = -torch.log(slope + 1e-12).sum(dim=-1)

        return u, logdet


# =========================================
# Full flow
# =========================================

class CubePiecewiseSplineFlow(nn.Module):
    """
    Cube-native flow:
    u ~ Uniform([0,1]^d)
    x = h(u)
    """

    def __init__(
        self,
        dim,
        n_blocks=4,
        hidden=128,
        n_bins=16,
        permute="reverse",
        seed=0,
        perm_sequence=None,
    ):
        super().__init__()

        self.dim = dim
        layers = []

        g = torch.Generator()
        g.manual_seed(seed)

        if perm_sequence is not None:
            if len(perm_sequence) != n_blocks:
                raise ValueError(
                    f"perm_sequence length must equal n_blocks. "
                    f"Got len={len(perm_sequence)}, n_blocks={n_blocks}."
                )

        for block_idx in range(n_blocks):

            # Fixed mask. Permutation layers decide which coordinate ordering is used.
            mask = torch.zeros(dim)
            mask[::2] = 1.0

            layers.append(
                PiecewiseLinearSplineCoupling(
                    dim=dim,
                    mask=mask,
                    hidden=hidden,
                    n_bins=n_bins,
                )
            )

            # -----------------------------------------
            # Deterministic permutation sequence
            # -----------------------------------------
            if perm_sequence is not None:
                digit = perm_sequence[block_idx]

                if digit == "0":
                    perm = torch.arange(dim)
                elif digit == "1":
                    perm = torch.arange(dim - 1, -1, -1)
                else:
                    raise ValueError(
                        f"Invalid digit in perm_sequence: {digit}. "
                        "Use only '0' or '1'."
                    )

            # -----------------------------------------
            # Original behavior
            # -----------------------------------------
            else:
                if permute == "reverse":
                    perm = torch.arange(dim - 1, -1, -1)
                elif permute == "random":
                    perm = torch.randperm(dim, generator=g)
                elif permute == "identity":
                    perm = torch.arange(dim)
                else:
                    raise ValueError("permute must be reverse, random, or identity")

            layers.append(Permute(perm))

        self.layers = nn.ModuleList(layers)
    def print_permutations(self):
        print("Permutation layers:")
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Permute):
                print(
                    f"  layer {i}: "
                    f"perm={layer.perm.detach().cpu().tolist()}, "
                    f"invperm={layer.invperm.detach().cpu().tolist()}"
                )

    def fwd(self, u):
        x = u
        logdet = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)

        for layer in self.layers:
            x, ld = layer.forward(x)
            logdet = logdet + ld

        return x, logdet

    def inv(self, x):
        u = x
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for layer in reversed(self.layers):
            u, ld = layer.inverse(u)
            logdet = logdet + ld

        return u, logdet

    def sample(self, n, device=None):
        if device is None:
            device = next(self.parameters()).device

        u = torch.rand(n, self.dim, device=device)
        x, _ = self.fwd(u)
        return x

    def log_prob(self, x):
        _, inv_logdet = self.inv(x)
        return inv_logdet
