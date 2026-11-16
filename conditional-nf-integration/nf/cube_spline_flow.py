import torch
import torch.nn as nn
from nf.logit_rotation import LogitRotation

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

        # identity-ish initialization
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


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


class RationalQuadraticSplineCoupling(nn.Module):
    """
    Smooth cube-native rational-quadratic spline coupling.

    Maps [0,1]^d -> [0,1]^d.
    No Gaussian base. No sigmoid.
    """

    def __init__(
        self,
        dim,
        mask,
        hidden=128,
        n_bins=16,
        min_height=1e-3,
        min_derivative=1e-3,
    ):
        super().__init__()

        self.dim = dim
        self.n_bins = n_bins
        self.min_height = min_height
        self.min_derivative = min_derivative

        self.register_buffer("mask", mask.float())

        free_idx = (mask == 0).nonzero(as_tuple=False).flatten()
        self.register_buffer("free_idx", free_idx)
        self.n_free = free_idx.numel()

        # heights: n_bins
        # derivatives: n_bins + 1
        out_dim = self.n_free * (n_bins + n_bins + 1)

        self.net = MLP(dim, out_dim, hidden)

    def _params(self, x_masked):
        raw = self.net(x_masked)
        raw = raw.view(-1, self.n_free, 2 * self.n_bins + 1)

        raw_heights = raw[:, :, :self.n_bins]
        raw_derivatives = raw[:, :, self.n_bins:]

        heights = torch.softmax(raw_heights, dim=-1)
        heights = self.min_height + (1.0 - self.n_bins * self.min_height) * heights
        heights = heights / heights.sum(dim=-1, keepdim=True)

        derivatives = self.min_derivative + torch.nn.functional.softplus(raw_derivatives)

        return heights, derivatives

    def forward(self, u):
        eps = 1e-6
        u = u.clamp(eps, 1.0 - eps)

        u_masked = u * self.mask
        heights, derivatives = self._params(u_masked)

        u_free = u[:, self.free_idx]

        K = self.n_bins
        width = 1.0 / K

        bin_idx = torch.floor(u_free * K).long().clamp(0, K - 1)

        cum_heights = torch.cat(
            [
                torch.zeros(heights.shape[0], self.n_free, 1, device=u.device, dtype=u.dtype),
                torch.cumsum(heights, dim=-1),
            ],
            dim=-1,
        )

        h = torch.gather(heights, 2, bin_idx.unsqueeze(-1)).squeeze(-1)
        y0 = torch.gather(cum_heights, 2, bin_idx.unsqueeze(-1)).squeeze(-1)

        d0 = torch.gather(derivatives, 2, bin_idx.unsqueeze(-1)).squeeze(-1)
        d1 = torch.gather(derivatives, 2, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

        x0 = bin_idx.to(u.dtype) * width
        theta = (u_free - x0) / width

        delta = h / width
        a = d0 + d1 - 2.0 * delta

        numerator = h * (delta * theta**2 + d0 * theta * (1.0 - theta))
        denominator = delta + a * theta * (1.0 - theta)

        v_free = y0 + numerator / denominator

        deriv_num = delta**2 * (
            d1 * theta**2
            + 2.0 * delta * theta * (1.0 - theta)
            + d0 * (1.0 - theta) ** 2
        )
        deriv = deriv_num / (denominator**2)

        v = u.clone()
        v[:, self.free_idx] = v_free

        logdet = torch.log(deriv + 1e-12).sum(dim=-1)

        return v, logdet

    def inverse(self, v):
        eps = 1e-6
        v = v.clamp(eps, 1.0 - eps)

        v_masked = v * self.mask
        heights, derivatives = self._params(v_masked)

        v_free = v[:, self.free_idx]

        K = self.n_bins
        width = 1.0 / K

        cum_heights = torch.cat(
            [
                torch.zeros(heights.shape[0], self.n_free, 1, device=v.device, dtype=v.dtype),
                torch.cumsum(heights, dim=-1),
            ],
            dim=-1,
        )

        bin_idx = torch.searchsorted(
            cum_heights.contiguous(),
            v_free.unsqueeze(-1).contiguous(),
            right=True,
        ).squeeze(-1) - 1

        bin_idx = bin_idx.clamp(0, K - 1)

        h = torch.gather(heights, 2, bin_idx.unsqueeze(-1)).squeeze(-1)
        y0 = torch.gather(cum_heights, 2, bin_idx.unsqueeze(-1)).squeeze(-1)

        d0 = torch.gather(derivatives, 2, bin_idx.unsqueeze(-1)).squeeze(-1)
        d1 = torch.gather(derivatives, 2, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

        delta = h / width
        a = d0 + d1 - 2.0 * delta

        y = v_free - y0

        A = y * a + h * (delta - d0)
        B = h * d0 - y * a
        C = -delta * y

        discriminant = B**2 - 4.0 * A * C
        discriminant = torch.clamp(discriminant, min=1e-12)

        theta = (2.0 * C) / (-B - torch.sqrt(discriminant))
        theta = theta.clamp(0.0, 1.0)

        x0 = bin_idx.to(v.dtype) * width
        u_free = x0 + theta * width

        denominator = delta + a * theta * (1.0 - theta)

        deriv_num = delta**2 * (
            d1 * theta**2
            + 2.0 * delta * theta * (1.0 - theta)
            + d0 * (1.0 - theta) ** 2
        )
        deriv = deriv_num / (denominator**2)

        u = v.clone()
        u[:, self.free_idx] = u_free

        logdet = -torch.log(deriv + 1e-12).sum(dim=-1)

        return u, logdet


class CubeSplineFlow(nn.Module):
    """
    Cube-native smooth spline flow:

        u ~ Uniform([0,1]^d)
        x = h_theta(u) in [0,1]^d
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
        use_rotation=False,
        rotation_angle=0.0,
    ):
        super().__init__()

        self.dim = dim
        self.n_blocks = n_blocks
        self.perm_sequence = perm_sequence
        self.use_rotation = use_rotation
        self.rotation_angle = rotation_angle

        layers = []

        g = torch.Generator()
        g.manual_seed(seed)

        if perm_sequence is not None:
            if len(perm_sequence) != n_blocks:
                raise ValueError(
                    f"perm_sequence length must equal n_blocks. "
                    f"Got len={len(perm_sequence)}, n_blocks={n_blocks}."
                )

        for k in range(n_blocks):
            mask = torch.zeros(dim)

            # Fixed mask; permutation layers control coordinate order.
            mask[::2] = 1.0

            layers.append(
                RationalQuadraticSplineCoupling(
                    dim=dim,
                    mask=mask,
                    hidden=hidden,
                    n_bins=n_bins,
                )
            )

            # --------------------------------------------------
            # Deterministic permutation sequence
            # --------------------------------------------------
            if perm_sequence is not None:
                digit = perm_sequence[k]

                if digit == "0":
                    perm = torch.arange(dim)
                elif digit == "1":
                    perm = torch.arange(dim - 1, -1, -1)
                else:
                    raise ValueError(
                        f"Invalid digit in perm_sequence: {digit}. "
                        "Use only '0' or '1'."
                    )

            # --------------------------------------------------
            # Original behavior
            # --------------------------------------------------
            else:
                if permute == "reverse":
                    perm = torch.arange(dim - 1, -1, -1)
                elif permute == "random":
                    perm = torch.randperm(dim, generator=g)
                elif permute == "identity":
                    perm = torch.arange(dim)
                else:
                    raise ValueError(
                        "permute must be 'reverse', 'random', or 'identity'"
                    )

            layers.append(Permute(perm))
            if self.use_rotation:
                if dim != 2:
                    raise ValueError("LogitRotation currently supports dim=2 only.")
                layers.append(LogitRotation(rotation_angle))

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