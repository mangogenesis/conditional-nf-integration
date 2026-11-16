import math
import torch
import torch.nn as nn

LOG_2PI = math.log(2.0 * math.pi)


def logit(x, eps=1e-6):
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


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


class CubeAffineCoupling(nn.Module):
    """
    Uniform base cube coupling:

        u in (0,1)^d
        y = logit(u_free)
        y' = exp(s) y + t
        v_free = sigmoid(y')

    This maps [0,1]^d -> [0,1]^d.
    """

    def __init__(self, dim, mask, hidden=128, clamp=2.0):
        super().__init__()
        self.dim = dim
        self.clamp = clamp
        self.register_buffer("mask", mask.float())

        self.st = MLP(dim, 2 * dim, hidden)

    def forward(self, u):
        eps = 1e-6
        u = u.clamp(eps, 1.0 - eps)

        u_masked = u * self.mask
        st = self.st(u_masked)
        s, t = st[:, :self.dim], st[:, self.dim:]

        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        s = self.clamp * torch.tanh(s / self.clamp)

        y = logit(u, eps=eps)
        yp = y * torch.exp(s) + t
        v = torch.sigmoid(yp)

        # keep masked coordinates unchanged
        v = u_masked + (1.0 - self.mask) * v

        # log dv/du for transformed dims:
        # y=logit(u): dy/du = 1/[u(1-u)]
        # yp=exp(s)y+t: dyp/dy=exp(s)
        # v=sigmoid(yp): dv/dyp = v(1-v)
        logdet_each = (
            s
            + torch.log(v.clamp(eps, 1 - eps) * (1 - v.clamp(eps, 1 - eps)))
            - torch.log(u * (1 - u))
        )
        logdet_each = logdet_each * (1.0 - self.mask)
        logdet = logdet_each.sum(dim=-1)

        return v, logdet

    def inverse(self, v):
        eps = 1e-6
        v = v.clamp(eps, 1.0 - eps)

        v_masked = v * self.mask
        st = self.st(v_masked)
        s, t = st[:, :self.dim], st[:, self.dim:]

        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        s = self.clamp * torch.tanh(s / self.clamp)

        yp = logit(v, eps=eps)
        y = (yp - t) * torch.exp(-s)
        u_free = torch.sigmoid(y)

        u = v_masked + (1.0 - self.mask) * u_free

        # log du/dv = - log dv/du
        logdet_each = (
            -s
            + torch.log(u.clamp(eps, 1 - eps) * (1 - u.clamp(eps, 1 - eps)))
            - torch.log(v * (1 - v))
        )
        logdet_each = logdet_each * (1.0 - self.mask)
        logdet = logdet_each.sum(dim=-1)

        return u, logdet


class CubeAffineFlow(nn.Module):
    """
    RealNVP-style affine coupling with uniform base:

        u ~ Uniform([0,1]^d)
        x = h(u) in [0,1]^d
    """

    def __init__(self, dim, n_blocks=8, hidden=128, permute="reverse", seed=0):
        super().__init__()
        self.dim = dim

        layers = []
        g = torch.Generator()
        g.manual_seed(seed)

        for _ in range(n_blocks):
            mask = torch.zeros(dim)
            mask[::2] = 1.0

            layers.append(CubeAffineCoupling(dim, mask, hidden))

            if permute == "reverse":
                perm = torch.arange(dim - 1, -1, -1)
            elif permute == "random":
                perm = torch.randperm(dim, generator=g)
            else:
                raise ValueError("permute must be reverse or random")

            layers.append(Permute(perm))

        self.layers = nn.ModuleList(layers)

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
