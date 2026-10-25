# nfmodel/flows/realnvp_nd.py
import math
import torch
import torch.nn as nn

LOG_2PI = math.log(2.0 * math.pi)

def standard_normal_logprob(z: torch.Tensor):
    # z: (B, D)
    return (-0.5 * (z**2 + LOG_2PI)).sum(dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class Permute(nn.Module):
    """
    Fixed permutation of dimensions. Invertible with zero logdet.
    
    """
    def __init__(self, perm: torch.Tensor):
        super().__init__()
        assert perm.ndim == 1
        self.register_buffer("perm", perm.long())
        inv = torch.empty_like(self.perm)
        inv[self.perm] = torch.arange(self.perm.numel(), device=self.perm.device)
        self.register_buffer("invperm", inv.long())

    def forward(self, x: torch.Tensor):
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)

    def inverse(self, y: torch.Tensor):
        return y[:, self.invperm], torch.zeros(y.shape[0], device=y.device)

class AffineCoupling(nn.Module):
    """
    RealNVP affine coupling:
      x_masked = x * mask
      s,t = NN(x_masked)
      y = x_masked + (1-mask) * (x * exp(s) + t)
    """
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 128, clamp: float = 2.0):
        super().__init__()
        assert mask.shape == (dim,)
        self.dim = dim
        self.register_buffer("mask", mask.float())
        self.st = MLP(in_dim=dim, out_dim=2 * dim, hidden=hidden)
        self.clamp = clamp

    def forward(self, x: torch.Tensor):
        x_masked = x * self.mask
        st = self.st(x_masked)
        s, t = st[:, :self.dim], st[:, self.dim:]

        # only transform unmasked dims
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        # stabilize scale
        s = self.clamp * torch.tanh(s / self.clamp)

        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = s.sum(dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor):
        y_masked = y * self.mask
        st = self.st(y_masked)
        s, t = st[:, :self.dim], st[:, self.dim:]

        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        s = self.clamp * torch.tanh(s / self.clamp)

        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = (-s).sum(dim=-1)
        return x, logdet

class RealNVP(nn.Module):
    """
    RealNVP with alternating masks + permutation layers.
    """
    def __init__(
        self,
        dim: int,
        n_blocks: int = 8,
        hidden: int = 128,
        permute: str = "reverse",  
        seed: int = 0,
    ):
        super().__init__()
        self.dim = dim

        # [Coupling, Permute, Coupling, Permute, ...]
        transforms = []

        g = torch.Generator()
        g.manual_seed(seed)

        for k in range(n_blocks):
            # Alternating binary mask: 0101... / 1010...
            mask = torch.zeros(dim)
            mask[k % 2 :: 2] = 1.0

            transforms.append(AffineCoupling(dim=dim, mask=mask, hidden=hidden))

            if permute == "reverse":
                perm = torch.arange(dim - 1, -1, -1)
            elif permute == "random":
                perm = torch.randperm(dim, generator=g)
            else:
                raise ValueError("permute must be 'reverse' or 'random'")

            transforms.append(Permute(perm))

        self.transforms = nn.ModuleList(transforms)

    def fwd(self, z: torch.Tensor):
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device)
        for tr in self.transforms:
            x, ld = tr.forward(x)
            logdet += ld
        return x, logdet

    def inv(self, x: torch.Tensor):
        z = x
        logdet = torch.zeros(x.shape[0], device=x.device)
        for tr in reversed(self.transforms):
            z, ld = tr.inverse(z)
            logdet += ld
        return z, logdet

    def log_prob(self, x: torch.Tensor):
        z, logdet = self.inv(x)
        return standard_normal_logprob(z) + logdet

    def sample(self, n: int, device="mps" if torch.backends.mps.is_available() else "cpu"):
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.fwd(z)
        return x