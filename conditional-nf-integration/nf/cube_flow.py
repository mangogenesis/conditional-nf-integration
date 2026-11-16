import torch
import torch.nn as nn

from nf.flows.realnvp_nd import RealNVP


def logit(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)


# class CubeRealNVP(nn.Module):
#     """
#     Flow on (0,1)^d:
#         z ~ N(0,I) -> y in R^d via RealNVP -> x = sigmoid(y)
#     """
#     def __init__(self, dim, n_blocks=8, hidden=128, permute="reverse", seed=0):
#         super().__init__()
#         self.dim = dim
#         self.flow = RealNVP(
#             dim=dim,
#             n_blocks=n_blocks,
#             hidden=hidden,
#             permute=permute,
#             seed=seed,
#         )

#     def sample(self, n, device=None):
#         if device is None:
#             device = next(self.parameters()).device
#         z = torch.randn(n, self.dim, device=device)
#         y, _ = self.flow.fwd(z)
#         x = torch.sigmoid(y)
#         return x

#     def log_prob(self, x, eps=1e-6):
#         x = x.clamp(eps, 1 - eps)
#         y = logit(x, eps=eps)

#         # log q_Y(y)
#         log_qy = self.flow.log_prob(y)

#         # y = logit(x), so |dy/dx| = 1 / (x(1-x))
#         logdet_logit = -torch.log(x * (1 - x)).sum(dim=-1)

#         return log_qy + logdet_logit

class CubeRealNVP(nn.Module):
    """
    Flow on (0,1)^d:
        z ~ N(0,I) -> y in R^d via RealNVP -> x = sigmoid(y)

    Optional mixture floor:
        q_mix(x) = (1 - eps_mix) q_nf(x) + eps_mix * Uniform([0,1]^d)
                 = (1 - eps_mix) q_nf(x) + eps_mix
    """

    def __init__(self, dim, n_blocks=8, hidden=128, permute="reverse", seed=0, eps_mix=0.01):
        super().__init__()
        self.dim = dim
        self.eps_mix = eps_mix

        self.flow = RealNVP(
            dim=dim,
            n_blocks=n_blocks,
            hidden=hidden,
            permute=permute,
            seed=seed,
        )

    def sample_nf(self, n, device=None):
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(n, self.dim, device=device)
        y, _ = self.flow.fwd(z)
        x = torch.sigmoid(y)
        return x

    def sample(self, n, device=None):
        """
        Sample from mixture:
            with prob 1 - eps_mix: NF
            with prob eps_mix: uniform
        """
        if device is None:
            device = next(self.parameters()).device

        x_nf = self.sample_nf(n, device=device)
        x_uni = torch.rand(n, self.dim, device=device)

        mask = (torch.rand(n, 1, device=device) < self.eps_mix).float()

        x = (1.0 - mask) * x_nf + mask * x_uni
        return x

    def log_prob_nf(self, x, eps=1e-6):
        x = x.clamp(eps, 1 - eps)
        y = logit(x, eps=eps)

        log_qy = self.flow.log_prob(y)

        logdet_logit = -torch.log(x * (1 - x)).sum(dim=-1)

        return log_qy + logdet_logit

    def log_prob(self, x, eps=1e-6):
        """
        log q_mix(x) = log((1-eps_mix) q_nf(x) + eps_mix)
        """
        log_q_nf = self.log_prob_nf(x, eps=eps)

        log_a = torch.log(torch.tensor(1.0 - self.eps_mix, device=x.device, dtype=x.dtype)) + log_q_nf
        log_b = torch.log(torch.tensor(self.eps_mix, device=x.device, dtype=x.dtype))

        return torch.logaddexp(log_a, log_b)