import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, n_layers=3):
        super().__init__()

        layers = []
        last = in_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(last, hidden))
            layers.append(nn.ReLU())
            last = hidden

        layers.append(nn.Linear(last, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConditionalRQFlow1D(nn.Module):
    """
    Conditional 1D rational-quadratic spline flow on [0,1].

    Base:
        u ~ Uniform(0,1)

    Transform:
        x = T_theta(u | c)

    Density:
        q_theta(x | c) = | d T^{-1}(x|c) / dx |
    """

    def __init__(
        self,
        cond_dim=2,
        hidden=128,
        n_bins=64,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.hidden = hidden
        self.n_bins = n_bins

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # widths: K, heights: K, derivatives: K+1
        out_dim = 3 * n_bins + 1

        self.net = MLP(
            in_dim=cond_dim,
            out_dim=out_dim,
            hidden=hidden,
            n_layers=3,
        )

    def _normalize_condition(self, cond):
        """
        Use [mu, log(sigma)] instead of [mu, sigma].
        This usually makes training more stable.
        """
        mu = cond[:, 0:1]
        sigma = cond[:, 1:2]

        sigma = torch.clamp(sigma, min=1e-6)

        cond_norm = torch.cat(
            [
                mu,
                torch.log(sigma),
            ],
            dim=1,
        )

        return cond_norm

    def _params(self, cond):
        """
        Convert network output into valid spline parameters.
        """

        cond = self._normalize_condition(cond)
        raw = self.net(cond)

        K = self.n_bins

        raw_widths = raw[:, :K]
        raw_heights = raw[:, K:2 * K]
        raw_derivatives = raw[:, 2 * K:]

        widths = F.softmax(raw_widths, dim=-1)
        heights = F.softmax(raw_heights, dim=-1)

        widths = (
            self.min_bin_width
            + (1.0 - self.min_bin_width * K) * widths
        )

        heights = (
            self.min_bin_height
            + (1.0 - self.min_bin_height * K) * heights
        )

        derivatives = F.softplus(raw_derivatives) + self.min_derivative

        cumwidths = torch.cumsum(widths, dim=-1)
        cumheights = torch.cumsum(heights, dim=-1)

        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)

        # Numerical safety
        cumwidths[:, 0] = 0.0
        cumwidths[:, -1] = 1.0
        cumheights[:, 0] = 0.0
        cumheights[:, -1] = 1.0

        return widths, heights, derivatives, cumwidths, cumheights

    @staticmethod
    def _gather(params, bin_idx):
        return params.gather(1, bin_idx.unsqueeze(1)).squeeze(1)

    def _bin_indices(self, inputs, cumvalues):
        """
        inputs: (N,)
        cumvalues: (N, K+1)

        Return bin indices in {0,...,K-1}.
        """
        K = cumvalues.shape[1] - 1

        bin_idx = torch.sum(inputs.unsqueeze(1) >= cumvalues[:, 1:-1], dim=1)
        bin_idx = torch.clamp(bin_idx, min=0, max=K - 1)

        return bin_idx

    def _spline_forward(self, u, cond):
        """
        Forward transform:
            u -> x

        Returns:
            x, log|dx/du|
        """

        eps = 1e-6

        u = torch.clamp(u.reshape(-1), eps, 1.0 - eps)

        widths, heights, derivatives, cumwidths, cumheights = self._params(cond)

        bin_idx = self._bin_indices(u, cumwidths)

        x_k = self._gather(cumwidths, bin_idx)
        y_k = self._gather(cumheights, bin_idx)

        w_k = self._gather(widths, bin_idx)
        h_k = self._gather(heights, bin_idx)

        d_k = self._gather(derivatives, bin_idx)
        d_kp1 = self._gather(derivatives, bin_idx + 1)

        theta = (u - x_k) / w_k
        theta = torch.clamp(theta, eps, 1.0 - eps)

        s_k = h_k / w_k

        numerator = h_k * (
            s_k * theta ** 2
            + d_k * theta * (1.0 - theta)
        )

        denominator = (
            s_k
            + (d_kp1 + d_k - 2.0 * s_k)
            * theta
            * (1.0 - theta)
        )

        x = y_k + numerator / denominator

        derivative_numer = s_k ** 2 * (
            d_kp1 * theta ** 2
            + 2.0 * s_k * theta * (1.0 - theta)
            + d_k * (1.0 - theta) ** 2
        )

        derivative_denom = denominator ** 2

        logabsdet = torch.log(derivative_numer + eps) - torch.log(derivative_denom + eps)

        return x.reshape(-1, 1), logabsdet.reshape(-1)

    def _spline_inverse(self, x, cond):
        """
        Inverse transform:
            x -> u

        Returns:
            u, log|dx/du| evaluated at u

        Then:
            log q(x|c) = - log|dx/du|
        """

        eps = 1e-6

        x = torch.clamp(x.reshape(-1), eps, 1.0 - eps)

        widths, heights, derivatives, cumwidths, cumheights = self._params(cond)

        bin_idx = self._bin_indices(x, cumheights)

        x_k = self._gather(cumwidths, bin_idx)
        y_k = self._gather(cumheights, bin_idx)

        w_k = self._gather(widths, bin_idx)
        h_k = self._gather(heights, bin_idx)

        d_k = self._gather(derivatives, bin_idx)
        d_kp1 = self._gather(derivatives, bin_idx + 1)

        s_k = h_k / w_k

        # Solve rational quadratic equation for theta.
        # Let t = (x - y_k) / h_k.
        t = (x - y_k) / h_k
        t = torch.clamp(t, eps, 1.0 - eps)

        a_tmp = d_kp1 + d_k - 2.0 * s_k

        a = (s_k - d_k) + t * a_tmp
        b = d_k - t * a_tmp
        c = -t * s_k

        discriminant = b ** 2 - 4.0 * a * c
        discriminant = torch.clamp(discriminant, min=eps)

        sqrt_disc = torch.sqrt(discriminant)

        theta_quadratic = (-b + sqrt_disc) / (2.0 * a + eps)
        theta_linear = -c / (b + eps)

        theta = torch.where(
            torch.abs(a) > 1e-8,
            theta_quadratic,
            theta_linear,
        )

        theta = torch.clamp(theta, eps, 1.0 - eps)

        u = x_k + theta * w_k

        denominator = (
            s_k
            + (d_kp1 + d_k - 2.0 * s_k)
            * theta
            * (1.0 - theta)
        )

        derivative_numer = s_k ** 2 * (
            d_kp1 * theta ** 2
            + 2.0 * s_k * theta * (1.0 - theta)
            + d_k * (1.0 - theta) ** 2
        )

        derivative_denom = denominator ** 2

        logabsdet = torch.log(derivative_numer + eps) - torch.log(derivative_denom + eps)

        return u.reshape(-1, 1), logabsdet.reshape(-1)

    def forward(self, u, cond):
        """
        u -> x
        """
        if cond.shape[0] == 1 and u.shape[0] > 1:
            cond = cond.expand(u.shape[0], -1)

        return self._spline_forward(u, cond)

    def inverse(self, x, cond):
        """
        x -> u
        """
        if cond.shape[0] == 1 and x.shape[0] > 1:
            cond = cond.expand(x.shape[0], -1)

        return self._spline_inverse(x, cond)

    def log_prob(self, x, cond):
        """
        log q_theta(x | cond)

        Since base density is Uniform(0,1), log p_U(u)=0.
        Therefore:

            log q(x|c) = - log|dx/du|.
        """

        _, logabsdet_forward = self.inverse(x, cond)

        log_q = -logabsdet_forward

        return log_q

    def spline_regularization(self, cond):
        """
        Smoothness regularization for the RQ spline derivatives.

        Penalizes sharp jumps in neighboring log-derivatives.

        This helps reduce artificial bumps/spikes in q_theta.

        """
        _, _, derivatives, _, _ = self._params(cond)

        log_deriv = torch.log(derivatives + 1e-12)

        smooth_penalty = (
            (log_deriv[:, 1:] - log_deriv[:, :-1]) ** 2
        ).mean()

        return smooth_penalty

    def sample(self, n, cond, device=None):
        """
        Sample x ~ q_theta(x | cond).
        """

        if device is None:
            device = cond.device

        if cond.shape[0] == 1:
            cond = cond.expand(n, -1)

        u = torch.rand(n, 1, device=device)

        x, _ = self.forward(u, cond)

        return x