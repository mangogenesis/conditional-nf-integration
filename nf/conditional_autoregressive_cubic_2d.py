import torch
import torch.nn as nn

from nf.conditional_cubic_2d import (
    MLP,
    _normalize_bins,
    _monotone_derivatives,
    cubic_forward_1d,
    cubic_inverse_1d,
)


# ============================================================
# One conditional 1D cubic spline
# ============================================================

class ConditionalCubicSpline1D(nn.Module):
    """
    One monotone cubic spline.

    The spline parameters are generated from a context vector.

    Network predicts:
        K widths
        K heights

    Knot derivatives are constructed from secant slopes using
    the monotonicity-preserving rule already defined in
    conditional_cubic_2d.py.
    """

    def __init__(
        self,
        context_dim,
        hidden=128,
        n_bins=16,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.hidden = hidden
        self.n_bins = n_bins

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height

        # K widths + K heights
        self.net = MLP(
            in_dim=context_dim,
            out_dim=2 * n_bins,
            hidden=hidden,
        )

    def _params(
        self,
        context,
    ):
        raw = self.net(
            context
        )

        K = self.n_bins

        raw_w = raw[:, :K]
        raw_h = raw[:, K:2*K]

        widths = _normalize_bins(
            raw_w,
            min_bin=self.min_bin_width,
        )

        heights = _normalize_bins(
            raw_h,
            min_bin=self.min_bin_height,
        )

        derivatives = _monotone_derivatives(
            widths,
            heights,
        )

        return (
            widths,
            heights,
            derivatives,
        )

    def forward(
        self,
        u,
        context,
    ):
        """
        u:       (N,)
        context: (N,context_dim)

        Returns:
            x
            log |dx/du|
        """

        (
            widths,
            heights,
            derivatives,
        ) = self._params(
            context
        )

        return cubic_forward_1d(
            u,
            widths,
            heights,
            derivatives,
        )

    def inverse(
        self,
        x,
        context,
    ):
        """
        Returns:
            u
            log |du/dx|
        """

        (
            widths,
            heights,
            derivatives,
        ) = self._params(
            context
        )

        return cubic_inverse_1d(
            x,
            widths,
            heights,
            derivatives,
        )


# ============================================================
# 2D autoregressive cubic flow
# ============================================================

class ConditionalAutoregressiveCubicFlow2D(nn.Module):
    """
    2D triangular autoregressive cubic flow.

    Ordering:

        x1 = T1(u1 | c)

        x2 = T2(u2 | x1, c)

    Therefore

        q(x1,x2 | c)
        =
        q(x1 | c)
        q(x2 | x1,c)

    The Jacobian is triangular.
    """

    def __init__(
        self,
        cond_dim=3,
        hidden=128,
        n_bins=16,
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.hidden = hidden
        self.n_bins = n_bins

        # ----------------------------------------------------
        # First coordinate:
        #
        # x1 = T1(u1 | c)
        # ----------------------------------------------------

        self.spline1 = ConditionalCubicSpline1D(
            context_dim=cond_dim,
            hidden=hidden,
            n_bins=n_bins,
        )

        # ----------------------------------------------------
        # Second coordinate:
        #
        # x2 = T2(u2 | x1,c)
        # ----------------------------------------------------

        self.spline2 = ConditionalCubicSpline1D(
            context_dim=cond_dim + 1,
            hidden=hidden,
            n_bins=n_bins,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        u,
        cond,
    ):
        """
        Base -> target.

        u:    (N,2)
        cond: (N,cond_dim)

        Returns:
            x
            log |dx/du|
        """

        u1 = u[:, 0]
        u2 = u[:, 1]

        # ----------------------------------------------------
        # First dimension
        # ----------------------------------------------------

        x1, logdet1 = self.spline1.forward(
            u1,
            cond,
        )

        # ----------------------------------------------------
        # Second dimension conditioned on x1
        # ----------------------------------------------------

        context2 = torch.cat(
            [
                x1[:, None],
                cond,
            ],
            dim=-1,
        )

        x2, logdet2 = self.spline2.forward(
            u2,
            context2,
        )

        x = torch.stack(
            [
                x1,
                x2,
            ],
            dim=-1,
        )

        total_logdet = (
            logdet1
            + logdet2
        )

        return (
            x,
            total_logdet,
        )

    # ========================================================
    # Inverse
    # ========================================================

    def inverse(
        self,
        x,
        cond,
    ):
        """
        Target -> base.

        Since the transformation is triangular:

            x1 = T1(u1|c)

        can be inverted first.

        Then x1 is already known, so:

            x2 = T2(u2|x1,c)

        can be inverted.
        """

        x1 = x[:, 0]
        x2 = x[:, 1]

        # ----------------------------------------------------
        # Recover u1
        # ----------------------------------------------------

        u1, logdet1 = self.spline1.inverse(
            x1,
            cond,
        )

        # ----------------------------------------------------
        # Recover u2
        #
        # The context uses x1, which is observed.
        # ----------------------------------------------------

        context2 = torch.cat(
            [
                x1[:, None],
                cond,
            ],
            dim=-1,
        )

        u2, logdet2 = self.spline2.inverse(
            x2,
            context2,
        )

        u = torch.stack(
            [
                u1,
                u2,
            ],
            dim=-1,
        )

        total_logdet = (
            logdet1
            + logdet2
        )

        return (
            u,
            total_logdet,
        )

    # ========================================================
    # Density
    # ========================================================

    def log_prob(
        self,
        x,
        cond,
    ):
        """
        Base distribution is Uniform([0,1]^2).

        Therefore

            log q(x|c)
            =
            log |du/dx|.
        """

        u, inverse_logdet = self.inverse(
            x,
            cond,
        )

        inside = (
            (u >= 0.0)
            &
            (u <= 1.0)
        ).all(
            dim=-1
        )

        log_q = torch.where(
            inside,
            inverse_logdet,
            torch.full_like(
                inverse_logdet,
                -float("inf"),
            ),
        )

        return log_q

    # ========================================================
    # Sampling
    # ========================================================

    @torch.no_grad()
    def sample(
        self,
        n,
        cond,
        device=None,
    ):
        if device is None:
            device = next(
                self.parameters()
            ).device

        if cond.shape[0] == 1:

            cond = cond.repeat(
                n,
                1,
            )

        elif cond.shape[0] != n:

            raise ValueError(
                "cond must have shape "
                "(1,cond_dim) or "
                "(n,cond_dim)"
            )

        u = torch.rand(
            n,
            2,
            device=device,
        )

        x, _ = self.forward(
            u,
            cond,
        )

        return x