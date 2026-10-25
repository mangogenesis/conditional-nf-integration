import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Small helper network
# ============================================================

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Bin utilities
# ============================================================

def _normalize_bins(
    raw,
    min_bin=1e-3,
):
    """
    raw: (..., K)

    Returns positive bins summing to 1.
    """

    K = raw.shape[-1]

    if min_bin * K >= 1.0:
        raise ValueError(
            "min_bin * n_bins must be < 1"
        )

    probs = F.softmax(
        raw,
        dim=-1,
    )

    bins = (
        min_bin
        +
        (1.0 - min_bin * K) * probs
    )

    return bins


def _cum_bins(bins):
    """
    bins: (..., K)

    returns:
        (..., K+1)

    Starts at 0 and ends at 1.
    """

    zeros = torch.zeros_like(
        bins[..., :1]
    )

    cumulative = torch.cumsum(
        bins,
        dim=-1,
    )

    return torch.cat(
        [
            zeros,
            cumulative,
        ],
        dim=-1,
    )


def _select_bin(
    values,
    knots,
):
    """
    values: (N,)
    knots:  (N,K+1)

    returns:
        bin index in [0,K-1]
    """

    idx = torch.sum(
        values[:, None]
        >= knots[:, 1:-1],
        dim=-1,
    )

    K = (
        knots.shape[-1]
        - 1
    )

    return idx.clamp(
        min=0,
        max=K - 1,
    )


def _gather(
    values,
    idx,
):
    """
    values: (N,K) or (N,K+1)
    idx:    (N,)
    """

    return values.gather(
        1,
        idx[:, None],
    ).squeeze(1)


# ============================================================
# Monotonicity-preserving cubic derivatives
# ============================================================

def _monotone_derivatives(
    widths,
    heights,
):
    """
    Construct monotone cubic Hermite knot derivatives.

    widths:  (N,K)
    heights: (N,K)

    The secant slope of each bin is

        m_k = height_k / width_k

    Interior knot derivatives are chosen using the harmonic
    mean of neighboring secant slopes.

    This avoids arbitrary learned derivative values and
    strongly suppresses cubic overshoot.
    """

    eps = 1e-12

    secants = (
        heights
        / (widths + eps)
    )

    N, K = secants.shape

    derivatives = torch.zeros(
        N,
        K + 1,
        device=widths.device,
        dtype=widths.dtype,
    )

    # --------------------------------------------------------
    # Boundary derivatives
    # --------------------------------------------------------

    derivatives[:, 0] = (
        secants[:, 0]
    )

    derivatives[:, -1] = (
        secants[:, -1]
    )

    # --------------------------------------------------------
    # Interior derivatives
    # --------------------------------------------------------

    if K > 1:

        m_left = (
            secants[:, :-1]
        )

        m_right = (
            secants[:, 1:]
        )

        derivatives[:, 1:-1] = (
            2.0
            * m_left
            * m_right
            /
            (
                m_left
                + m_right
                + eps
            )
        )

    return derivatives


# ============================================================
# Cubic Hermite forward
# ============================================================

def cubic_forward_1d(
    u,
    widths,
    heights,
    derivatives,
):
    """
    Forward cubic Hermite spline.

    u:           (N,)
    widths:      (N,K)
    heights:     (N,K)
    derivatives: (N,K+1)

    Returns:
        x
        log |dx/du|
    """

    eps = 1e-12

    x_knots = _cum_bins(
        widths
    )

    y_knots = _cum_bins(
        heights
    )

    idx = _select_bin(
        u,
        x_knots,
    )

    x0 = _gather(
        x_knots[:, :-1],
        idx,
    )

    y0 = _gather(
        y_knots[:, :-1],
        idx,
    )

    w = _gather(
        widths,
        idx,
    )

    h = _gather(
        heights,
        idx,
    )

    d0 = _gather(
        derivatives[:, :-1],
        idx,
    )

    d1 = _gather(
        derivatives[:, 1:],
        idx,
    )

    # --------------------------------------------------------
    # Local bin coordinate t in [0,1]
    # --------------------------------------------------------

    t = (
        (u - x0)
        / (w + eps)
    )

    t = t.clamp(
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Cubic Hermite basis
    # --------------------------------------------------------

    h00 = (
        2.0 * t**3
        - 3.0 * t**2
        + 1.0
    )

    h10 = (
        t**3
        - 2.0 * t**2
        + t
    )

    h01 = (
        -2.0 * t**3
        + 3.0 * t**2
    )

    h11 = (
        t**3
        - t**2
    )

    x = (
        h00 * y0
        + h10 * w * d0
        + h01 * (y0 + h)
        + h11 * w * d1
    )

    # --------------------------------------------------------
    # Derivative wrt t
    # --------------------------------------------------------

    dh00 = (
        6.0 * t**2
        - 6.0 * t
    )

    dh10 = (
        3.0 * t**2
        - 4.0 * t
        + 1.0
    )

    dh01 = (
        -6.0 * t**2
        + 6.0 * t
    )

    dh11 = (
        3.0 * t**2
        - 2.0 * t
    )

    dx_dt = (
        dh00 * y0
        + dh10 * w * d0
        + dh01 * (y0 + h)
        + dh11 * w * d1
    )

    dx_du = (
        dx_dt
        / (w + eps)
    )

    # --------------------------------------------------------
    # Important:
    # We do NOT clamp negative derivatives here.
    #
    # If monotonicity ever fails, we want to see it rather
    # than silently hiding the problem.
    # --------------------------------------------------------

    if torch.any(
        dx_du <= 0.0
    ):
        raise RuntimeError(
            "Non-positive spline derivative detected. "
            "Monotonicity has failed."
        )

    logdet = torch.log(
        dx_du
    )

    return (
        x,
        logdet,
    )


# ============================================================
# Cubic inverse
# ============================================================

def cubic_inverse_1d(
    x,
    widths,
    heights,
    derivatives,
    n_iter=20,
):
    """
    Inverse monotone cubic spline using bisection.

    x:           (N,)
    widths:      (N,K)
    heights:     (N,K)
    derivatives: (N,K+1)

    Returns:
        u
        log |du/dx|
    """

    x_knots = _cum_bins(
        widths
    )

    y_knots = _cum_bins(
        heights
    )

    idx = _select_bin(
        x,
        y_knots,
    )

    u0 = _gather(
        x_knots[:, :-1],
        idx,
    )

    w = _gather(
        widths,
        idx,
    )

    left = (
        u0.clone()
    )

    right = (
        u0 + w
    ).clone()

    # --------------------------------------------------------
    # Bisection
    # --------------------------------------------------------

    for _ in range(
        n_iter
    ):

        mid = (
            0.5
            * (
                left
                + right
            )
        )

        y_mid, _ = (
            cubic_forward_1d(
                mid,
                widths,
                heights,
                derivatives,
            )
        )

        go_right = (
            y_mid < x
        )

        left = torch.where(
            go_right,
            mid,
            left,
        )

        right = torch.where(
            go_right,
            right,
            mid,
        )

    u = (
        0.5
        * (
            left
            + right
        )
    )

    _, forward_logdet = (
        cubic_forward_1d(
            u,
            widths,
            heights,
            derivatives,
        )
    )

    inverse_logdet = (
        -forward_logdet
    )

    return (
        u,
        inverse_logdet,
    )


# ============================================================
# Conditional cubic coupling layer
# ============================================================

class ConditionalCubicCoupling2D(nn.Module):

    def __init__(
        self,
        cond_dim=3,
        hidden=128,
        n_bins=16,
        transform_index=1,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
    ):
        super().__init__()

        self.cond_dim = (
            cond_dim
        )

        self.hidden = (
            hidden
        )

        self.n_bins = (
            n_bins
        )

        self.transform_index = (
            transform_index
        )

        self.min_bin_width = (
            min_bin_width
        )

        self.min_bin_height = (
            min_bin_height
        )

        # ----------------------------------------------------
        # IMPORTANT CHANGE
        #
        # Network predicts ONLY:
        #
        #   K widths
        #   K heights
        #
        # Derivatives are determined analytically from
        # neighboring secant slopes.
        # ----------------------------------------------------

        out_dim = (
            2 * n_bins
        )

        self.net = MLP(
            in_dim=(
                1 + cond_dim
            ),
            out_dim=out_dim,
            hidden=hidden,
        )

    def _params(
        self,
        fixed,
        cond,
    ):

        inp = torch.cat(
            [
                fixed,
                cond,
            ],
            dim=-1,
        )

        raw = self.net(
            inp
        )

        K = self.n_bins

        raw_w = (
            raw[:, :K]
        )

        raw_h = (
            raw[:, K:2*K]
        )

        widths = _normalize_bins(
            raw_w,
            min_bin=(
                self.min_bin_width
            ),
        )

        heights = _normalize_bins(
            raw_h,
            min_bin=(
                self.min_bin_height
            ),
        )

        derivatives = (
            _monotone_derivatives(
                widths,
                heights,
            )
        )

        return (
            widths,
            heights,
            derivatives,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        cond,
    ):

        x1 = x[:, 0]
        x2 = x[:, 1]

        if (
            self.transform_index
            == 0
        ):

            # x2 fixed
            fixed = (
                x2[:, None]
            )

            (
                widths,
                heights,
                derivatives,
            ) = self._params(
                fixed,
                cond,
            )

            y1, logdet = (
                cubic_forward_1d(
                    x1,
                    widths,
                    heights,
                    derivatives,
                )
            )

            y2 = x2

        else:

            # x1 fixed
            fixed = (
                x1[:, None]
            )

            (
                widths,
                heights,
                derivatives,
            ) = self._params(
                fixed,
                cond,
            )

            y2, logdet = (
                cubic_forward_1d(
                    x2,
                    widths,
                    heights,
                    derivatives,
                )
            )

            y1 = x1

        y = torch.stack(
            [
                y1,
                y2,
            ],
            dim=-1,
        )

        return (
            y,
            logdet,
        )

    # ========================================================
    # Inverse
    # ========================================================

    def inverse(
        self,
        y,
        cond,
    ):

        y1 = y[:, 0]
        y2 = y[:, 1]

        if (
            self.transform_index
            == 0
        ):

            fixed = (
                y2[:, None]
            )

            (
                widths,
                heights,
                derivatives,
            ) = self._params(
                fixed,
                cond,
            )

            x1, logdet = (
                cubic_inverse_1d(
                    y1,
                    widths,
                    heights,
                    derivatives,
                )
            )

            x2 = y2

        else:

            fixed = (
                y1[:, None]
            )

            (
                widths,
                heights,
                derivatives,
            ) = self._params(
                fixed,
                cond,
            )

            x2, logdet = (
                cubic_inverse_1d(
                    y2,
                    widths,
                    heights,
                    derivatives,
                )
            )

            x1 = y1

        x = torch.stack(
            [
                x1,
                x2,
            ],
            dim=-1,
        )

        return (
            x,
            logdet,
        )


# ============================================================
# Full Conditional Cubic Flow
# ============================================================

class ConditionalCubicFlow2D(nn.Module):

    def __init__(
        self,
        cond_dim=3,
        hidden=128,
        n_bins=16,
        n_layers=6,
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.hidden = hidden
        self.n_bins = n_bins
        self.n_layers = n_layers

        layers = []

        for i in range(
            n_layers
        ):

            transform_index = (
                i % 2
            )

            layer = (
                ConditionalCubicCoupling2D(
                    cond_dim=cond_dim,
                    hidden=hidden,
                    n_bins=n_bins,
                    transform_index=(
                        transform_index
                    ),
                )
            )

            layers.append(
                layer
            )

        self.layers = (
            nn.ModuleList(
                layers
            )
        )

    # ========================================================
    # Base -> target
    # ========================================================

    def forward(
        self,
        u,
        cond,
    ):

        x = u

        total_logdet = (
            torch.zeros(
                u.shape[0],
                device=u.device,
                dtype=u.dtype,
            )
        )

        for layer in self.layers:

            x, logdet = (
                layer.forward(
                    x,
                    cond,
                )
            )

            total_logdet = (
                total_logdet
                + logdet
            )

        return (
            x,
            total_logdet,
        )

    # ========================================================
    # Target -> base
    # ========================================================

    def inverse(
        self,
        x,
        cond,
    ):

        u = x

        total_logdet = (
            torch.zeros(
                x.shape[0],
                device=x.device,
                dtype=x.dtype,
            )
        )

        for layer in reversed(
            self.layers
        ):

            u, logdet = (
                layer.inverse(
                    u,
                    cond,
                )
            )

            total_logdet = (
                total_logdet
                + logdet
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

        u, logdet_inv = (
            self.inverse(
                x,
                cond,
            )
        )

        inside = (
            (
                u >= 0.0
            )
            &
            (
                u <= 1.0
            )
        ).all(
            dim=-1
        )

        log_q = (
            logdet_inv
        )

        log_q = torch.where(
            inside,
            log_q,
            torch.full_like(
                log_q,
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
                "(1, cond_dim) or "
                "(n, cond_dim)"
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