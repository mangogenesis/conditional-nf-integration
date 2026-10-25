import torch
import torch.nn as nn


class LogitRotation(nn.Module):
    """
    Rotation layer on the unit cube.

    x in (0, 1)^2
      -> z = logit(x) in R^2
      -> z_rot = R z
      -> y = sigmoid(z_rot) in (0, 1)^2

    This keeps the support inside the unit square.
    """

    def __init__(self, theta, eps=1e-6):
        super().__init__()
        self.theta = float(theta)
        self.eps = eps

    def rotation_matrix(self, device, dtype):
        theta = torch.tensor(self.theta, device=device, dtype=dtype)
        c = torch.cos(theta)
        s = torch.sin(theta)

        R = torch.stack(
            [
                torch.stack([c, -s]),
                torch.stack([s, c]),
            ]
        )
        return R

    def _logit(self, x):
        x = x.clamp(self.eps, 1.0 - self.eps)
        return torch.log(x) - torch.log1p(-x)

    def forward(self, x):
        """
        Forward map x -> y.
        Returns:
            y, logdet_forward
        """
        x = x.clamp(self.eps, 1.0 - self.eps)

        z = self._logit(x)
        R = self.rotation_matrix(x.device, x.dtype)

        z_rot = z @ R.T
        y = torch.sigmoid(z_rot)

        # log |dz/dx|
        logdet_logit = -torch.log(x) - torch.log1p(-x)

        # log |dy/dz_rot|
        logdet_sigmoid = torch.log(y) + torch.log1p(-y)

        # det(R) = 1
        logdet = (logdet_logit + logdet_sigmoid).sum(dim=-1)

        return y, logdet

    def inverse(self, y):
        """
        Inverse map y -> x.
        Returns:
            x, logdet_inverse
        """
        y = y.clamp(self.eps, 1.0 - self.eps)

        z_rot = self._logit(y)
        R = self.rotation_matrix(y.device, y.dtype)

        # inverse of z_rot = z @ R.T is z = z_rot @ R
        z = z_rot @ R
        x = torch.sigmoid(z)

        # log |dz_rot/dy|
        logdet_logit = -torch.log(y) - torch.log1p(-y)

        # log |dx/dz|
        logdet_sigmoid = torch.log(x) + torch.log1p(-x)

        logdet = (logdet_logit + logdet_sigmoid).sum(dim=-1)

        return x, logdet