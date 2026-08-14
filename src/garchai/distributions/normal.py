"""Standard normal innovations."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import _LOG_2PI, Distribution, _as_tensor

_SQRT2 = math.sqrt(2.0)


class Normal(Distribution):
    """Standard normal innovations, :math:`z_t \\sim N(0,1)`.

    The resulting observation log-density is the familiar

    .. math::
        \\log f(\\epsilon_t \\mid h_t) =
            -\\tfrac{1}{2}\\left[\\log 2\\pi + \\log h_t
            + \\frac{\\epsilon_t^2}{h_t}\\right],

    which is the loss used by Zhao et al. (2024, Eq. 5) under the name *N-loss*
    and by Nguyen et al. (2022) for RECH.

    Examples
    --------
    >>> import torch
    >>> d = Normal()
    >>> float(d.log_prob(torch.tensor(0.0, dtype=torch.float64)))
    -0.9189385332046727
    >>> round(float(d.ppf(0.05)), 6)
    -1.644854
    """

    name = "normal"
    n_shape_params = 0

    def log_prob(self, z: Tensor) -> Tensor:
        return -0.5 * (_LOG_2PI + z * z)

    def cdf(self, z: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf(z / _SQRT2))

    def ppf(self, alpha: float | Tensor) -> Tensor:
        a = _as_tensor(alpha)
        return _SQRT2 * torch.erfinv(2.0 * a - 1.0)

    def expected_shortfall(self, alpha: float | Tensor) -> Tensor:
        r"""``E[z | z <= q] = -phi(q) / alpha`` for the standard normal."""
        a = _as_tensor(alpha)
        q = self.ppf(a)
        pdf = torch.exp(-0.5 * q * q) / math.sqrt(2.0 * math.pi)
        return -pdf / a
