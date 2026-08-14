"""Standardised Generalised Error Distribution (Nelson, 1991)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import Distribution, _as_tensor


def _lambda(nu: float) -> float:
    r"""The scale constant :math:`\lambda` that gives the GED unit variance.

    .. math::
        \lambda = \left[2^{-2/\nu}\,
                  \frac{\Gamma(1/\nu)}{\Gamma(3/\nu)}\right]^{1/2}
    """
    return math.sqrt(math.exp(math.lgamma(1.0 / nu) - math.lgamma(3.0 / nu)) * 2.0 ** (-2.0 / nu))


class GED(Distribution):
    r"""Generalised Error Distribution standardised to unit variance.

    .. math::
        f(z) = \frac{\nu \exp\!\left(-\tfrac{1}{2}|z/\lambda|^{\nu}\right)}
                    {\lambda\, 2^{1+1/\nu}\, \Gamma(1/\nu)}

    ``nu = 2`` recovers the standard normal, ``nu < 2`` gives fatter tails and
    ``nu > 2`` thinner ones. Unlike the Student-t, all moments exist for every
    ``nu > 0``, which makes the GED a useful robustness check: if a fitted t
    wants ``nu`` near its lower bound, refitting with a GED tells you whether
    that is genuine tail behaviour or the optimiser chasing an outlier.

    Examples
    --------
    >>> import torch
    >>> from garchai.distributions import Normal
    >>> g, n = GED(nu=2.0), Normal()
    >>> z = torch.tensor([-2.0, -0.5, 0.0, 1.3], dtype=torch.float64)
    >>> bool(torch.allclose(g.log_prob(z), n.log_prob(z), atol=1e-12))
    True
    """

    name = "ged"
    n_shape_params = 1

    def __init__(self, nu: float | Tensor = 1.5):
        self.nu = nu if isinstance(nu, Tensor) else torch.tensor(float(nu), dtype=torch.float64)
        if float(self.nu.detach()) <= 0.0:
            raise ValueError("GED shape parameter nu must be positive.")

    @property
    def lam(self) -> float:
        return _lambda(float(self.nu.detach()))

    def log_prob(self, z: Tensor) -> Tensor:
        nu = self.nu.to(z.dtype)
        lam = torch.as_tensor(self.lam, dtype=z.dtype)
        const = (
            torch.log(nu)
            - torch.log(lam)
            - (1.0 + 1.0 / nu) * math.log(2.0)
            - torch.lgamma(1.0 / nu)
        )
        return const - 0.5 * torch.abs(z / lam) ** nu

    def ppf(self, alpha: float | Tensor) -> Tensor:
        r"""Lower-tail quantile.

        With :math:`t_0 = \tfrac{1}{2}|q/\lambda|^{\nu}` the tail probability is
        :math:`\tfrac{1}{2}\big(1 - P(1/\nu, t_0)\big)`, where :math:`P` is the
        regularised lower incomplete gamma function. Inverting gives
        :math:`q = -\lambda (2 t_0)^{1/\nu}` for ``alpha < 0.5``, and the
        mirror image above it.
        """
        from scipy.special import gammaincinv
        import numpy as np

        a = _as_tensor(alpha)
        nu = float(self.nu.detach())
        lam = self.lam
        an = np.asarray(a.detach().cpu(), dtype=np.float64)
        lower = an < 0.5
        p = np.where(lower, 1.0 - 2.0 * an, 2.0 * an - 1.0)
        t0 = gammaincinv(1.0 / nu, np.clip(p, 0.0, 1.0 - 1e-15))
        magnitude = lam * (2.0 * t0) ** (1.0 / nu)
        return torch.as_tensor(np.where(lower, -magnitude, magnitude), dtype=a.dtype)

    def expected_shortfall(self, alpha: float | Tensor) -> Tensor:
        r"""``E[z | z <= q]`` in closed form.

        .. math::
            E[z \mid z \le q] = -\frac{\lambda\,2^{1/\nu}\,\Gamma(2/\nu)}
                                       {2\,\Gamma(1/\nu)\,\alpha}
                                 \big(1 - P(2/\nu, t_0)\big)

        with :math:`t_0 = \tfrac{1}{2}|q/\lambda|^{\nu}`. Valid for ``alpha < 0.5``,
        which is the only region VaR/ES is ever evaluated in.
        """
        from scipy.special import gammaincc
        import numpy as np

        a = _as_tensor(alpha)
        nu = float(self.nu.detach())
        lam = self.lam
        q = self.ppf(a)
        an = np.asarray(a.detach().cpu(), dtype=np.float64)
        qn = np.asarray(q.detach().cpu(), dtype=np.float64)
        t0 = 0.5 * np.abs(qn / lam) ** nu
        upper = gammaincc(2.0 / nu, t0)  # already regularised
        coef = lam * 2.0 ** (1.0 / nu) * math.exp(math.lgamma(2.0 / nu) - math.lgamma(1.0 / nu))
        return torch.as_tensor(-coef * upper / (2.0 * an), dtype=a.dtype)
