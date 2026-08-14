"""Hansen (1994) skewed Student-t, standardised to zero mean and unit variance.

Reference
---------
Hansen, B. E. (1994). Autoregressive conditional density estimation.
*International Economic Review* 35(3), 705-730.

This is the distribution GARCHNet (Buczynski & Chlebus, 2023) uses for its
skewed specification, and the one most VaR work reaches for when the normal and
the symmetric t both misprice the left tail.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import _LOG_PI, Distribution, _as_tensor


class SkewT(Distribution):
    r"""Hansen's skewed Student-t with zero mean and unit variance.

    With degrees of freedom :math:`\eta > 2` and skewness :math:`-1 < \lambda < 1`,

    .. math::
        c &= \frac{\Gamma\!\left(\frac{\eta+1}{2}\right)}
                  {\sqrt{\pi(\eta-2)}\;\Gamma\!\left(\frac{\eta}{2}\right)}, \\
        a &= 4\lambda c\,\frac{\eta-2}{\eta-1}, \qquad
        b^2 = 1 + 3\lambda^2 - a^2,

    and

    .. math::
        f(z) = \begin{cases}
          bc\left[1 + \frac{1}{\eta-2}
            \left(\frac{bz+a}{1-\lambda}\right)^2\right]^{-\frac{\eta+1}{2}},
            & z < -a/b, \\[2mm]
          bc\left[1 + \frac{1}{\eta-2}
            \left(\frac{bz+a}{1+\lambda}\right)^2\right]^{-\frac{\eta+1}{2}},
            & z \ge -a/b.
        \end{cases}

    ``lam = 0`` reduces exactly to the standardised Student-t.

    Parameters
    ----------
    df
        Degrees of freedom :math:`\eta`, must exceed 2.
    lam
        Skewness :math:`\lambda` in ``(-1, 1)``. Negative values put more mass in
        the left tail, which is the empirically usual direction for equity
        returns.

    Examples
    --------
    >>> import torch
    >>> from garchai.distributions import StudentT
    >>> s, t = SkewT(df=6.0, lam=0.0), StudentT(df=6.0)
    >>> z = torch.tensor([-3.0, -1.0, 0.0, 2.0], dtype=torch.float64)
    >>> bool(torch.allclose(s.log_prob(z), t.log_prob(z), atol=1e-12))
    True
    """

    name = "skew-t"
    n_shape_params = 2

    def __init__(self, df: float | Tensor = 8.0, lam: float | Tensor = 0.0):
        self.df = df if isinstance(df, Tensor) else torch.tensor(float(df), dtype=torch.float64)
        self.lam = lam if isinstance(lam, Tensor) else torch.tensor(float(lam), dtype=torch.float64)
        if float(self.df.detach()) <= 2.0:
            raise ValueError("Skewed-t degrees of freedom must exceed 2 for unit variance.")
        if not -1.0 < float(self.lam.detach()) < 1.0:
            raise ValueError("Skewness lambda must lie strictly between -1 and 1.")

    # -- Hansen constants --------------------------------------------------- #
    def _constants(self, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        eta = self.df.to(dtype)
        lam = self.lam.to(dtype)
        log_c = (
            torch.lgamma(0.5 * (eta + 1.0))
            - torch.lgamma(0.5 * eta)
            - 0.5 * (_LOG_PI + torch.log(eta - 2.0))
        )
        c = torch.exp(log_c)
        a = 4.0 * lam * c * (eta - 2.0) / (eta - 1.0)
        b = torch.sqrt(1.0 + 3.0 * lam * lam - a * a)
        return eta, lam, a, b

    def log_prob(self, z: Tensor) -> Tensor:
        eta, lam, a, b = self._constants(z.dtype)
        log_c = (
            torch.lgamma(0.5 * (eta + 1.0))
            - torch.lgamma(0.5 * eta)
            - 0.5 * (_LOG_PI + torch.log(eta - 2.0))
        )
        shifted = b * z + a
        denom = torch.where(z < -a / b, 1.0 - lam, 1.0 + lam)
        quad = (shifted / denom) ** 2 / (eta - 2.0)
        return torch.log(b) + log_c - 0.5 * (eta + 1.0) * torch.log1p(quad)

    def ppf(self, alpha: float | Tensor) -> Tensor:
        r"""Lower-tail quantile, Hansen's inverse CDF.

        .. math::
            q(\alpha) = \begin{cases}
              \frac{1}{b}\left[(1-\lambda)\sqrt{\frac{\eta-2}{\eta}}\;
                 t_\eta^{-1}\!\left(\frac{\alpha}{1-\lambda}\right) - a\right],
                 & \alpha < \frac{1-\lambda}{2},\\[2mm]
              \frac{1}{b}\left[(1+\lambda)\sqrt{\frac{\eta-2}{\eta}}\;
                 t_\eta^{-1}\!\left(\frac{\alpha+\lambda}{1+\lambda}\right) - a\right],
                 & \text{otherwise.}
            \end{cases}
        """
        from scipy.stats import t as _scipy_t
        import numpy as np

        a_t = _as_tensor(alpha)
        eta, lam, a, b = self._constants(a_t.dtype)
        eta_f, lam_f = float(eta.detach()), float(lam.detach())
        an = np.asarray(a_t.detach().cpu(), dtype=np.float64)

        left = an < (1.0 - lam_f) / 2.0
        u = np.where(left, an / (1.0 - lam_f), (an + lam_f) / (1.0 + lam_f))
        weight = np.where(left, 1.0 - lam_f, 1.0 + lam_f)
        raw = _scipy_t.ppf(np.clip(u, 1e-15, 1.0 - 1e-15), eta_f)
        scaled = weight * math.sqrt((eta_f - 2.0) / eta_f) * raw
        return torch.as_tensor(
            (scaled - float(a.detach())) / float(b.detach()), dtype=a_t.dtype
        )

    def expected_shortfall(self, alpha: float | Tensor) -> Tensor:
        r"""``E[z | z <= q]`` via the quantile-integral identity.

        Hansen's skewed-t has no compact closed-form tail expectation. Rather
        than integrate :math:`z f(z)` over a truncated left tail — which
        converges painfully slowly, since the integrand decays only like
        :math:`|z|^{-\eta}` — we substitute :math:`u = F(z)` and use

        .. math::
            E[z \mid z \le q_\alpha]
              = \frac{1}{\alpha}\int_{-\infty}^{q_\alpha} z f(z)\,\mathrm{d}z
              = \frac{1}{\alpha}\int_{0}^{\alpha} F^{-1}(u)\,\mathrm{d}u .

        The domain is now finite, so there is no truncation error at all. The
        integrand has an integrable singularity at :math:`u = 0` (it diverges
        like :math:`u^{-1/\eta}`, and :math:`\eta > 2`), which adaptive
        quadrature with the endpoint declared as a break point handles cleanly.

        Accurate to ~1e-10 absolute. Used for reporting, never inside a training
        loop.
        """
        from scipy.integrate import quad
        import numpy as np

        a_t = _as_tensor(alpha)
        flat = a_t.reshape(-1) if a_t.ndim else a_t.reshape(1)

        def integrand(u: float) -> float:
            return float(self.ppf(u))

        values = []
        for value in flat:
            a = float(value)
            total, _ = quad(integrand, 0.0, a, points=[0.0], limit=200)
            values.append(total / a)
        out = torch.as_tensor(np.asarray(values), dtype=torch.float64)
        return out.reshape(a_t.shape) if a_t.ndim else out.squeeze()
