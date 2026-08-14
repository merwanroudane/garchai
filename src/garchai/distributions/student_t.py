"""Standardised Student-t innovations, and the paper-faithful T-loss kernel."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import _LOG_PI, Distribution, _as_tensor


class StudentT(Distribution):
    r"""Student-t innovations standardised to unit variance.

    If :math:`T \sim t_\nu` then :math:`\operatorname{Var}(T) = \nu/(\nu-2)`, so
    the standardised innovation is :math:`z = T / \sqrt{\nu/(\nu-2)}` and its
    density is

    .. math::
        \log f(z) = \log\Gamma\!\left(\tfrac{\nu+1}{2}\right)
                  - \log\Gamma\!\left(\tfrac{\nu}{2}\right)
                  - \tfrac{1}{2}\log\!\big(\pi(\nu-2)\big)
                  - \tfrac{\nu+1}{2}\log\!\left(1 + \frac{z^2}{\nu-2}\right).

    This matches Liu & So (2020, sec. 2.2) exactly, and is the standardisation
    Bollerslev (1987) introduced. It requires ``nu > 2``.

    Parameters
    ----------
    df
        Degrees of freedom :math:`\nu`. May be a scalar tensor carrying
        gradient, so that ``nu`` is estimated jointly with the variance path,
        or a plain float when the paper fixes it (Zhao et al. use ``nu`` in
        ``{3, 5}``).

    Examples
    --------
    >>> import torch
    >>> d = StudentT(df=5.0)
    >>> z = torch.linspace(-80, 80, 200_001, dtype=torch.float64)
    >>> pdf = torch.exp(d.log_prob(z))
    >>> round(float(torch.trapz(pdf, z)), 4)          # integrates to one
    1.0
    >>> round(float(torch.trapz(z * z * pdf, z)), 3)  # and is standardised
    1.0
    """

    name = "student-t"
    n_shape_params = 1

    def __init__(self, df: float | Tensor = 8.0):
        self.df = df if isinstance(df, Tensor) else torch.tensor(float(df), dtype=torch.float64)
        if float(self.df.detach()) <= 2.0:
            raise ValueError("Student-t degrees of freedom must exceed 2 for unit variance.")

    # -- density ------------------------------------------------------------ #
    def log_prob(self, z: Tensor) -> Tensor:
        nu = self.df.to(z.dtype)
        nm2 = nu - 2.0
        const = (
            torch.lgamma(0.5 * (nu + 1.0))
            - torch.lgamma(0.5 * nu)
            - 0.5 * (_LOG_PI + torch.log(nm2))
        )
        return const - 0.5 * (nu + 1.0) * torch.log1p(z * z / nm2)

    # -- quantiles ---------------------------------------------------------- #
    def ppf(self, alpha: float | Tensor) -> Tensor:
        """Lower-tail quantile of the *standardised* t.

        Computed as the raw :math:`t_\\nu` quantile divided by
        :math:`\\sqrt{\\nu/(\\nu-2)}`. The raw quantile comes from the inverse
        regularised incomplete beta function; SciPy is used when available
        because ``torch`` has no ``betaincinv``.
        """
        nu = float(self.df.detach())
        a = _as_tensor(alpha)
        raw = _t_ppf(a, nu)
        return raw / math.sqrt(nu / (nu - 2.0))

    def expected_shortfall(self, alpha: float | Tensor) -> Tensor:
        r"""``E[z | z <= q]`` for the standardised t.

        For a raw :math:`t_\nu` with lower-tail quantile :math:`q_T`,

        .. math::
            E[T \mid T \le q_T]
              = -\frac{\nu + q_T^2}{\nu - 1}\,\frac{f_\nu(q_T)}{\alpha},

        which requires ``nu > 1``; the standardised value divides by
        :math:`\sqrt{\nu/(\nu-2)}`.
        """
        nu = float(self.df.detach())
        a = _as_tensor(alpha)
        q_raw = _t_ppf(a, nu)
        log_pdf = (
            math.lgamma(0.5 * (nu + 1.0))
            - math.lgamma(0.5 * nu)
            - 0.5 * (_LOG_PI + math.log(nu))
            - 0.5 * (nu + 1.0) * torch.log1p(q_raw * q_raw / nu)
        )
        pdf = torch.exp(log_pdf)
        es_raw = -(nu + q_raw * q_raw) / (nu - 1.0) * pdf / a
        return es_raw / math.sqrt(nu / (nu - 2.0))


def _t_ppf(alpha: Tensor, df: float) -> Tensor:
    """Lower-tail quantile of a raw ``t_df``, via SciPy with a bisection fallback."""
    try:
        from scipy.stats import t as _scipy_t
    except ImportError:  # pragma: no cover - SciPy is a hard dependency
        return _t_ppf_bisect(alpha, df)
    import numpy as np

    value = _scipy_t.ppf(np.asarray(alpha.detach().cpu(), dtype=np.float64), df)
    return torch.as_tensor(value, dtype=alpha.dtype)


def _t_ppf_bisect(alpha: Tensor, df: float, iters: int = 200) -> Tensor:  # pragma: no cover
    """Bisection fallback, used only if SciPy is unavailable."""
    lo = torch.full_like(alpha, -1e4)
    hi = torch.full_like(alpha, 1e4)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = _t_cdf(mid, df)
        lo = torch.where(cdf < alpha, mid, lo)
        hi = torch.where(cdf < alpha, hi, mid)
    return 0.5 * (lo + hi)


def _t_cdf(x: Tensor, df: float) -> Tensor:  # pragma: no cover
    from scipy.special import betainc
    import numpy as np

    xn = np.asarray(x.detach().cpu(), dtype=np.float64)
    z = df / (df + xn * xn)
    tail = 0.5 * betainc(0.5 * df, 0.5, z)
    return torch.as_tensor(np.where(xn > 0, 1.0 - tail, tail), dtype=x.dtype)


# --------------------------------------------------------------------------- #
# paper-faithful variant
# --------------------------------------------------------------------------- #
class StudentTKernel(StudentT):
    r"""Zhao et al. (2024) *T-loss*: the t log-density without its constants.

    Equation 6 of the paper drops the :math:`\log\Gamma` normalising terms:

    .. math::
        -\log L(\epsilon_t) = \tfrac{1}{2}\log \hat\sigma_t^2
            + \tfrac{\nu+1}{2}\log\!\left(1 +
              \frac{\epsilon_t^2}{(\nu-2)\hat\sigma_t^2}\right).

    Because the omitted terms do not depend on :math:`\hat\sigma_t`, the
    gradient with respect to the variance path — and therefore the fitted model —
    is identical to :class:`StudentT` at fixed ``nu``. What *does* change is the
    reported log-likelihood value, so this class must not be used for AIC, BIC or
    any cross-model likelihood comparison.

    It exists so that reproductions of Zhao et al. report the same loss numbers
    the paper does. Use :class:`StudentT` everywhere else.

    Warnings
    --------
    Not a normalised density. ``exp(log_prob(z))`` does not integrate to one, and
    ``nu`` must be held fixed: with ``nu`` free the dropped constants matter and
    the estimate would be inconsistent.
    """

    name = "student-t-kernel"

    def log_prob(self, z: Tensor) -> Tensor:
        nu = self.df.to(z.dtype)
        return -0.5 * (nu + 1.0) * torch.log1p(z * z / (nu - 2.0))

    def log_prob_return(self, eps: Tensor, h: Tensor) -> Tensor:
        return self.log_prob(eps * torch.rsqrt(h)) - 0.5 * torch.log(h)
