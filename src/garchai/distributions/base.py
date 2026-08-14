"""Standardised conditional distributions for the innovations.

Every distribution here is **standardised**: :math:`E[z_t] = 0` and
:math:`\\operatorname{Var}(z_t) = 1`. That is what makes

.. math::
    \\epsilon_t = \\sigma_t z_t, \\qquad
    \\operatorname{Var}(\\epsilon_t \\mid \\psi_{t-1}) = h_t

hold, so that :math:`h_t` really is the conditional variance rather than a
scale parameter that happens to be called one. Blueprint sec. 19 flags this as a
recurring source of silent error: a raw Student-t has variance
:math:`\\nu/(\\nu-2)`, so a model that plugs :math:`\\sigma_t` into an
unstandardised t is misreporting its own volatility by ~15% at ``nu = 8``.

Each distribution provides:

``log_prob(z)``
    log density of the standardised innovation.
``log_prob_return(eps, h)``
    log density of the *observation*, i.e. ``log_prob(eps/sqrt(h)) - 0.5*log(h)``.
    This is the quantity summed to form the log-likelihood.
``ppf(alpha)``
    lower-tail quantile of the standardised innovation.
``expected_shortfall(alpha)``
    :math:`E[z \\mid z \\le q_\\alpha]`, the standardised expected shortfall.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_PI = math.log(math.pi)


class Distribution(ABC):
    """Base class for standardised innovation distributions."""

    name: str = "distribution"
    n_shape_params: int = 0
    """Number of extra parameters beyond location and scale (e.g. df)."""

    @abstractmethod
    def log_prob(self, z: Tensor) -> Tensor:
        """Log density of the standardised innovation ``z``."""

    @abstractmethod
    def ppf(self, alpha: float | Tensor) -> Tensor:
        """Lower-tail quantile: the ``q`` with ``P(z <= q) = alpha``."""

    @abstractmethod
    def expected_shortfall(self, alpha: float | Tensor) -> Tensor:
        """``E[z | z <= ppf(alpha)]`` for the standardised innovation."""

    # -- shared machinery --------------------------------------------------- #
    def log_prob_return(self, eps: Tensor, h: Tensor) -> Tensor:
        """Log density of ``eps`` given conditional variance ``h``.

        The Jacobian of ``eps -> z = eps / sqrt(h)`` contributes ``-0.5*log(h)``.
        """
        return self.log_prob(eps * torch.rsqrt(h)) - 0.5 * torch.log(h)

    def nll(self, eps: Tensor, h: Tensor, reduction: str = "sum") -> Tensor:
        """Negative log-likelihood of ``eps`` under conditional variance ``h``."""
        out = -self.log_prob_return(eps, h)
        if reduction == "sum":
            return out.sum()
        if reduction == "mean":
            return out.mean()
        if reduction == "none":
            return out
        raise ValueError(f"Unknown reduction {reduction!r}.")

    def standardize(self, eps: Tensor, h: Tensor) -> Tensor:
        """Return ``z = eps / sqrt(h)``."""
        return eps * torch.rsqrt(h)

    def var(self, alpha: float, mu: Tensor, sigma: Tensor) -> Tensor:
        """Value-at-Risk as a return level: ``mu + sigma * ppf(alpha)``.

        Reported on the return scale and therefore **negative** for small
        ``alpha``. Sign conventions differ across the literature; ``garchai``
        always returns the quantile of the return itself, never its absolute
        value, so that ``returns < var`` is the exceedance test.
        """
        return mu + sigma * self.ppf(alpha).to(sigma.dtype)

    def es(self, alpha: float, mu: Tensor, sigma: Tensor) -> Tensor:
        """Expected shortfall as a return level, same sign convention as :meth:`var`."""
        return mu + sigma * self.expected_shortfall(alpha).to(sigma.dtype)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


def _as_tensor(x: float | Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(dtype)
    return torch.tensor(float(x), dtype=dtype)
