r"""Mean models: turning returns into innovations.

Blueprint sec. 21 is blunt about this: *do not silently assume returns equal
innovations*. The variance recursion models :math:`\epsilon_t`, not :math:`r_t`,
and

.. math::
    r_t = \mu_t + \epsilon_t .

Whether :math:`\mu_t` is zero, a constant, or an ARMA process is a modelling
choice with real consequences — and the papers in this collection disagree.
Zhao et al. (2024) invoke the efficient-market hypothesis and set
:math:`\mu_t = 0`; Wei et al. (2025) run a t-test, reject a zero mean for all
three indices, and adopt a constant; Xu et al. (2024) fit an AR model over a
rolling 90-day window. Reproducing any of them faithfully means being able to
say which.

Each mean model is an ``nn.Module`` whose parameters are optimised jointly with
the variance parameters, so the mean and variance are not estimated in two
disconnected stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "MeanModel",
    "ZeroMean",
    "ConstantMean",
    "ARMean",
    "ARMAMean",
    "get_mean_model",
]


class MeanModel(nn.Module, ABC):
    """Base class for conditional-mean specifications.

    Subclasses implement :meth:`conditional_mean`. The contract is that
    :meth:`residuals` returns a series *aligned with* ``r``, so that downstream
    code never has to reason about offsets; where a lag structure makes the
    first few fitted values unavailable, :attr:`n_presample` says how many
    observations should be dropped from the likelihood.
    """

    n_presample: int = 0

    @abstractmethod
    def conditional_mean(self, r: Tensor) -> Tensor:
        r"""Return :math:`\mu_t` for every ``t``, aligned with ``r``."""

    def residuals(self, r: Tensor) -> tuple[Tensor, Tensor]:
        r"""Return ``(eps, mu)`` where :math:`\epsilon_t = r_t - \mu_t`."""
        mu = self.conditional_mean(r)
        return r - mu, mu

    def parameter_dict(self) -> dict[str, Tensor]:
        """Parameters on their natural scale, for reporting."""
        return {name: p.detach().clone() for name, p in self.named_parameters()}

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        items = ", ".join(f"{k}={float(v):.5g}" for k, v in self.parameter_dict().items()
                          if v.numel() == 1)
        return items


class ZeroMean(MeanModel):
    r"""Fix :math:`\mu_t = 0`, so that :math:`\epsilon_t = r_t`.

    The right choice when the paper says so. Zhao et al. (2024) state it
    explicitly: "According to the efficient market hypothesis, in practice, we
    treat :math:`\mu_t = 0` and then :math:`\epsilon_t = r_t`."

    Examples
    --------
    >>> import torch
    >>> r = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    >>> eps, mu = ZeroMean().residuals(r)
    >>> bool(torch.equal(eps, r)), float(mu.abs().max())
    (True, 0.0)
    """

    name = "zero"

    def conditional_mean(self, r: Tensor) -> Tensor:
        return torch.zeros_like(r)


class ConstantMean(MeanModel):
    r"""A single estimated intercept, :math:`\mu_t = \mu`.

    This is the default and, for daily returns, almost always the honest
    minimum: a zero mean is a testable restriction, not a free lunch. Wei et al.
    (2025, Table 1) report two-sided t-tests rejecting a zero mean at the 5%
    level for the S&P 500 and NASDAQ and at 10% for the DJI, and adopt a
    constant mean on that basis.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> m = ConstantMean(mu=0.04)
    >>> float(m.conditional_mean(torch.zeros(5))[0])
    0.04
    """

    name = "constant"

    def __init__(self, mu: float = 0.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(mu), dtype=torch.float64))

    def conditional_mean(self, r: Tensor) -> Tensor:
        return self.mu.to(r.dtype).expand_as(r)


class ARMean(MeanModel):
    r"""Autoregressive mean of order ``p``.

    .. math::
        \mu_t = c + \sum_{i=1}^{p} \phi_i\, r_{t-i}

    The first ``p`` fitted values have no complete lag vector. Rather than
    fabricate them, we fill them with the unconditional mean
    :math:`c/(1-\sum_i \phi_i)` so that the variance filter has something to
    start from, and set :attr:`n_presample` to ``p`` so the likelihood can drop
    them. This is the convention :mod:`arch` uses, and it matters: including
    conditioning observations in the likelihood biases the intercept.

    Parameters
    ----------
    p
        Autoregressive order.
    include_constant
        Whether to estimate ``c``.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> m = ARMean(p=1)
    >>> with torch.no_grad():
    ...     _ = m.const.fill_(0.1); _ = m.phi.fill_(0.5)
    >>> r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> [round(float(x), 4) for x in m.conditional_mean(r)[1:]]
    [0.6, 1.1, 1.6]
    """

    name = "ar"

    def __init__(self, p: int = 1, include_constant: bool = True):
        super().__init__()
        if p < 1:
            raise ValueError("ARMean requires p >= 1; use ZeroMean or ConstantMean instead.")
        self.p = p
        self.n_presample = p
        self.include_constant = include_constant
        self.phi = nn.Parameter(torch.zeros(p, dtype=torch.float64))
        if include_constant:
            self.const = nn.Parameter(torch.zeros((), dtype=torch.float64))
        else:
            self.register_parameter("const", None)

    def _intercept(self, dtype: torch.dtype) -> Tensor:
        if self.const is None:
            return torch.zeros((), dtype=dtype)
        return self.const.to(dtype)

    def unconditional_mean(self, dtype: torch.dtype = torch.float64) -> Tensor:
        r"""Return :math:`c/(1-\sum_i \phi_i)`, or ``c`` when the AR is explosive."""
        c = self._intercept(dtype)
        denom = 1.0 - self.phi.to(dtype).sum()
        if float(denom.detach().abs()) < 1e-8:
            return c
        return c / denom

    def conditional_mean(self, r: Tensor) -> Tensor:
        c = self._intercept(r.dtype)
        phi = self.phi.to(r.dtype)
        head = self.unconditional_mean(r.dtype).expand(min(self.p, r.numel()))
        if r.numel() <= self.p:
            return head.clone()
        lagged = torch.stack([r[self.p - i - 1 : -i - 1] for i in range(self.p)], dim=-1)
        return torch.cat([head, c + lagged @ phi])


class ARMAMean(MeanModel):
    r"""ARMA(p, q) mean.

    .. math::
        \mu_t = c + \sum_{i=1}^{p}\phi_i r_{t-i}
                  + \sum_{j=1}^{q}\theta_j \epsilon_{t-j}

    The moving-average part makes this genuinely recursive: :math:`\epsilon_t`
    depends on earlier :math:`\epsilon`'s, which depend on the parameters. We
    condition on :math:`\epsilon_t = 0` for :math:`t < \max(p,q)` — the standard
    conditional-likelihood treatment — and report those as pre-sample.

    Because the recursion runs in Python, ARMA means are slower than AR means;
    for the daily series in this collection that cost is irrelevant next to the
    variance filter.
    """

    name = "arma"

    def __init__(self, p: int = 1, q: int = 1, include_constant: bool = True):
        super().__init__()
        if p < 0 or q < 0 or (p == 0 and q == 0):
            raise ValueError("ARMAMean requires p >= 0, q >= 0 and p + q >= 1.")
        self.p, self.q = p, q
        self.n_presample = max(p, q)
        self.phi = nn.Parameter(torch.zeros(max(p, 1), dtype=torch.float64))
        self.theta = nn.Parameter(torch.zeros(max(q, 1), dtype=torch.float64))
        if include_constant:
            self.const = nn.Parameter(torch.zeros((), dtype=torch.float64))
        else:
            self.register_parameter("const", None)

    def conditional_mean(self, r: Tensor) -> Tensor:
        c = torch.zeros((), dtype=r.dtype) if self.const is None else self.const.to(r.dtype)
        phi = self.phi.to(r.dtype)
        theta = self.theta.to(r.dtype)
        n = r.numel()
        start = self.n_presample

        mu_parts: list[Tensor] = [c.expand(start).clone()] if start else []
        eps_hist: list[Tensor] = [torch.zeros((), dtype=r.dtype) for _ in range(start)]

        for t in range(start, n):
            value = c
            for i in range(self.p):
                value = value + phi[i] * r[t - i - 1]
            for j in range(self.q):
                value = value + theta[j] * eps_hist[t - j - 1]
            mu_parts.append(value.reshape(1))
            eps_hist.append(r[t] - value)
        return torch.cat(mu_parts) if mu_parts else torch.zeros_like(r)


_REGISTRY: dict[str, type[MeanModel]] = {
    "zero": ZeroMean,
    "constant": ConstantMean,
    "ar": ARMean,
    "arma": ARMAMean,
}


def get_mean_model(name: str, **kwargs) -> MeanModel:
    """Build a mean model by name.

    Examples
    --------
    >>> get_mean_model("constant").name
    'constant'
    >>> get_mean_model("ar", p=2).n_presample
    2
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown mean model {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
