r"""Volatility targets and proxies, each named after the convention it follows.

Blueprint sec. 29 warns: *do not mix target definitions silently*. That warning
is not hypothetical. Two of the Tier-S papers in this collection both say they
use "5-day realized volatility" and mean genuinely different estimators:

Zhao et al. (2024)
    :math:`\sigma_t = \sqrt{\sum_{i=0}^{k-1}\epsilon_{t-i}^2}` with ``k = 5``.
    No division by ``k``, no demeaning, and the window **includes** the current
    observation.

Wei et al. (2025, Eq. 31)
    :math:`\sigma_t = \sqrt{\frac{1}{k}\sum_{i=1}^{k}(r_{t-i}-\bar r_k)^2}` with
    ``k = 5``. Divided by ``k``, demeaned, and the window is strictly **lagged**
    so no look-ahead is possible.

At ``k = 5`` the first is :math:`\sqrt{5}\approx 2.24` times larger than the
second even before the demeaning and the shift. A reproduction that quietly
swapped one for the other would report metrics off by a factor of five and still
look internally consistent. So each estimator is a separate, named, documented
function, and the paper reproductions call them by name.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "realized_volatility_zhao",
    "realized_volatility_wei",
    "rolling_std",
    "squared_returns",
    "absolute_returns",
    "realized_variance_intraday",
    "realized_volatility_intraday",
    "bipower_variation",
    "hansen_lunde_scale",
    "get_target",
]


def _check_1d(x: Tensor, name: str = "series") -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(x.shape)}.")


# --------------------------------------------------------------------------- #
# paper-specific daily proxies
# --------------------------------------------------------------------------- #
def realized_volatility_zhao(eps: Tensor, k: int = 5) -> Tensor:
    r"""Zhao et al. (2024) realized volatility.

    .. math::
        \sigma_t = \sqrt{\sum_{i=0}^{k-1}\epsilon_{t-i}^2}

    Note the three ways this differs from the more common estimator: the sum is
    **not** divided by ``k``, the innovations are **not** demeaned, and the
    window **includes** :math:`\epsilon_t` itself.

    That last point deserves care. As a *description* of period-``t`` volatility
    this is fine and is what the paper evaluates against. But it means the
    target at time ``t`` is not in the information set at time ``t-1``, so it
    must never be fed back as a predictor. :func:`realized_volatility_wei` is
    the lagged alternative.

    Returns
    -------
    Tensor
        Same length as ``eps``. The first ``k-1`` entries use the shorter window
        available to them rather than being dropped, so alignment is preserved.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> eps = torch.tensor([1.0, 2.0, 2.0])
    >>> [round(float(v), 6) for v in realized_volatility_zhao(eps, k=2)]
    [1.0, 2.236068, 2.828427]
    """
    _check_1d(eps, "eps")
    if k < 1:
        raise ValueError("k must be at least 1.")
    sq = eps.square()
    cumulative = torch.cumsum(sq, dim=0)
    shifted = torch.cat([torch.zeros(k, dtype=eps.dtype, device=eps.device), cumulative[:-k]])
    windowed = cumulative - shifted[: cumulative.numel()]
    return torch.sqrt(windowed.clamp(min=0.0))


def realized_volatility_wei(r: Tensor, k: int = 5) -> Tensor:
    r"""Wei, Yang & Cui (2025, Eq. 31) realized volatility.

    .. math::
        \sigma_t = \sqrt{\frac{1}{k}\sum_{i=1}^{k}\big(r_{t-i}-\bar r_k\big)^2},
        \qquad \bar r_k = \frac{1}{k}\sum_{i=1}^{k} r_{t-i}

    Strictly lagged: the window covers :math:`t-k, \dots, t-1`, so the target is
    measurable with respect to information at :math:`t-1` and can be used as a
    predictor without look-ahead. Demeaned within the window, and divided by
    ``k`` (not ``k-1``).

    Returns
    -------
    Tensor
        Same length as ``r``. The first ``k`` entries have no complete lagged
        window and are returned as ``nan``, which forces the caller to decide
        what to do rather than silently inserting zeros.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> r = torch.tensor([1.0, 3.0, 5.0, 7.0])
    >>> out = realized_volatility_wei(r, k=2)
    >>> bool(out[:2].isnan().all()), round(float(out[2]), 6)
    (True, 1.0)
    """
    _check_1d(r, "r")
    if k < 1:
        raise ValueError("k must be at least 1.")
    n = r.numel()
    out = torch.full((n,), float("nan"), dtype=r.dtype, device=r.device)
    if n <= k:
        return out
    windows = r.unfold(0, k, 1)[:-1]  # rows t-k..t-1 for t = k..n-1
    centred = windows - windows.mean(dim=1, keepdim=True)
    values = torch.sqrt(centred.square().mean(dim=1))
    return torch.cat([out[:k], values])


def rolling_std(r: Tensor, window: int = 21, unbiased: bool = True) -> Tensor:
    r"""Trailing rolling standard deviation, the common practitioner proxy.

    Strictly lagged, demeaned, and by default using the unbiased ``k-1``
    divisor. This is what community GARCH-LSTM implementations typically mean by
    "realized volatility" when they work from daily data only.
    """
    _check_1d(r, "r")
    n = r.numel()
    out = torch.full((n,), float("nan"), dtype=r.dtype, device=r.device)
    if n <= window:
        return out
    windows = r.unfold(0, window, 1)[:-1]
    centred = windows - windows.mean(dim=1, keepdim=True)
    denom = window - 1 if unbiased else window
    return torch.cat([out[:window], torch.sqrt(centred.square().sum(dim=1) / denom)])


def squared_returns(eps: Tensor) -> Tensor:
    r"""The crude one-observation variance proxy :math:`\epsilon_t^2`.

    Unbiased for :math:`h_t` but extremely noisy — its signal-to-noise ratio is
    poor enough that model rankings computed against it can invert relative to
    rankings against a realized measure. Useful as a *loss* target under QLIKE,
    which is robust to that noise; a poor choice for reporting MSE.
    """
    _check_1d(eps, "eps")
    return eps.square()


def absolute_returns(eps: Tensor) -> Tensor:
    r"""The proxy :math:`|\epsilon_t|`, less noisy than the square but biased.

    Under a normal innovation, :math:`E|\epsilon_t| = \sigma_t\sqrt{2/\pi}`, so
    comparing it directly to :math:`\sigma_t` understates volatility by about
    20%. Rescale by :math:`\sqrt{\pi/2}` if you intend it as a volatility level.
    """
    _check_1d(eps, "eps")
    return eps.abs()


# --------------------------------------------------------------------------- #
# intraday realized measures
# --------------------------------------------------------------------------- #
def realized_variance_intraday(intraday_returns: Tensor) -> Tensor:
    r"""Realized variance from intraday returns.

    .. math::
        RV_t = \sum_{i=1}^{M} r_{i,t}^2

    Parameters
    ----------
    intraday_returns
        Shape ``(T, M)``: ``M`` intraday returns for each of ``T`` days.

    Notes
    -----
    This is the estimator RealRECH uses at 5-minute sampling, and the one
    GENSHIN uses with ``M = 48``. It ignores the overnight gap, which is why
    RealRECH rescales it — see :func:`hansen_lunde_scale`.
    """
    if intraday_returns.ndim != 2:
        raise ValueError(
            f"intraday_returns must be (T, M), got shape {tuple(intraday_returns.shape)}."
        )
    return intraday_returns.square().sum(dim=1)


def realized_volatility_intraday(intraday_returns: Tensor, percent: bool = False) -> Tensor:
    r"""Realized volatility :math:`\sqrt{RV_t}`, optionally in percent.

    Yu et al. (2025) define their GENSHIN target as
    :math:`RV_t = \sqrt{\sum_i r_{i,t}^2}\times 100\%`; pass ``percent=True``
    for that convention.
    """
    rv = torch.sqrt(realized_variance_intraday(intraday_returns))
    return rv * 100.0 if percent else rv


def bipower_variation(intraday_returns: Tensor) -> Tensor:
    r"""Barndorff-Nielsen & Shephard bipower variation, robust to jumps.

    .. math::
        BV_t = \frac{\pi}{2}\,\frac{M}{M-1}
               \sum_{i=2}^{M} |r_{i,t}|\,|r_{i-1,t}|

    Where realized variance absorbs jumps into the continuous variance, bipower
    variation does not, so :math:`RV_t - BV_t` estimates the jump contribution.
    """
    if intraday_returns.ndim != 2:
        raise ValueError(
            f"intraday_returns must be (T, M), got shape {tuple(intraday_returns.shape)}."
        )
    m = intraday_returns.shape[1]
    if m < 2:
        raise ValueError("bipower variation needs at least 2 intraday returns per day.")
    absolute = intraday_returns.abs()
    products = (absolute[:, 1:] * absolute[:, :-1]).sum(dim=1)
    return (torch.pi / 2.0) * (m / (m - 1)) * products


def hansen_lunde_scale(returns: Tensor, rv: Tensor) -> tuple[Tensor, Tensor]:
    r"""Rescale a realized measure to match the variance of close-to-close returns.

    .. math::
        \tilde\sigma_t^2 = \hat c \cdot rv_t, \qquad
        \hat c = \frac{\sum_t y_t^2}{\sum_t rv_t}

    Realized measures built from trading-hours data miss the overnight move, so
    they systematically understate close-to-close variance. Hansen & Lunde
    (2005) correct this with a single scale factor, and RealRECH (Liu et al.,
    2023, Eq. 13) adopts it before using the realized measure as a proxy for the
    latent conditional variance.

    Returns
    -------
    scaled_rv, c
        The rescaled measure and the scale factor, so the factor can be
        reported alongside the results rather than disappearing into the data
        pipeline.
    """
    _check_1d(returns, "returns")
    _check_1d(rv, "rv")
    if returns.numel() != rv.numel():
        raise ValueError(
            f"returns and rv must align: got {returns.numel()} and {rv.numel()} observations."
        )
    total_rv = rv.sum()
    if float(total_rv) <= 0:
        raise ValueError("Sum of the realized measure must be positive to rescale.")
    c = returns.square().sum() / total_rv
    return c * rv, c


_REGISTRY = {
    "zhao2024": realized_volatility_zhao,
    "wei2025": realized_volatility_wei,
    "rolling_std": rolling_std,
    "squared_returns": squared_returns,
    "absolute_returns": absolute_returns,
}


def get_target(name: str):
    """Look up a daily volatility target by name.

    Examples
    --------
    >>> get_target("wei2025").__name__
    'realized_volatility_wei'
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown target {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
