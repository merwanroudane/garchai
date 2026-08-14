r"""Loss functions for volatility models.

Three families live here and they are not interchangeable:

**Likelihood losses** score a *distribution*. They are the right objective when
the model is probabilistic, and they are what makes the fitted :math:`h_t` a
conditional variance rather than an arbitrary positive number. Zhao et al.
(2024, Table 4) show this concretely: swapping MSE for a likelihood loss cuts
their RNN's MAE on the S&P 500 from 0.765 to 0.200.

**Point losses** (MSE, MAE, Huber) score a forecast against a *proxy*. They are
what most of the hybrid literature reports, and they inherit whatever noise the
proxy carries.

**Robust losses** (QLIKE) are consistent for the latent variance even when the
proxy is noisy, which is why they dominate the forecast-evaluation literature.

Losses that need a variance take ``h``; losses that compare volatilities take
``sigma``. Mixing the two is the single easiest way to be wrong by a square
root, so the argument names say which is expected and the docstrings repeat it.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "gaussian_nll",
    "qlike",
    "qlike_normalised",
    "mse",
    "mae",
    "rmse",
    "huber",
    "pinball",
    "smape",
    "r2_oos",
    "garch_informed_loss",
]


def _reduce(x: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"Unknown reduction {reduction!r}. Use 'mean', 'sum' or 'none'.")


# --------------------------------------------------------------------------- #
# likelihood
# --------------------------------------------------------------------------- #
def gaussian_nll(eps: Tensor, h: Tensor, reduction: str = "sum") -> Tensor:
    r"""Gaussian negative log-likelihood of innovations given a variance path.

    .. math::
        -\log L_t = \tfrac{1}{2}\left[\log 2\pi + \log h_t
                    + \frac{\epsilon_t^2}{h_t}\right]

    Provided as a free function for convenience; for anything other than the
    normal, go through :mod:`garchai.distributions`, whose ``nll`` handles the
    standardisation correctly.

    Parameters
    ----------
    eps
        Innovations :math:`\epsilon_t`, **not** returns.
    h
        Conditional **variance** :math:`h_t`, not volatility.
    """
    from ..distributions import Normal

    return Normal().nll(eps, h, reduction=reduction)


# --------------------------------------------------------------------------- #
# robust
# --------------------------------------------------------------------------- #
def qlike(proxy_variance: Tensor, h: Tensor, reduction: str = "mean") -> Tensor:
    r"""QLIKE loss, in the convention this package uses throughout.

    .. math::
        \mathrm{QLIKE}_t = \frac{\widehat{\sigma^2_t}}{h_t} + \log h_t

    Both arguments are **variances**, not volatilities.

    QLIKE is *robust* in the sense of Patton (2011): its ranking of forecasts is
    unaffected by noise in the volatility proxy, so squared returns can be used
    as ``proxy_variance`` without biasing model comparison. MSE shares that
    property; MAE does not, which is why MAE-based rankings in the hybrid
    literature should be read with care.

    Note that this form is not minimised at zero, and its level has no
    interpretation on its own — only differences between models do. Use
    :func:`qlike_normalised` if you want a loss whose minimum is zero.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> h = torch.tensor([1.0, 2.0])
    >>> float(qlike(h, h))          # perfect forecast: 1 + log(h), averaged
    1.3465735902799727
    """
    if proxy_variance.shape != h.shape:
        raise ValueError(
            f"proxy_variance and h must have the same shape, got "
            f"{tuple(proxy_variance.shape)} and {tuple(h.shape)}."
        )
    return _reduce(proxy_variance / h + torch.log(h), reduction)


def qlike_normalised(proxy_variance: Tensor, h: Tensor, reduction: str = "mean") -> Tensor:
    r"""QLIKE shifted so that a perfect forecast scores zero.

    .. math::
        \mathrm{QLIKE}^{0}_t = \frac{\widehat{\sigma^2_t}}{h_t}
            - \log\frac{\widehat{\sigma^2_t}}{h_t} - 1

    This is the Bregman-divergence form. It is non-negative, equal to zero
    exactly when :math:`h_t = \widehat{\sigma^2_t}`, and differs from
    :func:`qlike` by a term that does not depend on the model — so the two rank
    models identically.
    """
    ratio = proxy_variance / h
    return _reduce(ratio - torch.log(ratio) - 1.0, reduction)


# --------------------------------------------------------------------------- #
# point losses
# --------------------------------------------------------------------------- #
def mse(target: Tensor, prediction: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error. Both arguments must be on the same scale."""
    return _reduce((target - prediction).square(), reduction)


def rmse(target: Tensor, prediction: Tensor) -> Tensor:
    """Root mean squared error."""
    return torch.sqrt(mse(target, prediction, reduction="mean"))


def mae(target: Tensor, prediction: Tensor, reduction: str = "mean") -> Tensor:
    """Mean absolute error."""
    return _reduce((target - prediction).abs(), reduction)


def huber(target: Tensor, prediction: Tensor, delta: float = 1.0,
          reduction: str = "mean") -> Tensor:
    r"""Huber loss: quadratic within ``delta``, linear outside it.

    A compromise between MSE and MAE for volatility targets, where a handful of
    crisis observations can otherwise dominate the gradient entirely.
    """
    if delta <= 0:
        raise ValueError("delta must be positive.")
    residual = (target - prediction).abs()
    quadratic = 0.5 * residual.square()
    linear = delta * (residual - 0.5 * delta)
    return _reduce(torch.where(residual <= delta, quadratic, linear), reduction)


def pinball(target: Tensor, quantile_forecast: Tensor, alpha: float,
            reduction: str = "mean") -> Tensor:
    r"""Pinball (quantile) loss, the proper scoring rule for a VaR forecast.

    .. math::
        L_t = \begin{cases}
            \alpha\,(y_t - q_t), & y_t \ge q_t \\
            (\alpha - 1)(y_t - q_t), & y_t < q_t
        \end{cases}

    Wei et al. (2025) report it alongside violation ratios, and it is the
    sharper of the two: a violation ratio only counts exceedances, whereas the
    pinball loss also penalises a VaR that is needlessly far from the returns.

    Parameters
    ----------
    target
        Realised returns.
    quantile_forecast
        The forecast :math:`\alpha`-quantile, on the return scale (so negative
        for small ``alpha``).
    alpha
        Quantile level in ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    diff = target - quantile_forecast
    return _reduce(torch.maximum(alpha * diff, (alpha - 1.0) * diff), reduction)


def smape(target: Tensor, prediction: Tensor, reduction: str = "mean") -> Tensor:
    r"""Symmetric mean absolute percentage error, in the 0-200% convention.

    .. math::
        \mathrm{SMAPE}_t = \frac{200\,|y_t - \hat y_t|}{|y_t| + |\hat y_t|}

    Reported by Wei et al. (2025). Beware that a 0-100% variant (halving this)
    also circulates; the convention is stated here so comparisons are unambiguous.
    """
    denom = (target.abs() + prediction.abs()).clamp(min=torch.finfo(target.dtype).eps)
    return _reduce(200.0 * (target - prediction).abs() / denom, reduction)


def r2_oos(target: Tensor, prediction: Tensor) -> Tensor:
    r"""Out-of-sample :math:`R^2` against the mean of the evaluation sample.

    .. math::
        R^2_{oos} = 1 - \frac{\sum_t (y_t-\hat y_t)^2}{\sum_t (y_t-\bar y)^2}

    Unbounded below, which is not a defect: Wei et al. (2025, Table 3) report
    values near :math:`-500` for classical GARCH at the one-day horizon, and
    that is genuine information about how badly a variance forecast can track a
    realized-volatility target it was never fitted to.
    """
    ss_res = (target - prediction).square().sum()
    ss_tot = (target - target.mean()).square().sum()
    return 1.0 - ss_res / ss_tot.clamp(min=torch.finfo(target.dtype).eps)


# --------------------------------------------------------------------------- #
# informed
# --------------------------------------------------------------------------- #
def garch_informed_loss(
    prediction: Tensor,
    truth: Tensor,
    garch_prediction: Tensor,
    lam: float = 0.01,
    reduction: str = "mean",
) -> Tensor:
    r"""The GINN objective of Xu et al. (2024, Eq. 17).

    .. math::
        \mathcal{L} = \lambda\,\mathrm{MSE}\big(\sigma_t^2, \hat\sigma^2_{t,GINN}\big)
              + (1-\lambda)\,\mathrm{MSE}\big(\hat\sigma^2_{t,GARCH},
                                              \hat\sigma^2_{t,GINN}\big)

    The network is pulled toward the observed variance *and* toward a GARCH
    model's forecast — the same idea as a physics-informed network, with GARCH
    supplying the structural prior instead of a differential equation.

    The paper's tuned value is :math:`\lambda = 0.01`, which is worth pausing
    over: it puts 99% of the weight on agreeing with GARCH and only 1% on the
    observed variance. Xu et al. also report GINN-0, the :math:`\lambda = 0`
    limit, which trains purely to imitate GARCH and still performs comparably.
    That is a statement about how noisy :math:`(r_t-\hat\mu_t)^2` is as a target,
    not a claim that the observed data is unimportant.

    Parameters
    ----------
    prediction
        The network's variance forecast.
    truth
        The observed variance proxy, :math:`(r_t - \hat\mu_t)^2` in the paper.
    garch_prediction
        A GARCH model's variance forecast for the same periods. Detached from
        the graph internally, since it is a fixed target rather than something
        the network should push back on.
    lam
        Weight on the data term. ``0`` reproduces GINN-0.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1].")
    data_term = (truth - prediction).square()
    structure_term = (garch_prediction.detach() - prediction).square()
    return _reduce(lam * data_term + (1.0 - lam) * structure_term, reduction)
