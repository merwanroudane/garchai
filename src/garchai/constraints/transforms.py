"""Differentiable maps from unconstrained reals to valid GARCH parameters.

Every recursion in ``garchai`` stores *raw* parameters in :math:`\\mathbb{R}` and
applies one of these transforms on every forward pass. Two consequences matter:

1. The optimiser never has to respect a boundary, so plain Adam works.
2. For time-varying-parameter models the transform is applied *at every time
   step* (blueprint sec. 17.3), not once at the end of training.

The transforms are pure functions of tensors; the raw parameters themselves live
on the cells. That keeps constraints testable in isolation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "positive",
    "inverse_positive",
    "bounded",
    "stationary_simplex",
    "gjr_simplex",
    "egarch_persistence",
    "student_df",
    "GARCHConstraint",
    "GJRConstraint",
    "EGARCHConstraint",
]


# --------------------------------------------------------------------------- #
# elementary maps
# --------------------------------------------------------------------------- #
def positive(raw: Tensor, eps: float = 1e-8) -> Tensor:
    """Map :math:`\\mathbb{R} \\to (\\varepsilon, \\infty)` via softplus.

    ``softplus`` is preferred over ``exp`` because its gradient does not blow up
    for large inputs, and over ``ReLU`` because it has no zero-gradient region —
    the same reasoning Liu & So (2020, sec. 2.1) give for their variance readout.
    """
    return F.softplus(raw) + eps


def inverse_positive(value: Tensor | float, eps: float = 1e-8) -> Tensor:
    """Inverse of :func:`positive`, for initialising raw parameters at a target.

    The naive form ``log(expm1(y))`` overflows to ``inf`` for ``y`` beyond about
    709 in float64, because ``expm1`` overflows before the ``log`` can bring it
    back. We use the algebraically identical but stable

    .. math::
        \\operatorname{softplus}^{-1}(y) = y + \\log\\!\\big(1 - e^{-y}\\big),

    which is exact at both ends: it returns ``y`` for large ``y`` and
    ``log(y)`` for small ``y``.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> targets = torch.tensor([1e-6, 1.0, 1e4])
    >>> raw = torch.stack([inverse_positive(float(t)) for t in targets])
    >>> bool(torch.allclose(positive(raw), targets, rtol=1e-9))
    True
    """
    v = torch.as_tensor(value, dtype=torch.get_default_dtype())
    shifted = torch.clamp(v - eps, min=1e-300)
    return shifted + torch.log(-torch.expm1(-shifted))


def bounded(raw: Tensor, lower: float, upper: float) -> Tensor:
    """Map :math:`\\mathbb{R} \\to (\\text{lower}, \\text{upper})` via a sigmoid.

    The interval really is open, and keeping it open takes more than clamping
    the sigmoid. A plain sigmoid saturates to exactly ``0.0`` or ``1.0`` once
    ``|raw|`` exceeds about 37 in float64; nudging it to ``tiny`` does not help,
    because ``lower + (upper - lower) * 1e-308`` rounds straight back to
    ``lower``. We therefore clamp the *result* to the neighbouring representable
    values via ``nextafter``, which is exact regardless of how wide or how
    off-centre the interval is.

    This matters: :func:`student_df` maps onto ``(2, inf)``, and a degrees-of-
    freedom value sitting exactly on 2 is one where the variance of the
    standardised t does not exist.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> out = bounded(torch.tensor([-1e6, 0.0, 1e6]), -3.0, 7.5)
    >>> bool((out > -3.0).all() and (out < 7.5).all())
    True
    """
    if not upper > lower:
        raise ValueError("upper must exceed lower.")
    unit = torch.sigmoid(raw)
    out = lower + (upper - lower) * unit
    lo = torch.full_like(out, lower)
    hi = torch.full_like(out, upper)
    return out.clamp(min=torch.nextafter(lo, hi), max=torch.nextafter(hi, lo))


# --------------------------------------------------------------------------- #
# family-specific parameter sets
# --------------------------------------------------------------------------- #
def stationary_simplex(
    raw_alpha: Tensor,
    raw_beta: Tensor,
    raw_slack: Tensor,
    slack: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Constrain ``(alpha, beta)`` for a covariance-stationary GARCH(1,1).

    Guarantees, exactly and for any real inputs:

    .. math::
        \\alpha \\ge 0,\\quad \\beta \\ge 0,\\quad
        \\alpha + \\beta \\le 1 - \\delta < 1

    The construction is the one recommended in blueprint sec. 17.2: take a
    softmax over ``[raw_alpha, raw_beta, raw_slack]`` and scale the first two
    components by ``1 - delta``. The third component is a genuine free parameter,
    not a constant — it lets the model choose *how far* below unity the
    persistence sits, which a two-way softmax could not express.

    Parameters
    ----------
    raw_alpha, raw_beta, raw_slack
        Unconstrained reals, broadcastable to a common shape. For time-varying
        models these carry a time dimension.
    slack
        The ``delta`` above. Must lie in ``(0, 1)``.

    Returns
    -------
    alpha, beta
        Constrained parameters with the same shape as the broadcast inputs.
    """
    if not 0.0 < slack < 1.0:
        raise ValueError("slack must lie strictly between 0 and 1.")
    stacked = torch.stack(torch.broadcast_tensors(raw_alpha, raw_beta, raw_slack), dim=-1)
    weights = torch.softmax(stacked, dim=-1)
    budget = 1.0 - slack
    return budget * weights[..., 0], budget * weights[..., 1]


def gjr_simplex(
    raw_alpha: Tensor,
    raw_gamma: Tensor,
    raw_beta: Tensor,
    raw_slack: Tensor,
    slack: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Constrain ``(alpha, gamma, beta)`` for a stationary GJR-GARCH(1,1).

    Under innovations symmetric about zero, :math:`E[I(\\epsilon_{t-1}<0)] = 1/2`,
    so the covariance-stationarity condition is

    .. math::
        \\alpha + \\tfrac{1}{2}\\gamma + \\beta < 1 .

    We softmax over four components and set ``gamma = 2 (1-delta) w_1``, which
    makes the left-hand side equal to :math:`(1-\\delta)(w_0 + w_1 + w_2) < 1`.

    Note that this parameterisation enforces ``gamma >= 0``, i.e. the usual
    leverage direction where negative shocks raise volatility at least as much
    as positive ones. Estimated GJR gammas are overwhelmingly positive; if a
    series genuinely needs ``gamma < 0``, use ``enforce_stationarity=False``.
    """
    if not 0.0 < slack < 1.0:
        raise ValueError("slack must lie strictly between 0 and 1.")
    stacked = torch.stack(
        torch.broadcast_tensors(raw_alpha, raw_gamma, raw_beta, raw_slack), dim=-1
    )
    weights = torch.softmax(stacked, dim=-1)
    budget = 1.0 - slack
    return budget * weights[..., 0], 2.0 * budget * weights[..., 1], budget * weights[..., 2]


def egarch_persistence(raw_beta: Tensor, slack: float = 1e-4) -> Tensor:
    """Constrain the EGARCH log-variance persistence to ``|beta| < 1``.

    EGARCH models :math:`\\log h_t`, so positivity is automatic and the only
    stability requirement is that the autoregressive coefficient on
    :math:`\\log h_{t-1}` lies inside the unit circle. Unlike GARCH, ``beta`` may
    legitimately be negative here, so we use ``tanh`` rather than a simplex.
    """
    return (1.0 - slack) * torch.tanh(raw_beta)


def student_df(raw: Tensor, min_df: float = 2.05, max_df: float = 200.0) -> Tensor:
    """Constrain Student-t degrees of freedom to ``(min_df, max_df)``.

    The variance of a standardised Student-t exists only for ``nu > 2``, hence
    the floor. The ceiling exists because the t converges to the normal well
    before ``nu = 200``; without it the optimiser can wander to ``nu = 1e6``
    where the gradient is numerically zero and the parameter is unidentified.
    """
    return bounded(raw, min_df, max_df)


# --------------------------------------------------------------------------- #
# callable bundles
# --------------------------------------------------------------------------- #
class GARCHConstraint:
    """Constraint set for GARCH(1,1): ``omega>0, alpha,beta>=0, alpha+beta<1``.

    Examples
    --------
    >>> import torch
    >>> c = GARCHConstraint()
    >>> p = c(torch.tensor(-2.0), torch.tensor(0.0), torch.tensor(2.0),
    ...       torch.tensor(0.0))
    >>> bool(p["omega"] > 0), bool(p["alpha"] + p["beta"] < 1.0)
    (True, True)
    """

    family = "garch"

    def __init__(self, slack: float = 1e-4, eps: float = 1e-8, enforce: bool = True):
        self.slack = slack
        self.eps = eps
        self.enforce = enforce

    def __call__(
        self,
        raw_omega: Tensor,
        raw_alpha: Tensor,
        raw_beta: Tensor,
        raw_slack: Tensor,
    ) -> dict[str, Tensor]:
        omega = positive(raw_omega, self.eps)
        if self.enforce:
            alpha, beta = stationary_simplex(raw_alpha, raw_beta, raw_slack, self.slack)
        else:
            alpha, beta = positive(raw_alpha, 0.0), positive(raw_beta, 0.0)
        return {"omega": omega, "alpha": alpha, "beta": beta}

    @staticmethod
    def persistence(params: dict[str, Tensor]) -> Tensor:
        return params["alpha"] + params["beta"]


class GJRConstraint:
    """Constraint set for GJR-GARCH(1,1): adds ``alpha + gamma/2 + beta < 1``."""

    family = "gjr"

    def __init__(self, slack: float = 1e-4, eps: float = 1e-8, enforce: bool = True):
        self.slack = slack
        self.eps = eps
        self.enforce = enforce

    def __call__(
        self,
        raw_omega: Tensor,
        raw_alpha: Tensor,
        raw_gamma: Tensor,
        raw_beta: Tensor,
        raw_slack: Tensor,
    ) -> dict[str, Tensor]:
        omega = positive(raw_omega, self.eps)
        if self.enforce:
            alpha, gamma, beta = gjr_simplex(
                raw_alpha, raw_gamma, raw_beta, raw_slack, self.slack
            )
        else:
            alpha = positive(raw_alpha, 0.0)
            beta = positive(raw_beta, 0.0)
            gamma = raw_gamma
        return {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta}

    @staticmethod
    def persistence(params: dict[str, Tensor]) -> Tensor:
        return params["alpha"] + 0.5 * params["gamma"] + params["beta"]


class EGARCHConstraint:
    """Constraint set for EGARCH(1,1,1): only ``|beta| < 1`` is required.

    ``omega``, ``alpha`` and ``gamma`` are unconstrained because the model is
    written for :math:`\\log h_t`; positivity of the variance is structural.
    """

    family = "egarch"

    def __init__(self, slack: float = 1e-4, enforce: bool = True):
        self.slack = slack
        self.enforce = enforce

    def __call__(
        self,
        raw_omega: Tensor,
        raw_alpha: Tensor,
        raw_gamma: Tensor,
        raw_beta: Tensor,
    ) -> dict[str, Tensor]:
        beta = egarch_persistence(raw_beta, self.slack) if self.enforce else raw_beta
        return {"omega": raw_omega, "alpha": raw_alpha, "gamma": raw_gamma, "beta": beta}

    @staticmethod
    def persistence(params: dict[str, Tensor]) -> Tensor:
        return params["beta"]
