r"""GARCH kernels: the neural-network counterparts of GARCH-family recursions.

Zhao et al. (AAAI 2024) observe that each GARCH-family model *is* a neural
network once you look at it correctly, and that the correspondence is exact
rather than analogical:

===================  ==========================================================
GARCH(1,1)           an RNN cell with a scalar hidden state, no output layer and
                     no activation. :math:`\Theta = (\omega,\alpha,\beta)`,
                     :math:`X = (1, \epsilon_{t-1}^2, h_{t-1})`, and
                     :math:`h_t = \Theta\cdot X`.
GJR-GARCH            the same, but the cell consumes :math:`\epsilon_{t-1}`
                     rather than :math:`\epsilon_{t-1}^2`, because the *sign* is
                     needed: :math:`X = (1, \epsilon^2, I(\epsilon)\epsilon^2, h)`.
FI-GARCH             a **1-D convolution**. The fractional differencing expands
                     into weights :math:`\Lambda = (\lambda_0,\dots,\lambda_{T-1})`
                     that do not depend on :math:`t`, so applying them is exactly
                     a kernel sliding over :math:`\{\epsilon_t^2\}`.
===================  ==========================================================

A ``GARCHKernel`` is that counterpart, packaged so it can be dropped into a
larger network. It is what :class:`~garchai.structural.garch_nn.GARCHNN` and
:class:`~garchai.structural.garch_nn.StructuralGARCHLSTM` plug in as
:math:`K_{\text{garch}}` in the paper's Equation 7.

Reference
---------
Zhao, P., Zhu, H., Ng, W. S. H., & Lee, D. L. (2024). From GARCH to Neural
Network for Volatility Forecast. *AAAI 2024*. https://arxiv.org/abs/2402.06642
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from ..constraints.transforms import bounded, gjr_simplex, positive, stationary_simplex
from ..core.config import NumericalConfig

__all__ = [
    "GARCHKernel",
    "GARCHNNKernel",
    "GJRNNKernel",
    "FIGARCHNNKernel",
    "get_kernel",
    "figarch_weights",
]


class GARCHKernel(nn.Module, ABC):
    """A GARCH-family recursion usable as a neural building block.

    The interface differs from :class:`~garchai.cells.base.VarianceCell` in one
    way that matters: a kernel takes the *history* it needs as an explicit
    argument rather than owning a state object, because Zhao et al.'s FI-GARCH
    counterpart is convolutional and needs a window, not a state.
    """

    #: How many lagged squared innovations the kernel needs. 1 for GARCH/GJR.
    lookback: int = 1
    stylized_fact: str = ""

    @abstractmethod
    def transformed_parameters(self) -> dict[str, Tensor]:
        """Econometric parameters after the constraint transform."""

    @abstractmethod
    def forward(self, eps_history: Tensor, h_prev: Tensor) -> Tensor:
        r"""Return :math:`h_t`.

        Parameters
        ----------
        eps_history
            Innovations up to and including :math:`\epsilon_{t-1}`, most recent
            **last**. Kernels with ``lookback == 1`` use only the final element.
        h_prev
            The previous conditional variance :math:`h_{t-1}`. Ignored by the
            convolutional FI-GARCH kernel, whose memory lives in the window.
        """

    @abstractmethod
    def persistence(self) -> Tensor:
        """Volatility persistence implied by the current parameters."""

    def unconditional_variance(self) -> Tensor:
        r"""Long-run variance :math:`\omega/(1-P)`, or ``inf`` if not stationary."""
        params = self.transformed_parameters()
        p = self.persistence()
        denom = 1.0 - p
        if float(denom.detach()) <= 0:
            return torch.tensor(float("inf"), dtype=params["omega"].dtype)
        return params["omega"] / denom

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return ", ".join(
            f"{k}={float(v.detach()):.4f}"
            for k, v in self.transformed_parameters().items()
            if v.numel() == 1
        )


class GARCHNNKernel(GARCHKernel):
    r"""The NN counterpart of GARCH(1,1) — volatility clustering.

    .. math::
        h_t = \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}

    Zhao et al.'s Figure 1b: an RNN cell whose input is
    :math:`(\epsilon_{t-1}^2, h_{t-1})`, whose output feeds back as the next
    input, and which differs from a textbook RNN only by having the output layer
    and the ``tanh`` removed and the state constrained to a scalar.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> k = GARCHNNKernel(0.05, 0.10, 0.85)
    >>> eps = torch.tensor([2.0])
    >>> float(k(eps, torch.tensor(1.0)).detach())   # 0.05 + 0.10*4 + 0.85*1
    1.3
    """

    lookback = 1
    stylized_fact = "volatility clustering"

    def __init__(
        self,
        omega: float = 0.05,
        alpha: float = 0.10,
        beta: float = 0.85,
        enforce_stationarity: bool = True,
        numerical: NumericalConfig | None = None,
    ):
        super().__init__()
        self.numerical = numerical or NumericalConfig()
        self.enforce = enforce_stationarity
        from ..cells.garch_cell import _invert_garch_start
        from ..constraints.transforms import GARCHConstraint

        constraint = GARCHConstraint(
            slack=self.numerical.stationarity_slack,
            eps=self.numerical.eps,
            enforce=enforce_stationarity,
        )
        raw = _invert_garch_start(omega, alpha, beta, constraint)
        self.raw_omega = nn.Parameter(raw["omega"])
        self.raw_alpha = nn.Parameter(raw["alpha"])
        self.raw_beta = nn.Parameter(raw["beta"])
        self.raw_slack = nn.Parameter(raw["slack"])

    def transformed_parameters(self) -> dict[str, Tensor]:
        omega = positive(self.raw_omega, self.numerical.eps)
        if self.enforce:
            alpha, beta = stationary_simplex(
                self.raw_alpha, self.raw_beta, self.raw_slack,
                self.numerical.stationarity_slack,
            )
        else:
            alpha = positive(self.raw_alpha, 0.0)
            beta = positive(self.raw_beta, 0.0)
        return {"omega": omega, "alpha": alpha, "beta": beta}

    def forward(self, eps_history: Tensor, h_prev: Tensor) -> Tensor:
        p = self.transformed_parameters()
        eps_prev = eps_history[..., -1]
        h = p["omega"] + p["alpha"] * eps_prev.square() + p["beta"] * h_prev
        return h.clamp(min=self.numerical.variance_floor)

    def persistence(self) -> Tensor:
        p = self.transformed_parameters()
        return p["alpha"] + p["beta"]


class GJRNNKernel(GARCHKernel):
    r"""The NN counterpart of GJR-GARCH — asymmetry.

    .. math::
        h_t = \omega + \alpha\epsilon_{t-1}^2
              + \lambda\, I(\epsilon_{t-1}<0)\,\epsilon_{t-1}^2 + \beta h_{t-1}

    Zhao et al. name the leverage coefficient :math:`\lambda` here; this package
    calls it ``gamma`` throughout for consistency with
    :class:`~garchai.cells.garch_cell.GJRCell` and with the wider literature.
    The deviation is cosmetic and recorded on the model's provenance.

    This is the kernel Zhao et al. select for their headline GARCH-LSTM results.
    """

    lookback = 1
    stylized_fact = "asymmetry (leverage)"

    def __init__(
        self,
        omega: float = 0.05,
        alpha: float = 0.05,
        gamma: float = 0.08,
        beta: float = 0.85,
        enforce_stationarity: bool = True,
        numerical: NumericalConfig | None = None,
    ):
        super().__init__()
        self.numerical = numerical or NumericalConfig()
        self.enforce = enforce_stationarity
        from ..cells.garch_cell import _invert_gjr_start
        from ..constraints.transforms import GJRConstraint

        constraint = GJRConstraint(
            slack=self.numerical.stationarity_slack,
            eps=self.numerical.eps,
            enforce=enforce_stationarity,
        )
        raw = _invert_gjr_start(omega, alpha, gamma, beta, constraint)
        self.raw_omega = nn.Parameter(raw["omega"])
        self.raw_alpha = nn.Parameter(raw["alpha"])
        self.raw_gamma = nn.Parameter(raw["gamma"])
        self.raw_beta = nn.Parameter(raw["beta"])
        self.raw_slack = nn.Parameter(raw["slack"])

    def transformed_parameters(self) -> dict[str, Tensor]:
        omega = positive(self.raw_omega, self.numerical.eps)
        if self.enforce:
            alpha, gamma, beta = gjr_simplex(
                self.raw_alpha, self.raw_gamma, self.raw_beta, self.raw_slack,
                self.numerical.stationarity_slack,
            )
        else:
            alpha = positive(self.raw_alpha, 0.0)
            beta = positive(self.raw_beta, 0.0)
            gamma = self.raw_gamma
        return {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta}

    def forward(self, eps_history: Tensor, h_prev: Tensor) -> Tensor:
        p = self.transformed_parameters()
        eps_prev = eps_history[..., -1]
        sq = eps_prev.square()
        indicator = (eps_prev < 0).to(sq.dtype)
        h = p["omega"] + p["alpha"] * sq + p["gamma"] * indicator * sq + p["beta"] * h_prev
        return h.clamp(min=self.numerical.variance_floor)

    def persistence(self) -> Tensor:
        p = self.transformed_parameters()
        return p["alpha"] + 0.5 * p["gamma"] + p["beta"]


def figarch_weights(phi: Tensor, d: Tensor, beta: Tensor, truncation: int) -> Tensor:
    r"""Expand the FI-GARCH lag polynomial into convolution weights.

    FI-GARCH(1, d, 1) writes the conditional variance as

    .. math::
        \left[1 - \frac{1-\phi B}{1-\beta B}(1-B)^d\right]\epsilon_t^2
            = \left(\sum_{k=0}^{\infty}\lambda_k B^k\right)\epsilon_t^2 ,

    and Zhao et al. (Eq. 3) note that because :math:`\lambda_k` does not depend
    on :math:`t`, applying it is a 1-D convolution over
    :math:`\{\epsilon_t^2\}` with kernel size equal to the truncation.

    The expansion is built in three steps:

    1. Fractional differencing, :math:`\delta_0 = 1`,
       :math:`\delta_j = \delta_{j-1}\,\frac{j-1-d}{j}`.
    2. Multiply by :math:`(1-\phi B)`: :math:`g_j = \delta_j - \phi\,\delta_{j-1}`.
    3. Divide by :math:`(1-\beta B)`, i.e. :math:`u_j = g_j + \beta u_{j-1}`.

    Then :math:`\lambda_k = -u_k`, and :math:`\lambda_0 = 0` automatically since
    :math:`u_0 = 1`. That zero is not a convenience — it is what stops
    :math:`\epsilon_t^2` entering its own variance, which would be look-ahead.

    Returns
    -------
    Tensor
        Shape ``(truncation,)``, ordered ``[lambda_0, ..., lambda_{T-1}]``.
    """
    if truncation < 2:
        raise ValueError("FI-GARCH truncation must be at least 2.")
    dtype = phi.dtype

    deltas = [torch.ones((), dtype=dtype)]
    for j in range(1, truncation):
        deltas.append(deltas[-1] * (j - 1 - d) / j)

    u_prev = torch.zeros((), dtype=dtype)
    weights = []
    for j in range(truncation):
        g = deltas[j] - (phi * deltas[j - 1] if j >= 1 else torch.zeros((), dtype=dtype))
        u = g + beta * u_prev
        # lambda(B) = 1 - u(B): the leading 1 cancels u_0 = 1 exactly, so
        # lambda_0 = 0 and eps_t never enters its own conditional variance.
        weights.append(1.0 - u if j == 0 else -u)
        u_prev = u
    return torch.stack(weights)


class FIGARCHNNKernel(GARCHKernel):
    r"""The NN counterpart of FI-GARCH — long memory, as a 1-D convolution.

    .. math::
        h_t = \frac{\omega}{1-\beta}
              + \sum_{k=1}^{T-1}\lambda_k\,\epsilon_{t-k}^2

    The three GARCH parameters :math:`(\phi, d, \beta)` generate the whole
    kernel :math:`\Lambda` through :func:`figarch_weights`, so the convolution
    has ``T`` taps but only three degrees of freedom. That is the point of the
    correspondence: the long memory is *structured*, not fitted tap by tap.

    Parameters
    ----------
    truncation
        Kernel size ``T``. The hyperbolic decay of :math:`\lambda_k` is slow, so
        truncating too early throws away exactly the long memory the model
        exists to capture; 100-1000 is typical.

    Notes
    -----
    Unlike GARCH and GJR, non-negativity of every :math:`\lambda_k` is not
    guaranteed by simple box constraints on :math:`(\phi, d, \beta)` — the
    Bollerslev-Mikkelsen inequalities are considerably more involved. We
    constrain the three parameters to their usual ranges and clamp the resulting
    variance at the numerical floor, and :meth:`weights_are_non_negative` reports
    whether the sufficient condition actually holds at the current estimate.
    """

    stylized_fact = "long memory"

    def __init__(
        self,
        omega: float = 0.05,
        phi: float = 0.2,
        d: float = 0.4,
        beta: float = 0.6,
        truncation: int = 250,
        numerical: NumericalConfig | None = None,
    ):
        super().__init__()
        self.numerical = numerical or NumericalConfig()
        self.truncation = truncation
        self.lookback = truncation

        from ..constraints.transforms import inverse_positive

        dt = torch.float64
        self.raw_omega = nn.Parameter(inverse_positive(omega, self.numerical.eps).to(dt))
        self.raw_phi = nn.Parameter(torch.logit(torch.tensor(min(max(phi, 1e-4), 0.999), dtype=dt)))
        self.raw_d = nn.Parameter(torch.logit(torch.tensor(min(max(d, 1e-4), 0.999), dtype=dt)))
        self.raw_beta = nn.Parameter(torch.logit(torch.tensor(min(max(beta, 1e-4), 0.999), dtype=dt)))

    def transformed_parameters(self) -> dict[str, Tensor]:
        return {
            "omega": positive(self.raw_omega, self.numerical.eps),
            "phi": bounded(self.raw_phi, 0.0, 1.0),
            "d": bounded(self.raw_d, 0.0, 1.0),
            "beta": bounded(self.raw_beta, 0.0, 1.0),
        }

    def weights(self) -> Tensor:
        """The convolution kernel :math:`\\Lambda` at the current parameters."""
        p = self.transformed_parameters()
        return figarch_weights(p["phi"], p["d"], p["beta"], self.truncation)

    def weights_are_non_negative(self) -> bool:
        """Whether every :math:`\\lambda_k` is non-negative at the current fit."""
        return bool((self.weights()[1:] >= 0).all())

    def forward(self, eps_history: Tensor, h_prev: Tensor) -> Tensor:
        r"""Convolve the kernel over the window. ``h_prev`` is unused by design.

        Alignment matters here and is easy to get wrong. ``eps_history`` is
        most-recent-last, so with a window of length :math:`L` we need
        :math:`\sum_{k=1}^{L}\lambda_k\epsilon_{t-k}^2`, where
        :math:`\epsilon_{t-k}^2` sits at ``window[L-k]``. Reversing
        ``lambda_1..lambda_L`` lines the two up. Only ``truncation - 1`` taps are
        usable, because :math:`\lambda_0` multiplies :math:`\epsilon_t`, which is
        not in the conditioning set.
        """
        p = self.transformed_parameters()
        weights = self.weights()
        usable = self.truncation - 1
        window = eps_history[..., -usable:].square()
        length = window.shape[-1]
        taps = weights[1 : length + 1].flip(-1)
        intercept = p["omega"] / (1.0 - p["beta"]).clamp(min=1e-8)
        return (intercept + (window * taps).sum(dim=-1)).clamp(
            min=self.numerical.variance_floor
        )

    def persistence(self) -> Tensor:
        r"""Sum of the convolution weights, the long-memory analogue of ``alpha+beta``."""
        return self.weights()[1:].sum()

    def unconditional_variance(self) -> Tensor:
        r"""Infinite whenever :math:`d > 0`.

        FI-GARCH with a genuine fractional-differencing parameter has no finite
        unconditional variance — that is precisely what "long memory" means
        here. Returning ``inf`` rather than a plausible-looking number keeps the
        distinction visible in reported output.
        """
        d = self.transformed_parameters()["d"]
        if float(d.detach()) > 1e-8:
            return torch.tensor(float("inf"), dtype=d.dtype)
        p = self.transformed_parameters()
        denom = (1.0 - self.persistence()).clamp(min=1e-12)
        return p["omega"] / denom


_KERNELS = {
    "garch": GARCHNNKernel,
    "gjr": GJRNNKernel,
    "figarch": FIGARCHNNKernel,
}


def get_kernel(name: str, **kwargs) -> GARCHKernel:
    """Build a GARCH kernel by name.

    Examples
    --------
    >>> get_kernel("gjr").stylized_fact
    'asymmetry (leverage)'
    """
    key = name.lower().replace("-", "").replace("_", "")
    if key not in _KERNELS:
        raise KeyError(f"Unknown kernel {name!r}. Available: {sorted(_KERNELS)}")
    return _KERNELS[key](**kwargs)
