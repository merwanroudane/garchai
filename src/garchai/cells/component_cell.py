r"""Component GARCH (Engle & Lee, 1999) — the csGARCH of GENSHIN's model bank.

Splits the conditional variance into a slowly-moving **permanent** component and
a fast **transitory** deviation from it:

.. math::
    q_t &= \omega + \rho\, q_{t-1}
           + \varphi\big(\epsilon_{t-1}^2 - \sigma_{t-1}^2\big) \\
    \sigma_t^2 &= q_t + \alpha\big(\epsilon_{t-1}^2 - q_{t-1}\big)
                      + \beta\big(\sigma_{t-1}^2 - q_{t-1}\big)

A plain GARCH has one persistence, :math:`\alpha+\beta`, and so one answer to
"how long does a shock last". Empirically that is too crude: volatility has a
slow level that drifts over months and fast swings around it that die within
days. csGARCH gives each its own decay rate — :math:`\rho` for the permanent
component, :math:`\alpha+\beta` for the transitory one — and the model is only
sensible when :math:`\alpha+\beta < \rho`, i.e. the transitory part really is
the faster of the two.

``arch`` has no component model, which is why this is built here rather than
adapted: GENSHIN's 60-model feature bank includes csGARCH at four lag
structures and three distributions.

Reference
---------
Engle, R. F., & Lee, G. G. J. (1999). A permanent and transitory component model
of stock return volatility. In *Cointegration, Causality and Forecasting: A
Festschrift in Honour of Clive W. J. Granger*, 475-497.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..constraints.transforms import bounded, inverse_positive, positive
from ..core.config import NumericalConfig
from .base import CellState, InitMethod, VarianceCell

__all__ = ["ComponentGARCHCell"]


class ComponentGARCHCell(VarianceCell):
    r"""csGARCH(1,1): permanent plus transitory variance components.

    Parameters
    ----------
    omega
        Intercept of the permanent component. Its long-run level is
        :math:`\omega/(1-\rho)`.
    rho
        Persistence of the permanent component, constrained to ``(0, 1)``.
        Typically very close to 1 — the whole point is that this component is
        slow.
    phi
        Loading of the permanent component on the squared-return surprise.
    alpha, beta
        Transitory dynamics. Constrained so ``alpha + beta < rho``, which is
        what keeps the decomposition meaningful.

    Attributes
    ----------
    The permanent path :math:`q_t` is returned in the cell state's ``extra``
    dict, so it can be plotted against :math:`\sigma_t^2` — the standard way of
    showing the decomposition, and the reason to fit this model at all.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> cell = ComponentGARCHCell(omega=0.01, rho=0.99, phi=0.05,
    ...                           alpha=0.06, beta=0.80)
    >>> p = {k: float(v.detach()) for k, v in cell.transformed_parameters().items()}
    >>> p["alpha"] + p["beta"] < p["rho"]     # transitory decays faster
    True
    >>> h, state = cell.filter(torch.randn(200, dtype=torch.float64))
    >>> bool((h > 0).all()), "q" in (state.extra or {})
    (True, True)
    """

    constraint_family = "component"

    def __init__(
        self,
        omega: float = 0.01,
        rho: float = 0.99,
        phi: float = 0.05,
        alpha: float = 0.06,
        beta: float = 0.80,
        enforce_stationarity: bool = True,
        init_method: InitMethod = "arch_backcast",
        backcast_lambda: float | None = None,
        sample_n: int = 10,
        numerical: NumericalConfig | None = None,
    ):
        super().__init__(
            init_method=init_method,
            backcast_lambda=backcast_lambda,
            sample_n=sample_n,
            numerical=numerical,
        )
        self.enforce = enforce_stationarity
        dt = torch.float64
        eps = self.numerical.eps

        if enforce_stationarity and alpha + beta >= rho:
            raise ValueError(
                f"csGARCH requires alpha + beta < rho for the transitory component to "
                f"decay faster than the permanent one; got {alpha} + {beta} >= {rho}. "
                "Pass enforce_stationarity=False to fit it anyway."
            )

        self.raw_omega = nn.Parameter(inverse_positive(omega, eps).to(dt))
        self.raw_rho = nn.Parameter(
            torch.logit(torch.tensor(min(max(rho, 1e-4), 0.9999), dtype=dt))
        )
        self.raw_phi = nn.Parameter(inverse_positive(max(phi, 1e-8), 0.0).to(dt))
        # alpha and beta share rho's budget via a simplex, so alpha + beta < rho
        # holds for any optimiser step rather than only at the starting values.
        share = (alpha + beta) / rho
        weights = torch.tensor(
            [alpha / (alpha + beta), beta / (alpha + beta)], dtype=dt
        ).clamp(min=1e-8)
        self.raw_split = nn.Parameter(torch.log(weights))
        self.raw_share = nn.Parameter(
            torch.logit(torch.tensor(min(max(share, 1e-4), 0.9999), dtype=dt))
        )

    def transformed_parameters(self) -> dict[str, Tensor]:
        omega = positive(self.raw_omega, self.numerical.eps)
        rho = bounded(self.raw_rho, 0.0, 1.0 - self.numerical.stationarity_slack)
        phi = positive(self.raw_phi, 0.0)
        if self.enforce:
            share = bounded(self.raw_share, 0.0, 1.0 - self.numerical.stationarity_slack)
            split = torch.softmax(self.raw_split, dim=-1)
            transitory = rho * share
            alpha, beta = transitory * split[0], transitory * split[1]
        else:
            alpha = positive(self.raw_split[0], 0.0)
            beta = positive(self.raw_split[1], 0.0)
        return {"omega": omega, "rho": rho, "phi": phi, "alpha": alpha, "beta": beta}

    def initial_state(self, eps: Tensor) -> CellState:
        """Start both components at the same level, as is conventional."""
        state = super().initial_state(eps)
        state.extra = {"q": state.h.clone()}
        return state

    def step(
        self, eps_prev: Tensor, state: CellState, params: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, CellState]:
        p = params if params is not None else self.transformed_parameters()
        q_prev = (state.extra or {}).get("q", state.h)
        sq = eps_prev.square()

        q = p["omega"] + p["rho"] * q_prev + p["phi"] * (sq - state.h)
        q = q.clamp(min=self.numerical.variance_floor)
        h = q + p["alpha"] * (sq - q_prev) + p["beta"] * (state.h - q_prev)
        h = h.clamp(min=self.numerical.variance_floor)
        return h, CellState(h=h, extra={"q": q})

    def persistence(self) -> Tensor:
        r"""The **permanent** component's persistence, :math:`\rho`.

        Not :math:`\alpha+\beta`. In a csGARCH the long-run behaviour is
        governed by :math:`\rho`, and reporting the transitory persistence here
        would understate how long shocks actually last — often badly, since
        :math:`\rho` is typically near 0.99 while :math:`\alpha+\beta` may be
        0.85. :meth:`transitory_persistence` gives the other one.
        """
        return self.transformed_parameters()["rho"]

    def transitory_persistence(self) -> Tensor:
        r"""The transitory component's decay rate, :math:`\alpha+\beta`."""
        p = self.transformed_parameters()
        return p["alpha"] + p["beta"]

    def unconditional_variance(self) -> Tensor:
        r"""Long-run variance :math:`\omega/(1-\rho)`, set by the permanent component."""
        p = self.transformed_parameters()
        denominator = 1.0 - p["rho"]
        if float(denominator.detach()) <= 0:
            return torch.tensor(float("inf"), dtype=p["omega"].dtype)
        return p["omega"] / denominator

    def decompose(self, eps: Tensor) -> dict[str, Tensor]:
        """Return the permanent and transitory paths separately.

        The transitory part is :math:`\\sigma_t^2 - q_t`, which is signed: it
        goes negative when volatility sits below its slow-moving level, and that
        is exactly the information a single-component GARCH cannot express.
        """
        params = self.transformed_parameters()
        state = self.initial_state(eps)
        variances, permanents = [state.h], [(state.extra or {})["q"]]
        for t in range(1, eps.numel()):
            h, state = self.step(eps[t - 1], state, params)
            variances.append(h)
            permanents.append(state.extra["q"])
        variance = torch.stack(variances)
        permanent = torch.stack(permanents)
        return {
            "variance": variance,
            "permanent": permanent,
            "transitory": variance - permanent,
        }
