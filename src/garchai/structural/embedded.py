r"""Wei, Yang & Cui (2025): GARCH dynamics embedded in recurrent gating.

Where Zhao et al. replace the LSTM's *output gate* with a GARCH kernel, Wei et
al. inject a GARCH signal into the hidden-state update itself, and do it
differently for the two architectures:

**GARCH-GRU** (Eqs. 15-20) — additive fusion, so each hidden unit is nudged
individually:

.. math::
    g_t &= \phi\big(\omega_0 + \alpha\epsilon_{t-1}^2 + \beta\sigma^2_{t-1}\big) \\
    z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \\
    r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \\
    \tilde h_t &= \tanh\big(W_h x_t + U_h(r_t\odot h_{t-1}) + b_h\big) \\
    \hat h_t &= (1-z_t)\odot\tilde h_t + z_t\odot h_{t-1} \\
    h_t &= \tanh\big(\hat h_t + \gamma g_t\big)

**GARCH-LSTM** (Eqs. 22-29) — multiplicative fusion, so the whole hidden state
is scaled at once:

.. math::
    \tilde h_t = o_t\cdot\tanh(c_t), \qquad
    h_t = \big(1 + w\tanh(g_t)\big)\odot\tilde h_t

Here :math:`\phi(u) = W_g u + b_g` is a linear projection lifting the scalar
GARCH output to the hidden dimension, and :math:`\gamma`, :math:`w` are learned
scalars governing how much the econometrics is allowed to say.

The paper offers a useful reading of the GRU form: inverting the ``tanh`` gives
:math:`\gamma^{-1}\tanh^{-1}(h_t) = g_t + \gamma^{-1}\hat h_t`, so on the
pre-activation scale the GARCH signal is simply an additive, volatility-driven
bias shift.

A deviation worth stating plainly
---------------------------------
Equation 19 reads :math:`\hat h_t = (1-z_t)\odot\tilde h_t + z_t\odot h_{t-1}`,
which **swaps** the roles of the update gate relative to the paper's own recap
of the standard GRU in Equation 13. We implement Equation 19 as printed, because
that is the model the paper specifies and evaluates. ``gate_convention="standard"``
selects the textbook orientation, and the difference is recorded on the model's
provenance.

Reference
---------
Wei, J., Yang, S., & Cui, Z. (2025). Unified GARCH-Recurrent Neural Networks in
Financial Volatility Forecasting. https://arxiv.org/abs/2504.09380 (v2)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..constraints.transforms import positive, stationary_simplex, student_df
from ..core.base import VolatilityModel
from ..core.config import ModelConfig
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance
from ..distributions import Normal, StudentT
from ..distributions.base import Distribution
from ..mean.models import MeanModel, get_mean_model

__all__ = ["EmbeddedGARCHGRUCell", "EmbeddedGARCHLSTMCell", "EmbeddedGARCHGRU", "EmbeddedGARCHLSTM"]

_WEI = dict(
    paper_id="wei2025_garch_rnn",
    title="Unified GARCH-Recurrent Neural Networks in Financial Volatility Forecasting",
    authors=("Jingyi Wei", "Steve Yang", "Zhenyu Cui"),
    year=2025,
    paper_url="https://arxiv.org/abs/2504.09380",
    code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
    implementation_status=ImplementationStatus.PAPER_FAITHFUL,
)


class _GARCHSignal(nn.Module):
    r"""The shared GARCH component :math:`g_t = \phi(\omega_0 + \alpha\epsilon^2 + \beta\sigma^2)`.

    The reparameterisation enforcing :math:`\alpha+\beta<1` is applied here, so
    the econometric parameters stay interpretable after joint optimisation with
    the network — which is the property the paper repeatedly emphasises.
    """

    def __init__(
        self,
        hidden_size: int,
        omega: float = 0.05,
        alpha: float = 0.10,
        beta: float = 0.85,
        enforce_stationarity: bool = True,
        slack: float = 1e-4,
        eps: float = 1e-8,
    ):
        super().__init__()
        from ..cells.garch_cell import _invert_garch_start
        from ..constraints.transforms import GARCHConstraint

        self.slack, self.eps, self.enforce = slack, eps, enforce_stationarity
        constraint = GARCHConstraint(slack=slack, eps=eps, enforce=enforce_stationarity)
        raw = _invert_garch_start(omega, alpha, beta, constraint)
        self.raw_omega = nn.Parameter(raw["omega"])
        self.raw_alpha = nn.Parameter(raw["alpha"])
        self.raw_beta = nn.Parameter(raw["beta"])
        self.raw_slack = nn.Parameter(raw["slack"])

        # phi(u) = W_g u + b_g, the projection to the hidden dimension.
        self.W_g = nn.Parameter(torch.randn(hidden_size, dtype=torch.float64) * 0.1)
        self.b_g = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float64))

    def transformed_parameters(self) -> dict[str, Tensor]:
        omega = positive(self.raw_omega, self.eps)
        if self.enforce:
            alpha, beta = stationary_simplex(
                self.raw_alpha, self.raw_beta, self.raw_slack, self.slack
            )
        else:
            alpha = positive(self.raw_alpha, 0.0)
            beta = positive(self.raw_beta, 0.0)
        return {"omega": omega, "alpha": alpha, "beta": beta}

    def garch_variance(self, eps_prev: Tensor, sigma2_prev: Tensor) -> Tensor:
        """The scalar GARCH(1,1) update, before projection."""
        p = self.transformed_parameters()
        return p["omega"] + p["alpha"] * eps_prev.square() + p["beta"] * sigma2_prev

    def forward(self, eps_prev: Tensor, sigma2_prev: Tensor) -> Tensor:
        """The projected signal :math:`g_t`, shape ``(hidden_size,)``."""
        return self.W_g * self.garch_variance(eps_prev, sigma2_prev) + self.b_g

    def persistence(self) -> Tensor:
        p = self.transformed_parameters()
        return p["alpha"] + p["beta"]


class EmbeddedGARCHGRUCell(nn.Module):
    """One step of the GARCH-GRU (Eqs. 15-20)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        gate_convention: Literal["paper", "standard"] = "paper",
        gamma: float = 0.5,
        **garch_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_convention = gate_convention
        dt = torch.float64
        scale = hidden_size**-0.5

        self.W_z = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dt) * scale)
        self.U_z = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dt) * scale)
        self.b_z = nn.Parameter(torch.zeros(hidden_size, dtype=dt))
        self.W_r = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dt) * scale)
        self.U_r = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dt) * scale)
        self.b_r = nn.Parameter(torch.zeros(hidden_size, dtype=dt))
        self.W_h = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dt) * scale)
        self.U_h = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dt) * scale)
        self.b_h = nn.Parameter(torch.zeros(hidden_size, dtype=dt))

        self.garch = _GARCHSignal(hidden_size, **garch_kwargs)
        self.gamma = nn.Parameter(torch.tensor(float(gamma), dtype=dt))

    def forward(
        self, x_t: Tensor, h_prev: Tensor, eps_prev: Tensor, sigma2_prev: Tensor
    ) -> Tensor:
        g = self.garch(eps_prev, sigma2_prev)
        z = torch.sigmoid(self.W_z @ x_t + self.U_z @ h_prev + self.b_z)
        r = torch.sigmoid(self.W_r @ x_t + self.U_r @ h_prev + self.b_r)
        h_tilde = torch.tanh(self.W_h @ x_t + self.U_h @ (r * h_prev) + self.b_h)
        if self.gate_convention == "paper":
            # Eq. 19 exactly as printed.
            h_hat = (1.0 - z) * h_tilde + z * h_prev
        else:
            h_hat = (1.0 - z) * h_prev + z * h_tilde
        return torch.tanh(h_hat + self.gamma * g)


class EmbeddedGARCHLSTMCell(nn.Module):
    """One step of the GARCH-LSTM (Eqs. 22-29)."""

    def __init__(self, input_size: int, hidden_size: int, w: float = 0.5, **garch_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        dt = torch.float64
        scale = hidden_size**-0.5

        for gate in ("f", "i", "o", "c"):
            setattr(self, f"W_{gate}",
                    nn.Parameter(torch.randn(hidden_size, input_size, dtype=dt) * scale))
            setattr(self, f"U_{gate}",
                    nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dt) * scale))
            bias = torch.ones(hidden_size, dtype=dt) if gate == "f" else torch.zeros(
                hidden_size, dtype=dt
            )
            setattr(self, f"b_{gate}", nn.Parameter(bias))

        self.garch = _GARCHSignal(hidden_size, **garch_kwargs)
        self.w = nn.Parameter(torch.tensor(float(w), dtype=dt))

    def forward(
        self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor, eps_prev: Tensor, sigma2_prev: Tensor
    ) -> tuple[Tensor, Tensor]:
        g = self.garch(eps_prev, sigma2_prev)
        f = torch.sigmoid(self.W_f @ x_t + self.U_f @ h_prev + self.b_f)
        i = torch.sigmoid(self.W_i @ x_t + self.U_i @ h_prev + self.b_i)
        o = torch.sigmoid(self.W_o @ x_t + self.U_o @ h_prev + self.b_o)
        c_tilde = torch.tanh(self.W_c @ x_t + self.U_c @ h_prev + self.b_c)
        c = f * c_prev + i * c_tilde
        h_tilde = o * torch.tanh(c)
        h = (1.0 + self.w * torch.tanh(g)) * h_tilde
        return h, c


class _EmbeddedModel(VolatilityModel):
    r"""Shared scaffolding for the two Wei et al. architectures.

    The output layer is the paper's: ``softplus`` then square root, so the
    volatility is non-negative by construction. Because :math:`\sigma^2_{t-1}`
    feeds back into :math:`g_t`, the recursion is genuinely closed — the network
    and the econometric component see each other at every step, which is the
    whole claim of the paper.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        config: ModelConfig | None = None,
        include_return_input: bool = True,
        n_exog: int = 0,
    ):
        super().__init__(config)
        self._mean = get_mean_model(self.config.mean)
        self.hidden_size = hidden_size
        self.include_return_input = include_return_input
        self.n_exog = n_exog
        self.input_size = (1 if include_return_input else 0) + 1 + n_exog
        self.readout = nn.Linear(hidden_size, 1, dtype=torch.float64)
        if self.config.distribution == "student-t":
            self.raw_df = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    def distribution(self) -> Distribution:
        if self.config.distribution == "normal":
            return Normal()
        if self.config.distribution == "student-t":
            return StudentT(df=student_df(self.raw_df, self.config.numerical.min_df))
        raise ValueError("Embedded GARCH-RNN models support 'normal' or 'student-t'.")

    def _distribution_parameters(self) -> dict[str, float]:
        if self.config.distribution == "student-t":
            return {
                "df (nu)": float(
                    student_df(self.raw_df, self.config.numerical.min_df).detach()
                )
            }
        return {}

    def supports_analytic_forecast(self) -> bool:
        return False

    def _variance_from_hidden(self, h: Tensor) -> Tensor:
        """Paper's constrained output layer: softplus, then implicitly sqrt."""
        return F.softplus(self.readout(h).squeeze(-1)).clamp(
            min=self.config.numerical.variance_floor
        )

    def _inputs(self, eps: Tensor, sigma2_prev: Tensor, t: int, exog: Tensor | None) -> Tensor:
        parts = []
        if self.include_return_input:
            parts.append(eps[t - 1].reshape(1))
        parts.append(sigma2_prev.reshape(1))
        if exog is not None:
            parts.append(exog[t - 1])
        return torch.cat(parts)

    @property
    def cell(self):
        return self._cell

    def persistence_report(self) -> dict[str, float]:
        """The econometric parameters, still interpretable after joint training."""
        p = self._cell.garch.transformed_parameters()
        return {
            **{k: float(v.detach()) for k, v in p.items()},
            "persistence": float(self._cell.garch.persistence().detach()),
        }


class EmbeddedGARCHGRU(_EmbeddedModel):
    r"""GARCH-GRU: additive fusion of the GARCH signal into the hidden state.

    The paper's preferred model — it trains roughly three times faster than the
    LSTM variant while matching or beating its accuracy, and it is the more
    robust of the two at longer horizons.

    Parameters
    ----------
    hidden_size
        Hidden dimension. The paper tunes this with Optuna.
    gate_convention
        ``"paper"`` implements Eq. 19 as printed; ``"standard"`` uses the
        textbook GRU orientation. See the module docstring.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> model = EmbeddedGARCHGRU(hidden_size=4)
    >>> h, eps, mu = model.filter(torch.randn(50) * 0.9)
    >>> h.shape, bool((h > 0).all())
    (torch.Size([50]), True)
    """

    provenance = Provenance(
        **_WEI,
        equations="Eqs. 15-21 (GARCH-GRU)",
        deviations=(
            "Eq. 19 swaps the update-gate roles relative to the paper's own "
            "Eq. 13 recap of the standard GRU. Implemented as printed in Eq. 19; "
            "gate_convention='standard' selects the textbook orientation.",
        ),
    )

    def __init__(
        self,
        hidden_size: int = 16,
        config: ModelConfig | None = None,
        gate_convention: Literal["paper", "standard"] = "paper",
        gamma: float = 0.5,
        include_return_input: bool = True,
        n_exog: int = 0,
        **garch_kwargs,
    ):
        super().__init__(hidden_size, config, include_return_input, n_exog)
        self._cell = EmbeddedGARCHGRUCell(
            self.input_size, hidden_size, gate_convention=gate_convention,
            gamma=gamma, **garch_kwargs,
        )

    def filter(self, r: Tensor, exog: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(r)
        n = eps.numel()
        dtype = eps.dtype

        sigma2 = eps.var(unbiased=True).clamp(min=self.config.numerical.variance_floor)
        h = torch.zeros(self.hidden_size, dtype=dtype)
        variances = [sigma2]

        for t in range(1, n):
            x = self._inputs(eps, sigma2, t, exog)
            h = self._cell(x, h, eps[t - 1], sigma2)
            sigma2 = self._variance_from_hidden(h)
            variances.append(sigma2)
        return torch.stack(variances), eps, mu


class EmbeddedGARCHLSTM(_EmbeddedModel):
    r"""GARCH-LSTM: multiplicative fusion of the GARCH signal.

    The counterpart to :class:`EmbeddedGARCHGRU`, built for a like-for-like
    comparison. Where the GRU adds :math:`\gamma g_t` to individual hidden
    units, this scales the entire hidden state by
    :math:`1 + w\tanh(g_t)` — a global amplification in high-volatility regimes
    rather than a per-unit adjustment. The ``tanh`` bounds the scaling, which is
    what keeps gradients well behaved when volatility spikes.

    The paper notes this fusion is "partially inspired by" Zhao et al. (2024),
    and the resemblance to :class:`~garchai.structural.garch_nn.StructuralGARCHLSTM`
    is real. The difference is where the signal enters: Zhao modulates the
    *variance* directly, Wei modulates the *hidden state* which is then read out
    into a variance.
    """

    provenance = Provenance(
        **_WEI,
        equations="Eqs. 22-29 (GARCH-LSTM)",
        deviations=(
            "The paper does not constrain w; garchai leaves it free, matching "
            "the paper, and relies on the tanh(g_t) bound for stability.",
        ),
    )

    def __init__(
        self,
        hidden_size: int = 16,
        config: ModelConfig | None = None,
        w: float = 0.5,
        include_return_input: bool = True,
        n_exog: int = 0,
        **garch_kwargs,
    ):
        super().__init__(hidden_size, config, include_return_input, n_exog)
        self._cell = EmbeddedGARCHLSTMCell(self.input_size, hidden_size, w=w, **garch_kwargs)

    def filter(self, r: Tensor, exog: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(r)
        n = eps.numel()
        dtype = eps.dtype

        sigma2 = eps.var(unbiased=True).clamp(min=self.config.numerical.variance_floor)
        h = torch.zeros(self.hidden_size, dtype=dtype)
        c = torch.zeros(self.hidden_size, dtype=dtype)
        variances = [sigma2]

        for t in range(1, n):
            x = self._inputs(eps, sigma2, t, exog)
            h, c = self._cell(x, h, c, eps[t - 1], sigma2)
            sigma2 = self._variance_from_hidden(h)
            variances.append(sigma2)
        return torch.stack(variances), eps, mu


def _embedded_variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
    """Report the GARCH sub-component, which is what stays interpretable.

    The "cell" of an embedded model is a recurrent network, not a GARCH
    recursion, so the base class's default would find nothing to report. What a
    reader wants here is exactly the econometric part: the paper's claim is that
    ``omega``, ``alpha`` and ``beta`` remain economically meaningful *after*
    joint optimisation with the network weights, and this is where that claim
    becomes checkable.
    """
    garch = self._cell.garch
    params = {k: float(v.detach()) for k, v in garch.transformed_parameters().items()}
    persistence = float(garch.persistence().detach())
    denominator = 1.0 - persistence
    long_run = params["omega"] / denominator if denominator > 0 else float("inf")
    return params, persistence, long_run


_EmbeddedModel._variance_diagnostics = _embedded_variance_diagnostics
