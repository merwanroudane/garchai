r"""Rodikov & Antulov-Fantulin: the σ-LSTM cell (2022) and the σ-Cell family (2023).

Two papers, one idea: build the *shape* of a GARCH recursion into a recurrent
cell, then let a network learn the parameters non-linearly.

The σ-Cell (2023, Eqs. 13-14) is the cleaner statement:

.. math::
    \tilde\sigma^2_t &= \phi\big(\tilde\sigma^2_{t-1}W_s
                        + x^2_{t-1}W_r + b_h\big) \\
    \sigma^2_t &= \phi_o\big(\tilde\sigma^2_t W_o + b_o\big)

Compare with GARCH(1,1), :math:`\sigma^2_t = \omega + \alpha x^2_{t-1} +
\beta\sigma^2_{t-1}`: the same two inputs and the same additive structure, but a
hidden state of width 10 instead of 1, and a non-linearity :math:`\phi` wrapped
around the sum. GARCH is the special case where :math:`\phi` is the identity and
the hidden state is scalar.

Five variants, in increasing order of machinery:

============  =========================================================
``SigmaCell``      Eqs. 13-14. Deterministic.
``SigmaCellN``     Eqs. 15-17. Adds a **stochastic layer**:
                   :math:`\tilde x_{t-1} = x_{t-1} - N(0,\sigma_{t-1})`.
``SigmaCellRL``    Eqs. 18-22. Residuals from a learned RNN mean
                   :math:`G` instead of a random draw.
``SigmaCellNTV``   Eqs. 23-30. :math:`W_s, W_r` become **time-varying**,
                   generated from :math:`x_{t-1}` and split in half.
``SigmaCellRLTV``  Eqs. 31-38. Both at once.
============  =========================================================

All share the log-likelihood of Eq. 39,
:math:`L = \sum_t \big[\log\sigma^2_t + (x_t - G(x_t))^2/\sigma^2_t\big]`.

Resolving the 2022 paper's ambiguities
--------------------------------------
The earlier σ-LSTM paper is four pages and leaves three things unstated. All
three are settled by reading the 2023 paper, which is the same authors
elaborating the same construction — so these are resolutions, not guesses:

1. **Eq. 8**, ``o_t = N(0, W_o[C_t^2])``, does not say whether the second
   argument is a variance or a standard deviation. σ-Cell-N's Eq. 15 writes
   :math:`N(0, \sigma_{t-1})` with :math:`\sigma` the *volatility*, so the
   convention is **standard deviation**.
2. **Eq. 11**, :math:`\hat\sigma^2_t = \langle C_t\rangle^2`, uses an unglossed
   mean operator. σ-Cell's Eq. 14 replaces it with a learned linear readout
   :math:`\phi_o(\tilde\sigma^2_t W_o + b_o)`; the mean is the unweighted
   special case. We implement the learned readout and expose ``readout="mean"``
   for the 2022 form.
3. **Eq. 12**, :math:`L = \sum[-\ln\hat\sigma^2 - r^2/\hat\sigma^2]`, is
   labelled a loss but is written to be *maximised*. Eq. 39 of the 2023 paper is
   its exact negative and is minimised. So Eq. 12 is a log-likelihood with a
   loss's name.

References
----------
Rodikov, G., & Antulov-Fantulin, N. (2022). Volatility-inspired σ-LSTM cell.
https://arxiv.org/abs/2205.07022

Rodikov, G., & Antulov-Fantulin, N. (2023). Introducing the σ-Cell: Unifying
GARCH, Stochastic Fluctuations and Evolving Mechanisms in RNN-based Volatility
Forecasting. https://arxiv.org/abs/2309.01565
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.base import VolatilityModel
from ..core.config import ModelConfig
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance
from ..distributions import Normal
from ..distributions.base import Distribution
from ..mean.models import MeanModel, get_mean_model

__all__ = [
    "adjusted_softplus",
    "SigmaCell",
    "SigmaCellN",
    "SigmaCellRL",
    "SigmaCellNTV",
    "SigmaCellRLTV",
    "SigmaLSTM",
    "sigma_cell_loss",
]

_SIGMA_CELL = dict(
    paper_id="rodikov2023_sigma_cell",
    title=(
        "Introducing the sigma-Cell: Unifying GARCH, Stochastic Fluctuations and "
        "Evolving Mechanisms in RNN-based Volatility Forecasting"
    ),
    authors=("German Rodikov", "Nino Antulov-Fantulin"),
    year=2023,
    paper_url="https://arxiv.org/abs/2309.01565",
    code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
    implementation_status=ImplementationStatus.PAPER_FAITHFUL,
)


def adjusted_softplus(x: Tensor, beta: float = 1.0) -> Tensor:
    r"""The paper's Adjusted Softplus (Eq. 41).

    .. math::
        \mathrm{AS}(x) = \max\!\left(0,\;
            \frac{\frac{1}{\beta}\log(1+e^{\beta x}) - \frac{\log 2}{\beta}}
                 {\frac{1}{\beta}\log(1+e^{\beta}) - \frac{\log 2}{\beta}}\right)

    Standard softplus, shifted so :math:`\mathrm{AS}(0) = 0`, clamped at zero,
    and normalised so :math:`\mathrm{AS}(1) = 1`. The three properties matter in
    combination: non-negativity keeps the variance valid, the shift means a zero
    pre-activation contributes nothing, and the normalisation puts the cell on
    the same scale as a GARCH coefficient — so a learned weight near 0.9 means
    what a :math:`\beta` near 0.9 means.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = torch.tensor([-5.0, 0.0, 1.0])
    >>> [round(float(v), 6) for v in adjusted_softplus(x)]
    [0.0, 0.0, 1.0]
    """
    shift = math.log(2.0) / beta
    scale = math.log(1.0 + math.exp(beta)) / beta - shift
    return torch.clamp(F.softplus(x, beta=beta) - shift, min=0.0) / scale


def sigma_cell_loss(x: Tensor, variance: Tensor, mean: Tensor | None = None) -> Tensor:
    r"""Equation 39: :math:`L=\sum_t[\log\sigma^2_t + (x_t-G(x_t))^2/\sigma^2_t]`.

    This is twice the Gaussian negative log-likelihood minus its constant, so it
    ranks models identically to :func:`garchai.losses.gaussian_nll` while
    reporting the paper's numbers. ``mean`` is :math:`G(x_t)`, the learned mean
    for the variants that have one; ``None`` treats the series as zero-mean.
    """
    residual = x if mean is None else x - mean
    return (torch.log(variance) + residual.square() / variance).sum()


class _SigmaBase(VolatilityModel):
    """Shared scaffolding for the σ-Cell family."""

    def __init__(
        self,
        hidden_size: int = 10,
        config: ModelConfig | None = None,
        beta: float = 1.0,
        variance_floor: float = 1e-6,
        initial_variance: float = 1.0,
    ):
        super().__init__(config or ModelConfig(mean="zero", distribution="normal"))
        self.hidden_size = hidden_size
        self.beta = beta
        # The paper's output activation is a plain ReLU (sec. 3.1), which can
        # return *exactly* zero. Eq. 39 then evaluates x^2/0, and the loss goes
        # to 1e14 before training has taken a single useful step. The package
        # default floor of 1e-12 is far too small to absorb that, so these cells
        # carry their own, larger floor. It is a guard, not a model change: a
        # trained sigma-Cell never approaches it.
        self.variance_floor = variance_floor
        # sigma-Cell-N feeds sigma_{t-1} into the residual draw (Eq. 15), which
        # is a positive feedback loop: a large variance produces a large
        # residual, which produces a larger variance. At random initialisation
        # that can run to 1e22 within a few hundred steps. The ceiling is a
        # numerical guard on an unstable-at-init specification, not a change to
        # the model; a trained cell stays orders of magnitude below it.
        self._variance_ceiling = float(
            torch.exp(torch.tensor(self.config.numerical.max_log_variance))
        )
        self._mean = get_mean_model(self.config.mean)
        dt = torch.float64

        # Eq. 13. Ws and Wr act elementwise on the hidden state and on the
        # scalar squared input, which is what makes the paper's parameter count
        # (41 at hidden_size=10) come out exactly.
        self.W_s = nn.Parameter(torch.empty(hidden_size, dtype=dt))
        self.W_r = nn.Parameter(torch.empty(hidden_size, dtype=dt))
        self.b_h = nn.Parameter(torch.zeros(hidden_size, dtype=dt))
        self.W_o = nn.Parameter(torch.empty(hidden_size, dtype=dt))
        # The paper specifies zero biases (sec. 3.8). For b_o that is not
        # workable: the Adjusted Softplus clamps a negative pre-activation to
        # zero, so a zero hidden state meets a zero bias and Eq. 14 returns
        # sigma^2 = 0 -- an infinite loss at initialisation, with a gradient
        # that carries no information. b_o therefore starts at the expected
        # variance level. Every other bias is zero, as specified.
        self.b_o = nn.Parameter(torch.full((1,), float(initial_variance), dtype=dt))
        self._xavier_init()

    def _xavier_init(self) -> None:
        """Xavier uniform weights and zero biases (paper sec. 3.8).

        With one departure: ``W_o`` is initialised **positive**. A signed ``W_o``
        makes the ReLU readout return zero for roughly half of all random
        initialisations, which does not merely start the model badly - it starts
        it at an infinite loss, from which the gradient carries no information.
        """
        bound = math.sqrt(6.0 / (1 + self.hidden_size))
        for weight in (self.W_s, self.W_r):
            nn.init.uniform_(weight, -bound, bound)
        nn.init.uniform_(self.W_o, 0.1 * bound, bound)

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    @property
    def cell(self):
        return self

    def distribution(self) -> Distribution:
        return Normal()

    def supports_analytic_forecast(self) -> bool:
        return False

    def phi(self, x: Tensor) -> Tensor:
        """Hidden activation: the Adjusted Softplus."""
        return adjusted_softplus(x, self.beta)

    def phi_o(self, x: Tensor) -> Tensor:
        """Output activation: the paper's ReLU, lifted off zero (see __init__)."""
        return (torch.relu(x) + self.variance_floor).clamp(max=self._variance_ceiling)

    def _readout(self, hidden: Tensor) -> Tensor:
        """Equation 14."""
        return self.phi_o((hidden * self.W_o).sum() + self.b_o).squeeze()

    def _variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
        """A sigma-Cell has no closed-form persistence or long-run variance.

        Its recursion is non-linear in the hidden state, so neither quantity has
        the meaning it carries in a GARCH. Reporting ``nan`` is more honest than
        reporting a number computed from a formula that does not apply. What is
        reported instead is the mean of ``W_s``, the closest analogue of
        ``beta``, and the realised mean of the fitted variance path.
        """
        params = {
            "mean W_s (beta-like)": float(self.W_s.detach().mean()),
            "mean W_r (alpha-like)": float(self.W_r.detach().mean()),
            "b_o": float(self.b_o.detach().squeeze()),
        }
        result = getattr(self, "result_", None)
        long_run = (
            float(result.conditional_variance.mean()) if result is not None else float("nan")
        )
        return params, float("nan"), long_run

    def _residual(self, x_prev: Tensor, sigma_prev: Tensor, index: int) -> Tensor:
        """The variant-specific :math:`\\tilde x_{t-1}`. Plain σ-Cell uses ``x``."""
        return x_prev

    def _weights(self, x_prev: Tensor) -> tuple[Tensor, Tensor]:
        """The variant-specific ``(W_s, W_r)``. Fixed unless time-varying."""
        return self.W_s, self.W_r

    def _mean_prediction(self, x_prev: Tensor, index: int) -> Tensor | None:
        """:math:`G(x_t)` for the variants that learn a mean. ``None`` otherwise."""
        return None

    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(r)
        n = eps.numel()
        dtype = eps.dtype

        hidden = torch.zeros(self.hidden_size, dtype=dtype)
        sigma2 = eps.var(unbiased=True).clamp(min=self.config.numerical.variance_floor)
        variances = [sigma2]
        self._reset_state(n, dtype)

        for t in range(1, n):
            residual = self._residual(eps[t - 1], torch.sqrt(sigma2), t)
            w_s, w_r = self._weights(eps[t - 1])
            hidden = self.phi(hidden * w_s + residual.square() * w_r + self.b_h)
            sigma2 = self._readout(hidden)
            variances.append(sigma2)
        return torch.stack(variances), eps, mu

    def _reset_state(self, n: int, dtype: torch.dtype) -> None:
        """Hook for variants carrying extra recurrent state."""

    def negative_loglikelihood(self, r: Tensor) -> Tensor:
        """Equation 39, with :math:`G` supplied by the RL variants."""
        variance, eps, _ = self.filter(r)
        return sigma_cell_loss(eps, variance, getattr(self, "_last_mean", None))


class SigmaCell(_SigmaBase):
    r"""σ-Cell (Eqs. 13-14): the deterministic GARCH-shaped RNN cell.

    At ``hidden_size=10`` this has exactly the paper's 41 parameters:
    :math:`W_s, W_r, b_h, W_o` at 10 each, plus a scalar :math:`b_o`.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> from garchai.core import set_seed
    >>> set_seed(0)
    >>> model = SigmaCell(hidden_size=10)
    >>> sum(p.numel() for p in model.parameters())     # paper: 41
    41
    >>> h, _, _ = model.filter(torch.randn(100) * 0.9)
    >>> bool((h > 0).all())
    True
    """

    provenance = Provenance(**_SIGMA_CELL, equations="Eqs. 13-14, 39, 41")


class SigmaCellN(_SigmaBase):
    r"""σ-Cell-N (Eqs. 15-17): adds the stochastic layer.

    .. math::
        \tilde x_{t-1} = x_{t-1} - N(0, \sigma_{t-1})

    The draw uses the *previous* volatility as its standard deviation, so
    uncertainty propagates forward the way it does in a GARCH innovation rather
    than being injected as fixed-scale noise. In evaluation mode the draw is
    replaced by its mean (zero), which makes the filter deterministic and the
    forecast reproducible.
    """

    provenance = Provenance(
        **_SIGMA_CELL,
        equations="Eqs. 15-17, 39, 41",
        deviations=(
            "N(0, sigma) is read as a standard-deviation parameterisation, "
            "following Eq. 15's use of sigma (volatility) rather than sigma^2.",
            "In eval() the stochastic draw is replaced by its mean, so filtering "
            "and forecasting are reproducible.",
        ),
    )

    def _residual(self, x_prev: Tensor, sigma_prev: Tensor, index: int) -> Tensor:
        if not self.training:
            return x_prev
        return x_prev - torch.randn((), dtype=x_prev.dtype) * sigma_prev


class SigmaCellRL(_SigmaBase):
    r"""σ-Cell-RL (Eqs. 18-22): residuals from a learned recurrent mean.

    .. math::
        h_t = \varphi(x_{t-1}W_{xh} + h_{t-1}W_{hh} + b_h), \quad
        G(h_t) = \varphi(h_t W_{ho} + b_o), \quad
        \tilde x_{t-1} = x_{t-1} - G(h_t)

    Where σ-Cell-N subtracts *noise*, this subtracts a *prediction*. The variance
    equation then sees genuine forecast errors, so the model learns the mean and
    the variance jointly — which is what a GARCH with a non-trivial mean equation
    does, expressed as a network.
    """

    provenance = Provenance(**_SIGMA_CELL, equations="Eqs. 18-22, 39, 41")

    def __init__(self, hidden_size: int = 10, config: ModelConfig | None = None,
                 mean_hidden: int = 10, beta: float = 1.0, **kwargs):
        super().__init__(hidden_size, config, beta, **kwargs)
        dt = torch.float64
        self.mean_hidden = mean_hidden
        self.W_xh = nn.Parameter(torch.randn(mean_hidden, dtype=dt) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(mean_hidden, mean_hidden, dtype=dt) * 0.1)
        self.b_mh = nn.Parameter(torch.zeros(mean_hidden, dtype=dt))
        self.W_ho = nn.Parameter(torch.randn(mean_hidden, dtype=dt) * 0.1)
        self.b_mo = nn.Parameter(torch.zeros(1, dtype=dt))

    def _reset_state(self, n: int, dtype: torch.dtype) -> None:
        self._h_mean = torch.zeros(self.mean_hidden, dtype=dtype)
        self._means: list[Tensor] = [torch.zeros((), dtype=dtype)]

    def _residual(self, x_prev: Tensor, sigma_prev: Tensor, index: int) -> Tensor:
        # Eqs. 18-19: tanh throughout, per the paper's choice of phi = tanh here.
        self._h_mean = torch.tanh(x_prev * self.W_xh + self.W_hh @ self._h_mean + self.b_mh)
        prediction = torch.tanh((self._h_mean * self.W_ho).sum() + self.b_mo).squeeze()
        self._means.append(prediction)
        return x_prev - prediction

    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        variance, eps, mu = super().filter(r)
        # G(x_t) enters the likelihood (Eq. 39), so keep it aligned with eps.
        self._last_mean = torch.stack(self._means)
        return variance, eps, mu


class SigmaCellNTV(SigmaCellN):
    r"""σ-Cell-NTV (Eqs. 23-30): time-varying :math:`W_{s,t}, W_{r,t}`.

    .. math::
        w_{t-1} = \tilde\varphi(Wx_{t-1} + b), \quad
        W_{s,t} = \pi_1(w_{t-1}), \quad W_{r,t} = \pi_2(w_{t-1})

    A single projection produces a :math:`2n`-vector which is split down the
    middle: the first half weights the lagged variance, the second the squared
    residual. So the model's *persistence* and its *shock sensitivity* both move
    with the data — the GARCH analogue would be :math:`\alpha_t` and
    :math:`\beta_t` varying, as in Yin & Barucca, but generated pointwise from
    :math:`x_{t-1}` rather than from a latent state.
    """

    provenance = Provenance(**_SIGMA_CELL, equations="Eqs. 23-30, 39, 41")

    def __init__(self, hidden_size: int = 10, config: ModelConfig | None = None,
                 beta: float = 1.0, **kwargs):
        super().__init__(hidden_size, config, beta, **kwargs)
        dt = torch.float64
        # Eq. 23: one projection to 2n, then split (Eqs. 26-27).
        self.W_tv = nn.Parameter(torch.randn(2 * hidden_size, dtype=dt) * 0.1)
        self.b_tv = nn.Parameter(torch.zeros(2 * hidden_size, dtype=dt))
        # Eqs. 24-25 replace the fixed W_s, W_r outright, so the inherited
        # parameters are dead weight. Freezing them keeps them out of the
        # optimiser and out of the reported parameter count.
        self.W_s.requires_grad_(False)
        self.W_r.requires_grad_(False)

    def _weights(self, x_prev: Tensor) -> tuple[Tensor, Tensor]:
        w = torch.tanh(x_prev * self.W_tv + self.b_tv)
        return w[: self.hidden_size], w[self.hidden_size :]


class SigmaCellRLTV(SigmaCellRL):
    """σ-Cell-RLTV (Eqs. 31-38): learned residuals *and* time-varying weights."""

    provenance = Provenance(**_SIGMA_CELL, equations="Eqs. 31-38, 39, 41")

    def __init__(self, hidden_size: int = 10, config: ModelConfig | None = None,
                 mean_hidden: int = 10, beta: float = 1.0):
        super().__init__(hidden_size, config, mean_hidden, beta)
        dt = torch.float64
        self.W_tv = nn.Parameter(torch.randn(2 * hidden_size, dtype=dt) * 0.1)
        self.b_tv = nn.Parameter(torch.zeros(2 * hidden_size, dtype=dt))
        self.W_s.requires_grad_(False)   # replaced by Eqs. 32-33
        self.W_r.requires_grad_(False)

    def _weights(self, x_prev: Tensor) -> tuple[Tensor, Tensor]:
        w = torch.tanh(x_prev * self.W_tv + self.b_tv)
        return w[: self.hidden_size], w[self.hidden_size :]


class SigmaLSTM(VolatilityModel):
    r"""The σ-LSTM cell of Rodikov & Antulov-Fantulin (2022), Eqs. 4-12.

    .. math::
        f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f) \\
        i_t &= \sigma(W_i[h_{t-1}, x_t] + b_i) \\
        \tilde C_t &= \tanh(W_C[h_{t-1}, x_t] + b_C) \\
        C_t &= f_t \odot C_{t-1} + i_t \odot \tilde C_t \\
        o_t &= N\big(0, W_o[C_t^2]\big) \\
        h_t &= o_t \odot \varphi(C_t)

    An LSTM whose output gate is replaced by a **stochastic** draw scaled by the
    cell state — the cell state plays the role of a volatility, and the sampled
    gate the role of an innovation. The design goal is the HAR-RV cascade's
    long-and-short memory structure expressed inside a single cell.

    The three under-specified points in the 2022 paper are resolved against the
    2023 σ-Cell paper; see the module docstring. ``readout`` selects between the
    2022 mean form and the 2023 learned form.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> from garchai.core import set_seed
    >>> set_seed(0)
    >>> model = SigmaLSTM(hidden_size=8)
    >>> model.eval()                        # deterministic in eval mode
    SigmaLSTM(distribution=normal, unfitted)
    >>> h, _, _ = model.filter(torch.randn(80) * 0.9)
    >>> bool((h > 0).all())
    True
    """

    provenance = Provenance(
        paper_id="rodikov2022_sigma_lstm",
        title="Volatility-inspired sigma-LSTM cell",
        authors=("German Rodikov", "Nino Antulov-Fantulin"),
        year=2022,
        paper_url="https://arxiv.org/abs/2205.07022",
        equations="Eqs. 4-12",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.PAPER_FAITHFUL,
        deviations=(
            "Eq. 8's N(0, W_o[C_t^2]) does not say whether the argument is a "
            "variance or a standard deviation. Resolved as a standard deviation "
            "by reference to Eq. 15 of the 2023 sigma-Cell paper.",
            "Eq. 11's mean operator <C_t> is unglossed. Implemented as the "
            "learned readout of the 2023 paper's Eq. 14; readout='mean' gives "
            "the 2022 unweighted form.",
            "Eq. 12 is labelled a loss but written to be maximised; it is the "
            "exact negative of Eq. 39 of the 2023 paper. Minimised here.",
        ),
    )

    def __init__(
        self,
        hidden_size: int = 10,
        config: ModelConfig | None = None,
        readout: Literal["learned", "mean"] = "learned",
    ):
        super().__init__(config or ModelConfig(mean="zero", distribution="normal"))
        self.hidden_size = hidden_size
        self.readout = readout
        self._mean = get_mean_model(self.config.mean)
        dt = torch.float64

        for gate in ("f", "i", "C"):
            setattr(self, f"W_{gate}",
                    nn.Parameter(torch.randn(hidden_size, hidden_size + 1, dtype=dt) * 0.1))
            bias = torch.ones(hidden_size, dtype=dt) if gate == "f" else torch.zeros(
                hidden_size, dtype=dt
            )
            setattr(self, f"b_{gate}", nn.Parameter(bias))
        # Eq. 8: W_o maps the squared cell state to the gate's scale.
        self.W_o = nn.Parameter(torch.abs(torch.randn(hidden_size, dtype=dt)) * 0.1)
        if readout == "learned":
            self.W_readout = nn.Parameter(torch.abs(torch.randn(hidden_size, dtype=dt)) * 0.1)
            self.b_readout = nn.Parameter(torch.ones(1, dtype=dt))
        self.variance_floor = 1e-6

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    @property
    def cell(self):
        return self

    def distribution(self) -> Distribution:
        return Normal()

    def supports_analytic_forecast(self) -> bool:
        return False

    def _variance(self, c: Tensor) -> Tensor:
        if self.readout == "mean":
            return c.mean().square() + self.variance_floor                    # Eq. 11
        return torch.relu(
            (c.square() * self.W_readout).sum() + self.b_readout
        ).squeeze() + self.variance_floor

    def _variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
        """No closed-form persistence: the recursion is non-linear in the state."""
        result = getattr(self, "result_", None)
        long_run = (
            float(result.conditional_variance.mean()) if result is not None else float("nan")
        )
        return {"hidden_size": float(self.hidden_size)}, float("nan"), long_run

    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(r)
        n = eps.numel()
        dtype = eps.dtype

        h = torch.zeros(self.hidden_size, dtype=dtype)
        c = torch.full((self.hidden_size,), 0.1, dtype=dtype)
        variances = [self._variance(c)]

        for t in range(1, n):
            joint = torch.cat([h, eps[t - 1].reshape(1)])
            f = torch.sigmoid(self.W_f @ joint + self.b_f)
            i = torch.sigmoid(self.W_i @ joint + self.b_i)
            c_tilde = torch.tanh(self.W_C @ joint + self.b_C)
            c = f * c + i * c_tilde

            # Eq. 8: a stochastic output gate whose scale is the cell state.
            # Standard-deviation parameterisation, per the 2023 paper's Eq. 15.
            scale = torch.sqrt((self.W_o.abs() * c.square()).clamp(min=1e-12))
            o = scale * torch.randn(self.hidden_size, dtype=dtype) if self.training else scale
            h = o * torch.tanh(c)                                   # Eq. 9
            variances.append(self._variance(c))                     # Eq. 11
        return torch.stack(variances), eps, mu

    def negative_loglikelihood(self, r: Tensor) -> Tensor:
        """Equation 12, negated so it is minimised (see the module docstring)."""
        variance, eps, _ = self.filter(r)
        return sigma_cell_loss(eps, variance)
