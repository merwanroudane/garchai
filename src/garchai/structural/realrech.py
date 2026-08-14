r"""Liu, Wang, Tran & Kohn (2023): Deep Learning Enhanced Realized GARCH.

Also known as RealRECH. Where :mod:`garchai.structural.rech` lets a recurrent
network drive the constant of a *return*-only GARCH, this does the same for a
**Realized GARCH**, so the network also gets to see the realized measure:

.. math::
    y_t &= \sigma_t\epsilon_t \\
    \sigma_t^2 &= g(\omega_t) + \gamma\, rv_{t-1} + \beta\sigma^2_{t-1}
        \qquad\text{(Eq. 7b)} \\
    rv_t &= \xi + \varphi\sigma_t^2 + \tau(\epsilon_t) + u_t
        \qquad\text{(Eq. 7c, measurement equation)} \\
    \omega_t &= \beta_0 + \beta_1 h_t, \qquad h_t = \mathrm{LSTM}(x_t) \\
    x_t &= \big(\omega_{t-1},\, y_{t-1},\, \sigma^2_{t-1},\, rv_{t-1}\big)

with the leverage function :math:`\tau(\epsilon) = \tau_1\epsilon +
\tau_2(\epsilon^2-1)` and :math:`g = \mathrm{ReLU}`.

The measurement equation is what makes this a Realized GARCH rather than a GARCH
with an extra regressor. A realized measure is a *noisy proxy* for the latent
variance, and Eq. 7c models that noise explicitly instead of pretending
:math:`rv_t = \sigma_t^2`. Its :math:`\tau(\cdot)` also picks up the leverage
effect, which is why the variance equation itself can stay symmetric. Both
equations contribute to the likelihood (Eq. 8), so the realized measure informs
the parameters twice over.

Status of the reference implementation
--------------------------------------
``VBayesLab/RealRECH`` ships only ``SMC_for_RECH`` — the *earlier* RECH model —
and its README still says the Python version is forthcoming. So there is no
DeepRGARCH source to validate against. The equations above are complete enough
to implement faithfully; what the paper does not pin down numerically is the SMC
configuration (tempering schedule, ESS threshold, particle count, Markov kernel).
Those settings are taken from the RECH SMC code by the same authors and recorded
as deviations.

Data
----
The paper uses the Oxford-Man Realized Library, which was discontinued in 2022.
Per ``docs/03_DECISIONS_AND_GAPS.md`` the substitute is a realized measure
rebuilt at the same 5-minute sampling with the same Hansen-Lunde rescaling
(:func:`garchai.targets.hansen_lunde_scale`), and any reproduction says so.

Reference
---------
Liu, C., Wang, C., Tran, M.-N., & Kohn, R. (2023). Deep Learning Enhanced
Realized GARCH. https://arxiv.org/abs/2302.08002
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..constraints.transforms import bounded, inverse_positive, positive
from ..core.base import VolatilityModel
from ..core.config import ModelConfig
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance
from ..core.results import FitResult
from ..distributions import Normal
from ..distributions.base import Distribution
from ..mean.models import MeanModel, get_mean_model

__all__ = ["RealizedGARCH", "DeepRGARCH", "RealRECH"]

_LIU = dict(
    paper_id="liu2023_realrech",
    title="Deep Learning Enhanced Realized GARCH",
    authors=("Chen Liu", "Chao Wang", "Minh-Ngoc Tran", "Robert Kohn"),
    year=2023,
    paper_url="https://arxiv.org/abs/2302.08002",
    official_repo="https://github.com/VBayesLab/RealRECH",
    code_status=CodeStatus.OFFICIAL,
)

_LOG_2PI = math.log(2.0 * math.pi)


class _RealizedBase(VolatilityModel):
    """Shared machinery for the Realized GARCH family.

    Both models carry a **joint** likelihood over returns and the realized
    measure, so ``fit`` needs two series rather than one. That is handled here
    by stashing the realized measure before delegating to the standard training
    loop.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config or ModelConfig(mean="zero", distribution="normal"))
        self._mean = get_mean_model(self.config.mean)
        self._rv: Tensor | None = None
        self._last_paths: dict | None = None

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    def distribution(self) -> Distribution:
        return Normal()

    def supports_analytic_forecast(self) -> bool:
        return False

    # -- parameters --------------------------------------------------------- #
    def _register_common(self, gamma: float, beta: float, xi: float, phi: float,
                         tau1: float, tau2: float, sigma_u: float) -> None:
        dt = torch.float64
        eps = self.config.numerical.eps
        self.raw_gamma = nn.Parameter(inverse_positive(gamma, eps).to(dt))
        self.raw_beta = nn.Parameter(torch.logit(torch.tensor(min(max(beta, 1e-4), 0.999), dtype=dt)))
        self.xi = nn.Parameter(torch.tensor(float(xi), dtype=dt))
        self.raw_phi = nn.Parameter(inverse_positive(phi, eps).to(dt))
        self.tau1 = nn.Parameter(torch.tensor(float(tau1), dtype=dt))
        self.tau2 = nn.Parameter(torch.tensor(float(tau2), dtype=dt))
        self.raw_sigma_u = nn.Parameter(inverse_positive(sigma_u, eps).to(dt))

    def measurement_parameters(self) -> dict[str, Tensor]:
        """Constrained parameters of the variance and measurement equations."""
        return {
            "gamma": positive(self.raw_gamma, self.config.numerical.eps),
            "beta": bounded(self.raw_beta, 0.0, 1.0),
            "xi": self.xi,
            "phi": positive(self.raw_phi, self.config.numerical.eps),
            "tau1": self.tau1,
            "tau2": self.tau2,
            "sigma_u": positive(self.raw_sigma_u, self.config.numerical.eps),
        }

    def leverage(self, z: Tensor) -> Tensor:
        r""":math:`\tau(\epsilon) = \tau_1\epsilon + \tau_2(\epsilon^2 - 1)`.

        Centred so that :math:`E[\tau(\epsilon)] = 0` under a standardised
        innovation, which is what keeps :math:`\xi` interpretable as the
        measurement intercept rather than absorbing a leverage mean.
        """
        return self.tau1 * z + self.tau2 * (z.square() - 1.0)

    # -- likelihood --------------------------------------------------------- #
    def joint_negative_loglikelihood(self, y: Tensor, rv: Tensor) -> Tensor:
        r"""Equation 8: the return and measurement equations, added.

        .. math::
            \ell = -\tfrac{1}{2}\sum_t\Big[\log 2\pi + \log\sigma_t^2
                   + y_t^2/\sigma_t^2\Big]
                   -\tfrac{1}{2}\sum_t\Big[\log 2\pi + \log\sigma_u^2
                   + u_t^2/\sigma_u^2\Big]

        with :math:`u_t = rv_t - \xi - \varphi\sigma_t^2 - \tau_1 y_t/\sigma_t
        - \tau_2(y_t^2/\sigma_t^2 - 1)`.

        Both halves matter. Dropping the measurement term would leave
        :math:`\xi, \varphi, \tau_1, \tau_2, \sigma_u` unidentified — they appear
        nowhere else — which is a quiet way to end up fitting a different model
        than the one documented.
        """
        h, _, _ = self._filter_with_rv(y, rv)
        p = self.measurement_parameters()

        z = y * torch.rsqrt(h)
        return_term = 0.5 * (_LOG_2PI + torch.log(h) + y.square() / h).sum()

        u = rv - p["xi"] - p["phi"] * h - self.leverage(z)
        variance_u = p["sigma_u"].square()
        measurement_term = 0.5 * (
            _LOG_2PI + torch.log(variance_u) + u.square() / variance_u
        ).sum()
        return return_term + measurement_term

    def negative_loglikelihood(self, r: Tensor) -> Tensor:
        if self._rv is None:
            raise RuntimeError(
                "A realized measure is required. Call fit(returns, realized_variance=...)."
            )
        return self.joint_negative_loglikelihood(r, self._rv)

    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self._rv is None:
            raise RuntimeError("A realized measure is required; call fit(...) first.")
        return self._filter_with_rv(r, self._rv)

    def fit(
        self, returns, train=None, callback=None, *, realized_variance=None
    ) -> FitResult:
        """Fit on ``(returns, realized_variance)`` jointly.

        Parameters
        ----------
        realized_variance
            The realized measure, aligned with ``returns``. Keyword-only, so
            that ``fit(returns, TrainConfig(...))`` cannot bind a training
            config to it by accident. Rescale it first
            with :func:`garchai.targets.hansen_lunde_scale`: trading-hours
            measures miss the overnight move and systematically understate
            close-to-close variance, which biases ``phi`` toward one.
        """
        from ..core.results import to_tensor

        if realized_variance is None:
            raise ValueError(
                "RealizedGARCH-family models need a realized measure: "
                "fit(returns, realized_variance=rv)."
            )
        rv = to_tensor(realized_variance, dtype=self.config.torch_dtype)
        r = to_tensor(returns, dtype=self.config.torch_dtype)
        if rv.numel() != r.numel():
            raise ValueError(
                f"returns and realized_variance must align: {r.numel()} vs {rv.numel()}."
            )
        if bool((rv <= 0).any()):
            raise ValueError("The realized measure must be strictly positive.")

        from ..core.config import resolve_scale

        scale = resolve_scale(r, self.config.scale)
        self._rv = rv * scale**2          # a variance scales with the square
        return super().fit(returns, train=train, callback=callback)

    def _variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
        p = {k: float(v.detach()) for k, v in self.measurement_parameters().items()}
        # In a Realized GARCH the persistence is beta + gamma*phi: a shock feeds
        # into rv through the measurement equation and back into the variance.
        persistence = p["beta"] + p["gamma"] * p["phi"]
        p["persistence (beta + gamma*phi)"] = persistence
        constant = self._constant_term()
        long_run = (
            (constant + p["gamma"] * p["xi"]) / (1.0 - persistence)
            if persistence < 1.0
            else float("inf")
        )
        return p, persistence, long_run

    def _constant_term(self) -> float:
        raise NotImplementedError


class RealizedGARCH(_RealizedBase):
    r"""Hansen, Huang & Shek (2012) Realized GARCH(1,1), the baseline.

    .. math::
        \sigma_t^2 = \omega + \gamma\, rv_{t-1} + \beta\sigma_{t-1}^2,
        \qquad
        rv_t = \xi + \varphi\sigma_t^2 + \tau(\epsilon_t) + u_t

    Present because DeepRGARCH's whole claim is measured against it: the
    recurrent component earns its keep only if it beats this. Note the
    persistence here is :math:`\beta + \gamma\varphi`, not
    :math:`\alpha + \beta` — a shock reaches tomorrow's variance *through* the
    measurement equation.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> model = RealizedGARCH()
    >>> y = torch.randn(50) * 0.9
    >>> rv = (y.square() + 0.1)
    >>> h, _, _ = model._filter_with_rv(y, rv)
    >>> bool((h > 0).all())
    True
    """

    provenance = Provenance(
        paper_id="hansen2012_realized_garch",
        title="Realized GARCH: a joint model for returns and realized measures of volatility",
        authors=("Peter R. Hansen", "Zhuo Huang", "Howard Howan Shek"),
        year=2012,
        venue="Journal of Applied Econometrics 27(6), 877-906",
        doi="10.1002/jae.1234",
        equations="Realized GARCH(1,1) with a linear measurement equation",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.PAPER_FAITHFUL,
        notes="Baseline for DeepRGARCH; also the reference in Liu et al. (2023).",
    )

    def __init__(
        self,
        config: ModelConfig | None = None,
        omega: float = 0.05,
        gamma: float = 0.35,
        beta: float = 0.55,
        xi: float = -0.2,
        phi: float = 1.0,
        tau1: float = -0.07,
        tau2: float = 0.07,
        sigma_u: float = 0.4,
    ):
        super().__init__(config)
        self.raw_omega = nn.Parameter(
            inverse_positive(omega, self.config.numerical.eps).to(torch.float64)
        )
        self._register_common(gamma, beta, xi, phi, tau1, tau2, sigma_u)

    @property
    def cell(self):
        return self

    def transformed_parameters(self) -> dict[str, Tensor]:
        return {
            "omega": positive(self.raw_omega, self.config.numerical.eps),
            **self.measurement_parameters(),
        }

    def _constant_term(self) -> float:
        return float(positive(self.raw_omega, self.config.numerical.eps).detach())

    def _filter_with_rv(self, y: Tensor, rv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(y)
        p = self.measurement_parameters()
        omega = positive(self.raw_omega, self.config.numerical.eps)
        floor = self.config.numerical.variance_floor

        h = eps.var(unbiased=True).clamp(min=floor)
        out = [h]
        for t in range(1, eps.numel()):
            h = (omega + p["gamma"] * rv[t - 1] + p["beta"] * h).clamp(min=floor)
            out.append(h)
        return torch.stack(out), eps, mu


class DeepRGARCH(_RealizedBase):
    r"""RealRECH: Realized GARCH with an LSTM-driven constant (Eqs. 7a-7f).

    The only structural change from :class:`RealizedGARCH` is that
    :math:`\omega` becomes :math:`\omega_t = \beta_0 + \beta_1 h_t` with
    :math:`h_t = \mathrm{LSTM}(x_t)` and
    :math:`x_t = (\omega_{t-1}, y_{t-1}, \sigma^2_{t-1}, rv_{t-1})`.

    Feeding the realized measure into the LSTM is the point: intraday measures
    are far more informative about the current variance level than squared daily
    returns, so the network has something worth learning from. As with RECH,
    :math:`\beta_1 = 0` recovers the baseline exactly, which makes the
    contribution of the recurrent part directly measurable rather than asserted.

    Parameters
    ----------
    hidden_size
        LSTM width. The paper's parameter count (26 LSTM parameters, of which 24
        inside the cell) implies a **single** hidden unit.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> from garchai.core import set_seed
    >>> set_seed(0)
    >>> model = DeepRGARCH(beta_1=0.0)
    >>> y = torch.randn(80) * 0.9
    >>> rv = y.square() + 0.1
    >>> deep, _, _ = model._filter_with_rv(y, rv)
    >>> baseline = RealizedGARCH(omega=float(model.beta_0.detach()))
    >>> import garchai
    >>> with torch.no_grad():
    ...     _ = baseline.raw_gamma.copy_(model.raw_gamma)
    ...     _ = baseline.raw_beta.copy_(model.raw_beta)
    >>> plain, _, _ = baseline._filter_with_rv(y, rv)
    >>> bool(torch.allclose(deep, plain, atol=1e-12))
    True
    """

    provenance = Provenance(
        **_LIU,
        equations="Eqs. 6a-6e (general), Eqs. 7a-7f (DeepRGARCH(1,1)), Eq. 8 (likelihood)",
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "No reference implementation exists to validate against: "
            "VBayesLab/RealRECH ships only SMC_for_RECH (the earlier model) and "
            "its README still promises the Python version.",
            "The paper specifies Bayesian inference by SMC but does not give the "
            "tempering schedule, ESS threshold, particle count or Markov kernel "
            "numerically. Defaults follow the RECH SMC code by the same authors. "
            "Maximum likelihood is offered as the fast alternative and is what "
            "fit() uses unless SMC is requested.",
        ),
        data_substitutions=(
            "The Oxford-Man Realized Library was discontinued in 2022. "
            "Reproductions use a realized measure rebuilt at the same 5-minute "
            "sampling with the same Hansen-Lunde rescaling, and say so.",
        ),
    )

    def __init__(
        self,
        config: ModelConfig | None = None,
        hidden_size: int = 1,
        beta_0: float = 0.05,
        beta_1: float = 0.05,
        gamma: float = 0.35,
        beta: float = 0.55,
        xi: float = -0.2,
        phi: float = 1.0,
        tau1: float = -0.07,
        tau2: float = 0.07,
        sigma_u: float = 0.4,
    ):
        super().__init__(config)
        dt = torch.float64
        self.hidden_size = hidden_size
        # x_t = (omega_{t-1}, y_{t-1}, sigma^2_{t-1}, rv_{t-1}) -- Eq. 7f.
        self.lstm = nn.LSTMCell(4, hidden_size, dtype=dt)
        self.raw_beta_0 = nn.Parameter(
            inverse_positive(beta_0, self.config.numerical.eps).to(dt)
        )
        self.raw_beta_1 = nn.Parameter(
            inverse_positive(max(beta_1, 1e-8), 0.0).to(dt)
            if beta_1 > 0
            else torch.tensor(-40.0, dtype=dt)
        )
        self._register_common(gamma, beta, xi, phi, tau1, tau2, sigma_u)

    @property
    def cell(self):
        return self.lstm

    @property
    def beta_0(self) -> Tensor:
        return positive(self.raw_beta_0, self.config.numerical.eps)

    @property
    def beta_1(self) -> Tensor:
        return positive(self.raw_beta_1, 0.0)

    def _constant_term(self) -> float:
        return float(self.beta_0.detach())

    def _filter_with_rv(self, y: Tensor, rv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(y)
        p = self.measurement_parameters()
        beta_0, beta_1 = self.beta_0, self.beta_1
        floor = self.config.numerical.variance_floor
        dtype = eps.dtype

        state = torch.zeros(1, self.hidden_size, dtype=dtype)
        cell = torch.zeros(1, self.hidden_size, dtype=dtype)
        h = eps.var(unbiased=True).clamp(min=floor)
        # g = ReLU, applied to omega_t to keep the variance positive (Eq. 6b).
        omega = torch.relu(beta_0 + beta_1 * state.mean())

        variances, omegas = [h], [omega]
        for t in range(1, eps.numel()):
            x = torch.stack(
                [omega.reshape(()), eps[t - 1], h.reshape(()), rv[t - 1]]
            ).reshape(1, -1)
            state, cell = self.lstm(x, (state, cell))
            omega = torch.relu(beta_0 + beta_1 * state.mean())
            h = (omega + p["gamma"] * rv[t - 1] + p["beta"] * h).clamp(min=floor)
            variances.append(h)
            omegas.append(omega)

        self._last_paths = {
            "omega": torch.stack(omegas).detach().cpu().numpy(),
        }
        return torch.stack(variances), eps, mu

    def _build_result(self, r: Tensor, history: dict, message: str) -> FitResult:
        result = super()._build_result(r, history, message)
        if self._last_paths is not None:
            result.parameter_paths = {
                "omega": self._last_paths["omega"] / self.data_scale_**2
            }
        return result


#: The paper's own name for :class:`DeepRGARCH`.
RealRECH = DeepRGARCH
