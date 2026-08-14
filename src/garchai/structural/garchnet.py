r"""Buczyński & Chlebus (2023): GARCHNet, and Liu & So (2020): ANN-GARCH.

Two papers with the same core move: replace the GARCH *variance equation* with
a neural network, but keep the GARCH *likelihood* as the objective. The network
is free to be arbitrarily non-linear; what makes it a volatility model rather
than a regression is that it is fitted by (quasi-)maximum likelihood.

GARCHNet goes one step further than anything else in this collection. Not only
the variance but the **entire innovation distribution** is time-varying: the
last hidden layer feeds three separate heads,

.. math::
    \sigma_t^2 = \mathrm{softplus}(W_V l_t + b_V), \qquad
    \eta_t = \mathrm{softplus}(W_E l_t + b_E) + 2, \qquad
    \lambda_t = \tanh(W_S l_t + b_S)

so the degrees of freedom (Eq. 20) and the skewness (Eq. 21) move with the data
too. The ``+2`` is not cosmetic: it enforces :math:`\eta_t > 2`, without which
the variance of a standardised Student-t does not exist. The ``tanh`` likewise
enforces :math:`-1 < \lambda_t < 1`, Hansen's admissibility condition.

That is a genuinely different object from a GARCH-t. A GARCH-t says "the tails
are heavy, by this fixed amount". GARCHNet says "the tails are *this* heavy
today and *that* heavy tomorrow" — which is why the paper is framed around VaR,
where the tail shape is the entire question.

The authors are explicit about why they drop the usual non-negativity
constraints on :math:`\omega, \beta, \gamma`: for a neural network such
restrictions "could lead to the worsening of the accuracy of estimated
solution" (sec. 2.3). Positivity is instead structural, via the softplus.

Liu & So (2020) is the earlier, simpler version of the same idea: an LSTM over a
20-day window, a softplus-squared readout, and a standardised Student-t
likelihood with **fixed** degrees of freedom.

References
----------
Buczyński, M., & Chlebus, M. (2023). GARCHNet: Value-at-Risk Forecasting with
GARCH Models Based on Neural Networks. *Computational Economics*.
https://doi.org/10.1007/s10614-023-10390-7

Liu, W. K., & So, M. K. P. (2020). A GARCH Model with Artificial Neural
Networks. *Information* 11(10), 489. https://doi.org/10.3390/info11100489
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.base import VolatilityModel
from ..core.config import ModelConfig
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance
from ..distributions import Normal, SkewT, StudentT
from ..distributions.base import Distribution
from ..mean.models import MeanModel, get_mean_model

__all__ = ["GARCHNet", "ANNGARCH", "LikelihoodGARCHLSTM"]


class GARCHNet(VolatilityModel):
    r"""GARCHNet (Buczyński & Chlebus, 2023).

    An LSTM over a rolling window of past returns produces the conditional
    variance and, under the ``t`` and ``skew-t`` specifications, the
    distribution's shape parameters as well. Everything is fitted by QMLE using
    the paper's Eqs. 6-8.

    Parameters
    ----------
    window
        Lookback fed to the LSTM at each step.
    distribution
        ``"normal"`` (Eq. 6), ``"student-t"`` (Eq. 7) or ``"skew-t"`` (Eq. 8).
        Under the latter two the shape parameters are time-varying network
        outputs, per Eqs. 20-21.
    time_varying_shape
        Set ``False`` to estimate a single :math:`\eta` and :math:`\lambda` for
        the whole sample, which is the conventional GARCH-t/skew-t treatment and
        a useful ablation against the paper's contribution.

    Attributes
    ----------
    After fitting, ``result_.parameter_paths`` carries ``eta`` and ``lambda``
    where applicable — the time-varying tail shape, which is the quantity the
    paper's VaR results turn on.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> from garchai.core import ModelConfig, set_seed
    >>> set_seed(0)
    >>> model = GARCHNet(window=10, hidden_size=8,
    ...                  config=ModelConfig(mean="zero", distribution="student-t"))
    >>> h, eps, _ = model.filter(torch.randn(80) * 0.9)
    >>> h.shape, bool((h > 0).all())
    (torch.Size([80]), True)
    >>> bool((model._last_paths["eta"] > 2.0).all())     # Eq. 20's "+2"
    True
    """

    provenance = Provenance(
        paper_id="buczynski2023_garchnet",
        title="GARCHNet: Value-at-Risk Forecasting with GARCH Models Based on Neural Networks",
        authors=("Mateusz Buczynski", "Marcin Chlebus"),
        year=2023,
        venue="Computational Economics",
        doi="10.1007/s10614-023-10390-7",
        equations="Eqs. 6-8 (QMLE), Eqs. 20-21 (time-varying eta, lambda), Eq. 22 (softplus)",
        code_status=CodeStatus.COMMUNITY_REPLICATION,
        community_repos=("https://github.com/e-rrrr-or-404/garchnet-replication",),
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "Official code is 'available from the corresponding author'; the "
            "community replication was consulted for hyper-parameters only.",
            "The paper does not state the LSTM depth or width numerically; both "
            "are exposed as arguments and default to a single layer of 32.",
        ),
        notes=(
            "Unusually, the innovation distribution's shape is time-varying, not "
            "just its scale. Eq. 20's '+2' is what keeps the Student-t variance "
            "finite and Eq. 21's tanh is Hansen's admissibility condition."
        ),
    )

    def __init__(
        self,
        window: int = 20,
        hidden_size: int = 32,
        layers: int = 1,
        config: ModelConfig | None = None,
        time_varying_shape: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(config or ModelConfig(mean="zero", distribution="student-t"))
        name = self.config.distribution
        if name not in ("normal", "student-t", "skew-t"):
            raise ValueError("GARCHNet supports 'normal', 'student-t' or 'skew-t'.")

        self.window = window
        self.hidden_size = hidden_size
        self.time_varying_shape = time_varying_shape
        self._mean = get_mean_model(self.config.mean)
        dt = torch.float64

        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0, dtype=dt,
        )
        # Eq. 22: a separate head per output, softplus for the variance.
        self.head_variance = nn.Linear(hidden_size, 1, dtype=dt)
        if name in ("student-t", "skew-t"):
            if time_varying_shape:
                self.head_eta = nn.Linear(hidden_size, 1, dtype=dt)   # Eq. 20
            else:
                self.raw_eta = nn.Parameter(torch.tensor(2.0, dtype=dt))
        if name == "skew-t":
            if time_varying_shape:
                self.head_lambda = nn.Linear(hidden_size, 1, dtype=dt)  # Eq. 21
            else:
                self.raw_lambda = nn.Parameter(torch.zeros((), dtype=dt))
        self._last_paths: dict | None = None
        self._shape: dict[str, Tensor] = {}

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    @property
    def cell(self):
        return self.lstm

    def distribution(self) -> Distribution:
        """A representative distribution; the real one is per-period."""
        name = self.config.distribution
        if name == "normal":
            return Normal()
        eta = self._shape.get("eta")
        value = float(eta.mean().detach()) if eta is not None else 8.0
        if name == "student-t":
            return StudentT(df=max(value, 2.05))
        lam = self._shape.get("lambda")
        skew = float(lam.mean().detach()) if lam is not None else 0.0
        return SkewT(df=max(value, 2.05), lam=min(max(skew, -0.99), 0.99))

    def supports_analytic_forecast(self) -> bool:
        return False

    # -- the network -------------------------------------------------------- #
    def _windows(self, eps: Tensor) -> Tensor:
        """Rolling windows, left-padded so the output aligns with ``eps``."""
        padded = torch.cat([torch.zeros(self.window, dtype=eps.dtype), eps])
        return padded.unfold(0, self.window, 1)[:-1]

    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(r)
        windows = self._windows(eps)
        out, _ = self.lstm(windows.unsqueeze(-1))
        last = out[:, -1, :]

        variance = F.softplus(self.head_variance(last).squeeze(-1)).clamp(
            min=self.config.numerical.variance_floor
        )

        name = self.config.distribution
        shape: dict[str, Tensor] = {}
        if name in ("student-t", "skew-t"):
            if self.time_varying_shape:
                # Eq. 20: eta_t = softplus(...) + 2, so eta_t > 2 always.
                shape["eta"] = F.softplus(self.head_eta(last).squeeze(-1)) + 2.0
            else:
                shape["eta"] = (F.softplus(self.raw_eta) + 2.0).expand_as(variance)
        if name == "skew-t":
            if self.time_varying_shape:
                shape["lambda"] = torch.tanh(self.head_lambda(last).squeeze(-1))  # Eq. 21
            else:
                shape["lambda"] = torch.tanh(self.raw_lambda).expand_as(variance)

        self._shape = shape
        self._last_paths = {k: v.detach().cpu().numpy() for k, v in shape.items()}
        return variance, eps, mu

    def negative_loglikelihood(self, r: Tensor) -> Tensor:
        """QMLE under the chosen distribution (Eqs. 6, 7 or 8).

        With time-varying shape the density is evaluated period by period,
        because :math:`\\eta_t` and :math:`\\lambda_t` change every step. That is
        slower than a vectorised call and it is the price of the paper's actual
        contribution.
        """
        variance, eps, _ = self.filter(r)
        name = self.config.distribution
        if name == "normal":
            return Normal().nll(eps, variance, reduction="sum")

        z = eps * torch.rsqrt(variance)
        jacobian = 0.5 * torch.log(variance)
        eta = self._shape["eta"]

        if name == "student-t":
            log_prob = torch.stack(
                [StudentT(df=eta[i]).log_prob(z[i]) for i in range(z.numel())]
            )
        else:
            lam = self._shape["lambda"]
            log_prob = torch.stack(
                [
                    SkewT(df=eta[i], lam=lam[i].clamp(-0.999, 0.999)).log_prob(z[i])
                    for i in range(z.numel())
                ]
            )
        return -(log_prob - jacobian).sum()

    def _variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
        params: dict[str, float] = {}
        if self._last_paths:
            for key, values in self._last_paths.items():
                params[f"mean {key}"] = float(values.mean())
                params[f"min {key}"] = float(values.min())
                params[f"max {key}"] = float(values.max())
        result = getattr(self, "result_", None)
        long_run = (
            float(result.conditional_variance.mean()) if result is not None else float("nan")
        )
        return params, float("nan"), long_run

    def _build_result(self, r: Tensor, history: dict, message: str):
        result = super()._build_result(r, history, message)
        if self._last_paths:
            result.parameter_paths = dict(self._last_paths)
        return result

    def var_path(self, alpha: float = 0.01) -> Tensor:
        r"""Value-at-Risk using the **period-specific** distribution.

        This is the point of the model. A GARCH-t applies one quantile
        :math:`F^{-1}(\alpha)` scaled by :math:`\sigma_t`; GARCHNet applies
        :math:`F_t^{-1}(\alpha)`, because the tail shape itself moves. Using the
        average :math:`\eta` instead would discard exactly the feature the paper
        exists to demonstrate.
        """
        result = self._require_fit()
        sigma = torch.tensor(result.conditional_volatility)
        mu = torch.tensor(result.conditional_mean)
        name = self.config.distribution
        if name == "normal" or not self._last_paths:
            return mu + sigma * float(self.distribution().ppf(alpha))

        eta = torch.tensor(self._last_paths["eta"])
        if name == "student-t":
            quantiles = torch.stack(
                [StudentT(df=eta[i]).ppf(alpha) for i in range(eta.numel())]
            )
        else:
            lam = torch.tensor(self._last_paths["lambda"])
            quantiles = torch.stack(
                [
                    SkewT(df=eta[i], lam=lam[i].clamp(-0.999, 0.999)).ppf(alpha)
                    for i in range(eta.numel())
                ]
            )
        return mu + sigma * quantiles


class ANNGARCH(VolatilityModel):
    r"""Liu & So (2020): a GARCH model with an LSTM variance equation.

    .. math::
        r_t = \epsilon_t\sqrt{h_t},\quad \epsilon_t\sim T(0,1,\nu),\qquad
        h_t = \big(\log(\exp(D(y_{c_1}(t)))+1)\big)^2

    Equation 7's readout is worth reading twice: the network output passes
    through a softplus and is then **squared**. Squaring a softplus is not the
    obvious way to make a variance positive, and it has a real consequence —
    the map is much steeper, so a given change in the network output moves the
    variance further. The authors adopt softplus over ReLU explicitly because
    ReLU "may introduce a zero gradient in some regions" (sec. 2.1), the same
    failure the σ-Cell family hits.

    The objective is the standardised Student-t negative log-likelihood with
    ``nu`` estimated, and the lookback is 20 days.

    Notes
    -----
    The paper's covariates are 17 Bloomberg Terminal fundamentals, which are
    licensed and cannot be redistributed. Per ``docs/03_DECISIONS_AND_GAPS.md``
    the substitute is a public analogue feature set; ``n_exog`` accepts any
    exogenous panel, and the substitution is declared on the provenance.

    The paper's variable-importance measure — the drop in log-likelihood when a
    covariate is removed — is available as :meth:`variable_importance`.
    """

    provenance = Provenance(
        paper_id="liu2020_ann_garch",
        title="A GARCH Model with Artificial Neural Networks",
        authors=("Wing Ki Liu", "Mike K. P. So"),
        year=2020,
        venue="Information 11(10), 489",
        doi="10.3390/info11100489",
        equations="Eqs. 1-2 (GARCH), Eqs. 3-6 (LSTM cell), Eq. 7 (readout), sec. 2.2 (NLL)",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.PAPER_FAITHFUL,
        deviations=(
            "The paper uses TensorFlow defaults for the LSTM gates and an "
            "identity activation in the dense head, chosen over ReLU to avoid "
            "zero-gradient regions. Both are reproduced.",
            "The dense head's depth and width follow Feng et al. and are not "
            "given numerically; exposed as arguments.",
        ),
        data_substitutions=(
            "The paper's 17 covariates come from a Bloomberg Terminal and are "
            "licensed. Any exogenous panel may be supplied via n_exog; the "
            "public analogue used in the examples is declared where it appears.",
        ),
    )

    def __init__(
        self,
        window: int = 20,
        hidden_size: int = 32,
        head_width: int = 16,
        n_exog: int = 0,
        config: ModelConfig | None = None,
    ):
        super().__init__(config or ModelConfig(mean="zero", distribution="student-t"))
        # The paper specifies a standardised Student-t (sec. 2.2), which is the
        # default here. A Gaussian is still accepted: that is ordinary quasi-MLE,
        # and rejecting it would also reject the common `ModelConfig(mean="zero")`
        # whose own default distribution is normal.
        if self.config.distribution not in ("student-t", "normal"):
            raise ValueError(
                "ANN-GARCH supports 'student-t' (the paper's specification) or "
                f"'normal' (Gaussian QMLE); got {self.config.distribution!r}."
            )
        self.use_student_t = self.config.distribution == "student-t"
        self.window = window
        self.n_exog = n_exog
        self._mean = get_mean_model(self.config.mean)
        dt = torch.float64

        self.lstm = nn.LSTM(
            input_size=1 + n_exog, hidden_size=hidden_size, batch_first=True, dtype=dt
        )
        # "the identity function works better for this network because RELU may
        # introduce a zero gradient in some regions" (sec. 2.1).
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_width, dtype=dt),
            nn.Identity(),
            nn.Linear(head_width, 1, dtype=dt),
        )
        self.raw_df = nn.Parameter(torch.tensor(0.0, dtype=dt))
        self._exog: Tensor | None = None

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    @property
    def cell(self):
        return self.lstm

    def distribution(self) -> Distribution:
        from ..constraints.transforms import student_df

        if not self.use_student_t:
            return Normal()
        return StudentT(df=student_df(self.raw_df, self.config.numerical.min_df))

    def _distribution_parameters(self) -> dict[str, float]:
        from ..constraints.transforms import student_df

        if not self.use_student_t:
            return {}
        return {
            "df (nu)": float(
                student_df(self.raw_df, self.config.numerical.min_df).detach()
            )
        }

    def supports_analytic_forecast(self) -> bool:
        return False

    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(r)
        n = eps.numel()
        padded = torch.cat([torch.zeros(self.window, dtype=eps.dtype), eps])
        windows = padded.unfold(0, self.window, 1)[:-1].unsqueeze(-1)

        if self.n_exog:
            if self._exog is None:
                raise RuntimeError("This model expects exogenous inputs; pass exog= to fit().")
            padded_x = torch.cat(
                [torch.zeros(self.window, self.n_exog, dtype=eps.dtype), self._exog]
            )
            extra = torch.stack(
                [padded_x[i : i + self.window] for i in range(n)]
            )
            windows = torch.cat([windows, extra], dim=-1)

        out, _ = self.lstm(windows)
        # Eq. 7: softplus, then squared.
        variance = F.softplus(self.head(out[:, -1, :]).squeeze(-1)).square().clamp(
            min=self.config.numerical.variance_floor
        )
        return variance, eps, mu

    def fit(self, returns, train=None, callback=None, *, exog=None):
        """Fit, optionally with an exogenous panel aligned to ``returns``.

        ``exog`` is keyword-only on purpose. With it in the second positional
        slot, the natural call ``fit(returns, TrainConfig(...))`` silently
        binds the training config to ``exog`` and fails deep inside a tensor
        conversion.
        """
        if exog is not None:
            from ..core.results import to_numpy

            values = torch.as_tensor(
                to_numpy(exog) if not isinstance(exog, torch.Tensor) else exog,
                dtype=self.config.torch_dtype,
            )
            if values.ndim != 2 or values.shape[1] != self.n_exog:
                raise ValueError(
                    f"exog must have shape (T, {self.n_exog}), got {tuple(values.shape)}."
                )
            self._exog = values
        return super().fit(returns, train=train, callback=callback)

    def _variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
        result = getattr(self, "result_", None)
        long_run = (
            float(result.conditional_variance.mean()) if result is not None else float("nan")
        )
        return {"window": float(self.window)}, float("nan"), long_run

    def variable_importance(self, returns, exog) -> dict[int, float]:
        r"""The paper's importance measure: :math:`\Delta` log-likelihood on removal.

        Each covariate is zeroed in turn and the model re-evaluated. A large
        drop means the variable was carrying real information. Note this is an
        *ablation on a fitted model*, not a refit, which is what the paper does
        and is far cheaper — but it measures sensitivity of the fitted network
        rather than the value of the variable to a model that could adapt
        without it.
        """
        if not self.n_exog:
            return {}
        self._require_fit()
        baseline_exog = self._exog
        r = torch.as_tensor(returns, dtype=self.config.torch_dtype) * self.data_scale_

        with torch.no_grad():
            base = -float(self.negative_loglikelihood(r))
            out: dict[int, float] = {}
            for j in range(self.n_exog):
                ablated = baseline_exog.clone()
                ablated[:, j] = 0.0
                self._exog = ablated
                out[j] = base - (-float(self.negative_loglikelihood(r)))
        self._exog = baseline_exog
        return out


#: The paper's own framing of :class:`ANNGARCH`.
LikelihoodGARCHLSTM = ANNGARCH
