r"""Nguyen, Tran & Kohn (2022): Recurrent Conditional Heteroskedasticity.

RECH's idea is unusually clean. Take any GARCH-type recursion and let its
**constant term** be driven by a recurrent network:

.. math::
    y_t &= \sigma_t\epsilon_t \\
    \sigma_t^2 &= g(\omega_t) + f(\sigma^2_{t-1},\dots,y_{t-1},\dots)
        \qquad\text{(Eq. 7b)} \\
    \omega_t &= \beta_0 + \beta_1 h_t \qquad\text{(Eq. 7c)} \\
    h_t &= \mathrm{RNN}(x_t, h_{t-1}), \quad h_1 \equiv 0
        \qquad\text{(Eq. 7d)}

The additive split is what makes this work. :math:`f(\cdot)` is an ordinary
GARCH component, so persistence and leverage keep their usual meaning no matter
how elaborate the recurrent part becomes; and :math:`\beta_1 = 0` collapses the
whole model back to plain GARCH. Complexity is added *beside* the econometrics,
not *on top of* it.

The SRN-GARCH specification (Eqs. 8a-8d) is the paper's headline:

.. math::
    \sigma_t^2 = \omega_t + \alpha y_{t-1}^2 + \beta\sigma_{t-1}^2,
    \qquad
    h_t = \varphi\big(v^{\!\top}x_t + w h_{t-1} + b\big),
    \qquad
    x_t = (\omega_{t-1},\, y_{t-1},\, \sigma^2_{t-1})^{\!\top}

Two details in that input vector repay attention. Including :math:`y_{t-1}`
itself — not :math:`y_{t-1}^2` — is what lets a *symmetric* GARCH component
still capture leverage, because the network sees the sign. And including
:math:`\sigma^2_{t-1}` lets the recurrent part express **non-linear** dependence
on past variance, where :math:`\beta` only expresses linear dependence.

Paper versus official code
--------------------------
The authors say so themselves, in the repository README: *"in the JAE paper, we
implemented RNN-GARCH for the RECH specification. The LSTM-tGARCH specification
is more sophisticated and works better."* So the published equations and the
shipped software are two different models. Rather than pick one,
``implementation`` selects:

``"paper"``
    SRN (simple recurrent) component, Gaussian errors — Eqs. 8a-8d exactly.
``"official_code"``
    LSTM component, Student-t errors — what ``VBayesLab/RECH`` actually runs.

This is the situation blueprint sec. 64 anticipates, and exposing both is more
useful than adjudicating between them.

Reference
---------
Nguyen, T.-N., Tran, M.-N., & Kohn, R. (2022). Recurrent Conditional
Heteroskedasticity. *Journal of Applied Econometrics* 37(5), 1031-1054.
https://arxiv.org/abs/2010.13061 — official code: https://github.com/VBayesLab/RECH
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..constraints.transforms import positive, student_df
from ..core.base import VolatilityModel
from ..core.config import ModelConfig
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance
from ..distributions import Normal, StudentT
from ..distributions.base import Distribution
from ..mean.models import MeanModel, get_mean_model

__all__ = ["RECH", "SRNGARCH", "SRNGJR", "LSTMtGARCH", "bounded_relu"]

_RECH = dict(
    paper_id="nguyen2022_rech",
    title="Recurrent Conditional Heteroskedasticity",
    authors=("T.-N. Nguyen", "M.-N. Tran", "R. Kohn"),
    year=2022,
    venue="Journal of Applied Econometrics 37(5), 1031-1054",
    paper_url="https://arxiv.org/abs/2010.13061",
    official_repo="https://github.com/VBayesLab/RECH",
    code_status=CodeStatus.OFFICIAL,
)


def bounded_relu(x: Tensor, upper: float = 1.0) -> Tensor:
    r"""Bounded ReLU, :math:`\varphi(z) = \min(\max(0, z), A)`.

    The paper picks this activation for two distinct reasons, and only one of
    them is about training. Liew et al. (2016) recommend it for stability; but
    the paper's Theorem 1 additionally *requires* a bounded recurrent component
    — :math:`\omega_t \le M` almost surely — to guarantee that
    :math:`\operatorname{Var}(y_t)` is finite. Swapping in an unbounded ReLU
    would train, and would silently forfeit that guarantee.
    """
    return torch.clamp(x, min=0.0, max=upper)


class _RecurrentComponent(nn.Module):
    """The RNN driving :math:`\\omega_t`. SRN in the paper, LSTM in the code."""

    def __init__(
        self,
        kind: Literal["srn", "lstm"] = "srn",
        input_size: int = 3,
        hidden_size: int = 1,
        activation_bound: float = 1.0,
    ):
        super().__init__()
        self.kind = kind
        self.hidden_size = hidden_size
        self.activation_bound = activation_bound
        dt = torch.float64

        if kind == "srn":
            # Eq. 8d: h_t = phi(v' x_t + w h_{t-1} + b), a scalar state by default.
            self.v = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dt) * 0.1)
            self.w = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dt))
            self.b = nn.Parameter(torch.zeros(hidden_size, dtype=dt))
        elif kind == "lstm":
            self.cell = nn.LSTMCell(input_size, hidden_size, dtype=dt)
        else:
            raise ValueError(f"Unknown recurrent component {kind!r}; use 'srn' or 'lstm'.")

    def initial_state(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        """``h_1 = 0`` — the network starts memoryless, as Eq. 8d specifies."""
        zeros = torch.zeros(self.hidden_size, dtype=dtype)
        return zeros, (torch.zeros(self.hidden_size, dtype=dtype) if self.kind == "lstm" else None)

    def step(
        self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        if self.kind == "srn":
            pre = self.v @ x_t + self.w @ h_prev + self.b
            return bounded_relu(pre, self.activation_bound), None
        h, c = self.cell(x_t.reshape(1, -1), (h_prev.reshape(1, -1), c_prev.reshape(1, -1)))
        return h.reshape(-1), c.reshape(-1)


class RECH(VolatilityModel):
    r"""Recurrent Conditional Heteroskedasticity.

    Parameters
    ----------
    garch_component
        ``"garch"`` for Eq. 8b, ``"gjr"`` for the SRN-GJR specification of
        Table 1.
    implementation
        ``"paper"`` — SRN + Gaussian, exactly Eqs. 8a-8d.
        ``"official_code"`` — LSTM + Student-t, matching ``VBayesLab/RECH``.
        Anything explicitly passed for ``recurrent``/``distribution`` overrides
        the preset.
    hidden_size
        The paper uses a **scalar** recurrent state and notes that this suffices;
        larger states are permitted by Eq. 7d.
    activation_bound
        The :math:`A` in the bounded ReLU. Governs the bound on
        :math:`\omega_t` and hence Theorem 1.

    Attributes
    ----------
    After fitting, ``result_.parameter_paths`` carries ``omega`` and ``h``, so
    the recurrent component's contribution can be plotted against the constant
    :math:`\beta_0` it replaces. That decomposition is the model's main
    interpretive payoff, and :func:`garchai.reporting.plot_rech_decomposition`
    draws it.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> from garchai.core import set_seed
    >>> set_seed(0)
    >>> model = RECH(implementation="paper")
    >>> h, eps, _ = model.filter(torch.randn(100) * 0.9)
    >>> h.shape, bool((h > 0).all())
    (torch.Size([100]), True)

    With ``beta_1 = 0`` the recurrent component is switched off and RECH is
    exactly its GARCH component:

    >>> import torch
    >>> from garchai.cells import GARCHCell
    >>> model = RECH(implementation="paper", beta_1=0.0)
    >>> r = torch.randn(200) * 0.9
    >>> rech_path, eps, _ = model.filter(r)
    >>> cell = GARCHCell(0.05, 0.1, 0.85, init_method="sample")
    >>> garch_path, _ = cell.filter(eps)
    >>> bool(torch.allclose(rech_path, garch_path, atol=1e-12))
    True
    """

    provenance = Provenance(
        **_RECH,
        equations="Eqs. 7a-7d (general), Eqs. 8a-8d (SRN-GARCH), Theorem 1",
        implementation_status=ImplementationStatus.PAPER_FAITHFUL,
        deviations=(
            "The paper and the official code implement different models; the "
            "authors say so in the repository README. `implementation` selects "
            "between SRN+Gaussian (paper) and LSTM+Student-t (official code).",
            "The bounded-ReLU upper bound A is not stated numerically in the "
            "paper; it defaults to 1.0 and is exposed as activation_bound. It "
            "matters, because Theorem 1's finite-variance guarantee rests on it.",
        ),
    )

    def __init__(
        self,
        garch_component: Literal["garch", "gjr"] = "garch",
        implementation: Literal["paper", "official_code"] = "paper",
        config: ModelConfig | None = None,
        hidden_size: int = 1,
        recurrent: Literal["srn", "lstm"] | None = None,
        activation_bound: float = 1.0,
        omega: float = 0.05,
        alpha: float = 0.10,
        gamma: float = 0.05,
        beta: float = 0.85,
        beta_1: float = 0.05,
    ):
        if implementation == "official_code":
            recurrent = recurrent or "lstm"
            if config is None:
                config = ModelConfig(mean="zero", distribution="student-t")
        else:
            recurrent = recurrent or "srn"
            if config is None:
                config = ModelConfig(mean="zero", distribution="normal")
        super().__init__(config)

        self.implementation = implementation
        self.garch_component = garch_component
        self.use_student_t = self.config.distribution == "student-t"
        if self.config.distribution not in ("normal", "student-t"):
            raise ValueError("RECH supports 'normal' or 'student-t' innovations.")

        self._mean = get_mean_model(self.config.mean)
        self.hidden_size = hidden_size
        # x_t = (omega_{t-1}, y_{t-1}, sigma^2_{t-1}), per the text below Eq. 8d.
        self.network = _RecurrentComponent(recurrent, 3, hidden_size, activation_bound)

        dt = torch.float64
        from ..constraints.transforms import inverse_positive

        # beta_0, beta_1 >= 0 keep omega_t positive (paper, below Eq. 8d).
        self.raw_beta_0 = nn.Parameter(inverse_positive(omega, self.config.numerical.eps).to(dt))
        self.raw_beta_1 = nn.Parameter(
            inverse_positive(max(beta_1, 1e-8), 0.0).to(dt)
            if beta_1 > 0
            else torch.tensor(-40.0, dtype=dt)  # softplus(-40) is 0 to machine precision
        )
        # The GARCH component keeps the standard simplex parameterisation.
        from ..cells.garch_cell import _invert_garch_start, _invert_gjr_start
        from ..constraints.transforms import GARCHConstraint, GJRConstraint

        if garch_component == "garch":
            self._constraint = GARCHConstraint(
                slack=self.config.numerical.stationarity_slack, eps=self.config.numerical.eps
            )
            raw = _invert_garch_start(1.0, alpha, beta, self._constraint)
        elif garch_component == "gjr":
            self._constraint = GJRConstraint(
                slack=self.config.numerical.stationarity_slack, eps=self.config.numerical.eps
            )
            raw = _invert_gjr_start(1.0, alpha, gamma, beta, self._constraint)
            self.raw_gamma = nn.Parameter(raw["gamma"])
        else:
            raise ValueError(f"Unknown garch_component {garch_component!r}.")

        self.raw_alpha = nn.Parameter(raw["alpha"])
        self.raw_beta = nn.Parameter(raw["beta"])
        self.raw_slack = nn.Parameter(raw["slack"])
        if self.use_student_t:
            self.raw_df = nn.Parameter(torch.tensor(0.0, dtype=dt))

        self._last_paths: dict | None = None

    # -- interface ---------------------------------------------------------- #
    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    @property
    def cell(self):
        return self.network

    def distribution(self) -> Distribution:
        if self.use_student_t:
            return StudentT(df=student_df(self.raw_df, self.config.numerical.min_df))
        return Normal()

    def _distribution_parameters(self) -> dict[str, float]:
        if self.use_student_t:
            return {
                "df (nu)": float(
                    student_df(self.raw_df, self.config.numerical.min_df).detach()
                )
            }
        return {}

    def supports_analytic_forecast(self) -> bool:
        return False

    def garch_parameters(self) -> dict[str, Tensor]:
        """The GARCH component's parameters, with the usual constraints applied."""
        if self.garch_component == "garch":
            p = self._constraint(
                torch.zeros((), dtype=torch.float64), self.raw_alpha, self.raw_beta, self.raw_slack
            )
        else:
            p = self._constraint(
                torch.zeros((), dtype=torch.float64),
                self.raw_alpha,
                self.raw_gamma,
                self.raw_beta,
                self.raw_slack,
            )
        p.pop("omega", None)  # omega is supplied by the recurrent component
        return p

    @property
    def beta_0(self) -> Tensor:
        """The constant part of :math:`\\omega_t`."""
        return positive(self.raw_beta_0, self.config.numerical.eps)

    @property
    def beta_1(self) -> Tensor:
        r"""The recurrent loading. ``beta_1 = 0`` collapses RECH to plain GARCH."""
        return positive(self.raw_beta_1, 0.0)

    # -- the recursion ------------------------------------------------------ #
    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(r)
        n = eps.numel()
        dtype = eps.dtype
        p = self.garch_parameters()
        beta_0, beta_1 = self.beta_0, self.beta_1

        h, c = self.network.initial_state(dtype)
        sigma2 = eps.var(unbiased=True).clamp(min=self.config.numerical.variance_floor)
        omega = beta_0 + beta_1 * h.mean()

        variances = [sigma2]
        omegas = [omega]
        states = [h.mean()]

        for t in range(1, n):
            # x_t = (omega_{t-1}, y_{t-1}, sigma^2_{t-1}); including y (not y^2)
            # is what lets a symmetric GARCH component still see leverage.
            x = torch.stack([omega.reshape(()), eps[t - 1], sigma2.reshape(())])
            h, c = self.network.step(x, h, c)
            omega = beta_0 + beta_1 * h.mean()

            shock = eps[t - 1].square()
            asymmetric = (
                p["gamma"] * (eps[t - 1] < 0).to(dtype) * shock
                if self.garch_component == "gjr"
                else torch.zeros((), dtype=dtype)
            )
            sigma2 = (
                omega + p["alpha"] * shock + asymmetric + p["beta"] * sigma2
            ).clamp(min=self.config.numerical.variance_floor)

            variances.append(sigma2)
            omegas.append(omega)
            states.append(h.mean())

        self._last_paths = {
            "omega": torch.stack(omegas).detach().cpu().numpy(),
            "h": torch.stack(states).detach().cpu().numpy(),
        }
        return torch.stack(variances), eps, mu

    def _variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
        """Report the GARCH component plus the average recurrent constant."""
        p = {k: float(v.detach()) for k, v in self.garch_parameters().items()}
        p["beta_0"] = float(self.beta_0.detach())
        p["beta_1"] = float(self.beta_1.detach())
        persistence = float(self._constraint.persistence(self.garch_parameters()).detach())
        omega_bar = (
            float(self._last_paths["omega"].mean()) if self._last_paths else p["beta_0"]
        )
        p["mean omega_t"] = omega_bar
        long_run = omega_bar / (1.0 - persistence) if persistence < 1.0 else float("inf")
        return p, persistence, long_run

    def _build_result(self, r: Tensor, history: dict, message: str):
        result = super()._build_result(r, history, message)
        if self._last_paths is not None:
            scale = self.data_scale_
            result.parameter_paths = {
                "omega": self._last_paths["omega"] / scale**2,
                "h": self._last_paths["h"],
            }
        return result

    def decomposition(self) -> dict:
        r"""Split the fitted variance into its econometric and recurrent parts.

        Returns ``beta_0`` (the constant a plain GARCH would have), the fitted
        :math:`\omega_t` path, and their difference :math:`\beta_1 h_t` — the
        part the recurrent component is contributing. Plotting the difference
        against zero shows directly *when* the network is doing work, which is
        the question a referee will ask.
        """
        result = self._require_fit()
        paths = result.parameter_paths or {}
        beta_0 = float(self.beta_0.detach()) / self.data_scale_**2
        omega = paths.get("omega")
        return {
            "beta_0": beta_0,
            "omega_t": omega,
            "recurrent_contribution": None if omega is None else omega - beta_0,
            "hidden_state": paths.get("h"),
        }


class SRNGARCH(RECH):
    """The paper's headline specification: SRN recurrent component, GARCH(1,1)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("garch_component", "garch")
        kwargs.setdefault("implementation", "paper")
        super().__init__(**kwargs)


class SRNGJR(RECH):
    """SRN recurrent component with a GJR(1,1,1) GARCH component (Table 1)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("garch_component", "gjr")
        kwargs.setdefault("implementation", "paper")
        super().__init__(**kwargs)


class LSTMtGARCH(RECH):
    """What the official code runs: LSTM recurrent component, Student-t errors.

    The authors describe this as "more sophisticated and works better" than the
    RNN-GARCH of the published paper.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("garch_component", "garch")
        kwargs.setdefault("implementation", "official_code")
        super().__init__(**kwargs)
