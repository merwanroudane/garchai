"""Classical GARCH-family models, estimated by maximum likelihood in PyTorch.

These exist for three reasons, in order of importance:

1. **Baselines.** Every hybrid claim in the collection is a claim relative to
   GARCH(1,1). Hansen & Lunde (2005) is the standing reminder that beating it is
   harder than it looks.
2. **Differentiability.** ``arch`` estimates these faster and is the reference
   we validate against — but its variance path is not an autograd graph, so it
   cannot be a component of a neural model. These can.
3. **Initialisation.** A structural model that starts from a converged GARCH
   solution trains far more reliably than one started at random.

For pure classical estimation with no neural component, prefer
:mod:`garchai.adapters.arch_adapter`; it is faster and better tested. Use these
when the variance path needs to carry gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..cells.base import InitMethod, VarianceCell
from ..cells.garch_cell import EGARCHCell, GARCHCell, GJRCell
from ..constraints.transforms import student_df
from ..core.base import VolatilityModel
from ..core.config import ModelConfig
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance
from ..distributions import GED, Normal, SkewT, StudentT
from ..distributions.base import Distribution
from ..mean.models import ConstantMean, MeanModel, get_mean_model

__all__ = ["GARCH", "GJRGARCH", "EGARCH"]


class _ClassicalModel(VolatilityModel):
    """Shared plumbing for the classical families."""

    def __init__(
        self,
        cell: VarianceCell,
        config: ModelConfig | None = None,
        mean_kwargs: dict | None = None,
    ):
        super().__init__(config)
        self._cell = cell
        self._mean = self._build_mean(mean_kwargs or {})
        self._build_distribution_parameters()

    def _build_mean(self, kwargs: dict) -> MeanModel:
        spec = self.config.mean
        if spec == "ar":
            kwargs.setdefault("p", max(self.config.mean_order[0], 1))
        elif spec == "arma":
            kwargs.setdefault("p", self.config.mean_order[0])
            kwargs.setdefault("q", self.config.mean_order[1])
        return get_mean_model(spec, **kwargs)

    def _build_distribution_parameters(self) -> None:
        """Register shape parameters so they are estimated with everything else."""
        name = self.config.distribution
        if name in ("student-t", "skew-t"):
            # raw scale; student_df() maps it into (min_df, 200)
            self.raw_df = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        if name == "skew-t":
            self.raw_skew = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        if name == "ged":
            self.raw_ged = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    @property
    def cell(self) -> VarianceCell:
        return self._cell

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    def distribution(self) -> Distribution:
        name = self.config.distribution
        if name == "normal":
            return Normal()
        if name == "student-t":
            return StudentT(df=student_df(self.raw_df, self.config.numerical.min_df))
        if name == "skew-t":
            return SkewT(
                df=student_df(self.raw_df, self.config.numerical.min_df),
                lam=0.999 * torch.tanh(self.raw_skew),
            )
        if name == "ged":
            from ..constraints.transforms import bounded

            return GED(nu=bounded(self.raw_ged, 0.2, 6.0))
        raise ValueError(f"Unsupported distribution {name!r}.")

    def _distribution_parameters(self) -> dict[str, float]:
        name = self.config.distribution
        out: dict[str, float] = {}
        if name in ("student-t", "skew-t"):
            out["df (nu)"] = float(student_df(self.raw_df, self.config.numerical.min_df).detach())
        if name == "skew-t":
            out["skew (lambda)"] = float((0.999 * torch.tanh(self.raw_skew)).detach())
        if name == "ged":
            from ..constraints.transforms import bounded

            out["shape (nu)"] = float(bounded(self.raw_ged, 0.2, 6.0).detach())
        return out

    def _forecast_mean(self, horizon: int):
        import numpy as np

        if isinstance(self._mean, ConstantMean):
            return np.full(horizon, float(self._mean.mu.detach()))
        if hasattr(self._mean, "unconditional_mean"):
            return np.full(horizon, float(self._mean.unconditional_mean().detach()))
        return np.zeros(horizon)


class GARCH(_ClassicalModel):
    r"""GARCH(1,1) — Bollerslev (1986).

    .. math::
        r_t = \mu_t + \epsilon_t, \qquad \epsilon_t = \sqrt{h_t}\, z_t, \qquad
        h_t = \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}

    Examples
    --------
    >>> import numpy as np, torch
    >>> torch.set_default_dtype(torch.float64)
    >>> from garchai.core import ModelConfig, set_seed
    >>> set_seed(0)
    >>> rng = np.random.default_rng(0)
    >>> r = rng.standard_normal(800) * 1.5          # doctest: +SKIP
    >>> model = GARCH(config=ModelConfig(mean="constant"))   # doctest: +SKIP
    >>> res = model.fit(r)                          # doctest: +SKIP
    >>> res.persistence < 1.0                       # doctest: +SKIP
    True
    """

    provenance = Provenance(
        paper_id="bollerslev1986",
        title="Generalized autoregressive conditional heteroskedasticity",
        authors=("Tim Bollerslev",),
        year=1986,
        venue="Journal of Econometrics 31(3), 307-327",
        doi="10.1016/0304-4076(86)90063-1",
        equations="GARCH(1,1) conditional variance",
        code_status=CodeStatus.PACKAGE_SOURCE,
        official_repo="https://github.com/bashtage/arch",
        implementation_status=ImplementationStatus.PAPER_FAITHFUL,
        notes=(
            "Validated against `arch` to 1e-12 on the full conditional-variance "
            "path; see tests/numerical/test_arch_parity.py."
        ),
    )

    def __init__(
        self,
        omega: float = 0.05,
        alpha: float = 0.10,
        beta: float = 0.85,
        config: ModelConfig | None = None,
        init_method: InitMethod = "arch_backcast",
        mean_kwargs: dict | None = None,
    ):
        cfg = config or ModelConfig()
        cell = GARCHCell(
            omega, alpha, beta,
            enforce_stationarity=cfg.enforce_stationarity,
            init_method=init_method,
            numerical=cfg.numerical,
        )
        super().__init__(cell, cfg, mean_kwargs)


class GJRGARCH(_ClassicalModel):
    r"""GJR-GARCH(1,1,1) — Glosten, Jagannathan & Runkle (1993).

    .. math::
        h_t = \omega + \alpha \epsilon_{t-1}^2
              + \gamma\, I(\epsilon_{t-1}<0)\,\epsilon_{t-1}^2 + \beta h_{t-1}

    ``gamma`` measures the leverage effect. Zhao et al. (2024) select GJR as the
    GARCH kernel inside their best GARCH-LSTM.
    """

    provenance = Provenance(
        paper_id="glosten1993",
        title=(
            "On the Relation between the Expected Value and the Volatility of "
            "the Nominal Excess Return on Stocks"
        ),
        authors=("Lawrence R. Glosten", "Ravi Jagannathan", "David E. Runkle"),
        year=1993,
        venue="Journal of Finance 48(5), 1779-1801",
        doi="10.1111/j.1540-6261.1993.tb05128.x",
        equations="GJR-GARCH(1,1,1) conditional variance",
        code_status=CodeStatus.PACKAGE_SOURCE,
        implementation_status=ImplementationStatus.PAPER_FAITHFUL,
        deviations=(
            "The stationary parameterisation enforces gamma >= 0. Pass "
            "enforce_stationarity=False to admit negative leverage.",
        ),
    )

    def __init__(
        self,
        omega: float = 0.05,
        alpha: float = 0.05,
        gamma: float = 0.08,
        beta: float = 0.85,
        config: ModelConfig | None = None,
        init_method: InitMethod = "arch_backcast",
        mean_kwargs: dict | None = None,
    ):
        cfg = config or ModelConfig()
        cell = GJRCell(
            omega, alpha, gamma, beta,
            enforce_stationarity=cfg.enforce_stationarity,
            init_method=init_method,
            numerical=cfg.numerical,
        )
        super().__init__(cell, cfg, mean_kwargs)


class EGARCH(_ClassicalModel):
    r"""EGARCH(1,1,1) — Nelson (1991).

    .. math::
        \log h_t = \omega + \beta \log h_{t-1}
                   + \alpha(|z_{t-1}| - E|z|) + \gamma z_{t-1}

    Has **no closed-form multi-step variance forecast**: iterating the recursion
    requires :math:`E[\exp(\cdot)]`, not :math:`\exp(E[\cdot])`, and the two
    differ by a Jensen gap that grows with the horizon. ``forecast()`` therefore
    simulates, and refuses ``method="analytic"`` rather than returning a
    plausible-looking wrong number.
    """

    provenance = Provenance(
        paper_id="nelson1991",
        title="Conditional Heteroskedasticity in Asset Returns: A New Approach",
        authors=("Daniel B. Nelson",),
        year=1991,
        venue="Econometrica 59(2), 347-370",
        doi="10.2307/2938260",
        equations="EGARCH(1,1,1) log-variance recursion",
        code_status=CodeStatus.PACKAGE_SOURCE,
        implementation_status=ImplementationStatus.PAPER_FAITHFUL,
        deviations=(
            "E|z| defaults to the normal value sqrt(2/pi); pass expected_abs_z "
            "explicitly when using a non-normal innovation distribution.",
        ),
    )

    def __init__(
        self,
        omega: float = -0.05,
        alpha: float = 0.15,
        gamma: float = -0.08,
        beta: float = 0.97,
        config: ModelConfig | None = None,
        init_method: InitMethod = "arch_backcast",
        expected_abs_z: float | None = None,
        mean_kwargs: dict | None = None,
    ):
        cfg = config or ModelConfig()
        kwargs = {} if expected_abs_z is None else {"expected_abs_z": expected_abs_z}
        cell = EGARCHCell(
            omega, alpha, gamma, beta,
            enforce_stationarity=cfg.enforce_stationarity,
            init_method=init_method,
            numerical=cfg.numerical,
            **kwargs,
        )
        super().__init__(cell, cfg, mean_kwargs)

    def supports_analytic_forecast(self) -> bool:
        return False
