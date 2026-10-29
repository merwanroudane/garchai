"""Result containers.

A fitted model returns data, not just printed output. Everything a user might
want to plot, tabulate or feed into a backtest is an attribute; ``summary()``
is a rendering of that data, never the only place it exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class FitResult:
    """Everything produced by fitting a volatility model.

    All series are on the **original data scale**, with any internal rescaling
    already undone. ``scale`` records what that rescaling was so the choice
    stays visible.

    Attributes
    ----------
    conditional_variance
        :math:`h_t`, aligned with the input returns.
    conditional_volatility
        :math:`\\sigma_t = \\sqrt{h_t}`.
    standardized_residuals
        :math:`z_t = \\epsilon_t / \\sigma_t`. Should look i.i.d. under the
        assumed distribution; :mod:`garchai.diagnostics` checks that.
    loglikelihood
        Maximised log-likelihood, evaluated over the estimation sample only
        (pre-sample observations required by the mean model are excluded).
    """

    conditional_variance: np.ndarray
    conditional_volatility: np.ndarray
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    conditional_mean: np.ndarray
    loglikelihood: float
    n_obs: int
    n_params: int
    n_presample: int
    parameters: dict[str, float]
    mean_parameters: dict[str, float] = field(default_factory=dict)
    distribution_parameters: dict[str, float] = field(default_factory=dict)
    persistence: float | None = None
    long_run_variance: float | None = None
    scale: float = 1.0
    history: dict[str, list[float]] = field(default_factory=dict)
    converged: bool = True
    message: str = ""
    parameter_paths: dict[str, np.ndarray] | None = None
    """Time-varying parameter paths, for models that have them. ``None`` otherwise."""

    @property
    def aic(self) -> float:
        """Akaike information criterion, ``2k - 2 log L``."""
        return 2.0 * self.n_params - 2.0 * self.loglikelihood

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``k log n - 2 log L``."""
        return self.n_params * np.log(self.n_obs) - 2.0 * self.loglikelihood

    def to_frame(self):
        """Return the fitted series as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            {
                "conditional_mean": self.conditional_mean,
                "residual": self.residuals,
                "conditional_variance": self.conditional_variance,
                "conditional_volatility": self.conditional_volatility,
                "standardized_residual": self.standardized_residuals,
            }
        )


@dataclass
class ForecastResult:
    """Multi-step-ahead variance forecasts.

    Attributes
    ----------
    variance
        Forecast :math:`h_{T+1}, \\dots, h_{T+H}` on the original data scale.
    method
        ``"analytic"``, ``"recursive"`` or ``"simulation"``. Recorded because
        blueprint sec. 27 forbids silently substituting one for another: an
        analytic multi-step GARCH forecast and a simulated one answer different
        questions once the model is nonlinear.
    simulations
        The raw simulated paths when ``method == "simulation"``, so that
        forecast intervals can be computed rather than only point forecasts.
    """

    variance: np.ndarray
    volatility: np.ndarray
    mean: np.ndarray
    horizon: int
    method: str
    scale: float = 1.0
    simulations: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def var(self, alpha: float, distribution) -> np.ndarray:
        """Value-at-Risk at each horizon, as a return level."""
        q = float(distribution.ppf(alpha))
        return self.mean + self.volatility * q

    def es(self, alpha: float, distribution) -> np.ndarray:
        """Expected shortfall at each horizon, as a return level."""
        e = float(distribution.expected_shortfall(alpha))
        return self.mean + self.volatility * e

    def to_frame(self):
        """Return the forecasts as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            {
                "horizon": np.arange(1, self.horizon + 1),
                "mean": self.mean,
                "variance": self.variance,
                "volatility": self.volatility,
            }
        ).set_index("horizon")


def to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """Detach a tensor to a NumPy array, leaving arrays untouched."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x, dtype: torch.dtype = torch.float64) -> Tensor:
    """Coerce array-likes, pandas Series and tensors to a 1-D tensor.

    Rejects non-finite input rather than letting a ``nan`` propagate silently
    into a variance recursion, where it would turn the whole path to ``nan``
    several hundred steps later and look like an optimiser failure.
    """
    if isinstance(x, Tensor):
        out = x.to(dtype)
    else:
        values = getattr(x, "values", x)
        # np.array copies: pandas >= 3 hands out read-only views, which torch must not alias
        out = torch.as_tensor(np.array(values, dtype=np.float64), dtype=dtype)
    out = out.squeeze()
    if out.ndim != 1:
        raise ValueError(f"Expected a 1-D series, got shape {tuple(out.shape)}.")
    if not torch.isfinite(out).all():
        n_bad = int((~torch.isfinite(out)).sum())
        raise ValueError(
            f"Input series contains {n_bad} non-finite value(s). Drop or impute "
            "them before fitting: a single NaN silently poisons the entire "
            "variance recursion downstream of it."
        )
    return out
