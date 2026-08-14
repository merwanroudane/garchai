r"""Fatima & Uddin (2022): DCC-GARCH hybridised with multivariate ANNs.

Two hybrids, differing only in which model feeds which — and the paper is
explicit that the ordering is the experiment, not an implementation detail:

``direction="mann_to_dcc"``
    **MANN-DCC-GARCH.** The network forecasts the return series; DCC-GARCH is
    then fitted to the network's residuals. The network absorbs conditional
    *mean* structure, leaving DCC a cleaner series in which to find the
    volatility and co-movement.

``direction="dcc_to_mann"``
    **DCC-GARCH-MANN.** DCC-GARCH is fitted first, and its conditional
    volatilities and dynamic correlations become inputs to the network. The
    econometric model supplies the second moments; the network is free to use
    them nonlinearly.

The multivariate ANN is a single network with :math:`n` outputs rather than
:math:`n` separate networks — that is what makes it *multivariate*. It can use
the UK's lagged return when predicting the US, which is exactly the cross-market
spillover a set of independent univariate networks structurally cannot see.

DCC (Engle 2002) itself is two-stage:

.. math::
    Q_t &= (1 - a - b)\bar{Q} + a\,z_{t-1}z_{t-1}' + b\,Q_{t-1} \\
    R_t &= \mathrm{diag}(Q_t)^{-1/2}\,Q_t\,\mathrm{diag}(Q_t)^{-1/2}

with :math:`a, b \ge 0` and :math:`a + b < 1`. The normalisation on the second
line is what makes :math:`R_t` a correlation matrix — unit diagonal, and
positive definite whenever :math:`Q_t` is.

Reference
---------
Fatima, S., & Uddin, M. (2022). On the forecasting of multivariate financial time
series using hybridization of DCC-GARCH model and multivariate ANNs. *Neural
Computing and Applications*, 34, 21911-21925.
https://doi.org/10.1007/s00521-022-07631-5
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..core.config import TrainConfig, set_seed
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = ["DCCGARCHMANN", "DCCResult", "MultivariateANN", "fit_dcc"]


@dataclass
class DCCResult:
    """Fitted DCC-GARCH: univariate volatilities plus the correlation path."""

    sigma: np.ndarray
    """Shape ``(T, n)`` — conditional standard deviations from stage one."""

    correlations: np.ndarray
    """Shape ``(T, n, n)`` — the dynamic conditional correlation matrices."""

    a: float
    b: float
    standardised: np.ndarray
    """Shape ``(T, n)`` — the :math:`z_t` that stage two operates on."""

    @property
    def persistence(self) -> float:
        """:math:`a + b`. Must be < 1 for the correlation process to be stationary."""
        return self.a + self.b

    def pairwise(self, i: int, j: int) -> np.ndarray:
        """The correlation path between two series — the paper's headline figure."""
        return self.correlations[:, i, j]


def fit_dcc(returns, *, max_iter: int = 200) -> DCCResult:
    r"""Two-stage DCC-GARCH(1,1) estimation (Engle 2002).

    Stage one fits a univariate GARCH(1,1) per series; stage two fits
    :math:`(a, b)` on the standardised residuals by quasi-maximum likelihood.
    """
    from arch import arch_model
    from scipy.optimize import minimize

    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, n); got shape {r.shape}.")
    n_obs, n_series = r.shape
    if n_series < 2:
        raise ValueError("DCC needs at least two series.")

    sigma = np.empty_like(r)
    for j in range(n_series):
        fitted = arch_model(r[:, j], mean="Constant", vol="GARCH", p=1, q=1).fit(
            disp="off", show_warning=False
        )
        sigma[:, j] = np.sqrt(fitted.conditional_volatility**2)

    z = r / np.where(sigma > 1e-12, sigma, 1.0)
    q_bar = np.cov(z, rowvar=False)

    def negative_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        if a < 0 or b < 0 or a + b >= 0.9999:
            return 1e10
        q = q_bar.copy()
        total = 0.0
        for t in range(n_obs):
            diagonal = np.sqrt(np.diag(q))
            correlation = q / np.outer(diagonal, diagonal)
            try:
                sign, logdet = np.linalg.slogdet(correlation)
                if sign <= 0:
                    return 1e10
                solved = np.linalg.solve(correlation, z[t])
            except np.linalg.LinAlgError:
                return 1e10
            total += logdet + z[t] @ solved
            q = (1 - a - b) * q_bar + a * np.outer(z[t], z[t]) + b * q
        return 0.5 * total

    result = minimize(
        negative_log_likelihood,
        x0=np.array([0.02, 0.95]),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-6, "fatol": 1e-6},
    )
    a, b = float(result.x[0]), float(result.x[1])
    a, b = max(a, 0.0), max(b, 0.0)
    if a + b >= 1.0:  # numerical safety; the optimiser is penalised, not constrained
        scale = 0.999 / (a + b)
        a, b = a * scale, b * scale

    correlations = np.empty((n_obs, n_series, n_series))
    q = q_bar.copy()
    for t in range(n_obs):
        diagonal = np.sqrt(np.diag(q))
        correlations[t] = q / np.outer(diagonal, diagonal)
        q = (1 - a - b) * q_bar + a * np.outer(z[t], z[t]) + b * q

    return DCCResult(sigma=sigma, correlations=correlations, a=a, b=b, standardised=z)


class MultivariateANN(nn.Module):
    """One network, ``n_outputs`` heads — cross-series information is shared."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_sizes: tuple[int, ...] = (32, 16),
        activation: str = "tanh",
    ):
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}[activation]
        layers: list[nn.Module] = []
        previous = n_inputs
        for size in hidden_sizes:
            layers += [nn.Linear(previous, size), act()]
            previous = size
        layers.append(nn.Linear(previous, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCCGARCHMANN:
    r"""DCC-GARCH hybridised with a multivariate ANN, in either direction.

    Parameters
    ----------
    direction
        ``"dcc_to_mann"`` or ``"mann_to_dcc"`` — the paper's two hybrids.
    n_lags
        Lagged returns of *every* series fed to the network.

    Examples
    --------
    >>> from garchai.multivariate import DCCGARCHMANN
    >>> model = DCCGARCHMANN(direction="dcc_to_mann")
    >>> model.direction
    'dcc_to_mann'
    """

    provenance = Provenance(
        paper_id="fatima2022_dcc_mann",
        title=(
            "On the forecasting of multivariate financial time series using "
            "hybridization of DCC-GARCH model and multivariate ANNs"
        ),
        authors=("Samreen Fatima", "Mudassir Uddin"),
        year=2022,
        paper_url="https://doi.org/10.1007/s00521-022-07631-5",
        equations="DCC-GARCH(1,1) (Engle 2002); both hybridisation directions, sec. 3",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "The paper does not report the ANN hidden-layer sizes, activation or "
            "training schedule; defaults here are (32, 16) with tanh, and all are "
            "constructor arguments.",
            "DCC stage two is estimated by Nelder-Mead on the QMLE objective. The "
            "paper does not state its optimiser.",
        ),
        data_substitutions=(
            "The paper uses S&P 500, FTSE-100, KSE-100, KLSE and BSESN over "
            "2013-01-01 to 2020-03-17. KSE-100 and KLSE are not available from the "
            "bundled public sources; the worked example substitutes freely "
            "available indices and says so in the results table.",
        ),
    )

    def __init__(
        self,
        direction: str = "dcc_to_mann",
        n_lags: int = 3,
        hidden_sizes: tuple[int, ...] = (32, 16),
        activation: str = "tanh",
        seed: int = 0,
    ):
        if direction not in ("dcc_to_mann", "mann_to_dcc"):
            raise ValueError(
                f"direction must be 'dcc_to_mann' or 'mann_to_dcc'; got {direction!r}."
            )
        self.direction = direction
        self.n_lags = n_lags
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.seed = seed
        self.dcc: DCCResult | None = None
        self.network: MultivariateANN | None = None
        self._mean = self._scale = None

    # -- feature assembly --------------------------------------------------- #
    def _lagged(self, r: np.ndarray) -> np.ndarray:
        """All series, all lags — the cross-market block."""
        return np.column_stack(
            [r[i : r.shape[0] - self.n_lags + i, :] for i in range(self.n_lags)]
        )

    def build_inputs(self, returns) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(returns, dtype=float)
        if r.ndim != 2:
            raise ValueError(f"returns must be 2-D (T, n); got shape {r.shape}.")
        lagged = self._lagged(r)
        target = r[self.n_lags :, :]

        if self.direction == "dcc_to_mann":
            if self.dcc is None:
                self.dcc = fit_dcc(r)
            sigma = self.dcc.sigma[self.n_lags :, :]
            # The upper triangle of R_t, excluding the unit diagonal, is the
            # non-redundant part of the correlation matrix.
            iu = np.triu_indices(r.shape[1], k=1)
            correlations = self.dcc.correlations[self.n_lags :][:, iu[0], iu[1]]
            features = np.column_stack([lagged, sigma, correlations])
        else:
            features = lagged
        return features, target

    # -- fitting ------------------------------------------------------------ #
    def fit(self, returns, *, train: TrainConfig | None = None) -> DCCGARCHMANN:
        train = train or TrainConfig(lr=1e-3, max_epochs=300, patience=30)
        set_seed(self.seed)
        r = np.asarray(returns, dtype=float)

        features, target = self.build_inputs(r)
        self._mean = features.mean(axis=0)
        self._scale = np.where(features.std(axis=0) > 1e-12, features.std(axis=0), 1.0)
        dtype = torch.get_default_dtype()
        x = torch.as_tensor((features - self._mean) / self._scale, dtype=dtype)
        y = torch.as_tensor(target, dtype=dtype)

        self.network = MultivariateANN(
            x.shape[1], y.shape[1], self.hidden_sizes, self.activation
        )
        optimiser = torch.optim.Adam(self.network.parameters(), lr=train.lr)
        best, patience, best_state = float("inf"), 0, None
        for _ in range(train.max_epochs):
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(self.network(x), y)
            loss.backward()
            optimiser.step()
            value = float(loss.detach())
            if value < best - 1e-10:
                best, patience = value, 0
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
            else:
                patience += 1
                if patience >= train.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)

        if self.direction == "mann_to_dcc":
            # DCC on the network's residuals — the second half of this direction.
            with torch.no_grad():
                fitted = self.network(x).numpy()
            self.dcc = fit_dcc(target - fitted)
        return self

    def predict(self, returns) -> np.ndarray:
        """Conditional mean forecasts, shape ``(T - n_lags, n)``."""
        if self.network is None:
            raise RuntimeError("Model has not been fitted; call .fit(returns) first.")
        features, _ = self.build_inputs(np.asarray(returns, dtype=float))
        with torch.no_grad():
            return self.network(
                torch.as_tensor(
                    (features - self._mean) / self._scale,
                    dtype=torch.get_default_dtype(),
                )
            ).numpy()

    def conditional_correlations(self) -> np.ndarray:
        if self.dcc is None:
            raise RuntimeError("No DCC fit available; call .fit(returns) first.")
        return self.dcc.correlations

    def evaluate(self, returns) -> dict[str, float]:
        """Per-series and pooled RMSE/MAE — the paper's Table 5 layout."""
        r = np.asarray(returns, dtype=float)
        prediction = self.predict(r)
        truth = r[self.n_lags :, :]
        residual = truth - prediction
        metrics: dict[str, float] = {
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "MAE": float(np.mean(np.abs(residual))),
        }
        for j in range(r.shape[1]):
            metrics[f"RMSE_series{j}"] = float(np.sqrt(np.mean(residual[:, j] ** 2)))
            metrics[f"MAE_series{j}"] = float(np.mean(np.abs(residual[:, j])))
        if self.dcc is not None:
            metrics["dcc_a"] = self.dcc.a
            metrics["dcc_b"] = self.dcc.b
            metrics["dcc_persistence"] = self.dcc.persistence
        return metrics
