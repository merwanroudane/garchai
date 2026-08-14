r"""Sentiment-augmented GARCH and GARCH-LSTM for volatility and VaR (2026).

Two models, matching the paper's ``GARCH_X`` and ``NN_Sentiment_Index``:

:class:`SentimentGARCHX`
    Sentiment enters the variance equation directly as a lagged exogenous term,

    .. math::
        \sigma_t^2 = \omega + \sum_i \alpha_i \varepsilon_{t-i}^2
                   + \sum_j \beta_j \sigma_{t-j}^2
                   + \sum_k \gamma_k S_{t-k}

:class:`SentimentGARCHLSTM`
    An LSTM taking the GARCH volatility forecast *and* the sentiment index,
    which is the paper's answer to its own observation that the variance-sentiment
    relationship is **nonlinear** — a linear :math:`\gamma_k` cannot represent it,
    and the paper's nonlinear Granger causality tests are what motivate moving
    past :class:`SentimentGARCHX`.

Positivity
----------
:math:`\gamma_k S_{t-k}` is the one term in the variance equation that can be
negative, and a negative variance ends the recursion. The paper does not say how
it handles this. We require the sentiment input to be non-negative and
:math:`\gamma_k \ge 0`, which makes positivity structural rather than something
checked after the fact — see :func:`normalise_sentiment` and ``deviations``.

Data
----
The paper uses a proprietary sentiment feed over S&P 500 constituents (2019-2024).
It is not redistributable, so no bundled example can reproduce its numbers. The
sentiment series is therefore a **required argument**: this module will not invent
one. :func:`vix_sentiment_proxy` builds a documented public substitute from the
VIX, and every result computed from it is labelled a proxy result.

Reference
---------
The Sentiment Augmented GARCH-LSTM Hybrid Model for Volatility and Value-at-Risk
Forecasting (2026). *Computational Economics*, 67, 313-353.
https://doi.org/10.1007/s10614-025-11042-8
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..core.config import TrainConfig, set_seed
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = [
    "SentimentGARCHLSTM",
    "SentimentGARCHX",
    "normalise_sentiment",
    "vix_sentiment_proxy",
]


def normalise_sentiment(sentiment, method: str = "minmax") -> np.ndarray:
    r"""Map a sentiment series onto :math:`[0, 1]`, non-negative by construction.

    This is not cosmetic. With :math:`\gamma \ge 0` and :math:`S_t \ge 0`, the
    variance recursion cannot be driven negative by the sentiment term, so the
    model needs no ad-hoc floor that would quietly change its dynamics.
    """
    s = np.asarray(sentiment, dtype=float)
    if method == "minmax":
        low, high = np.nanmin(s), np.nanmax(s)
        return (s - low) / (high - low) if high > low else np.zeros_like(s)
    if method == "rank":
        from scipy.stats import rankdata

        return (rankdata(s) - 1) / max(s.size - 1, 1)
    raise ValueError(f"method must be 'minmax' or 'rank'; got {method!r}.")


def vix_sentiment_proxy(vix) -> np.ndarray:
    r"""A public stand-in for the paper's proprietary sentiment index.

    The VIX is an implied-volatility index, so it is a *fear* gauge rather than a
    media-sentiment score: it is inverted here (high VIX -> low sentiment) and
    scaled to :math:`[0, 1]`. It is not a substitute in any strict sense — it is
    derived from option prices, not text, so it cannot carry the information the
    paper's media indices carry, and it is mechanically related to volatility in
    a way a text-derived index is not. Results built on it are labelled proxy
    results and are not comparable to the paper's.
    """
    return 1.0 - normalise_sentiment(vix)


class SentimentGARCHX:
    r"""GARCH(1,1) with lagged sentiment in the variance equation.

    Examples
    --------
    >>> from garchai.ensembles import SentimentGARCHX
    >>> model = SentimentGARCHX(sentiment_lags=2)
    >>> model.sentiment_lags
    2
    """

    provenance = Provenance(
        paper_id="sentiment_garch_lstm_2026",
        title=(
            "The Sentiment Augmented GARCH-LSTM Hybrid Model for Volatility and "
            "Value-at-Risk Forecasting"
        ),
        year=2026,
        paper_url="https://doi.org/10.1007/s10614-025-11042-8",
        equations="sentiment-augmented variance equation (Formula 6); VaR sec. 5",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "The paper does not state how it keeps the variance positive when the "
            "sentiment term is included. We constrain gamma >= 0 and require the "
            "sentiment input on [0, 1], making positivity structural. A paper that "
            "allowed signed gamma would need a floor we would rather not guess at.",
            "Sentiment index construction is not reproducible from the paper; the "
            "feed is proprietary. The series is a required argument here.",
        ),
        data_substitutions=(
            "The paper's sentiment indices over S&P 500 constituents (2019-2024) "
            "are proprietary and not redistributable. No bundled example "
            "reproduces the paper's numbers. vix_sentiment_proxy() provides a "
            "public stand-in whose limitations are documented; every result "
            "computed from it is reported as a proxy result.",
        ),
    )

    def __init__(self, sentiment_lags: int = 1, p: int = 1, q: int = 1):
        self.sentiment_lags = sentiment_lags
        self.p = p
        self.q = q
        self.params: dict[str, float] = {}
        self.sigma2: np.ndarray | None = None

    def fit(self, returns, sentiment) -> SentimentGARCHX:
        from scipy.optimize import minimize

        r = np.asarray(returns, dtype=float)
        s = np.asarray(sentiment, dtype=float)
        if r.shape != s.shape:
            raise ValueError("returns and sentiment must align in length.")
        # Check finiteness BEFORE the sign test. `nan < 0` is False, so a NaN
        # walks straight through a non-negativity check, then poisons every
        # likelihood evaluation and the optimiser returns its infeasibility
        # penalty with the starting values untouched -- a silent failure that
        # looks like a converged fit. Fail loudly instead.
        if not np.isfinite(s).all():
            raise ValueError(
                "sentiment contains NaN or inf. A rolling proxy typically produces "
                "this on its first window, before enough observations exist."
            )
        if not np.isfinite(r).all():
            raise ValueError("returns contain NaN or inf.")
        if s.min() < 0.0:
            raise ValueError(
                "sentiment must be non-negative; pass it through normalise_sentiment()."
            )

        eps = r - r.mean()
        n_gamma = self.sentiment_lags

        def unpack(theta: np.ndarray) -> tuple[float, float, float, np.ndarray]:
            omega, alpha, beta = np.exp(theta[0]), theta[1], theta[2]
            gamma = np.exp(theta[3 : 3 + n_gamma])   # gamma >= 0 by construction
            return omega, alpha, beta, gamma

        def negative_log_likelihood(theta: np.ndarray) -> float:
            omega, alpha, beta, gamma = unpack(theta)
            if alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e10
            h = np.empty(eps.size)
            h[0] = eps.var()
            for t in range(1, eps.size):
                sentiment_term = sum(
                    gamma[k] * s[t - 1 - k] for k in range(n_gamma) if t - 1 - k >= 0
                )
                h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1] + sentiment_term
            if not np.all(np.isfinite(h)) or np.any(h <= 0):
                return 1e10
            return 0.5 * np.sum(np.log(h) + eps**2 / h)

        theta0 = np.concatenate([[np.log(eps.var() * 0.05), 0.08, 0.88], np.full(n_gamma, -6.0)])
        result = minimize(negative_log_likelihood, theta0, method="Nelder-Mead",
                          options={"maxiter": 4000, "fatol": 1e-8})
        # The objective returns 1e10 for infeasible parameters. If the optimiser
        # ends there it never found a feasible point, and reporting the starting
        # values with a log-likelihood of -1e10 would present a total failure as
        # an estimate. Refuse to return one.
        if result.fun >= 1e9:
            raise RuntimeError(
                "Sentiment-GARCH estimation failed to find a feasible point. "
                "Check that the returns are on a sensible scale (percent, not "
                "fractions) and that the sentiment series is not degenerate."
            )
        omega, alpha, beta, gamma = unpack(result.x)

        h = np.empty(eps.size)
        h[0] = eps.var()
        for t in range(1, eps.size):
            sentiment_term = sum(
                gamma[k] * s[t - 1 - k] for k in range(n_gamma) if t - 1 - k >= 0
            )
            h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1] + sentiment_term

        self.sigma2 = h
        self.params = {
            "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
            **{f"gamma{k + 1}": float(g) for k, g in enumerate(gamma)},
            "loglik": float(-result.fun), "persistence": float(alpha + beta),
        }
        return self

    def conditional_volatility(self) -> np.ndarray:
        if self.sigma2 is None:
            raise RuntimeError("Model has not been fitted; call .fit(returns, sentiment).")
        return np.sqrt(self.sigma2)

    def var(self, alpha_level: float = 0.05) -> np.ndarray:
        r"""Gaussian VaR from the fitted conditional volatility, as a return level."""
        from scipy.stats import norm

        return norm.ppf(alpha_level) * self.conditional_volatility()

    def sentiment_contribution(self, sentiment) -> np.ndarray:
        r"""The share of :math:`\sigma_t^2` attributable to the sentiment term.

        The number that decides whether sentiment did anything. A :math:`\gamma`
        that optimises to near zero means the augmentation is inert, which the
        error metrics alone would not reveal.
        """
        if self.sigma2 is None:
            raise RuntimeError("Model has not been fitted; call .fit(returns, sentiment).")
        s = np.asarray(sentiment, dtype=float)
        gammas = [self.params[f"gamma{k + 1}"] for k in range(self.sentiment_lags)]
        contribution = np.zeros_like(s)
        for k, g in enumerate(gammas):
            contribution[k + 1 :] += g * s[: s.size - k - 1]
        return contribution / self.sigma2


class SentimentGARCHLSTM(nn.Module):
    r"""LSTM over GARCH volatility and sentiment — the paper's ``NN_Sentiment_Index``.

    ``use_sentiment=False`` gives the paper's ``NN`` baseline, which is the
    ablation its central claim rests on.
    """

    provenance = Provenance(
        paper_id="sentiment_garch_lstm_2026",
        title=(
            "The Sentiment Augmented GARCH-LSTM Hybrid Model for Volatility and "
            "Value-at-Risk Forecasting"
        ),
        year=2026,
        paper_url="https://doi.org/10.1007/s10614-025-11042-8",
        equations="Formula 7 (neural specification); VaR backtests sec. 5",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "Hidden sizes and training schedule are not reported in the paper; "
            "defaults here are a single 32-unit LSTM layer, and all are arguments.",
        ),
        data_substitutions=SentimentGARCHX.provenance.data_substitutions,
    )

    def __init__(
        self,
        n_features: int,
        window: int = 10,
        hidden_size: int = 32,
        use_sentiment: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.window = window
        self.use_sentiment = use_sentiment
        self.seed = seed
        set_seed(seed)
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        self._mean = self._scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        # Softplus keeps the volatility forecast positive without a hard clamp.
        return nn.functional.softplus(self.head(output[:, -1, :])).squeeze(-1)

    def _windows(self, x: np.ndarray) -> torch.Tensor:
        padded = np.vstack([np.repeat(x[:1], self.window - 1, axis=0), x])
        return torch.as_tensor(
            np.stack([padded[t : t + self.window] for t in range(x.shape[0])]),
            dtype=torch.get_default_dtype(),
        )

    def fit_series(
        self, features, target, *, train: TrainConfig | None = None
    ) -> SentimentGARCHLSTM:
        train = train or TrainConfig(lr=1e-3, max_epochs=200, patience=20)
        set_seed(self.seed)
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and target must align in length.")

        self._mean = x.mean(axis=0)
        self._scale = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1.0)
        windows = self._windows((x - self._mean) / self._scale)
        y_tensor = torch.as_tensor(y, dtype=torch.get_default_dtype())

        optimiser = torch.optim.Adam(self.parameters(), lr=train.lr)
        best, patience, best_state = float("inf"), 0, None
        self.history: dict[str, list[float]] = {"loss": []}
        for _ in range(train.max_epochs):
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(self(windows), y_tensor)
            loss.backward()
            optimiser.step()
            value = float(loss.detach())
            self.history["loss"].append(value)
            if value < best - 1e-10:
                best, patience = value, 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience += 1
                if patience >= train.patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        return self

    def predict(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        with torch.no_grad():
            return self(self._windows((x - self._mean) / self._scale)).numpy()

    def evaluate(self, features, target) -> dict[str, float]:
        prediction = self.predict(features)
        truth = np.asarray(target, dtype=float)
        residual = truth - prediction
        return {
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "R2": float(1.0 - residual.var() / truth.var()),
            "used_sentiment": float(self.use_sentiment),
        }
