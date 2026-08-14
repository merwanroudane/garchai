r"""Li & Zhang (2026): GARCH-FIS — volatility-driven fuzzy parameter adaptation.

A Takagi-Sugeno fuzzy inference system whose **structure never changes** — the
rule base and the universe partition are fixed — while the *widths* of its
antecedent membership functions are rescaled each period by the GARCH volatility
forecast:

.. math::
    \mu_{ij}(x) = \exp\!\left(
        -\frac{(x - c_{ij})^2}{2\,\big(\kappa\,\hat\sigma_{t+1}\,w_{ij}\big)^2}
    \right)

That single idea is what the paper is about. A fuzzy set answers "is the return
*high*?", and the honest answer depends on the regime: a 2% move is extreme in a
calm market and unremarkable in a crisis. Fixing the membership widths forces one
answer for both. Scaling them by :math:`\hat\sigma_{t+1}` makes "high" mean
*high relative to current conditions*, so the same rule base stays meaningful
across regimes without being refitted.

The loop (paper Fig. 1) is closed:

.. code-block:: text

    X_t --FIS(Theta_t)--> P_hat_{t+1} --GARCH--> sigma_{t+1}
      ^                                              |
      |______ Theta_{t+1} <-- embed (P_hat, sigma) __|

Only the antecedent parameters move. Consequents are fitted once by least
squares, which keeps the system interpretable — every rule remains readable as
"if the inputs are like *this*, the output is that linear function".

Reference
---------
Li, W.-J., & Zhang, D.-Q. (2026). GARCH-FIS: A Hybrid Forecasting Model with
Dynamic Volatility-Driven Parameter Adaptation. https://arxiv.org/abs/2603.14793
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = ["GARCHFIS", "FuzzyRuleBase"]


@dataclass
class FuzzyRuleBase:
    """A fixed Takagi-Sugeno rule base: Gaussian antecedents, linear consequents."""

    centres: np.ndarray
    """Shape ``(n_rules, n_inputs)`` — the fixed universe partition."""

    base_widths: np.ndarray
    """Shape ``(n_rules, n_inputs)`` — widths *before* volatility scaling."""

    consequents: np.ndarray | None = None
    """Shape ``(n_rules, n_inputs + 1)`` — fitted by least squares."""

    @property
    def n_rules(self) -> int:
        return self.centres.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.centres.shape[1]

    def firing_strengths(self, x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        r"""Rule activations, with every width multiplied by ``scale``.

        ``scale`` is where the GARCH volatility enters. At ``scale = 1`` this is
        an ordinary TS-FIS; raising it widens every fuzzy set, so more rules fire
        and the output is smoother — which is the desired behaviour when
        volatility is high and sharp categorical judgements are least reliable.
        """
        widths = np.maximum(self.base_widths * scale, 1e-8)
        exponent = -0.5 * ((x[None, :] - self.centres) / widths) ** 2
        strengths = np.exp(exponent).prod(axis=1)
        total = strengths.sum()
        return strengths / total if total > 1e-12 else np.full(self.n_rules, 1.0 / self.n_rules)


class GARCHFIS:
    r"""GARCH-FIS: a fuzzy forecaster whose fuzzy sets breathe with volatility.

    Parameters
    ----------
    n_rules
        Size of the rule base. Fixed for the life of the model, per the paper.
    kappa
        Sensitivity of the membership widths to volatility. ``kappa = 0`` freezes
        them, which reduces the model to a plain TS-FIS and is the natural
        ablation for measuring what the adaptation contributes.
    window
        Rolling window for the GARCH fit.

    Examples
    --------
    >>> import numpy as np
    >>> from garchai.fuzzy import GARCHFIS
    >>> model = GARCHFIS(n_rules=4, kappa=0.0)
    >>> model.kappa                  # adaptation switched off
    0.0
    """

    provenance = Provenance(
        paper_id="li2026_garch_fis",
        title=(
            "GARCH-FIS: A Hybrid Forecasting Model with Dynamic Volatility-Driven "
            "Parameter Adaptation"
        ),
        authors=("Wen-Jing Li", "Da-Qing Zhang"),
        year=2026,
        paper_url="https://arxiv.org/abs/2603.14793",
        equations="sec. 3.1 (recursive rolling loop), sec. 3.2 (parameter identification)",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "The paper does not give the membership-width scaling law in closed "
            "form; it states only that width and universe coverage are adjusted "
            "by the volatility forecast. We use a multiplicative Gaussian scaling, "
            "width * kappa * sigma_hat, and expose kappa. kappa = 0 recovers a "
            "static FIS, which is the ablation for the paper's contribution.",
            "Rule centres are placed by k-means on the training inputs; the paper "
            "does not state its partitioning method.",
            "Consequents are fitted once by least squares rather than re-estimated "
            "each step, matching the paper's statement that only antecedent "
            "parameters adapt.",
        ),
    )

    def __init__(
        self,
        n_rules: int = 8,
        kappa: float = 1.0,
        window: int = 250,
        n_lags: int = 3,
        refit_every: int = 1,
        seed: int = 0,
    ):
        self.n_rules = n_rules
        self.kappa = kappa
        self.window = window
        self.n_lags = n_lags
        if refit_every < 1:
            raise ValueError("refit_every must be at least 1.")
        self.refit_every = refit_every
        self.seed = seed
        self.rules: FuzzyRuleBase | None = None
        self._sigma_reference = 1.0
        self._filter_state = None

    @property
    def uses_exact_protocol(self) -> bool:
        """``True`` only when GARCH is re-estimated at every step.

        Exposed so a results table can state which protocol produced it. The
        GINN reproduction in this package flipped its conclusion between
        ``refit_every=10`` and ``refit_every=1``, so the distinction is not
        cosmetic.
        """
        return self.refit_every == 1

    # -- construction ------------------------------------------------------- #
    @staticmethod
    def _design(series: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
        """Lagged design matrix and the aligned target."""
        rows = [series[i : len(series) - n_lags + i] for i in range(n_lags)]
        return np.column_stack(rows), series[n_lags:]

    def _build_rules(self, x: np.ndarray) -> FuzzyRuleBase:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.n_rules, random_state=self.seed, n_init=10)
        kmeans.fit(x)
        centres = kmeans.cluster_centers_
        # A width of half the mean inter-centre distance gives neighbouring sets
        # meaningful overlap without collapsing them into one another.
        distances = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        spacing = distances.min(axis=1)[:, None]
        widths = np.repeat(np.maximum(spacing * 0.5, 1e-6), centres.shape[1], axis=1)
        return FuzzyRuleBase(centres=centres, base_widths=widths)

    def _volatility_path(self, series: np.ndarray) -> np.ndarray:
        r"""Rolling one-step GARCH(1,1) volatility forecasts.

        With ``refit_every = 1`` this is the exact protocol: a full MLE fit per
        step. That is O(T) maximum-likelihood optimisations and dominates the
        runtime of everything else in the class. ``refit_every = k`` re-estimates
        the parameters every k-th step and *filters* the variance forward in
        between, which is an approximation — the parameters lag by up to k
        observations — but never uses future data, so it does not leak.
        """
        from arch import arch_model

        out = np.full(series.size, np.nan)
        last: float | None = None
        for t in range(self.window, series.size):
            if (t - self.window) % self.refit_every == 0 or last is None:
                try:
                    fitted = arch_model(
                        series[t - self.window : t], mean="Constant", vol="GARCH", p=1, q=1
                    ).fit(disp="off", show_warning=False)
                    parameters = fitted.params
                    residual = series[t - self.window : t] - parameters["mu"]
                    variance = float(fitted.conditional_volatility[-1] ** 2)
                    last = float(
                        np.sqrt(
                            parameters["omega"]
                            + parameters["alpha[1]"] * residual[-1] ** 2
                            + parameters["beta[1]"] * variance
                        )
                    )
                    self._filter_state = (parameters, variance)
                except Exception:  # noqa: BLE001
                    last = out[t - 1] if t > self.window else float(np.nanstd(series[:t]))
            else:
                # Carry the fitted parameters forward one step of the recursion.
                parameters, variance = self._filter_state
                residual = series[t - 1] - parameters["mu"]
                variance = (
                    parameters["omega"]
                    + parameters["alpha[1]"] * residual**2
                    + parameters["beta[1]"] * variance
                )
                self._filter_state = (parameters, variance)
                last = float(np.sqrt(variance))
            out[t] = last
        # Before the first full window, fall back to the running standard
        # deviation rather than dropping the rows entirely.
        out[: self.window] = np.nanstd(series[: self.window])
        return out

    # -- fitting ------------------------------------------------------------ #
    def fit(self, series) -> GARCHFIS:
        r"""Fit the rule base and consequents on a return series.

        The GARCH volatility path is computed by a **rolling** fit, so the scale
        applied at time ``t`` uses only information available then.
        """
        y = np.asarray(series, dtype=float)
        if y.size <= self.window + self.n_lags + 1:
            raise ValueError(f"Need more than {self.window + self.n_lags + 1} observations.")

        x, target = self._design(y, self.n_lags)
        volatility = self._volatility_path(y)[self.n_lags :]
        self._sigma_reference = float(np.nanmedian(volatility))
        self.rules = self._build_rules(x)

        # Consequents by least squares on the firing-strength-weighted design
        # (the standard TS-FIS identification).
        design = self._weighted_design(x, volatility)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        self.rules.consequents = coefficients.reshape(self.n_rules, self.n_lags + 1)
        return self

    def _scale(self, sigma: float) -> float:
        r"""Width multiplier :math:`\kappa\,\hat\sigma/\bar\sigma`, floored at a small value.

        Normalising by the median volatility keeps the multiplier near 1 in
        ordinary conditions, so ``kappa`` has the same meaning across assets
        whatever their volatility level.
        """
        if self.kappa == 0.0:
            return 1.0
        reference = self._sigma_reference if self._sigma_reference > 1e-12 else 1.0
        return max(self.kappa * float(sigma) / reference, 1e-3)

    def _weighted_design(self, x: np.ndarray, volatility: np.ndarray) -> np.ndarray:
        rows = []
        for i in range(x.shape[0]):
            strengths = self.rules.firing_strengths(x[i], self._scale(volatility[i]))
            augmented = np.concatenate([x[i], [1.0]])
            rows.append(np.outer(strengths, augmented).ravel())
        return np.stack(rows)

    # -- prediction --------------------------------------------------------- #
    def predict(self, series) -> np.ndarray:
        """One-step-ahead forecasts over a series."""
        if self.rules is None or self.rules.consequents is None:
            raise RuntimeError("GARCH-FIS has not been fitted; call .fit(series) first.")
        y = np.asarray(series, dtype=float)
        x, _ = self._design(y, self.n_lags)
        volatility = self._volatility_path(y)[self.n_lags :]

        out = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            strengths = self.rules.firing_strengths(x[i], self._scale(volatility[i]))
            augmented = np.concatenate([x[i], [1.0]])
            out[i] = float(strengths @ (self.rules.consequents @ augmented))
        return out

    def evaluate(self, series) -> dict[str, float]:
        y = np.asarray(series, dtype=float)
        prediction = self.predict(y)
        truth = y[self.n_lags :]
        residual = truth - prediction
        return {
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "R2": float(1.0 - residual.var() / truth.var()),
        }

    def membership_widths(self, sigma: float) -> np.ndarray:
        """The effective widths at a given volatility — the adaptation, visible.

        Plotting these against :math:`\\hat\\sigma_t` is the clearest way to show
        what the model does: the fuzzy sets should visibly widen in turbulent
        periods and tighten in calm ones.
        """
        if self.rules is None:
            raise RuntimeError("GARCH-FIS has not been fitted.")
        return self.rules.base_widths * self._scale(sigma)
