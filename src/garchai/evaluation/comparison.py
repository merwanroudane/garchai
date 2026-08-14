r"""Formal forecast-comparison tests.

Reporting that model A has a lower RMSE than model B is not evidence that A is
better: the difference has a sampling distribution, and for volatility forecasts
it is usually wide. These are the tests that turn a table of numbers into a
claim.

* :func:`diebold_mariano` — is the loss difference between two models
  significantly different from zero?
* :func:`giacomini_white` — is one model's *conditional* predictive ability
  better, given what we knew at forecast time?
* :func:`model_confidence_set` — which subset of models cannot be distinguished
  from the best at a given confidence level?

The MCS is the right tool when comparing more than two models, because running
pairwise DM tests over ``M`` models invites exactly the multiple-comparison
problem it was designed to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ComparisonResult",
    "newey_west_variance",
    "diebold_mariano",
    "giacomini_white",
    "model_confidence_set",
]


@dataclass
class ComparisonResult:
    """Outcome of a pairwise forecast-comparison test."""

    name: str
    statistic: float
    p_value: float
    mean_loss_difference: float
    n_obs: int
    favours: str
    extra: dict[str, float] | None = None

    @property
    def significant(self) -> bool:
        """Significant at the 5% level."""
        return self.p_value < 0.05

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        verdict = f"favours {self.favours}" if self.significant else "no significant difference"
        return f"{self.name}: stat={self.statistic:.4f}, p={self.p_value:.4f} ({verdict})"


def newey_west_variance(x: np.ndarray, lags: int | None = None) -> float:
    r"""Newey-West HAC estimate of the long-run variance of a series mean.

    .. math::
        \hat\sigma^2_{LR} = \gamma_0
            + 2\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)\gamma_k

    The Bartlett weights guarantee a non-negative estimate. When ``lags`` is
    ``None`` the usual :math:`\lfloor 4(T/100)^{2/9}\rfloor` rule is used.

    Loss differentials from ``h``-step-ahead forecasts are MA(h-1) by
    construction, so ignoring autocorrelation here inflates every test statistic.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))

    centred = x - x.mean()
    variance = float(centred @ centred) / n
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1.0)
        cov = float(centred[k:] @ centred[:-k]) / n
        variance += 2.0 * weight * cov
    return max(variance, 1e-300)


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    lags: int | None = None,
    harvey_correction: bool = True,
    names: tuple[str, str] = ("A", "B"),
) -> ComparisonResult:
    r"""Diebold-Mariano test of equal predictive accuracy.

    With :math:`d_t = L_t^{A} - L_t^{B}`,

    .. math::
        DM = \frac{\bar d}{\sqrt{\hat\sigma^2_{LR}/T}}
        \;\overset{d}{\to}\; N(0,1).

    A **negative** statistic favours model A (it has the lower loss).

    Parameters
    ----------
    horizon
        Forecast horizon. An ``h``-step loss differential is MA(h-1), so the
        default HAC lag length is set to ``h - 1`` when ``lags`` is ``None``.
    harvey_correction
        Apply the Harvey, Leybourne & Newbold (1997) small-sample correction and
        refer the statistic to a :math:`t_{T-1}` distribution. The uncorrected DM
        test over-rejects badly in samples of a few hundred observations, which
        is the norm for out-of-sample volatility evaluation, so this defaults to
        ``True``.

    Notes
    -----
    The test compares *losses*, not forecasts, so it inherits whatever the loss
    function values. Running it on MSE against a squared-return proxy and on
    QLIKE against the same proxy can give different answers, and both are valid
    answers to different questions.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Loss series must align: {a.shape} vs {b.shape}.")
    d = a - b
    n = d.size
    if n < 3:
        raise ValueError("Need at least 3 observations for a Diebold-Mariano test.")

    if lags is None:
        lags = max(horizon - 1, 0)
    variance = newey_west_variance(d, lags=lags)
    statistic = float(d.mean() / np.sqrt(variance / n))

    from scipy.stats import norm, t as student_t

    if harvey_correction:
        factor = (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n
        statistic *= np.sqrt(max(factor, 1e-12))
        p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
    else:
        p_value = float(2.0 * norm.sf(abs(statistic)))

    return ComparisonResult(
        name="Diebold-Mariano" + (" (Harvey-corrected)" if harvey_correction else ""),
        statistic=statistic,
        p_value=p_value,
        mean_loss_difference=float(d.mean()),
        n_obs=n,
        favours=names[0] if d.mean() < 0 else names[1],
        extra={"hac_lags": float(lags), "long_run_variance": variance},
    )


def giacomini_white(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    instruments: np.ndarray | None = None,
    names: tuple[str, str] = ("A", "B"),
) -> ComparisonResult:
    r"""Giacomini-White test of equal *conditional* predictive ability.

    Diebold-Mariano asks whether one model is better on average. Giacomini-White
    asks a sharper question: given information available at forecast time, can we
    *predict* which model will do better? A model that wins only in calm periods
    and loses in crises can pass DM and fail this.

    Regressing :math:`d_t` on instruments :math:`h_{t-1}` (a constant and the
    lagged differential by default), the test statistic is

    .. math::
        GW = T \cdot \bar Z' \hat\Omega^{-1} \bar Z \sim \chi^2_q,
        \qquad Z_t = h_{t-1} d_t .

    Parameters
    ----------
    instruments
        Shape ``(T, q)`` of information available when the forecast was made.
        Defaults to a constant and the lagged loss differential.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Loss series must align: {a.shape} vs {b.shape}.")
    d = a - b

    if instruments is None:
        instruments = np.column_stack([np.ones(d.size - 1), d[:-1]])
        d = d[1:]
    instruments = np.atleast_2d(np.asarray(instruments, dtype=float))
    if instruments.shape[0] != d.size:
        raise ValueError(
            f"instruments must have {d.size} rows, got {instruments.shape[0]}."
        )

    n, q = instruments.shape
    z = instruments * d[:, None]
    z_bar = z.mean(axis=0)
    centred = z - z_bar
    omega = centred.T @ centred / n

    try:
        statistic = float(n * z_bar @ np.linalg.solve(omega, z_bar))
    except np.linalg.LinAlgError:
        statistic = float(n * z_bar @ np.linalg.pinv(omega) @ z_bar)

    from scipy.stats import chi2

    return ComparisonResult(
        name="Giacomini-White (conditional predictive ability)",
        statistic=statistic,
        p_value=float(chi2.sf(statistic, q)),
        mean_loss_difference=float(d.mean()),
        n_obs=n,
        favours=names[0] if d.mean() < 0 else names[1],
        extra={"q": float(q)},
    )


def model_confidence_set(
    losses,
    alpha: float = 0.10,
    n_bootstrap: int = 5_000,
    block_length: int = 5,
    seed: int = 42,
):
    r"""Hansen, Lunde & Nason (2011) Model Confidence Set.

    Iteratively eliminates models until the hypothesis of equal predictive
    ability among the survivors can no longer be rejected. What remains is the
    set of models that cannot be distinguished from the best at confidence
    :math:`1-\alpha` — which is usually more than one, and saying so is more
    honest than declaring a winner.

    Uses the range statistic :math:`T_{\max}` with deviations taken from the
    cross-model average loss, and a stationary bootstrap to get its distribution.

    Parameters
    ----------
    losses
        ``(T, M)`` array or DataFrame; column ``m`` is model ``m``'s loss series.
    alpha
        ``0.10`` gives the 90% MCS.
    block_length
        Expected block length of the stationary bootstrap, which must be long
        enough to preserve the serial dependence in the loss series.

    Returns
    -------
    pandas.DataFrame
        Columns ``model``, ``rank``, ``mean_loss``, ``p_value``, ``in_mcs``.

    Notes
    -----
    The studentisation here divides each :math:`\bar d_i` by the **bootstrap**
    standard deviation of that same quantity, estimated across replications.
    That is what Hansen et al. specify. A tempting shortcut is to divide by the
    spread *across models within* a replication; that quantity has no
    relationship to the sampling variability of :math:`\bar d_i` and produces
    badly miscalibrated p-values.
    """
    import pandas as pd

    if isinstance(losses, pd.DataFrame):
        names = list(losses.columns)
        matrix = losses.to_numpy(dtype=float)
    else:
        matrix = np.asarray(losses, dtype=float)
        names = [f"model_{i}" for i in range(matrix.shape[1])]

    n_periods, n_models = matrix.shape
    if n_models < 2:
        raise ValueError("The MCS needs at least two models.")

    rng = np.random.default_rng(seed)
    # Shared bootstrap indices: every elimination round must resample the same
    # way, or the sequence of tests is not comparable.
    index_draws = np.empty((n_bootstrap, n_periods), dtype=np.int64)
    for b in range(n_bootstrap):
        idx = np.empty(n_periods, dtype=np.int64)
        idx[0] = rng.integers(0, n_periods)
        restart = rng.random(n_periods) < 1.0 / block_length
        jumps = rng.integers(0, n_periods, size=n_periods)
        for t in range(1, n_periods):
            idx[t] = jumps[t] if restart[t] else (idx[t - 1] + 1) % n_periods
        index_draws[b] = idx

    surviving = list(range(n_models))
    p_values: dict[int, float] = {}
    running_max = 0.0

    while len(surviving) > 1:
        block = matrix[:, surviving]
        d = block - block.mean(axis=1, keepdims=True)
        d_bar = d.mean(axis=0)

        boot_means = np.stack([d[index_draws[b]].mean(axis=0) for b in range(n_bootstrap)])
        deviations = boot_means - d_bar
        # Bootstrap variance of d_bar_i, across replications - the studentiser.
        scale = np.sqrt(np.maximum((deviations**2).mean(axis=0), 1e-300))

        t_observed = float(np.max(np.abs(d_bar) / scale))
        t_bootstrap = np.max(np.abs(deviations) / scale, axis=1)
        p = float(np.mean(t_bootstrap >= t_observed))

        # p-values must be monotone along the elimination sequence.
        running_max = max(running_max, p)
        if running_max >= alpha:
            for i in surviving:
                p_values.setdefault(i, running_max)
            break

        worst = surviving[int(np.argmax(d_bar))]
        p_values[worst] = running_max
        surviving.remove(worst)

    for i in surviving:
        p_values.setdefault(i, 1.0)

    mean_loss = matrix.mean(axis=0)
    order = np.argsort(mean_loss)
    rank = {int(model): position + 1 for position, model in enumerate(order)}

    return (
        pd.DataFrame(
            [
                {
                    "model": names[i],
                    "rank": rank[i],
                    "mean_loss": float(mean_loss[i]),
                    "p_value": p_values[i],
                    "in_mcs": bool(p_values[i] >= alpha),
                }
                for i in range(n_models)
            ]
        )
        .sort_values("rank")
        .reset_index(drop=True)
    )
