r"""Value-at-Risk backtests.

Sign convention, stated once and used everywhere: VaR is reported as a **return
level**, so at :math:`\alpha = 0.01` it is a negative number and a violation is
:math:`r_t < \mathrm{VaR}_t`. The literature is split between this and the
positive-loss convention; mixing them silently flips every exceedance count.

Four tests are provided, and they answer different questions:

* :func:`kupiec_pof` — is the *number* of violations right?
* :func:`christoffersen_independence` — are violations *clustered*?
* :func:`christoffersen_conditional_coverage` — both at once.
* :func:`dynamic_quantile_test` — are violations predictable from anything
  observable, including the VaR forecast itself?

A model can pass Kupiec and still be useless: getting 1% violations is easy if
they all arrive in the same fortnight. That is exactly what the independence
tests are for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "BacktestResult",
    "violations",
    "violation_ratio",
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_conditional_coverage",
    "dynamic_quantile_test",
    "expected_shortfall_backtest",
    "var_backtest_table",
]


@dataclass
class BacktestResult:
    """Outcome of a single backtest."""

    name: str
    statistic: float
    p_value: float
    df: int
    alpha: float
    n_obs: int
    n_violations: int
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        """Reject the null at the 5% level."""
        return self.p_value < 0.05

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        verdict = "REJECT" if self.rejected else "pass"
        return (
            f"{self.name}: stat={self.statistic:.4f}, p={self.p_value:.4f} "
            f"[{verdict} at 5%]"
        )


def _chi2_sf(statistic: float, df: int) -> float:
    """Upper-tail probability of a chi-squared variate."""
    if not math.isfinite(statistic) or statistic < 0:
        return float("nan")
    try:
        from scipy.stats import chi2

        return float(chi2.sf(statistic, df))
    except ImportError:  # pragma: no cover - SciPy is a hard dependency
        from math import erfc, exp, sqrt

        if df == 1:
            return erfc(sqrt(statistic / 2.0))
        if df == 2:
            return exp(-statistic / 2.0)
        raise


def _as_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def violations(returns: Tensor | np.ndarray, var: Tensor | np.ndarray) -> np.ndarray:
    r"""Boolean exceedance indicator, ``returns < var``.

    Both arguments are on the **return** scale, so ``var`` is negative for small
    ``alpha``.
    """
    r, v = _as_numpy(returns), _as_numpy(var)
    if r.shape != v.shape:
        raise ValueError(f"returns and var must align: {r.shape} vs {v.shape}.")
    return r < v


def violation_ratio(returns, var, alpha: float) -> float:
    r"""Observed violation rate divided by ``alpha``.

    A well-calibrated model gives 1. Above 1 the model understates risk; below 1
    it is too conservative. Wei et al. (2025) report this alongside the pinball
    loss.
    """
    hits = violations(returns, var)
    return float(hits.mean() / alpha)


def kupiec_pof(returns, var, alpha: float) -> BacktestResult:
    r"""Kupiec (1995) proportion-of-failures test of unconditional coverage.

    Under the null that violations occur with probability :math:`\alpha`,

    .. math::
        LR_{uc} = -2\log\frac{(1-\alpha)^{n-x}\alpha^{x}}
                             {(1-\hat\pi)^{n-x}\hat\pi^{x}}
        \;\sim\; \chi^2_1, \qquad \hat\pi = x/n

    where ``x`` is the number of violations in ``n`` observations.

    Tests only the *count*. A model that produces the right number of
    violations, all in one week, passes this comfortably.
    """
    hits = violations(returns, var)
    n, x = hits.size, int(hits.sum())
    pi_hat = x / n if n else float("nan")

    if x == 0 or x == n:
        # The unrestricted likelihood is degenerate; fall back to the
        # restricted-only form, which is still the correct LR here.
        log_restricted = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
        statistic = -2.0 * log_restricted
    else:
        log_restricted = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
        log_unrestricted = (n - x) * math.log(1 - pi_hat) + x * math.log(pi_hat)
        statistic = -2.0 * (log_restricted - log_unrestricted)

    return BacktestResult(
        name="Kupiec POF (unconditional coverage)",
        statistic=statistic,
        p_value=_chi2_sf(statistic, 1),
        df=1,
        alpha=alpha,
        n_obs=n,
        n_violations=x,
        extra={"observed_rate": pi_hat, "expected_rate": alpha},
    )


def _transition_counts(hits: np.ndarray) -> tuple[int, int, int, int]:
    """Counts ``n_ij`` of transitions from state ``i`` to state ``j``."""
    previous, current = hits[:-1], hits[1:]
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))
    return n00, n01, n10, n11


def christoffersen_independence(returns, var, alpha: float) -> BacktestResult:
    r"""Christoffersen (1998) test that violations are serially independent.

    Compares a first-order Markov chain for the hit sequence against an i.i.d.
    Bernoulli, with :math:`LR_{ind} \sim \chi^2_1`. A rejection means violations
    cluster — the model is slow to widen its intervals when volatility rises,
    which is the failure mode that actually costs money.
    """
    hits = violations(returns, var)
    n00, n01, n10, n11 = _transition_counts(hits)

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if hits.size > 1 else 0.0

    def _term(count: int, prob: float) -> float:
        return count * math.log(prob) if count and 0.0 < prob < 1.0 else 0.0

    log_markov = _term(n00, 1 - pi01) + _term(n01, pi01) + _term(n10, 1 - pi11) + _term(n11, pi11)
    log_iid = _term(n00 + n10, 1 - pi) + _term(n01 + n11, pi)
    statistic = -2.0 * (log_iid - log_markov)
    statistic = max(statistic, 0.0)

    return BacktestResult(
        name="Christoffersen independence",
        statistic=statistic,
        p_value=_chi2_sf(statistic, 1),
        df=1,
        alpha=alpha,
        n_obs=hits.size,
        n_violations=int(hits.sum()),
        extra={"pi01": pi01, "pi11": pi11, "n00": n00, "n01": n01, "n10": n10, "n11": n11},
    )


def christoffersen_conditional_coverage(returns, var, alpha: float) -> BacktestResult:
    r"""Joint test of correct coverage *and* independence.

    .. math::
        LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2_2

    This is the test to quote if you quote only one, but report the components
    too: they say *how* a model failed, and the two failure modes call for
    different fixes.
    """
    uc = kupiec_pof(returns, var, alpha)
    ind = christoffersen_independence(returns, var, alpha)
    statistic = uc.statistic + ind.statistic
    return BacktestResult(
        name="Christoffersen conditional coverage",
        statistic=statistic,
        p_value=_chi2_sf(statistic, 2),
        df=2,
        alpha=alpha,
        n_obs=uc.n_obs,
        n_violations=uc.n_violations,
        extra={"lr_uc": uc.statistic, "lr_ind": ind.statistic},
    )


def dynamic_quantile_test(returns, var, alpha: float, lags: int = 4) -> BacktestResult:
    r"""Engle & Manganelli (2004) dynamic quantile test.

    Regresses the demeaned hit sequence :math:`\mathrm{Hit}_t = I_t - \alpha` on
    a constant, its own lags, and the contemporaneous VaR forecast:

    .. math::
        \mathrm{Hit}_t = \delta_0 + \sum_{k=1}^{K}\delta_k \mathrm{Hit}_{t-k}
                       + \delta_{K+1}\mathrm{VaR}_t + u_t

    Under correct conditional coverage every coefficient is zero, and

    .. math::
        DQ = \frac{\hat\delta' X'X \hat\delta}{\alpha(1-\alpha)}
             \sim \chi^2_{K+2}.

    Strictly more demanding than Christoffersen: it rejects when violations are
    predictable from *anything* in the regressor set, including the model's own
    forecast — which catches a model whose VaR is systematically too tight
    precisely when it predicts high volatility.
    """
    hits = violations(returns, var).astype(np.float64)
    v = _as_numpy(var).astype(np.float64)
    n = hits.size
    if n <= lags + 2:
        raise ValueError(f"Need more than {lags + 2} observations for a DQ test with {lags} lags.")

    y = hits[lags:] - alpha
    columns = [np.ones(n - lags)]
    for k in range(1, lags + 1):
        columns.append(hits[lags - k : n - k] - alpha)
    columns.append(v[lags:])
    x = np.column_stack(columns)

    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    statistic = float(coefficients @ (x.T @ x) @ coefficients / (alpha * (1.0 - alpha)))
    df = x.shape[1]

    return BacktestResult(
        name=f"Dynamic quantile (K={lags})",
        statistic=statistic,
        p_value=_chi2_sf(statistic, df),
        df=df,
        alpha=alpha,
        n_obs=n,
        n_violations=int(hits.sum()),
    )


def expected_shortfall_backtest(returns, var, es, alpha: float) -> BacktestResult:
    r"""McNeil & Frey (2000) exceedance-residual test for expected shortfall.

    On violation days, the standardised residual :math:`r_t - \mathrm{ES}_t`
    should have zero mean. A one-sided t-test on those residuals asks whether
    losses on bad days are systematically worse than the model said.

    Reported with the number of exceedances, because with 1% VaR on two years of
    daily data there may be only five of them and the test has almost no power.
    """
    r, e = _as_numpy(returns), _as_numpy(es)
    hits = violations(returns, var)
    n_hits = int(hits.sum())
    if n_hits < 2:
        return BacktestResult(
            name="McNeil-Frey ES exceedance residual",
            statistic=float("nan"),
            p_value=float("nan"),
            df=0,
            alpha=alpha,
            n_obs=r.size,
            n_violations=n_hits,
            extra={"note": float(n_hits)},
        )

    residuals = r[hits] - e[hits]
    mean = float(residuals.mean())
    sd = float(residuals.std(ddof=1))
    statistic = mean / (sd / math.sqrt(n_hits)) if sd > 0 else float("nan")

    try:
        from scipy.stats import t as _t

        p_value = float(_t.cdf(statistic, df=n_hits - 1))
    except ImportError:  # pragma: no cover
        p_value = float("nan")

    return BacktestResult(
        name="McNeil-Frey ES exceedance residual",
        statistic=statistic,
        p_value=p_value,
        df=n_hits - 1,
        alpha=alpha,
        n_obs=r.size,
        n_violations=n_hits,
        extra={"mean_residual": mean, "sd_residual": sd},
    )


def var_backtest_table(returns, var, alpha: float, es=None, dq_lags: int = 4):
    """Run the full battery and return it as a tidy :class:`pandas.DataFrame`.

    Examples
    --------
    >>> import numpy as np, torch
    >>> rng = np.random.default_rng(0)
    >>> r = torch.tensor(rng.standard_normal(2000))
    >>> v = torch.full((2000,), -2.3263478740408408)   # exact 1% normal VaR
    >>> table = var_backtest_table(r, v, alpha=0.01)
    >>> bool((table["p_value"] > 0.05).all())          # a correct model passes
    True
    """
    import pandas as pd

    tests = [
        kupiec_pof(returns, var, alpha),
        christoffersen_independence(returns, var, alpha),
        christoffersen_conditional_coverage(returns, var, alpha),
        dynamic_quantile_test(returns, var, alpha, lags=dq_lags),
    ]
    if es is not None:
        tests.append(expected_shortfall_backtest(returns, var, es, alpha))

    return pd.DataFrame(
        [
            {
                "test": t.name,
                "statistic": t.statistic,
                "p_value": t.p_value,
                "df": t.df,
                "n_obs": t.n_obs,
                "n_violations": t.n_violations,
                "violation_rate": t.n_violations / t.n_obs if t.n_obs else float("nan"),
                "expected_rate": t.alpha,
                "reject_5pct": t.rejected,
            }
            for t in tests
        ]
    )
