"""End-to-end estimation, forecasting and risk output on real market data.

These use the bundled S&P 500 series rather than simulated returns, both because
the build requirements call for it and because real data exercises paths that
Gaussian simulation never does: fat tails, volatility clustering that actually
persists, and the 2008 and 2020 crises.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from garchai import EGARCH, GARCH, GJRGARCH
from garchai.core import ModelConfig, TrainConfig
from garchai.datasets import load_returns
from garchai.risk import var_backtest_table, violation_ratio

torch.set_default_dtype(torch.float64)

FAST = TrainConfig(lr=0.1, max_epochs=120, patience=25, refine_with_lbfgs=True)


@pytest.fixture(scope="module")
def sp500() -> np.ndarray:
    """Daily S&P 500 log returns in percent, 2010-2019."""
    return load_returns("sp500", start="2010-01-01", end="2019-12-31").to_numpy()


@pytest.fixture(scope="module")
def fitted(sp500):
    model = GARCH(config=ModelConfig(mean="constant", distribution="normal", scale=None))
    result = model.fit(sp500, FAST)
    return model, result


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #
def test_fit_produces_a_sane_variance_path(fitted, sp500):
    _, res = fitted
    assert res.conditional_variance.shape == sp500.shape
    assert np.all(res.conditional_variance > 0)
    assert np.all(np.isfinite(res.conditional_variance))
    assert np.isfinite(res.loglikelihood)


def test_estimated_parameters_are_econometrically_valid(fitted):
    _, res = fitted
    assert res.parameters["omega"] > 0
    assert res.parameters["alpha"] >= 0
    assert res.parameters["beta"] >= 0
    assert 0.0 < res.persistence < 1.0


def test_equity_returns_show_high_persistence(fitted):
    """A decade of daily equity data should give persistence around 0.95-0.999."""
    _, res = fitted
    assert 0.90 < res.persistence < 1.0


def test_standardized_residuals_have_roughly_unit_variance(fitted):
    _, res = fitted
    z = res.standardized_residuals
    assert abs(float(np.std(z)) - 1.0) < 0.15
    assert abs(float(np.mean(z))) < 0.15


def test_information_criteria_are_consistent(fitted):
    _, res = fitted
    assert res.aic == pytest.approx(2 * res.n_params - 2 * res.loglikelihood)
    assert res.bic == pytest.approx(
        res.n_params * np.log(res.n_obs) - 2 * res.loglikelihood
    )
    assert res.bic > res.aic  # log(n) > 2 for any realistic sample


def test_summary_reports_the_key_quantities(fitted):
    model, _ = fitted
    text = model.summary()
    for expected in ("omega", "alpha", "beta", "Persistence", "Log-likelihood", "Provenance"):
        assert expected in text


def test_result_converts_to_a_dataframe(fitted):
    _, res = fitted
    frame = res.to_frame()
    assert list(frame.columns) == [
        "conditional_mean", "residual", "conditional_variance",
        "conditional_volatility", "standardized_residual",
    ]


def test_unfitted_model_refuses_to_report_results():
    model = GARCH()
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.conditional_variance_


def test_scaling_is_undone_in_the_reported_results(sp500):
    """Fitting on a rescaled series must not change what the user sees."""
    plain = GARCH(config=ModelConfig(mean="constant", scale=None)).fit(sp500, FAST)
    scaled = GARCH(config=ModelConfig(mean="constant", scale=10.0)).fit(sp500, FAST)
    assert scaled.scale == 10.0
    # The variance path is on the original scale in both cases.
    ratio = scaled.conditional_variance.mean() / plain.conditional_variance.mean()
    assert 0.7 < ratio < 1.4


def test_non_finite_input_is_rejected_with_a_clear_message(sp500):
    dirty = sp500.copy()
    dirty[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        GARCH().fit(dirty, FAST)


@pytest.mark.parametrize("distribution", ["normal", "student-t", "ged"])
def test_alternative_distributions_fit(sp500, distribution):
    model = GARCH(config=ModelConfig(mean="constant", distribution=distribution, scale=None))
    res = model.fit(sp500[:1200], FAST)
    assert np.isfinite(res.loglikelihood)
    if distribution != "normal":
        assert res.distribution_parameters


def test_student_t_beats_the_normal_on_equity_returns(sp500):
    """Daily equity returns are leptokurtic, so the t should fit better."""
    normal = GARCH(config=ModelConfig(mean="constant", distribution="normal",
                                      scale=None)).fit(sp500, FAST)
    student = GARCH(config=ModelConfig(mean="constant", distribution="student-t",
                                       scale=None)).fit(sp500, FAST)
    assert student.loglikelihood > normal.loglikelihood


def test_gjr_finds_a_positive_leverage_effect(sp500):
    """Negative shocks raise equity volatility more than positive ones."""
    res = GJRGARCH(config=ModelConfig(mean="constant", scale=None)).fit(sp500, FAST)
    assert res.parameters["gamma"] > 0.0


# --------------------------------------------------------------------------- #
# forecasting
# --------------------------------------------------------------------------- #
def test_analytic_forecast_decays_toward_the_long_run_variance(fitted):
    model, res = fitted
    forecast = model.forecast(horizon=250, method="analytic")
    assert forecast.method == "analytic"
    assert forecast.variance.shape == (250,)
    assert np.all(forecast.variance > 0)
    # far-horizon forecast converges on the unconditional variance
    assert forecast.variance[-1] == pytest.approx(res.long_run_variance, rel=0.05)


def test_forecast_is_monotone_between_the_two_ends(fitted):
    model, res = fitted
    forecast = model.forecast(horizon=100, method="analytic")
    gaps = np.abs(forecast.variance - res.long_run_variance)
    assert np.all(np.diff(gaps) <= 1e-12)  # gap shrinks every step


def test_one_step_forecast_continues_the_filtered_path(fitted):
    model, _ = fitted
    forecast = model.forecast(horizon=1)
    assert forecast.horizon == 1
    assert np.isfinite(forecast.variance[0]) and forecast.variance[0] > 0


def test_egarch_refuses_an_analytic_forecast(sp500):
    """It has no closed form, and garchai will not fake one."""
    model = EGARCH(config=ModelConfig(mean="constant", scale=None))
    model.fit(sp500[:800], FAST)
    with pytest.raises(ValueError, match="no closed-form"):
        model.forecast(horizon=5, method="analytic")


@pytest.mark.slow
def test_egarch_auto_forecast_falls_back_to_simulation(sp500):
    model = EGARCH(config=ModelConfig(mean="constant", scale=None))
    model.fit(sp500[:800], FAST)
    forecast = model.forecast(horizon=3, method="auto", n_simulations=200, seed=0)
    assert forecast.method == "simulation"
    assert forecast.simulations is not None
    assert forecast.simulations.shape == (200, 3)


def test_forecast_requires_a_fitted_model():
    with pytest.raises(RuntimeError, match="not been fitted"):
        GARCH().forecast(horizon=1)


def test_forecast_rejects_a_zero_horizon(fitted):
    model, _ = fitted
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        model.forecast(horizon=0)


def test_forecast_frame_and_risk_helpers(fitted):
    model, _ = fitted
    forecast = model.forecast(horizon=5)
    frame = forecast.to_frame()
    assert list(frame.index) == [1, 2, 3, 4, 5]
    var = forecast.var(0.05, model.distribution())
    es = forecast.es(0.05, model.distribution())
    assert np.all(es < var)  # expected shortfall is always deeper in the tail


# --------------------------------------------------------------------------- #
# risk output
# --------------------------------------------------------------------------- #
def test_gaussian_var_undercovers_the_far_tail(fitted, sp500):
    """A documented failure mode, asserted rather than hidden.

    Equity returns are leptokurtic, so a Gaussian innovation puts too little
    mass in the far tail and the 1% VaR is breached far more than 1% of the
    time. This is exactly why the VaR literature reaches for the Student-t, and
    it is a property of the model, not a defect in the implementation. The 5%
    level, which sits well inside the tail, is close to nominal.
    """
    model, _ = fitted
    far_tail = violation_ratio(sp500, model.var(0.01), 0.01)
    near_tail = violation_ratio(sp500, model.var(0.05), 0.05)
    assert far_tail > 1.5, "Gaussian GARCH should over-violate the 1% VaR"
    assert 0.7 < near_tail < 1.5, f"5% violation ratio {near_tail:.2f} is implausible"
    assert far_tail < 4.0, f"1% violation ratio {far_tail:.2f} signals a bug, not fat tails"


def test_student_t_improves_far_tail_coverage(sp500):
    """The fix for the above: a fatter-tailed innovation covers the 1% tail better."""
    normal = GARCH(config=ModelConfig(mean="constant", distribution="normal", scale=None))
    student = GARCH(config=ModelConfig(mean="constant", distribution="student-t", scale=None))
    normal.fit(sp500, FAST)
    student.fit(sp500, FAST)

    normal_error = abs(violation_ratio(sp500, normal.var(0.01), 0.01) - 1.0)
    student_error = abs(violation_ratio(sp500, student.var(0.01), 0.01) - 1.0)
    assert student_error < normal_error


def test_es_is_always_more_extreme_than_var(fitted):
    model, _ = fitted
    for alpha in (0.01, 0.05):
        assert np.all(model.es(alpha) < model.var(alpha))


def test_var_backtest_table_runs_on_a_fitted_model(fitted, sp500):
    model, _ = fitted
    table = var_backtest_table(sp500, model.var(0.05), alpha=0.05, es=model.es(0.05))
    assert len(table) == 5
    assert set(table.columns) >= {"test", "statistic", "p_value", "reject_5pct"}
