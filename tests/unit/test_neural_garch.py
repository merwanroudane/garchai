"""Yin & Barucca (2022): time-varying coefficients and the neural BEKK.

The property that matters most here is structural, not statistical: because the
BEKK quadratic form is preserved, :math:`\\Sigma_t` must be symmetric positive
definite at *every* step for *any* network output. If that ever fails, the
model has stopped being a covariance model, and no amount of good forecasting
accuracy would redeem it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from garchai.core import ModelConfig, TrainConfig, set_seed
from garchai.structural import NeuralBEKK, NeuralGARCH
from garchai.structural.neural_garch import _gaussian_kl

torch.set_default_dtype(torch.float64)


@pytest.fixture
def returns() -> torch.Tensor:
    set_seed(0)
    return torch.randn(200) * 0.9


@pytest.fixture
def panel() -> torch.Tensor:
    set_seed(0)
    return torch.randn(120, 3) * 0.8


# --------------------------------------------------------------------------- #
# KL divergence
# --------------------------------------------------------------------------- #
def test_kl_is_zero_between_identical_gaussians():
    mu = torch.tensor([0.3, -1.2])
    var = torch.tensor([0.5, 2.0])
    assert float(_gaussian_kl(mu, var, mu, var)) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_non_negative_and_grows_with_separation():
    mu_p, var = torch.tensor([0.0]), torch.tensor([1.0])
    values = [float(_gaussian_kl(torch.tensor([m]), var, mu_p, var)) for m in (0.5, 1.0, 2.0)]
    assert all(v > 0 for v in values)
    assert values[0] < values[1] < values[2]


def test_kl_matches_the_closed_form_for_a_known_pair():
    """Univariate: KL = log(sp/sq) + (sq^2 + (mq-mp)^2)/(2 sp^2) - 1/2."""
    mu_q, var_q = torch.tensor([1.0]), torch.tensor([0.25])
    mu_p, var_p = torch.tensor([0.0]), torch.tensor([1.0])
    expected = np.log(1.0 / 0.5) + (0.25 + 1.0) / 2.0 - 0.5
    assert float(_gaussian_kl(mu_q, var_q, mu_p, var_p)) == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# univariate Neural GARCH
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("distribution", ["normal", "student-t"])
def test_filter_produces_valid_variance(returns, distribution):
    set_seed(0)
    model = NeuralGARCH(
        hidden_size=8, config=ModelConfig(mean="zero", distribution=distribution)
    )
    h, eps, _ = model.filter(returns)
    assert h.shape == returns.shape
    assert bool((h > 0).all()) and bool(torch.isfinite(h).all())


def test_coefficients_are_time_varying(returns):
    """The whole point: omega, alpha and beta must actually move."""
    set_seed(0)
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    model.train()
    model.filter(returns)
    paths = model._last_paths
    for name in ("omega", "alpha", "beta"):
        assert paths[name].std() > 0, f"{name} is constant"


def test_initialisation_starts_in_a_sane_econometric_region(returns):
    """Guards the bias initialisation: a bad start opened the ELBO at 1e7."""
    set_seed(0)
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    model.train()
    model.filter(returns)
    paths = model._last_paths
    assert paths["beta"].mean() > 0.5, "beta should start high, as GARCH betas are"
    assert paths["alpha"].mean() < 0.4, "alpha should start small"
    assert float(model.negative_loglikelihood(returns).detach()) < 1e5


def test_coefficients_are_never_negative(returns):
    """A negative alpha or beta would make the 'variance' negative."""
    set_seed(0)
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    model.train()
    model.filter(returns)
    for name in ("omega", "alpha", "beta"):
        assert (model._last_paths[name] >= 0).all()


def test_student_t_degrees_of_freedom_stay_above_two(returns):
    """Below 2 the variance of a standardised t does not exist."""
    set_seed(0)
    model = NeuralGARCH(
        hidden_size=8, config=ModelConfig(mean="zero", distribution="student-t")
    )
    model.train()
    model.filter(returns)
    assert (model._last_paths["nu"] > 2.0).all()


def test_elbo_is_finite_and_gradients_flow(returns):
    set_seed(0)
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    model.train()
    loss = model.negative_loglikelihood(returns)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


def test_kl_term_is_actually_included(returns):
    """Setting kl_weight to zero must change the objective."""
    set_seed(0)
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    model.train()
    with_kl = float(model.negative_loglikelihood(returns).detach())
    model.kl_weight = 0.0
    without_kl = float(model.negative_loglikelihood(returns).detach())
    assert with_kl != pytest.approx(without_kl)


def test_enforce_stationarity_bounds_the_persistence(returns):
    set_seed(0)
    model = NeuralGARCH(
        hidden_size=8, config=ModelConfig(mean="zero"), enforce_stationarity=True
    )
    model.train()
    model.filter(returns)
    paths = model._last_paths
    assert (paths["alpha"] + paths["beta"] < 1.0).all()


def test_evaluation_mode_is_deterministic(returns):
    """In eval mode the posterior mean is used, so two passes must agree."""
    set_seed(0)
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    model.eval()
    first, _, _ = model.filter(returns)
    second, _, _ = model.filter(returns)
    assert torch.allclose(first, second, atol=1e-14)


def test_training_mode_samples(returns):
    """In training mode the reparameterised draw makes passes differ."""
    set_seed(0)
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    model.train()
    first, _, _ = model.filter(returns)
    second, _, _ = model.filter(returns)
    assert not torch.allclose(first, second, atol=1e-10)


def test_forecast_path_returns_a_positive_scalar(returns):
    set_seed(0)
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    forecast = model.forecast_path(returns, horizon=1)
    assert float(forecast) > 0 and np.isfinite(float(forecast))


def test_multi_step_forecast_is_refused_rather_than_faked(returns):
    model = NeuralGARCH(hidden_size=8, config=ModelConfig(mean="zero"))
    with pytest.raises(NotImplementedError, match="one-step-ahead"):
        model.forecast_path(returns, horizon=5)


def test_rejects_unsupported_distribution():
    with pytest.raises(ValueError, match="normal|student-t"):
        NeuralGARCH(config=ModelConfig(distribution="ged"))


def test_provenance_records_the_likelihood_typo():
    text = " ".join(NeuralGARCH.provenance.deviations)
    assert "logGamma" in text
    assert "stuid-maker" in text


# --------------------------------------------------------------------------- #
# multivariate Neural BEKK
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_assets", [2, 3, 5])
def test_covariance_is_positive_definite_at_every_step(n_assets):
    """The structural guarantee. Must hold for any network output."""
    set_seed(0)
    model = NeuralBEKK(n_assets=n_assets, hidden_size=8)
    model.train()
    panel = torch.randn(80, n_assets) * 0.8
    covariance = model.filter_covariance(panel)
    assert covariance.shape == (80, n_assets, n_assets)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    assert bool((eigenvalues > 0).all())


@pytest.mark.parametrize("n_assets", [2, 4])
def test_covariance_is_symmetric(n_assets):
    set_seed(0)
    model = NeuralBEKK(n_assets=n_assets, hidden_size=8)
    panel = torch.randn(60, n_assets) * 0.8
    covariance = model.filter_covariance(panel)
    asymmetry = float((covariance - covariance.transpose(-1, -2)).abs().max().detach())
    assert asymmetry < 1e-12


def test_positive_definiteness_survives_extreme_inputs():
    """Even a crisis-sized shock must not break the quadratic form."""
    set_seed(0)
    model = NeuralBEKK(n_assets=3, hidden_size=8)
    model.train()
    panel = torch.randn(60, 3) * 0.8
    panel[30] = torch.tensor([50.0, -50.0, 40.0])
    covariance = model.filter_covariance(panel)
    assert bool((torch.linalg.eigvalsh(covariance) > 0).all())
    assert bool(torch.isfinite(covariance).all())


def test_coefficient_vector_has_the_papers_dimension():
    """2n + n(n+1)/2: the diagonals of A and B, and the upper triangle of Omega."""
    for n in (2, 3, 5):
        model = NeuralBEKK(n_assets=n, hidden_size=4)
        assert model.n_coefficients == 2 * n + n * (n + 1) // 2


def test_bekk_elbo_and_gradients(panel):
    set_seed(0)
    model = NeuralBEKK(n_assets=3, hidden_size=8)
    model.train()
    loss = model.negative_loglikelihood(panel)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_bekk_requires_at_least_two_assets():
    with pytest.raises(ValueError, match="at least 2 assets"):
        NeuralBEKK(n_assets=1)


def test_bekk_rejects_a_mismatched_panel():
    model = NeuralBEKK(n_assets=3, hidden_size=4)
    with pytest.raises(ValueError, match=r"shape \(T, 3\)"):
        model.filter_covariance(torch.randn(50, 4))


def test_bekk_fit_and_correlation_output(panel):
    set_seed(0)
    model = NeuralBEKK(n_assets=3, hidden_size=6)
    result = model.fit(panel, TrainConfig(lr=0.02, max_epochs=6, patience=5))
    assert np.isfinite(result.loglikelihood)
    assert "ELBO" in result.message

    correlation = model.correlation_()
    assert correlation.shape == (panel.shape[0], 3, 3)
    diagonal = np.diagonal(correlation, axis1=-2, axis2=-1)
    assert np.allclose(diagonal, 1.0, atol=1e-10)
    assert np.all(np.abs(correlation) <= 1.0 + 1e-10)


def test_unfitted_bekk_refuses_to_report_covariance():
    model = NeuralBEKK(n_assets=2, hidden_size=4)
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.covariance_
