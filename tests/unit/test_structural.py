"""Zhao et al. (2024) and Wei et al. (2025): the structural neural-GARCH models.

The tests here check the properties that make each design *correct*, not merely
that the code runs:

* Zhao's GARCH-LSTM must collapse **exactly** to its GARCH kernel at ``w = 0``.
  That is the design's central claim; if it fails, the neural part is not a
  modulation of a valid variance but a replacement for it.
* The FI-GARCH kernel's :math:`\\lambda_0` must be exactly zero, or
  :math:`\\epsilon_t` enters its own conditional variance and the model is
  looking ahead.
* Wei's econometric parameters must remain interpretable after joint training,
  which is what the paper repeatedly claims for the design.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from garchai.core import ModelConfig, TrainConfig, set_seed
from garchai.distributions import Normal
from garchai.structural import (
    FIGARCHNNKernel,
    GARCHNN,
    GARCHNNKernel,
    GJRNNKernel,
    EmbeddedGARCHGRU,
    EmbeddedGARCHLSTM,
    StructuralGARCHLSTM,
    figarch_weights,
    get_kernel,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture
def returns() -> torch.Tensor:
    set_seed(0)
    return torch.randn(300) * 0.9


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
def test_garch_kernel_matches_the_recursion_by_hand():
    kernel = GARCHNNKernel(0.05, 0.10, 0.85)
    out = kernel(torch.tensor([2.0]), torch.tensor(1.0)).detach()
    assert float(out) == pytest.approx(0.05 + 0.10 * 4.0 + 0.85 * 1.0)


def test_gjr_kernel_treats_the_sign_asymmetrically():
    kernel = GJRNNKernel(0.05, 0.05, 0.20, 0.70)
    up = float(kernel(torch.tensor([2.0]), torch.tensor(1.0)).detach())
    down = float(kernel(torch.tensor([-2.0]), torch.tensor(1.0)).detach())
    assert down > up
    assert down - up == pytest.approx(0.20 * 4.0)


def test_kernels_expose_their_stylized_fact():
    assert GARCHNNKernel().stylized_fact == "volatility clustering"
    assert GJRNNKernel().stylized_fact == "asymmetry (leverage)"
    assert FIGARCHNNKernel().stylized_fact == "long memory"


def test_kernel_registry():
    assert isinstance(get_kernel("gjr"), GJRNNKernel)
    assert isinstance(get_kernel("FI-GARCH".lower().replace("-", "")), FIGARCHNNKernel)
    with pytest.raises(KeyError, match="Unknown kernel"):
        get_kernel("aparch")


# --------------------------------------------------------------------------- #
# FI-GARCH expansion
# --------------------------------------------------------------------------- #
def test_figarch_lambda_zero_is_exactly_zero():
    """Non-negotiable: a non-zero lambda_0 would be look-ahead."""
    for phi, d, beta in [(0.2, 0.4, 0.6), (0.0, 0.3, 0.5), (0.5, 0.8, 0.2)]:
        weights = figarch_weights(
            torch.tensor(phi), torch.tensor(d), torch.tensor(beta), truncation=50
        )
        assert float(weights[0]) == pytest.approx(0.0, abs=1e-14)


def test_figarch_first_weight_matches_the_closed_form():
    """lambda_1 = d + phi - beta, from the first term of the expansion."""
    phi, d, beta = 0.2, 0.4, 0.6
    weights = figarch_weights(
        torch.tensor(phi), torch.tensor(d), torch.tensor(beta), truncation=20
    )
    assert float(weights[1]) == pytest.approx(d + phi - beta, abs=1e-12)


def test_figarch_decay_is_hyperbolic_not_exponential():
    """Long memory means power-law decay; that is the whole point of FI-GARCH."""
    weights = figarch_weights(
        torch.tensor(0.2), torch.tensor(0.4), torch.tensor(0.6), truncation=200
    ).detach()
    ratio = float(weights[100] / weights[20])
    exponential = 0.6 ** 80  # what a GARCH beta would give over the same span
    assert ratio > 1e3 * exponential
    assert 0.0 < ratio < 1.0


def test_figarch_rejects_a_degenerate_truncation():
    with pytest.raises(ValueError, match="at least 2"):
        figarch_weights(torch.tensor(0.2), torch.tensor(0.4), torch.tensor(0.6), truncation=1)


def test_figarch_kernel_produces_positive_finite_variance(returns):
    kernel = FIGARCHNNKernel(truncation=40)
    out = kernel(returns[:40], torch.tensor(1.0))
    value = float(out.detach())
    assert value > 0 and np.isfinite(value)


def test_figarch_weights_are_differentiable():
    kernel = FIGARCHNNKernel(truncation=30)
    kernel.weights().sum().backward()
    assert kernel.raw_d.grad is not None and torch.isfinite(kernel.raw_d.grad)


# --------------------------------------------------------------------------- #
# Zhao et al. GARCH-LSTM
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", ["garch", "gjr"])
def test_garch_lstm_collapses_to_its_kernel_at_zero_coupling(returns, kernel):
    """Equation 7's defining property: w = 0 switches the LSTM off entirely."""
    model = StructuralGARCHLSTM(kernel=kernel, w=0.0)
    embedded, _, _ = model.filter(returns)
    plain, _, _ = GARCHNN(kernel=model.kernel).filter(returns)
    assert torch.allclose(embedded, plain, atol=1e-14)


def test_garch_lstm_coupling_actually_does_something(returns):
    model = StructuralGARCHLSTM(kernel="gjr", w=0.5)
    with_lstm, _, _ = model.filter(returns)
    without, _, _ = StructuralGARCHLSTM(kernel=model.kernel, w=0.0).filter(returns)
    assert float((with_lstm - without).abs().max()) > 1e-3


def test_garch_lstm_coupling_is_bounded(returns):
    """|w| < 1 guarantees 1 + w*tanh(c) > 0, hence a positive variance."""
    model = StructuralGARCHLSTM(kernel="gjr", w=0.9)
    with torch.no_grad():
        model.raw_w.fill_(50.0)  # push the raw parameter to an extreme
    assert abs(float(model.coupling.detach())) < 1.0
    h, _, _ = model.filter(returns)
    assert bool((h > 0).all())


def test_garch_lstm_variance_stays_positive_under_large_shocks():
    model = StructuralGARCHLSTM(kernel="gjr", w=0.8)
    shocked = torch.cat([torch.randn(50), torch.tensor([30.0, -30.0, 0.0])])
    h, _, _ = model.filter(shocked)
    assert bool((h > 0).all()) and bool(torch.isfinite(h).all())


def test_garch_lstm_gradients_reach_both_components(returns):
    model = StructuralGARCHLSTM(kernel="gjr", w=0.3)
    h, eps, _ = model.filter(returns)
    Normal().nll(eps, h).backward()
    assert torch.isfinite(model.raw_w.grad)
    assert torch.isfinite(model.W_f.grad).all()
    assert torch.isfinite(model.kernel.raw_alpha.grad)  # econometric side too


@pytest.mark.slow
def test_garch_nn_recovers_a_known_garch_process():
    """The paper's Table 2 claim: backprop estimates GARCH parameters correctly."""
    rng = np.random.default_rng(0)
    n, omega, alpha, beta = 3000, 0.05, 0.09, 0.88
    h = np.zeros(n)
    e = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()

    set_seed(0)
    model = GARCHNN(kernel="garch", config=ModelConfig(mean="zero", scale=None))
    result = model.fit(e, TrainConfig(lr=0.08, max_epochs=250, patience=40))
    assert result.parameters["alpha"] == pytest.approx(alpha, abs=0.05)
    assert result.parameters["beta"] == pytest.approx(beta, abs=0.06)


def test_zhao_models_refuse_an_analytic_forecast(returns):
    model = GARCHNN(kernel="garch", config=ModelConfig(mean="zero", scale=None))
    model.fit(returns.numpy(), TrainConfig(max_epochs=5, patience=3))
    with pytest.raises(ValueError, match="no closed-form"):
        model.forecast(horizon=3, method="analytic")


@pytest.mark.parametrize("factory", [
    lambda cfg: GARCHNN(kernel="garch", config=cfg),
    lambda cfg: GARCHNN(kernel="figarch", config=cfg),
    lambda cfg: StructuralGARCHLSTM(kernel="gjr", hidden_size=3, config=cfg),
])
def test_zhao_forecast_one_step_is_exact_and_paths_are_valid(returns, factory):
    """forecast() used to call cell.initial_state, which a kernel does not have."""
    model = factory(ModelConfig(mean="zero", scale=None))
    model.fit(returns.numpy(), TrainConfig(max_epochs=5, patience=3))
    one = model.forecast(horizon=1)
    with torch.no_grad():
        exact = float(model._one_step_variance(model._fitted_returns))
    assert one.variance[0] == pytest.approx(exact, rel=1e-10)

    path = model.forecast(horizon=6, n_simulations=500, seed=0)
    assert path.method == "simulation"
    assert path.variance[0] == pytest.approx(exact, rel=1e-10)
    assert np.all(np.isfinite(path.variance)) and np.all(path.variance > 0)
    assert path.simulations.shape == (500, 6)


def test_garch_nn_simulated_forecast_matches_the_closed_form(returns):
    """For a GARCH(1,1) kernel the Monte Carlo mean must reproduce the analytic decay."""
    model = GARCHNN(kernel="garch", config=ModelConfig(mean="zero", scale=None))
    model.fit(returns.numpy(), TrainConfig(max_epochs=20, patience=5))
    fc = model.forecast(horizon=8, n_simulations=40_000, seed=1)
    p = {k: float(v) for k, v in model.kernel.transformed_parameters().items()}
    persistence = p["alpha"] + p["beta"]
    long_run = p["omega"] / (1.0 - persistence)
    closed = long_run + persistence ** np.arange(8) * (fc.variance[0] - long_run)
    np.testing.assert_allclose(fc.variance, closed, rtol=0.02)


def test_zhao_rejects_unsupported_distributions():
    model = GARCHNN(config=ModelConfig(distribution="skew-t"))
    with pytest.raises(ValueError, match="supports"):
        model.distribution()


# --------------------------------------------------------------------------- #
# Wei et al. embedded models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [EmbeddedGARCHGRU, EmbeddedGARCHLSTM])
def test_embedded_models_produce_valid_variance(returns, factory):
    model = factory(hidden_size=8, config=ModelConfig(mean="constant"))
    h, _, _ = model.filter(returns)
    assert h.shape == returns.shape
    assert bool((h > 0).all()) and bool(torch.isfinite(h).all())


@pytest.mark.parametrize("factory", [EmbeddedGARCHGRU, EmbeddedGARCHLSTM])
def test_embedded_gradients_reach_the_econometric_parameters(returns, factory):
    """The joint-optimisation claim: GARCH parameters must receive gradient."""
    model = factory(hidden_size=6, config=ModelConfig(mean="constant"))
    h, eps, _ = model.filter(returns)
    Normal().nll(eps, h).backward()
    garch = model.cell.garch
    for name in ("raw_omega", "raw_alpha", "raw_beta", "W_g"):
        grad = getattr(garch, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name


@pytest.mark.parametrize("factory", [EmbeddedGARCHGRU, EmbeddedGARCHLSTM])
def test_embedded_econometric_parameters_stay_valid(returns, factory):
    model = factory(hidden_size=6, config=ModelConfig(mean="constant"))
    report = model.persistence_report()
    assert report["omega"] > 0
    assert 0.0 <= report["alpha"] and 0.0 <= report["beta"]
    assert report["persistence"] < 1.0


def test_gru_gate_conventions_differ(returns):
    """Eq. 19 as printed is not the textbook GRU; both must be available."""
    set_seed(0)
    paper = EmbeddedGARCHGRU(hidden_size=6, gate_convention="paper")
    set_seed(0)
    standard = EmbeddedGARCHGRU(hidden_size=6, gate_convention="standard")
    a, _, _ = paper.filter(returns)
    b, _, _ = standard.filter(returns)
    assert float((a - b).abs().max()) > 1e-8


def test_embedded_models_accept_exogenous_inputs(returns):
    exog = torch.randn(returns.numel(), 2)
    model = EmbeddedGARCHGRU(hidden_size=6, n_exog=2, config=ModelConfig(mean="constant"))
    h, _, _ = model.filter(returns, exog=exog)
    assert bool((h > 0).all())


@pytest.mark.slow
def test_embedded_models_train_and_improve_the_likelihood(returns):
    set_seed(0)
    model = EmbeddedGARCHGRU(hidden_size=6, config=ModelConfig(mean="constant", scale=None))
    with torch.no_grad():
        before = float(model.negative_loglikelihood(returns))
    result = model.fit(returns.numpy(), TrainConfig(lr=0.05, max_epochs=40, patience=15))
    assert -result.loglikelihood < before


@pytest.mark.parametrize("factory", [EmbeddedGARCHGRU, EmbeddedGARCHLSTM])
def test_embedded_models_reject_unsupported_distributions(factory):
    model = factory(hidden_size=4, config=ModelConfig(distribution="ged"))
    with pytest.raises(ValueError, match="normal|student-t"):
        model.distribution()


def test_provenance_records_the_gate_deviation():
    text = " ".join(EmbeddedGARCHGRU.provenance.deviations)
    assert "Eq. 19" in text
    assert EmbeddedGARCHGRU.provenance.paper_id == "wei2025_garch_rnn"
    assert StructuralGARCHLSTM.provenance.paper_id == "zhao2024_garch_nn"
