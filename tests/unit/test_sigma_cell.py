"""The σ-LSTM cell (2022) and the σ-Cell family (2023).

Rodikov & Antulov-Fantulin's construction is a GARCH recursion with a wider,
non-linear hidden state. Two things are checked here beyond "it runs":

* the Adjusted Softplus really has the three properties Eq. 41 claims, since
  the whole cell's scaling rests on ``AS(1) = 1``;
* the paper's own parameter count (41 at ``hidden_size=10``) comes out exactly,
  which is the cheapest possible check that the weight *shapes* were read
  correctly — a matrix where the paper meant a vector would sail past every
  other test while quadrupling the model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from garchai.core import ModelConfig, TrainConfig, set_seed
from garchai.structural import (
    SigmaCell,
    SigmaCellN,
    SigmaCellNTV,
    SigmaCellRL,
    SigmaCellRLTV,
    SigmaLSTM,
    adjusted_softplus,
    sigma_cell_loss,
)

torch.set_default_dtype(torch.float64)

ALL_VARIANTS = [SigmaCell, SigmaCellN, SigmaCellRL, SigmaCellNTV, SigmaCellRLTV, SigmaLSTM]


@pytest.fixture
def returns() -> torch.Tensor:
    set_seed(0)
    return torch.randn(400) * 0.9


# --------------------------------------------------------------------------- #
# Adjusted Softplus (Eq. 41)
# --------------------------------------------------------------------------- #
def test_adjusted_softplus_is_zero_at_zero():
    assert float(adjusted_softplus(torch.tensor(0.0))) == pytest.approx(0.0, abs=1e-15)


def test_adjusted_softplus_is_one_at_one():
    """The normalisation that puts the cell on a GARCH-coefficient scale."""
    assert float(adjusted_softplus(torch.tensor(1.0))) == pytest.approx(1.0, abs=1e-12)


def test_adjusted_softplus_is_non_negative_everywhere():
    x = torch.linspace(-50.0, 50.0, 10_001)
    assert bool((adjusted_softplus(x) >= 0.0).all())


def test_adjusted_softplus_is_monotone():
    x = torch.linspace(-10.0, 10.0, 2001)
    assert bool((torch.diff(adjusted_softplus(x)) >= -1e-15).all())


def test_adjusted_softplus_matches_its_closed_form():
    beta = 1.5
    x = torch.tensor([-2.0, 0.3, 2.0])
    shift = math.log(2.0) / beta
    scale = math.log(1.0 + math.exp(beta)) / beta - shift
    expected = torch.clamp(
        torch.log(1.0 + torch.exp(beta * x)) / beta - shift, min=0.0
    ) / scale
    assert torch.allclose(adjusted_softplus(x, beta), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# the loss (Eq. 39)
# --------------------------------------------------------------------------- #
def test_sigma_cell_loss_matches_its_formula():
    x = torch.tensor([1.0, -2.0])
    variance = torch.tensor([1.0, 4.0])
    expected = (math.log(1.0) + 1.0) + (math.log(4.0) + 4.0 / 4.0)
    assert float(sigma_cell_loss(x, variance)) == pytest.approx(expected, abs=1e-12)


def test_sigma_cell_loss_is_twice_the_gaussian_nll_up_to_a_constant():
    """Eq. 39 and the Gaussian NLL rank models identically."""
    from garchai.losses import gaussian_nll

    x = torch.randn(50)
    variance = torch.rand(50) + 0.5
    gap = float(sigma_cell_loss(x, variance)) - 2.0 * float(gaussian_nll(x, variance))
    assert gap == pytest.approx(-50 * math.log(2 * math.pi), abs=1e-9)


def test_sigma_cell_loss_uses_the_learned_mean_when_given():
    x = torch.tensor([1.0, 1.0])
    variance = torch.tensor([1.0, 1.0])
    mean = torch.tensor([1.0, 1.0])
    assert float(sigma_cell_loss(x, variance, mean)) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def test_parameter_count_matches_the_paper():
    """41 parameters at hidden_size=10 (paper sec. 3.1)."""
    assert sum(p.numel() for p in SigmaCell(hidden_size=10).parameters()) == 41


def test_time_varying_variants_freeze_the_replaced_weights():
    """Eqs. 24-25 replace W_s and W_r, so the fixed ones must not be optimised."""
    for factory in (SigmaCellNTV, SigmaCellRLTV):
        model = factory(hidden_size=10)
        assert not model.W_s.requires_grad
        assert not model.W_r.requires_grad


@pytest.mark.parametrize("factory", ALL_VARIANTS)
def test_variance_is_positive_and_finite(factory, returns):
    set_seed(0)
    model = factory(hidden_size=10)
    model.train()
    h, _, _ = model.filter(returns)
    assert h.shape == returns.shape
    assert bool((h > 0).all()) and bool(torch.isfinite(h).all())


@pytest.mark.parametrize("factory", ALL_VARIANTS)
def test_gradients_reach_every_trainable_parameter(factory, returns):
    set_seed(0)
    model = factory(hidden_size=10)
    model.train()
    model.negative_loglikelihood(returns).backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)


@pytest.mark.parametrize("factory", ALL_VARIANTS)
def test_initialisation_does_not_start_at_an_infinite_loss(factory, returns):
    """A ReLU readout with a zero bias returns sigma^2 = 0 and an infinite loss."""
    set_seed(0)
    model = factory(hidden_size=10)
    model.train()
    assert float(model.negative_loglikelihood(returns).detach()) < 1e6


@pytest.mark.slow
@pytest.mark.parametrize("factory", [SigmaCell, SigmaCellN, SigmaCellRL, SigmaLSTM])
def test_training_reduces_the_loss(factory, returns):
    set_seed(0)
    model = factory(hidden_size=10, config=ModelConfig(mean="zero", scale=None))
    model.train()
    before = float(model.negative_loglikelihood(returns).detach())
    model.fit(returns.numpy(), TrainConfig(lr=0.02, max_epochs=60, patience=20))
    model.eval()
    after = float(model.negative_loglikelihood(returns).detach())
    assert after < before


# --------------------------------------------------------------------------- #
# the stochastic layer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [SigmaCellN, SigmaCellNTV, SigmaLSTM])
def test_stochastic_variants_are_deterministic_in_eval(factory, returns):
    """Forecasts must be reproducible; the draw is replaced by its mean."""
    set_seed(0)
    model = factory(hidden_size=8)
    model.eval()
    first, _, _ = model.filter(returns)
    second, _, _ = model.filter(returns)
    assert torch.allclose(first, second, atol=1e-14)


@pytest.mark.parametrize("factory", [SigmaCellN, SigmaLSTM])
def test_stochastic_variants_sample_in_training(factory, returns):
    set_seed(0)
    model = factory(hidden_size=8)
    model.train()
    first, _, _ = model.filter(returns)
    second, _, _ = model.filter(returns)
    assert not torch.allclose(first, second, atol=1e-10)


def test_variance_ceiling_contains_the_feedback_loop(returns):
    """sigma-Cell-N's residual draw is a positive feedback loop; it must be capped."""
    set_seed(0)
    model = SigmaCellN(hidden_size=10)
    model.train()
    h, _, _ = model.filter(returns)
    assert float(h.max().detach()) <= model._variance_ceiling
    assert bool(torch.isfinite(h).all())


# --------------------------------------------------------------------------- #
# sigma-LSTM specifics
# --------------------------------------------------------------------------- #
def test_sigma_lstm_offers_both_readouts(returns):
    """Eq. 11's mean form and the 2023 paper's learned form."""
    set_seed(0)
    mean_form = SigmaLSTM(hidden_size=8, readout="mean")
    set_seed(0)
    learned = SigmaLSTM(hidden_size=8, readout="learned")
    mean_form.eval()
    learned.eval()
    a, _, _ = mean_form.filter(returns)
    b, _, _ = learned.filter(returns)
    assert not torch.allclose(a, b, atol=1e-8)
    assert bool((a > 0).all()) and bool((b > 0).all())


def test_sigma_lstm_records_all_three_resolved_ambiguities():
    text = " ".join(SigmaLSTM.provenance.deviations)
    assert "Eq. 8" in text and "standard deviation" in text
    assert "Eq. 11" in text
    assert "Eq. 12" in text
    assert SigmaLSTM.provenance.paper_id == "rodikov2022_sigma_lstm"


def test_sigma_cell_family_declares_the_2023_paper():
    for factory in (SigmaCell, SigmaCellN, SigmaCellRL, SigmaCellNTV, SigmaCellRLTV):
        assert factory.provenance.paper_id == "rodikov2023_sigma_cell"


def test_diagnostics_report_nan_persistence_rather_than_a_wrong_number(returns):
    """A non-linear recursion has no GARCH-style persistence; say so."""
    set_seed(0)
    model = SigmaCell(hidden_size=10, config=ModelConfig(mean="zero", scale=None))
    result = model.fit(returns.numpy(), TrainConfig(max_epochs=5, patience=3))
    assert np.isnan(result.persistence)
    assert "mean W_s (beta-like)" in result.parameters
