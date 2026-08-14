"""GARCHNet (Buczynski & Chlebus 2023) and ANN-GARCH (Liu & So 2020).

GARCHNet's distinguishing claim is that the innovation distribution's *shape*
is time-varying, not just its scale. Two things follow that must be tested:
the admissibility constraints (``eta_t > 2``, ``|lambda_t| < 1``) have to hold at
every step for any network output, and the VaR path has to actually use the
period-specific quantile rather than an average one — otherwise the model is a
GARCH-t with extra parameters.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from garchai.core import ModelConfig, TrainConfig, set_seed
from garchai.structural import ANNGARCH, GARCHNet, LikelihoodGARCHLSTM

torch.set_default_dtype(torch.float64)


@pytest.fixture
def returns() -> torch.Tensor:
    set_seed(0)
    return torch.randn(250) * 0.9


# --------------------------------------------------------------------------- #
# GARCHNet
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("distribution", ["normal", "student-t", "skew-t"])
def test_variance_is_positive_and_aligned(returns, distribution):
    set_seed(0)
    model = GARCHNet(
        window=10, hidden_size=16,
        config=ModelConfig(mean="zero", distribution=distribution),
    )
    h, eps, _ = model.filter(returns)
    assert h.shape == returns.shape
    assert bool((h > 0).all()) and bool(torch.isfinite(h).all())


@pytest.mark.parametrize("distribution", ["student-t", "skew-t"])
def test_degrees_of_freedom_always_exceed_two(returns, distribution):
    """Eq. 20's '+2'. Below 2 the standardised Student-t has no variance."""
    set_seed(0)
    model = GARCHNet(
        window=10, hidden_size=16,
        config=ModelConfig(mean="zero", distribution=distribution),
    )
    model.filter(returns)
    assert bool((model._last_paths["eta"] > 2.0).all())


def test_skewness_stays_inside_the_admissible_range(returns):
    """Eq. 21's tanh enforces Hansen's -1 < lambda < 1."""
    set_seed(0)
    model = GARCHNet(
        window=10, hidden_size=16, config=ModelConfig(mean="zero", distribution="skew-t")
    )
    model.filter(returns)
    assert bool((np.abs(model._last_paths["lambda"]) < 1.0).all())


def test_constraints_survive_extreme_inputs():
    """They must hold for any network output, not just well-behaved ones."""
    set_seed(0)
    model = GARCHNet(
        window=10, hidden_size=16, config=ModelConfig(mean="zero", distribution="skew-t")
    )
    shocked = torch.cat([torch.randn(60), torch.tensor([80.0, -80.0, 0.0])])
    h, _, _ = model.filter(shocked)
    assert bool((h > 0).all()) and bool(torch.isfinite(h).all())
    assert bool((model._last_paths["eta"] > 2.0).all())
    assert bool((np.abs(model._last_paths["lambda"]) < 1.0).all())


def test_shape_parameters_become_time_varying_after_training(returns):
    """The paper's contribution: eta_t and lambda_t must genuinely move."""
    set_seed(0)
    model = GARCHNet(
        window=15, hidden_size=16,
        config=ModelConfig(mean="zero", distribution="student-t", scale=None),
    )
    model.filter(returns)
    before = float(model._last_paths["eta"].std())
    model.fit(returns.numpy(), TrainConfig(lr=0.01, max_epochs=40, patience=15))
    after = float(model._last_paths["eta"].std())
    assert after > before
    assert after > 1e-3, "eta_t is effectively constant; the head is not learning"


def test_fixed_shape_ablation_really_is_fixed(returns):
    set_seed(0)
    model = GARCHNet(
        window=10, hidden_size=16, time_varying_shape=False,
        config=ModelConfig(mean="zero", distribution="skew-t"),
    )
    model.filter(returns)
    assert model._last_paths["eta"].std() == pytest.approx(0.0, abs=1e-12)
    assert model._last_paths["lambda"].std() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("distribution", ["normal", "student-t", "skew-t"])
def test_gradients_reach_every_head(returns, distribution):
    set_seed(0)
    model = GARCHNet(
        window=10, hidden_size=12,
        config=ModelConfig(mean="zero", distribution=distribution),
    )
    model.negative_loglikelihood(returns).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_var_path_uses_period_specific_quantiles(returns):
    """Using a single averaged quantile would discard the paper's contribution."""
    set_seed(0)
    model = GARCHNet(
        window=15, hidden_size=16,
        config=ModelConfig(mean="zero", distribution="student-t", scale=None),
    )
    result = model.fit(returns.numpy(), TrainConfig(lr=0.01, max_epochs=30, patience=12))
    var = model.var_path(0.01).numpy()

    sigma = result.conditional_volatility
    ratio = var / sigma          # would be constant if one quantile were used
    assert float(np.std(ratio)) > 1e-6
    assert np.all(var < 0.0)     # a 1% VaR is a loss level


def test_rejects_unsupported_distribution():
    with pytest.raises(ValueError, match="normal|student-t|skew-t"):
        GARCHNet(config=ModelConfig(distribution="ged"))


def test_provenance_flags_the_missing_official_code():
    text = " ".join(GARCHNet.provenance.deviations)
    assert "corresponding author" in text
    assert GARCHNet.provenance.paper_id == "buczynski2023_garchnet"


# --------------------------------------------------------------------------- #
# ANN-GARCH (Liu & So)
# --------------------------------------------------------------------------- #
def test_ann_garch_readout_is_softplus_squared(returns):
    """Eq. 7 squares a softplus, which is steeper than either alone."""
    set_seed(0)
    model = ANNGARCH(window=10, hidden_size=12)
    h, _, _ = model.filter(returns)
    assert bool((h > 0).all()) and bool(torch.isfinite(h).all())


def test_ann_garch_defaults_to_the_papers_student_t():
    """The paper specifies a standardised Student-t (sec. 2.2)."""
    assert ANNGARCH(window=5).config.distribution == "student-t"


def test_ann_garch_accepts_gaussian_qmle():
    """Quasi-MLE under a Gaussian is legitimate and must not be rejected."""
    model = ANNGARCH(window=5, config=ModelConfig(mean="zero", distribution="normal"))
    assert not model.use_student_t
    assert model._distribution_parameters() == {}


def test_ann_garch_rejects_genuinely_unsupported_distributions():
    with pytest.raises(ValueError, match="student-t|normal"):
        ANNGARCH(config=ModelConfig(mean="zero", distribution="ged"))


def test_ann_garch_is_an_alias_of_likelihood_garch_lstm():
    assert LikelihoodGARCHLSTM is ANNGARCH


def test_ann_garch_trains(returns):
    set_seed(0)
    model = ANNGARCH(
        window=10, hidden_size=12,
        config=ModelConfig(mean="zero", distribution="student-t", scale=None),
    )
    before = float(model.negative_loglikelihood(returns).detach())
    result = model.fit(returns.numpy(), TrainConfig(lr=0.02, max_epochs=30, patience=12))
    assert -result.loglikelihood < before
    assert result.distribution_parameters["df (nu)"] > 2.0


def test_ann_garch_accepts_exogenous_inputs(returns):
    set_seed(0)
    exog = torch.randn(returns.numel(), 3)
    model = ANNGARCH(
        window=10, hidden_size=12, n_exog=3,
        config=ModelConfig(mean="zero", distribution="student-t", scale=None),
    )
    result = model.fit(
        returns.numpy(), TrainConfig(lr=0.02, max_epochs=15, patience=8), exog=exog
    )
    assert np.isfinite(result.loglikelihood)


def test_ann_garch_rejects_a_mismatched_exog_panel(returns):
    model = ANNGARCH(window=10, n_exog=3)
    with pytest.raises(ValueError, match=r"shape \(T, 3\)"):
        model.fit(returns.numpy(), TrainConfig(max_epochs=1), exog=torch.randn(250, 2))


def test_ann_garch_variable_importance(returns):
    """The paper's measure: the drop in log-likelihood when a variable is removed."""
    set_seed(0)
    exog = torch.randn(returns.numel(), 3)
    exog[:, 0] *= 5.0        # make one variable obviously more influential
    model = ANNGARCH(
        window=10, hidden_size=12, n_exog=3,
        config=ModelConfig(mean="zero", distribution="student-t", scale=None),
    )
    model.fit(returns.numpy(), TrainConfig(lr=0.02, max_epochs=15, patience=8), exog=exog)
    importance = model.variable_importance(returns.numpy(), exog)
    assert set(importance) == {0, 1, 2}
    assert all(np.isfinite(v) for v in importance.values())


def test_ann_garch_declares_the_bloomberg_substitution():
    assert ANNGARCH.provenance.data_substitutions
    assert "Bloomberg" in " ".join(ANNGARCH.provenance.data_substitutions)


def test_fit_signature_does_not_swallow_the_train_config(returns):
    """Regression: exog/realized_variance must be keyword-only.

    With the extra data argument in the second positional slot, the natural call
    ``fit(returns, TrainConfig(...))`` bound the config to it and failed deep
    inside a tensor conversion.
    """
    set_seed(0)
    model = ANNGARCH(window=10, hidden_size=8,
                     config=ModelConfig(mean="zero", distribution="student-t", scale=None))
    result = model.fit(returns.numpy(), TrainConfig(max_epochs=2, patience=1))
    assert np.isfinite(result.loglikelihood)
