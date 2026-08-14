"""Loss functions, forecast-comparison tests and the model confidence set."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from garchai.evaluation import (
    diebold_mariano,
    giacomini_white,
    model_confidence_set,
    newey_west_variance,
)
from garchai.losses import (
    garch_informed_loss,
    gaussian_nll,
    huber,
    mae,
    mse,
    pinball,
    qlike,
    qlike_normalised,
    r2_oos,
    rmse,
    smape,
)

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def test_gaussian_nll_matches_the_closed_form():
    eps = torch.tensor([0.5, -1.5])
    h = torch.tensor([1.0, 4.0])
    expected = sum(
        0.5 * (math.log(2 * math.pi) + math.log(hi) + ei**2 / hi)
        for ei, hi in zip([0.5, -1.5], [1.0, 4.0])
    )
    assert float(gaussian_nll(eps, h)) == pytest.approx(expected, abs=1e-12)


def test_qlike_is_minimised_at_the_truth():
    truth = torch.tensor([1.0, 2.0, 0.5])
    best = float(qlike(truth, truth))
    for wrong in (0.5, 0.8, 1.3, 2.0):
        assert float(qlike(truth, truth * wrong)) > best


def test_qlike_normalised_is_zero_at_the_truth_and_non_negative():
    truth = torch.tensor([1.0, 2.0, 0.5])
    assert float(qlike_normalised(truth, truth)) == pytest.approx(0.0, abs=1e-14)
    for wrong in (0.3, 0.7, 1.5, 4.0):
        assert float(qlike_normalised(truth, truth * wrong)) > 0.0


def test_qlike_variants_rank_models_identically():
    torch.manual_seed(0)
    proxy = torch.rand(200) + 0.5
    a, b = proxy * 1.1, proxy * 1.6
    plain = float(qlike(proxy, a)) - float(qlike(proxy, b))
    shifted = float(qlike_normalised(proxy, a)) - float(qlike_normalised(proxy, b))
    assert plain == pytest.approx(shifted, abs=1e-12)


def test_qlike_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        qlike(torch.rand(5), torch.rand(6))


def test_point_losses():
    y = torch.tensor([1.0, 2.0, 3.0])
    p = torch.tensor([1.0, 2.0, 5.0])
    assert float(mse(y, p)) == pytest.approx(4.0 / 3.0)
    assert float(rmse(y, p)) == pytest.approx(math.sqrt(4.0 / 3.0))
    assert float(mae(y, p)) == pytest.approx(2.0 / 3.0)


def test_huber_is_quadratic_inside_and_linear_outside_delta():
    y = torch.tensor([0.0])
    small = huber(y, torch.tensor([0.5]), delta=1.0)
    large = huber(y, torch.tensor([10.0]), delta=1.0)
    assert float(small) == pytest.approx(0.125)
    assert float(large) == pytest.approx(1.0 * (10.0 - 0.5))


def test_huber_rejects_non_positive_delta():
    with pytest.raises(ValueError, match="delta must be positive"):
        huber(torch.zeros(3), torch.zeros(3), delta=0.0)


def test_pinball_penalises_the_two_sides_asymmetrically():
    q = torch.tensor([-2.0])
    below = pinball(torch.tensor([-3.0]), q, alpha=0.05)
    above = pinball(torch.tensor([-1.0]), q, alpha=0.05)
    # a violation costs (1 - alpha) per unit; an over-wide VaR costs alpha
    assert float(below) == pytest.approx(0.95)
    assert float(above) == pytest.approx(0.05)


def test_pinball_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        pinball(torch.zeros(3), torch.zeros(3), alpha=1.0)


def test_smape_uses_the_0_to_200_convention():
    y = torch.tensor([1.0])
    assert float(smape(y, torch.tensor([1.0]))) == pytest.approx(0.0)
    assert float(smape(y, torch.tensor([-1.0]))) == pytest.approx(200.0)


def test_r2_oos_is_one_for_perfect_and_zero_for_the_mean():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(r2_oos(y, y)) == pytest.approx(1.0)
    assert float(r2_oos(y, y.mean().expand_as(y))) == pytest.approx(0.0, abs=1e-12)
    assert float(r2_oos(y, torch.zeros_like(y))) < 0.0


def test_garch_informed_loss_endpoints():
    pred = torch.tensor([1.0, 2.0])
    truth = torch.tensor([1.5, 2.5])
    garch = torch.tensor([0.5, 1.5])
    assert float(garch_informed_loss(pred, truth, garch, lam=1.0)) == pytest.approx(
        float(mse(truth, pred))
    )
    # lam = 0 is GINN-0: train purely to imitate GARCH
    assert float(garch_informed_loss(pred, truth, garch, lam=0.0)) == pytest.approx(
        float(mse(garch, pred))
    )


def test_garch_informed_loss_uses_the_papers_weighting():
    pred = torch.tensor([1.0])
    truth = torch.tensor([2.0])
    garch = torch.tensor([0.0])
    expected = 0.01 * 1.0 + 0.99 * 1.0
    assert float(garch_informed_loss(pred, truth, garch, lam=0.01)) == pytest.approx(expected)


def test_garch_informed_loss_does_not_backpropagate_into_the_garch_target():
    pred = torch.tensor([1.0], requires_grad=True)
    garch = torch.tensor([0.5], requires_grad=True)
    garch_informed_loss(pred, torch.tensor([1.5]), garch, lam=0.5).backward()
    assert pred.grad is not None
    assert garch.grad is None


def test_garch_informed_loss_rejects_lambda_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"lam must lie in \[0, 1\]"):
        garch_informed_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), lam=1.5)


# --------------------------------------------------------------------------- #
# comparison tests
# --------------------------------------------------------------------------- #
def test_newey_west_reduces_to_the_sample_variance_at_zero_lags():
    x = np.random.default_rng(0).standard_normal(500)
    assert newey_west_variance(x, lags=0) == pytest.approx(x.var(), rel=1e-12)


def test_newey_west_accounts_for_positive_autocorrelation():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(4000)
    persistent = np.convolve(noise, np.ones(10) / 10.0, mode="same")
    assert newey_west_variance(persistent, lags=20) > newey_west_variance(persistent, lags=0)


def test_diebold_mariano_finds_no_difference_between_identical_models():
    rng = np.random.default_rng(0)
    loss = rng.random(500)
    result = diebold_mariano(loss, loss.copy())
    assert result.statistic == pytest.approx(0.0, abs=1e-9)
    assert not result.significant


def test_diebold_mariano_detects_a_genuinely_better_model():
    rng = np.random.default_rng(0)
    a = rng.random(1000)
    b = a + 0.15  # model B is uniformly worse
    result = diebold_mariano(a, b, names=("A", "B"))
    assert result.significant
    assert result.favours == "A"
    assert result.statistic < 0


def test_diebold_mariano_is_correctly_sized():
    """Under equal predictive ability it must reject about 5% of the time."""
    rejections = 0
    trials = 300
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        a, b = rng.standard_normal(400) ** 2, rng.standard_normal(400) ** 2
        rejections += diebold_mariano(a, b).significant
    assert 0.01 < rejections / trials < 0.11


def test_harvey_correction_shrinks_the_statistic():
    rng = np.random.default_rng(0)
    a, b = rng.random(80), rng.random(80) + 0.05
    corrected = diebold_mariano(a, b, horizon=5, harvey_correction=True)
    plain = diebold_mariano(a, b, horizon=5, harvey_correction=False)
    assert abs(corrected.statistic) < abs(plain.statistic)


def test_diebold_mariano_requires_aligned_series():
    with pytest.raises(ValueError, match="align"):
        diebold_mariano(np.zeros(10), np.zeros(11))


def test_giacomini_white_runs_and_detects_a_difference():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    b = a + 0.2
    result = giacomini_white(a, b, names=("A", "B"))
    assert result.significant and result.favours == "A"


# --------------------------------------------------------------------------- #
# model confidence set
# --------------------------------------------------------------------------- #
def test_mcs_keeps_only_the_dominant_model_when_one_clearly_wins():
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 600
    base = rng.random(n)
    losses = pd.DataFrame(
        {"good": base, "poor": base + 1.0, "awful": base + 2.0}
    )
    table = model_confidence_set(losses, alpha=0.10, n_bootstrap=300, seed=0)
    assert table.iloc[0]["model"] == "good"
    assert bool(table.set_index("model").loc["good", "in_mcs"])
    assert not bool(table.set_index("model").loc["awful", "in_mcs"])


def test_mcs_retains_every_model_when_they_are_indistinguishable():
    import pandas as pd

    rng = np.random.default_rng(1)
    losses = pd.DataFrame({f"m{i}": rng.random(400) for i in range(4)})
    table = model_confidence_set(losses, alpha=0.10, n_bootstrap=300, seed=0)
    assert bool(table["in_mcs"].all())


def test_mcs_ranks_by_mean_loss():
    import pandas as pd

    rng = np.random.default_rng(2)
    base = rng.random(300)
    losses = pd.DataFrame({"c": base + 0.6, "a": base, "b": base + 0.3})
    table = model_confidence_set(losses, alpha=0.10, n_bootstrap=200, seed=0)
    assert list(table["model"]) == ["a", "b", "c"]


def test_mcs_needs_at_least_two_models():
    with pytest.raises(ValueError, match="at least two models"):
        model_confidence_set(np.random.default_rng(0).random((100, 1)))
