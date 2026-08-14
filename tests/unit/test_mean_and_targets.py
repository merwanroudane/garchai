"""Mean models, and the volatility targets that papers disagree about."""

from __future__ import annotations

import math

import pytest
import torch

from garchai.mean import ARMAMean, ARMean, ConstantMean, ZeroMean, get_mean_model
from garchai.targets import (
    absolute_returns,
    bipower_variation,
    get_target,
    hansen_lunde_scale,
    realized_variance_intraday,
    realized_volatility_intraday,
    realized_volatility_wei,
    realized_volatility_zhao,
    rolling_std,
    squared_returns,
)

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
# mean models
# --------------------------------------------------------------------------- #
def test_zero_mean_leaves_returns_untouched():
    r = torch.tensor([0.5, -1.0, 2.0])
    eps, mu = ZeroMean().residuals(r)
    assert torch.equal(eps, r)
    assert float(mu.abs().max()) == 0.0
    assert ZeroMean().n_presample == 0


def test_constant_mean_subtracts_its_intercept():
    m = ConstantMean(mu=0.04)
    r = torch.tensor([1.0, 2.0])
    eps, mu = m.residuals(r)
    assert torch.allclose(mu, torch.full_like(r, 0.04))
    assert torch.allclose(eps, r - 0.04)


def test_ar_mean_matches_hand_computation():
    m = ARMean(p=1)
    with torch.no_grad():
        m.const.fill_(0.1)
        m.phi.fill_(0.5)
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mu = m.conditional_mean(r)
    assert torch.allclose(mu[1:], torch.tensor([0.6, 1.1, 1.6]), atol=1e-12)
    # the first entry is the unconditional mean, c / (1 - phi)
    assert float(mu[0]) == pytest.approx(0.1 / 0.5, abs=1e-12)


def test_ar_mean_declares_its_presample():
    assert ARMean(p=3).n_presample == 3


def test_ar_mean_handles_explosive_roots_without_dividing_by_zero():
    m = ARMean(p=1)
    with torch.no_grad():
        m.const.fill_(0.1)
        m.phi.fill_(1.0)  # unit root: 1 - sum(phi) == 0
    assert float(m.unconditional_mean()) == pytest.approx(0.1)


def test_ar_mean_rejects_order_zero():
    with pytest.raises(ValueError, match="p >= 1"):
        ARMean(p=0)


def test_arma_mean_matches_hand_computation():
    m = ARMAMean(p=1, q=1)
    with torch.no_grad():
        m.const.fill_(0.1)
        m.phi.fill_(0.5)
        m.theta.fill_(0.3)
    r = torch.tensor([1.0, 3.0, 5.0, 7.0])
    mu = m.conditional_mean(r)

    expected = [0.1]
    eps = [0.0]
    for t in range(1, 4):
        value = 0.1 + 0.5 * float(r[t - 1]) + 0.3 * eps[t - 1]
        expected.append(value)
        eps.append(float(r[t]) - value)
    assert torch.allclose(mu, torch.tensor(expected), atol=1e-12)


def test_arma_mean_is_differentiable():
    m = ARMAMean(p=1, q=1)
    r = torch.randn(30)
    eps, _ = m.residuals(r)
    eps.square().sum().backward()
    assert m.theta.grad is not None and torch.isfinite(m.theta.grad).all()


def test_mean_registry():
    assert get_mean_model("constant").name == "constant"
    assert get_mean_model("ar", p=2).n_presample == 2
    with pytest.raises(KeyError, match="Unknown mean model"):
        get_mean_model("kalman")


# --------------------------------------------------------------------------- #
# targets: the two "5-day realized volatility" estimators are NOT the same
# --------------------------------------------------------------------------- #
def test_zhao_target_matches_its_formula():
    """Sum of squares over a window that includes t, with no division by k."""
    eps = torch.tensor([1.0, 2.0, 2.0])
    out = realized_volatility_zhao(eps, k=2)
    assert float(out[0]) == pytest.approx(1.0)
    assert float(out[1]) == pytest.approx(math.sqrt(1.0 + 4.0))
    assert float(out[2]) == pytest.approx(math.sqrt(4.0 + 4.0))


def test_wei_target_matches_its_formula():
    """Demeaned, divided by k, and strictly lagged."""
    r = torch.tensor([1.0, 3.0, 5.0, 7.0])
    out = realized_volatility_wei(r, k=2)
    assert bool(out[:2].isnan().all())
    # window for t=2 is (r0, r1) = (1, 3): mean 2, mean squared deviation 1
    assert float(out[2]) == pytest.approx(1.0, abs=1e-12)
    assert float(out[3]) == pytest.approx(1.0, abs=1e-12)


def test_wei_target_is_strictly_lagged():
    """Changing r_t must not change the target at t - that would be look-ahead."""
    r = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    base = realized_volatility_wei(r, k=2)
    bumped = r.clone()
    bumped[3] = 100.0
    after = realized_volatility_wei(bumped, k=2)
    assert float(base[3]) == pytest.approx(float(after[3]), abs=1e-12)


def test_zhao_target_includes_the_current_observation():
    """The mirror image: Zhao's target does depend on eps_t, by construction."""
    eps = torch.tensor([1.0, 1.0, 1.0, 1.0])
    base = realized_volatility_zhao(eps, k=2)
    bumped = eps.clone()
    bumped[3] = 10.0
    after = realized_volatility_zhao(bumped, k=2)
    assert float(base[3]) != pytest.approx(float(after[3]))


def test_the_two_estimators_genuinely_differ():
    """Guards against anyone 'simplifying' them into one function later."""
    torch.manual_seed(0)
    x = torch.randn(50)
    zhao = realized_volatility_zhao(x, k=5)[5:]
    wei = realized_volatility_wei(x, k=5)[5:]
    ratio = (zhao / wei).mean()
    assert float(ratio) > 1.5


@pytest.mark.parametrize("k", [0, -1])
def test_targets_reject_invalid_windows(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        realized_volatility_zhao(torch.randn(10), k=k)


def test_rolling_std_is_lagged_and_unbiased():
    r = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    out = rolling_std(r, window=2)
    assert bool(out[:2].isnan().all())
    # window (1, 3): unbiased sd = sqrt(((1-2)^2 + (3-2)^2)/1) = sqrt(2)
    assert float(out[2]) == pytest.approx(math.sqrt(2.0), abs=1e-12)


def test_simple_proxies():
    eps = torch.tensor([-2.0, 3.0])
    assert torch.allclose(squared_returns(eps), torch.tensor([4.0, 9.0]))
    assert torch.allclose(absolute_returns(eps), torch.tensor([2.0, 3.0]))


def test_intraday_realized_measures():
    intraday = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.5, -0.5]])
    rv = realized_variance_intraday(intraday)
    assert torch.allclose(rv, torch.tensor([0.14, 0.50]), atol=1e-12)
    assert torch.allclose(realized_volatility_intraday(intraday), torch.sqrt(rv))
    percent = realized_volatility_intraday(intraday, percent=True)
    assert torch.allclose(percent, torch.sqrt(rv) * 100.0)


def test_bipower_variation_is_smaller_than_rv_under_a_jump():
    torch.manual_seed(0)
    intraday = torch.randn(1, 200) * 0.01
    intraday[0, 100] = 1.0  # a single large jump
    rv = float(realized_variance_intraday(intraday))
    bv = float(bipower_variation(intraday))
    assert bv < rv


def test_intraday_measures_reject_1d_input():
    with pytest.raises(ValueError, match=r"\(T, M\)"):
        realized_variance_intraday(torch.randn(10))


def test_hansen_lunde_scale_equalises_total_variation():
    torch.manual_seed(0)
    returns = torch.randn(500)
    rv = torch.rand(500) * 0.5  # deliberately understated, as trading-hours RV is
    scaled, c = hansen_lunde_scale(returns, rv)
    assert float(scaled.sum()) == pytest.approx(float(returns.square().sum()), rel=1e-12)
    assert float(c) > 1.0


def test_hansen_lunde_scale_requires_alignment():
    with pytest.raises(ValueError, match="align"):
        hansen_lunde_scale(torch.randn(10), torch.rand(9))


def test_target_registry():
    assert get_target("wei2025") is realized_volatility_wei
    assert get_target("zhao2024") is realized_volatility_zhao
    with pytest.raises(KeyError, match="Unknown target"):
        get_target("garman_klass")
