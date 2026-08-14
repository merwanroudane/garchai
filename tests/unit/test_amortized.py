"""De Clerk & Savel'ev (2022): amortized GARCH parameter estimation.

The construction rests on two closed forms, so the strongest tests are exact
rather than statistical: the moment map must round-trip to machine precision,
and the feasibility region must reject pairs that no GARCH can produce. If
either fails, the network is learning the wrong function.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from garchai.core import TrainConfig, set_seed
from garchai.structural import (
    AmortizedGARCHInference,
    GARCHParameters,
    NeuralGARCHEstimator,
    empirical_moments,
    fourth_standardised_moment,
    garch_moment_map,
    normalised_autocovariance,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def garch_series() -> np.ndarray:
    """A simulated GARCH(1,1) with known parameters."""
    rng = np.random.default_rng(0)
    n, omega, alpha, beta = 3000, 0.05, 0.09, 0.88
    h = np.zeros(n)
    e = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    return e[1:]


@pytest.fixture(scope="module")
def estimator() -> AmortizedGARCHInference:
    set_seed(0)
    model = AmortizedGARCHInference(lag=15, n_samples=8000)
    model.fit(TrainConfig(lr=1e-3, max_epochs=25, patience=10))
    return model


# --------------------------------------------------------------------------- #
# the closed forms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "alpha,beta", [(0.09, 0.88), (0.05, 0.90), (0.15, 0.75), (0.02, 0.95)]
)
def test_moment_map_round_trips_exactly(alpha, beta):
    """Eq. 3 inverted must return the beta that produced the kurtosis."""
    _, kurtosis = garch_moment_map(alpha, beta, lag=15)
    recovered = AmortizedGARCHInference(lag=15).beta_from_moments(alpha, kurtosis)
    assert recovered == pytest.approx(beta, abs=1e-12)


def test_kurtosis_exceeds_three_for_any_valid_garch():
    """Eq. 3's second term is strictly positive whenever alpha > 0."""
    for alpha, beta in [(0.05, 0.9), (0.1, 0.85), (0.2, 0.7)]:
        assert fourth_standardised_moment(alpha, beta) > 3.0


def test_autocovariance_decays_geometrically_in_the_lag():
    r"""Eq. 2 carries the factor :math:`(\alpha+\beta)^{n-1}`."""
    alpha, beta = 0.09, 0.88
    first = normalised_autocovariance(alpha, beta, lag=1)
    tenth = normalised_autocovariance(alpha, beta, lag=10)
    assert tenth == pytest.approx(first * (alpha + beta) ** 9, rel=1e-12)


def test_kurtosis_rises_with_alpha():
    """More ARCH means fatter tails, holding persistence roughly fixed."""
    low = fourth_standardised_moment(0.05, 0.90)
    high = fourth_standardised_moment(0.15, 0.80)
    assert high > low


def test_beta_inversion_rejects_an_impossible_kurtosis():
    estimator = AmortizedGARCHInference()
    with pytest.raises(ValueError, match="must exceed 3"):
        estimator.beta_from_moments(0.09, 2.5)


def test_beta_inversion_rejects_an_infeasible_pair():
    estimator = AmortizedGARCHInference()
    with pytest.raises(ValueError, match="outside the GARCH-feasible region"):
        estimator.beta_from_moments(0.45, 3.001)


# --------------------------------------------------------------------------- #
# empirical moments
# --------------------------------------------------------------------------- #
def test_empirical_moments_recover_the_theoretical_ones(garch_series):
    """On a long simulated path the sample moments should be near the closed forms."""
    gamma, kurtosis = empirical_moments(garch_series, lag=15)
    _, theoretical = garch_moment_map(0.09, 0.88, lag=15)
    assert kurtosis > 3.0
    assert gamma > 0.0
    assert kurtosis == pytest.approx(theoretical, rel=0.6)


def test_empirical_moments_reject_a_series_shorter_than_the_lag():
    with pytest.raises(ValueError, match="Need more than"):
        empirical_moments(np.random.default_rng(0).standard_normal(10), lag=15)


def test_gaussian_noise_has_kurtosis_near_three():
    """A series with no ARCH effect should sit at the edge of the region."""
    x = np.random.default_rng(0).standard_normal(20_000)
    _, kurtosis = empirical_moments(x, lag=15)
    assert kurtosis == pytest.approx(3.0, abs=0.2)


# --------------------------------------------------------------------------- #
# the feasibility region (paper Fig. 1)
# --------------------------------------------------------------------------- #
def test_feasibility_accepts_reachable_pairs():
    estimator = AmortizedGARCHInference(lag=15)
    gamma, kurtosis = garch_moment_map(0.09, 0.88, lag=15)
    assert estimator.is_feasible(gamma, kurtosis)


@pytest.mark.parametrize(
    "gamma,kurtosis,reason",
    [
        (0.05, 2.5, "kurtosis below 3"),
        (-0.10, 5.0, "negative autocovariance"),
        (0.05, float("nan"), "non-finite"),
        (0.05, 3.0, "kurtosis exactly 3"),
    ],
)
def test_feasibility_rejects_unreachable_pairs(gamma, kurtosis, reason):
    assert not AmortizedGARCHInference(lag=15).is_feasible(gamma, kurtosis), reason


def test_strict_mode_refuses_rather_than_fabricating():
    """The paper discards infeasible windows; inventing a number would be worse."""
    estimator = AmortizedGARCHInference(lag=15)
    gaussian = np.random.default_rng(1).standard_normal(500)   # no ARCH effect
    gamma, kurtosis = empirical_moments(gaussian, lag=15)
    if not estimator.is_feasible(gamma, kurtosis):
        with pytest.raises(ValueError, match="outside the GARCH-feasible region"):
            estimator.estimate(gaussian, strict=True)


def test_non_strict_mode_flags_rather_than_hides(garch_series):
    estimator = AmortizedGARCHInference(lag=15)
    result = estimator.estimate(garch_series, strict=False)
    assert isinstance(result, GARCHParameters)
    assert isinstance(result.feasible, bool)


# --------------------------------------------------------------------------- #
# the network
# --------------------------------------------------------------------------- #
def test_training_set_is_all_admissible():
    """Every synthetic row must be a pair the moment map can actually reach."""
    estimator = AmortizedGARCHInference(lag=15, n_samples=2000)
    x, y = estimator.make_training_set()
    assert x.shape == (2000, 2)
    assert y.shape == (2000,)
    assert bool((y > 0).all()) and bool((y < 1).all())
    assert bool((x[:, 1] > 3.0).all())      # kurtosis above 3
    assert bool(torch.isfinite(x).all())


def test_training_reduces_validation_error():
    set_seed(0)
    estimator = AmortizedGARCHInference(lag=15, n_samples=4000)
    history = estimator.fit(TrainConfig(lr=1e-3, max_epochs=25, patience=10))
    assert history["val_loss"][-1] < history["val_loss"][0]


def test_alpha_output_is_always_admissible(estimator, garch_series):
    """The sigmoid keeps alpha_1 in (0, 1) whatever the network emits."""
    result = estimator.estimate(garch_series, strict=False)
    assert 0.0 < result.alpha1 < 1.0
    assert result.beta1 >= 0.0


def test_estimate_returns_a_usable_parameter_triple(estimator, garch_series):
    result = estimator.estimate(garch_series, strict=False)
    parameters = result.as_dict()
    assert set(parameters) == {"omega", "alpha", "beta"}
    assert all(np.isfinite(v) for v in parameters.values())
    assert result.persistence == pytest.approx(result.alpha1 + result.beta1)


def test_warm_start_builds_a_valid_garch(estimator, garch_series):
    """The paper's practical pitch: seed a likelihood optimiser cheaply."""
    model = estimator.warm_start(garch_series)
    parameters = model.cell.transformed_parameters()
    assert float(parameters["omega"].detach()) > 0
    assert float(model.cell.persistence().detach()) < 1.0


def test_estimation_is_fast(estimator, garch_series):
    """Speed is the entire point; a slow estimator is a failed one."""
    import time

    start = time.perf_counter()
    for _ in range(10):
        estimator.estimate(garch_series, strict=False)
    elapsed = (time.perf_counter() - start) / 10
    assert elapsed < 0.5, f"{elapsed * 1000:.0f} ms per estimate is too slow"


def test_alias_and_provenance():
    assert NeuralGARCHEstimator is AmortizedGARCHInference
    assert AmortizedGARCHInference.provenance.paper_id == "declerk2022_amortized_garch"
    notes = AmortizedGARCHInference.provenance.notes or ""
    assert "synthetic by construction" in notes
