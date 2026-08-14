"""Constraint transforms must produce valid parameters for *any* real input."""

from __future__ import annotations

import pytest
import torch

from garchai.constraints import (
    EGARCHConstraint,
    GARCHConstraint,
    GJRConstraint,
    bounded,
    egarch_persistence,
    gjr_simplex,
    inverse_positive,
    positive,
    stationary_simplex,
    student_df,
)

torch.set_default_dtype(torch.float64)

# Deliberately extreme: the optimiser can and does visit these.
WILD = [-1e6, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1e6]


@pytest.fixture
def wild_draws() -> torch.Tensor:
    torch.manual_seed(0)
    fixed = torch.tensor(WILD)
    random = torch.randn(20_000) * 50.0
    return torch.cat([fixed, random])


def test_positive_is_strictly_positive(wild_draws):
    out = positive(wild_draws, eps=1e-8)
    assert torch.isfinite(out).all()
    assert (out > 0).all()


def test_positive_inverse_round_trip():
    targets = torch.tensor([1e-6, 1e-3, 0.05, 1.0, 25.0, 1e4])
    raw = torch.stack([inverse_positive(float(t)) for t in targets])
    assert torch.allclose(positive(raw), targets, rtol=1e-9, atol=1e-12)


def test_bounded_respects_its_interval(wild_draws):
    out = bounded(wild_draws, -3.0, 7.5)
    assert (out > -3.0).all() and (out < 7.5).all()


def test_bounded_rejects_inverted_interval():
    with pytest.raises(ValueError, match="upper must exceed lower"):
        bounded(torch.zeros(3), 1.0, 1.0)


def test_stationary_simplex_guarantees_stationarity(wild_draws):
    n = wild_draws.numel() // 3 * 3
    a, b, s = wild_draws[:n].reshape(3, -1)
    alpha, beta = stationary_simplex(a, b, s, slack=1e-4)
    assert (alpha >= 0).all()
    assert (beta >= 0).all()
    assert (alpha + beta < 1.0).all()
    assert (alpha + beta <= 1.0 - 1e-4 + 1e-12).all()


def test_gjr_simplex_guarantees_stationarity(wild_draws):
    n = wild_draws.numel() // 4 * 4
    a, g, b, s = wild_draws[:n].reshape(4, -1)
    alpha, gamma, beta = gjr_simplex(a, g, b, s, slack=1e-4)
    assert (alpha >= 0).all() and (gamma >= 0).all() and (beta >= 0).all()
    assert (alpha + 0.5 * gamma + beta < 1.0).all()


def test_egarch_persistence_inside_unit_circle(wild_draws):
    beta = egarch_persistence(wild_draws, slack=1e-4)
    assert (beta.abs() < 1.0).all()


def test_student_df_respects_variance_existence(wild_draws):
    nu = student_df(wild_draws, min_df=2.05, max_df=200.0)
    assert (nu > 2.0).all() and (nu < 200.0).all()


@pytest.mark.parametrize("slack", [0.0, 1.0, -0.1, 1.5])
def test_simplex_rejects_invalid_slack(slack):
    z = torch.zeros(())
    with pytest.raises(ValueError, match="slack"):
        stationary_simplex(z, z, z, slack=slack)


def test_garch_constraint_bundle_and_persistence():
    c = GARCHConstraint()
    p = c(torch.tensor(-2.0), torch.tensor(0.5), torch.tensor(2.0), torch.tensor(0.1))
    assert set(p) == {"omega", "alpha", "beta"}
    assert float(p["omega"]) > 0
    assert float(c.persistence(p)) == pytest.approx(float(p["alpha"] + p["beta"]))


def test_gjr_persistence_uses_half_gamma():
    c = GJRConstraint()
    p = c(*[torch.tensor(v) for v in (-2.0, 0.3, 0.4, 1.5, 0.2)])
    expected = float(p["alpha"] + 0.5 * p["gamma"] + p["beta"])
    assert float(c.persistence(p)) == pytest.approx(expected)


def test_egarch_constraint_leaves_omega_alpha_gamma_free():
    c = EGARCHConstraint()
    raws = [torch.tensor(v) for v in (-0.5, 1.2, -0.9, 3.0)]
    p = c(*raws)
    assert float(p["omega"]) == pytest.approx(-0.5)
    assert float(p["alpha"]) == pytest.approx(1.2)
    assert float(p["gamma"]) == pytest.approx(-0.9)
    assert abs(float(p["beta"])) < 1.0


def test_unenforced_mode_still_yields_non_negative_alpha_beta(wild_draws):
    c = GARCHConstraint(enforce=False)
    n = wild_draws.numel() // 3 * 3
    a, b, s = wild_draws[:n].reshape(3, -1)
    p = c(a, a, b, s)
    assert (p["alpha"] >= 0).all() and (p["beta"] >= 0).all()


def test_transforms_are_differentiable():
    raw = torch.randn(4, requires_grad=True)
    alpha, beta = stationary_simplex(raw[0], raw[1], raw[2])
    (alpha + beta + positive(raw[3])).backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
