"""Recurrence, initialisation and gradient correctness for the variance cells."""

from __future__ import annotations

import math

import pytest
import torch

from garchai.cells import CellState, EGARCHCell, GARCHCell, GJRCell
from garchai.distributions import Normal, StudentT

torch.set_default_dtype(torch.float64)


@pytest.fixture
def eps() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(200) * 1.1


# --------------------------------------------------------------------------- #
# round-trip: what you ask for is what you get
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "omega,alpha,beta", [(0.05, 0.10, 0.85), (0.01, 0.05, 0.90), (0.2, 0.30, 0.40)]
)
def test_garch_starting_values_survive_the_transform(omega, alpha, beta):
    p = {k: v.detach() for k, v in GARCHCell(omega, alpha, beta).transformed_parameters().items()}
    assert float(p["omega"]) == pytest.approx(omega, abs=1e-12)
    assert float(p["alpha"]) == pytest.approx(alpha, abs=1e-12)
    assert float(p["beta"]) == pytest.approx(beta, abs=1e-12)


@pytest.mark.parametrize(
    "omega,alpha,gamma,beta", [(0.05, 0.05, 0.08, 0.85), (0.02, 0.02, 0.15, 0.80)]
)
def test_gjr_starting_values_survive_the_transform(omega, alpha, gamma, beta):
    p = {k: v.detach() for k, v in GJRCell(omega, alpha, gamma, beta).transformed_parameters().items()}
    for name, want in [("omega", omega), ("alpha", alpha), ("gamma", gamma), ("beta", beta)]:
        assert float(p[name]) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize(
    "omega,alpha,gamma,beta", [(-0.05, 0.15, -0.08, 0.97), (0.01, 0.20, 0.05, 0.90)]
)
def test_egarch_starting_values_survive_the_transform(omega, alpha, gamma, beta):
    p = {k: v.detach() for k, v in EGARCHCell(omega, alpha, gamma, beta).transformed_parameters().items()}
    for name, want in [("omega", omega), ("alpha", alpha), ("gamma", gamma), ("beta", beta)]:
        assert float(p[name]) == pytest.approx(want, abs=1e-12)


def test_garch_rejects_non_stationary_starting_values():
    with pytest.raises(ValueError, match="violates stationarity"):
        GARCHCell(0.05, 0.5, 0.6)


def test_gjr_rejects_negative_gamma_when_enforcing():
    with pytest.raises(ValueError, match="gamma >= 0"):
        GJRCell(0.05, 0.05, -0.02, 0.85)


def test_gjr_allows_negative_gamma_when_not_enforcing():
    p = {k: v.detach() for k, v in GJRCell(
        0.05, 0.05, -0.02, 0.85, enforce_stationarity=False
    ).transformed_parameters().items()}
    assert float(p["gamma"]) == pytest.approx(-0.02, abs=1e-12)


# --------------------------------------------------------------------------- #
# the recursions themselves
# --------------------------------------------------------------------------- #
def test_garch_recursion_matches_hand_computation():
    cell = GARCHCell(0.05, 0.10, 0.85, init_method="sample")
    e = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.5])
    h, _ = cell.filter(e)

    manual = [float(e.var(unbiased=True))]
    for t in range(1, 5):
        manual.append(0.05 + 0.10 * float(e[t - 1]) ** 2 + 0.85 * manual[-1])
    assert torch.allclose(h, torch.tensor(manual), atol=1e-14)


def test_gjr_recursion_matches_hand_computation():
    cell = GJRCell(0.05, 0.05, 0.08, 0.85, init_method="sample")
    e = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.5])
    h, _ = cell.filter(e)

    manual = [float(e.var(unbiased=True))]
    for t in range(1, 5):
        prev = float(e[t - 1])
        manual.append(
            0.05 + 0.05 * prev**2 + 0.08 * (prev < 0) * prev**2 + 0.85 * manual[-1]
        )
    assert torch.allclose(h, torch.tensor(manual), atol=1e-14)


def test_egarch_recursion_matches_hand_computation():
    cell = EGARCHCell(-0.05, 0.15, -0.08, 0.97, init_method="sample")
    e = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.5])
    h, _ = cell.filter(e)

    eabs = math.sqrt(2.0 / math.pi)
    manual = [float(e.var(unbiased=True))]
    for t in range(1, 5):
        z = float(e[t - 1]) / math.sqrt(manual[-1])
        log_h = -0.05 + 0.97 * math.log(manual[-1]) + 0.15 * (abs(z) - eabs) - 0.08 * z
        manual.append(math.exp(log_h))
    assert torch.allclose(h, torch.tensor(manual), rtol=1e-12)


def test_gjr_reduces_to_garch_when_gamma_is_zero(eps):
    g, j = GARCHCell(0.05, 0.10, 0.85), GJRCell(0.05, 0.10, 0.0, 0.85)
    assert torch.allclose(g.filter(eps)[0], j.filter(eps)[0], atol=1e-12)


def test_negative_shocks_raise_variance_more_under_gjr():
    # alpha + gamma/2 + beta = 0.05 + 0.10 + 0.75 = 0.90 < 1
    cell = GJRCell(0.05, 0.05, 0.20, 0.75, init_method="sample")
    up, _ = cell.filter(torch.tensor([2.0, 0.0, 0.0]))
    down, _ = cell.filter(torch.tensor([-2.0, 0.0, 0.0]))
    assert float(down[1]) > float(up[1])


@pytest.mark.parametrize("cell_factory", [GARCHCell, GJRCell, EGARCHCell])
def test_variance_is_always_positive_and_finite(cell_factory, eps):
    h, _ = cell_factory().filter(torch.cat([eps, torch.tensor([50.0, -50.0, 0.0])]))
    assert torch.isfinite(h).all()
    assert (h > 0).all()


def test_filter_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1-D"):
        GARCHCell().filter(torch.randn(3, 4))


def test_filter_output_is_aligned_with_input(eps):
    h, _ = GARCHCell().filter(eps)
    assert h.shape == eps.shape


def test_filter_can_resume_from_a_state(eps):
    cell = GARCHCell(0.05, 0.10, 0.85)
    full, _ = cell.filter(eps)

    first, state = cell.filter(eps[:100])
    # continuing needs the step that bridges eps[99] -> h[100]
    h100, state = cell.step(eps[99], state)
    rest, _ = cell.filter(eps[100:], state=CellState(h=h100))
    joined = torch.cat([first, rest])
    assert torch.allclose(joined, full, atol=1e-12)


# --------------------------------------------------------------------------- #
# initialisation
# --------------------------------------------------------------------------- #
def test_tsgarch_backcast_matches_its_published_formula(eps):
    cell = GARCHCell(init_method="tsgarch_backcast")
    lam, n = cell.TSGARCH_LAMBDA, eps.numel()
    sq = eps.square()
    want = (lam**n) * sq.mean() + (1 - lam) * ((lam ** torch.arange(n)) * sq).sum()
    assert float(cell.tsgarch_backcast(eps)) == pytest.approx(float(want), abs=1e-14)


def test_arch_backcast_truncates_at_75_observations(eps):
    cell = GARCHCell()
    lam = cell.ARCH_LAMBDA
    w = lam ** torch.arange(75)
    w = w / w.sum()
    want = (w * eps[:75].square()).sum()
    assert float(cell.arch_backcast(eps)) == pytest.approx(float(want), abs=1e-14)


def test_arch_backcast_applies_one_presample_step(eps):
    """``arch`` sets h1 by recursing on the backcast; ``tsgarch`` uses it directly."""
    cell = GARCHCell(0.05, 0.10, 0.85, init_method="arch_backcast")
    bc = cell.arch_backcast(eps)
    h, _ = cell.filter(eps)
    assert float(h[0]) == pytest.approx(0.05 + (0.10 + 0.85) * float(bc), abs=1e-14)
    assert float(h[0]) != pytest.approx(float(bc), abs=1e-6)


def test_tsgarch_backcast_is_used_directly(eps):
    cell = GARCHCell(0.05, 0.10, 0.85, init_method="tsgarch_backcast")
    h, _ = cell.filter(eps)
    assert float(h[0]) == pytest.approx(float(cell.tsgarch_backcast(eps)), abs=1e-14)


def test_first_n_uses_only_the_leading_observations(eps):
    cell = GARCHCell(init_method="first_n", sample_n=10)
    assert float(cell.initial_variance(eps)) == pytest.approx(
        float(eps[:10].square().mean()), abs=1e-14
    )


def test_unconditional_init_uses_the_model_implied_variance(eps):
    cell = GARCHCell(0.05, 0.10, 0.85, init_method="unconditional")
    assert float(cell.initial_variance(eps)) == pytest.approx(0.05 / (1 - 0.95), rel=1e-6)


def test_trainable_init_is_a_parameter(eps):
    cell = GARCHCell(init_method="trainable")
    assert cell.raw_h0 is not None
    assert "raw_h0" in dict(cell.named_parameters())
    assert float(cell.initial_variance(eps)) > 0


def test_unknown_init_method_is_rejected(eps):
    cell = GARCHCell()
    cell.init_method = "wishful"
    with pytest.raises(ValueError, match="Unknown init_method"):
        cell.initial_variance(eps)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def test_persistence_and_long_run_variance():
    cell = GARCHCell(0.05, 0.10, 0.85)
    assert float(cell.persistence()) == pytest.approx(0.95, abs=1e-12)
    assert float(cell.unconditional_variance()) == pytest.approx(1.0, rel=1e-9)


def test_egarch_long_run_variance_is_exponential():
    cell = EGARCHCell(-0.05, 0.15, -0.08, 0.90)
    assert float(cell.unconditional_variance()) == pytest.approx(math.exp(-0.05 / 0.10), rel=1e-9)


def test_summary_row_reports_econometric_quantities():
    row = GARCHCell(0.05, 0.10, 0.85).summary_row()
    assert row["alpha"] == pytest.approx(0.10, abs=1e-12)
    assert row["persistence"] == pytest.approx(0.95, abs=1e-12)
    assert row["long_run_variance"] == pytest.approx(1.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "factory,dist",
    [
        (lambda: GARCHCell(0.05, 0.10, 0.85, init_method="sample"), Normal()),
        (lambda: GJRCell(0.05, 0.05, 0.08, 0.85, init_method="sample"), StudentT(6.0)),
        (lambda: EGARCHCell(init_method="sample"), Normal()),
    ],
    ids=["garch", "gjr", "egarch"],
)
def test_analytic_gradients_match_central_differences(factory, dist):
    torch.manual_seed(0)
    e = torch.randn(120)
    cell = factory()

    def loss() -> torch.Tensor:
        h, _ = cell.filter(e)
        return dist.nll(e, h)

    loss().backward()
    step = 1e-6
    for name, param in cell.named_parameters():
        analytic = float(param.grad)
        base = float(param.data)
        param.data = torch.tensor(base + step)
        up = float(loss())
        param.data = torch.tensor(base - step)
        down = float(loss())
        param.data = torch.tensor(base)
        numeric = (up - down) / (2 * step)
        assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-6), name


def test_state_is_not_detached_by_default():
    """A silently truncated gradient would make long-memory training wrong."""
    cell = GARCHCell(0.05, 0.10, 0.85, init_method="sample")
    e = torch.randn(60)
    h, _ = cell.filter(e)
    h[-1].backward()
    assert cell.raw_beta.grad is not None
    assert abs(float(cell.raw_beta.grad)) > 0


def test_truncated_bptt_changes_the_gradient():
    e = torch.randn(60)
    full = GARCHCell(0.05, 0.10, 0.85, init_method="sample")
    trunc = GARCHCell(0.05, 0.10, 0.85, init_method="sample")
    full.filter(e)[0][-1].backward()
    trunc.filter(e, truncate_bptt=5)[0][-1].backward()
    assert float(full.raw_beta.grad) != pytest.approx(float(trunc.raw_beta.grad), rel=1e-3)
