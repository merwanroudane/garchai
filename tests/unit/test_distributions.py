"""Every innovation distribution must be a genuine standardised density.

If these tests pass, ``h_t`` really is the conditional variance. If the
standardisation were wrong, every VaR number in the package would be silently
mis-scaled while every other test still passed.
"""

from __future__ import annotations

import math

import pytest
import torch

from garchai.distributions import GED, Normal, SkewT, StudentT, StudentTKernel, get_distribution

torch.set_default_dtype(torch.float64)

# nu = 3 is excluded from the moment checks: its variance integral converges too
# slowly for quadrature on any practical grid (the integrand decays like z^-2).
PROPER = [
    Normal(),
    StudentT(5.0),
    StudentT(8.0),
    StudentT(30.0),
    SkewT(6.0, -0.3),
    SkewT(10.0, 0.5),
    SkewT(8.0, 0.0),
    GED(1.2),
    GED(2.0),
    GED(3.0),
]

GRID = torch.linspace(-60.0, 60.0, 600_001)


def _moments(dist):
    pdf = torch.exp(dist.log_prob(GRID))
    mass = torch.trapz(pdf, GRID)
    mean = torch.trapz(GRID * pdf, GRID)
    var = torch.trapz(GRID * GRID * pdf, GRID)
    return float(mass), float(mean), float(var)


@pytest.mark.parametrize("dist", PROPER, ids=lambda d: f"{d.name}")
def test_density_integrates_to_one(dist):
    mass, _, _ = _moments(dist)
    assert mass == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("dist", PROPER, ids=lambda d: f"{d.name}")
def test_zero_mean_and_unit_variance(dist):
    _, mean, var = _moments(dist)
    assert mean == pytest.approx(0.0, abs=1e-5)
    assert var == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize("dist", PROPER, ids=lambda d: f"{d.name}")
@pytest.mark.parametrize("alpha", [0.01, 0.025, 0.05, 0.10])
def test_ppf_inverts_the_cdf(dist, alpha):
    q = float(dist.ppf(alpha))
    grid = torch.linspace(-400.0, q, 400_001)
    mass = float(torch.trapz(torch.exp(dist.log_prob(grid)), grid))
    assert mass == pytest.approx(alpha, rel=2e-3)


@pytest.mark.parametrize("dist", PROPER, ids=lambda d: f"{d.name}")
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_expected_shortfall_matches_quadrature(dist, alpha):
    q = float(dist.ppf(alpha))
    grid = torch.linspace(-600.0, q, 1_200_001)
    pdf = torch.exp(dist.log_prob(grid))
    reference = float(torch.trapz(grid * pdf, grid)) / alpha
    assert float(dist.expected_shortfall(alpha)) == pytest.approx(reference, rel=1e-4)


@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_es_is_more_extreme_than_var(alpha):
    for dist in PROPER:
        assert float(dist.expected_shortfall(alpha)) < float(dist.ppf(alpha))


def test_normal_matches_closed_form():
    d = Normal()
    assert float(d.ppf(0.05)) == pytest.approx(-1.6448536269514722, abs=1e-12)
    assert float(d.ppf(0.01)) == pytest.approx(-2.3263478740408408, abs=1e-12)
    # ES = -phi(q)/alpha
    q = float(d.ppf(0.01))
    expected = -math.exp(-0.5 * q * q) / math.sqrt(2 * math.pi) / 0.01
    assert float(d.expected_shortfall(0.01)) == pytest.approx(expected, abs=1e-12)


def test_ged_nu_two_is_the_normal():
    z = torch.tensor([-3.0, -1.0, -0.25, 0.0, 0.5, 2.0])
    assert torch.allclose(GED(2.0).log_prob(z), Normal().log_prob(z), atol=1e-12)


def test_skew_t_with_zero_skew_is_the_student_t():
    z = torch.tensor([-4.0, -1.5, 0.0, 0.7, 3.0])
    s, t = SkewT(6.0, 0.0), StudentT(6.0)
    assert torch.allclose(s.log_prob(z), t.log_prob(z), atol=1e-12)
    for a in (0.01, 0.05, 0.10):
        assert float(s.ppf(a)) == pytest.approx(float(t.ppf(a)), abs=1e-10)
        assert float(s.expected_shortfall(a)) == pytest.approx(
            float(t.expected_shortfall(a)), abs=1e-8
        )


def test_negative_skew_fattens_the_left_tail():
    left, right = SkewT(6.0, -0.5), SkewT(6.0, 0.5)
    assert float(left.ppf(0.01)) < float(SkewT(6.0, 0.0).ppf(0.01)) < float(right.ppf(0.01))


def test_log_prob_return_includes_the_jacobian():
    d = StudentT(7.0)
    eps = torch.tensor([0.5, -1.2, 2.0])
    h = torch.tensor([0.25, 1.0, 4.0])
    expected = d.log_prob(eps / h.sqrt()) - 0.5 * torch.log(h)
    assert torch.allclose(d.log_prob_return(eps, h), expected, atol=1e-14)


def test_nll_reductions_agree():
    d = Normal()
    eps, h = torch.randn(50), torch.rand(50) + 0.5
    assert float(d.nll(eps, h, "sum")) == pytest.approx(float(d.nll(eps, h, "none").sum()))
    assert float(d.nll(eps, h, "mean")) == pytest.approx(float(d.nll(eps, h, "none").mean()))
    with pytest.raises(ValueError, match="reduction"):
        d.nll(eps, h, "median")


def test_student_t_kernel_differs_only_by_a_constant():
    """The dropped normalising terms must not depend on the variance path."""
    full, kernel = StudentT(5.0), StudentTKernel(5.0)
    eps = torch.tensor([0.3, -1.1, 2.4, -0.8])
    for h in (torch.full((4,), 0.5), torch.full((4,), 2.0), torch.rand(4) + 0.2):
        gap = full.log_prob_return(eps, h) - kernel.log_prob_return(eps, h)
        assert torch.allclose(gap, gap[0].expand_as(gap), atol=1e-12)


def test_student_t_rejects_df_at_or_below_two():
    with pytest.raises(ValueError, match="exceed 2"):
        StudentT(2.0)


def test_skew_t_rejects_out_of_range_lambda():
    with pytest.raises(ValueError, match="lambda"):
        SkewT(6.0, 1.0)


def test_var_and_es_scale_with_sigma():
    d = StudentT(6.0)
    mu = torch.tensor([0.01, 0.01])
    sigma = torch.tensor([1.0, 3.0])
    var = d.var(0.05, mu, sigma)
    assert float(var[1] - mu[1]) == pytest.approx(3.0 * float(var[0] - mu[0]), rel=1e-12)


def test_registry_lookup_and_error():
    assert get_distribution("student-t", df=5.0).name == "student-t"
    assert get_distribution("Normal").name == "normal"
    with pytest.raises(KeyError, match="Unknown distribution"):
        get_distribution("cauchy")
