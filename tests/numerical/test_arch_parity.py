"""Bit-parity with the ``arch`` package.

This is the Phase 1 gate from ``docs/02_ARCHITECTURE.md``: a ``garchai`` cell and
an ``arch`` volatility process, given identical parameters and identical data,
must produce the *same* conditional-variance path — not merely one that agrees
after a burn-in.

Getting this to hold required matching three conventions exactly, each of which
was verified against the source rather than assumed:

* the backcast weights (``lambda = 0.94``, truncated at 75 observations),
* the pre-sample step (``arch`` recurses once on the backcast rather than using
  it as ``h_1`` directly),
* and, for EGARCH, that the backcast is taken on the log scale.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from garchai.cells import EGARCHCell, GARCHCell, GJRCell

torch.set_default_dtype(torch.float64)

arch_univariate = pytest.importorskip("arch.univariate", reason="requires the `arch` package")
ArchGARCH = arch_univariate.GARCH
ArchEGARCH = arch_univariate.EGARCH

TOL = 1e-12


@pytest.fixture(params=[500, 1500, 4000], ids=lambda n: f"n{n}")
def resids(request) -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.standard_normal(request.param) * 1.2


def _arch_path(process, params: list[float], e: np.ndarray) -> np.ndarray:
    sigma2 = np.zeros(e.shape[0])
    process.compute_variance(
        np.array(params), e, sigma2, process.backcast(e), process.variance_bounds(e)
    )
    return sigma2


@pytest.mark.external
@pytest.mark.parametrize(
    "omega,alpha,beta",
    [(0.05, 0.10, 0.85), (0.02, 0.07, 0.90), (0.10, 0.20, 0.70), (0.30, 0.05, 0.60)],
)
def test_garch_path_matches_arch(resids, omega, alpha, beta):
    reference = _arch_path(ArchGARCH(p=1, o=0, q=1), [omega, alpha, beta], resids)
    ours, _ = GARCHCell(omega, alpha, beta, init_method="arch_backcast").filter(
        torch.tensor(resids)
    )
    assert np.abs(reference - ours.detach().numpy()).max() < TOL


@pytest.mark.external
@pytest.mark.parametrize(
    "omega,alpha,gamma,beta",
    [(0.05, 0.05, 0.08, 0.85), (0.02, 0.02, 0.15, 0.80), (0.08, 0.03, 0.20, 0.75)],
)
def test_gjr_path_matches_arch(resids, omega, alpha, gamma, beta):
    reference = _arch_path(ArchGARCH(p=1, o=1, q=1), [omega, alpha, gamma, beta], resids)
    ours, _ = GJRCell(omega, alpha, gamma, beta, init_method="arch_backcast").filter(
        torch.tensor(resids)
    )
    assert np.abs(reference - ours.detach().numpy()).max() < TOL


@pytest.mark.external
@pytest.mark.parametrize(
    "omega,alpha,gamma,beta",
    [(-0.05, 0.15, -0.08, 0.97), (-0.02, 0.10, -0.05, 0.99), (0.01, 0.20, 0.05, 0.90)],
)
def test_egarch_path_matches_arch(resids, omega, alpha, gamma, beta):
    reference = _arch_path(ArchEGARCH(p=1, o=1, q=1), [omega, alpha, gamma, beta], resids)
    ours, _ = EGARCHCell(omega, alpha, gamma, beta, init_method="arch_backcast").filter(
        torch.tensor(resids)
    )
    assert np.abs(reference - ours.detach().numpy()).max() < TOL


@pytest.mark.external
def test_backcast_matches_arch_exactly(resids):
    reference = ArchGARCH(p=1, q=1).backcast(resids)
    ours = GARCHCell().arch_backcast(torch.tensor(resids))
    assert float(ours) == pytest.approx(reference, abs=1e-15)


@pytest.mark.external
def test_egarch_backcast_is_the_log_of_the_variance_backcast(resids):
    """``arch``'s EGARCH backcast is ``log`` of the GARCH one; ours must agree."""
    variance_backcast = ArchGARCH(p=1, q=1).backcast(resids)
    egarch_backcast = ArchEGARCH(p=1, o=1, q=1).backcast(resids)
    assert egarch_backcast == pytest.approx(np.log(variance_backcast), abs=1e-14)

    cell = EGARCHCell(-0.05, 0.15, -0.08, 0.97)
    h1 = cell.h1_from_backcast(cell.arch_backcast(torch.tensor(resids)))
    expected = np.exp(-0.05 + 0.97 * egarch_backcast)
    assert float(h1.detach()) == pytest.approx(expected, rel=1e-13)


@pytest.mark.external
def test_tsgarch_convention_differs_from_arch(resids):
    """The two packages genuinely disagree; garchai must not silently pick one."""
    e = torch.tensor(resids)
    cell = GARCHCell(0.05, 0.10, 0.85)
    assert float(cell.arch_backcast(e)) != pytest.approx(float(cell.tsgarch_backcast(e)), rel=1e-3)
