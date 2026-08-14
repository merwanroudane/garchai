"""GENSHIN (Yu et al. 2025): VMD, the component GARCH, and MSGNet.

GENSHIN's contribution is mostly in feature construction, so most of what needs
testing is upstream of the network: VMD must actually separate frequencies, the
component GARCH must actually separate timescales, and the narrowing shortcut
must be recorded rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from garchai.cells import ComponentGARCHCell
from garchai.core import TrainConfig, set_seed
from garchai.graph import (
    DISTRIBUTIONS,
    GARCHFeatureBank,
    GENSHIN,
    LAG_STRUCTURES,
    MSGNet,
    SPECIFICATIONS,
    variational_mode_decomposition,
    vmd_reconstruct,
)

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
# VMD
# --------------------------------------------------------------------------- #
def test_vmd_separates_known_frequencies():
    """The decisive test: put in tones, get the tones back."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    signal = np.cos(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 40 * t)
    modes, frequencies = variational_mode_decomposition(signal, n_modes=2, alpha=2000)
    assert modes.shape == (2, 1000)
    recovered = sorted(frequencies * 1000)
    assert recovered[0] == pytest.approx(5.0, abs=1.5)
    assert recovered[1] == pytest.approx(40.0, abs=1.5)


def test_vmd_modes_are_ordered_by_frequency():
    t = np.linspace(0, 1, 800, endpoint=False)
    signal = sum(np.cos(2 * np.pi * f * t) for f in (3, 25, 90))
    _, frequencies = variational_mode_decomposition(signal, n_modes=3)
    assert np.all(np.diff(frequencies) > 0)


def test_vmd_energy_shares_track_amplitudes():
    t = np.linspace(0, 1, 1000, endpoint=False)
    signal = np.cos(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 40 * t)
    modes, _ = variational_mode_decomposition(signal, n_modes=2, alpha=2000)
    shares = [(m**2).sum() / (signal**2).sum() for m in modes]
    assert shares[0] > shares[1]                 # the larger tone carries more
    assert shares[0] / shares[1] == pytest.approx(4.0, rel=0.5)   # amplitude squared


def test_vmd_reconstruction_is_close_but_lossy():
    """With tau = 0 the residual is the noise VMD declined to assign."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 500, endpoint=False)
    signal = np.cos(2 * np.pi * 7 * t) + 0.05 * rng.standard_normal(500)
    modes, _ = variational_mode_decomposition(signal, n_modes=2)
    error = np.abs(signal - vmd_reconstruct(modes)).max()
    assert error < 0.5 * np.abs(signal).max()


def test_vmd_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 4 samples"):
        variational_mode_decomposition([1.0, 2.0])
    with pytest.raises(ValueError, match="n_modes must be at least 1"):
        variational_mode_decomposition(np.random.randn(100), n_modes=0)
    with pytest.raises(ValueError, match="Unknown init"):
        variational_mode_decomposition(np.random.randn(100), init="clever")


# --------------------------------------------------------------------------- #
# component GARCH
# --------------------------------------------------------------------------- #
def test_component_garch_enforces_the_ordering_of_timescales():
    """alpha + beta < rho: the transitory part must decay faster."""
    cell = ComponentGARCHCell(omega=0.01, rho=0.99, phi=0.05, alpha=0.06, beta=0.80)
    p = {k: float(v.detach()) for k, v in cell.transformed_parameters().items()}
    assert p["alpha"] + p["beta"] < p["rho"]


def test_component_garch_rejects_an_inverted_ordering():
    with pytest.raises(ValueError, match="alpha \\+ beta < rho"):
        ComponentGARCHCell(rho=0.90, alpha=0.50, beta=0.50)


def test_component_garch_persistence_is_the_permanent_one():
    """Reporting alpha+beta here would understate how long shocks last."""
    cell = ComponentGARCHCell(rho=0.99, alpha=0.06, beta=0.80)
    assert float(cell.persistence().detach()) == pytest.approx(0.99, abs=1e-3)
    assert float(cell.transitory_persistence().detach()) == pytest.approx(0.86, abs=1e-2)
    assert float(cell.persistence().detach()) > float(cell.transitory_persistence().detach())


def test_component_garch_decomposition_sums_back():
    set_seed(0)
    cell = ComponentGARCHCell()
    eps = torch.randn(300) * 0.9
    parts = cell.decompose(eps)
    assert torch.allclose(
        parts["permanent"] + parts["transitory"], parts["variance"], atol=1e-12
    )


def test_transitory_component_can_go_negative():
    """Volatility below its slow level - what a one-component GARCH cannot say."""
    set_seed(0)
    cell = ComponentGARCHCell()
    eps = torch.randn(400) * 0.9
    eps[200:215] *= 5.0
    parts = cell.decompose(eps)
    assert bool((parts["transitory"] < 0).any())


def test_component_garch_variance_is_valid_and_differentiable():
    from garchai.distributions import Normal

    set_seed(0)
    cell = ComponentGARCHCell()
    eps = torch.randn(200) * 0.9
    h, state = cell.filter(eps)
    assert bool((h > 0).all()) and bool(torch.isfinite(h).all())
    assert "q" in (state.extra or {})
    Normal().nll(eps, h).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters()
    )


def test_component_long_run_variance_uses_rho():
    cell = ComponentGARCHCell(omega=0.01, rho=0.99)
    p = cell.transformed_parameters()
    expected = float(p["omega"].detach()) / (1.0 - float(p["rho"].detach()))
    assert float(cell.unconditional_variance().detach()) == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# MSGNet
# --------------------------------------------------------------------------- #
def test_msgnet_forward_shape():
    set_seed(0)
    net = MSGNet(n_series=6, sequence_length=20, d_model=16, n_scales=3)
    out = net(torch.randn(8, 20, 6))
    assert out.shape == (8, 1)


def test_adaptive_adjacency_is_row_normalised():
    set_seed(0)
    net = MSGNet(n_series=6, sequence_length=20, d_model=16)
    adjacency = net._graph(0).detach()
    assert torch.allclose(adjacency.sum(dim=-1), torch.ones(adjacency.shape[0]), atol=1e-12)
    assert bool((adjacency >= 0).all())


def test_adjacency_is_soft_not_pruned():
    """ReLU floors the logits; the softmax still gives every pair positive weight.

    Guards the docstring against re-acquiring the claim that ReLU sparsifies the
    graph - it does not, and reading it that way overstates the layer.
    """
    set_seed(0)
    net = MSGNet(n_series=6, sequence_length=20, d_model=16)
    adjacency = net._graph(0).detach()
    assert bool((adjacency > 0).all())


def test_fft_detects_a_planted_period():
    set_seed(0)
    net = MSGNet(n_series=1, sequence_length=64, d_model=8, n_scales=1)
    t = torch.arange(64, dtype=torch.float64)
    signal = torch.sin(2 * torch.pi * t / 16).reshape(1, 64, 1)
    periods, amplitudes = net._periods(net.embed(signal), 1)
    assert periods[0] == pytest.approx(16, abs=6)
    assert float(amplitudes[0].detach()) > 0


def test_msgnet_gradients_flow():
    set_seed(0)
    net = MSGNet(n_series=4, sequence_length=16, d_model=12)
    net(torch.randn(4, 16, 4)).sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
def test_full_grid_is_sixty_models():
    """5 specifications x 4 lag structures x 3 distributions (paper sec. 2.1)."""
    assert len(SPECIFICATIONS) * len(LAG_STRUCTURES) * len(DISTRIBUTIONS) == 60
    assert GARCHFeatureBank().n_models == 60
    assert GENSHIN().bank.n_models == 60


def test_narrowing_the_grid_is_recorded_not_hidden():
    """A speed shortcut that changes the model must be visible in the output."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(300) * 0.9
    target = np.abs(returns) + 0.1

    model = GENSHIN(
        n_modes=2, n_components=3, sequence_length=10,
        specifications=("sgarch",), lags=((1, 1),), distributions=("normal",),
    )
    features = model.build_features(returns, target)
    assert features.narrowed is True
    assert "narrowed" in features.note.lower()
    assert features.n_models_fitted == 1


def test_pipeline_runs_end_to_end():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(400) * 0.9
    target = np.abs(returns) + 0.1

    model = GENSHIN(
        n_modes=2, n_components=3, sequence_length=10,
        specifications=("sgarch", "gjr"), lags=((1, 1),), distributions=("normal",),
    )
    features = model.build_features(returns, target)
    assert features.components.shape == (400, 3)
    model.fit(features, TrainConfig(lr=1e-3, max_epochs=10, patience=5))
    metrics = model.evaluate(features)
    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_predict_requires_a_fitted_network():
    model = GENSHIN(specifications=("sgarch",), lags=((1, 1),), distributions=("normal",))
    rng = np.random.default_rng(0)
    features = model.build_features(
        rng.standard_normal(200) * 0.9, np.abs(rng.standard_normal(200)) + 0.1
    )
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(features)


def test_features_reject_misaligned_input():
    model = GENSHIN(specifications=("sgarch",), lags=((1, 1),), distributions=("normal",))
    with pytest.raises(ValueError, match="must align"):
        model.build_features(np.random.randn(100), np.random.randn(90))


def test_provenance_declares_the_data_substitution():
    substitutions = " ".join(GENSHIN.provenance.data_substitutions)
    assert "Chinese" in substitutions
    text = " ".join(GENSHIN.provenance.deviations)
    assert "csGARCH" in text
    assert GENSHIN.provenance.paper_id == "yu2025_genshin"
