"""Feature hybrids and ensembles: Hu 2020, Maingo 2025, Rubio 2026, Gebreslase 2026.

This family's characteristic failure is look-ahead — an in-sample GARCH fit used
as a feature, or random K-fold stacking on time-ordered data — which produces
spectacular accuracy that vanishes out of sample. Most of the tests here are
about that rather than about accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from garchai.core import TrainConfig, set_seed
from garchai.ensembles import (
    GARCHFeatures,
    GARCHXGBoost,
    LSTMANNGARCH,
    MultiGARCHTransformer,
    StackedMLGARCH,
    build_garch_features,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def features() -> GARCHFeatures:
    rng = np.random.default_rng(0)
    n, omega, alpha, beta = 900, 0.05, 0.09, 0.88
    h = np.zeros(n)
    e = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    target = np.sqrt(h)
    return build_garch_features(
        e, target, specifications=("garch",), window=300, refit_every=100, n_lags=3
    )


# --------------------------------------------------------------------------- #
# feature construction — where look-ahead would enter
# --------------------------------------------------------------------------- #
def test_features_are_finite_and_aligned(features):
    assert features.features.shape[0] == features.target.shape[0]
    assert np.isfinite(features.features).all()
    assert np.isfinite(features.target).all()
    assert features.features.shape[1] == len(features.names)


def test_refit_shortcut_is_declared(features):
    """A speed shortcut that changes the protocol must be visible."""
    assert features.refit_every == 100
    assert "approximation" in features.note


def test_exact_protocol_says_nothing_about_approximation():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(400) * 0.9
    exact = build_garch_features(
        r, np.abs(r) + 0.1, specifications=("garch",), window=200, refit_every=1, n_lags=2
    )
    assert exact.refit_every == 1
    assert "approximation" not in exact.note


def test_split_is_chronological(features):
    """Random splitting would leak the future into the training set."""
    train, test = features.split(0.8)
    assert len(train) + len(test) == len(features)
    assert np.allclose(train.features[0], features.features[0])
    assert np.allclose(test.features[-1], features.features[-1])


def test_features_reject_misaligned_input():
    with pytest.raises(ValueError, match="must align"):
        build_garch_features(np.random.randn(200), np.random.randn(190))


def test_features_reject_too_short_a_series():
    with pytest.raises(ValueError, match="Need more than"):
        build_garch_features(np.random.randn(50), np.abs(np.random.randn(50)), window=100)


# --------------------------------------------------------------------------- #
# the sequence hybrids
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bidirectional", [False, True])
def test_lstm_ann_garch_trains_and_predicts(features, bidirectional):
    train, test = features.split(0.8)
    set_seed(0)
    model = LSTMANNGARCH(
        features.features.shape[1], window=10, hidden_size=16,
        bidirectional=bidirectional,
    )
    history = model.fit_features(train, TrainConfig(lr=1e-3, max_epochs=20, patience=8))
    assert history["loss"][-1] < history["loss"][0]
    prediction = model.predict(test)
    assert prediction.shape == test.target.shape
    assert np.all(prediction > 0)          # softplus output


def test_multi_garch_transformer_trains(features):
    train, test = features.split(0.8)
    set_seed(0)
    model = MultiGARCHTransformer(features.features.shape[1], window=10, d_model=16)
    history = model.fit_features(train, TrainConfig(lr=1e-3, max_epochs=20, patience=8))
    assert history["loss"][-1] < history["loss"][0]
    assert np.all(model.predict(test) > 0)


def test_windowing_keeps_alignment(features):
    """Left-padding must preserve one prediction per observation."""
    set_seed(0)
    model = LSTMANNGARCH(features.features.shape[1], window=15, hidden_size=8)
    x = model._reshape(model._standardise(features.features, fit=True))
    assert x.shape[0] == features.features.shape[0]
    assert x.shape[1] == 15


def test_sequence_models_report_the_standard_metrics(features):
    train, test = features.split(0.8)
    set_seed(0)
    model = LSTMANNGARCH(features.features.shape[1], window=10, hidden_size=8)
    model.fit_features(train, TrainConfig(max_epochs=5, patience=3))
    metrics = model.evaluate(test)
    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert all(np.isfinite(v) for v in metrics.values())


# --------------------------------------------------------------------------- #
# residual correction
# --------------------------------------------------------------------------- #
def test_xgboost_fits_the_residual_not_the_target(features):
    """Fitting the target directly would let the booster relearn the GARCH signal."""
    train, _ = features.split(0.8)
    model = GARCHXGBoost().fit(train)
    baseline = train.features[:, 0]
    fitted_correction = model.model.predict(train.features)
    # The correction should be small relative to the level it corrects.
    assert np.mean(np.abs(fitted_correction)) < np.mean(np.abs(baseline))


def test_xgboost_reports_its_own_baseline(features):
    """The gain over plain GARCH must be visible, not implied."""
    train, test = features.split(0.8)
    metrics = GARCHXGBoost().fit(train).evaluate(test)
    assert "baseline_RMSE" in metrics
    assert np.isfinite(metrics["baseline_RMSE"])


def test_xgboost_predictions_are_positive(features):
    train, test = features.split(0.8)
    assert np.all(GARCHXGBoost().fit(train).predict(test) > 0)


def test_xgboost_records_which_backend_it_used():
    assert GARCHXGBoost().backend in ("xgboost", "sklearn")


# --------------------------------------------------------------------------- #
# stacking
# --------------------------------------------------------------------------- #
def test_stacking_uses_blocked_out_of_fold_predictions(features):
    """Random K-fold on time-ordered data is look-ahead; blocks are not."""
    train, _ = features.split(0.8)
    model = StackedMLGARCH(n_folds=4)
    oof = model._out_of_fold(train.features, train.target)
    assert oof.shape == (len(train), len(model.base_models))
    # The first block has no earlier data to train on, so it must be excluded.
    assert np.isnan(oof[0]).all()
    assert np.isfinite(oof[-1]).all()


def test_stacking_reports_whether_it_beat_its_best_base(features):
    """An ensemble that cannot beat its best member is not adding anything."""
    train, test = features.split(0.8)
    metrics = StackedMLGARCH(n_folds=4).fit(train).evaluate(test)
    assert "best_base_RMSE" in metrics
    assert "beats_best_base" in metrics
    assert isinstance(metrics["beats_best_base"], bool)
    assert metrics["beats_best_base"] == (metrics["RMSE"] < metrics["best_base_RMSE"])


def test_stacking_predictions_are_positive(features):
    train, test = features.split(0.8)
    assert np.all(StackedMLGARCH(n_folds=4).fit(train).predict(test) > 0)


def test_stacking_accepts_custom_base_models(features):
    from sklearn.linear_model import Ridge

    train, test = features.split(0.8)
    model = StackedMLGARCH(base_models={"ridge": Ridge()}, n_folds=4).fit(train)
    metrics = model.evaluate(test)
    assert "base_ridge_RMSE" in metrics


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls,paper_id",
    [
        (LSTMANNGARCH, "hu2020_lstm_ann_garch"),
        (GARCHXGBoost, "maingo2025_statistical_ml_fusion"),
        (StackedMLGARCH, "rubio2026_stacked_ml_garch"),
        (MultiGARCHTransformer, "gebreslase2026_multi_garch_transformer"),
    ],
)
def test_provenance_is_declared(cls, paper_id):
    assert cls.provenance.paper_id == paper_id
    assert cls.provenance.implementation_status.value == "REPRODUCTION"


def test_stacking_declares_the_fold_deviation():
    text = " ".join(StackedMLGARCH.provenance.deviations)
    assert "look-ahead" in text
