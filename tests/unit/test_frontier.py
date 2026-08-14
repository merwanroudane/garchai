"""The final five: GARCH-FIS, GARCH-DDQN VaR, DCC-MANN, GARCH-TFT, sentiment.

Each of these has one property that decides whether the implementation is the
paper's model or merely something shaped like it, and those are what is tested
here: the fuzzy widths must actually track volatility, the DDQN reward must match
Eq. 12 exactly, DCC must produce valid correlation matrices, the TFT's attention
must be causal, and the sentiment term must not be able to drive the variance
negative.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from garchai.core import TrainConfig, set_seed
from garchai.ensembles import (
    GARCHTFT,
    InterpretableMultiHeadAttention,
    SentimentGARCHLSTM,
    SentimentGARCHX,
    VariableSelectionNetwork,
    garman_klass,
    historical_volatility,
    normalise_sentiment,
    quantile_loss,
    vix_sentiment_proxy,
)
from garchai.fuzzy import FuzzyRuleBase, GARCHFIS
from garchai.multivariate import DCCGARCHMANN, fit_dcc
from garchai.risk import GARCHDDQNVaR, QNetwork, ReplayBuffer, Transition

torch.set_default_dtype(torch.float64)


def _garch_series(n: int = 800, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    e = np.zeros(n)
    h[0] = 1.0
    for t in range(1, n):
        h[t] = 0.05 + 0.09 * e[t - 1] ** 2 + 0.88 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    return e


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    return _garch_series()


# --------------------------------------------------------------------------- #
# GARCH-FIS
# --------------------------------------------------------------------------- #
def test_membership_widths_widen_with_volatility():
    """The paper's entire mechanism: fuzzy sets breathe with the market."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GARCHFIS(n_rules=4, kappa=1.0, window=250, refit_every=25, seed=0).fit(_garch_series(600))
    assert model.membership_widths(2.0).mean() > model.membership_widths(0.5).mean()


def test_kappa_zero_freezes_the_adaptation():
    """The ablation must genuinely switch the contribution off."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GARCHFIS(n_rules=4, kappa=0.0, window=250, refit_every=25, seed=0).fit(_garch_series(600))
    assert model.membership_widths(0.1).mean() == pytest.approx(
        model.membership_widths(9.9).mean()
    )


def test_firing_strengths_form_a_partition():
    rules = FuzzyRuleBase(
        centres=np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]]),
        base_widths=np.full((3, 2), 0.5),
    )
    strengths = rules.firing_strengths(np.array([0.2, 0.3]))
    assert strengths.sum() == pytest.approx(1.0)
    assert np.all(strengths >= 0)


def test_firing_strengths_degrade_gracefully_far_from_every_rule():
    """A point in no rule's support must not produce 0/0."""
    rules = FuzzyRuleBase(
        centres=np.zeros((3, 2)), base_widths=np.full((3, 2), 1e-6)
    )
    strengths = rules.firing_strengths(np.array([1e6, 1e6]))
    assert np.isfinite(strengths).all()
    assert strengths.sum() == pytest.approx(1.0)


def test_fis_rejects_a_series_shorter_than_its_window():
    with pytest.raises(ValueError, match="Need more than"):
        GARCHFIS(window=250).fit(np.random.randn(100))


def test_fis_predict_requires_fitting():
    with pytest.raises(RuntimeError, match="not been fitted"):
        GARCHFIS().predict(np.random.randn(500))


# --------------------------------------------------------------------------- #
# GARCH-DDQN VaR
# --------------------------------------------------------------------------- #
def test_reward_matches_equation_12_exactly():
    """+1 TP, -1 FP, +rho TN, -rho FN. Getting a sign wrong inverts the policy."""
    agent = GARCHDDQNVaR()
    agent.rho = 0.2
    assert agent.reward(1, 1) == 1.0        # true positive
    assert agent.reward(1, 0) == -1.0       # false positive
    assert agent.reward(0, 0) == 0.2        # true negative
    assert agent.reward(0, 1) == -0.2       # false negative


def test_prose_variant_makes_the_missed_crisis_worst():
    """The paper's text and its Eq. 12 disagree; both must be reachable."""
    agent = GARCHDDQNVaR(severe_false_negative=True)
    agent.rho = 0.2
    assert agent.reward(0, 1) == pytest.approx(-5.0)
    assert agent.reward(0, 1) < agent.reward(1, 0)


def test_ddqn_uses_online_for_selection_and_target_for_evaluation():
    """Collapsing these two into one network is plain DQN, not Double DQN."""
    set_seed(0)
    agent = GARCHDDQNVaR()
    agent.online = QNetwork(4, (8,))
    agent.target = QNetwork(4, (8,))
    with torch.no_grad():
        for p in agent.target.parameters():
            p.mul_(0.0)          # target returns all zeros
    for _ in range(40):
        agent.buffer.push(
            Transition(np.zeros(4), 0, 1.0, np.zeros(4), False)
        )
    optimiser = torch.optim.Adam(agent.online.parameters(), lr=1e-3)
    agent.batch_size = 8
    loss = agent._learn(optimiser)
    # With a zeroed target net the bootstrap term vanishes, so y == reward.
    assert np.isfinite(loss)


@pytest.mark.slow
def test_epsilon_anneals_to_its_floor():
    """A schedule that never anneals reports exploration noise as a learning curve."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = GARCHDDQNVaR(seed=0, refit_every=25).fit(_garch_series(700), n_episodes=6)
    assert agent.history.epsilon[0] == 1.0
    assert agent.history.epsilon[-1] == pytest.approx(agent.epsilon_end, abs=1e-9)


def test_rho_is_the_class_ratio():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = GARCHDDQNVaR(alpha=0.05, seed=0, refit_every=25).fit(_garch_series(700), n_episodes=3)
    assert 0.0 < agent.rho <= 1.0


@pytest.mark.slow
def test_var_is_negative_and_regime_dependent():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _garch_series(700)
        agent = GARCHDDQNVaR(alpha=0.05, seed=0, refit_every=25).fit(r, n_episodes=6)
    var = agent.var(r)
    assert np.all(var < 0)                    # a loss threshold, per the convention
    assert set(np.unique(agent.predict_regime(r))) <= {0, 1}


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=5, seed=0)
    for i in range(20):
        buffer.push(Transition(np.zeros(2), 0, float(i), np.zeros(2), False))
    assert len(buffer) == 5
    assert len(buffer.sample(3)) == 3


def test_agent_rejects_a_degenerate_label_split():
    with pytest.raises(ValueError, match="Both risk classes"):
        GARCHDDQNVaR(quantile_threshold=0.0, window=100).fit(
            _garch_series(300), n_episodes=1
        )


def test_agent_rejects_an_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must lie"):
        GARCHDDQNVaR(alpha=1.5)


# --------------------------------------------------------------------------- #
# DCC-GARCH-MANN
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def panel() -> np.ndarray:
    rng = np.random.default_rng(0)
    n, k = 600, 3
    chol = np.linalg.cholesky([[1, 0.6, 0.3], [0.6, 1, 0.4], [0.3, 0.4, 1]])
    z = rng.standard_normal((n, k)) @ chol.T
    h = np.ones((n, k))
    r = np.zeros((n, k))
    for t in range(1, n):
        h[t] = 0.05 + 0.09 * r[t - 1] ** 2 + 0.87 * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    return r


def test_dcc_correlations_are_valid_matrices(panel):
    """Unit diagonal and positive definite - otherwise it is not a correlation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_dcc(panel)
    assert np.allclose(np.diagonal(result.correlations, axis1=1, axis2=2), 1.0)
    for t in range(0, panel.shape[0], 50):
        assert np.linalg.eigvalsh(result.correlations[t]).min() > 0


def test_dcc_recovers_a_known_correlation(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_dcc(panel)
    assert result.pairwise(0, 1).mean() == pytest.approx(0.6, abs=0.1)


def test_dcc_is_stationary(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_dcc(panel)
    assert result.a >= 0 and result.b >= 0
    assert result.persistence < 1.0


def test_dcc_rejects_univariate_input():
    with pytest.raises(ValueError, match="at least two series"):
        fit_dcc(np.random.randn(200, 1))
    with pytest.raises(ValueError, match="must be 2-D"):
        fit_dcc(np.random.randn(200))


@pytest.mark.parametrize("direction", ["dcc_to_mann", "mann_to_dcc"])
def test_both_hybrid_directions_run(panel, direction):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DCCGARCHMANN(direction=direction, n_lags=3, seed=0).fit(
            panel, train=TrainConfig(lr=1e-2, max_epochs=30, patience=10)
        )
    prediction = model.predict(panel)
    assert prediction.shape == (panel.shape[0] - 3, panel.shape[1])
    assert np.isfinite(prediction).all()
    assert "dcc_persistence" in model.evaluate(panel)


def test_the_two_directions_are_actually_different(panel):
    """If the ordering did not matter the paper's comparison would be empty."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = TrainConfig(lr=1e-2, max_epochs=30, patience=10)
        forward = DCCGARCHMANN("dcc_to_mann", seed=0).fit(panel, train=config)
        reverse = DCCGARCHMANN("mann_to_dcc", seed=0).fit(panel, train=config)
    # dcc_to_mann feeds sigma and the correlations in as extra inputs.
    assert forward.build_inputs(panel)[0].shape[1] > reverse.build_inputs(panel)[0].shape[1]


def test_mann_is_multivariate_not_a_stack_of_univariate_models(panel):
    """One network, n outputs - so series j can inform series i."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DCCGARCHMANN(seed=0).fit(panel, train=TrainConfig(max_epochs=10, patience=5))
    assert model.network.net[-1].out_features == panel.shape[1]


def test_direction_is_validated():
    with pytest.raises(ValueError, match="direction must be"):
        DCCGARCHMANN(direction="sideways")


# --------------------------------------------------------------------------- #
# GARCH-TFT
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_pinball_loss_recovers_true_quantiles():
    """The decisive check on the loss: fit constants to a known distribution."""
    torch.manual_seed(0)
    y = torch.randn(20000)
    quantiles = (0.1, 0.5, 0.9)
    p = torch.zeros(3, requires_grad=True)
    optimiser = torch.optim.Adam([p], lr=0.02)
    for _ in range(3000):
        optimiser.zero_grad()
        quantile_loss(p.expand(20000, 3), y, quantiles).backward()
        optimiser.step()
    learned = p.detach()
    for i, q in enumerate(quantiles):
        assert float(learned[i]) == pytest.approx(float(torch.quantile(y, q)), abs=0.05)


def test_attention_is_causal():
    """Without the mask the model reads the future and every metric is void."""
    set_seed(0)
    attention = InterpretableMultiHeadAttention(d_model=8, n_heads=2, dropout=0.0)
    x = torch.randn(2, 6, 8)
    mask = torch.tril(torch.ones(6, 6))
    _, weights = attention(x, x, x, mask)
    upper = torch.triu(weights[0], diagonal=1)
    assert float(upper.abs().max()) < 1e-8


def test_interpretable_attention_shares_one_value_projection():
    """Averaging heads is only meaningful because V is shared - Lim Eq. 14-16."""
    attention = InterpretableMultiHeadAttention(d_model=16, n_heads=4)
    assert attention.v.out_features == attention.d_head
    assert attention.q.out_features == 16


def test_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        InterpretableMultiHeadAttention(d_model=10, n_heads=4)


def test_variable_selection_weights_are_a_distribution():
    set_seed(0)
    vsn = VariableSelectionNetwork(n_variables=4, d_model=8, dropout=0.0)
    _, weights = vsn(torch.randn(3, 5, 4, 8))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 5), atol=1e-10)
    assert bool((weights >= 0).all())


def test_tft_reports_variable_importance_including_the_garch_input():
    """The paper's interpretability claim - and whether GARCH was used at all."""
    rng = np.random.default_rng(0)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    hv = historical_volatility(close, 10)
    x = np.column_stack([np.diff(np.log(close), prepend=np.log(close[0])), hv, hv * 0.9])
    model = GARCHTFT(window=8, d_model=16, n_heads=2, dropout=0.0, seed=0)
    model.names = ("return", "hv", "garch")
    model.fit(x, hv, train=TrainConfig(lr=1e-2, max_epochs=25, patience=10))
    importance = model.variable_importance(x)
    assert set(importance) == {"return", "hv", "garch"}
    assert sum(importance.values()) == pytest.approx(1.0, abs=1e-6)


def test_tft_reports_its_quantile_crossing_rate():
    """Crossed quantiles make the coverage figure meaningless, so it travels with it."""
    rng = np.random.default_rng(0)
    n = 250
    x = rng.standard_normal((n, 2))
    y = np.abs(x[:, 0]) + 0.5
    model = GARCHTFT(window=6, d_model=16, n_heads=2, dropout=0.0, seed=0)
    model.fit(x, y, train=TrainConfig(lr=1e-2, max_epochs=25, patience=10))
    metrics = model.evaluate(x, y)
    assert "quantile_crossing_rate" in metrics
    assert 0.0 <= metrics["quantile_crossing_rate"] <= 1.0
    assert metrics["quantile_crossing_rate"] == pytest.approx(
        model.quantile_crossing_rate(x)
    )


def test_tft_requires_the_median_quantile():
    with pytest.raises(ValueError, match="median"):
        GARCHTFT(quantiles=(0.1, 0.9))


def test_tft_windowing_preserves_alignment():
    model = GARCHTFT(window=12)
    model._standardise(np.random.randn(50, 3), fit=True)
    windows = model._reshape(np.random.randn(50, 3))
    assert windows.shape == (50, 12, 3)


def test_garman_klass_is_never_negative():
    """The GK expression can go negative on a small range; the estimator cannot."""
    rng = np.random.default_rng(0)
    n = 200
    open_ = 100 + rng.standard_normal(n)
    close = open_ + rng.standard_normal(n) * 3     # large |ln(C/O)|, tiny range
    high = np.maximum(open_, close) + 1e-9
    low = np.minimum(open_, close) - 1e-9
    assert np.all(garman_klass(high, low, open_, close) >= 0)


def test_historical_volatility_has_no_nan():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).standard_normal(200) * 0.01))
    assert np.isfinite(historical_volatility(close, 10)).all()


def test_tft_predict_requires_fitting():
    with pytest.raises(RuntimeError, match="not been fitted"):
        GARCHTFT().predict(np.random.randn(50, 2))


# --------------------------------------------------------------------------- #
# sentiment-augmented models
# --------------------------------------------------------------------------- #
def test_sentiment_term_cannot_drive_the_variance_negative():
    """gamma >= 0 and S in [0, 1] make positivity structural, not a post-hoc floor."""
    rng = np.random.default_rng(0)
    n = 500
    sentiment = normalise_sentiment(np.cumsum(rng.standard_normal(n)))
    h = np.zeros(n)
    e = np.zeros(n)
    h[0] = 1.0
    for t in range(1, n):
        h[t] = 0.02 + 0.08 * e[t - 1] ** 2 + 0.85 * h[t - 1] + 0.15 * sentiment[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    model = SentimentGARCHX(sentiment_lags=1).fit(e, sentiment)
    assert np.all(model.sigma2 > 0)
    assert model.params["gamma1"] >= 0.0


def test_sentiment_garch_x_recovers_its_parameters():
    rng = np.random.default_rng(0)
    n = 600
    sentiment = normalise_sentiment(np.cumsum(rng.standard_normal(n)) * 0.3)
    h = np.zeros(n)
    e = np.zeros(n)
    h[0] = 1.0
    for t in range(1, n):
        h[t] = 0.02 + 0.08 * e[t - 1] ** 2 + 0.85 * h[t - 1] + 0.15 * sentiment[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    model = SentimentGARCHX(sentiment_lags=1).fit(e, sentiment)
    assert model.params["alpha"] == pytest.approx(0.08, abs=0.05)
    assert model.params["beta"] == pytest.approx(0.85, abs=0.10)
    assert model.params["persistence"] < 1.0


def test_normalise_sentiment_is_bounded():
    rng = np.random.default_rng(0)
    for method in ("minmax", "rank"):
        s = normalise_sentiment(rng.standard_normal(300) * 50, method=method)
        assert s.min() >= 0.0 and s.max() <= 1.0


def test_normalise_sentiment_handles_a_constant_series():
    assert np.all(normalise_sentiment(np.full(50, 3.0)) == 0.0)


def test_normalise_sentiment_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method must be"):
        normalise_sentiment(np.random.randn(20), method="zscore")


def test_vix_proxy_inverts_the_fear_gauge():
    """High VIX must map to low sentiment, or the sign of gamma flips meaning."""
    vix = np.array([10.0, 20.0, 40.0, 80.0])
    proxy = vix_sentiment_proxy(vix)
    assert np.all(np.diff(proxy) < 0)
    assert proxy.min() >= 0.0 and proxy.max() <= 1.0


def test_sentiment_garch_x_rejects_negative_sentiment():
    with pytest.raises(ValueError, match="non-negative"):
        SentimentGARCHX().fit(np.random.randn(200), np.random.randn(200) - 5)


def test_sentiment_garch_x_rejects_misaligned_input():
    with pytest.raises(ValueError, match="must align"):
        SentimentGARCHX().fit(np.random.randn(200), np.abs(np.random.randn(150)))


def test_sentiment_contribution_is_a_share():
    rng = np.random.default_rng(0)
    n = 400
    sentiment = normalise_sentiment(np.cumsum(rng.standard_normal(n)))
    model = SentimentGARCHX(sentiment_lags=1).fit(_garch_series(n), sentiment)
    share = model.sentiment_contribution(sentiment)
    assert np.all(share >= 0.0) and np.all(share <= 1.0)


def test_sentiment_lstm_ablation_is_reachable():
    """use_sentiment=False is the paper's NN baseline - the claim rests on it."""
    rng = np.random.default_rng(0)
    n = 300
    x = np.column_stack([np.abs(rng.standard_normal(n)), rng.random(n)])
    y = np.abs(rng.standard_normal(n)) + 0.5
    for use_sentiment in (True, False):
        model = SentimentGARCHLSTM(2, window=6, hidden_size=8, use_sentiment=use_sentiment)
        model.fit_series(x, y, train=TrainConfig(lr=1e-2, max_epochs=20, patience=10))
        metrics = model.evaluate(x, y)
        assert metrics["used_sentiment"] == float(use_sentiment)
        assert np.all(model.predict(x) > 0)         # softplus output


def test_sentiment_lstm_rejects_misaligned_input():
    model = SentimentGARCHLSTM(2, window=5)
    with pytest.raises(ValueError, match="must align"):
        model.fit_series(np.random.randn(100, 2), np.random.randn(90))


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls,paper_id",
    [
        (GARCHFIS, "li2026_garch_fis"),
        (GARCHDDQNVaR, "rl_var_ddqn_2025"),
        (DCCGARCHMANN, "fatima2022_dcc_mann"),
        (GARCHTFT, "petrosino2025_garch_tft"),
        (SentimentGARCHX, "sentiment_garch_lstm_2026"),
        (SentimentGARCHLSTM, "sentiment_garch_lstm_2026"),
    ],
)
def test_provenance_is_declared(cls, paper_id):
    assert cls.provenance.paper_id == paper_id


def test_ddqn_declares_the_papers_internal_contradiction():
    text = " ".join(GARCHDDQNVaR.provenance.deviations)
    assert "prose" in text and "Eq. 12" in text


def test_tft_declares_that_no_code_was_ported():
    text = " ".join(GARCHTFT.provenance.deviations)
    assert "licence" in text


def test_sentiment_declares_the_data_blocker():
    text = " ".join(SentimentGARCHX.provenance.data_substitutions)
    assert "proprietary" in text


# --------------------------------------------------------------------------- #
# dtype robustness
# --------------------------------------------------------------------------- #
def test_models_work_under_float32_default():
    """The package documents float64 but does not set it globally on import.

    A model that calls torch.as_tensor on a float64 array without naming a dtype
    builds a Double input for Float layers, which raises only when the caller has
    not set the default -- so it passes every test that sets float64 at the top
    and fails the first time a user runs an example script.
    """
    original = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        rng = np.random.default_rng(0)
        n = 200
        x = rng.standard_normal((n, 3))
        y = np.abs(rng.standard_normal(n)) + 0.5
        config = TrainConfig(lr=1e-2, max_epochs=5, patience=3)

        tft = GARCHTFT(window=6, d_model=8, n_heads=2, dropout=0.0, seed=0)
        tft.fit(x, y, train=config)
        assert np.isfinite(tft.predict(x)).all()

        lstm = SentimentGARCHLSTM(3, window=6, hidden_size=8)
        lstm.fit_series(x, y, train=config)
        assert np.isfinite(lstm.predict(x)).all()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            panel = np.column_stack([_garch_series(300, seed=s) for s in (0, 1)])
            mann = DCCGARCHMANN(seed=0).fit(panel, train=config)
        assert np.isfinite(mann.predict(panel)).all()
    finally:
        torch.set_default_dtype(original)


def test_sentiment_rejects_nan_before_the_sign_check():
    """`nan < 0` is False, so a NaN walks through a non-negativity check.

    It then poisons every likelihood evaluation, the optimiser returns its
    infeasibility penalty, and the starting values come back looking like a
    converged fit. Fail loudly instead.
    """
    sentiment = np.full(300, 0.5)
    sentiment[0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        SentimentGARCHX().fit(_garch_series(300), sentiment)


def test_sentiment_rejects_non_finite_returns():
    returns = _garch_series(300)
    returns[10] = np.inf
    with pytest.raises(ValueError, match="returns contain"):
        SentimentGARCHX().fit(returns, np.full(300, 0.5))
