r"""The frontier family on real market data: fuzzy, RL, multivariate, TFT, sentiment.

Runs the five models added last, each on real data, and writes journal-format
tables and figures to ``examples/outputs/``.

Every result here is reported with the comparison that makes it interpretable —
GARCH-FIS against its own no-adaptation ablation, GARCH-TFT against a
stand-alone TFT, the DDQN against unconditional VaR, and the sentiment model
against the ``NN`` baseline. A number without its ablation is not evidence about
the mechanism the paper proposes, only about the pipeline as a whole.

Run::

    python examples/07_frontier_models.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import torch

from garchai.core import TrainConfig, set_seed
from garchai.datasets import load
from garchai.ensembles import (
    GARCHTFT,
    SentimentGARCHX,
    garman_klass,
    historical_volatility,
    normalise_sentiment,
)
from garchai.fuzzy import GARCHFIS
from garchai.multivariate import DCCGARCHMANN, fit_dcc
from garchai.reporting import Table, comparison_table, use_journal_style
from garchai.risk import GARCHDDQNVaR

warnings.filterwarnings("ignore")

# The package documents float64 as its numerical default but does not set it
# globally on import, so an example must ask for it explicitly.
torch.set_default_dtype(torch.float64)

OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)
START, END = "2015-01-01", "2020-12-31"


def _returns(key: str, start: str = START, end: str = END) -> np.ndarray:
    frame = load(key, start=start, end=end)
    return 100.0 * np.diff(np.log(frame["close"].to_numpy()))


def _write(name: str, table: Table) -> None:
    (OUTPUT / f"{name}.md").write_text(table.to_markdown(), encoding="utf-8")
    (OUTPUT / f"{name}.tex").write_text(table.to_latex(), encoding="utf-8")
    print(table.to_markdown(), "\n")


# --------------------------------------------------------------------------- #
# 1. GARCH-FIS — does the volatility adaptation earn its place?
# --------------------------------------------------------------------------- #
def run_garch_fis() -> None:
    print("\n" + "=" * 78 + "\n GARCH-FIS (Li & Zhang 2026) — S&P 500\n" + "=" * 78)
    returns = _returns("sp500")
    split = int(0.8 * returns.size)

    metrics = {}
    for label, kappa in (("FIS (no adaptation, k=0)", 0.0), ("GARCH-FIS (k=1)", 1.0)):
        model = GARCHFIS(n_rules=6, kappa=kappa, window=250, n_lags=3, seed=0)
        model.fit(returns[:split])
        metrics[label] = model.evaluate(returns[split:])

    table = comparison_table(
        metrics,
        caption=(
            "GARCH-FIS on S&P 500 daily returns, out-of-sample. The k=0 row is the "
            "same rule base with the volatility adaptation switched off."
        ),
    )
    table.note = (
        "The adaptation is the paper's sole contribution, so the two rows differ "
        "only in whether membership widths scale with the GARCH forecast. Note "
        "that the target here is the daily return, which is close to "
        "unpredictable; the paper forecasts prices, which are strongly "
        "autocorrelated, so these R-squared values are not comparable to its own."
    )
    _write("frontier_garch_fis", table)


# --------------------------------------------------------------------------- #
# 2. GARCH-DDQN — VaR as classification
# --------------------------------------------------------------------------- #
def run_ddqn() -> None:
    print("\n" + "=" * 78 + "\n GARCH-DDQN VaR (Pokou et al. 2025) — FTSE 100\n" + "=" * 78)
    returns = _returns("ftse")
    agent = GARCHDDQNVaR(alpha=0.05, window=250, n_lags=5, seed=0)
    agent.fit(returns, n_episodes=25)
    result = agent.evaluate(returns)

    benchmark = _unconditional_var(returns, 0.05)

    # Built directly rather than through comparison_table: several of these
    # columns have no "best" value. mean_VaR is a descriptive statistic (a more
    # negative threshold is more conservative, not better), expected_rate is a
    # constant, and a p-value is evidence against a null rather than a score to
    # maximise. Bolding a maximum in those columns would assert a ranking that
    # does not exist, so violation error is reported as |actual - nominal| and
    # nothing is bolded automatically.
    import pandas as pd

    frame = pd.DataFrame(
        {
            "GARCH-DDQN": {
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "Violation rate": result["violation_rate"],
                "|rate - 0.05|": abs(result["violation_rate"] - 0.05),
                "Kupiec p": result["kupiec_p"],
                "Christoffersen p": result["christoffersen_p"],
                "Mean VaR": result["mean_VaR"],
            },
            "Unconditional quantile": {
                "Accuracy": float("nan"),
                "Precision": float("nan"),
                "Recall": float("nan"),
                "F1": float("nan"),
                "Violation rate": benchmark["violation_rate"],
                "|rate - 0.05|": abs(benchmark["violation_rate"] - 0.05),
                "Kupiec p": benchmark["kupiec_p"],
                "Christoffersen p": benchmark["christoffersen_p"],
                "Mean VaR": benchmark["mean_VaR"],
            },
        }
    ).T.round(4)
    frame.index.name = "Model"

    table = Table(
        frame=frame,
        caption="Risk-regime classification and VaR backtests, FTSE 100 (2015-2020).",
        label="tab:ddqn_var",
        note=(
            "Kupiec and Christoffersen p-values above 0.05 mean the model is NOT "
            "rejected; they are evidence, not scores, so no column is marked "
            "'best'. The comparison that matters is Christoffersen: both models "
            "hit the 5% rate, but the unconditional quantile clusters its "
            "violations while the conditional model does not, which is the "
            "paper's stated claim. Classification columns are blank for the "
            "unconditional benchmark because it makes no regime call. Recall is "
            "the share of true high-risk days detected - the quantity Eq. 12's "
            "reward is shaped to raise, and the one plain accuracy conceals."
        ),
    )
    _write("frontier_ddqn_var", table)

    _plot_ddqn(agent, returns)


def _unconditional_var(returns: np.ndarray, alpha: float) -> dict[str, float]:
    """The benchmark any conditional VaR model must beat to be worth its cost."""
    from garchai.risk import christoffersen_conditional_coverage, kupiec_pof

    threshold = np.quantile(returns, alpha)
    var = np.full(returns.size, threshold)
    return {
        "violation_rate": float((returns < var).mean()),
        "expected_rate": alpha,
        "kupiec_p": float(kupiec_pof(returns, var, alpha).p_value),
        "christoffersen_p": float(
            christoffersen_conditional_coverage(returns, var, alpha).p_value
        ),
        "mean_VaR": float(var.mean()),
    }


def _plot_ddqn(agent: GARCHDDQNVaR, returns: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    use_journal_style()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), height_ratios=[2, 1])

    var = agent.var(returns)
    aligned = returns[agent.n_lags :]
    breaches = aligned < var
    axes[0].plot(aligned, lw=0.6, color="0.55", label="FTSE 100 return")
    axes[0].plot(var, lw=1.1, color="C3", label="DDQN VaR (5%)")
    axes[0].scatter(
        np.flatnonzero(breaches), aligned[breaches], s=9, color="C1", zorder=3,
        label=f"exceedance ({breaches.sum()})",
    )
    axes[0].set_ylabel("return (%)")
    axes[0].legend(loc="lower left", ncol=3, fontsize=8)
    axes[0].set_title("Regime-conditional VaR and its exceedances")

    axes[1].plot(agent.history.reward, color="C0", lw=1.2)
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("episode reward")
    axes[1].set_title("Learning curve (reward under Eq. 12)")
    figure.tight_layout()
    figure.savefig(OUTPUT / "frontier_ddqn_var.png")
    plt.close(figure)
    print(f"  figure -> {OUTPUT / 'frontier_ddqn_var.png'}")


# --------------------------------------------------------------------------- #
# 3. DCC-GARCH-MANN — both hybridisation directions
# --------------------------------------------------------------------------- #
def run_dcc_mann() -> None:
    print("\n" + "=" * 78 + "\n DCC-GARCH-MANN (Fatima & Uddin 2022)\n" + "=" * 78)
    keys = ("sp500", "ftse", "nikkei", "hangseng")
    series = [_returns(k) for k in keys]
    length = min(s.size for s in series)
    panel = np.column_stack([s[-length:] for s in series])

    metrics = {}
    for direction, label in (
        ("dcc_to_mann", "DCC-GARCH -> MANN"),
        ("mann_to_dcc", "MANN -> DCC-GARCH"),
    ):
        model = DCCGARCHMANN(direction=direction, n_lags=3, seed=0)
        model.fit(panel, train=TrainConfig(lr=1e-3, max_epochs=300, patience=30))
        result = model.evaluate(panel)
        metrics[label] = {
            "RMSE": result["RMSE"], "MAE": result["MAE"],
            "dcc_a": result["dcc_a"], "dcc_b": result["dcc_b"],
            "dcc_persistence": result["dcc_persistence"],
        }

    # Only RMSE and MAE are ranked. a, b and their sum describe the estimated
    # correlation process; a larger persistence is not a better model, so
    # bolding a maximum there would assert a ranking that does not exist.
    accuracy = comparison_table(
        {k: {"RMSE": v["RMSE"], "MAE": v["MAE"]} for k, v in metrics.items()},
        caption=(
            "Both hybridisation directions on a four-market panel "
            "(S&P 500, FTSE 100, Nikkei 225, Hang Seng), 2015-2020."
        ),
        lower_is_better=("RMSE", "MAE"),
    )
    accuracy.note = (
        "DATA SUBSTITUTION: the paper uses S&P 500, FTSE-100, KSE-100, KLSE and "
        "BSESN. KSE-100, KLSE and BSESN are not in the bundled public sources, so "
        "Nikkei 225 and Hang Seng stand in. These figures are therefore not "
        "comparable to the paper's Table 5."
    )
    _write("frontier_dcc_mann", accuracy)

    import pandas as pd

    parameters = pd.DataFrame(
        {k: {"a": v["dcc_a"], "b": v["dcc_b"], "a + b": v["dcc_persistence"]}
         for k, v in metrics.items()}
    ).T.round(4)
    parameters.index.name = "Model"
    _write(
        "frontier_dcc_parameters",
        Table(
            frame=parameters,
            caption="Estimated DCC correlation dynamics (Engle 2002).",
            label="tab:dcc_parameters",
            note=(
                "Descriptive parameters, not scores - nothing is ranked. a + b "
                "must stay below 1 for the correlation process to be stationary; "
                "values close to 1 mean correlations move slowly and shocks to "
                "co-movement persist, which is typical of equity index panels."
            ),
        ),
    )

    _plot_correlations(panel, keys)


def _plot_correlations(panel: np.ndarray, keys: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt

    use_journal_style()
    result = fit_dcc(panel)
    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            axis.plot(result.pairwise(i, j), lw=0.9, label=f"{keys[i]}-{keys[j]}")
    axis.set_ylabel("conditional correlation")
    axis.set_xlabel("trading day (2015-2020)")
    axis.set_title(
        f"Dynamic conditional correlations (a={result.a:.3f}, b={result.b:.3f})"
    )
    axis.legend(ncol=3, fontsize=7)
    figure.tight_layout()
    figure.savefig(OUTPUT / "frontier_dcc_correlations.png")
    plt.close(figure)
    print(f"  figure -> {OUTPUT / 'frontier_dcc_correlations.png'}")


# --------------------------------------------------------------------------- #
# 4. GARCH-TFT — the paper's HV-vs-GK distinction, kept
# --------------------------------------------------------------------------- #
def run_garch_tft() -> None:
    print("\n" + "=" * 78 + "\n GARCH-TFT (Petrosino et al. 2025) — S&P 500\n" + "=" * 78)
    frame = load("sp500", start=START, end=END)
    close = frame["close"].to_numpy()
    returns = 100.0 * np.diff(np.log(close))

    hv = historical_volatility(close, 10)[1:]
    gk = garman_klass(
        frame["high"].to_numpy(), frame["low"].to_numpy(),
        frame["open"].to_numpy(), close,
    )[1:]
    garch_sigma = _rolling_garch_sigma(returns)

    metrics: dict[str, dict[str, float]] = {}
    for proxy_name, target in (("HV", hv), ("GK", gk)):
        for use_garch in (True, False):
            features = (
                np.column_stack([returns, np.roll(target, 1), garch_sigma])
                if use_garch
                else np.column_stack([returns, np.roll(target, 1)])
            )
            features, aligned = features[1:], target[1:]
            split = int(0.8 * aligned.size)

            set_seed(0)
            model = GARCHTFT(window=20, d_model=32, n_heads=4, dropout=0.1, seed=0)
            model.names = ("return", "lagged_proxy", "garch") if use_garch else (
                "return", "lagged_proxy"
            )
            model.fit(
                features[:split], aligned[:split],
                train=TrainConfig(lr=1e-3, max_epochs=250, patience=25),
            )
            label = f"{proxy_name}: {'GARCH-TFT' if use_garch else 'TFT (no GARCH)'}"
            metrics[label] = model.evaluate(features[split:], aligned[split:])
            if use_garch:
                importance = model.variable_importance(features[split:])
                metrics[label]["garch_weight"] = importance["garch"]

    # nominal_coverage is a constant (0.8) and interval_coverage is best when it
    # is CLOSEST to it, not when it is largest, so neither belongs in a
    # bolded-maximum column. Coverage is reported as its signed gap instead.
    for row in metrics.values():
        if "interval_coverage" in row:
            row["coverage_gap"] = row.pop("interval_coverage") - row.pop(
                "nominal_coverage"
            )

    table = comparison_table(
        metrics,
        caption=(
            "GARCH-TFT against a stand-alone TFT on two volatility proxies, "
            "S&P 500 out-of-sample."
        ),
        lower_is_better=("MAE", "RMSE", "quantile_crossing_rate"),
    )
    table.note = (
        "The paper's claim is proxy-specific: a significant gain on Garman-Klass "
        "and mere parity on historical volatility. Both proxies are reported so "
        "that claim can be checked rather than assumed. garch_weight is the mean "
        "variable-selection weight the model itself puts on the GARCH input - a "
        "near-zero value means the hybrid is not using its econometric term "
        "whatever the error columns say. coverage_gap is realised minus nominal "
        "(0.80) 80% interval coverage, so zero is ideal and the sign says "
        "whether the intervals are too narrow (negative) or too wide. "
        "quantile_crossing_rate above a few percent invalidates the coverage "
        "figure on the same row, because a crossed interval is empty."
    )
    _write("frontier_garch_tft", table)


def _rolling_garch_sigma(returns: np.ndarray, window: int = 250) -> np.ndarray:
    from arch import arch_model

    out = np.full(returns.size, np.nan)
    for t in range(window, returns.size):
        try:
            fitted = arch_model(
                returns[t - window : t], mean="Constant", vol="GARCH", p=1, q=1
            ).fit(disp="off", show_warning=False)
            out[t] = float(
                np.sqrt(fitted.forecast(horizon=1, reindex=False).variance.values[-1, 0])
            )
        except Exception:  # noqa: BLE001
            out[t] = out[t - 1]
    out[:window] = np.nanstd(returns[:window])
    return out


# --------------------------------------------------------------------------- #
# 5. Sentiment-augmented GARCH — on a declared public proxy
# --------------------------------------------------------------------------- #
def run_sentiment() -> None:
    print("\n" + "=" * 78 + "\n Sentiment-augmented GARCH (2026) — S&P 500\n" + "=" * 78)
    print(
        "  NOTE: the paper's sentiment feed is proprietary. A public proxy built\n"
        "  from realised downside pressure is used instead, so these numbers do\n"
        "  NOT reproduce the paper's and are not presented as doing so."
    )
    returns = _returns("sp500")

    # A transparent, fully reproducible stand-in: rolling share of down-days,
    # inverted. It is a market-derived mood proxy, not a media-sentiment index,
    # and it cannot carry the textual information the paper's feed carries.
    # The first window is empty, and the mean of an empty slice is NaN. Seeding
    # it at 0.5 (neutral) keeps the series finite; leaving it NaN used to slip
    # past the non-negativity check, because `nan < 0` is False.
    window = 20
    down_share = np.array(
        [
            (returns[max(0, t - window) : t] < 0).mean() if t > 0 else 0.5
            for t in range(returns.size)
        ]
    )
    sentiment = normalise_sentiment(1.0 - down_share)
    assert np.isfinite(sentiment).all()

    metrics = {}
    baseline = SentimentGARCHX(sentiment_lags=1).fit(returns, np.zeros_like(sentiment))
    augmented = SentimentGARCHX(sentiment_lags=1).fit(returns, sentiment)
    for label, model, series in (
        ("GARCH (no sentiment)", baseline, np.zeros_like(sentiment)),
        ("GARCH-X (sentiment)", augmented, sentiment),
    ):
        # In the baseline the sentiment series is identically zero, so gamma
        # multiplies zero and is not identified. Whatever value the optimiser
        # leaves there is arbitrary, and printing it would invite a comparison
        # against the augmented model's gamma that means nothing.
        identified = bool(np.any(series > 0))
        metrics[label] = {
            "loglik": model.params["loglik"],
            "alpha": model.params["alpha"],
            "beta": model.params["beta"],
            "gamma1": model.params["gamma1"] if identified else float("nan"),
            "persistence": model.params["persistence"],
            "sentiment_share": float(model.sentiment_contribution(series).mean()),
        }

    # Every column here is an estimated parameter or a likelihood, none of which
    # ranks one row above the other -- a larger alpha is not "better". Built as
    # a plain estimation table so nothing is bolded as a winner. The one
    # comparison that IS meaningful, the likelihood-ratio test, is computed
    # explicitly below.
    import pandas as pd
    from scipy.stats import chi2

    frame = pd.DataFrame(metrics).T.round(4)
    frame.index.name = "Model"

    statistic = 2.0 * (
        metrics["GARCH-X (sentiment)"]["loglik"] - metrics["GARCH (no sentiment)"]["loglik"]
    )
    p_value = float(chi2.sf(max(statistic, 0.0), df=1))

    table = Table(
        frame=frame,
        caption="Sentiment-augmented GARCH on S&P 500 daily returns, 2015-2020.",
        label="tab:sentiment_garch",
    )
    table.note = (
        f"Likelihood-ratio test of gamma = 0: LR = {statistic:.3f}, p = {p_value:.4f} "
        f"({'sentiment term is significant' if p_value < 0.05 else 'sentiment term is NOT significant'} "
        "at 5%). Parameters are not ranked - a larger alpha is not a better "
        "model - so no column is bolded. "
        "DATA SUBSTITUTION: the paper's proprietary sentiment indices are not "
        "redistributable; a public down-day proxy stands in, so this table does "
        "not reproduce the paper. sentiment_share is the mean fraction of the "
        "conditional variance attributable to the sentiment term - if it is "
        "near zero the augmentation is inert regardless of the likelihood."
    )
    _write("frontier_sentiment", table)


if __name__ == "__main__":
    run_garch_fis()
    run_ddqn()
    run_dcc_mann()
    run_garch_tft()
    run_sentiment()
    print(f"\nAll tables and figures written to {OUTPUT}")
