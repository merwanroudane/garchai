"""
Getting started with garchai: the complete workflow on real market data
======================================================================

Run me::

    python examples/01_getting_started.py

This walks the whole path a volatility study takes — load data, fit competing
models, compare them formally, evaluate the risk numbers, and produce the tables
and figures you would put in a paper. Every number comes from real S&P 500 data
bundled with the package, so this script reproduces byte-for-byte offline.

What each step is doing, and why, is explained inline. If you have never fitted
a GARCH model before, read top to bottom; nothing is assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from garchai import EGARCH, GARCH, GJRGARCH
from garchai.core import ModelConfig, TrainConfig, set_seed
from garchai.datasets import load_returns
from garchai.evaluation import diebold_mariano, model_confidence_set
from garchai.losses import mae, mse, qlike
from garchai.reporting import comparison_table, estimation_table
from garchai.risk import var_backtest_table, violation_ratio
from garchai.targets import realized_volatility_wei

import torch

OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    set_seed(0)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    # Daily log returns in percent. Working in percent rather than raw
    # fractions keeps omega around 0.05 instead of 5e-6, which matters for
    # the conditioning of the likelihood surface.
    print("1. Loading data")
    returns = load_returns("sp500", start="2010-01-01", end="2019-12-31")
    train, test = returns[:"2018-12-31"], returns["2019-01-01":]
    print(f"   S&P 500 daily log returns (%)")
    print(f"   train: {len(train):5d} obs  {train.index[0].date()} to {train.index[-1].date()}")
    print(f"   test : {len(test):5d} obs  {test.index[0].date()} to {test.index[-1].date()}")
    print(f"   mean {train.mean():+.4f}   sd {train.std():.4f}   "
          f"skew {train.skew():+.2f}   kurtosis {train.kurtosis():.2f}")
    print("   Excess kurtosis well above 0 is the classic sign that a Gaussian")
    print("   innovation will under-price the tails. We test that below.\n")

    # ------------------------------------------------------------------ #
    # 2. Fit competing models
    # ------------------------------------------------------------------ #
    # A constant mean, not zero: for equity indices the daily drift is small
    # but not zero, and forcing it to zero pushes that drift into the
    # residuals, where the variance model then has to absorb it.
    print("2. Fitting models by maximum likelihood")
    settings = TrainConfig(lr=0.1, max_epochs=200, patience=30, refine_with_lbfgs=True)
    specifications = {
        "GARCH-N": GARCH(config=ModelConfig(mean="constant", distribution="normal")),
        "GARCH-t": GARCH(config=ModelConfig(mean="constant", distribution="student-t")),
        "GJR-t": GJRGARCH(config=ModelConfig(mean="constant", distribution="student-t")),
        "EGARCH-t": EGARCH(config=ModelConfig(mean="constant", distribution="student-t")),
    }

    fits = {}
    for name, model in specifications.items():
        fits[name] = model.fit(train.to_numpy(), settings)
        print(f"   {name:10s} loglik {fits[name].loglikelihood:10.2f}   "
              f"persistence {fits[name].persistence:.4f}")
    print()

    # ------------------------------------------------------------------ #
    # 3. Estimation table
    # ------------------------------------------------------------------ #
    print("3. Estimation table\n")
    table = estimation_table(fits, caption="Table 1. GARCH-family estimates, S&P 500 2010-2018")
    print(table.to_text())
    (OUTPUT / "table1_estimation.tex").write_text(table.to_latex(), encoding="utf-8")
    (OUTPUT / "table1_estimation.md").write_text(table.to_markdown(), encoding="utf-8")
    print(f"\n   written to {OUTPUT / 'table1_estimation.tex'}\n")

    # The Student-t degrees of freedom is the number to look at here: a low
    # value (say under 10) says the data has tails a normal cannot reach.
    df = fits["GARCH-t"].distribution_parameters.get("df (nu)")
    if df is not None:
        print(f"   Fitted Student-t degrees of freedom: {df:.2f}")
        print("   Low df confirms the excess kurtosis seen in step 1.\n")

    # ------------------------------------------------------------------ #
    # 4. Out-of-sample forecast accuracy
    # ------------------------------------------------------------------ #
    # We need a volatility *target* to score against. Note that this is
    # Wei et al.'s estimator specifically: 5-day, demeaned, and strictly
    # lagged so nothing from period t leaks into the target at t. Zhao et
    # al. use a different one; garchai keeps them as separate named
    # functions precisely so a comparison never mixes them by accident.
    print("4. Out-of-sample evaluation (2019)")
    full = returns.to_numpy()
    split = len(train)

    target = realized_volatility_wei(torch.tensor(full), k=5).numpy()[split:]
    valid = np.isfinite(target)

    metrics, loss_series = {}, {}
    for name, model in specifications.items():
        # Filter the whole sample with the parameters estimated on the
        # training window only, then read off the test-period variances.
        # No refitting: this is genuine out-of-sample.
        with torch.no_grad():
            h, _, _ = model.filter(torch.tensor(full) * model.data_scale_)
        predicted_var = (h.numpy() / model.data_scale_**2)[split:]
        predicted_vol = np.sqrt(predicted_var)

        t = torch.tensor(target[valid])
        p_vol = torch.tensor(predicted_vol[valid])
        p_var = torch.tensor(predicted_var[valid])

        metrics[name] = {
            "MSE": float(mse(t, p_vol)),
            "MAE": float(mae(t, p_vol)),
            "QLIKE": float(qlike(t.square(), p_var)),
        }
        loss_series[name] = ((t - p_vol) ** 2).numpy()

    comparison = comparison_table(
        metrics, caption="Table 2. Out-of-sample accuracy, S&P 500 2019"
    )
    print()
    print(comparison.to_text())
    (OUTPUT / "table2_accuracy.tex").write_text(comparison.to_latex(), encoding="utf-8")
    print()

    # ------------------------------------------------------------------ #
    # 5. Is the difference significant?
    # ------------------------------------------------------------------ #
    # A lower MSE is not evidence on its own. Diebold-Mariano asks whether
    # the loss difference is distinguishable from zero; the MCS asks which
    # models survive as candidates for best once you account for having
    # compared several at once.
    print("5. Formal comparison")
    best = min(metrics, key=lambda k: metrics[k]["MSE"])
    for name in metrics:
        if name == best:
            continue
        result = diebold_mariano(
            loss_series[best], loss_series[name], names=(best, name)
        )
        verdict = "significant" if result.significant else "not significant"
        print(f"   {best} vs {name:10s}: DM = {result.statistic:+.3f}, "
              f"p = {result.p_value:.3f}  ({verdict})")

    import pandas as pd

    mcs = model_confidence_set(pd.DataFrame(loss_series), alpha=0.10, n_bootstrap=1000)
    print("\n   Model confidence set (90%):")
    print("   " + mcs.to_string(index=False).replace("\n", "\n   "))
    survivors = list(mcs.loc[mcs["in_mcs"], "model"])
    print(f"\n   {len(survivors)} of {len(metrics)} models survive: {', '.join(survivors)}")
    print("   Anything more than one survivor means the data cannot separate them.\n")

    # ------------------------------------------------------------------ #
    # 6. Risk: Value-at-Risk and backtests
    # ------------------------------------------------------------------ #
    print("6. Value-at-Risk backtests (in-sample, 2010-2018)")
    for name in ("GARCH-N", "GARCH-t"):
        model = specifications[name]
        var = model.var(0.01)
        ratio = violation_ratio(train.to_numpy(), var, 0.01)
        table = var_backtest_table(train.to_numpy(), var, alpha=0.01, es=model.es(0.01))
        rejected = int(table["reject_5pct"].sum())
        print(f"   {name:10s} 1% VaR violation ratio {ratio:.2f}   "
              f"tests rejected: {rejected}/{len(table)}")
    print("   A ratio near 1 is well calibrated. The Gaussian model should be")
    print("   noticeably above 1 - that is the fat-tail problem, made concrete.\n")

    # ------------------------------------------------------------------ #
    # 7. Forecasting
    # ------------------------------------------------------------------ #
    print("7. Multi-step forecasting")
    model = specifications["GARCH-t"]
    forecast = model.forecast(horizon=60, method="analytic")
    long_run = fits["GARCH-t"].long_run_variance
    print(f"   1-day  ahead volatility: {forecast.volatility[0]:.4f}%")
    print(f"   60-day ahead volatility: {forecast.volatility[-1]:.4f}%")
    print(f"   long-run volatility    : {np.sqrt(long_run):.4f}%")
    print("   The forecast decays geometrically toward the long-run level at")
    print(f"   rate {fits['GARCH-t'].persistence:.4f} per day. That decay rate *is* the")
    print("   persistence parameter - which is why it is the number to check\n"
          "   when a forecast looks implausibly flat or implausibly jumpy.\n")

    # ------------------------------------------------------------------ #
    # 8. Figures
    # ------------------------------------------------------------------ #
    print("8. Figures")
    try:
        from garchai.reporting import (
            plot_forecast_fan, plot_news_impact, plot_var_exceedances, plot_volatility,
        )

        result = fits["GARCH-t"]
        plot_volatility(
            train.to_numpy(), result.conditional_volatility, index=train.index,
            title="S&P 500 returns and fitted conditional volatility (GARCH-t)",
        ).savefig(OUTPUT / "fig1_volatility.png")

        plot_var_exceedances(
            train.to_numpy(), specifications["GARCH-t"].var(0.01), alpha=0.01,
            index=train.index,
        ).savefig(OUTPUT / "fig2_var.png")

        plot_forecast_fan(
            result.conditional_volatility, forecast.volatility, long_run=long_run,
        ).savefig(OUTPUT / "fig3_forecast.png")

        plot_news_impact(
            {n: specifications[n] for n in ("GARCH-t", "GJR-t", "EGARCH-t")},
        ).savefig(OUTPUT / "fig4_news_impact.png")

        print(f"   4 figures written to {OUTPUT}")
        print("   fig4 is the one to look at: the GJR and EGARCH curves are")
        print("   visibly tilted, which is the leverage effect made graphical.")
    except ImportError:
        print("   matplotlib not installed; skipping. `pip install garchai[plots]`")

    print("\nDone. Outputs in", OUTPUT)


if __name__ == "__main__":
    main()
