r"""Generate the light-themed figures used by the GitHub Pages site.

Every figure is built from **real market data** bundled with the package, on a
white ground with a bright, high-contrast palette — no dark backgrounds, no dark
fills. Output goes to ``docs/assets/`` as SVG so it stays sharp at any zoom and
adds almost nothing to the repository size.

Run::

    python docs/make_site_figures.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)

from garchai.core import ModelConfig, TrainConfig
from garchai.datasets import load, load_returns
from garchai.econometric import EGARCH, GARCH, GJRGARCH

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

# A sunny palette: warm, saturated strokes on a bright white ground. Warm hues
# lead (gold, sunset, coral) with a cool sky blue for contrast, so the charts
# read as bright rather than clinical. Backgrounds stay white -- a dark plot
# ground would defeat the point.
C = {
    "gold": "#f0a323",
    "sun": "#e8890c",
    "sunset": "#f2704e",
    "coral": "#e2574c",
    "sky": "#2f9fd6",
    "teal": "#17a398",
    "green": "#7cb342",
    "rose": "#e05b8a",
    "grey": "#b9c2cc",
    "grid": "#f0e9dd",
    "ink": "#3a3226",
}
# Warm-first ordering so the first series a reader sees is sunny, not blue.
SERIES = [C["sun"], C["sky"], C["sunset"], C["teal"], C["rose"], C["green"]]
# Back-compat aliases for the figure functions below.
C["blue"] = C["sky"]
C["amber"] = C["gold"]
C["violet"] = C["rose"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#ddd2c0",
        "axes.labelcolor": C["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 1.0,
        "text.color": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.4,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def save(fig, name: str) -> None:
    fig.savefig(ASSETS / f"{name}.svg", format="svg")
    plt.close(fig)
    print(f"  wrote assets/{name}.svg")


# --------------------------------------------------------------------------- #
# 1. Stylised facts — why GARCH exists at all
# --------------------------------------------------------------------------- #
def figure_stylised_facts(returns: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))

    axes[0].plot(returns, color=C["blue"], lw=0.6)
    axes[0].set_title("1. Volatility clusters")
    axes[0].set_ylabel("return (%)")
    axes[0].set_xlabel("trading day")

    # Fat tails against the normal it is so often assumed to follow
    z = (returns - returns.mean()) / returns.std()
    axes[1].hist(z, bins=80, density=True, color=C["blue"], alpha=0.32,
                 edgecolor=C["blue"], linewidth=0.4)
    grid = np.linspace(-6, 6, 400)
    axes[1].plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi),
                 color=C["coral"], lw=1.8, label="Normal")
    axes[1].set_title(f"2. Fat tails (kurtosis {float((z**4).mean()):.1f} vs 3.0)")
    axes[1].set_xlim(-6, 6)
    axes[1].legend()
    axes[1].set_xlabel("standardised return")

    # The decisive plot: returns are unpredictable, their magnitude is not
    def acf(x, n):
        x = x - x.mean()
        d = float((x * x).sum())
        return np.array([float((x[:-k] * x[k:]).sum()) / d for k in range(1, n + 1)])

    lags = np.arange(1, 41)
    width = 0.42
    axes[2].bar(lags - width / 2, acf(returns, 40), width,
                color=C["grey"], label="returns")
    axes[2].bar(lags + width / 2, acf(returns**2, 40), width,
                color=C["teal"], label="squared returns")
    axes[2].axhline(0, color="#c7d2de", lw=1)
    axes[2].set_title("3. Magnitude is predictable, direction is not")
    axes[2].set_xlabel("lag")
    axes[2].legend()

    fig.tight_layout()
    save(fig, "stylised_facts")


# --------------------------------------------------------------------------- #
# 2. Fitted conditional volatility
# --------------------------------------------------------------------------- #
def figure_conditional_volatility(returns, results) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.0))
    vol = results["GJR"].conditional_volatility
    ax.fill_between(range(len(returns)), -2 * vol, 2 * vol,
                    color=C["blue"], alpha=0.16, label="±2σ from GJR-GARCH")
    ax.plot(returns, color=C["grey"], lw=0.55, label="S&P 500 return")
    ax.plot(2 * vol, color=C["blue"], lw=1.3)
    ax.plot(-2 * vol, color=C["blue"], lw=1.3)
    ax.set_title("Conditional volatility tracks the market: S&P 500, 2015–2020")
    ax.set_ylabel("return (%)")
    ax.set_xlabel("trading day")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, "conditional_volatility")


# --------------------------------------------------------------------------- #
# 3. News impact — what asymmetry means
# --------------------------------------------------------------------------- #
def figure_news_impact(models) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    shocks = np.linspace(-4, 4, 401)
    for (name, model), colour in zip(models.items(), SERIES):
        p = {k: float(v.detach()) for k, v in model.cell.transformed_parameters().items()}
        h_bar = float(model.cell.unconditional_variance().detach())
        if name == "EGARCH":
            z = shocks / np.sqrt(h_bar)
            log_h = (p["omega"] + p["beta"] * np.log(h_bar)
                     + p["alpha"] * (np.abs(z) - np.sqrt(2 / np.pi)) + p.get("gamma", 0.0) * z)
            h = np.exp(log_h)
        elif "gamma" in p:
            h = (p["omega"] + p["alpha"] * shocks**2
                 + p["gamma"] * shocks**2 * (shocks < 0) + p["beta"] * h_bar)
        else:
            h = p["omega"] + p["alpha"] * shocks**2 + p["beta"] * h_bar
        ax.plot(shocks, np.sqrt(h), color=colour, lw=2.0, label=name)
    ax.set_title("News impact: bad news raises volatility more")
    ax.set_xlabel("shock at $t-1$ (%)")
    ax.set_ylabel("resulting volatility $\\sigma_t$")
    ax.legend()
    fig.tight_layout()
    save(fig, "news_impact")


# --------------------------------------------------------------------------- #
# 4. VaR and its exceedances
# --------------------------------------------------------------------------- #
def figure_var(returns, model) -> None:
    var = model.var(alpha=0.05)
    hits = returns < var
    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    ax.plot(returns, color=C["grey"], lw=0.55, label="S&P 500 return")
    ax.plot(var, color=C["coral"], lw=1.5, label="5% Value-at-Risk")
    ax.scatter(np.flatnonzero(hits), returns[hits], s=22, color=C["rose"],
               zorder=3, label=f"exceedance ({int(hits.sum())})",
               edgecolor="white", linewidth=0.6)
    ax.set_title(
        f"VaR exceedances: {hits.mean():.2%} observed against 5.00% promised"
    )
    ax.set_ylabel("return (%)")
    ax.set_xlabel("trading day")
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    save(fig, "var_exceedances")


# --------------------------------------------------------------------------- #
# 5. Multi-step forecast fan
# --------------------------------------------------------------------------- #
def figure_forecast(models) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for (name, model), colour in zip(models.items(), SERIES):
        try:
            f = model.forecast(horizon=20, method="simulation",
                               n_simulations=4000, seed=0)
        except Exception:
            f = model.forecast(horizon=20)
        ax.plot(range(1, 21), np.sqrt(f.variance), color=colour, lw=2.0,
                marker="o", markersize=3.2, label=name)
        lr = float(np.sqrt(model.cell.unconditional_variance().detach()))
        ax.axhline(lr, color=colour, lw=0.9, ls=":", alpha=0.7)
    ax.set_title("Forecasts decay toward the long-run level (dotted)")
    ax.set_xlabel("horizon (trading days)")
    ax.set_ylabel("forecast volatility")
    ax.legend()
    fig.tight_layout()
    save(fig, "forecast_fan")


# --------------------------------------------------------------------------- #
# 6. Dynamic conditional correlations across four markets
# --------------------------------------------------------------------------- #
def figure_dcc() -> None:
    from garchai.multivariate import fit_dcc

    keys = ("sp500", "ftse", "nikkei", "hangseng")
    labels = {"sp500": "S&P 500", "ftse": "FTSE 100",
              "nikkei": "Nikkei 225", "hangseng": "Hang Seng"}
    series = [load_returns(k, start="2015-01-01", end="2020-12-31").to_numpy()
              for k in keys]
    length = min(s.size for s in series)
    panel = np.column_stack([s[-length:] for s in series])

    result = fit_dcc(panel)
    fig, ax = plt.subplots(figsize=(13.5, 4.0))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for (i, j), colour in zip(pairs, SERIES + [C["rose"]]):
        ax.plot(result.pairwise(i, j), lw=1.3, color=colour,
                label=f"{labels[keys[i]]} – {labels[keys[j]]}")
    ax.set_title(
        f"Dynamic conditional correlations  (a = {result.a:.3f}, "
        f"b = {result.b:.3f}, a + b = {result.persistence:.3f})"
    )
    ax.set_ylabel("conditional correlation")
    ax.set_xlabel("trading day, 2015–2020")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    save(fig, "dcc_correlations")


# --------------------------------------------------------------------------- #
# 7. Model-comparison bars, straight from the reproduction log
# --------------------------------------------------------------------------- #
def figure_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2))

    names = ["GARCH\nbaseline", "GARCH-\nXGBoost", "Stacked\nML-GARCH",
             "LSTM-ANN\nGARCH", "Multi-GARCH\nTransformer", "BLSTM-ANN\nGARCH"]
    rmse = [0.5445, 0.3322, 0.1057, 0.2401, 0.0858, 0.0873]
    colours = [C["grey"]] + [C["amber"], C["violet"], C["teal"], C["blue"], C["green"]]
    bars = axes[0].bar(names, rmse, color=colours, edgecolor="white", linewidth=1.2)
    for bar, value in zip(bars, rmse):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                     f"{value:.3f}", ha="center", fontsize=9, color=C["ink"])
    axes[0].set_title("Feature hybrids vs the GARCH baseline (RMSE, lower is better)")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", labelsize=8.5)

    # The honest one: GARCH-FIS does not beat its own ablation
    labels = ["FIS, adaptation OFF\n(κ = 0)", "GARCH-FIS\n(κ = 1)"]
    metrics = {"RMSE": [2.0038, 2.0311], "MAE": [1.2030, 1.1894]}
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width / 2, metrics["RMSE"], width, color=C["coral"],
                edgecolor="white", linewidth=1.2, label="RMSE")
    axes[1].bar(x + width / 2, metrics["MAE"], width, color=C["blue"],
                edgecolor="white", linewidth=1.2, label="MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_title("A negative result we report: the adaptation does not pay")
    axes[1].legend()

    fig.tight_layout()
    save(fig, "model_comparison")


def main() -> None:
    style()
    print("Generating site figures from real market data...")

    returns = load_returns("sp500", start="2015-01-01", end="2020-12-31")
    array = returns.to_numpy()

    train = TrainConfig(max_epochs=600, refine_with_lbfgs=True)
    config = ModelConfig(distribution="student-t")
    models, results = {}, {}
    for name, cls in (("GARCH", GARCH), ("GJR", GJRGARCH), ("EGARCH", EGARCH)):
        model = cls(config=config)
        results[name] = model.fit(returns, train)
        models[name] = model
        print(f"  fitted {name}: loglik {results[name].loglikelihood:.2f}, "
              f"AIC {results[name].aic:.2f}")

    figure_stylised_facts(array)
    figure_conditional_volatility(array, results)
    figure_news_impact(models)
    figure_var(array, models["GJR"])
    figure_forecast(models)
    figure_comparison()
    figure_dcc()

    print(f"\nAll figures written to {ASSETS}")


if __name__ == "__main__":
    main()
