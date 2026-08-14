r"""Publication-quality figures.

One house style, applied everywhere, so that a figure lifted from a notebook
into a manuscript needs no rework: serif type, no top or right spine, a muted
palette that survives greyscale printing, and enough resolution to be embedded
at full column width.

Each function returns the Matplotlib ``Figure`` so a caller can adjust anything
before saving. Nothing here calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "use_journal_style",
    "PALETTE",
    "plot_volatility",
    "plot_var_exceedances",
    "plot_forecast_fan",
    "plot_news_impact",
    "plot_training_history",
]

# Muted, colour-blind-safe, and distinguishable in greyscale by lightness.
PALETTE = {
    "data": "#9aa0a6",
    "primary": "#1f4e79",
    "secondary": "#c0504d",
    "tertiary": "#4f8a3d",
    "quaternary": "#8064a2",
    "accent": "#e8a33d",
    "grid": "#d9d9d9",
}
_SERIES_COLOURS = [
    PALETTE["primary"], PALETTE["secondary"], PALETTE["tertiary"],
    PALETTE["quaternary"], PALETTE["accent"],
]


def use_journal_style() -> None:
    """Apply the package's figure style to the global Matplotlib settings."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def _new_axes(figsize):
    import matplotlib.pyplot as plt

    use_journal_style()
    return plt.subplots(figsize=figsize)


def plot_volatility(
    returns,
    volatility,
    index=None,
    title: str = "Conditional volatility",
    label: str = "Fitted",
    figsize: tuple[float, float] = (9.0, 4.0),
):
    r"""Returns with the fitted conditional volatility band overlaid.

    Plots :math:`\pm\sigma_t` rather than :math:`\sigma_t` alone, because the
    band is what makes volatility clustering visible: the returns should fill it
    evenly through calm and turbulent periods alike, and a model that is
    systematically too wide or too narrow shows up immediately.
    """
    x = np.arange(len(returns)) if index is None else index
    returns = np.asarray(returns)
    volatility = np.asarray(volatility)

    fig, ax = _new_axes(figsize)
    ax.plot(x, returns, color=PALETTE["data"], linewidth=0.6, label="Returns")
    ax.plot(x, volatility, color=PALETTE["primary"], label=f"{label} $\\sigma_t$")
    ax.plot(x, -volatility, color=PALETTE["primary"])
    ax.fill_between(x, -volatility, volatility, color=PALETTE["primary"], alpha=0.12)
    ax.set_title(title)
    ax.set_ylabel("Return (%)")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def plot_var_exceedances(
    returns,
    var,
    alpha: float,
    index=None,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 4.0),
):
    r"""Returns against a VaR line, with exceedances marked.

    This is the figure Zhao et al. (2024, Fig. 2) use. Its value is that it shows
    *when* violations happen, not just how many: a model with correct
    unconditional coverage whose exceedances all cluster in one month is visibly
    broken here, while a table of violation counts would call it well calibrated.
    """
    x = np.arange(len(returns)) if index is None else index
    returns = np.asarray(returns)
    var = np.asarray(var)
    breaches = returns < var
    rate = float(breaches.mean())

    fig, ax = _new_axes(figsize)
    ax.plot(x, returns, color=PALETTE["data"], linewidth=0.6, label="Returns")
    ax.plot(x, var, color=PALETTE["secondary"], linewidth=1.0,
            label=f"VaR ({alpha:.0%})")
    ax.scatter(
        np.asarray(x)[breaches], returns[breaches],
        color=PALETTE["secondary"], s=14, zorder=3, marker="v",
        label=f"Exceedances: {breaches.sum()} ({rate:.2%})",
    )
    ax.set_title(title or f"{alpha:.0%} Value-at-Risk and exceedances")
    ax.set_ylabel("Return (%)")
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    return fig


def plot_forecast_fan(
    history,
    forecast_volatility,
    long_run: float | None = None,
    history_tail: int = 250,
    title: str = "Volatility forecast",
    figsize: tuple[float, float] = (9.0, 4.0),
):
    """Recent fitted volatility followed by the forecast path.

    The long-run level is drawn as a reference because the whole content of a
    multi-step GARCH forecast is geometric decay toward it; seeing the two
    together tells you the persistence at a glance.
    """
    history = np.asarray(history)[-history_tail:]
    forecast_volatility = np.asarray(forecast_volatility)
    x_hist = np.arange(-len(history), 0)
    x_fore = np.arange(len(forecast_volatility))

    fig, ax = _new_axes(figsize)
    ax.plot(x_hist, history, color=PALETTE["primary"], label="Fitted $\\sigma_t$")
    ax.plot(x_fore, forecast_volatility, color=PALETTE["secondary"],
            linestyle="--", label="Forecast")
    if long_run is not None:
        ax.axhline(np.sqrt(long_run), color=PALETTE["tertiary"], linestyle=":",
                   label="Long-run $\\sigma$")
    ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Days from end of sample")
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_news_impact(
    models: dict,
    shock_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 401,
    title: str = "News impact curves",
    figsize: tuple[float, float] = (6.0, 4.0),
):
    r"""News impact curves: :math:`h_t` as a function of :math:`\epsilon_{t-1}`.

    Engle & Ng's diagnostic, and the clearest single picture of what
    distinguishes the GARCH families. GARCH gives a symmetric parabola; GJR and
    EGARCH tilt it, so the asymmetry a leverage parameter encodes becomes
    something you can see rather than a coefficient you have to trust.

    Each model is evaluated at its own long-run variance, which is the standard
    convention and what makes the curves comparable across specifications.
    """
    import torch

    from ..cells.base import CellState

    shocks = np.linspace(*shock_range, n_points)
    fig, ax = _new_axes(figsize)

    for i, (name, model) in enumerate(models.items()):
        cell = model.cell if hasattr(model, "cell") else model
        with torch.no_grad():
            long_run = float(cell.unconditional_variance())
            if not np.isfinite(long_run) or long_run <= 0:
                long_run = 1.0
            state = CellState(
                h=torch.tensor(long_run, dtype=torch.float64),
                log_h=torch.tensor(np.log(long_run), dtype=torch.float64),
            )
            curve = [
                float(cell.step(torch.tensor(s, dtype=torch.float64), state)[0])
                for s in shocks
            ]
        ax.plot(shocks, curve, color=_SERIES_COLOURS[i % len(_SERIES_COLOURS)], label=name)

    ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Shock $\\epsilon_{t-1}$")
    ax.set_ylabel("Conditional variance $h_t$")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_training_history(
    history: dict,
    title: str = "Training history",
    figsize: tuple[float, float] = (7.0, 4.0),
):
    """Loss (and validation loss) against epoch, with the learning rate beneath."""
    import matplotlib.pyplot as plt

    use_journal_style()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, height_ratios=[3, 1]
    )
    epochs = np.arange(len(history.get("loss", [])))
    top.plot(epochs, history["loss"], color=PALETTE["primary"], label="Training")
    validation = np.asarray(history.get("val_loss", []), dtype=float)
    if validation.size and np.isfinite(validation).any():
        top.plot(epochs, validation, color=PALETTE["secondary"], label="Validation")
    top.set_ylabel("Negative log-likelihood")
    top.set_title(title)
    top.legend(loc="best")

    if history.get("lr"):
        bottom.plot(epochs, history["lr"], color=PALETTE["tertiary"])
        bottom.set_yscale("log")
    bottom.set_ylabel("Learning rate")
    bottom.set_xlabel("Epoch")
    fig.tight_layout()
    return fig


def plot_rech_decomposition(
    result,
    beta_0: float,
    index=None,
    title: str = "RECH: econometric constant vs recurrent component",
    figsize: tuple[float, float] = (9.0, 4.5),
):
    r"""Split a fitted RECH constant into its baseline and recurrent parts.

    RECH replaces GARCH's constant :math:`\omega` with
    :math:`\omega_t = \beta_0 + \beta_1 h_t`. Plotting :math:`\omega_t` against
    the flat :math:`\beta_0` a plain GARCH would have shows *when* the recurrent
    component is contributing, and by how much — which is the question that
    decides whether the extra machinery earned its place.

    A model whose :math:`\omega_t` hugs :math:`\beta_0` throughout is a GARCH
    wearing a network; one whose :math:`\omega_t` lifts during turbulent periods
    is doing what the paper claims.

    Parameters
    ----------
    result
        A :class:`~garchai.core.results.FitResult` whose ``parameter_paths``
        contains ``omega`` — i.e. from :class:`~garchai.structural.rech.RECH`
        or :class:`~garchai.structural.realrech.DeepRGARCH`.
    beta_0
        The fitted constant, from ``model.beta_0``.
    """
    import matplotlib.pyplot as plt

    paths = result.parameter_paths or {}
    if "omega" not in paths:
        raise ValueError(
            "result.parameter_paths has no 'omega'. Fit a RECH-family model first."
        )
    omega = np.asarray(paths["omega"])
    x = np.arange(len(omega)) if index is None else index

    use_journal_style()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1]
    )
    top.plot(x, omega, color=PALETTE["primary"], label=r"$\omega_t = \beta_0 + \beta_1 h_t$")
    top.axhline(beta_0, color=PALETTE["secondary"], linestyle="--",
                label=r"$\beta_0$ (plain GARCH constant)")
    top.set_ylabel(r"$\omega_t$")
    top.set_title(title)
    top.legend(loc="best")

    contribution = omega - beta_0
    bottom.fill_between(x, 0.0, contribution, color=PALETTE["tertiary"], alpha=0.55)
    bottom.axhline(0.0, color="black", linewidth=0.6)
    bottom.set_ylabel(r"$\beta_1 h_t$")
    bottom.set_xlabel("Time")
    share = 100.0 * contribution.mean() / omega.mean() if omega.mean() else 0.0
    bottom.annotate(
        f"recurrent share of mean $\omega_t$: {share:.1f}%",
        xy=(0.01, 0.85), xycoords="axes fraction", fontsize=9,
    )
    fig.tight_layout()
    return fig
