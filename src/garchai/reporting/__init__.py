"""Publication-quality tables and figures.

One house style for both, so a table or figure moves from a notebook into a
manuscript without rework.
"""

from .figures import (
    PALETTE, plot_forecast_fan, plot_news_impact, plot_rech_decomposition,
    plot_training_history,
    plot_var_exceedances, plot_volatility, use_journal_style,
)
from .tables import Table, comparison_table, estimation_table, significance_stars

__all__ = [
    "Table", "estimation_table", "comparison_table", "significance_stars",
    "use_journal_style", "PALETTE", "plot_volatility", "plot_var_exceedances",
    "plot_forecast_fan", "plot_news_impact", "plot_rech_decomposition",
    "plot_training_history",
]
