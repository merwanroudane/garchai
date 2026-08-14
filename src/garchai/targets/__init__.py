"""Volatility targets and proxies, each named after the paper it follows."""

from .realized import (
    absolute_returns,
    bipower_variation,
    get_target,
    hansen_lunde_scale,
    realized_variance_intraday,
    realized_volatility_intraday,
    realized_volatility_wei,
    realized_volatility_zhao,
    rolling_std,
    squared_returns,
)

__all__ = [
    "realized_volatility_zhao", "realized_volatility_wei", "rolling_std",
    "squared_returns", "absolute_returns", "realized_variance_intraday",
    "realized_volatility_intraday", "bipower_variation", "hansen_lunde_scale",
    "get_target",
]
