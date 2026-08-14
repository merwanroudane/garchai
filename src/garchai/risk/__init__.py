"""Value-at-Risk, expected shortfall and their backtests.

Sign convention: VaR and ES are reported as **return levels**, so they are
negative for small ``alpha`` and a violation is ``returns < var``.
"""

from .backtests import (
    BacktestResult, christoffersen_conditional_coverage, christoffersen_independence,
    dynamic_quantile_test, expected_shortfall_backtest, kupiec_pof,
    var_backtest_table, violation_ratio, violations,
)

__all__ = [
    "BacktestResult", "violations", "violation_ratio", "kupiec_pof",
    "christoffersen_independence", "christoffersen_conditional_coverage",
    "dynamic_quantile_test", "expected_shortfall_backtest", "var_backtest_table",
]

from .garch_ddqn import GARCHDDQNVaR, QNetwork, ReplayBuffer, Transition

__all__ += ["GARCHDDQNVaR", "QNetwork", "ReplayBuffer", "Transition"]
