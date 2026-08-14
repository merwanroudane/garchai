"""Real financial datasets bundled with the package.

The build requirements forbid simulated data in teaching examples, so these
series ship with the package and every example runs offline and reproducibly.
"""

from .loaders import AVAILABLE, DatasetInfo, fetch_yahoo, list_datasets, load, load_returns, log_returns

__all__ = [
    "DatasetInfo", "AVAILABLE", "load", "load_returns", "list_datasets",
    "fetch_yahoo", "log_returns",
]
