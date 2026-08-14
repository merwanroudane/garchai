r"""Real financial datasets, bundled with the package.

The build requirements are explicit that teaching examples must use real
economic or financial data, not simulation. Two ways to honour that were
available, and this module takes the first:

1. **Bundle** compressed daily CSVs for the exact series the Tier-S papers use.
   Every example then runs offline, reproduces byte-for-byte on any machine, and
   keeps reproducing in five years when the vendor API has changed twice.
2. Fetch live at example-run time. Rejected: it makes the documentation fragile,
   non-reproducible, and dependent on a network at exactly the moment a new user
   is trying the package for the first time.

:func:`fetch_yahoo` is still provided, for users who want current data or a
series we do not ship. It is never used by the examples or the tests.

Simulated data has one legitimate home in this package and it is not here: the
amortized estimator of De Clerk & Savel'ev (2022) is *trained* on analytically
generated moment pairs. That is training data for an estimator, not a worked
example, and :mod:`garchai.structural.amortized` generates it explicitly.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DatasetInfo",
    "AVAILABLE",
    "load",
    "load_returns",
    "list_datasets",
    "fetch_yahoo",
    "log_returns",
]


@dataclass(frozen=True)
class DatasetInfo:
    """Provenance of a bundled series."""

    key: str
    name: str
    ticker: str
    description: str
    used_by: tuple[str, ...]
    """Paper ids (see ``resources/papers.yaml``) that use this series."""

    source: str = "Yahoo Finance"


AVAILABLE: dict[str, DatasetInfo] = {
    "sp500": DatasetInfo(
        key="sp500",
        name="S&P 500 Index",
        ticker="^GSPC",
        description="US large-cap equity index, daily close.",
        used_by=("zhao2024_garch_nn", "wei2025_garch_rnn", "xu2024_ginn",
                 "yin2022_neural_garch", "rodikov2022_sigma_lstm", "liu2020_ann_garch"),
    ),
    "dji": DatasetInfo(
        key="dji",
        name="Dow Jones Industrial Average",
        ticker="^DJI",
        description="US blue-chip equity index, daily close.",
        used_by=("zhao2024_garch_nn", "wei2025_garch_rnn", "xu2024_ginn",
                 "yin2022_neural_garch"),
    ),
    "nasdaq": DatasetInfo(
        key="nasdaq",
        name="NASDAQ Composite",
        ticker="^IXIC",
        description="US technology-weighted equity index, daily close.",
        used_by=("zhao2024_garch_nn", "wei2025_garch_rnn", "xu2024_ginn",
                 "yin2022_neural_garch"),
    ),
    "eurusd": DatasetInfo(
        key="eurusd",
        name="EUR/USD exchange rate",
        ticker="EURUSD=X",
        description="Euro against the US dollar, daily close.",
        used_by=("zhao2024_garch_nn", "yin2022_neural_garch"),
    ),
    "gold": DatasetInfo(
        key="gold",
        name="Gold futures",
        ticker="GC=F",
        description="COMEX gold front-month futures, daily close.",
        used_by=("zhao2024_garch_nn", "yin2022_neural_garch"),
    ),
    "bitcoin": DatasetInfo(
        key="bitcoin",
        name="Bitcoin / USD",
        ticker="BTC-USD",
        description="Bitcoin against the US dollar, daily close.",
        used_by=("rodikov2022_sigma_lstm", "rubio2026_stacked_ml_garch"),
    ),
    "ftse": DatasetInfo(
        key="ftse",
        name="FTSE 100",
        ticker="^FTSE",
        description="UK large-cap equity index, daily close.",
        used_by=("xu2024_ginn", "yin2022_neural_garch"),
    ),
    "nikkei": DatasetInfo(
        key="nikkei",
        name="Nikkei 225",
        ticker="^N225",
        description="Japanese equity index, daily close.",
        used_by=("xu2024_ginn",),
    ),
    "hangseng": DatasetInfo(
        key="hangseng",
        name="Hang Seng Index",
        ticker="^HSI",
        description="Hong Kong equity index, daily close.",
        used_by=("xu2024_ginn",),
    ),
}


def _data_dir() -> Path:
    return Path(resources.files("garchai") / "datasets" / "data")


def list_datasets() -> pd.DataFrame:
    """Return a table of the bundled datasets and which papers use each.

    Examples
    --------
    >>> table = list_datasets()
    >>> "sp500" in set(table["key"])
    True
    """
    rows = []
    for info in AVAILABLE.values():
        path = _data_dir() / f"{info.key}.csv.gz"
        rows.append(
            {
                "key": info.key,
                "name": info.name,
                "ticker": info.ticker,
                "bundled": path.exists(),
                "used_by": ", ".join(info.used_by),
            }
        )
    return pd.DataFrame(rows)


def load(key: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load a bundled daily price series.

    Parameters
    ----------
    key
        One of :data:`AVAILABLE`.
    start, end
        Optional ISO date bounds, inclusive.

    Returns
    -------
    pandas.DataFrame
        Indexed by date, with a ``close`` column and, where available,
        ``open``, ``high``, ``low`` and ``volume``.
    """
    if key not in AVAILABLE:
        raise KeyError(f"Unknown dataset {key!r}. Available: {sorted(AVAILABLE)}")
    path = _data_dir() / f"{key}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled data for {key!r} is missing at {path}. Regenerate it with "
            "`python -m garchai.datasets.build`, or fetch a fresh copy with "
            "`garchai.datasets.fetch_yahoo`."
        )
    # Each file carries a '#'-prefixed provenance header recording the ticker
    # and download date, so a bundled series can be traced to what produced it.
    with gzip.open(path, "rt") as handle:
        frame = pd.read_csv(handle, comment="#", parse_dates=["date"], index_col="date")
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    return frame


def log_returns(prices: pd.Series | np.ndarray, percent: bool = True) -> pd.Series:
    r"""Daily log returns, :math:`r_t = \log(p_t/p_{t-1})`.

    Parameters
    ----------
    percent
        Multiply by 100. Every paper in this collection works in percent, and
        it keeps :math:`\omega` at a numerically sane magnitude — see
        :func:`garchai.core.config.resolve_scale` for why that matters.
    """
    series = pd.Series(prices).astype(float)
    out = np.log(series).diff().dropna()
    return out * 100.0 if percent else out


def load_returns(
    key: str,
    start: str | None = None,
    end: str | None = None,
    percent: bool = True,
) -> pd.Series:
    """Load a bundled series and return its daily log returns.

    Examples
    --------
    >>> r = load_returns("sp500", start="2015-01-01", end="2019-12-31")  # doctest: +SKIP
    >>> float(r.std()) > 0                                              # doctest: +SKIP
    True
    """
    frame = load(key, start=start, end=end)
    return log_returns(frame["close"], percent=percent)


def fetch_yahoo(
    ticker: str,
    start: str = "2000-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download a daily price series from Yahoo Finance.

    Requires ``yfinance``. Provided for users who want current data or a series
    this package does not bundle; the examples and tests never call it, so that
    they keep working without a network.
    """
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fetch_yahoo requires yfinance. Install it with "
            "`pip install garchai[data]`."
        ) from exc

    raw = yfinance.download(
        ticker, start=start, end=end, progress=False, auto_adjust=False
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
    frame = frame[keep].dropna(subset=["close"])
    frame.index.name = "date"
    return frame
