"""Regenerate the bundled datasets from Yahoo Finance.

Run as ``python -m garchai.datasets.build``. This is a maintainer script, not
part of the user-facing API: the CSVs it writes are committed to the repository
so that end users never need a network to run an example.

Every file is written with an explicit provenance header recording the ticker,
the download date and the row count, so a bundled series can always be traced
back to what produced it.
"""

from __future__ import annotations

import argparse
import gzip
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .loaders import AVAILABLE, fetch_yahoo

DEFAULT_START = "1990-01-01"


def build(
    keys: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Download and write the compressed CSVs. Returns a summary table."""
    out_dir = out_dir or Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = keys or list(AVAILABLE)

    rows = []
    for key in targets:
        info = AVAILABLE[key]
        try:
            frame = fetch_yahoo(info.ticker, start=start, end=end)
        except Exception as exc:  # noqa: BLE001 - report and continue
            rows.append({"key": key, "ticker": info.ticker, "rows": 0,
                         "status": f"FAILED: {exc}"})
            continue

        path = out_dir / f"{key}.csv.gz"
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        header = (
            f"# garchai bundled dataset\n"
            f"# key={key} name={info.name} ticker={info.ticker}\n"
            f"# source={info.source} downloaded={stamp}\n"
            f"# rows={len(frame)} range={frame.index[0].date()}..{frame.index[-1].date()}\n"
        )
        with gzip.open(path, "wt", newline="") as handle:
            handle.write(header)
            frame.to_csv(handle, index_label="date")

        rows.append(
            {
                "key": key,
                "ticker": info.ticker,
                "rows": len(frame),
                "start": str(frame.index[0].date()),
                "end": str(frame.index[-1].date()),
                "kb": round(path.stat().st_size / 1024, 1),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="*", default=None, help="subset of dataset keys")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    summary = build(keys=args.keys, start=args.start, end=args.end)
    print(summary.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
