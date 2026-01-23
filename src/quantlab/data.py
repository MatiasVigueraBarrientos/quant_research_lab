from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PriceRequest:
    tickers: tuple[str, ...]
    start: str
    end: str

    def cache_name(self) -> str:
        key = "|".join(self.tickers) + f"|{self.start}|{self.end}"
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        return f"prices_{h}.csv"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fetch_prices_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/raw",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily adjusted prices from Yahoo via yfinance and cache to CSV.

    Returns a DataFrame indexed by date, columns=tickers, values=adjusted close prices.
    """
    tickers_t = tuple(tickers)
    req = PriceRequest(tickers=tickers_t, start=start, end=end)

    cache_dir_p = Path(cache_dir)
    _ensure_dir(cache_dir_p)
    cache_path = cache_dir_p / req.cache_name()

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        return df.sort_index()

    import yfinance as yf  # local import to keep module import fast

    raw = yf.download(
        list(tickers_t),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # yfinance returns either a single-index (single ticker) or multi-index columns
    if isinstance(raw.columns, pd.MultiIndex):
        # auto_adjust=True -> use 'Close'
        close = raw["Close"].copy()
    else:
        # single ticker case
        close = raw.rename("Close").to_frame()

    close.index.name = "Date"
    close = close.sort_index()

    # Ensure columns are tickers (not weird single column name)
    if close.shape[1] == 1 and close.columns[0] == "Close" and len(tickers_t) == 1:
        close.columns = [tickers_t[0]]

    # Clean: drop days with all NaN, forward-fill small gaps
    close = close.dropna(how="all")
    close = close.ffill(limit=5)

    close.to_csv(cache_path)
    return close
