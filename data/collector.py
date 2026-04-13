"""
data/collector.py
Fetches OHLCV data from pykrx for three time horizons:
  - macro  : ~10 years of daily bars
  - mid    : 120 trading days
  - micro  : 60 trading days  (used to derive the latest signals)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock as krx


def _trading_days_start(n_days: int, reference: datetime | None = None) -> str:
    """
    Returns a date string (YYYYMMDD) that is approximately n_days trading days
    before the reference date.  Uses a 1.5x calendar-day multiplier to account
    for weekends and holidays, then pykrx trims to actual trading days.
    """
    ref = reference or datetime.today()
    calendar_days = int(n_days * 1.5)
    start = ref - timedelta(days=calendar_days)
    return start.strftime("%Y%m%d")


class OHLCVCollector:
    """
    Collects OHLCV data for a single Korean stock ticker across three horizons.

    Parameters
    ----------
    ticker : str
        KRX stock code, e.g. "005930"
    macro_context_days : int
        Number of trading days for the macro horizon (~10 yr = 2520)
    mid_context_days : int
        Number of trading days for the mid horizon (120)
    micro_context_days : int
        Number of trading days for the micro horizon (60)
    cache_dir : str | Path
        Directory to cache downloaded data as parquet files
    """

    def __init__(
        self,
        ticker: str,
        macro_context_days: int = 2520,
        mid_context_days: int = 120,
        micro_context_days: int = 60,
        cache_dir: str | Path = ".cache/data",
    ) -> None:
        self.ticker = ticker
        self.macro_context_days = macro_context_days
        self.mid_context_days = mid_context_days
        self.micro_context_days = micro_context_days
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._today = datetime.today()
        self._today_str = self._today.strftime("%Y%m%d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """
        Returns a dict with keys 'macro', 'mid', 'micro', each a DataFrame
        with columns: Open, High, Low, Close, Volume, Change
        sorted ascending by date index.
        """
        return {
            "macro": self._fetch("macro", self.macro_context_days),
            "mid":   self._fetch("mid",   self.mid_context_days),
            "micro": self._fetch("micro", self.micro_context_days),
        }

    def get_stock_name(self) -> str:
        try:
            name = krx.get_market_ticker_name(self.ticker)
            return name if name else self.ticker
        except Exception:
            return self.ticker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, horizon: str) -> Path:
        date_tag = self._today.strftime("%Y%m%d")
        return self.cache_dir / f"{self.ticker}_{horizon}_{date_tag}.parquet"

    def _fetch(self, horizon: str, n_days: int) -> pd.DataFrame:
        cache_file = self._cache_path(horizon)
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        start = _trading_days_start(n_days, self._today)
        df = krx.get_market_ohlcv_by_date(start, self._today_str, self.ticker)

        if df is None or df.empty:
            raise ValueError(
                f"No OHLCV data returned for ticker={self.ticker} "
                f"horizon={horizon} start={start} end={self._today_str}"
            )

        df = df.sort_index()
        df.index = pd.to_datetime(df.index)

        # pykrx returns Korean column names — rename to English
        df = df.rename(columns={
            "시가":   "Open",
            "고가":   "High",
            "저가":   "Low",
            "종가":   "Close",
            "거래량": "Volume",
            "등락률": "Change",
        })

        # Keep only standard OHLCV columns that exist
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "Change"] if c in df.columns]
        df = df[keep]

        # Trim to exactly n_days (pykrx may return extra rows due to calendar padding)
        if len(df) > n_days:
            df = df.iloc[-n_days:]

        df.to_parquet(cache_file)
        return df

    def describe(self) -> dict:
        """Returns a summary dict for logging / debugging."""
        return {
            "ticker": self.ticker,
            "name": self.get_stock_name(),
            "today": self._today_str,
            "horizons": {
                "macro": self.macro_context_days,
                "mid":   self.mid_context_days,
                "micro": self.micro_context_days,
            },
        }
