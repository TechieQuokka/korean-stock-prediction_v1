"""
data/preprocessor.py
Computes technical indicators on top of raw OHLCV DataFrames.
All indicators are added as new columns; no rows are dropped — NaN
values appear only at the start of the series where the window has not
filled yet.

Indicators computed:
  RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ATR(14),
  SMA(20), SMA(60), SMA(200), EMA(12), EMA(26), OBV
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    """
    Adds technical indicators to a dict of OHLCV DataFrames.

    Usage
    -----
    pp = Preprocessor()
    enriched = pp.process_all({"macro": df_macro, "mid": df_mid, "micro": df_micro})
    """

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_all(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {horizon: self._enrich(df.copy()) for horizon, df in data.items()}

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._enrich(df.copy())

    # ------------------------------------------------------------------
    # Core enrichment
    # ------------------------------------------------------------------

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # Moving averages
        df["SMA_20"]  = close.rolling(20).mean()
        df["SMA_60"]  = close.rolling(60).mean()
        df["SMA_200"] = close.rolling(200).mean()
        df["EMA_12"]  = close.ewm(span=12, adjust=False).mean()
        df["EMA_26"]  = close.ewm(span=26, adjust=False).mean()

        # RSI
        df["RSI_14"] = self._rsi(close, period=14)

        # MACD
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = self._macd(close)

        # Bollinger Bands
        df["BB_mid"], df["BB_upper"], df["BB_lower"] = self._bollinger(close)
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]

        # ATR
        df["ATR_14"] = self._atr(high, low, close, period=14)

        # OBV
        df["OBV"] = self._obv(close, volume)

        # Daily return
        df["Return_1d"] = close.pct_change()

        return df

    # ------------------------------------------------------------------
    # Indicator implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast   = close.ewm(span=fast,   adjust=False).mean()
        ema_slow   = close.ewm(span=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram  = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _bollinger(
        close: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid   = close.rolling(window).mean()
        std   = close.rolling(window).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        return mid, upper, lower

    @staticmethod
    def _atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    # ------------------------------------------------------------------
    # Summary helpers (used by analyzers)
    # ------------------------------------------------------------------

    @staticmethod
    def latest_signals(df: pd.DataFrame) -> dict:
        """Returns the most recent row's key indicators as a plain dict."""
        last = df.iloc[-1]
        return {
            "date":        str(df.index[-1].date()),
            "close":       round(float(last["Close"]), 2),
            "RSI_14":      round(float(last["RSI_14"]), 2) if not np.isnan(last["RSI_14"]) else None,
            "MACD":        round(float(last["MACD"]), 4)   if not np.isnan(last["MACD"])   else None,
            "MACD_signal": round(float(last["MACD_signal"]), 4) if not np.isnan(last["MACD_signal"]) else None,
            "MACD_hist":   round(float(last["MACD_hist"]), 4)   if not np.isnan(last["MACD_hist"])   else None,
            "BB_upper":    round(float(last["BB_upper"]), 2) if not np.isnan(last["BB_upper"]) else None,
            "BB_lower":    round(float(last["BB_lower"]), 2) if not np.isnan(last["BB_lower"]) else None,
            "SMA_20":      round(float(last["SMA_20"]), 2)  if not np.isnan(last["SMA_20"])  else None,
            "SMA_60":      round(float(last["SMA_60"]), 2)  if not np.isnan(last["SMA_60"])  else None,
            "ATR_14":      round(float(last["ATR_14"]), 2)  if not np.isnan(last["ATR_14"])  else None,
            "volume":      int(last["Volume"]),
        }
