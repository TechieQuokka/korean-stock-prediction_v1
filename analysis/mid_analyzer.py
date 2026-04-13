"""
analysis/mid_analyzer.py
Builds the mid-term (120-day) context block for the Gemma4 prompt.
Focuses on:
  - Medium-term momentum (MACD, RSI)
  - 60-day SMA trend
  - Volume trend
  - TimesFM monthly forecast interpretation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd


class MidAnalyzer:
    """
    Converts 120-day OHLCV + TimesFM mid forecast into a prompt fragment.
    """

    def __init__(self, ticker: str, stock_name: str) -> None:
        self.ticker     = ticker
        self.stock_name = stock_name
        self._today     = datetime.today().strftime("%Y-%m-%d")

    def build_context(
        self,
        df: pd.DataFrame,
        forecast: Dict[str, Any],
    ) -> str:
        lines = [
            "=== MID-TERM ANALYSIS (120-Day Perspective) ===",
            f"Analysis Date: {self._today}",
            f"Data Range   : {str(df.index[0].date())} to {str(df.index[-1].date())}",
            "",
        ]

        last  = df.iloc[-1]
        close = df["Close"]

        # --- Price action ---
        period_high = float(close.max())
        period_low  = float(close.min())
        current     = float(close.iloc[-1])
        lines += [
            "[ Price Action - 120 Days ]",
            f"  Current Price: {int(current):,}",
            f"  120d High    : {int(period_high):,}",
            f"  120d Low     : {int(period_low):,}",
            f"  Position in range: {((current - period_low) / (period_high - period_low)) * 100:.1f}%",
            "",
        ]

        # --- RSI ---
        rsi = last.get("RSI_14", np.nan)
        if not np.isnan(rsi):
            rsi_f  = float(rsi)
            rsi_zone = (
                "Overbought (>70)" if rsi_f > 70
                else "Oversold (<30)"  if rsi_f < 30
                else "Neutral"
            )
            lines += [
                "[ RSI(14) ]",
                f"  Value : {rsi_f:.1f}",
                f"  Zone  : {rsi_zone}",
                "",
            ]

        # --- MACD ---
        macd      = last.get("MACD",        np.nan)
        macd_sig  = last.get("MACD_signal", np.nan)
        macd_hist = last.get("MACD_hist",   np.nan)
        if not np.isnan(macd):
            cross = "Bullish" if float(macd) > float(macd_sig) else "Bearish"
            lines += [
                "[ MACD(12,26,9) ]",
                f"  MACD Line   : {float(macd):.2f}",
                f"  Signal Line : {float(macd_sig):.2f}",
                f"  Histogram   : {float(macd_hist):.2f}",
                f"  Cross       : {cross}",
                "",
            ]

        # --- 60-day SMA ---
        sma60 = last.get("SMA_60", np.nan)
        if not np.isnan(sma60):
            diff_pct = ((current - float(sma60)) / float(sma60)) * 100
            lines += [
                "[ SMA-60 ]",
                f"  Value  : {int(sma60):,}",
                f"  Offset : {diff_pct:+.1f}%",
                "",
            ]

        # --- Volume trend ---
        vol_series = df["Volume"]
        avg_vol    = float(vol_series.mean())
        last_vol   = float(vol_series.iloc[-1])
        vol_ratio  = last_vol / avg_vol if avg_vol > 0 else 1.0
        lines += [
            "[ Volume Trend ]",
            f"  120d Avg Volume : {int(avg_vol):,}",
            f"  Latest Volume   : {int(last_vol):,}",
            f"  Ratio vs Avg    : {vol_ratio:.2f}x",
            "",
        ]

        # --- TimesFM mid forecast ---
        q50   = forecast["q50"]
        q10   = forecast["q10"]
        q90   = forecast["q90"]
        dates = forecast["dates"]
        lines += [
            f"[ TimesFM Monthly Forecast (next {forecast['forecast_len']} trading days) ]",
            f"  Last Close     : {int(forecast['last_close']):,}",
            f"  Forecast End   : {dates[-1]}",
            f"  Median  (Q50)  : {int(q50[-1]):,}",
            f"  Optimistic(Q90): {int(q90[-1]):,}",
            f"  Pessimistic(Q10): {int(q10[-1]):,}",
            f"  Expected Change: {((q50[-1] / forecast['last_close']) - 1) * 100:+.1f}%",
            "",
        ]

        return "\n".join(lines)
