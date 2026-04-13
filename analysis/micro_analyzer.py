"""
analysis/micro_analyzer.py
Builds the micro (short-term, 1-week horizon) context block for the Gemma4 prompt.
Focuses on:
  - Bollinger Band position
  - ATR-based volatility
  - Recent 5-day price action
  - Foreign/institution net buying
  - TimesFM weekly forecast
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd


class MicroAnalyzer:
    """
    Converts 60-day OHLCV + TimesFM micro forecast + scrape data
    into a short-term prompt fragment.
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
            "=== MICRO ANALYSIS (Short-Term / 1-Week Perspective) ===",
            f"Analysis Date: {self._today}",
            f"Data Range   : {str(df.index[0].date())} to {str(df.index[-1].date())}",
            "",
        ]

        last  = df.iloc[-1]
        close = df["Close"]

        # --- Last 5 trading days ---
        recent5 = df.tail(5)
        lines.append("[ Last 5 Trading Days ]")
        for idx, row in recent5.iterrows():
            chg = row.get("Return_1d", np.nan)
            chg_str = f"{chg * 100:+.2f}%" if not np.isnan(chg) else "N/A"
            lines.append(
                f"  {str(idx.date())}  Close={int(row['Close']):>9,}  "
                f"Vol={int(row['Volume']):>12,}  Change={chg_str}"
            )
        lines.append("")

        # --- Bollinger Bands ---
        bb_upper = last.get("BB_upper", np.nan)
        bb_lower = last.get("BB_lower", np.nan)
        bb_mid   = last.get("BB_mid",   np.nan)
        current  = float(last["Close"])
        if not np.isnan(bb_upper) and not np.isnan(bb_lower):
            bb_width = float(last.get("BB_width", np.nan))
            position = (current - float(bb_lower)) / (float(bb_upper) - float(bb_lower)) * 100
            bb_signal = (
                "Near Upper Band (potential resistance)"
                if position > 80
                else "Near Lower Band (potential support)"
                if position < 20
                else "Mid Band (neutral)"
            )
            lines += [
                "[ Bollinger Bands (20,2) ]",
                f"  Upper : {int(bb_upper):,}",
                f"  Mid   : {int(bb_mid):,}",
                f"  Lower : {int(bb_lower):,}",
                f"  Position in band: {position:.1f}%  -> {bb_signal}",
                f"  Band Width: {bb_width:.4f}",
                "",
            ]

        # --- ATR / Volatility ---
        atr = last.get("ATR_14", np.nan)
        if not np.isnan(atr):
            atr_pct = float(atr) / current * 100
            lines += [
                "[ ATR(14) Volatility ]",
                f"  ATR Value : {int(atr):,}",
                f"  ATR as %  : {atr_pct:.2f}% of current price",
                "",
            ]

        # --- TimesFM weekly forecast ---
        q50   = forecast["q50"]
        q10   = forecast["q10"]
        q90   = forecast["q90"]
        dates = forecast["dates"]
        lines += [
            f"[ TimesFM Weekly Forecast (next {forecast['forecast_len']} trading days) ]",
            f"  Last Close     : {int(forecast['last_close']):,}",
        ]
        for i, (d, p50, p10, p90) in enumerate(zip(dates, q50, q10, q90)):
            chg = ((p50 / forecast["last_close"]) - 1) * 100
            lines.append(
                f"  Day {i+1} ({d}): "
                f"Q50={int(p50):,}  Q10={int(p10):,}  Q90={int(p90):,}  "
                f"({chg:+.1f}%)"
            )
        lines.append("")

        return "\n".join(lines)
