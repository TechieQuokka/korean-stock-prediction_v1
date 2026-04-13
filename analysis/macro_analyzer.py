"""
analysis/macro_analyzer.py
Builds the macro (10-year) context block that is injected into the Gemma4 prompt.
Focuses on:
  - Long-term price trend and major cycles
  - 200-day SMA relationship (golden/death cross)
  - Annual return summary
  - TimesFM quarterly forecast interpretation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd


class MacroAnalyzer:
    """
    Converts 10-year OHLCV + TimesFM macro forecast into a prompt fragment.
    """

    def __init__(self, ticker: str, stock_name: str) -> None:
        self.ticker     = ticker
        self.stock_name = stock_name
        self._today     = datetime.today().strftime("%Y-%m-%d")

    def build_context(
        self,
        df: pd.DataFrame,
        forecast: Dict[str, Any],
        news: Dict[str, Any] | None = None,
    ) -> str:
        """
        Returns a text block describing the macro perspective.
        """
        lines = [
            f"=== MACRO ANALYSIS (10-Year Perspective) ===",
            f"Ticker      : {self.ticker}",
            f"Stock Name  : {self.stock_name}",
            f"Analysis Date: {self._today}",
            f"Data Range  : {str(df.index[0].date())} to {str(df.index[-1].date())}",
            f"Total Trading Days: {len(df)}",
            "",
        ]

        # --- Price summary ---
        close = df["Close"]
        lines += [
            "[ Price Summary - 10 Year ]",
            f"  Start Price : {int(close.iloc[0]):,}",
            f"  End Price   : {int(close.iloc[-1]):,}",
            f"  All-Time High (in period): {int(close.max()):,}",
            f"  All-Time Low  (in period): {int(close.min()):,}",
            f"  10-Year Return: {((close.iloc[-1] / close.iloc[0]) - 1) * 100:.1f}%",
            "",
        ]

        # --- Yearly return breakdown ---
        df_copy = df.copy()
        df_copy["Year"] = df_copy.index.year
        yearly = df_copy.groupby("Year")["Close"].agg(["first", "last"])
        yearly["return_pct"] = (yearly["last"] / yearly["first"] - 1) * 100
        lines.append("[ Annual Returns ]")
        for year, row in yearly.iterrows():
            direction = "+" if row["return_pct"] >= 0 else ""
            lines.append(f"  {year}: {direction}{row['return_pct']:.1f}%")
        lines.append("")

        # --- SMA trend ---
        last = df.iloc[-1]
        sma200 = last.get("SMA_200", np.nan)
        current_price = float(last["Close"])
        if not np.isnan(sma200):
            diff_pct = ((current_price - float(sma200)) / float(sma200)) * 100
            trend    = "above" if diff_pct >= 0 else "below"
            lines += [
                "[ Long-Term Trend (200-day SMA) ]",
                f"  Current Price : {int(current_price):,}",
                f"  SMA-200       : {int(sma200):,}",
                f"  Price is {abs(diff_pct):.1f}% {trend} SMA-200",
                "",
            ]

        # --- TimesFM macro forecast ---
        q50  = forecast["q50"]
        q10  = forecast["q10"]
        q90  = forecast["q90"]
        dates = forecast["dates"]
        lines += [
            f"[ TimesFM Quarterly Forecast (next {forecast['forecast_len']} trading days) ]",
            f"  Last Close    : {int(forecast['last_close']):,}",
            f"  Forecast End Date: {dates[-1]}",
            f"  Median (Q50)  : {int(q50[-1]):,}",
            f"  Optimistic (Q90): {int(q90[-1]):,}",
            f"  Pessimistic (Q10): {int(q10[-1]):,}",
            f"  Expected Change: {((q50[-1] / forecast['last_close']) - 1) * 100:+.1f}%",
            "",
        ]

        # --- Market news summary (macro-relevant) ---
        if news and news.get("news"):
            lines.append("[ Recent News Headlines (as of {}) ]".format(self._today))
            for item in news["news"][:5]:
                lines.append(f"  - {item['headline']} ({item.get('date', '')})")
            lines.append("")

        return "\n".join(lines)
