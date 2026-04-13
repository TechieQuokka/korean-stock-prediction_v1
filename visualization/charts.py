"""
visualization/charts.py
Builds interactive Plotly HTML charts for all three time horizons
plus a summary dashboard.

Charts produced:
  1. macro_chart.html   - 10yr candlestick + SMA200 + quarterly forecast band
  2. mid_chart.html     - 120d candlestick + MACD + RSI + monthly forecast band
  3. micro_chart.html   - 60d candlestick + Bollinger Bands + weekly forecast
  4. dashboard.html     - single-page summary with all three forecast overlays
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartBuilder:
    """
    Parameters
    ----------
    ticker : str
    stock_name : str
    output_dir : str | Path
        Directory where HTML files are written
    """

    def __init__(
        self,
        ticker: str,
        stock_name: str,
        output_dir: str | Path = "reports/charts",
    ) -> None:
        self.ticker     = ticker
        self.stock_name = stock_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._today     = datetime.today().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_all(
        self,
        ohlcv: Dict[str, pd.DataFrame],
        forecasts: Dict[str, Dict[str, Any]],
        gemma_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Path]:
        """
        Builds all charts and returns a dict of {chart_name: Path}.
        """
        paths = {}
        paths["macro"]     = self._macro_chart(ohlcv["macro"],  forecasts["macro"])
        paths["mid"]       = self._mid_chart(ohlcv["mid"],      forecasts["mid"])
        paths["micro"]     = self._micro_chart(ohlcv["micro"],  forecasts["micro"])
        paths["dashboard"] = self._dashboard(ohlcv, forecasts, gemma_result)
        return paths

    # ------------------------------------------------------------------
    # Individual charts
    # ------------------------------------------------------------------

    def _macro_chart(self, df: pd.DataFrame, forecast: Dict) -> Path:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            subplot_titles=["Price + SMA200 + Quarterly Forecast", "Volume"],
            vertical_spacing=0.05,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLCV", increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # SMA200
        if "SMA_200" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["SMA_200"],
                name="SMA-200", line=dict(color="orange", width=1.5),
            ), row=1, col=1)

        # Forecast band
        self._add_forecast_band(fig, forecast, row=1)

        # Volume
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.6,
        ), row=2, col=1)

        fig.update_layout(
            title=f"{self.stock_name} ({self.ticker}) — Macro View (10 Year) | {self._today}",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700,
        )

        path = self.output_dir / f"{self.ticker}_macro.html"
        fig.write_html(str(path))
        print(f"[Chart] Saved: {path}")
        return path

    def _mid_chart(self, df: pd.DataFrame, forecast: Dict) -> Path:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=["Price + SMA60 + Monthly Forecast", "MACD", "RSI(14)"],
            vertical_spacing=0.05,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLCV", increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # SMA lines
        for col, color in [("SMA_20", "#64b5f6"), ("SMA_60", "orange")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=col.replace("_", "-"),
                    line=dict(color=color, width=1.2),
                ), row=1, col=1)

        # Forecast band
        self._add_forecast_band(fig, forecast, row=1)

        # MACD
        if "MACD" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["MACD"], name="MACD",
                line=dict(color="#80cbc4", width=1),
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["MACD_signal"], name="Signal",
                line=dict(color="#ffcc02", width=1),
            ), row=2, col=1)
            hist_colors = ["#26a69a" if v >= 0 else "#ef5350"
                           for v in df["MACD_hist"].fillna(0)]
            fig.add_trace(go.Bar(
                x=df.index, y=df["MACD_hist"], name="Histogram",
                marker_color=hist_colors, opacity=0.7,
            ), row=2, col=1)

        # RSI
        if "RSI_14" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["RSI_14"], name="RSI(14)",
                line=dict(color="#ce93d8", width=1.2),
            ), row=3, col=1)
            for level, color in [(70, "red"), (30, "green")]:
                fig.add_hline(y=level, line_dash="dash",
                              line_color=color, opacity=0.5, row=3, col=1)

        fig.update_layout(
            title=f"{self.stock_name} ({self.ticker}) — Mid-Term View (120 Days) | {self._today}",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=800,
        )

        path = self.output_dir / f"{self.ticker}_mid.html"
        fig.write_html(str(path))
        print(f"[Chart] Saved: {path}")
        return path

    def _micro_chart(self, df: pd.DataFrame, forecast: Dict) -> Path:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            subplot_titles=["Price + Bollinger Bands + Weekly Forecast", "Volume"],
            vertical_spacing=0.05,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLCV", increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # Bollinger Bands
        if "BB_upper" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_upper"], name="BB Upper",
                line=dict(color="rgba(100,181,246,0.6)", width=1),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_mid"], name="BB Mid",
                line=dict(color="rgba(100,181,246,0.4)", width=1, dash="dash"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_lower"], name="BB Lower",
                line=dict(color="rgba(100,181,246,0.6)", width=1),
                fill="tonexty", fillcolor="rgba(100,181,246,0.05)",
            ), row=1, col=1)

        # Forecast band (weekly)
        self._add_forecast_band(fig, forecast, row=1)

        # Volume
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.6,
        ), row=2, col=1)

        fig.update_layout(
            title=f"{self.stock_name} ({self.ticker}) — Micro View (60 Days) | {self._today}",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=650,
        )

        path = self.output_dir / f"{self.ticker}_micro.html"
        fig.write_html(str(path))
        print(f"[Chart] Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _dashboard(
        self,
        ohlcv: Dict[str, pd.DataFrame],
        forecasts: Dict[str, Dict],
        gemma_result: Dict | None,
    ) -> Path:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                "Macro — Quarterly Forecast",
                "Mid — Monthly Forecast",
                "Micro — Weekly Forecast",
            ],
            horizontal_spacing=0.06,
        )

        horizon_col_map = {"macro": 1, "mid": 2, "micro": 3}

        for horizon, col in horizon_col_map.items():
            df  = ohlcv[horizon]
            fc  = forecasts[horizon]

            # Show only last 60 bars of each horizon for dashboard readability
            df_disp = df.tail(60)

            fig.add_trace(go.Scatter(
                x=df_disp.index, y=df_disp["Close"],
                name=f"{horizon.capitalize()} Close",
                line=dict(width=1.5),
            ), row=1, col=col)

            # Add forecast median line
            future_x = pd.to_datetime(fc["dates"])
            fig.add_trace(go.Scatter(
                x=future_x, y=fc["q50"],
                name=f"{horizon.capitalize()} Q50",
                line=dict(dash="dash", width=1.5),
            ), row=1, col=col)

            # Add uncertainty band
            fig.add_trace(go.Scatter(
                x=list(future_x) + list(future_x[::-1]),
                y=fc["q90"] + fc["q10"][::-1],
                fill="toself",
                fillcolor="rgba(255,165,0,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{horizon.capitalize()} Q10-Q90",
            ), row=1, col=col)

        # Gemma recommendation annotation
        recommendation_text = ""
        if gemma_result:
            rec    = gemma_result.get("recommendation", "N/A")
            conf   = gemma_result.get("confidence", 0)
            colors = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
            recommendation_text = (
                f"Gemma4 Recommendation: <b>{rec}</b> "
                f"(Confidence: {conf}%)"
            )
            fig.add_annotation(
                text=recommendation_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.08,
                showarrow=False,
                font=dict(size=16, color=colors.get(rec, "white")),
            )

        fig.update_layout(
            title=(
                f"{self.stock_name} ({self.ticker}) — "
                f"Multi-Horizon Dashboard | {self._today}"
            ),
            template="plotly_dark",
            height=500,
            showlegend=False,
        )

        path = self.output_dir / f"{self.ticker}_dashboard.html"
        fig.write_html(str(path))
        print(f"[Chart] Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _add_forecast_band(
        self,
        fig: go.Figure,
        forecast: Dict[str, Any],
        row: int,
        col: int = 1,
    ) -> None:
        future_x = pd.to_datetime(forecast["dates"])
        q10 = forecast["q10"]
        q50 = forecast["q50"]
        q90 = forecast["q90"]

        # Median line
        fig.add_trace(go.Scatter(
            x=future_x, y=q50,
            name="Forecast Q50",
            line=dict(color="gold", width=2, dash="dash"),
        ), row=row, col=col)

        # Uncertainty band
        fig.add_trace(go.Scatter(
            x=list(future_x) + list(future_x[::-1]),
            y=q90 + q10[::-1],
            fill="toself",
            fillcolor="rgba(255,165,0,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Forecast Q10-Q90",
        ), row=row, col=col)
