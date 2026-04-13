"""
models/timesfm_runner.py
Runs TimesFM 2.5-200M for three time horizons sequentially.
After all three forecasts are done, the model is explicitly
released from GPU memory before Gemma4 is called.

Forecast output per horizon:
  {
    "horizon":      "macro" | "mid" | "micro",
    "context_len":  int,
    "forecast_len": int,
    "dates":        List[str],        # forecast date strings
    "q10":          List[float],      # pessimistic band
    "q50":          List[float],      # median (point forecast)
    "q90":          List[float],      # optimistic band
    "last_close":   float,
  }
"""

from __future__ import annotations

import gc
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import timesfm


_TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"


def _future_dates(last_date: pd.Timestamp, n: int) -> List[str]:
    """Returns n business-day date strings after last_date."""
    dates = []
    current = last_date
    while len(dates) < n:
        current = current + timedelta(days=1)
        if current.weekday() < 5:   # Mon-Fri only
            dates.append(current.strftime("%Y-%m-%d"))
    return dates


class TimesFMRunner:
    """
    Loads TimesFM once and runs macro / mid / micro forecasts sequentially.

    Parameters
    ----------
    repo : str
        HuggingFace repo id (resolved from local cache automatically)
    macro_context : int
        Number of trading days for macro context
    mid_context : int
        Number of trading days for mid context
    micro_context : int
        Number of trading days for micro context
    macro_horizon : int
        Forecast length for macro
    mid_horizon : int
        Forecast length for mid
    micro_horizon : int
        Forecast length for micro
    """

    def __init__(
        self,
        repo: str = _TIMESFM_REPO,
        macro_context: int  = 2520,
        mid_context: int    = 120,
        micro_context: int  = 60,
        macro_horizon: int  = 60,
        mid_horizon: int    = 20,
        micro_horizon: int  = 5,
    ) -> None:
        self.repo           = repo
        self.macro_context  = macro_context
        self.mid_context    = mid_context
        self.micro_context  = micro_context
        self.macro_horizon  = macro_horizon
        self.mid_horizon    = mid_horizon
        self.micro_horizon  = micro_horizon

        self._model: timesfm.TimesFM_2p5_200M_torch | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_all(self, ohlcv: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """
        Runs macro, mid, micro forecasts and returns results dict.
        The model is loaded once and released after all three calls.
        """
        self._load_model()
        try:
            results = {
                "macro": self._forecast(ohlcv["macro"], "macro",
                                        self.macro_context, self.macro_horizon),
                "mid":   self._forecast(ohlcv["mid"],   "mid",
                                        self.mid_context,   self.mid_horizon),
                "micro": self._forecast(ohlcv["micro"], "micro",
                                        self.micro_context, self.micro_horizon),
            }
        finally:
            self._release_model()

        return results

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        print("[TimesFM] Loading model...")
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo)

        forecast_config = timesfm.ForecastConfig(
            max_context=self.macro_context,          # largest context across all horizons
            max_horizon=self.macro_horizon,           # largest horizon across all horizons
            normalize_inputs=True,
            use_continuous_quantile_head=True,        # enables Q10/Q50/Q90 output
        )
        self._model.compile(forecast_config)
        print("[TimesFM] Model loaded and compiled.")

    def _release_model(self) -> None:
        del self._model
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[TimesFM] Model released from GPU memory.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forecast(
        self,
        df: pd.DataFrame,
        horizon_name: str,
        context_len: int,
        forecast_len: int,
    ) -> dict:
        print(f"[TimesFM] Forecasting horizon={horizon_name} "
              f"context={context_len} forecast={forecast_len}")

        close_series = df["Close"].values.astype(np.float32)

        # Trim to context_len
        if len(close_series) > context_len:
            close_series = close_series[-context_len:]

        # API: forecast(horizon, inputs) → (point_forecasts, quantile_forecasts)
        # point_forecasts  shape: (batch, horizon)
        # quantile_forecasts shape: (batch, horizon, n_quantiles)
        point_forecasts, quantile_forecasts = self._model.forecast(
            horizon=forecast_len,
            inputs=[close_series],
        )

        q50 = point_forecasts[0][:forecast_len].tolist()

        # quantile_forecasts may be None or shaped differently depending on version
        if quantile_forecasts is not None and len(quantile_forecasts) > 0:
            qf = quantile_forecasts[0]  # shape: (horizon, n_quantiles) or (n_quantiles, horizon)
            qf = np.array(qf)
            if qf.ndim == 2 and qf.shape[0] == forecast_len:
                # shape: (horizon, n_quantiles)
                q10 = qf[:, 0].tolist()
                q90 = qf[:, -1].tolist()
            elif qf.ndim == 2 and qf.shape[1] == forecast_len:
                # shape: (n_quantiles, horizon)
                q10 = qf[0, :].tolist()
                q90 = qf[-1, :].tolist()
            else:
                q10 = q50
                q90 = q50
        else:
            q10 = q50
            q90 = q50

        last_date   = df.index[-1]
        last_close  = float(df["Close"].iloc[-1])
        future_dates = _future_dates(last_date, forecast_len)

        return {
            "horizon":      horizon_name,
            "context_len":  context_len,
            "forecast_len": forecast_len,
            "dates":        future_dates,
            "q10":          q10,
            "q50":          q50,
            "q90":          q90,
            "last_close":   last_close,
        }
