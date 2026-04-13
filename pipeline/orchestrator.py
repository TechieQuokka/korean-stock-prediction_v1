"""
pipeline/orchestrator.py
Coordinates the full prediction pipeline in strict sequential order
to stay within 12 GB VRAM:

  Step 1  Collect OHLCV data (pykrx)
  Step 2  Fetch news — Google News RSS (stock-specific + macro topics)
  Step 3  Compute technical indicators
  Step 4  Run TimesFM (macro + mid + micro) — GPU
  Step 5  Release TimesFM from GPU
  Step 6  Build analysis context (macro / mid / micro text blocks)
  Step 7  Gemma4 — news event analysis (stock sentiment + macro impact)
  Step 8  Gemma4 — final investment judgment (TimesFM + indicators + news)
  Step 9  Generate charts (plotly)
  Step 10 Save JSON report + print console summary
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from config.settings import Settings
from data.collector import OHLCVCollector
from data.news_fetcher import NewsFetcher
from data.preprocessor import Preprocessor
from models.timesfm_runner import TimesFMRunner
from models.gemma_client import GemmaClient
from analysis.macro_analyzer import MacroAnalyzer
from analysis.mid_analyzer import MidAnalyzer
from analysis.micro_analyzer import MicroAnalyzer
from analysis.news_analyzer import NewsAnalyzer
from visualization.charts import ChartBuilder


class Orchestrator:
    """
    Parameters
    ----------
    ticker : str
        KRX stock code (e.g. "005930")
    settings : Settings
        Loaded from config.toml
    output_dir : str | Path | None
        Override the default reports directory from settings
    """

    def __init__(
        self,
        ticker: str,
        settings: Settings,
        output_dir: str | Path | None = None,
    ) -> None:
        self.ticker  = ticker
        self.cfg     = settings
        self.out_dir = Path(output_dir) if output_dir else Path(settings.output.reports_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        today = datetime.today().strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"  Korean Stock Prediction Pipeline")
        print(f"  Ticker : {self.ticker}")
        print(f"  Date   : {today}")
        print(f"{'='*60}\n")

        # ----------------------------------------------------------
        # Step 1: OHLCV collection
        # ----------------------------------------------------------
        print("[Step 1/10] Collecting OHLCV data...")
        collector = OHLCVCollector(
            ticker=self.ticker,
            macro_context_days=self.cfg.timesfm.macro_context_days,
            mid_context_days=self.cfg.timesfm.mid_context_days,
            micro_context_days=self.cfg.timesfm.micro_context_days,
            cache_dir=self.cfg.data.cache_dir,
        )
        ohlcv_raw  = collector.fetch_all()
        stock_name = collector.get_stock_name()
        print(f"  Stock name : {stock_name}")
        for h, df in ohlcv_raw.items():
            print(f"  {h:6s} : {len(df)} rows  "
                  f"({str(df.index[0].date())} ~ {str(df.index[-1].date())})")

        # ----------------------------------------------------------
        # Step 2: News collection (Google News RSS)
        # ----------------------------------------------------------
        print("\n[Step 2/10] Fetching news (Google News RSS)...")
        news_fetcher = NewsFetcher(
            stock_name=stock_name,
            ticker=self.ticker,
            max_stock_news=self.cfg.scraping.max_news_items,
            max_macro_per_topic=3,
            delay_sec=self.cfg.scraping.naver_delay_sec,
            timeout=self.cfg.scraping.request_timeout,
        )
        news_data = news_fetcher.fetch_all()
        n_stock = len(news_data.get("stock_news", []))
        n_macro = sum(len(v) for v in news_data.get("macro_news", {}).values())
        print(f"  Stock news   : {n_stock} items")
        print(f"  Macro news   : {n_macro} items across "
              f"{len(news_data.get('macro_news', {}))} topics")

        # ----------------------------------------------------------
        # Step 3: Technical indicators
        # ----------------------------------------------------------
        print("\n[Step 3/10] Computing technical indicators...")
        pp     = Preprocessor()
        ohlcv  = pp.process_all(ohlcv_raw)
        latest = Preprocessor.latest_signals(ohlcv["micro"])
        print(f"  Close={latest['close']:,}  RSI={latest['RSI_14']}  "
              f"MACD_hist={latest['MACD_hist']}")

        # ----------------------------------------------------------
        # Step 4 & 5: TimesFM inference + GPU release
        # ----------------------------------------------------------
        print("\n[Step 4/10] Running TimesFM (macro / mid / micro)...")
        runner = TimesFMRunner(
            repo=self.cfg.model.timesfm_repo,
            macro_context=self.cfg.timesfm.macro_context_days,
            mid_context=self.cfg.timesfm.mid_context_days,
            micro_context=self.cfg.timesfm.micro_context_days,
            macro_horizon=self.cfg.timesfm.macro_horizon_days,
            mid_horizon=self.cfg.timesfm.mid_horizon_days,
            micro_horizon=self.cfg.timesfm.micro_horizon_days,
        )
        forecasts = runner.run_all(ohlcv)
        print("[Step 5/10] TimesFM released from GPU.")

        # ----------------------------------------------------------
        # Step 6: Build analysis context blocks
        # ----------------------------------------------------------
        print("\n[Step 6/10] Building analysis context blocks...")
        date_header = (
            f"[DATE ANCHOR] All data and news fetched on {today}. "
            f"Reason about the stock as of {today}.\n"
        )
        macro_ctx = MacroAnalyzer(self.ticker, stock_name).build_context(
            ohlcv["macro"], forecasts["macro"]
        )
        mid_ctx = MidAnalyzer(self.ticker, stock_name).build_context(
            ohlcv["mid"], forecasts["mid"]
        )
        micro_ctx = MicroAnalyzer(self.ticker, stock_name).build_context(
            ohlcv["micro"], forecasts["micro"]
        )

        # ----------------------------------------------------------
        # Step 7: Gemma4 — news analysis
        # ----------------------------------------------------------
        print("\n[Step 7/10] Gemma4 news event analysis...")
        gemma_client = GemmaClient(base_url=self.cfg.model.ollama_base_url)
        if not gemma_client.health_check():
            raise RuntimeError(
                f"Ollama server is not reachable at {self.cfg.model.ollama_base_url}"
            )

        news_analyzer  = NewsAnalyzer(gemma_client, self.ticker, stock_name)
        news_result    = news_analyzer.analyze(news_data)

        # Print news console summary
        print(news_result["console_summary"])

        # ----------------------------------------------------------
        # Step 8: Gemma4 — final investment judgment
        # ----------------------------------------------------------
        print("\n[Step 8/10] Gemma4 final investment judgment...")
        prompt = self._build_prompt(
            date_header, macro_ctx, mid_ctx, micro_ctx,
            latest, news_result
        )
        gemma_result = gemma_client.analyze(prompt)
        print(f"  Recommendation : {gemma_result['recommendation']}")
        print(f"  Confidence     : {gemma_result['confidence']}%")

        # ----------------------------------------------------------
        # Step 9: Charts
        # ----------------------------------------------------------
        print("\n[Step 9/10] Generating charts...")
        chart_builder = ChartBuilder(
            ticker=self.ticker,
            stock_name=stock_name,
            output_dir=Path(self.cfg.output.charts_dir),
        )
        chart_paths = chart_builder.build_all(ohlcv, forecasts, gemma_result)

        # ----------------------------------------------------------
        # Step 10: Save JSON report
        # ----------------------------------------------------------
        print("\n[Step 10/10] Saving JSON report...")
        report = {
            "ticker":       self.ticker,
            "stock_name":   stock_name,
            "date":         today,
            "forecasts":    forecasts,
            "latest_signals": latest,
            "news": {
                "stock_news":    news_data.get("stock_news", []),
                "macro_news":    news_data.get("macro_news", {}),
                "sentiment":     news_result["stock_sentiment"],
                "sentiment_score": news_result["sentiment_score"],
                "macro_risks":   news_result["macro_risks"],
                "macro_opportunities": news_result["macro_opportunities"],
                "news_adjustment": news_result["news_recommendation_adjustment"],
            },
            "gemma": {
                "recommendation": gemma_result["recommendation"],
                "confidence":     gemma_result["confidence"],
                "reasoning":      gemma_result["reasoning"],
                "model":          gemma_result["model"],
            },
            "charts": {k: str(v) for k, v in chart_paths.items()},
        }

        report_path = self.out_dir / f"{self.ticker}_{today}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Report saved : {report_path}")

        # ----------------------------------------------------------
        # Final console summary
        # ----------------------------------------------------------
        self._print_final_summary(
            stock_name, today, gemma_result, news_result, chart_paths
        )

        return report

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        date_header: str,
        macro_ctx: str,
        mid_ctx: str,
        micro_ctx: str,
        latest_signals: dict,
        news_result: dict,
    ) -> str:
        signals_block = "[ Latest Technical Signals ]\n"
        for k, v in latest_signals.items():
            signals_block += f"  {k}: {v}\n"

        news_block = (
            f"[ News Analysis Summary ]\n"
            f"  Stock Sentiment     : {news_result['stock_sentiment']} "
            f"(score: {news_result['sentiment_score']:+.2f})\n"
            f"  Event Tags          : {', '.join(news_result['stock_news_tags'])}\n"
            f"  News Adjustment     : {news_result['news_recommendation_adjustment']}\n"
        )
        if news_result["macro_risks"]:
            news_block += "  Macro Risks:\n"
            for r in news_result["macro_risks"]:
                news_block += f"    - {r}\n"
        if news_result["macro_opportunities"]:
            news_block += "  Macro Opportunities:\n"
            for o in news_result["macro_opportunities"]:
                news_block += f"    + {o}\n"
        if news_result["macro_summary"]:
            news_block += f"  Macro Summary: {news_result['macro_summary']}\n"

        prompt = f"""{date_header}

You are analyzing a Korean stock from three time perspectives simultaneously,
incorporating both quantitative forecasts and qualitative news analysis.

{macro_ctx}

{mid_ctx}

{micro_ctx}

{signals_block}

{news_block}

=== INSTRUCTIONS ===
Synthesize all signals: macro 10-year trend, mid 120-day momentum,
micro 1-week signals, technical indicators, and news sentiment.

1. Identify agreements and conflicts across horizons.
2. Weight the news sentiment appropriately given recent events.
3. Provide a final recommendation.

Output in this exact format:
RECOMMENDATION: <BUY | SELL | HOLD>
CONFIDENCE: <0-100>
REASONING:
<Detailed multi-horizon + news analysis>
RISKS:
<Key risk factors>
"""
        return prompt

    # ------------------------------------------------------------------
    # Final console print
    # ------------------------------------------------------------------

    @staticmethod
    def _print_final_summary(
        stock_name: str,
        today: str,
        gemma_result: dict,
        news_result: dict,
        chart_paths: dict,
    ) -> None:
        bar = "=" * 60
        rec  = gemma_result["recommendation"]
        conf = gemma_result["confidence"]
        adj  = news_result["news_recommendation_adjustment"]
        sent = news_result["stock_sentiment"]

        rec_icon = {"BUY": "▲", "SELL": "▼", "HOLD": "■"}.get(rec, "?")
        adj_icon = {"BOOST_BUY": "[+]", "BOOST_SELL": "[-]", "NEUTRAL": "[~]"}.get(adj, "")

        print(f"\n{bar}")
        print(f"  FINAL RESULT — {stock_name} | {today}")
        print(f"{bar}")
        print(f"  TimesFM + Gemma4 Recommendation : {rec_icon} {rec}")
        print(f"  Confidence                       : {conf}%")
        print(f"  News Sentiment                   : {sent}  {adj_icon} {adj}")
        print(f"  Dashboard → {chart_paths['dashboard']}")
        print(f"{bar}\n")
