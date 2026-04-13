"""
pipeline/orchestrator.py
Coordinates the full prediction pipeline in a strict sequential order
to stay within 12 GB VRAM:

  Step 1  Collect OHLCV data (pykrx)
  Step 2  Scrape Naver Finance (news, net buying, market index)
  Step 3  Compute technical indicators
  Step 4  Run TimesFM (macro + mid + micro) — GPU
  Step 5  Release TimesFM from GPU
  Step 6  Build analysis context (macro / mid / micro text blocks)
  Step 7  Call Gemma4 via Ollama — GPU (after TimesFM freed)
  Step 8  Generate charts (plotly)
  Step 9  Save JSON report
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from config.settings import Settings
from data.collector import OHLCVCollector
from data.naver_scraper import NaverScraper
from data.preprocessor import Preprocessor
from models.timesfm_runner import TimesFMRunner
from models.gemma_client import GemmaClient
from analysis.macro_analyzer import MacroAnalyzer
from analysis.mid_analyzer import MidAnalyzer
from analysis.micro_analyzer import MicroAnalyzer
from search.date_aware_search import DateAwareSearch
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
        self.ticker   = ticker
        self.cfg      = settings
        self.out_dir  = Path(output_dir) if output_dir else Path(settings.output.reports_dir)
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
        print("[Step 1/9] Collecting OHLCV data...")
        collector = OHLCVCollector(
            ticker=self.ticker,
            macro_context_days=self.cfg.timesfm.macro_context_days,
            mid_context_days=self.cfg.timesfm.mid_context_days,
            micro_context_days=self.cfg.timesfm.micro_context_days,
            cache_dir=self.cfg.data.cache_dir,
        )
        ohlcv_raw   = collector.fetch_all()
        stock_name  = collector.get_stock_name()
        print(f"  Stock name : {stock_name}")
        for h, df in ohlcv_raw.items():
            print(f"  {h:6s} : {len(df)} rows  "
                  f"({str(df.index[0].date())} ~ {str(df.index[-1].date())})")

        # ----------------------------------------------------------
        # Step 2: Naver Finance scraping
        # ----------------------------------------------------------
        print("\n[Step 2/9] Scraping Naver Finance (date-aware)...")
        scraper = NaverScraper(
            ticker=self.ticker,
            delay_sec=self.cfg.scraping.naver_delay_sec,
            max_news=self.cfg.scraping.max_news_items,
            timeout=self.cfg.scraping.request_timeout,
        )
        scrape_data = scraper.fetch_all()
        print(f"  News items   : {len(scrape_data.get('news', []))}")
        print(f"  Net buying   : {scrape_data.get('net_buying', {})}")
        print(f"  Market index : {scrape_data.get('market_index', {})}")

        searcher = DateAwareSearch(
            ticker=self.ticker,
            delay_sec=self.cfg.scraping.naver_delay_sec,
            timeout=self.cfg.scraping.request_timeout,
        )
        disclosures  = searcher.search_disclosures(max_items=10)
        date_header  = searcher.build_date_header()
        print(f"  Disclosures  : {len(disclosures)}")

        # ----------------------------------------------------------
        # Step 3: Technical indicators
        # ----------------------------------------------------------
        print("\n[Step 3/9] Computing technical indicators...")
        pp   = Preprocessor()
        ohlcv = pp.process_all(ohlcv_raw)
        latest_signals = Preprocessor.latest_signals(ohlcv["micro"])
        print(f"  Latest signals: {latest_signals}")

        # ----------------------------------------------------------
        # Step 4 & 5: TimesFM inference + release
        # ----------------------------------------------------------
        print("\n[Step 4/9] Running TimesFM (macro / mid / micro)...")
        runner = TimesFMRunner(
            repo=self.cfg.model.timesfm_repo,
            macro_context=self.cfg.timesfm.macro_context_days,
            mid_context=self.cfg.timesfm.mid_context_days,
            micro_context=self.cfg.timesfm.micro_context_days,
            macro_horizon=self.cfg.timesfm.macro_horizon_days,
            mid_horizon=self.cfg.timesfm.mid_horizon_days,
            micro_horizon=self.cfg.timesfm.micro_horizon_days,
        )
        forecasts = runner.run_all(ohlcv)   # releases GPU memory internally
        print("[Step 5/9] TimesFM released from GPU.")

        # ----------------------------------------------------------
        # Step 6: Build analysis context
        # ----------------------------------------------------------
        print("\n[Step 6/9] Building analysis context blocks...")
        macro_ctx = MacroAnalyzer(self.ticker, stock_name).build_context(
            ohlcv["macro"], forecasts["macro"], scrape_data
        )
        mid_ctx = MidAnalyzer(self.ticker, stock_name).build_context(
            ohlcv["mid"], forecasts["mid"]
        )
        micro_ctx = MicroAnalyzer(self.ticker, stock_name).build_context(
            ohlcv["micro"], forecasts["micro"], scrape_data
        )

        prompt = self._build_prompt(
            date_header, macro_ctx, mid_ctx, micro_ctx,
            latest_signals, disclosures
        )

        # ----------------------------------------------------------
        # Step 7: Gemma4 judgment
        # ----------------------------------------------------------
        print("\n[Step 7/9] Calling Gemma4 via Ollama...")
        client = GemmaClient(base_url=self.cfg.model.ollama_base_url)
        if not client.health_check():
            raise RuntimeError("Ollama server is not reachable at "
                               f"{self.cfg.model.ollama_base_url}")
        gemma_result = client.analyze(prompt)
        print(f"  Recommendation : {gemma_result['recommendation']}")
        print(f"  Confidence     : {gemma_result['confidence']}%")

        # ----------------------------------------------------------
        # Step 8: Charts
        # ----------------------------------------------------------
        print("\n[Step 8/9] Generating charts...")
        charts_dir = Path(self.cfg.output.charts_dir)
        chart_builder = ChartBuilder(
            ticker=self.ticker,
            stock_name=stock_name,
            output_dir=charts_dir,
        )
        chart_paths = chart_builder.build_all(ohlcv, forecasts, gemma_result)

        # ----------------------------------------------------------
        # Step 9: Save JSON report
        # ----------------------------------------------------------
        print("\n[Step 9/9] Saving JSON report...")
        report = {
            "ticker":       self.ticker,
            "stock_name":   stock_name,
            "date":         today,
            "forecasts":    forecasts,
            "latest_signals": latest_signals,
            "gemma":        {
                "recommendation": gemma_result["recommendation"],
                "confidence":     gemma_result["confidence"],
                "reasoning":      gemma_result["reasoning"],
                "model":          gemma_result["model"],
            },
            "news":         scrape_data.get("news", []),
            "disclosures":  disclosures,
            "charts":       {k: str(v) for k, v in chart_paths.items()},
        }

        report_path = self.out_dir / f"{self.ticker}_{today}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Report saved : {report_path}")

        # ----------------------------------------------------------
        # Final summary
        # ----------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"  RESULT: {stock_name} ({self.ticker})")
        print(f"  Recommendation : {gemma_result['recommendation']}")
        print(f"  Confidence     : {gemma_result['confidence']}%")
        print(f"  Dashboard      : {chart_paths['dashboard']}")
        print(f"{'='*60}\n")

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
        disclosures: list,
    ) -> str:
        disc_block = ""
        if disclosures:
            lines = ["[ Recent Disclosures ]"]
            for d in disclosures[:5]:
                lines.append(f"  - {d['title']} ({d.get('date', '')}) [{d.get('source', '')}]")
            disc_block = "\n".join(lines) + "\n"

        signals_block = "[ Latest Technical Signals ]\n"
        for k, v in latest_signals.items():
            signals_block += f"  {k}: {v}\n"

        prompt = f"""{date_header}

You are analyzing a Korean stock from three time perspectives simultaneously.
Your task is to synthesize all three perspectives and provide a final investment recommendation.

{macro_ctx}

{mid_ctx}

{micro_ctx}

{signals_block}
{disc_block}

=== INSTRUCTIONS ===
Based on all three time horizon analyses above (macro 10-year, mid 120-day, micro 1-week):

1. Synthesize the signals from each horizon.
2. Identify any conflicts between horizons and explain how you resolve them.
3. Provide your final recommendation.

Output your response in this exact format:
RECOMMENDATION: <BUY | SELL | HOLD>
CONFIDENCE: <0-100>
REASONING:
<Your detailed multi-horizon analysis here>
RISKS:
<Key risk factors>
"""
        return prompt
