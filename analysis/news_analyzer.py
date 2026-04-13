"""
analysis/news_analyzer.py
Uses Gemma4:e4b to analyze news impact on a specific stock.

Two-stage analysis:
  Stage 1 — Per-article sentiment batch
      Gemma4 classifies each news item as POSITIVE / NEGATIVE / NEUTRAL
      and identifies the event type.

  Stage 2 — Macro impact assessment
      Gemma4 reasons about how macro events (oil, war, rates, etc.)
      indirectly affect this specific stock/sector.

Output dict:
  {
    "stock_sentiment":  "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "sentiment_score":  float,   # -1.0 to +1.0
    "stock_news_tags":  List[str],
    "macro_risks":      List[str],
    "macro_opportunities": List[str],
    "macro_summary":    str,
    "news_recommendation_adjustment": str,  # "BOOST_BUY" | "BOOST_SELL" | "NEUTRAL"
    "console_summary":  str,   # pre-formatted for print()
    "raw_stock":        str,
    "raw_macro":        str,
  }
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from models.gemma_client import GemmaClient


_SENTIMENT_MAP = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}


class NewsAnalyzer:
    """
    Parameters
    ----------
    client : GemmaClient
        Shared Ollama client instance
    ticker : str
    stock_name : str
    """

    def __init__(self, client: GemmaClient, ticker: str, stock_name: str) -> None:
        self.client     = client
        self.ticker     = ticker
        self.stock_name = stock_name
        self._today     = datetime.today().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def analyze(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        stock_news = news_data.get("stock_news", [])
        macro_news = news_data.get("macro_news", {})

        raw_stock = ""
        raw_macro = ""

        # Stage 1: stock-specific news sentiment
        stock_result = {"sentiment": "NEUTRAL", "score": 0.0, "tags": [], "raw": ""}
        if stock_news:
            stock_result = self._analyze_stock_news(stock_news)
            raw_stock = stock_result["raw"]

        # Stage 2: macro impact
        macro_result = {
            "risks": [], "opportunities": [], "summary": "No macro news available.",
            "adjustment": "NEUTRAL", "raw": "",
        }
        if macro_news:
            macro_result = self._analyze_macro_news(macro_news)
            raw_macro = macro_result["raw"]

        result = {
            "stock_sentiment":              stock_result["sentiment"],
            "sentiment_score":              stock_result["score"],
            "stock_news_tags":              stock_result["tags"],
            "macro_risks":                  macro_result["risks"],
            "macro_opportunities":          macro_result["opportunities"],
            "macro_summary":                macro_result["summary"],
            "news_recommendation_adjustment": macro_result["adjustment"],
            "raw_stock":                    raw_stock,
            "raw_macro":                    raw_macro,
        }

        result["console_summary"] = self._build_console_summary(
            stock_news, macro_news, result
        )

        return result

    # ------------------------------------------------------------------
    # Stage 1 — stock news
    # ------------------------------------------------------------------

    def _analyze_stock_news(self, items: List[Dict]) -> Dict:
        headlines = "\n".join(
            f"  [{i+1}] {it['title']} ({it['date']}) [{it['source']}]"
            for i, it in enumerate(items)
        )

        prompt = f"""Today is {self._today}.

You are analyzing news headlines for {self.stock_name} (KRX: {self.ticker}).

NEWS HEADLINES:
{headlines}

For each headline, classify it as POSITIVE, NEGATIVE, or NEUTRAL for this stock.
Then provide an overall sentiment.

Respond in this exact format:
OVERALL_SENTIMENT: <POSITIVE | NEGATIVE | NEUTRAL>
OVERALL_SCORE: <float from -1.0 (very negative) to +1.0 (very positive)>
EVENT_TAGS: <comma-separated list of event types, e.g. earnings, regulation, partnership, macro>
SUMMARY: <2-3 sentences summarizing the stock news impact>
"""
        raw = self.client._call_ollama(prompt)

        sentiment = "NEUTRAL"
        score     = 0.0
        tags: List[str] = []

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("OVERALL_SENTIMENT:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in _SENTIMENT_MAP:
                    sentiment = val
            elif line.startswith("OVERALL_SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    score = max(-1.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith("EVENT_TAGS:"):
                tags = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]

        return {"sentiment": sentiment, "score": score, "tags": tags, "raw": raw}

    # ------------------------------------------------------------------
    # Stage 2 — macro impact
    # ------------------------------------------------------------------

    def _analyze_macro_news(self, macro_news: Dict[str, List[Dict]]) -> Dict:
        blocks = []
        for topic, items in macro_news.items():
            if not items:
                continue
            headlines = "; ".join(it["title"] for it in items)
            blocks.append(f"[{topic.upper()}] {headlines}")

        macro_text = "\n".join(blocks)

        prompt = f"""Today is {self._today}.

You are assessing how macro news events may indirectly affect {self.stock_name} (KRX: {self.ticker}).
The company is in the semiconductor / consumer electronics sector.

MACRO NEWS BY TOPIC:
{macro_text}

Analyze the indirect impact on this stock and respond in this exact format:
MACRO_RISKS: <bullet list of risk factors, one per line starting with ->
MACRO_OPPORTUNITIES: <bullet list of opportunity factors, one per line starting with ->
MACRO_SUMMARY: <2-3 sentences overall macro impact assessment>
RECOMMENDATION_ADJUSTMENT: <BOOST_BUY | BOOST_SELL | NEUTRAL>
"""
        raw = self.client._call_ollama(prompt)

        risks: List[str]   = []
        opps:  List[str]   = []
        summary            = ""
        adjustment         = "NEUTRAL"
        section            = None

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("MACRO_RISKS:"):
                section = "risks"
            elif line.startswith("MACRO_OPPORTUNITIES:"):
                section = "opps"
            elif line.startswith("MACRO_SUMMARY:"):
                summary = line.split(":", 1)[1].strip()
                section = "summary"
            elif line.startswith("RECOMMENDATION_ADJUSTMENT:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in ("BOOST_BUY", "BOOST_SELL", "NEUTRAL"):
                    adjustment = val
                section = None
            elif line.startswith("->"):
                content = line[2:].strip()
                if section == "risks" and content:
                    risks.append(content)
                elif section == "opps" and content:
                    opps.append(content)
            elif section == "summary" and line and not line.startswith("RECOMMENDATION"):
                summary = (summary + " " + line).strip()

        return {
            "risks":        risks,
            "opportunities": opps,
            "summary":      summary,
            "adjustment":   adjustment,
            "raw":          raw,
        }

    # ------------------------------------------------------------------
    # Console summary builder
    # ------------------------------------------------------------------

    def _build_console_summary(
        self,
        stock_news: List[Dict],
        macro_news: Dict[str, List[Dict]],
        result: Dict,
    ) -> str:
        bar  = "=" * 60
        thin = "-" * 60

        sentiment = result["stock_sentiment"]
        score     = result["sentiment_score"]
        icon_map  = {"POSITIVE": "[+]", "NEGATIVE": "[-]", "NEUTRAL": "[~]"}
        icon      = icon_map.get(sentiment, "[~]")

        lines = [
            bar,
            f"  NEWS ANALYSIS — {self.stock_name} ({self.ticker}) | {self._today}",
            bar,
        ]

        # Stock news
        lines.append(f"\n  Stock News ({len(stock_news)} items):")
        if stock_news:
            for item in stock_news:
                lines.append(f"    • {item['title']}")
                lines.append(f"      {item['date']}  [{item['source']}]")
        else:
            lines.append("    No stock-specific news found.")

        # Stock sentiment
        lines += [
            thin,
            f"  Stock Sentiment  : {icon} {sentiment}  (score: {score:+.2f})",
        ]
        if result["stock_news_tags"]:
            lines.append(f"  Event Tags       : {', '.join(result['stock_news_tags'])}")

        # Macro news
        lines.append(f"\n  Macro News by Topic:")
        macro_icon = {"oil_price": "Oil", "interest_rate": "Rates",
                      "geopolitics": "Geopolitics", "korea_economy": "KR Economy",
                      "semiconductor": "Semiconductors", "fx_krw": "FX (KRW)"}
        for topic, items in macro_news.items():
            label = macro_icon.get(topic, topic)
            if items:
                lines.append(f"    [{label}]")
                for it in items:
                    lines.append(f"      • {it['title']}")
            else:
                lines.append(f"    [{label}] No news.")

        # Macro impact
        lines.append(thin)
        lines.append(f"  Macro Impact (Gemma4 Assessment):")
        if result["macro_risks"]:
            lines.append("    Risks:")
            for r in result["macro_risks"]:
                lines.append(f"      [-] {r}")
        if result["macro_opportunities"]:
            lines.append("    Opportunities:")
            for o in result["macro_opportunities"]:
                lines.append(f"      [+] {o}")
        if result["macro_summary"]:
            lines.append(f"    Summary: {result['macro_summary']}")

        adj = result["news_recommendation_adjustment"]
        adj_display = {
            "BOOST_BUY":  ">> Leans BUY  (macro tailwind)",
            "BOOST_SELL": ">> Leans SELL (macro headwind)",
            "NEUTRAL":    ">> Neutral impact",
        }.get(adj, adj)
        lines.append(f"\n  News Adjustment  : {adj_display}")
        lines.append(bar)

        return "\n".join(lines)
