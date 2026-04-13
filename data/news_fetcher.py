"""
data/news_fetcher.py
Fetches news via Google News RSS — no API key required, no bot blocking.

Two categories:
  1. Stock-specific news  : searches by company name + ticker
  2. Macro news           : searches predefined macro topics
                            (oil, rates, war/geopolitics, Korea economy,
                             semiconductors, USD/KRW)

Google News RSS endpoint:
  https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko
  https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List
from urllib.parse import quote_plus

import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

_GOOGLE_NEWS_RSS_KO = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
_GOOGLE_NEWS_RSS_EN = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# Macro topics to monitor — each tuple is (label, korean_query, english_query)
_MACRO_TOPICS = [
    ("oil_price",      "유가 원유",                    "oil price crude"),
    ("interest_rate",  "금리 한국은행 연준",             "interest rate Fed Korea BOK"),
    ("geopolitics",    "전쟁 지정학 분쟁",              "war geopolitical conflict"),
    ("korea_economy",  "한국 경제 성장률",              "Korea economy GDP growth"),
    ("semiconductor",  "반도체 메모리 파운드리",         "semiconductor memory chip foundry"),
    ("fx_krw",         "원달러 환율 달러",              "KRW USD exchange rate"),
]


class NewsFetcher:
    """
    Fetches Google News RSS for stock-specific and macro topics.

    Parameters
    ----------
    stock_name : str
        Korean stock name (e.g. "삼성전자")
    ticker : str
        KRX ticker code (e.g. "005930")
    max_stock_news : int
        Max number of stock-specific news items
    max_macro_per_topic : int
        Max number of items per macro topic
    delay_sec : float
        Polite delay between RSS requests
    timeout : int
        HTTP request timeout in seconds
    """

    def __init__(
        self,
        stock_name: str,
        ticker: str,
        max_stock_news: int = 10,
        max_macro_per_topic: int = 3,
        delay_sec: float = 0.5,
        timeout: int = 10,
    ) -> None:
        self.stock_name         = stock_name
        self.ticker             = ticker
        self.max_stock_news     = max_stock_news
        self.max_macro_per_topic = max_macro_per_topic
        self.delay_sec          = delay_sec
        self.timeout            = timeout
        self._today             = datetime.today().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch_all(self) -> Dict:
        """
        Returns:
          {
            "today":       str,
            "stock_news":  List[Dict],   # company-specific
            "macro_news":  Dict[str, List[Dict]],  # by topic label
          }
        """
        stock_news = self._fetch_stock_news()
        time.sleep(self.delay_sec)
        macro_news = self._fetch_macro_news()

        return {
            "today":      self._today,
            "stock_news": stock_news,
            "macro_news": macro_news,
        }

    # ------------------------------------------------------------------
    # Stock news
    # ------------------------------------------------------------------

    def _fetch_stock_news(self) -> List[Dict]:
        # Search in both Korean and English to maximise coverage
        query_ko = f"{self.stock_name} 주식"
        query_en = f"{self.stock_name} stock"

        items: List[Dict] = []
        for query, lang in [(query_ko, "ko"), (query_en, "en")]:
            items += self._rss_fetch(query, lang, self.max_stock_news)
            time.sleep(self.delay_sec)
            if len(items) >= self.max_stock_news:
                break

        # Deduplicate by title
        seen: set = set()
        unique = []
        for item in items:
            if item["title"] not in seen:
                seen.add(item["title"])
                unique.append(item)

        return unique[: self.max_stock_news]

    # ------------------------------------------------------------------
    # Macro news
    # ------------------------------------------------------------------

    def _fetch_macro_news(self) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {}
        for label, query_ko, query_en in _MACRO_TOPICS:
            items = self._rss_fetch(query_ko, "ko", self.max_macro_per_topic)
            if len(items) < self.max_macro_per_topic:
                time.sleep(self.delay_sec)
                items += self._rss_fetch(query_en, "en", self.max_macro_per_topic - len(items))
            result[label] = items[: self.max_macro_per_topic]
            time.sleep(self.delay_sec)
        return result

    # ------------------------------------------------------------------
    # RSS fetch helper
    # ------------------------------------------------------------------

    def _rss_fetch(self, query: str, lang: str, max_items: int) -> List[Dict]:
        template = _GOOGLE_NEWS_RSS_KO if lang == "ko" else _GOOGLE_NEWS_RSS_EN
        url = template.format(query=quote_plus(query))

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[NewsFetcher] RSS fetch failed ({query}): {exc}")
            return []

        return self._parse_rss(resp.text, max_items)

    @staticmethod
    def _parse_rss(xml_text: str, max_items: int) -> List[Dict]:
        items: List[Dict] = []
        try:
            root = ET.fromstring(xml_text)
            channel = root.find("channel")
            if channel is None:
                return []

            for item in channel.findall("item"):
                title   = item.findtext("title", "").strip()
                link    = item.findtext("link", "").strip()
                pub_raw = item.findtext("pubDate", "")
                source  = ""
                src_tag = item.find("source")
                if src_tag is not None:
                    source = src_tag.text or ""

                # Parse published date
                try:
                    pub_dt = parsedate_to_datetime(pub_raw)
                    pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pub_str = pub_raw

                if not title:
                    continue

                items.append({
                    "title":  title,
                    "url":    link,
                    "date":   pub_str,
                    "source": source,
                })

                if len(items) >= max_items:
                    break

        except ET.ParseError as exc:
            print(f"[NewsFetcher] RSS parse error: {exc}")

        return items
