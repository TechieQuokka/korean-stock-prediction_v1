"""
search/date_aware_search.py
Date-aware search over Naver Finance.

Problem this solves:
  LLMs often reason as if they are at their training cutoff date.
  To force recency we:
    1. Inject today's date into every search query string.
    2. Use date-parameterised Naver Finance URLs.
    3. Return a search_date field in every result so the LLM prompt
       can explicitly state "this data was fetched on <date>".

All text returned is in the raw form scraped from Naver — the
prompt builder in each analyzer is responsible for formatting it.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}

_BASE = "https://finance.naver.com"


class DateAwareSearch:
    """
    Searches Naver Finance with the current date always embedded in
    the query and URL parameters.

    Parameters
    ----------
    ticker : str
        KRX stock code
    delay_sec : float
        Polite delay between requests
    timeout : int
        Per-request HTTP timeout
    """

    def __init__(
        self,
        ticker: str,
        delay_sec: float = 1.0,
        timeout: int = 10,
    ) -> None:
        self.ticker     = ticker
        self.delay_sec  = delay_sec
        self.timeout    = timeout

        # Three date formats used in different contexts
        self._today          = datetime.today()
        self._today_iso      = self._today.strftime("%Y-%m-%d")       # 2026-04-13
        self._today_compact  = self._today.strftime("%Y%m%d")          # 20260413
        self._today_display  = self._today.strftime("%Y.%m.%d")        # 2026.04.13

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search_news(self, max_items: int = 20) -> List[Dict[str, str]]:
        """
        Fetches stock-specific news from Naver Finance.
        The URL explicitly includes today's date to anchor the search.
        """
        # Layer 1: date in URL parameter
        url = (
            f"{_BASE}/item/news.naver"
            f"?code={self.ticker}&date={self._today_compact}"
        )
        soup = self._get(url)
        if soup is None:
            return []

        items: List[Dict[str, str]] = []
        for row in soup.select("table.type5 tr"):
            title_tag = row.select_one("td.title a")
            date_tag  = row.select_one("td.date")
            if not title_tag or not date_tag:
                continue
            items.append({
                "headline":   title_tag.get_text(strip=True),
                "date":       date_tag.get_text(strip=True),
                "url":        _BASE + title_tag["href"],
                # Layer 3: explicit fetch timestamp in every record
                "fetched_on": self._today_iso,
            })
            if len(items) >= max_items:
                break

        return items

    def search_disclosures(self, max_items: int = 10) -> List[Dict[str, str]]:
        """
        Fetches recent DART disclosures listed on Naver Finance.
        """
        url = f"{_BASE}/item/news_news.naver?code={self.ticker}&page=1"
        soup = self._get(url)
        if soup is None:
            return []

        items: List[Dict[str, str]] = []
        for row in soup.select("table.type5 tr"):
            title_tag  = row.select_one("td.title a")
            date_tag   = row.select_one("td.date")
            source_tag = row.select_one("td.info")
            if not title_tag:
                continue
            items.append({
                "title":      title_tag.get_text(strip=True),
                "date":       date_tag.get_text(strip=True) if date_tag else "",
                "source":     source_tag.get_text(strip=True) if source_tag else "",
                "url":        _BASE + title_tag["href"],
                "fetched_on": self._today_iso,
            })
            if len(items) >= max_items:
                break

        return items

    def build_date_header(self) -> str:
        """
        Returns a short text header to prepend to any context block,
        forcing the LLM to anchor its reasoning to today's date.
        """
        return (
            f"[SEARCH DATE ANCHOR]\n"
            f"All data below was fetched on {self._today_iso}.\n"
            f"Treat all information as current as of {self._today_iso}.\n"
        )

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _get(self, url: str) -> BeautifulSoup | None:
        try:
            time.sleep(self.delay_sec)
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            print(f"[DateAwareSearch] Request failed: {url}  error={exc}")
            return None
