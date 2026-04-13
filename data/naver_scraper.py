"""
data/naver_scraper.py
Scrapes Naver Finance for:
  - Stock news headlines (date-aware)
  - Foreign / institutional net buying data
  - Market overview (KOSPI / KOSDAQ index)
All requests inject today's date to ensure recency.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Dict, Any

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

_NAVER_FINANCE_BASE = "https://finance.naver.com"


class NaverScraper:
    """
    Date-aware Naver Finance scraper.

    Parameters
    ----------
    ticker : str
        KRX stock code, e.g. "005930"
    delay_sec : float
        Polite delay between HTTP requests
    max_news : int
        Maximum number of news items to collect
    timeout : int
        HTTP request timeout in seconds
    """

    def __init__(
        self,
        ticker: str,
        delay_sec: float = 1.0,
        max_news: int = 20,
        timeout: int = 10,
    ) -> None:
        self.ticker = ticker
        self.delay_sec = delay_sec
        self.max_news = max_news
        self.timeout = timeout

        self._today = datetime.today()
        self._today_str = self._today.strftime("%Y-%m-%d")           # human-readable
        self._today_compact = self._today.strftime("%Y%m%d")          # URL param format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> Dict[str, Any]:
        """
        Returns a dict with:
          - today        : str  (ISO date string injected into every result)
          - news         : List[Dict]  (headline, date, url)
          - net_buying   : Dict  (foreign / institution net buy amounts)
          - market_index : Dict  (KOSPI, KOSDAQ latest values)
        """
        result: Dict[str, Any] = {"today": self._today_str}
        result["news"] = self._fetch_news()
        time.sleep(self.delay_sec)
        result["net_buying"] = self._fetch_net_buying()
        time.sleep(self.delay_sec)
        result["market_index"] = self._fetch_market_index()
        return result

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def _fetch_news(self) -> List[Dict[str, str]]:
        url = (
            f"{_NAVER_FINANCE_BASE}/item/news.naver"
            f"?code={self.ticker}&date={self._today_compact}"
        )
        soup = self._get_soup(url)
        if soup is None:
            return []

        items: List[Dict[str, str]] = []
        for row in soup.select("table.type5 tr"):
            title_tag = row.select_one("td.title a")
            date_tag  = row.select_one("td.date")
            if not title_tag or not date_tag:
                continue

            items.append({
                "headline": title_tag.get_text(strip=True),
                "date":     date_tag.get_text(strip=True),
                "url":      _NAVER_FINANCE_BASE + title_tag["href"],
                "fetched_on": self._today_str,   # explicit date stamp
            })

            if len(items) >= self.max_news:
                break

        return items

    # ------------------------------------------------------------------
    # Foreign / institution net buying
    # ------------------------------------------------------------------

    def _fetch_net_buying(self) -> Dict[str, Any]:
        url = f"{_NAVER_FINANCE_BASE}/item/frgn.naver?code={self.ticker}"
        soup = self._get_soup(url)
        if soup is None:
            return {}

        result: Dict[str, Any] = {"date": self._today_str}
        try:
            rows = soup.select("table.type2 tr")
            for row in rows:
                cols = row.select("td")
                if len(cols) < 4:
                    continue
                label = cols[0].get_text(strip=True)
                if "foreign" in label.lower() or label in ("외국인", "기관"):
                    result[label] = cols[1].get_text(strip=True)
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # Market index snapshot
    # ------------------------------------------------------------------

    def _fetch_market_index(self) -> Dict[str, str]:
        url = f"{_NAVER_FINANCE_BASE}/sise/sise_index.naver?code=KOSPI"
        soup = self._get_soup(url)
        result: Dict[str, str] = {"date": self._today_str}
        if soup is None:
            return result

        try:
            value_tag = soup.select_one("#now_value")
            change_tag = soup.select_one("#change_value")
            if value_tag:
                result["KOSPI"] = value_tag.get_text(strip=True)
            if change_tag:
                result["KOSPI_change"] = change_tag.get_text(strip=True)
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _get_soup(self, url: str) -> BeautifulSoup | None:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            print(f"[NaverScraper] Request failed for {url}: {exc}")
            return None
