"""
models/gemma_client.py
Ollama REST API wrapper for gemma4:e4b.
The model name is fixed — changing it requires reworking the prompt strategy.

Date injection:
  Every request includes today's date in the system prompt so the model
  reasons about recency correctly without hallucinating a stale date.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import requests

# Fixed — do not parameterise.
OLLAMA_MODEL = "gemma4:e4b"

_SYSTEM_TEMPLATE = (
    "Today is {today} (KST). "
    "You are a professional Korean stock market analyst. "
    "All your analysis and reasoning must be grounded in information "
    "available as of {today}. "
    "Respond only in English. "
    "When you output a final recommendation, use exactly one of: "
    "BUY / SELL / HOLD."
)


class GemmaClient:
    """
    Thin wrapper around the Ollama /api/generate endpoint.

    Parameters
    ----------
    base_url : str
        Ollama server base URL, e.g. "http://localhost:11434"
    timeout : int
        HTTP request timeout in seconds (large because generation is slow)
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300) -> None:
        self.base_url  = base_url.rstrip("/")
        self.timeout   = timeout
        self._today    = datetime.today().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Sends a prompt to gemma4:e4b and returns a structured dict:
          {
            "raw":            str,   # full model text
            "recommendation": str,   # BUY | SELL | HOLD
            "confidence":     int,   # 0-100
            "reasoning":      str,   # extracted reasoning block
          }
        """
        system = _SYSTEM_TEMPLATE.format(today=self._today)
        full_prompt = f"{system}\n\n{prompt}"

        raw = self._call_ollama(full_prompt)
        return self._parse_response(raw)

    def health_check(self) -> bool:
        """Returns True if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,   # low temperature for more deterministic analysis
                "num_predict": 2048,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.RequestException as exc:
            raise RuntimeError(f"[GemmaClient] Ollama request failed: {exc}") from exc

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """
        Extracts structured fields from the raw text.
        Looks for:
          RECOMMENDATION: BUY | SELL | HOLD
          CONFIDENCE: <number>
        Falls back to searching the text if structured tags are absent.
        """
        recommendation = "HOLD"  # safe default
        confidence     = 50
        reasoning      = raw.strip()

        upper = raw.upper()

        # --- recommendation ---
        for keyword in ("BUY", "SELL", "HOLD"):
            marker = f"RECOMMENDATION: {keyword}"
            if marker in upper:
                recommendation = keyword
                break
        else:
            # fallback: check if any keyword appears prominently
            for keyword in ("BUY", "SELL", "HOLD"):
                if keyword in upper:
                    recommendation = keyword
                    break

        # --- confidence ---
        for line in raw.splitlines():
            if "CONFIDENCE" in line.upper() and ":" in line:
                try:
                    val = int("".join(filter(str.isdigit, line.split(":")[-1])))
                    if 0 <= val <= 100:
                        confidence = val
                except ValueError:
                    pass
                break

        return {
            "raw":            raw,
            "recommendation": recommendation,
            "confidence":     confidence,
            "reasoning":      reasoning,
            "model":          OLLAMA_MODEL,
            "date":           datetime.today().strftime("%Y-%m-%d"),
        }
