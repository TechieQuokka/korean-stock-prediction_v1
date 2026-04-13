"""
config/settings.py
Loads config.toml and exposes a typed Settings dataclass.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class ModelConfig:
    timesfm_repo: str
    ollama_base_url: str


@dataclass
class TimesFMConfig:
    macro_context_days: int
    mid_context_days: int
    micro_context_days: int
    macro_horizon_days: int
    mid_horizon_days: int
    micro_horizon_days: int
    quantile_levels: List[float]


@dataclass
class DataConfig:
    market: str
    cache_dir: str


@dataclass
class ScrapingConfig:
    naver_delay_sec: float
    max_news_items: int
    request_timeout: int


@dataclass
class OutputConfig:
    reports_dir: str
    charts_dir: str


@dataclass
class Settings:
    model: ModelConfig
    timesfm: TimesFMConfig
    data: DataConfig
    scraping: ScrapingConfig
    output: OutputConfig

    @classmethod
    def load(cls, path: str | Path = "config.toml") -> "Settings":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

        with open(config_path, "rb") as f:
            raw = tomllib.load(f)

        return cls(
            model=ModelConfig(**raw["model"]),
            timesfm=TimesFMConfig(**raw["timesfm"]),
            data=DataConfig(**raw["data"]),
            scraping=ScrapingConfig(**raw["scraping"]),
            output=OutputConfig(**raw["output"]),
        )
