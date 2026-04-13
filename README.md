# Korean Stock Prediction v1

Multi-horizon Korean stock prediction pipeline combining **Google TimesFM 2.5** (time series forecasting) and **Gemma4:e4b** (LLM-based judgment) via Ollama.

## Overview

```
OHLCV Data (pykrx)
    + Google News RSS (stock + macro)
            ↓
    Technical Indicators
    (RSI, MACD, Bollinger Bands, ATR, OBV)
            ↓
    TimesFM 2.5-200M
    ┌─────────────────────────────────┐
    │ Macro  : 10yr context → 60d forecast  │
    │ Mid    : 120d context → 20d forecast  │
    │ Micro  : 60d  context →  5d forecast  │
    └─────────────────────────────────┘
            ↓
    Gemma4:e4b (Ollama)
    ┌─────────────────────────────────┐
    │ Stage 1: News sentiment analysis      │
    │ Stage 2: Macro event impact           │
    │ Stage 3: Final investment judgment    │
    └─────────────────────────────────┘
            ↓
    BUY / SELL / HOLD + Confidence %
    + Plotly Interactive Charts
    + JSON Report
```

## Sample Output

```
============================================================
  FINAL RESULT — 삼성전자 | 2026-04-13
============================================================
  TimesFM + Gemma4 Recommendation : ■ HOLD
  Confidence                       : 75%
  News Sentiment                   : NEGATIVE  [+] BOOST_BUY
  Dashboard → reports/charts/005930_dashboard.html
============================================================
```

### News Analysis Console Output (excerpt)

```
============================================================
  NEWS ANALYSIS — 삼성전자 (005930) | 2026-04-13
============================================================
  Stock News (20 items):
    • 홍라희, 삼성전자 주식 3조원어치 매각…상속세 곧 완납
      2026-04-09  [한겨레]
    • 삼성전자, 갤S26 흥행에 1분기 글로벌 스마트폰 1위 탈환
      2026-04-12  [서울경제]
  Stock Sentiment  : [-] NEGATIVE  (score: -0.50)
  Event Tags       : Shareholder Activity, Block Deal, Tax/Regulation, Product

  Macro News by Topic:
    [Oil]        • 트럼프 "이란 원유도 못 나가"…호르무즈 역봉쇄 예고
    [Rates]      • 한국은행, 4월 기준금리 연 2.50% 동결
    [Geopolitics]• 전쟁이 만든 지정학 프리미엄…에너지·방산·반도체 위기 속 강해진다
    [KR Economy] • 이란 전쟁 여파에 성장률 전망 뚝…나틱시스 1% 제시
    [Semiconductors] • 아마존의 반도체 독립 선언…삼성 파운드리에 새 기회
    [FX (KRW)]   • 원/달러 환율 1,495원대 출발
  News Adjustment  : >> Leans BUY (macro tailwind)
============================================================
```

## Requirements

### Hardware
- GPU: NVIDIA GPU with 12 GB+ VRAM (tested on RTX 3060 12GB)
- CUDA 12.x+

### Models (local cache required)
| Model | Size | Role |
|---|---|---|
| `google/timesfm-2.5-200m-pytorch` | ~1 GB | Time series forecasting |
| `gemma4:e4b` (Ollama) | 9.6 GB | News analysis + investment judgment |

> Both models run **sequentially** — TimesFM is released from GPU before Ollama calls Gemma4.

### Software
- Python 3.11+
- [Ollama](https://ollama.com) server running locally

```bash
pip install -r requirements.txt
```

```
pykrx
timesfm
torch
transformers
plotly
pandas
numpy
requests
beautifulsoup4
tomli
```

## Usage

```bash
# Basic
python main.py --ticker 005930

# Custom config
python main.py --ticker 000660 --config config.toml

# Custom output directory
python main.py --ticker 035420 --output-dir ./reports/naver
```

`--ticker` is **required**. Use KRX stock codes:
- `005930` → 삼성전자
- `000660` → SK하이닉스
- `035420` → NAVER

## Project Structure

```
korean-stock-prediction_v1/
│
├── config.toml                  # All parameters (no hardcoding)
├── main.py                      # CLI entry point
├── requirements.txt
│
├── config/
│   └── settings.py              # config.toml loader
│
├── data/
│   ├── collector.py             # pykrx OHLCV (3 horizons + daily cache)
│   ├── news_fetcher.py          # Google News RSS (stock + 6 macro topics)
│   └── preprocessor.py         # RSI, MACD, Bollinger Bands, ATR, OBV
│
├── models/
│   ├── timesfm_runner.py        # TimesFM sequential 3-horizon inference
│   └── gemma_client.py          # Ollama REST API (gemma4:e4b fixed)
│
├── analysis/
│   ├── macro_analyzer.py        # 10-year context → prompt fragment
│   ├── mid_analyzer.py          # 120-day context → prompt fragment
│   ├── micro_analyzer.py        # 1-week signals → prompt fragment
│   └── news_analyzer.py         # Gemma4 news sentiment + macro impact
│
├── search/
│   └── date_aware_search.py     # Today-date forced into every query
│
├── visualization/
│   └── charts.py                # Plotly HTML (macro / mid / micro / dashboard)
│
├── pipeline/
│   └── orchestrator.py          # 10-step sequential pipeline
│
└── reports/                     # Generated outputs (gitignored)
    ├── {ticker}_{date}.json
    └── charts/
        ├── {ticker}_macro.html
        ├── {ticker}_mid.html
        ├── {ticker}_micro.html
        └── {ticker}_dashboard.html
```

## Configuration

Edit `config.toml` to adjust parameters without touching code:

```toml
[timesfm]
macro_context_days  = 2520   # ~10 years
mid_context_days    = 120
micro_context_days  = 60

macro_horizon_days  = 60     # quarterly forecast
mid_horizon_days    = 20     # monthly forecast
micro_horizon_days  = 5      # weekly forecast

[scraping]
max_news_items  = 20
naver_delay_sec = 1.0
```

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| LLM model | `gemma4:e4b` (hardcoded) | Changing model requires prompt restructuring |
| News source | Google News RSS | No API key, no bot blocking, Korean + English |
| VRAM strategy | Sequential (TimesFM → free → Gemma4) | RTX 3060 12GB limit |
| Visualization | Plotly (interactive HTML) | No server needed, works offline |
| Scheduler | None | Manual execution on demand |
| Config | `config.toml` + CLI args | No hardcoding |

## Date-Aware Design

LLMs tend to reason from their training cutoff date. This pipeline injects today's date at three layers:

1. **Search queries** — `"삼성전자 news 2026-04-13"`
2. **RSS URL parameters** — date-anchored endpoints
3. **System prompt** — `"Today is 2026-04-13. All analysis must reflect information as of this date."`

## Output

Each run produces:
- `reports/{ticker}_{date}.json` — full structured report
- `reports/charts/{ticker}_macro.html` — 10-year candlestick + SMA200 + forecast band
- `reports/charts/{ticker}_mid.html` — 120-day candlestick + MACD + RSI + forecast
- `reports/charts/{ticker}_micro.html` — 60-day candlestick + Bollinger Bands + forecast
- `reports/charts/{ticker}_dashboard.html` — all three horizons in one view

## License

MIT
