"""
main.py
CLI entry point for the Korean Stock Prediction pipeline.

Usage:
  python main.py --ticker 005930
  python main.py --ticker 005930 --config config.toml
  python main.py --ticker 005930 --output-dir ./reports/samsung
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Korean Stock Prediction: TimesFM + Gemma4 multi-horizon analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker 005930
  python main.py --ticker 000660 --config config.toml
  python main.py --ticker 035420 --output-dir ./reports/naver
        """,
    )

    parser.add_argument(
        "--ticker",
        required=True,
        metavar="TICKER",
        help="KRX stock code (required). e.g. 005930 for Samsung Electronics",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        metavar="PATH",
        help="Path to config.toml (default: config.toml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Override output directory for reports (default: from config.toml)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate config file exists before doing any heavy imports
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Lazy imports after arg validation to keep --help fast
    from config.settings import Settings
    from pipeline.orchestrator import Orchestrator

    settings = Settings.load(config_path)

    orchestrator = Orchestrator(
        ticker=args.ticker,
        settings=settings,
        output_dir=args.output_dir,
    )

    try:
        report = orchestrator.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
