"""
AI Trading Agent — Main Entry Point.

Initializes all components, wires the event-driven pipeline,
and starts the live trading loop:

    WebSocket Feed → Candle Buffer → Indicator Registry
    → Forecast Inference → Decision Engine → Execution Engine

Usage:
    python main.py                  # Live mode
    python main.py --backtest       # Run backtest
    python main.py --dry-run        # Dry run with mock feed
"""
from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from core.event_bus import EventBus
from core.candle_buffer import CandleBuffer
from feed.websocket_feed import WebSocketFeed
from indicators.registry import IndicatorRegistry
from decision.signal_generator import SignalGenerator
from execution.engine import ExecutionEngine
from monitoring.logger import setup_logging

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with env var overrides."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    # Override from environment variables
    feed = config.get("feed", {})
    feed["api_key"] = os.environ.get("OANDA_API_KEY", feed.get("api_key", ""))
    feed["account_id"] = os.environ.get("OANDA_ACCOUNT_ID", feed.get("account_id", ""))
    feed["twelvedata_api_key"] = os.environ.get(
        "TWELVEDATA_API_KEY", feed.get("twelvedata_api_key", "")
    )
    feed["finnhub_api_key"] = os.environ.get(
        "FINNHUB_API_KEY", feed.get("finnhub_api_key", "")
    )
    config["feed"] = feed

    return config


async def run_live(config: dict) -> None:
    """Run the live trading agent."""
    logger.info("=" * 60)
    logger.info("  AI Trading Agent — NSE NIFTY50 M5 (₹ INR)")
    logger.info("  Starting in LIVE mode")
    logger.info("=" * 60)

    # Initialize event bus
    event_bus = EventBus()

    # Initialize candle buffer
    buffer_cfg = config.get("buffer", {})
    candle_buffer = CandleBuffer(
        event_bus=event_bus,
        max_candles=buffer_cfg.get("max_candles", 500),
        timeframe_seconds=config.get("timeframe_seconds", 300),
        warmup_candles=buffer_cfg.get("warmup_candles", 200),
    )

    # Initialize indicator registry
    indicator_registry = IndicatorRegistry(
        event_bus=event_bus,
        candle_buffer=candle_buffer,
        config=config,
    )

    # Initialize forecast engine (lazy import — torch is heavy)
    try:
        from forecast.inference import InferenceEngine
        inference_engine = InferenceEngine(
            event_bus=event_bus,
            candle_buffer=candle_buffer,
            config=config,
        )
    except Exception as exc:
        logger.warning(f"Forecast engine disabled: {exc}")
        inference_engine = None

    # Initialize decision engine
    signal_generator = SignalGenerator(
        event_bus=event_bus,
        candle_buffer=candle_buffer,
        config=config,
    )

    # Initialize execution engine
    execution_engine = ExecutionEngine(
        event_bus=event_bus,
        config=config,
    )

    # Initialize WebSocket feed
    ws_feed = WebSocketFeed(
        event_bus=event_bus,
        config=config,
    )

    # Start event bus
    await event_bus.start()
    logger.info("Event bus started")

    # Start feed
    try:
        logger.info("Starting WebSocket feed...")
        await ws_feed.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await ws_feed.stop()
        await event_bus.stop()

        # Print final stats
        logger.info("=" * 60)
        logger.info("  Final Statistics")
        logger.info(f"  Event Bus: {json.dumps(event_bus.stats)}")
        logger.info(f"  Feed: {json.dumps(ws_feed.stats)}")
        if inference_engine:
            logger.info(f"  Inference: {json.dumps(inference_engine.stats)}")
        logger.info(f"  Execution: {json.dumps(execution_engine.stats)}")
        logger.info(f"  Buffer: {candle_buffer.size} candles")
        logger.info(f"  Indicator latency: {indicator_registry.last_compute_ms:.2f}ms")
        logger.info("=" * 60)


async def run_dry(config: dict) -> None:
    """Run in dry-run mode with synthetic tick data."""
    import time
    import numpy as np

    logger.info("=" * 60)
    logger.info("  AI Trading Agent — DRY RUN MODE")
    logger.info("=" * 60)

    event_bus = EventBus()
    buffer_cfg = config.get("buffer", {})

    candle_buffer = CandleBuffer(
        event_bus=event_bus,
        max_candles=buffer_cfg.get("max_candles", 500),
        timeframe_seconds=config.get("timeframe_seconds", 300),
        warmup_candles=50,  # Reduced for dry run
    )

    indicator_registry = IndicatorRegistry(
        event_bus=event_bus,
        candle_buffer=candle_buffer,
        config=config,
    )

    try:
        from forecast.inference import InferenceEngine
        inference_engine = InferenceEngine(
            event_bus=event_bus,
            candle_buffer=candle_buffer,
            config=config,
        )
    except Exception as exc:
        logger.warning(f"Forecast engine disabled: {exc}")
        inference_engine = None

    signal_generator = SignalGenerator(
        event_bus=event_bus,
        candle_buffer=candle_buffer,
        config=config,
    )

    execution_engine = ExecutionEngine(
        event_bus=event_bus,
        config=config,
    )

    ws_feed = WebSocketFeed(
        event_bus=event_bus,
        config=config,
    )

    await event_bus.start()

    # Generate synthetic NSE NIFTY50 (₹) candles
    np.random.seed(42)
    base_price = 24000.0
    n_candles = 300

    synthetic_candles = []
    price = base_price

    for i in range(n_candles):
        change = np.random.normal(0, 15.0)
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.normal(0, 8.0))
        l = min(o, c) - abs(np.random.normal(0, 8.0))
        v = max(1000, np.random.normal(50000, 15000))

        synthetic_candles.append({
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })
        price = c

    logger.info(f"Feeding {n_candles} synthetic NIFTY50 candles...")
    await ws_feed.feed_historical(synthetic_candles, delay=0.01)

    # Wait for pipeline to process
    await asyncio.sleep(2.0)

    await event_bus.stop()

    logger.info("=" * 60)
    logger.info("  Dry Run Complete")
    logger.info(f"  Candles processed: {candle_buffer.size}")
    logger.info(f"  Signals: {json.dumps(execution_engine.stats)}")
    if inference_engine:
        logger.info(f"  Inference: {json.dumps(inference_engine.stats)}")
    logger.info("=" * 60)


def run_backtest(config: dict) -> None:
    """Run backtest on historical data."""
    from backtest.engine import BacktestEngine

    bt_cfg = config.get("backtest", {})
    data_path = bt_cfg.get("data_path", "data/nifty50_5m.csv")

    engine = BacktestEngine(config)
    result = engine.run(data_path)

    logger.info("=" * 60)
    logger.info("  Backtest Results")
    logger.info("=" * 60)
    for key, val in result.summary().items():
        logger.info(f"  {key}: {val}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AI Trading Agent — NSE NIFTY50 M5 (₹ INR)"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtest mode"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run dry-run with synthetic data"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tracker = setup_logging(config)

    if args.backtest:
        run_backtest(config)
    elif args.dry_run:
        asyncio.run(run_dry(config))
    else:
        asyncio.run(run_live(config))


if __name__ == "__main__":
    main()
