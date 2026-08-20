"""
Structured JSON Logging & Performance Monitoring.

Configures structured JSON logging for Docker-compatible output.
Tracks pipeline latency and component performance metrics.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for Docker/stdout compatibility."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class PerformanceTracker:
    """Track component latency and throughput metrics."""

    def __init__(self):
        self._timings: dict[str, list[float]] = {}
        self._counters: dict[str, int] = {}

    def record_timing(self, component: str, duration_ms: float) -> None:
        """Record a component's execution time."""
        if component not in self._timings:
            self._timings[component] = []
        self._timings[component].append(duration_ms)
        # Keep only last 100 readings
        if len(self._timings[component]) > 100:
            self._timings[component] = self._timings[component][-100:]

    def increment(self, counter: str) -> None:
        """Increment a named counter."""
        self._counters[counter] = self._counters.get(counter, 0) + 1

    def get_stats(self) -> dict:
        """Get performance summary."""
        stats = {"counters": dict(self._counters), "latency": {}}

        for component, times in self._timings.items():
            if times:
                stats["latency"][component] = {
                    "avg_ms": round(sum(times) / len(times), 2),
                    "max_ms": round(max(times), 2),
                    "min_ms": round(min(times), 2),
                    "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 2),
                    "count": len(times),
                }

        return stats


def setup_logging(config: dict) -> PerformanceTracker:
    """Configure the logging system.

    Args:
        config: Full application config dict.

    Returns:
        PerformanceTracker instance for latency monitoring.
    """
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "json")

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(module)s.%(funcName)s | %(message)s"
        ))

    root.addHandler(handler)

    # Suppress noisy libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    tracker = PerformanceTracker()
    return tracker
