"""
Rolling in-memory candle buffer.

Aggregates raw ticks into M5 OHLCV candles and maintains a fixed-size
deque for all downstream indicator computations. Zero disk I/O.
"""
from __future__ import annotations

import time
import logging
from collections import deque
from typing import Optional

import numpy as np

from core.types import Candle
from core.events import TickEvent, CandleCloseEvent, EventType
from core.event_bus import EventBus

logger = logging.getLogger(__name__)


class CandleBuffer:
    """Rolling OHLCV candle buffer with live tick aggregation.

    Ticks are aggregated into candles of `timeframe_seconds` duration.
    When a candle completes, it is pushed into the buffer and a
    CandleCloseEvent is published to the event bus.
    """

    def __init__(
        self,
        event_bus: EventBus,
        max_candles: int = 500,
        timeframe_seconds: int = 300,
        warmup_candles: int = 200,
    ):
        self._bus = event_bus
        self._max = max_candles
        self._tf = timeframe_seconds
        self._warmup = warmup_candles
        self._buffer: deque[Candle] = deque(maxlen=max_candles)

        # Current forming candle
        self._current: Optional[Candle] = None
        self._tick_count: int = 0

        # Pre-allocated numpy arrays for fast access
        self._arrays_dirty: bool = True
        self._cached_arrays: dict[str, np.ndarray] = {}

        # Register tick handler
        self._bus.subscribe(EventType.TICK, self._on_tick)

    def _candle_start(self, ts: float) -> float:
        """Compute the candle start timestamp for a given tick timestamp."""
        return (ts // self._tf) * self._tf

    async def _on_tick(self, event: TickEvent) -> None:
        """Process incoming tick and aggregate into candles."""
        price = event.mid
        ts = event.timestamp
        self._tick_count += 1

        candle_start = self._candle_start(ts)

        if self._current is None:
            # First tick ever
            self._current = Candle(
                timestamp=candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1.0,
            )
            return

        if candle_start > self._current.timestamp:
            # New candle period → close the current candle
            completed = self._current
            self._buffer.append(completed)
            self._arrays_dirty = True

            logger.debug(
                f"Candle closed: O={completed.open:.5f} H={completed.high:.5f} "
                f"L={completed.low:.5f} C={completed.close:.5f} V={completed.volume}"
            )

            # Publish candle close event
            await self._bus.publish(CandleCloseEvent(
                timestamp=completed.timestamp,
                open=completed.open,
                high=completed.high,
                low=completed.low,
                close=completed.close,
                volume=completed.volume,
            ))

            # Start new candle
            self._current = Candle(
                timestamp=candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1.0,
            )
        else:
            # Update current candle
            self._current.high = max(self._current.high, price)
            self._current.low = min(self._current.low, price)
            self._current.close = price
            self._current.volume += 1.0

    def get_array(self, field: str, n: Optional[int] = None) -> np.ndarray:
        """Return numpy array of the last n values for a field.

        Fields: 'open', 'high', 'low', 'close', 'volume', 'timestamp'
        If n is None, returns all buffered candles.
        """
        if self._arrays_dirty:
            self._rebuild_arrays()

        arr = self._cached_arrays.get(field)
        if arr is None:
            raise ValueError(f"Unknown field: {field}")

        if n is not None:
            return arr[-n:]
        return arr

    def _rebuild_arrays(self) -> None:
        """Rebuild numpy arrays from the deque (only on dirty flag)."""
        if not self._buffer:
            for f in ("open", "high", "low", "close", "volume", "timestamp"):
                self._cached_arrays[f] = np.empty(0, dtype=np.float64)
            self._arrays_dirty = False
            return

        n = len(self._buffer)
        for f in ("open", "high", "low", "close", "volume", "timestamp"):
            arr = np.empty(n, dtype=np.float64)
            for i, c in enumerate(self._buffer):
                arr[i] = getattr(c, f)
            self._cached_arrays[f] = arr

        self._arrays_dirty = False

    def seed(self, candles: list[Candle]) -> None:
        """Seed the buffer with historical candles (for warmup)."""
        for c in candles:
            self._buffer.append(c)
        self._arrays_dirty = True
        logger.info(f"Buffer seeded with {len(candles)} candles")

    @property
    def is_warm(self) -> bool:
        """True if buffer has enough candles for indicator computation."""
        return len(self._buffer) >= self._warmup

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def current_candle(self) -> Optional[Candle]:
        return self._current

    @property
    def last_close(self) -> float:
        if self._buffer:
            return self._buffer[-1].close
        return 0.0
