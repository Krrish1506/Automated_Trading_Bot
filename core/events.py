"""
Event definitions for the async event-driven architecture.

Each event is a typed dataclass carrying its payload. The EventBus
dispatches these to subscribers based on event type.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from core.types import IndicatorState, TradeSignal


class EventType(str, enum.Enum):
    TICK = "TICK"
    CANDLE_CLOSE = "CANDLE_CLOSE"
    INDICATORS_READY = "INDICATORS_READY"
    FORECAST_READY = "FORECAST_READY"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


@dataclass(slots=True)
class Event:
    """Base event with type discriminator."""
    event_type: EventType = EventType.ERROR  # Overridden by subclass __post_init__
    timestamp: float = 0.0


@dataclass(slots=True)
class TickEvent(Event):
    """Raw price tick from the WebSocket feed."""
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.TICK
        if self.mid == 0.0 and self.bid > 0 and self.ask > 0:
            self.mid = (self.bid + self.ask) / 2.0


@dataclass(slots=True)
class CandleCloseEvent(Event):
    """Fired when a 5-minute candle completes."""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.CANDLE_CLOSE


@dataclass(slots=True)
class IndicatorsReadyEvent(Event):
    """All indicator modules have finished computation."""
    state: IndicatorState = field(default_factory=IndicatorState)

    def __post_init__(self):
        self.event_type = EventType.INDICATORS_READY


@dataclass(slots=True)
class ForecastReadyEvent(Event):
    """AI forecast model has produced a prediction."""
    probability: float = 0.5
    direction: str = "neutral"  # up / down / neutral

    def __post_init__(self):
        self.event_type = EventType.FORECAST_READY


@dataclass(slots=True)
class SignalGeneratedEvent(Event):
    """Final trade signal from the decision engine."""
    signal: TradeSignal = field(default_factory=TradeSignal)

    def __post_init__(self):
        self.event_type = EventType.SIGNAL_GENERATED


@dataclass(slots=True)
class ErrorEvent(Event):
    """System error event."""
    component: str = ""
    message: str = ""
    details: Any = None

    def __post_init__(self):
        self.event_type = EventType.ERROR
