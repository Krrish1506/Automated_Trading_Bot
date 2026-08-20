"""
Type definitions for the AI Trading Agent.

All shared data structures used across modules — candles, signals,
zones, structure states, and enumerations.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enumerations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MarketState(str, enum.Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"


class SignalType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TrendDirection(str, enum.Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class StructureBreak(str, enum.Enum):
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(slots=True)
class Candle:
    """Single OHLCV candle."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class ZoneInfo:
    """Supply or demand zone."""
    price_high: float
    price_low: float
    strength: float
    touch_count: int
    zone_type: str  # "demand" or "supply"
    age_bars: int


@dataclass(slots=True)
class OrderBlock:
    """Smart Money order block."""
    price_high: float
    price_low: float
    direction: str  # "bullish" or "bearish"
    strength: float
    age_bars: int


@dataclass(slots=True)
class StructureState:
    """Current market structure state from SMC engine."""
    bos: bool = False
    bos_direction: StructureBreak = StructureBreak.NONE
    choch: bool = False
    choch_direction: StructureBreak = StructureBreak.NONE
    liquidity_sweep: bool = False
    structure_bias: TrendDirection = TrendDirection.NEUTRAL
    order_blocks: List[OrderBlock] = field(default_factory=list)


@dataclass(slots=True)
class HarmonicPattern:
    """Detected harmonic XABCD pattern."""
    pattern_type: str  # gartley, butterfly, bat, crab
    prz_high: float
    prz_low: float
    completion_pct: float
    predicted_prz: Optional[float] = None
    direction: str = ""  # bullish or bearish


@dataclass(slots=True)
class FibonacciLevels:
    """Fibonacci extension and retracement levels."""
    extensions: dict = field(default_factory=dict)   # {1.272: price, ...}
    retracements: dict = field(default_factory=dict)  # {0.618: price, ...}
    vw_pivot_levels: dict = field(default_factory=dict)
    nearest_fib_dist: float = 0.0
    confluence_count: int = 0


@dataclass(slots=True)
class IndicatorState:
    """Complete state from all indicator modules."""
    # Holt-Winters RSI
    rsi: float = 50.0
    hw_rsi: float = 50.0
    momentum_shift: str = "neutral"  # bullish/bearish/neutral

    # Predictive MACD
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    crossover_target: float = 0.0
    crossover_direction: str = "none"

    # Pivot Trend
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0
    pivot_break: StructureBreak = StructureBreak.NONE
    trend_reversal: bool = False

    # AlphaTrend
    alpha_trend_line: float = 0.0
    trend_direction: TrendDirection = TrendDirection.NEUTRAL
    is_sideways: bool = True

    # AlphaTrend++
    alpha_plus_trend: float = 0.0
    trend_confirmed: bool = False
    trend_strength: float = 0.0

    # SMC
    structure: StructureState = field(default_factory=StructureState)

    # Hidden Zones
    demand_zones: List[ZoneInfo] = field(default_factory=list)
    supply_zones: List[ZoneInfo] = field(default_factory=list)
    nearest_demand_dist: float = float('inf')
    nearest_supply_dist: float = float('inf')

    # Fibonacci
    fibonacci: FibonacciLevels = field(default_factory=FibonacciLevels)

    # Harmonics
    active_pattern: Optional[HarmonicPattern] = None

    # ATR (shared)
    atr: float = 0.0

    # Current price
    current_price: float = 0.0


@dataclass(slots=True)
class TradeSignal:
    """Final structured trade signal output."""
    symbol: str = "EUR/USD"
    current_price: float = 0.0
    market_state: MarketState = MarketState.SIDEWAYS
    signal: SignalType = SignalType.HOLD
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    fib_level: float = 0.0
    pattern: str = ""
    structure_state: StructureState = field(default_factory=StructureState)
    active_confluence: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to the strict JSON output format."""
        return {
            "symbol": self.symbol,
            "current_price": round(self.current_price, 5),
            "market_state": self.market_state.value,
            "signal": self.signal.value,
            "confidence": round(self.confidence, 1),
            "entry_price": round(self.entry_price, 5),
            "stop_loss": round(self.stop_loss, 5),
            "take_profit": round(self.take_profit, 5),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "fib_level": round(self.fib_level, 3),
            "pattern": self.pattern,
            "structure_state": {
                "BOS": self.structure_state.bos,
                "CHOCH": self.structure_state.choch,
                "liquidity_sweep": self.structure_state.liquidity_sweep,
            },
            "active_confluence": self.active_confluence,
        }
