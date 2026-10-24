"""
Signal Generator — Final BUY/SELL/HOLD Decision Logic.

Combines composite score, risk filters, confluence checks, and
position sizing to produce the final structured trade signal.
Subscribes to FORECAST_READY and publishes SIGNAL_GENERATED.
"""
from __future__ import annotations

import time
import logging

from core.event_bus import EventBus
from core.candle_buffer import CandleBuffer
from core.events import (
    ForecastReadyEvent, IndicatorsReadyEvent,
    SignalGeneratedEvent, EventType,
)
from core.types import (
    TradeSignal, IndicatorState, MarketState, SignalType,
    TrendDirection, StructureBreak,
)
from decision.scorer import Scorer
from decision.confluence import ConfluenceDetector
from decision.risk_filter import RiskFilter
from decision.position_sizing import PositionSizer
from indicators.candle_patterns import CandlestickPatternDetector

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Final trade signal generation engine.

    Decision flow:
    1. Wait for both indicators and forecast to be ready
    2. Compute weighted composite score
    3. Detect confluences
    4. Run risk filters
    5. Generate BUY/SELL/HOLD signal
    6. Compute SL/TP/RR
    7. Publish structured signal
    """

    def __init__(
        self,
        event_bus: EventBus,
        candle_buffer: CandleBuffer,
        config: dict,
    ):
        self._bus = event_bus
        self._buffer = candle_buffer
        self._symbol = config.get("symbol", "NIFTY50")

        decision_cfg = config.get("decision", {})
        risk_cfg = config.get("risk", {})
        pos_cfg = config.get("position", {})

        self._scorer = Scorer(weights=decision_cfg.get("weights"))
        self._confluence = ConfluenceDetector(
            min_confluences=decision_cfg.get("confluence_min", 3)
        )
        self._risk_filter = RiskFilter(
            max_spread_pips=risk_cfg.get("max_spread_pips", 3.0),
            volatility_spike_mult=risk_cfg.get("volatility_spike_mult", 3.0),
            daily_loss_limit_pips=risk_cfg.get("daily_loss_limit_pips", 50.0),
            adx_sideways_threshold=risk_cfg.get("adx_sideways_threshold", 20),
        )
        self._position_sizer = PositionSizer(
            sl_atr_multiplier=pos_cfg.get("sl_atr_multiplier", 1.5),
            sl_min_pips=pos_cfg.get("sl_min_pips", 5.0),
            sl_spread_buffer_pips=pos_cfg.get("sl_spread_buffer_pips", 1.0),
            min_risk_reward=pos_cfg.get("min_risk_reward", 1.5),
        )
        self._pattern_detector = CandlestickPatternDetector(
            config.get("candle_patterns", {})
        )

        forecast_cfg = config.get("forecast", {})
        self._buy_threshold = forecast_cfg.get("probability_threshold_buy", 0.65)
        self._sell_threshold = forecast_cfg.get("probability_threshold_sell", 0.35)

        # State tracking
        self._latest_state: IndicatorState | None = None
        self._latest_forecast_prob: float = 0.5
        self._latest_forecast_dir: str = "neutral"

        # Subscribe to events
        self._bus.subscribe(EventType.INDICATORS_READY, self._on_indicators)
        self._bus.subscribe(EventType.FORECAST_READY, self._on_forecast)

    async def _on_indicators(self, event: IndicatorsReadyEvent) -> None:
        """Cache latest indicator state."""
        self._latest_state = event.state

    async def _on_forecast(self, event: ForecastReadyEvent) -> None:
        """Process forecast and generate final signal."""
        self._latest_forecast_prob = event.probability
        self._latest_forecast_dir = event.direction

        if self._latest_state is None:
            return

        signal = self._generate_signal()

        logger.info(
            f"Signal: {signal.signal.value} | "
            f"Confidence: {signal.confidence:.0f}% | "
            f"SL: {signal.stop_loss:.5f} | "
            f"TP: {signal.take_profit:.5f} | "
            f"RR: {signal.risk_reward_ratio:.2f} | "
            f"Confluences: {len(signal.active_confluence)}"
        )

        await self._bus.publish(SignalGeneratedEvent(
            timestamp=time.time(),
            signal=signal,
        ))

    def _generate_signal(self) -> TradeSignal:
        """Core signal generation logic."""
        state = self._latest_state
        prob = self._latest_forecast_prob
        fdir = self._latest_forecast_dir

        # 1. Compute weighted scores
        composite, individual_scores = self._scorer.score(state, prob, fdir)

        # 2. Detect confluences
        confluences = self._confluence.detect(state, fdir, individual_scores)

        # 3. Determine market state
        market_state = self._determine_market_state(state)

        # 4. Risk filter check
        rejected, reject_reasons = self._risk_filter.check(state)

        # 5. Signal decision
        signal_type = SignalType.HOLD
        pattern_name = ""
        fib_level = 0.0

        if not rejected:
            signal_type, pattern_name, fib_level = self._decide_signal(
                state, composite, prob, confluences
            )

        # 6. Confidence score (0-100)
        confidence = abs(composite) * 100
        if len(confluences) >= 3:
            confidence = min(confidence * 1.2, 100.0)
        if rejected:
            confidence = 0.0

        # 7. Position sizing
        entry_price = state.current_price
        pos = self._position_sizer.compute(signal_type, state, entry_price)

        return TradeSignal(
            symbol=self._symbol,
            current_price=state.current_price,
            market_state=market_state,
            signal=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=pos["stop_loss"],
            take_profit=pos["take_profit"],
            risk_reward_ratio=pos["risk_reward_ratio"],
            fib_level=fib_level,
            pattern=pattern_name,
            structure_state=state.structure,
            active_confluence=confluences,
        )

    def _decide_signal(
        self,
        state: IndicatorState,
        composite: float,
        prob: float,
        confluences: list[str],
    ) -> tuple[SignalType, str, float]:
        """Apply strict BUY/SELL/HOLD rules (ML + Fibonacci + Pattern Sniper Strategy)."""
        
        # 1. Fibonacci Golden Zone Check
        candles = self._buffer.get_candles(limit=20)
        in_golden_zone_buy = False
        in_golden_zone_sell = False
        fib_level = 0.0
        
        if len(candles) >= 20:
            recent_high = max(c.high for c in candles)
            recent_low = min(c.low for c in candles)
            swing_range = recent_high - recent_low
            if swing_range > 0:
                fib_level = (state.current_price - recent_low) / swing_range
                in_golden_zone_buy = 0.20 <= fib_level <= 0.65
                in_golden_zone_sell = 0.35 <= fib_level <= 0.80

        # 2. Candlestick Pattern Confirmation Check
        detected_patterns = self._pattern_detector.detect_all(candles)
        bullish_pattern = None
        bearish_pattern = None
        for name, data in detected_patterns.items():
            if data["direction"] == "bullish":
                bullish_pattern = name
            elif data["direction"] == "bearish":
                bearish_pattern = name

        # BUY conditions
        bullish_bos = (
            state.structure.bos and
            state.structure.bos_direction == StructureBreak.UP
        )
        alpha_bullish = state.trend_direction == TrendDirection.BULLISH
        forecast_bullish = prob >= 0.60  # Strict ML Confidence
        near_demand = (
            state.nearest_demand_dist < 3.0 or
            any(ob.direction == "bullish" for ob in state.structure.order_blocks)
        )
        sufficient_confluence = self._confluence.has_sufficient_confluence(confluences)

        # Sniper Gate: ALL conditions must be met
        if (bullish_bos and alpha_bullish and forecast_bullish and
                near_demand and sufficient_confluence and in_golden_zone_buy and bullish_pattern):
            return SignalType.BUY, bullish_pattern, fib_level

        # SELL conditions
        bearish_bos = (
            state.structure.bos and
            state.structure.bos_direction == StructureBreak.DOWN
        )
        alpha_bearish = state.trend_direction == TrendDirection.BEARISH
        forecast_bearish = prob <= 0.40  # Strict ML Confidence (1.0 - 0.60 = 0.40)
        near_supply = (
            state.nearest_supply_dist < 3.0 or
            any(ob.direction == "bearish" for ob in state.structure.order_blocks)
        )

        # Sniper Gate: ALL conditions must be met
        if (bearish_bos and alpha_bearish and forecast_bearish and
                near_supply and sufficient_confluence and in_golden_zone_sell and bearish_pattern):
            return SignalType.SELL, bearish_pattern, fib_level

        return SignalType.HOLD, "", fib_level

    def _determine_market_state(self, state: IndicatorState) -> MarketState:
        """Determine overall market state from indicators."""
        if state.is_sideways and not state.trend_confirmed:
            return MarketState.SIDEWAYS

        if state.trend_direction == TrendDirection.BULLISH:
            return MarketState.TRENDING_UP
        elif state.trend_direction == TrendDirection.BEARISH:
            return MarketState.TRENDING_DOWN

        return MarketState.SIDEWAYS
