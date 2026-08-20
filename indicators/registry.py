"""
Indicator Registry — Central Orchestrator for All Indicator Modules.

Manages initialization and execution of all indicator modules in
dependency order. Collects all outputs into a unified IndicatorState
and publishes INDICATORS_READY event after each candle close.
"""
from __future__ import annotations

import time
import logging

import numpy as np

from core.event_bus import EventBus
from core.candle_buffer import CandleBuffer
from core.events import CandleCloseEvent, IndicatorsReadyEvent, EventType
from core.types import IndicatorState

from indicators.holt_winters_rsi import HoltWintersRSI
from indicators.predictive_macd import PredictiveMACD
from indicators.pivot_trend import PivotTrend
from indicators.alpha_trend import AlphaTrend
from indicators.alpha_trend_plus import AlphaTrendPlus
from indicators.smc_engine import SMCEngine
from indicators.hidden_zone import HiddenZoneDetector
from indicators.fibonacci import FibonacciEngine
from indicators.harmonic import HarmonicScanner

logger = logging.getLogger(__name__)


class IndicatorRegistry:
    """Central registry and orchestrator for all indicator modules.

    On each CANDLE_CLOSE event:
    1. Extracts arrays from the candle buffer
    2. Runs all indicator modules in sequence
    3. Collects outputs into IndicatorState
    4. Publishes INDICATORS_READY event

    All computation happens in-memory with zero disk I/O.
    """

    def __init__(
        self,
        event_bus: EventBus,
        candle_buffer: CandleBuffer,
        config: dict,
    ):
        self._bus = event_bus
        self._buffer = candle_buffer
        self._cfg = config.get("indicators", {})

        # Initialize all modules
        rsi_cfg = self._cfg.get("rsi", {})
        hw_cfg = self._cfg.get("holt_winters", {})
        self._hw_rsi = HoltWintersRSI(
            rsi_period=rsi_cfg.get("period", 14),
            alpha=hw_cfg.get("alpha", 0.3),
            beta=hw_cfg.get("beta", 0.05),
            gamma=hw_cfg.get("gamma", 0.1),
        )

        macd_cfg = self._cfg.get("macd", {})
        self._macd = PredictiveMACD(
            fast=macd_cfg.get("fast", 12),
            slow=macd_cfg.get("slow", 26),
            signal=macd_cfg.get("signal", 9),
        )

        pivot_cfg = self._cfg.get("pivot", {})
        self._pivot = PivotTrend(swing_window=pivot_cfg.get("swing_window", 5))

        at_cfg = self._cfg.get("alpha_trend", {})
        self._alpha_trend = AlphaTrend(
            atr_period=at_cfg.get("atr_period", 14),
            atr_multiplier=at_cfg.get("atr_multiplier", 1.0),
        )

        atp_cfg = self._cfg.get("alpha_trend_plus", {})
        self._alpha_plus = AlphaTrendPlus(
            atr_period=atp_cfg.get("atr_period", 14),
            atr_multiplier=atp_cfg.get("atr_multiplier", 1.5),
            adx_period=atp_cfg.get("adx_period", 14),
            adx_threshold=atp_cfg.get("adx_threshold", 20.0),
            volume_surge_mult=atp_cfg.get("volume_surge_mult", 1.5),
        )

        smc_cfg = self._cfg.get("smc", {})
        self._smc = SMCEngine(
            swing_window=smc_cfg.get("swing_window", 5),
            order_block_lookback=smc_cfg.get("order_block_lookback", 10),
        )

        hz_cfg = self._cfg.get("hidden_zone", {})
        self._hidden_zone = HiddenZoneDetector(
            vol_cluster_window=hz_cfg.get("vol_cluster_window", 20),
            imbalance_threshold=hz_cfg.get("imbalance_threshold", 1.5),
            zone_max_age=hz_cfg.get("zone_max_age", 100),
        )

        fib_cfg = self._cfg.get("fibonacci", {})
        self._fibonacci = FibonacciEngine(
            extension_levels=fib_cfg.get("extension_levels", [1.272, 1.618, 2.0]),
            retracement_levels=fib_cfg.get("retracement_levels", [0.382, 0.5, 0.618]),
            vwap_period=fib_cfg.get("vwap_period", 20),
        )

        hmn_cfg = self._cfg.get("harmonic", {})
        self._harmonic = HarmonicScanner(
            tolerance=hmn_cfg.get("tolerance", 0.05),
            min_leg_bars=hmn_cfg.get("min_leg_bars", 5),
        )

        # Track computation times
        self._last_compute_ms: float = 0.0

        # Subscribe to candle close events
        self._bus.subscribe(EventType.CANDLE_CLOSE, self._on_candle_close)

    async def _on_candle_close(self, event: CandleCloseEvent) -> None:
        """Process candle close: run all indicators and publish state."""
        if not self._buffer.is_warm:
            logger.debug(
                f"Buffer warming up: {self._buffer.size} candles "
                f"(need {self._buffer._warmup})"
            )
            return

        t0 = time.perf_counter()

        state = self._compute_all()

        self._last_compute_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Indicators computed in {self._last_compute_ms:.2f}ms | "
            f"RSI={state.rsi:.1f} | AT={state.trend_direction.value} | "
            f"BOS={state.structure.bos}"
        )

        await self._bus.publish(IndicatorsReadyEvent(
            timestamp=event.timestamp,
            state=state,
        ))

    def _compute_all(self) -> IndicatorState:
        """Execute all indicator modules and collect results."""
        state = IndicatorState()

        # Extract arrays from buffer
        opens = self._buffer.get_array("open")
        highs = self._buffer.get_array("high")
        lows = self._buffer.get_array("low")
        closes = self._buffer.get_array("close")
        volumes = self._buffer.get_array("volume")

        state.current_price = float(closes[-1]) if len(closes) > 0 else 0.0

        # 1. Holt-Winters RSI
        rsi_result = self._hw_rsi.compute(closes)
        state.rsi = rsi_result["rsi"]
        state.hw_rsi = rsi_result["hw_rsi"]
        state.momentum_shift = rsi_result["momentum_shift"]

        # 2. Predictive MACD
        macd_result = self._macd.compute(closes)
        state.macd = macd_result["macd"]
        state.macd_signal = macd_result["signal_line"]
        state.macd_histogram = macd_result["histogram"]
        state.crossover_target = macd_result["crossover_target"]
        state.crossover_direction = macd_result["crossover_direction"]

        # 3. Pivot Trend
        pivot_result = self._pivot.compute(highs, lows, closes)
        state.last_swing_high = pivot_result["last_swing_high"]
        state.last_swing_low = pivot_result["last_swing_low"]
        state.pivot_break = pivot_result["pivot_break"]
        state.trend_reversal = pivot_result["trend_reversal"]

        # 4. AlphaTrend
        at_result = self._alpha_trend.compute(highs, lows, closes)
        state.alpha_trend_line = at_result["alpha_trend_line"]
        state.trend_direction = at_result["trend_direction"]
        state.is_sideways = at_result["is_sideways"]
        state.atr = at_result["atr"]

        # 5. AlphaTrend++
        atp_result = self._alpha_plus.compute(highs, lows, closes, volumes)
        state.alpha_plus_trend = atp_result["alpha_plus_trend"]
        state.trend_confirmed = atp_result["trend_confirmed"]
        state.trend_strength = atp_result["trend_strength"]

        # 6. SMC Engine
        state.structure = self._smc.compute(opens, highs, lows, closes)

        # 7. Hidden Zone Detector
        zone_result = self._hidden_zone.compute(
            opens, highs, lows, closes, volumes, state.atr
        )
        state.demand_zones = zone_result["demand_zones"]
        state.supply_zones = zone_result["supply_zones"]
        state.nearest_demand_dist = zone_result["nearest_demand_dist"]
        state.nearest_supply_dist = zone_result["nearest_supply_dist"]

        # 8. Fibonacci
        state.fibonacci = self._fibonacci.compute(
            highs, lows, closes, volumes, state.atr
        )

        # 9. Harmonic Scanner
        harmonic_result = self._harmonic.compute(highs, lows, closes)
        state.active_pattern = harmonic_result["active_pattern"]

        return state

    @property
    def last_compute_ms(self) -> float:
        return self._last_compute_ms
