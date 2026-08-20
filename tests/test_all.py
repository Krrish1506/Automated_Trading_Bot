"""
Comprehensive Test Suite — Unit Tests for All Modules.

Tests core infrastructure, indicators, forecast, decision engine,
and execution logic.
"""
from __future__ import annotations

import asyncio
import time
import pytest
import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════
# Core Tests
# ═══════════════════════════════════════════════════════════════

class TestEventBus:
    """Tests for the async event bus."""

    def test_subscribe_and_publish(self):
        from core.event_bus import EventBus
        from core.events import EventType, TickEvent

        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(EventType.TICK, handler)

        async def run():
            await bus.start()
            await bus.publish(TickEvent(
                timestamp=time.time(),
                bid=1.08500,
                ask=1.08515,
            ))
            await asyncio.sleep(0.1)
            await bus.stop()

        asyncio.run(run())
        assert len(received) == 1
        assert isinstance(received[0], TickEvent)
        assert received[0].bid == 1.08500

    def test_multiple_subscribers(self):
        from core.event_bus import EventBus
        from core.events import EventType, TickEvent

        bus = EventBus()
        count = [0, 0]

        async def handler1(event):
            count[0] += 1

        async def handler2(event):
            count[1] += 1

        bus.subscribe(EventType.TICK, handler1)
        bus.subscribe(EventType.TICK, handler2)

        async def run():
            await bus.start()
            await bus.publish(TickEvent(
                timestamp=time.time(), bid=1.0, ask=1.0
            ))
            await asyncio.sleep(0.1)
            await bus.stop()

        asyncio.run(run())
        assert count[0] == 1
        assert count[1] == 1


class TestCandleBuffer:
    """Tests for the candle buffer."""

    def test_tick_aggregation(self):
        from core.event_bus import EventBus
        from core.candle_buffer import CandleBuffer
        from core.events import TickEvent

        bus = EventBus()
        buffer = CandleBuffer(
            event_bus=bus,
            max_candles=100,
            timeframe_seconds=300,
            warmup_candles=2,
        )

        async def run():
            await bus.start()
            t_base = 1700000000.0

            # Feed ticks for first candle
            for i in range(10):
                await bus.publish(TickEvent(
                    timestamp=t_base + i,
                    bid=1.08500 + i * 0.00001,
                    ask=1.08515 + i * 0.00001,
                ))

            # Feed tick for next candle period
            await bus.publish(TickEvent(
                timestamp=t_base + 300,
                bid=1.08510,
                ask=1.08525,
            ))

            await asyncio.sleep(0.2)
            await bus.stop()

        asyncio.run(run())
        assert buffer.size >= 1


# ═══════════════════════════════════════════════════════════════
# Indicator Tests
# ═══════════════════════════════════════════════════════════════

class TestHoltWintersRSI:
    """Tests for Holt-Winters RSI module."""

    def test_rsi_basic(self):
        from indicators.holt_winters_rsi import HoltWintersRSI

        hw = HoltWintersRSI(rsi_period=14)
        # Generate trending data
        closes = np.linspace(1.08, 1.09, 100)
        result = hw.compute(closes)

        assert "rsi" in result
        assert 0 <= result["rsi"] <= 100
        assert "hw_rsi" in result
        assert "momentum_shift" in result

    def test_rsi_insufficient_data(self):
        from indicators.holt_winters_rsi import HoltWintersRSI

        hw = HoltWintersRSI(rsi_period=14)
        closes = np.array([1.08, 1.09, 1.10])  # Too few
        result = hw.compute(closes)
        assert result["rsi"] == 50.0  # Default


class TestPredictiveMACD:
    """Tests for Predictive MACD module."""

    def test_macd_computation(self):
        from indicators.predictive_macd import PredictiveMACD

        macd = PredictiveMACD(fast=12, slow=26, signal=9)
        closes = np.sin(np.linspace(0, 4 * np.pi, 200)) * 0.01 + 1.085
        result = macd.compute(closes)

        assert "macd" in result
        assert "signal_line" in result
        assert "histogram" in result
        assert "crossover_target" in result


class TestAlphaTrend:
    """Tests for AlphaTrend module."""

    def test_alpha_trend_uptrend(self):
        from indicators.alpha_trend import AlphaTrend

        at = AlphaTrend(atr_period=14, atr_multiplier=1.0)
        n = 200
        highs = np.linspace(1.08, 1.10, n) + 0.001
        lows = np.linspace(1.08, 1.10, n) - 0.001
        closes = np.linspace(1.08, 1.10, n)

        result = at.compute(highs, lows, closes)
        assert result["trend_direction"].value == "bullish"
        assert result["atr"] > 0

    def test_alpha_trend_downtrend(self):
        from indicators.alpha_trend import AlphaTrend

        at = AlphaTrend(atr_period=14, atr_multiplier=1.0)
        n = 200
        highs = np.linspace(1.10, 1.08, n) + 0.001
        lows = np.linspace(1.10, 1.08, n) - 0.001
        closes = np.linspace(1.10, 1.08, n)

        result = at.compute(highs, lows, closes)
        assert result["trend_direction"].value == "bearish"


class TestSMCEngine:
    """Tests for Smart Money Concepts engine."""

    def test_structure_detection(self):
        from indicators.smc_engine import SMCEngine

        smc = SMCEngine(swing_window=5, order_block_lookback=10)

        # Create data with clear uptrend structure
        n = 100
        np.random.seed(42)
        base = np.linspace(1.08, 1.10, n)
        noise = np.random.normal(0, 0.0003, n)
        opens = base + noise
        closes = base + noise + 0.0002
        highs = np.maximum(opens, closes) + abs(np.random.normal(0, 0.0002, n))
        lows = np.minimum(opens, closes) - abs(np.random.normal(0, 0.0002, n))

        result = smc.compute(opens, highs, lows, closes)
        assert hasattr(result, 'bos')
        assert hasattr(result, 'choch')
        assert hasattr(result, 'order_blocks')


class TestHiddenZone:
    """Tests for Hidden Zone detector."""

    def test_zone_detection(self):
        from indicators.hidden_zone import HiddenZoneDetector

        hz = HiddenZoneDetector()
        n = 200
        np.random.seed(42)
        base = np.linspace(1.08, 1.10, n)
        opens = base + np.random.normal(0, 0.0003, n)
        closes = base + np.random.normal(0, 0.0003, n)
        highs = np.maximum(opens, closes) + abs(np.random.normal(0, 0.0002, n))
        lows = np.minimum(opens, closes) - abs(np.random.normal(0, 0.0002, n))
        volumes = np.random.uniform(100, 1000, n)
        atr = 0.0005

        result = hz.compute(opens, highs, lows, closes, volumes, atr)
        assert "demand_zones" in result
        assert "supply_zones" in result
        assert "nearest_demand_dist" in result


# ═══════════════════════════════════════════════════════════════
# Forecast Tests
# ═══════════════════════════════════════════════════════════════

class TestLSTMModel:
    """Tests for the LSTM forecast model."""

    def test_forward_pass(self):
        from forecast.model import LSTMForecaster

        model = LSTMForecaster(input_features=12, hidden_dim=64, num_layers=2)
        x = torch.randn(1, 30, 12)  # (batch, seq, features)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1  # Sigmoid output

    def test_inference_latency(self):
        from forecast.model import LSTMForecaster

        model = LSTMForecaster(input_features=12, hidden_dim=128, num_layers=2)
        model.eval()
        x = torch.randn(1, 30, 12)

        # Warmup
        with torch.no_grad():
            model(x)

        # Measure
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        assert avg_ms < 50  # Should be well under 50ms on CPU


class TestTransformerModel:
    """Tests for the Transformer forecast model."""

    def test_forward_pass(self):
        from forecast.model import TransformerForecaster

        model = TransformerForecaster(input_features=12, d_model=64)
        x = torch.randn(1, 30, 12)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1


class TestFeatureBuilder:
    """Tests for the feature builder."""

    def test_build_returns_tensor(self):
        from forecast.feature_builder import FeatureBuilder
        from core.event_bus import EventBus
        from core.candle_buffer import CandleBuffer
        from core.types import IndicatorState, TrendDirection, StructureState

        # Create a mock buffer with enough data
        bus = EventBus()
        buffer = CandleBuffer(bus, max_candles=100, timeframe_seconds=300, warmup_candles=2)

        # Manually fill buffer
        from core.types import Candle
        from collections import deque
        buffer._buffer = deque(maxlen=100)
        for i in range(50):
            buffer._buffer.append(Candle(
                timestamp=i * 300.0,
                open=1.085 + i * 0.0001,
                high=1.087 + i * 0.0001,
                low=1.083 + i * 0.0001,
                close=1.086 + i * 0.0001,
                volume=float(500 + i),
            ))
        buffer._arrays_dirty = True

        state = IndicatorState()
        state.atr = 0.0005
        state.rsi = 55.0
        state.macd_histogram = 0.0001
        state.structure = StructureState()
        state.trend_direction = TrendDirection.BULLISH

        fb = FeatureBuilder(sequence_length=30, input_features=12)
        tensor = fb.build(buffer, state)

        assert tensor is not None
        assert tensor.shape == (1, 30, 12)


# ═══════════════════════════════════════════════════════════════
# Decision Engine Tests
# ═══════════════════════════════════════════════════════════════

class TestScorer:
    """Tests for the weighted scorer."""

    def test_bullish_scoring(self):
        from decision.scorer import Scorer
        from core.types import IndicatorState, TrendDirection, StructureState, StructureBreak

        scorer = Scorer()
        state = IndicatorState()
        state.trend_direction = TrendDirection.BULLISH
        state.trend_confirmed = True
        state.trend_strength = 0.8
        state.structure = StructureState()
        state.structure.bos = True
        state.structure.bos_direction = StructureBreak.UP
        state.rsi = 45.0
        state.momentum_shift = "bullish"
        state.macd_histogram = 0.001
        state.crossover_direction = "up"

        composite, individual = scorer.score(state, 0.75, "up")

        assert composite > 0  # Should be bullish
        assert len(individual) == 9  # All modules scored

    def test_neutral_scoring(self):
        from decision.scorer import Scorer
        from core.types import IndicatorState, StructureState

        scorer = Scorer()
        state = IndicatorState()
        state.structure = StructureState()

        composite, _ = scorer.score(state, 0.5, "neutral")
        assert abs(composite) < 0.3  # Should be near neutral


class TestRiskFilter:
    """Tests for the risk filter."""

    def test_spread_rejection(self):
        from decision.risk_filter import RiskFilter
        from core.types import IndicatorState, StructureState

        rf = RiskFilter(max_spread_pips=3.0)
        state = IndicatorState()
        state.structure = StructureState()

        rejected, reasons = rf.check(state, spread_pips=5.0)
        assert rejected
        assert any("Spread" in r for r in reasons)

    def test_daily_loss_rejection(self):
        from decision.risk_filter import RiskFilter
        from core.types import IndicatorState, StructureState

        rf = RiskFilter(daily_loss_limit_pips=50.0)
        rf.update_pnl(-60.0)  # Exceed limit

        state = IndicatorState()
        state.structure = StructureState()

        rejected, reasons = rf.check(state)
        assert rejected
        assert any("Daily loss" in r for r in reasons)


class TestPositionSizer:
    """Tests for position sizing."""

    def test_buy_position(self):
        from decision.position_sizing import PositionSizer
        from core.types import IndicatorState, SignalType, StructureState, FibonacciLevels

        ps = PositionSizer(sl_atr_multiplier=1.5, min_risk_reward=1.5)
        state = IndicatorState()
        state.atr = 0.00050
        state.last_swing_low = 1.08300
        state.last_swing_high = 1.08800
        state.structure = StructureState()
        state.fibonacci = FibonacciLevels()

        result = ps.compute(SignalType.BUY, state, 1.08500)

        assert result["stop_loss"] > 0
        assert result["take_profit"] > 1.08500
        assert result["risk_reward_ratio"] >= 1.5


# ═══════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """End-to-end pipeline integration test."""

    def test_indicator_pipeline(self):
        """Test full indicator computation on synthetic data."""
        from indicators.holt_winters_rsi import HoltWintersRSI
        from indicators.predictive_macd import PredictiveMACD
        from indicators.pivot_trend import PivotTrend
        from indicators.alpha_trend import AlphaTrend
        from indicators.smc_engine import SMCEngine

        n = 200
        np.random.seed(42)
        base = np.linspace(1.08, 1.10, n)
        noise = np.random.normal(0, 0.0003, n)
        closes = base + noise
        opens = closes - np.random.normal(0, 0.0001, n)
        highs = np.maximum(opens, closes) + abs(np.random.normal(0, 0.0002, n))
        lows = np.minimum(opens, closes) - abs(np.random.normal(0, 0.0002, n))

        # All indicators should compute without error
        rsi = HoltWintersRSI().compute(closes)
        assert rsi["rsi"] > 0

        macd = PredictiveMACD().compute(closes)
        assert "macd" in macd

        pivot = PivotTrend().compute(highs, lows, closes)
        assert "last_swing_high" in pivot

        alpha = AlphaTrend().compute(highs, lows, closes)
        assert "trend_direction" in alpha

        smc = SMCEngine().compute(opens, highs, lows, closes)
        assert hasattr(smc, "bos")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
