"""
Candlestick Pattern Engine.

Detects high-probability candlestick patterns for sniper trade confirmation:
- W / M Patterns (Double Bottom / Double Top)
- Hammer / Shooting Star
- Bullish / Bearish Engulfing
- Pin Bar Rejections
"""
from __future__ import annotations

import numpy as np
from core.types import Candle

class CandlestickPatternDetector:
    """Detects candlestick patterns for trade confirmation."""

    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        # Sensitivity thresholds
        self.wick_body_ratio = config.get("wick_body_ratio", 2.0)
        self.engulfing_min_size = config.get("engulfing_min_size", 0.0005) # Minimum candle size for engulfing

    def detect_all(self, candles: list[Candle]) -> dict[str, dict]:
        """Detect all patterns on the most recent candles."""
        if len(candles) < 5:
            return {}

        results = {}
        
        # Single and Dual Candle Patterns (checking the latest and previous)
        latest = candles[-1]
        previous = candles[-2]
        
        # 1. Hammer / Shooting Star
        hammer = self._is_hammer(latest)
        if hammer:
            results["hammer"] = {"detected": True, "direction": "bullish", "strength": 4}
            
        shooting_star = self._is_shooting_star(latest)
        if shooting_star:
            results["shooting_star"] = {"detected": True, "direction": "bearish", "strength": 4}

        # 2. Engulfing
        engulfing = self._is_engulfing(previous, latest)
        if engulfing == "bullish":
            results["bullish_engulfing"] = {"detected": True, "direction": "bullish", "strength": 4}
        elif engulfing == "bearish":
            results["bearish_engulfing"] = {"detected": True, "direction": "bearish", "strength": 4}

        # 3. Pin Bar
        pin_bar = self._is_pin_bar(latest)
        if pin_bar == "bullish":
            results["bullish_pin_bar"] = {"detected": True, "direction": "bullish", "strength": 5}
        elif pin_bar == "bearish":
            results["bearish_pin_bar"] = {"detected": True, "direction": "bearish", "strength": 5}

        # 4. W / M Pattern (Double Bottom/Top)
        # Needs a slightly longer lookback (e.g. 10-20 candles)
        wm_pattern = self._detect_w_m_pattern(candles[-20:])
        if wm_pattern == "W":
            results["w_pattern"] = {"detected": True, "direction": "bullish", "strength": 5}
        elif wm_pattern == "M":
            results["m_pattern"] = {"detected": True, "direction": "bearish", "strength": 5}

        return results

    def _is_hammer(self, candle: Candle) -> bool:
        """A hammer has a small real body at the top and a long lower wick."""
        body = abs(candle.close - candle.open)
        lower_wick = min(candle.open, candle.close) - candle.low
        upper_wick = candle.high - max(candle.open, candle.close)
        
        if body == 0:
            return False
            
        return lower_wick >= self.wick_body_ratio * body and upper_wick <= 0.5 * body

    def _is_shooting_star(self, candle: Candle) -> bool:
        """A shooting star has a small real body at the bottom and a long upper wick."""
        body = abs(candle.close - candle.open)
        lower_wick = min(candle.open, candle.close) - candle.low
        upper_wick = candle.high - max(candle.open, candle.close)
        
        if body == 0:
            return False
            
        return upper_wick >= self.wick_body_ratio * body and lower_wick <= 0.5 * body

    def _is_pin_bar(self, candle: Candle) -> str | None:
        """A pin bar is a strong rejection candle."""
        body = abs(candle.close - candle.open)
        lower_wick = min(candle.open, candle.close) - candle.low
        upper_wick = candle.high - max(candle.open, candle.close)
        
        if body == 0:
            return None
            
        # Bullish Pin Bar
        if lower_wick >= 3 * body and upper_wick <= 0.5 * body:
            return "bullish"
        # Bearish Pin Bar
        elif upper_wick >= 3 * body and lower_wick <= 0.5 * body:
            return "bearish"
            
        return None

    def _is_engulfing(self, prev: Candle, curr: Candle) -> str | None:
        """Detects bullish or bearish engulfing patterns."""
        curr_body = abs(curr.close - curr.open)
        if curr_body < self.engulfing_min_size:
            return None
            
        prev_is_red = prev.close < prev.open
        prev_is_green = prev.close > prev.open
        curr_is_red = curr.close < curr.open
        curr_is_green = curr.close > curr.open
        
        # Bullish Engulfing: previous red, current green, current body engulfs previous
        if prev_is_red and curr_is_green:
            if curr.open <= prev.close and curr.close >= prev.open:
                return "bullish"
                
        # Bearish Engulfing: previous green, current red, current body engulfs previous
        if prev_is_green and curr_is_red:
            if curr.open >= prev.close and curr.close <= prev.open:
                return "bearish"
                
        return None

    def _detect_w_m_pattern(self, candles: list[Candle]) -> str | None:
        """Simple heuristic for W (Double Bottom) or M (Double Top)."""
        if len(candles) < 10:
            return None
            
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        # Very simple detection:
        # W Pattern: Price makes a low, bounces, makes another low near the first, then bounces.
        # Find the two lowest points.
        sorted_lows = sorted(enumerate(lows), key=lambda x: x[1])
        lowest1_idx, lowest1_val = sorted_lows[0]
        lowest2_idx, lowest2_val = sorted_lows[1]
        
        # Ensure they are separated by a bounce (at least 3 candles apart)
        if abs(lowest1_idx - lowest2_idx) >= 3:
            # Ensure the second low is recent (within last 3 candles)
            if max(lowest1_idx, lowest2_idx) >= len(candles) - 3:
                # Ensure the lows are relatively equal (within 0.05% of each other)
                if abs(lowest1_val - lowest2_val) / max(lowest1_val, 1e-8) < 0.0005:
                    return "W"
                    
        # M Pattern
        sorted_highs = sorted(enumerate(highs), key=lambda x: x[1], reverse=True)
        highest1_idx, highest1_val = sorted_highs[0]
        highest2_idx, highest2_val = sorted_highs[1]
        
        if abs(highest1_idx - highest2_idx) >= 3:
            if max(highest1_idx, highest2_idx) >= len(candles) - 3:
                if abs(highest1_val - highest2_val) / max(highest1_val, 1e-8) < 0.0005:
                    return "M"
                    
        return None
