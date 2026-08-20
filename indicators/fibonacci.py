"""
Dynamic Fibonacci Extensions & Volume-Weighted Pivot Module.

Computes:
- Dynamic Fibonacci extensions from latest swing structure (1.272/1.618/2.0)
- Volume-Weighted Fibonacci Pivots using VWAP integration
- Confluence counting (fib levels near other key levels)
"""
from __future__ import annotations

import numpy as np
from core.types import FibonacciLevels


class FibonacciEngine:
    """Fibonacci extension and volume-weighted pivot computation.

    Identifies the latest complete swing leg and projects Fibonacci
    extension levels. Volume-weighted pivots adjust traditional
    S/R levels using volume cluster analysis.
    """

    def __init__(
        self,
        extension_levels: list[float] | None = None,
        retracement_levels: list[float] | None = None,
        vwap_period: int = 20,
        swing_window: int = 5,
    ):
        self._ext_levels = extension_levels or [1.272, 1.618, 2.0]
        self._ret_levels = retracement_levels or [0.382, 0.5, 0.618]
        self._vwap_period = vwap_period
        self._swing_window = swing_window

    def compute(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        atr: float,
    ) -> FibonacciLevels:
        """Compute Fibonacci levels from latest swing structure.

        Returns:
            FibonacciLevels with extensions, retracements, VW pivots,
            nearest distance, and confluence count.
        """
        result = FibonacciLevels()
        n = len(closes)

        if n < self._swing_window * 2 + 3:
            return result

        # Find latest swing high and swing low
        swing_high, sh_idx = self._find_last_swing_high(highs, n)
        swing_low, sl_idx = self._find_last_swing_low(lows, n)

        if swing_high is None or swing_low is None:
            return result

        current = closes[-1]

        # Determine swing direction (which came first)
        leg_range = abs(swing_high - swing_low)
        if leg_range < 1e-10:
            return result

        if sh_idx < sl_idx:
            # Downswing: high → low → extensions project down
            extensions = {}
            for level in self._ext_levels:
                ext_price = swing_high - leg_range * level
                extensions[level] = float(ext_price)

            retracements = {}
            for level in self._ret_levels:
                ret_price = swing_low + leg_range * level
                retracements[level] = float(ret_price)
        else:
            # Upswing: low → high → extensions project up
            extensions = {}
            for level in self._ext_levels:
                ext_price = swing_low + leg_range * level
                extensions[level] = float(ext_price)

            retracements = {}
            for level in self._ret_levels:
                ret_price = swing_high - leg_range * level
                retracements[level] = float(ret_price)

        # Volume-weighted pivots
        vw_pivots = self._compute_vw_pivots(highs, lows, closes, volumes)

        # Nearest fibonacci distance (ATR-normalized)
        all_levels = list(extensions.values()) + list(retracements.values())
        if all_levels and atr > 0:
            distances = [abs(current - lvl) / atr for lvl in all_levels]
            nearest_fib_dist = min(distances)
        else:
            nearest_fib_dist = float('inf')

        # Confluence count: how many fib levels are within 0.5 ATR of each other
        confluence = 0
        if atr > 0:
            all_lvls = all_levels + list(vw_pivots.values())
            for i in range(len(all_lvls)):
                for j in range(i + 1, len(all_lvls)):
                    if abs(all_lvls[i] - all_lvls[j]) < atr * 0.5:
                        confluence += 1

        result.extensions = extensions
        result.retracements = retracements
        result.vw_pivot_levels = vw_pivots
        result.nearest_fib_dist = float(nearest_fib_dist)
        result.confluence_count = confluence

        return result

    def _find_last_swing_high(
        self, highs: np.ndarray, n: int,
    ) -> tuple[float | None, int]:
        """Find the most recent swing high."""
        w = self._swing_window
        for i in range(n - w - 1, w - 1, -1):
            is_sh = True
            for j in range(i - w, i + w + 1):
                if j != i and j < n and highs[j] > highs[i]:
                    is_sh = False
                    break
            if is_sh:
                return float(highs[i]), i
        return None, -1

    def _find_last_swing_low(
        self, lows: np.ndarray, n: int,
    ) -> tuple[float | None, int]:
        """Find the most recent swing low."""
        w = self._swing_window
        for i in range(n - w - 1, w - 1, -1):
            is_sl = True
            for j in range(i - w, i + w + 1):
                if j != i and j < n and lows[j] < lows[i]:
                    is_sl = False
                    break
            if is_sl:
                return float(lows[i]), i
        return None, -1

    def _compute_vw_pivots(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> dict:
        """Compute volume-weighted pivot levels.

        Traditional pivot: (H + L + C) / 3, weighted by volume
        clusters to produce more accurate S/R levels.
        """
        n = len(closes)
        w = min(self._vwap_period, n)

        h = highs[-w:]
        l = lows[-w:]
        c = closes[-w:]
        v = volumes[-w:]

        total_vol = np.sum(v)
        if total_vol <= 0:
            pp = (h[-1] + l[-1] + c[-1]) / 3
        else:
            # Volume-weighted typical price
            typical = (h + l + c) / 3
            pp = float(np.sum(typical * v) / total_vol)

        # Volume-weighted high/low
        vw_high = float(np.sum(h * v) / total_vol) if total_vol > 0 else float(h[-1])
        vw_low = float(np.sum(l * v) / total_vol) if total_vol > 0 else float(l[-1])

        return {
            "pivot": pp,
            "r1": 2 * pp - vw_low,
            "s1": 2 * pp - vw_high,
            "r2": pp + (vw_high - vw_low),
            "s2": pp - (vw_high - vw_low),
        }
