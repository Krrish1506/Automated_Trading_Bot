"""
AlphaTrend Module — ATR-Based Trend Filter.

Computes a dynamic trailing trend line based on ATR that adapts to
volatility. Filters sideways markets using ATR percentile ranking.
"""
from __future__ import annotations

import numpy as np
from core.types import TrendDirection


class AlphaTrend:
    """ATR-filtered dynamic trend direction indicator.

    AlphaTrend creates a trailing line that only moves in the direction
    of the trend. When price is above/below this line, the trend is
    bullish/bearish. ATR percentile ranking filters sideways conditions.
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 1.0,
        sideways_percentile: float = 25.0,
        percentile_window: int = 100,
    ):
        self._atr_period = atr_period
        self._multiplier = atr_multiplier
        self._sideways_pct = sideways_percentile
        self._pct_window = percentile_window

    def compute(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> dict:
        """Compute AlphaTrend line, direction, and sideways filter.

        Returns:
            dict with: alpha_trend_line, trend_direction, is_sideways, atr
        """
        n = len(closes)
        if n < self._atr_period + 2:
            return {
                "alpha_trend_line": float(closes[-1]) if n > 0 else 0.0,
                "trend_direction": TrendDirection.NEUTRAL,
                "is_sideways": True,
                "atr": 0.0,
            }

        # ATR computation
        atr = self._compute_atr(highs, lows, closes)
        current_atr = atr[-1]

        # ATR percentile for sideways detection
        atr_pct_rank = self._atr_percentile(atr)
        is_sideways = atr_pct_rank < self._sideways_pct

        # AlphaTrend line computation
        at_line = self._compute_alpha_trend(closes, atr)
        current_at = at_line[-1]

        # Trend direction
        if closes[-1] > current_at:
            direction = TrendDirection.BULLISH
        elif closes[-1] < current_at:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL

        return {
            "alpha_trend_line": float(current_at),
            "trend_direction": direction,
            "is_sideways": bool(is_sideways),
            "atr": float(current_atr),
        }

    def _compute_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """Compute Average True Range using Wilder's smoothing."""
        n = len(highs)
        tr = np.empty(n, dtype=np.float64)
        tr[0] = highs[0] - lows[0]

        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.empty(n, dtype=np.float64)
        atr[:self._atr_period] = np.nan

        # Seed with SMA
        atr[self._atr_period - 1] = np.mean(tr[:self._atr_period])

        # Wilder's smoothing
        for i in range(self._atr_period, n):
            atr[i] = (atr[i - 1] * (self._atr_period - 1) + tr[i]) / self._atr_period

        return atr

    def _atr_percentile(self, atr: np.ndarray) -> float:
        """Compute percentile rank of current ATR vs rolling window."""
        valid = atr[~np.isnan(atr)]
        if len(valid) < 2:
            return 50.0

        window = valid[-self._pct_window:]
        current = valid[-1]
        rank = np.sum(window <= current) / len(window) * 100
        return float(rank)

    def _compute_alpha_trend(
        self,
        closes: np.ndarray,
        atr: np.ndarray,
    ) -> np.ndarray:
        """Compute the AlphaTrend line.

        The line trails price by ATR * multiplier and only moves
        in the trend direction — up in uptrend, down in downtrend.
        """
        n = len(closes)
        at = np.empty(n, dtype=np.float64)
        at[0] = closes[0]

        for i in range(1, n):
            atr_val = atr[i] if not np.isnan(atr[i]) else atr[max(0, i - 1)]
            if np.isnan(atr_val):
                atr_val = 0.0

            offset = atr_val * self._multiplier

            up_line = closes[i] - offset
            dn_line = closes[i] + offset

            if closes[i] > at[i - 1]:
                # Uptrend: only move up
                at[i] = max(at[i - 1], up_line)
            else:
                # Downtrend: only move down
                at[i] = min(at[i - 1], dn_line)

        return at
