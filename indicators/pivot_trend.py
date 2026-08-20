"""
Pivot Trend Module — Swing Detection & Pivot Break Logic.

Detects swing highs/lows using local extrema, identifies pivot breaks,
and determines trend reversal state from sequential pivot analysis.
"""
from __future__ import annotations

import numpy as np
from core.types import StructureBreak


class PivotTrend:
    """Swing high/low detection and pivot break trend analysis.

    Uses a configurable window to identify local extrema (pivot points),
    then tracks whether price breaks beyond these pivots to determine
    trend state and reversal signals.
    """

    def __init__(self, swing_window: int = 5):
        self._window = swing_window
        self._swing_highs: list[tuple[int, float]] = []  # (index, price)
        self._swing_lows: list[tuple[int, float]] = []
        self._last_break: StructureBreak = StructureBreak.NONE
        self._trend_state: str = "neutral"

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> dict:
        """Compute pivot points and break logic.

        Args:
            highs: Array of high prices.
            lows: Array of low prices.
            closes: Array of close prices.

        Returns:
            dict with: last_swing_high, last_swing_low, pivot_break,
                       trend_reversal
        """
        n = len(highs)
        if n < self._window * 2 + 1:
            return {
                "last_swing_high": 0.0,
                "last_swing_low": 0.0,
                "pivot_break": StructureBreak.NONE,
                "trend_reversal": False,
            }

        # Detect swing points
        self._detect_swings(highs, lows, n)

        last_sh = self._swing_highs[-1][1] if self._swing_highs else highs[-1]
        last_sl = self._swing_lows[-1][1] if self._swing_lows else lows[-1]

        # Check for pivot break
        current_close = closes[-1]
        pivot_break = StructureBreak.NONE
        trend_reversal = False

        if self._swing_highs and current_close > last_sh:
            pivot_break = StructureBreak.UP
        elif self._swing_lows and current_close < last_sl:
            pivot_break = StructureBreak.DOWN

        # Detect trend reversal (change in break direction)
        if pivot_break != StructureBreak.NONE and pivot_break != self._last_break:
            if self._last_break != StructureBreak.NONE:
                trend_reversal = True
            self._last_break = pivot_break

        return {
            "last_swing_high": float(last_sh),
            "last_swing_low": float(last_sl),
            "pivot_break": pivot_break,
            "trend_reversal": trend_reversal,
        }

    def _detect_swings(self, highs: np.ndarray, lows: np.ndarray, n: int) -> None:
        """Detect swing highs and lows using local extrema.

        A swing high at index i requires:
            high[i] >= high[i-w:i] and high[i] >= high[i+1:i+w+1]
        Symmetric for swing lows.
        """
        w = self._window
        self._swing_highs.clear()
        self._swing_lows.clear()

        for i in range(w, n - w):
            # Swing high: highest in window
            is_swing_high = True
            for j in range(i - w, i + w + 1):
                if j != i and highs[j] > highs[i]:
                    is_swing_high = False
                    break

            if is_swing_high:
                self._swing_highs.append((i, float(highs[i])))

            # Swing low: lowest in window
            is_swing_low = True
            for j in range(i - w, i + w + 1):
                if j != i and lows[j] < lows[i]:
                    is_swing_low = False
                    break

            if is_swing_low:
                self._swing_lows.append((i, float(lows[i])))

    @property
    def swing_highs(self) -> list[tuple[int, float]]:
        return self._swing_highs

    @property
    def swing_lows(self) -> list[tuple[int, float]]:
        return self._swing_lows
