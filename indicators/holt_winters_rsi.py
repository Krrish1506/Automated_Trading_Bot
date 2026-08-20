"""
Holt-Winters Smoothed RSI Module.

Applies standard RSI(14) then triple exponential smoothing (Holt-Winters
additive) on the RSI series to detect early momentum shifts before raw
RSI generates signals.
"""
from __future__ import annotations

import numpy as np


class HoltWintersRSI:
    """RSI with Holt-Winters exponential smoothing overlay.

    The smoothed RSI leads raw RSI at turning points, providing
    early momentum shift detection.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        alpha: float = 0.3,
        beta: float = 0.05,
        gamma: float = 0.1,
    ):
        self._period = rsi_period
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        # Internal state for incremental RSI
        self._prev_close: float = 0.0
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._rsi_values: list[float] = []
        self._initialized: bool = False
        self._count: int = 0

        # Holt-Winters state
        self._hw_level: float = 50.0
        self._hw_trend: float = 0.0
        self._hw_initialized: bool = False

    def compute(self, closes: np.ndarray) -> dict:
        """Compute RSI and Holt-Winters smoothed RSI from close prices.

        Args:
            closes: Array of close prices (full buffer).

        Returns:
            dict with keys: rsi, hw_rsi, momentum_shift
        """
        if len(closes) < self._period + 1:
            return {"rsi": 50.0, "hw_rsi": 50.0, "momentum_shift": "neutral"}

        # Compute full RSI series
        rsi_series = self._compute_rsi_series(closes)

        if len(rsi_series) == 0:
            return {"rsi": 50.0, "hw_rsi": 50.0, "momentum_shift": "neutral"}

        current_rsi = rsi_series[-1]

        # Apply Holt-Winters smoothing
        hw_rsi = self._apply_holt_winters(rsi_series)

        # Detect momentum shift
        momentum_shift = self._detect_shift(rsi_series, hw_rsi)

        return {
            "rsi": float(current_rsi),
            "hw_rsi": float(hw_rsi),
            "momentum_shift": momentum_shift,
        }

    def _compute_rsi_series(self, closes: np.ndarray) -> np.ndarray:
        """Compute RSI series using Wilder's smoothing."""
        deltas = np.diff(closes)
        n = len(deltas)
        if n < self._period:
            return np.array([50.0])

        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi_values = np.empty(n - self._period + 1, dtype=np.float64)

        # Seed: simple average of first `period` changes
        avg_gain = np.mean(gains[:self._period])
        avg_loss = np.mean(losses[:self._period])

        if avg_loss == 0:
            rsi_values[0] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[0] = 100.0 - 100.0 / (1.0 + rs)

        # Wilder's smoothed RSI
        for i in range(self._period, n):
            avg_gain = (avg_gain * (self._period - 1) + gains[i]) / self._period
            avg_loss = (avg_loss * (self._period - 1) + losses[i]) / self._period

            if avg_loss == 0:
                rsi_values[i - self._period + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i - self._period + 1] = 100.0 - 100.0 / (1.0 + rs)

        return rsi_values

    def _apply_holt_winters(self, rsi_series: np.ndarray) -> float:
        """Apply double exponential smoothing (Holt's linear) on RSI.

        Uses level + trend components for adaptive smoothing.
        Returns the current smoothed RSI value.
        """
        if len(rsi_series) < 2:
            return float(rsi_series[-1])

        # Initialize
        level = rsi_series[0]
        trend = rsi_series[1] - rsi_series[0]

        for val in rsi_series[1:]:
            prev_level = level
            level = self._alpha * val + (1 - self._alpha) * (prev_level + trend)
            trend = self._beta * (level - prev_level) + (1 - self._beta) * trend

        return float(level + trend)

    def _detect_shift(self, rsi_series: np.ndarray, hw_rsi: float) -> str:
        """Detect early momentum shift from smoothed vs raw RSI.

        Bullish shift: smoothed RSI crosses above raw RSI in oversold territory
        Bearish shift: smoothed RSI crosses below raw RSI in overbought territory
        """
        if len(rsi_series) < 3:
            return "neutral"

        current_rsi = rsi_series[-1]
        prev_rsi = rsi_series[-2]

        # Compute previous HW value (approximate)
        hw_prev = self._apply_holt_winters(rsi_series[:-1])

        # Cross detection
        hw_above_now = hw_rsi > current_rsi
        hw_above_prev = hw_prev > prev_rsi

        if hw_above_now and not hw_above_prev:
            # Smoothed crossed above raw → bullish shift
            if current_rsi < 45:
                return "bullish"
        elif not hw_above_now and hw_above_prev:
            # Smoothed crossed below raw → bearish shift
            if current_rsi > 55:
                return "bearish"

        return "neutral"
