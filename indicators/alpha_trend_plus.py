"""
AlphaTrend++ Module — Strict Volatility & Trend Confirmation Filter.

Enhanced version of AlphaTrend with additional ADX confirmation,
volume surge detection, and stricter entry conditions for
high-probability trend states only.
"""
from __future__ import annotations

import numpy as np
from core.types import TrendDirection


class AlphaTrendPlus:
    """Stricter trend confirmation with volatility and volume filters.

    Extends AlphaTrend with:
    - ADX confirmation (directional strength)
    - Volume surge detection
    - Only confirms trend when multiple conditions align
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        volume_surge_mult: float = 1.5,
    ):
        self._atr_period = atr_period
        self._multiplier = atr_multiplier
        self._adx_period = adx_period
        self._adx_threshold = adx_threshold
        self._vol_surge = volume_surge_mult

    def compute(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> dict:
        """Compute AlphaTrend++ with strict confirmation.

        Returns:
            dict with: alpha_plus_trend, trend_confirmed, trend_strength
        """
        n = len(closes)
        if n < self._adx_period * 2 + 1:
            return {
                "alpha_plus_trend": float(closes[-1]) if n > 0 else 0.0,
                "trend_confirmed": False,
                "trend_strength": 0.0,
            }

        # ADX computation
        adx, plus_di, minus_di = self._compute_adx(highs, lows, closes)

        # ATR
        atr = self._compute_atr(highs, lows, closes)

        # Enhanced AlphaTrend line with strict multiplier
        at_line = self._compute_strict_alpha(closes, atr)
        current_at = at_line[-1]

        # Volume surge check
        vol_sma = self._sma(volumes, 20)
        vol_surge = volumes[-1] > vol_sma[-1] * self._vol_surge if len(vol_sma) > 0 else False

        # Determine strict trend confirmation
        current_adx = adx[-1] if len(adx) > 0 else 0.0
        current_plus = plus_di[-1] if len(plus_di) > 0 else 0.0
        current_minus = minus_di[-1] if len(minus_di) > 0 else 0.0

        trend_confirmed = False
        trend_strength = 0.0

        adx_strong = current_adx > self._adx_threshold
        price_above_at = closes[-1] > current_at
        price_below_at = closes[-1] < current_at
        di_bullish = current_plus > current_minus
        di_bearish = current_minus > current_plus

        if price_above_at and adx_strong and di_bullish:
            trend_confirmed = True
            trend_strength = min(current_adx / 50.0, 1.0)
            if vol_surge:
                trend_strength = min(trend_strength * 1.2, 1.0)
        elif price_below_at and adx_strong and di_bearish:
            trend_confirmed = True
            trend_strength = min(current_adx / 50.0, 1.0)
            if vol_surge:
                trend_strength = min(trend_strength * 1.2, 1.0)

        return {
            "alpha_plus_trend": float(current_at),
            "trend_confirmed": bool(trend_confirmed),
            "trend_strength": float(trend_strength),
        }

    def _compute_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """ATR using Wilder's smoothing."""
        n = len(highs)
        tr = np.empty(n, dtype=np.float64)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        atr = np.empty(n, dtype=np.float64)
        atr[:self._atr_period] = np.nan
        atr[self._atr_period - 1] = np.mean(tr[:self._atr_period])
        for i in range(self._atr_period, n):
            atr[i] = (atr[i - 1] * (self._atr_period - 1) + tr[i]) / self._atr_period
        return atr

    def _compute_adx(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ADX, +DI, -DI using Wilder's method."""
        n = len(highs)
        p = self._adx_period

        # Directional movement
        plus_dm = np.zeros(n, dtype=np.float64)
        minus_dm = np.zeros(n, dtype=np.float64)
        tr = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]

            plus_dm[i] = up if up > down and up > 0 else 0.0
            minus_dm[i] = down if down > up and down > 0 else 0.0

            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        # Smooth with Wilder's
        sm_tr = self._wilder_smooth(tr, p)
        sm_plus = self._wilder_smooth(plus_dm, p)
        sm_minus = self._wilder_smooth(minus_dm, p)

        # DI
        plus_di = np.where(sm_tr > 0, 100.0 * sm_plus / sm_tr, 0.0)
        minus_di = np.where(sm_tr > 0, 100.0 * sm_minus / sm_tr, 0.0)

        # DX
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

        # ADX = smoothed DX
        adx = self._wilder_smooth(dx, p)

        return adx, plus_di, minus_di

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's exponential smoothing."""
        n = len(data)
        out = np.empty(n, dtype=np.float64)
        out[:period] = np.nan
        out[period - 1] = np.mean(data[1:period + 1]) if period + 1 <= n else np.mean(data[1:])
        for i in range(period, n):
            prev = out[i - 1] if not np.isnan(out[i - 1]) else data[i]
            out[i] = (prev * (period - 1) + data[i]) / period
        return out

    def _compute_strict_alpha(self, closes: np.ndarray, atr: np.ndarray) -> np.ndarray:
        """Strict AlphaTrend line with higher ATR multiplier."""
        n = len(closes)
        at = np.empty(n, dtype=np.float64)
        at[0] = closes[0]
        for i in range(1, n):
            a = atr[i] if not np.isnan(atr[i]) else 0.0
            offset = a * self._multiplier
            if closes[i] > at[i - 1]:
                at[i] = max(at[i - 1], closes[i] - offset)
            else:
                at[i] = min(at[i - 1], closes[i] + offset)
        return at

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        if len(data) < period:
            return np.array([np.mean(data)])
        return np.convolve(data, np.ones(period) / period, mode='valid')
