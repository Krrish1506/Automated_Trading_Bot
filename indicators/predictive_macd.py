"""
Predictive MACD Targets Module.

Standard MACD computation plus forward-looking crossover estimation:
calculates the price level required for a future MACD crossover,
providing predictive price targets.
"""
from __future__ import annotations

import numpy as np


class PredictiveMACD:
    """MACD with predictive crossover target estimation.

    Beyond standard MACD(12,26,9), this module estimates the price
    level that would cause a MACD line-signal crossover, giving
    a projected target for trend confirmation or reversal.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self._fast = fast
        self._slow = slow
        self._signal = signal

    def compute(self, closes: np.ndarray) -> dict:
        """Compute MACD and projected crossover target.

        Args:
            closes: Array of close prices.

        Returns:
            dict with: macd, signal_line, histogram, crossover_target,
                       crossover_direction
        """
        n = len(closes)
        if n < self._slow + self._signal:
            return {
                "macd": 0.0,
                "signal_line": 0.0,
                "histogram": 0.0,
                "crossover_target": 0.0,
                "crossover_direction": "none",
            }

        # EMA computation
        ema_fast = self._ema(closes, self._fast)
        ema_slow = self._ema(closes, self._slow)

        # MACD line = fast EMA - slow EMA
        macd_line = ema_fast - ema_slow

        # Signal line = EMA of MACD line
        signal_line = self._ema(macd_line, self._signal)

        # Histogram
        histogram = macd_line - signal_line

        current_macd = float(macd_line[-1])
        current_signal = float(signal_line[-1])
        current_histogram = float(histogram[-1])

        # Predict crossover target price
        crossover_target, crossover_dir = self._predict_crossover(
            closes, ema_fast, ema_slow, macd_line, signal_line
        )

        return {
            "macd": current_macd,
            "signal_line": current_signal,
            "histogram": current_histogram,
            "crossover_target": crossover_target,
            "crossover_direction": crossover_dir,
        }

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA using the standard multiplier method."""
        alpha = 2.0 / (period + 1)
        ema = np.empty_like(data, dtype=np.float64)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _predict_crossover(
        self,
        closes: np.ndarray,
        ema_fast: np.ndarray,
        ema_slow: np.ndarray,
        macd_line: np.ndarray,
        signal_line: np.ndarray,
    ) -> tuple[float, str]:
        """Estimate the price level that would cause a MACD crossover.

        The crossover occurs when MACD line = signal line, i.e.,
        histogram = 0. We estimate the price change needed to
        achieve this using the EMA update equations.

        Key insight: A new close price C would change the fast EMA by:
            new_fast = α_f * C + (1-α_f) * old_fast
        And slow EMA by:
            new_slow = α_s * C + (1-α_s) * old_slow

        Setting new_macd = new_signal and solving for C gives the
        projected crossover target.
        """
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = current_macd - current_signal

        if abs(current_histogram) < 1e-10:
            return float(closes[-1]), "at_crossover"

        alpha_f = 2.0 / (self._fast + 1)
        alpha_s = 2.0 / (self._slow + 1)
        alpha_sig = 2.0 / (self._signal + 1)

        # Current EMAs
        ef = ema_fast[-1]
        es = ema_slow[-1]

        # For MACD-signal crossover, we need:
        # new_macd = new_signal
        # (α_f * C + (1-α_f)*ef) - (α_s * C + (1-α_s)*es) =
        #   α_sig * new_macd + (1-α_sig) * current_signal
        #
        # Simplify: Let Δα = α_f - α_s
        # new_macd = Δα * C + (1-α_f)*ef - (1-α_s)*es
        # Setting new_macd = α_sig * new_macd + (1-α_sig) * current_signal
        # → (1-α_sig) * new_macd = (1-α_sig) * current_signal
        # → new_macd = current_signal
        #
        # So: Δα * C + (1-α_f)*ef - (1-α_s)*es = current_signal
        delta_alpha = alpha_f - alpha_s
        if abs(delta_alpha) < 1e-12:
            return float(closes[-1]), "none"

        rhs = current_signal - (1 - alpha_f) * ef + (1 - alpha_s) * es
        target_price = rhs / delta_alpha

        # Determine direction
        if target_price > closes[-1]:
            direction = "up"
        elif target_price < closes[-1]:
            direction = "down"
        else:
            direction = "none"

        return float(target_price), direction
