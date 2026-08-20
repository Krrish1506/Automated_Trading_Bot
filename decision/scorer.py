"""
Weighted Module Scorer.

Assigns normalized scores [-1, +1] to each indicator module output
and produces a composite weighted score for the decision engine.
"""
from __future__ import annotations

from core.types import IndicatorState, TrendDirection, StructureBreak


class Scorer:
    """Weighted scoring of all indicator module outputs.

    Each module's output is converted to a normalized score in [-1, +1]:
      +1 = strongly bullish
      -1 = strongly bearish
       0 = neutral

    The composite score is the weighted sum of all module scores.
    """

    def __init__(self, weights: dict | None = None):
        self._weights = weights or {
            "alpha_trend": 0.15,
            "alpha_trend_plus": 0.10,
            "smc_structure": 0.20,
            "forecast": 0.20,
            "harmonic_prz": 0.10,
            "fibonacci": 0.05,
            "hidden_zone": 0.10,
            "holt_winters_rsi": 0.05,
            "predictive_macd": 0.05,
        }

    def score(
        self,
        state: IndicatorState,
        forecast_prob: float,
        forecast_dir: str,
    ) -> tuple[float, dict[str, float]]:
        """Compute weighted composite score.

        Args:
            state: Complete indicator state.
            forecast_prob: AI forecast probability (0-1).
            forecast_dir: 'up', 'down', or 'neutral'.

        Returns:
            (composite_score, individual_scores) where composite is
            in [-1, +1] and individual is a dict of module scores.
        """
        scores: dict[str, float] = {}

        # 1. AlphaTrend
        if state.trend_direction == TrendDirection.BULLISH:
            scores["alpha_trend"] = 1.0
        elif state.trend_direction == TrendDirection.BEARISH:
            scores["alpha_trend"] = -1.0
        else:
            scores["alpha_trend"] = 0.0

        # Penalize sideways
        if state.is_sideways:
            scores["alpha_trend"] *= 0.3

        # 2. AlphaTrend++
        if state.trend_confirmed:
            trend_score = state.trend_strength
            if state.trend_direction == TrendDirection.BEARISH:
                trend_score = -trend_score
            scores["alpha_trend_plus"] = trend_score
        else:
            scores["alpha_trend_plus"] = 0.0

        # 3. SMC Structure
        smc_score = 0.0
        if state.structure.bos:
            if state.structure.bos_direction == StructureBreak.UP:
                smc_score += 0.6
            elif state.structure.bos_direction == StructureBreak.DOWN:
                smc_score -= 0.6

        if state.structure.choch:
            if state.structure.choch_direction == StructureBreak.UP:
                smc_score += 0.4
            elif state.structure.choch_direction == StructureBreak.DOWN:
                smc_score -= 0.4

        # Liquidity sweep is a reversal signal
        if state.structure.liquidity_sweep:
            smc_score *= 0.5  # Reduce conviction during sweeps

        scores["smc_structure"] = max(-1.0, min(1.0, smc_score))

        # 4. Forecast
        # Map probability to [-1, +1]: 0.5 → 0, 1.0 → +1, 0.0 → -1
        scores["forecast"] = (forecast_prob - 0.5) * 2.0

        # 5. Harmonic PRZ
        if state.active_pattern is not None:
            pat = state.active_pattern
            current = state.current_price

            # Check if price is near PRZ
            if pat.prz_low <= current <= pat.prz_high:
                # At PRZ → reversal expected
                prz_score = 0.8
                if pat.direction == "bearish":
                    prz_score = -prz_score
            else:
                # Not at PRZ → use completion %
                prz_score = (pat.completion_pct / 100.0) * 0.3
                if pat.direction == "bearish":
                    prz_score = -prz_score

            scores["harmonic_prz"] = prz_score
        else:
            scores["harmonic_prz"] = 0.0

        # 6. Fibonacci
        fib_score = 0.0
        if state.fibonacci.confluence_count > 0:
            fib_score = min(state.fibonacci.confluence_count / 3.0, 1.0)
            # Direction from retracement proximity
            if state.fibonacci.nearest_fib_dist < 1.0:
                if scores["alpha_trend"] > 0:
                    fib_score = abs(fib_score)
                else:
                    fib_score = -abs(fib_score)

        scores["fibonacci"] = fib_score

        # 7. Hidden Zone
        zone_score = 0.0
        # Near demand zone → bullish bias
        if state.nearest_demand_dist < 2.0:
            zone_score += (2.0 - state.nearest_demand_dist) / 2.0

        # Near supply zone → bearish bias
        if state.nearest_supply_dist < 2.0:
            zone_score -= (2.0 - state.nearest_supply_dist) / 2.0

        scores["hidden_zone"] = max(-1.0, min(1.0, zone_score))

        # 8. Holt-Winters RSI
        rsi_score = 0.0
        if state.momentum_shift == "bullish":
            rsi_score = 0.7
        elif state.momentum_shift == "bearish":
            rsi_score = -0.7

        # Oversold/overbought extremes
        if state.rsi < 30:
            rsi_score += 0.3
        elif state.rsi > 70:
            rsi_score -= 0.3

        scores["holt_winters_rsi"] = max(-1.0, min(1.0, rsi_score))

        # 9. Predictive MACD
        macd_score = 0.0
        if state.crossover_direction == "up":
            macd_score = 0.7
        elif state.crossover_direction == "down":
            macd_score = -0.7

        # Histogram momentum
        if state.macd_histogram > 0:
            macd_score += 0.3
        elif state.macd_histogram < 0:
            macd_score -= 0.3

        scores["predictive_macd"] = max(-1.0, min(1.0, macd_score))

        # Composite weighted score
        composite = 0.0
        for module, weight in self._weights.items():
            composite += scores.get(module, 0.0) * weight

        composite = max(-1.0, min(1.0, composite))

        return composite, scores
