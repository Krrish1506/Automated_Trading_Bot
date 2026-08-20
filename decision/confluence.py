"""
Confluence Detector.

Identifies alignment between multiple indicator modules to
confirm high-probability trade setups.
"""
from __future__ import annotations

from core.types import IndicatorState, TrendDirection, StructureBreak


class ConfluenceDetector:
    """Detects confluence between indicator modules.

    A confluence exists when multiple independent analysis methods
    agree on direction. More confluences = higher probability setup.
    """

    def __init__(self, min_confluences: int = 3):
        self._min = min_confluences

    def detect(
        self,
        state: IndicatorState,
        forecast_dir: str,
        individual_scores: dict[str, float],
    ) -> list[str]:
        """Detect active confluences.

        Returns list of confluence labels for the output JSON.
        """
        confluences: list[str] = []

        # Determine dominant direction from composite
        bullish_count = sum(1 for s in individual_scores.values() if s > 0.2)
        bearish_count = sum(1 for s in individual_scores.values() if s < -0.2)

        # 1. Trend Alignment: AlphaTrend + Alpha++ agree
        if state.trend_direction == TrendDirection.BULLISH and state.trend_confirmed:
            confluences.append("AlphaTrend Bullish Alignment")
        elif state.trend_direction == TrendDirection.BEARISH and state.trend_confirmed:
            confluences.append("AlphaTrend Bearish Alignment")

        # 2. Structural Confirmation: BOS matches trend direction
        if state.structure.bos:
            if (state.structure.bos_direction == StructureBreak.UP and
                    state.trend_direction == TrendDirection.BULLISH):
                confluences.append("Bullish Structure Confirmation")
            elif (state.structure.bos_direction == StructureBreak.DOWN and
                  state.trend_direction == TrendDirection.BEARISH):
                confluences.append("Bearish Structure Confirmation")

        # 3. Zone Alignment: price near demand/supply matching bias
        if state.nearest_demand_dist < 2.0 and individual_scores.get("alpha_trend", 0) > 0:
            confluences.append("Demand Zone Alignment")
        if state.nearest_supply_dist < 2.0 and individual_scores.get("alpha_trend", 0) < 0:
            confluences.append("Supply Zone Alignment")

        # 4. Harmonic PRZ Alignment
        if state.active_pattern is not None:
            pat = state.active_pattern
            if pat.prz_low <= state.current_price <= pat.prz_high:
                confluences.append(f"Harmonic {pat.pattern_type.title()} PRZ")

        # 5. Forecast alignment with structure
        if forecast_dir == "up" and state.structure.bos_direction == StructureBreak.UP:
            confluences.append("Forecast + Structure Alignment")
        elif forecast_dir == "down" and state.structure.bos_direction == StructureBreak.DOWN:
            confluences.append("Forecast + Structure Alignment")

        # 6. RSI Momentum confirmation
        if state.momentum_shift == "bullish" and individual_scores.get("alpha_trend", 0) > 0:
            confluences.append("RSI Momentum Confirmation")
        elif state.momentum_shift == "bearish" and individual_scores.get("alpha_trend", 0) < 0:
            confluences.append("RSI Momentum Confirmation")

        # 7. MACD Crossover alignment
        if (state.crossover_direction == "up" and
                individual_scores.get("smc_structure", 0) > 0):
            confluences.append("MACD Crossover Alignment")
        elif (state.crossover_direction == "down" and
              individual_scores.get("smc_structure", 0) < 0):
            confluences.append("MACD Crossover Alignment")

        # 8. Fibonacci confluence
        if state.fibonacci.confluence_count >= 2:
            confluences.append("Multi-Fibonacci Confluence")

        # 9. Order block alignment
        for ob in state.structure.order_blocks:
            if ob.direction == "bullish" and state.current_price >= ob.price_low and state.current_price <= ob.price_high:
                confluences.append("Bullish Order Block")
                break
            elif ob.direction == "bearish" and state.current_price >= ob.price_low and state.current_price <= ob.price_high:
                confluences.append("Bearish Order Block")
                break

        return confluences

    def has_sufficient_confluence(self, confluences: list[str]) -> bool:
        """Check if minimum confluence threshold is met."""
        return len(confluences) >= self._min
