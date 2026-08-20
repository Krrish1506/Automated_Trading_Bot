"""
Harmonic Pattern Scanner & Auto Harmonic Predictor.

Detects XABCD harmonic patterns (Gartley, Butterfly, Bat, Crab),
validates Fibonacci ratios between legs, identifies Potential Reversal
Zones (PRZ), and predicts PRZ before full pattern completion.
"""
from __future__ import annotations

import numpy as np
from core.types import HarmonicPattern


# Standard Fibonacci ratios for each harmonic pattern
DEFAULT_PATTERNS = {
    "gartley": {
        "XB": (0.618, 0.618),
        "AC": (0.382, 0.886),
        "BD": (1.272, 1.618),
        "XD": (0.786, 0.786),
    },
    "butterfly": {
        "XB": (0.786, 0.786),
        "AC": (0.382, 0.886),
        "BD": (1.618, 2.618),
        "XD": (1.272, 1.618),
    },
    "bat": {
        "XB": (0.382, 0.5),
        "AC": (0.382, 0.886),
        "BD": (1.618, 2.618),
        "XD": (0.886, 0.886),
    },
    "crab": {
        "XB": (0.382, 0.618),
        "AC": (0.382, 0.886),
        "BD": (2.618, 3.618),
        "XD": (1.618, 1.618),
    },
}


class HarmonicScanner:
    """XABCD harmonic pattern detector with auto-prediction.

    Scans for completed and partial harmonic patterns, validates
    Fibonacci ratios within configured tolerance, and projects
    potential reversal zones.
    """

    def __init__(
        self,
        tolerance: float = 0.05,
        min_leg_bars: int = 5,
        patterns: dict | None = None,
    ):
        self._tolerance = tolerance
        self._min_leg = min_leg_bars
        self._patterns = patterns or DEFAULT_PATTERNS

    def compute(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> dict:
        """Scan for harmonic patterns.

        Returns:
            dict with: active_pattern (HarmonicPattern or None),
                       pattern_type, prz_high, prz_low,
                       completion_pct, predicted_prz
        """
        n = len(closes)
        if n < self._min_leg * 4 + 1:
            return {
                "active_pattern": None,
                "pattern_type": "none",
                "prz_high": 0.0,
                "prz_low": 0.0,
                "completion_pct": 0.0,
                "predicted_prz": 0.0,
            }

        # Find swing points for XABCD detection
        pivots = self._find_pivots(highs, lows, n)

        if len(pivots) < 4:
            # Try auto-prediction with partial pattern
            predicted = self._auto_predict(pivots, closes[-1])
            return {
                "active_pattern": predicted,
                "pattern_type": predicted.pattern_type if predicted else "none",
                "prz_high": predicted.prz_high if predicted else 0.0,
                "prz_low": predicted.prz_low if predicted else 0.0,
                "completion_pct": predicted.completion_pct if predicted else 0.0,
                "predicted_prz": predicted.predicted_prz if predicted else 0.0,
            }

        # Try to match complete XABCD patterns
        best_pattern = None
        best_score = 0.0

        # Use the last 5 pivots as potential XABCD
        for start_idx in range(max(0, len(pivots) - 8), len(pivots) - 4):
            candidates = pivots[start_idx:start_idx + 5]
            if len(candidates) < 5:
                continue

            x, a, b, c, d = [p[1] for p in candidates]

            for pattern_name, ratios in self._patterns.items():
                score = self._validate_pattern(x, a, b, c, d, ratios)
                if score > best_score:
                    best_score = score

                    # Determine direction from X-A leg
                    direction = "bullish" if a > x else "bearish"

                    # PRZ computation
                    xa_range = abs(a - x)
                    xd_ratio = ratios["XD"]
                    xd_mid = (xd_ratio[0] + xd_ratio[1]) / 2

                    if direction == "bullish":
                        prz_center = x + xa_range * xd_mid
                        prz_high = prz_center + xa_range * 0.05
                        prz_low = prz_center - xa_range * 0.05
                    else:
                        prz_center = x - xa_range * xd_mid
                        prz_high = prz_center + xa_range * 0.05
                        prz_low = prz_center - xa_range * 0.05

                    best_pattern = HarmonicPattern(
                        pattern_type=pattern_name,
                        prz_high=float(prz_high),
                        prz_low=float(prz_low),
                        completion_pct=min(best_score * 100, 100.0),
                        direction=direction,
                    )

        # If no complete pattern, try auto-prediction
        if best_pattern is None:
            best_pattern = self._auto_predict(pivots, closes[-1])

        return {
            "active_pattern": best_pattern,
            "pattern_type": best_pattern.pattern_type if best_pattern else "none",
            "prz_high": best_pattern.prz_high if best_pattern else 0.0,
            "prz_low": best_pattern.prz_low if best_pattern else 0.0,
            "completion_pct": best_pattern.completion_pct if best_pattern else 0.0,
            "predicted_prz": best_pattern.predicted_prz if best_pattern else 0.0,
        }

    def _find_pivots(
        self, highs: np.ndarray, lows: np.ndarray, n: int,
    ) -> list[tuple[int, float, str]]:
        """Find alternating swing highs and lows.

        Returns list of (index, price, type) tuples.
        """
        w = max(3, self._min_leg // 2)
        pivots: list[tuple[int, float, str]] = []

        for i in range(w, n - w):
            is_high = all(highs[i] >= highs[j] for j in range(i - w, i + w + 1) if j != i)
            is_low = all(lows[i] <= lows[j] for j in range(i - w, i + w + 1) if j != i)

            if is_high:
                if not pivots or pivots[-1][2] != "high":
                    pivots.append((i, float(highs[i]), "high"))
                elif highs[i] > pivots[-1][1]:
                    pivots[-1] = (i, float(highs[i]), "high")

            elif is_low:
                if not pivots or pivots[-1][2] != "low":
                    pivots.append((i, float(lows[i]), "low"))
                elif lows[i] < pivots[-1][1]:
                    pivots[-1] = (i, float(lows[i]), "low")

        return pivots

    def _validate_pattern(
        self,
        x: float, a: float, b: float, c: float, d: float,
        ratios: dict,
    ) -> float:
        """Validate XABCD pattern against expected Fibonacci ratios.

        Returns a score 0-1 indicating how well the pattern matches.
        """
        xa = abs(a - x)
        ab = abs(b - a)
        bc = abs(c - b)
        cd = abs(d - c)
        xd = abs(d - x)

        if xa == 0 or ab == 0:
            return 0.0

        # XB ratio
        xb_actual = ab / xa
        xb_min, xb_max = ratios["XB"]
        xb_score = self._ratio_score(xb_actual, xb_min, xb_max)

        # AC ratio
        if ab > 0:
            ac_actual = bc / ab
        else:
            return 0.0
        ac_min, ac_max = ratios["AC"]
        ac_score = self._ratio_score(ac_actual, ac_min, ac_max)

        # BD ratio
        if bc > 0:
            bd_actual = cd / bc
        else:
            return 0.0
        bd_min, bd_max = ratios["BD"]
        bd_score = self._ratio_score(bd_actual, bd_min, bd_max)

        # XD ratio
        xd_actual = xd / xa
        xd_min, xd_max = ratios["XD"]
        xd_score = self._ratio_score(xd_actual, xd_min, xd_max)

        # All ratios must score above threshold
        scores = [xb_score, ac_score, bd_score, xd_score]
        if any(s == 0 for s in scores):
            return 0.0

        return sum(scores) / len(scores)

    def _ratio_score(self, actual: float, target_min: float, target_max: float) -> float:
        """Score a ratio against a target range with tolerance."""
        tol = self._tolerance
        low = target_min - tol
        high = target_max + tol

        if low <= actual <= high:
            # Perfect range → score based on distance to center
            center = (target_min + target_max) / 2
            max_dist = (high - low) / 2
            if max_dist == 0:
                return 1.0
            dist = abs(actual - center) / max_dist
            return max(0.0, 1.0 - dist * 0.5)
        return 0.0

    def _auto_predict(
        self,
        pivots: list[tuple[int, float, str]],
        current_price: float,
    ) -> HarmonicPattern | None:
        """Predict PRZ before full pattern completion.

        Uses partial XABC structure to project where D would complete
        each pattern type, returning the best match.
        """
        if len(pivots) < 3:
            return None

        # Use last 3-4 pivots as XAB[C]
        x_price = pivots[-3][1]
        a_price = pivots[-2][1]
        b_price = pivots[-1][1]

        xa = abs(a_price - x_price)
        ab = abs(b_price - a_price)

        if xa == 0:
            return None

        xb_ratio = ab / xa

        # Find best matching partial pattern
        best_match = None
        best_diff = float('inf')

        for pattern_name, ratios in self._patterns.items():
            xb_min, xb_max = ratios["XB"]
            if xb_min - self._tolerance <= xb_ratio <= xb_max + self._tolerance:
                diff = abs(xb_ratio - (xb_min + xb_max) / 2)
                if diff < best_diff:
                    best_diff = diff
                    best_match = pattern_name

        if best_match is None:
            return None

        # Project D point
        xd_ratios = self._patterns[best_match]["XD"]
        xd_mid = (xd_ratios[0] + xd_ratios[1]) / 2

        # Determine direction
        if a_price > x_price:
            # Bullish XA → D should be near X + xd * XA
            predicted_d = x_price + xa * xd_mid
            direction = "bullish"
        else:
            predicted_d = x_price - xa * xd_mid
            direction = "bearish"

        prz_range = xa * 0.05
        completion = min(abs(current_price - x_price) / (xa * xd_mid) * 100, 95.0) if xa * xd_mid > 0 else 0.0

        return HarmonicPattern(
            pattern_type=best_match,
            prz_high=float(predicted_d + prz_range),
            prz_low=float(predicted_d - prz_range),
            completion_pct=float(completion),
            predicted_prz=float(predicted_d),
            direction=direction,
        )
