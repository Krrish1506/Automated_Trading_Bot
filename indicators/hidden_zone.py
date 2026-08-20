"""
Hidden Zone Detector — Supply/Demand via Volatility Clustering.

Detects institutional supply/demand zones using:
- Volatility clustering analysis (GARCH-like fingerprint)
- Fair value gap (imbalance) detection
- Zone strength scoring by departure move magnitude
"""
from __future__ import annotations

import numpy as np
from core.types import ZoneInfo


class HiddenZoneDetector:
    """Supply/demand zone detector using volatility and imbalance analysis.

    Identifies zones where institutional activity left footprints
    via unusual volatility patterns and price imbalances (fair value gaps).
    Scores each zone by the strength of the departure move.
    """

    def __init__(
        self,
        vol_cluster_window: int = 20,
        imbalance_threshold: float = 1.5,
        zone_max_age: int = 100,
    ):
        self._vol_window = vol_cluster_window
        self._imb_threshold = imbalance_threshold
        self._max_age = zone_max_age

    def compute(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        atr: float,
    ) -> dict:
        """Detect hidden supply and demand zones.

        Returns:
            dict with: demand_zones, supply_zones, nearest_demand_dist,
                       nearest_supply_dist, zone_strength
        """
        n = len(closes)
        if n < self._vol_window + 5:
            return {
                "demand_zones": [],
                "supply_zones": [],
                "nearest_demand_dist": float('inf'),
                "nearest_supply_dist": float('inf'),
                "zone_strength": 0.0,
            }

        demand_zones: list[ZoneInfo] = []
        supply_zones: list[ZoneInfo] = []

        # Method 1: Fair Value Gap (FVG) / Imbalance detection
        fvg_zones = self._detect_fvg(highs, lows, closes, n, atr)
        for z in fvg_zones:
            if z.zone_type == "demand":
                demand_zones.append(z)
            else:
                supply_zones.append(z)

        # Method 2: Volatility clustering zones
        vol_zones = self._detect_volatility_clusters(
            opens, highs, lows, closes, volumes, n, atr
        )
        for z in vol_zones:
            if z.zone_type == "demand":
                demand_zones.append(z)
            else:
                supply_zones.append(z)

        # Merge overlapping zones
        demand_zones = self._merge_zones(demand_zones)
        supply_zones = self._merge_zones(supply_zones)

        # Filter by max age
        demand_zones = [z for z in demand_zones if z.age_bars <= self._max_age]
        supply_zones = [z for z in supply_zones if z.age_bars <= self._max_age]

        # Compute distances
        current = closes[-1]
        nearest_demand = float('inf')
        nearest_supply = float('inf')
        best_strength = 0.0

        for z in demand_zones:
            dist = current - z.price_high
            if dist >= 0:
                nearest_demand = min(nearest_demand, dist)
            elif abs(dist) < nearest_demand:
                nearest_demand = abs(dist)
            best_strength = max(best_strength, z.strength)

        for z in supply_zones:
            dist = z.price_low - current
            if dist >= 0:
                nearest_supply = min(nearest_supply, dist)
            elif abs(dist) < nearest_supply:
                nearest_supply = abs(dist)
            best_strength = max(best_strength, z.strength)

        # Normalize distances by ATR
        if atr > 0:
            nearest_demand = nearest_demand / atr if nearest_demand != float('inf') else float('inf')
            nearest_supply = nearest_supply / atr if nearest_supply != float('inf') else float('inf')

        return {
            "demand_zones": demand_zones,
            "supply_zones": supply_zones,
            "nearest_demand_dist": float(nearest_demand),
            "nearest_supply_dist": float(nearest_supply),
            "zone_strength": float(best_strength),
        }

    def _detect_fvg(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        n: int,
        atr: float,
    ) -> list[ZoneInfo]:
        """Detect Fair Value Gaps (price imbalances).

        Bullish FVG: gap between candle[i-1].high and candle[i+1].low
        Bearish FVG: gap between candle[i+1].high and candle[i-1].low
        """
        zones: list[ZoneInfo] = []
        if atr <= 0:
            return zones

        for i in range(1, n - 1):
            # Bullish FVG: strong up move creates gap
            if lows[i + 1] > highs[i - 1]:
                gap_size = lows[i + 1] - highs[i - 1]
                if gap_size > atr * 0.3:
                    body = abs(closes[i] - closes[i - 1])
                    strength = min(body / atr, 2.0)

                    # Count touches (how many times price revisited)
                    touches = 0
                    for j in range(i + 2, n):
                        if lows[j] <= lows[i + 1] and lows[j] >= highs[i - 1]:
                            touches += 1

                    zones.append(ZoneInfo(
                        price_high=float(lows[i + 1]),
                        price_low=float(highs[i - 1]),
                        strength=float(strength),
                        touch_count=touches,
                        zone_type="demand",
                        age_bars=n - 1 - i,
                    ))

            # Bearish FVG: strong down move creates gap
            if highs[i + 1] < lows[i - 1]:
                gap_size = lows[i - 1] - highs[i + 1]
                if gap_size > atr * 0.3:
                    body = abs(closes[i] - closes[i - 1])
                    strength = min(body / atr, 2.0)

                    touches = 0
                    for j in range(i + 2, n):
                        if highs[j] >= highs[i + 1] and highs[j] <= lows[i - 1]:
                            touches += 1

                    zones.append(ZoneInfo(
                        price_high=float(lows[i - 1]),
                        price_low=float(highs[i + 1]),
                        strength=float(strength),
                        touch_count=touches,
                        zone_type="supply",
                        age_bars=n - 1 - i,
                    ))

        return zones

    def _detect_volatility_clusters(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        n: int,
        atr: float,
    ) -> list[ZoneInfo]:
        """Detect zones via volatility clustering.

        Areas where volatility contracted (base formation) followed
        by explosive moves indicate institutional accumulation/distribution.
        """
        zones: list[ZoneInfo] = []
        w = self._vol_window

        if n < w + 5 or atr <= 0:
            return zones

        ranges = highs - lows

        for i in range(w, n - 3):
            # Look for contraction: current window avg range < 0.5 * ATR
            window_avg_range = np.mean(ranges[i - w:i])

            if window_avg_range < atr * 0.5:
                # Check for explosive move after contraction
                departure = abs(closes[i + 2] - closes[i])
                if departure > atr * self._imb_threshold:
                    zone_high = float(np.max(highs[i - w:i]))
                    zone_low = float(np.min(lows[i - w:i]))
                    strength = min(departure / atr, 3.0)

                    # Volume confirmation
                    vol_avg = np.mean(volumes[i - w:i]) if np.mean(volumes[i - w:i]) > 0 else 1.0
                    vol_ratio = volumes[i] / vol_avg

                    if closes[i + 2] > closes[i]:
                        zone_type = "demand"
                    else:
                        zone_type = "supply"

                    zones.append(ZoneInfo(
                        price_high=zone_high,
                        price_low=zone_low,
                        strength=float(strength * min(vol_ratio, 2.0)),
                        touch_count=0,
                        zone_type=zone_type,
                        age_bars=n - 1 - i,
                    ))

        return zones

    def _merge_zones(self, zones: list[ZoneInfo]) -> list[ZoneInfo]:
        """Merge overlapping zones of the same type."""
        if len(zones) <= 1:
            return zones

        zones.sort(key=lambda z: z.price_low)
        merged: list[ZoneInfo] = [zones[0]]

        for z in zones[1:]:
            last = merged[-1]
            if z.price_low <= last.price_high:
                # Overlapping — merge
                merged[-1] = ZoneInfo(
                    price_high=max(last.price_high, z.price_high),
                    price_low=min(last.price_low, z.price_low),
                    strength=max(last.strength, z.strength),
                    touch_count=last.touch_count + z.touch_count,
                    zone_type=last.zone_type,
                    age_bars=min(last.age_bars, z.age_bars),
                )
            else:
                merged.append(z)

        return merged
