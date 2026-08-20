"""
AI Smart Money Concepts (SMC) Engine.

Full structural analysis module detecting:
- BOS (Break of Structure) — price closing beyond last swing
- CHoCH (Change of Character) — reversal break opposite to prior trend
- Liquidity sweeps — wick beyond structure, close back inside
- Order blocks — last opposing candle before impulsive move
- Internal vs external structure tracking
"""
from __future__ import annotations

import numpy as np
from core.types import (
    StructureState, OrderBlock, StructureBreak, TrendDirection,
)


class SMCEngine:
    """Smart Money Concepts structural analysis engine.

    Tracks market structure via swing points and detects institutional
    footprints: BOS, CHoCH, liquidity sweeps, and order blocks.
    """

    def __init__(self, swing_window: int = 5, order_block_lookback: int = 10):
        self._window = swing_window
        self._ob_lookback = order_block_lookback
        self._prev_bias: TrendDirection = TrendDirection.NEUTRAL
        self._prev_bos_dir: StructureBreak = StructureBreak.NONE

    def compute(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> StructureState:
        """Run full SMC analysis on buffered candle data.

        Returns:
            StructureState with all SMC fields populated.
        """
        n = len(closes)
        state = StructureState()

        if n < self._window * 2 + 2:
            return state

        # Detect swing points
        swing_highs, swing_lows = self._detect_swings(highs, lows, n)

        if not swing_highs or not swing_lows:
            return state

        last_sh_idx, last_sh = swing_highs[-1]
        last_sl_idx, last_sl = swing_lows[-1]

        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        # ── BOS Detection ─────────────────────────────────────────
        bos = False
        bos_direction = StructureBreak.NONE

        if current_close > last_sh:
            bos = True
            bos_direction = StructureBreak.UP
        elif current_close < last_sl:
            bos = True
            bos_direction = StructureBreak.DOWN

        # ── CHoCH Detection ───────────────────────────────────────
        choch = False
        choch_direction = StructureBreak.NONE

        if bos and self._prev_bos_dir != StructureBreak.NONE:
            if bos_direction != self._prev_bos_dir:
                choch = True
                choch_direction = bos_direction

        # ── Liquidity Sweep ───────────────────────────────────────
        liquidity_sweep = False

        # Sweep above: wick exceeds swing high but close stays below
        if current_high > last_sh and current_close <= last_sh:
            liquidity_sweep = True

        # Sweep below: wick below swing low but close stays above
        if current_low < last_sl and current_close >= last_sl:
            liquidity_sweep = True

        # ── Order Block Detection ─────────────────────────────────
        order_blocks = self._detect_order_blocks(opens, highs, lows, closes, n)

        # ── Structure Bias ────────────────────────────────────────
        if bos_direction == StructureBreak.UP:
            bias = TrendDirection.BULLISH
        elif bos_direction == StructureBreak.DOWN:
            bias = TrendDirection.BEARISH
        elif choch:
            bias = (
                TrendDirection.BULLISH
                if choch_direction == StructureBreak.UP
                else TrendDirection.BEARISH
            )
        else:
            bias = self._prev_bias

        # Update state for next call
        if bos:
            self._prev_bos_dir = bos_direction
        self._prev_bias = bias

        state.bos = bos
        state.bos_direction = bos_direction
        state.choch = choch
        state.choch_direction = choch_direction
        state.liquidity_sweep = liquidity_sweep
        state.order_blocks = order_blocks
        state.structure_bias = bias

        return state

    def _detect_swings(
        self, highs: np.ndarray, lows: np.ndarray, n: int,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Detect swing highs and lows via local extrema."""
        w = self._window
        sh: list[tuple[int, float]] = []
        sl: list[tuple[int, float]] = []

        for i in range(w, n - w):
            is_high = True
            is_low = True

            for j in range(i - w, i + w + 1):
                if j == i:
                    continue
                if highs[j] > highs[i]:
                    is_high = False
                if lows[j] < lows[i]:
                    is_low = False
                if not is_high and not is_low:
                    break

            if is_high:
                sh.append((i, float(highs[i])))
            if is_low:
                sl.append((i, float(lows[i])))

        return sh, sl

    def _detect_order_blocks(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        n: int,
    ) -> list[OrderBlock]:
        """Detect order blocks: last opposing candle before impulsive move.

        Bullish OB: last bearish candle before a strong bullish move
        Bearish OB: last bullish candle before a strong bearish move
        """
        obs: list[OrderBlock] = []
        lookback = min(self._ob_lookback, n - 2)

        for i in range(n - lookback, n - 1):
            if i < 1:
                continue

            body = closes[i] - opens[i]
            body_next = closes[i + 1] - opens[i + 1]
            range_next = highs[i + 1] - lows[i + 1]

            if range_next == 0:
                continue

            # Strong move: next candle body > 60% of its range
            if abs(body_next) / range_next < 0.6:
                continue

            # Bullish OB: current candle bearish, next candle strongly bullish
            if body < 0 and body_next > 0 and abs(body_next) > abs(body) * 1.5:
                obs.append(OrderBlock(
                    price_high=float(highs[i]),
                    price_low=float(lows[i]),
                    direction="bullish",
                    strength=float(abs(body_next) / range_next),
                    age_bars=n - 1 - i,
                ))

            # Bearish OB: current candle bullish, next candle strongly bearish
            elif body > 0 and body_next < 0 and abs(body_next) > abs(body) * 1.5:
                obs.append(OrderBlock(
                    price_high=float(highs[i]),
                    price_low=float(lows[i]),
                    direction="bearish",
                    strength=float(abs(body_next) / range_next),
                    age_bars=n - 1 - i,
                ))

        return obs
