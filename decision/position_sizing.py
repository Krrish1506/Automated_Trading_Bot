"""
Position Sizing — Stop Loss, Take Profit, and Risk-Reward.

Computes precise SL/TP levels using:
- Structure-based SL (below/above swing points)
- ATR buffer and spread adjustment
- Fibonacci extension TP (primary)
- Harmonic PRZ TP (secondary)
- Dynamic RR enforcement
"""
from __future__ import annotations

from core.types import IndicatorState, SignalType


class PositionSizer:
    """Stop loss, take profit, and risk-reward computation.

    Produces spread-adjusted SL/TP levels that respect minimum
    distance and risk-reward requirements.
    """

    def __init__(
        self,
        sl_atr_multiplier: float = 1.5,
        sl_min_pips: float = 5.0,
        sl_spread_buffer_pips: float = 1.0,
        min_risk_reward: float = 1.5,
    ):
        self._sl_atr_mult = sl_atr_multiplier
        self._sl_min_raw = sl_min_pips
        self._sl_spread_raw = sl_spread_buffer_pips
        self._min_rr = min_risk_reward

    def compute(
        self,
        signal: SignalType,
        state: IndicatorState,
        entry_price: float,
    ) -> dict:
        """Compute SL, TP, and RR for the given signal.

        Returns:
            dict with: stop_loss, take_profit, risk_reward_ratio
        """
        if signal == SignalType.HOLD:
            return {
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "risk_reward_ratio": 0.0,
            }

        # Adapt point unit to Indian Rupee (₹1/pt) vs Forex (0.0001/pip)
        is_rupee = entry_price > 50.0
        pip_unit = 1.0 if is_rupee else 0.0001
        sl_min_dist = self._sl_min_raw * pip_unit
        sl_spread_buf = self._sl_spread_raw * pip_unit
        default_atr = 15.0 if is_rupee else 0.0005
        atr = state.atr if state.atr > 0 else default_atr

        # ── Stop Loss ─────────────────────────────────────────────
        if signal == SignalType.BUY:
            # SL below structure low + ATR buffer + spread
            structure_sl = state.last_swing_low
            atr_sl = entry_price - atr * self._sl_atr_mult

            # Use the more conservative (closer to entry) of the two
            sl = max(structure_sl, atr_sl) if structure_sl > 0 else atr_sl
            sl -= sl_spread_buf  # Spread buffer

            # Enforce minimum distance
            if entry_price - sl < sl_min_dist:
                sl = entry_price - sl_min_dist

        else:  # SELL
            structure_sl = state.last_swing_high
            atr_sl = entry_price + atr * self._sl_atr_mult

            sl = min(structure_sl, atr_sl) if structure_sl > 0 else atr_sl
            sl += sl_spread_buf

            if sl - entry_price < sl_min_dist:
                sl = entry_price + sl_min_dist

        # ── Take Profit ───────────────────────────────────────────
        risk = abs(entry_price - sl)

        # Primary: Fibonacci extension
        tp = self._fibonacci_tp(signal, state, entry_price)

        # Secondary: Harmonic PRZ
        if tp is None or tp == 0.0:
            tp = self._harmonic_tp(signal, state, entry_price)

        # Dynamic: Ensure minimum RR
        if tp is None or tp == 0.0:
            # Default to minimum RR × risk
            if signal == SignalType.BUY:
                tp = entry_price + risk * self._min_rr
            else:
                tp = entry_price - risk * self._min_rr
        else:
            # Check RR meets minimum
            reward = abs(tp - entry_price)
            if risk > 0 and reward / risk < self._min_rr:
                if signal == SignalType.BUY:
                    tp = entry_price + risk * self._min_rr
                else:
                    tp = entry_price - risk * self._min_rr

        # Final RR ratio
        reward = abs(tp - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        return {
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "risk_reward_ratio": round(float(rr), 2),
        }

    def _fibonacci_tp(
        self,
        signal: SignalType,
        state: IndicatorState,
        entry: float,
    ) -> float | None:
        """Take profit from Fibonacci extension levels."""
        extensions = state.fibonacci.extensions
        if not extensions:
            return None

        if signal == SignalType.BUY:
            # Use the nearest extension level above entry
            candidates = [v for v in extensions.values() if v > entry]
            return min(candidates) if candidates else None
        else:
            # Nearest extension below entry
            candidates = [v for v in extensions.values() if v < entry]
            return max(candidates) if candidates else None

    def _harmonic_tp(
        self,
        signal: SignalType,
        state: IndicatorState,
        entry: float,
    ) -> float | None:
        """Take profit from harmonic PRZ target."""
        if state.active_pattern is None:
            return None

        prz = state.active_pattern.predicted_prz
        if prz is None or prz == 0:
            prz = (state.active_pattern.prz_high + state.active_pattern.prz_low) / 2

        if signal == SignalType.BUY and prz > entry:
            return float(prz)
        elif signal == SignalType.SELL and prz < entry:
            return float(prz)

        return None
