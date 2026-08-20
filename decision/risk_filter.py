"""
Risk Filter — Trade Rejection Logic.

Rejects trades when market conditions are unfavorable:
- Excessive spread
- Volatility spike
- Daily loss limit reached
- Sideways market filter
"""
from __future__ import annotations

import logging

from core.types import IndicatorState

logger = logging.getLogger(__name__)


class RiskFilter:
    """Risk management filter for trade rejection.

    If any filter condition is triggered, the trade is rejected
    regardless of the signal strength.
    """

    def __init__(
        self,
        max_spread_pips: float = 3.0,
        volatility_spike_mult: float = 3.0,
        daily_loss_limit_pips: float = 50.0,
        adx_sideways_threshold: float = 20.0,
    ):
        self._max_spread = max_spread_pips
        self._vol_spike_mult = volatility_spike_mult
        self._daily_loss_limit = daily_loss_limit_pips
        self._adx_threshold = adx_sideways_threshold

        # Track daily P&L
        self._daily_pnl_pips: float = 0.0
        self._trade_count_today: int = 0

    def check(
        self,
        state: IndicatorState,
        spread_pips: float = 1.5,
        atr_sma: float = 0.0,
    ) -> tuple[bool, list[str]]:
        """Check all risk filters.

        Args:
            state: Current indicator state.
            spread_pips: Current bid-ask spread in pips.
            atr_sma: SMA of ATR for volatility spike detection.

        Returns:
            (should_reject, reasons) where should_reject is True
            if any filter is triggered.
        """
        reasons: list[str] = []

        # 1. Spread filter
        if spread_pips > self._max_spread:
            reasons.append(
                f"Spread too high: {spread_pips:.1f} > {self._max_spread:.1f} pips"
            )

        # 2. Volatility spike
        if atr_sma > 0 and state.atr > atr_sma * self._vol_spike_mult:
            reasons.append(
                f"Volatility spike: ATR={state.atr:.5f} > "
                f"{self._vol_spike_mult}x SMA={atr_sma:.5f}"
            )

        # 3. Daily loss limit
        if self._daily_pnl_pips < -self._daily_loss_limit:
            reasons.append(
                f"Daily loss limit: {self._daily_pnl_pips:.1f} < "
                f"-{self._daily_loss_limit:.1f} pips"
            )

        # 4. Sideways filter
        if state.is_sideways and not state.trend_confirmed:
            reasons.append("Sideways market: AlphaTrend sideways, no ADX confirmation")

        should_reject = len(reasons) > 0

        if should_reject:
            logger.warning(f"Trade rejected: {'; '.join(reasons)}")

        return should_reject, reasons

    def update_pnl(self, pnl_pips: float) -> None:
        """Update daily P&L tracking."""
        self._daily_pnl_pips += pnl_pips
        self._trade_count_today += 1

    def reset_daily(self) -> None:
        """Reset daily tracking (call at session start)."""
        self._daily_pnl_pips = 0.0
        self._trade_count_today = 0

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl_pips
