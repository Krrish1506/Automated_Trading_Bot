"""
Trade Execution Engine.

Receives SIGNAL_GENERATED events, formats structured JSON output,
logs to monitoring, and (in live mode) places orders via broker API.
Tracks open positions to avoid overexposure.
"""
from __future__ import annotations

import json
import time
import logging
from collections import deque

from core.event_bus import EventBus
from core.events import SignalGeneratedEvent, EventType, TickEvent
from core.types import TradeSignal, SignalType
from execution.reporter import TradeReporter

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Trade execution and output formatting engine.

    Responsibilities:
    - Format trade signals into strict JSON structure
    - Log all decisions with full context
    - Track open positions
    - Rate-limit order placement
    - In live mode: interface with broker API (future extension)
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: dict,
    ):
        self._bus = event_bus
        self._cfg = config

        risk_cfg = config.get("risk", {})
        self._max_positions = risk_cfg.get("max_open_positions", 1)

        # State
        self._open_positions: list[dict] = []
        self._trade_history: deque[dict] = deque(maxlen=1000)
        self._signal_count: int = 0
        self._trade_count: int = 0
        
        # Risk & Session Tracking
        risk_cfg = config.get("risk", {})
        self._session_duration = risk_cfg.get("session_duration_minutes", 180) * 60
        
        bt_cfg = config.get("backtest", {})
        initial_cap = bt_cfg.get("initial_equity", 1000000.0)
        self._reporter = TradeReporter(initial_capital=initial_cap)
        
        max_dd_pct = risk_cfg.get("max_drawdown_pct", 15.0) / 100.0
        self._max_dd_amount = initial_cap * max_dd_pct
        
        self._start_time = time.time()
        self._session_stopped = False

        # Subscribe
        self._bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal)
        self._bus.subscribe(EventType.TICK, self._on_tick)

    async def _on_signal(self, event: SignalGeneratedEvent) -> None:
        """Process generated signal."""
        signal = event.signal
        self._signal_count += 1

        # Format output
        output = signal.to_dict()

        # Always log the structured output
        logger.info(f"SIGNAL_OUTPUT: {json.dumps(output, indent=2)}")

        # Only execute trades for BUY/SELL
        if signal.signal == SignalType.HOLD:
            return

        # Check position limits
        if len(self._open_positions) >= self._max_positions:
            logger.warning(
                f"Max positions ({self._max_positions}) reached, "
                f"skipping {signal.signal.value}"
            )
            return

        # Record the trade intent
        trade_record = {
            "timestamp": time.time(),
            "signal": output,
            "signal_obj": signal,
            "status": "pending",
        }

        self._open_positions.append(trade_record)
        self._trade_history.append(trade_record)
        self._trade_count += 1

        fmt = ".2f" if signal.entry_price > 50 else ".5f"
        logger.info(
            f"TRADE EXECUTED: {signal.signal.value} {signal.symbol} @ "
            f"₹{signal.entry_price:{fmt}} | SL=₹{signal.stop_loss:{fmt}} | "
            f"TP=₹{signal.take_profit:{fmt}} | RR={signal.risk_reward_ratio:.2f} | "
            f"Confidence={signal.confidence:.0f}%"
        )

    async def _on_tick(self, event: TickEvent) -> None:
        """Process incoming ticks to check TP/SL and session timers."""
        if self._session_stopped:
            return

        elapsed = time.time() - self._start_time
        
        # Auto-stop when session duration reached
        if elapsed >= self._session_duration:
            logger.info("Session duration reached. Stopping trading and generating report.")
            self._session_stopped = True
            self._close_all_positions(event.mid)
            self._reporter.generate_reports()
            return
            
        # Close positions 30 minutes before session ends
        if elapsed >= self._session_duration - 1800:
            if self._open_positions:
                logger.info("30 minutes left in session. Closing all open positions.")
                self._close_all_positions(event.mid)

        # Check TP/SL
        for i in range(len(self._open_positions)-1, -1, -1):
            pos = self._open_positions[i]
            signal: TradeSignal = pos["signal_obj"]
            direction = signal.signal.value
            tp = signal.take_profit
            sl = signal.stop_loss
            
            close_price = None
            if direction == "BUY":
                if event.mid >= tp or event.mid <= sl:
                    close_price = event.mid
            elif direction == "SELL":
                if event.mid <= tp or event.mid >= sl:
                    close_price = event.mid
                    
            if close_price is not None:
                self.close_position(i, close_price)
                
    def _close_all_positions(self, market_price: float) -> None:
        """Force close all open positions at current market price."""
        while self._open_positions:
            self.close_position(0, market_price)

    def close_position(self, index: int, close_price: float) -> None:
        """Close an open position and log it via TradeReporter."""
        if index < len(self._open_positions):
            pos = self._open_positions.pop(index)
            pos["status"] = "closed"
            pos["close_time"] = time.time()
            
            signal: TradeSignal = pos["signal_obj"]
            direction = signal.signal.value
            entry = signal.entry_price
            
            # Simplified risk modeling (assume 2% risk sizing applied)
            risk_amount = self._reporter.initial_capital * 0.02
            sl_dist = abs(entry - signal.stop_loss)
            pips_gained = (close_price - entry) if direction == "BUY" else (entry - close_price)
            
            # PnL = (pips_gained / sl_dist) * risk_amount
            pnl_inr = (pips_gained / max(sl_dist, 0.0001)) * risk_amount
            pos["pnl_inr"] = pnl_inr
            
            # Record trade
            self._reporter.record_trade(
                direction=direction,
                entry_price=entry,
                exit_price=close_price,
                fib_level=signal.fib_level,
                ml_confidence=signal.confidence,
                pattern=signal.pattern or "No Pattern",
                confluences=signal.active_confluence
            )
            # Update PnL manually since record_trade generated a zero PnL
            self._reporter.update_last_trade_pnl(pnl_inr)
            
            logger.info(f"Position closed: PnL=₹{pnl_inr:.2f} at price {close_price:.5f}")
            
            # Drawdown check
            current_loss = self._reporter.initial_capital - self._reporter.current_capital
            if current_loss >= self._max_dd_amount:
                logger.error("EMERGENCY STOP: Max drawdown exceeded.")
                self._session_stopped = True
                self._reporter.generate_reports()

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def stats(self) -> dict:
        return {
            "signals_received": self._signal_count,
            "trades_executed": self._trade_count,
            "open_positions": len(self._open_positions),
            "pnl_inr": self._reporter.current_capital - self._reporter.initial_capital
        }

    @property
    def trade_history(self) -> list[dict]:
        return list(self._trade_history)
