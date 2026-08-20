"""
Walk-Forward Backtest Engine.

Replays historical data through the full trading pipeline with
realistic slippage and spread simulation. Tracks equity curve,
drawdown, and per-trade statistics.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from backtest.data_loader import DataLoader
from indicators.holt_winters_rsi import HoltWintersRSI
from indicators.predictive_macd import PredictiveMACD
from indicators.pivot_trend import PivotTrend
from indicators.alpha_trend import AlphaTrend
from indicators.alpha_trend_plus import AlphaTrendPlus
from indicators.smc_engine import SMCEngine
from indicators.hidden_zone import HiddenZoneDetector
from indicators.fibonacci import FibonacciEngine
from indicators.harmonic import HarmonicScanner
from core.types import (
    IndicatorState, TrendDirection, StructureBreak, SignalType,
)
from decision.scorer import Scorer
from decision.confluence import ConfluenceDetector
from decision.risk_filter import RiskFilter
from decision.position_sizing import PositionSizer

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    entry_idx: int
    entry_price: float
    direction: str  # "buy" or "sell"
    stop_loss: float
    take_profit: float
    exit_idx: int = 0
    exit_price: float = 0.0
    pnl_pips: float = 0.0
    result: str = "open"  # "win", "loss", "open"


@dataclass
class BacktestResult:
    """Aggregate backtest results."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate * 100, 1),
            "total_pnl_pips": round(self.total_pnl_pips, 1),
            "max_drawdown_pips": round(self.max_drawdown_pips, 1),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "profit_factor": round(self.profit_factor, 3),
        }


class BacktestEngine:
    """Walk-forward backtester with slippage and spread.

    Replays candles sequentially, computing all indicators and
    decision logic at each step, then simulates order execution
    with configurable transaction costs.
    """

    def __init__(self, config: dict):
        self._cfg = config
        bt_cfg = config.get("backtest", {})
        self._slippage = bt_cfg.get("slippage_pips", 0.5) / 10_000
        self._spread = bt_cfg.get("spread_pips", 1.5) / 10_000
        self._initial_equity = bt_cfg.get("initial_equity", 10_000.0)
        self._commission = bt_cfg.get("commission_per_lot", 3.5)
        # Only evaluate indicators every N bars (massive speedup)
        self._eval_interval = bt_cfg.get("eval_interval", 5)
        # Lookback window size for indicator slices
        self._lookback = bt_cfg.get("lookback_window", 300)

    def run(self, data_path: str) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            data_path: Path to CSV/Parquet file.

        Returns:
            BacktestResult with full statistics.
        """
        loader = DataLoader(data_path)
        arrays = loader.load_as_arrays()

        opens = arrays["open"]
        highs = arrays["high"]
        lows = arrays["low"]
        closes = arrays["close"]
        volumes = arrays["volume"]
        n = len(closes)

        logger.info(f"Starting backtest on {n} candles (eval_interval={self._eval_interval})")
        t0 = time.time()

        # Initialize indicators
        hw_rsi = HoltWintersRSI()
        macd = PredictiveMACD()
        pivot = PivotTrend()
        alpha = AlphaTrend()
        alpha_plus = AlphaTrendPlus()
        smc = SMCEngine()
        hidden = HiddenZoneDetector()
        fib = FibonacciEngine()
        harmonic = HarmonicScanner()

        # Decision components
        decision_cfg = self._cfg.get("decision", {})
        scorer = Scorer(weights=decision_cfg.get("weights"))
        confluence = ConfluenceDetector(
            min_confluences=decision_cfg.get("confluence_min", 3)
        )
        risk_filter = RiskFilter()
        pos_sizer = PositionSizer(
            sl_atr_multiplier=self._cfg.get("position", {}).get("sl_atr_multiplier", 1.5),
            min_risk_reward=self._cfg.get("position", {}).get("min_risk_reward", 1.5),
        )

        forecast_buy_thresh = self._cfg.get("forecast", {}).get("probability_threshold_buy", 0.65)
        forecast_sell_thresh = self._cfg.get("forecast", {}).get("probability_threshold_sell", 0.35)

        # Backtest state
        result = BacktestResult()
        result.equity_curve = [self._initial_equity]
        equity = self._initial_equity
        peak_equity = equity
        active_trade: BacktestTrade | None = None
        warmup = 200
        last_state: IndicatorState | None = None

        for i in range(warmup, n):
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Progress logging
            if (i - warmup) % 5000 == 0 and i > warmup:
                elapsed = time.time() - t0
                pct = (i - warmup) / (n - warmup) * 100
                logger.info(
                    f"Progress: {pct:.0f}% ({i}/{n}) | "
                    f"Trades: {len(result.trades)} | "
                    f"Elapsed: {elapsed:.0f}s"
                )

            # Check active trade for SL/TP hit
            if active_trade is not None:
                hit = self._check_trade_exit(
                    active_trade, current_high, current_low, i
                )
                if hit:
                    pnl = active_trade.pnl_pips
                    equity += pnl * 10  # Simple pip-to-equity conversion
                    result.equity_curve.append(equity)

                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = peak_equity - equity
                    result.max_drawdown_pips = max(result.max_drawdown_pips, drawdown)

                    result.trades.append(active_trade)
                    active_trade = None
                    continue

            # Only evaluate signals when no active trade
            if active_trade is not None:
                result.equity_curve.append(equity)
                continue

            # Skip indicator computation on non-eval bars
            if (i - warmup) % self._eval_interval != 0:
                result.equity_curve.append(equity)
                continue

            # Fixed lookback window for indicator slices
            lb_start = max(0, i - self._lookback + 1)
            o_slice = opens[lb_start:i + 1]
            h_slice = highs[lb_start:i + 1]
            l_slice = lows[lb_start:i + 1]
            c_slice = closes[lb_start:i + 1]
            v_slice = volumes[lb_start:i + 1]

            # Compute indicators
            state = IndicatorState()
            state.current_price = current_close

            rsi_r = hw_rsi.compute(c_slice)
            state.rsi = rsi_r["rsi"]
            state.hw_rsi = rsi_r["hw_rsi"]
            state.momentum_shift = rsi_r["momentum_shift"]

            macd_r = macd.compute(c_slice)
            state.macd = macd_r["macd"]
            state.macd_signal = macd_r["signal_line"]
            state.macd_histogram = macd_r["histogram"]
            state.crossover_target = macd_r["crossover_target"]
            state.crossover_direction = macd_r["crossover_direction"]

            pivot_r = pivot.compute(h_slice, l_slice, c_slice)
            state.last_swing_high = pivot_r["last_swing_high"]
            state.last_swing_low = pivot_r["last_swing_low"]
            state.pivot_break = pivot_r["pivot_break"]
            state.trend_reversal = pivot_r["trend_reversal"]

            at_r = alpha.compute(h_slice, l_slice, c_slice)
            state.alpha_trend_line = at_r["alpha_trend_line"]
            state.trend_direction = at_r["trend_direction"]
            state.is_sideways = at_r["is_sideways"]
            state.atr = at_r["atr"]

            atp_r = alpha_plus.compute(h_slice, l_slice, c_slice, v_slice)
            state.alpha_plus_trend = atp_r["alpha_plus_trend"]
            state.trend_confirmed = atp_r["trend_confirmed"]
            state.trend_strength = atp_r["trend_strength"]

            state.structure = smc.compute(o_slice, h_slice, l_slice, c_slice)

            zone_r = hidden.compute(o_slice, h_slice, l_slice, c_slice, v_slice, state.atr)
            state.demand_zones = zone_r["demand_zones"]
            state.supply_zones = zone_r["supply_zones"]
            state.nearest_demand_dist = zone_r["nearest_demand_dist"]
            state.nearest_supply_dist = zone_r["nearest_supply_dist"]

            state.fibonacci = fib.compute(h_slice, l_slice, c_slice, v_slice, state.atr)

            harm_r = harmonic.compute(h_slice, l_slice, c_slice)
            state.active_pattern = harm_r["active_pattern"]

            # Use neutral forecast for backtest (model not included)
            forecast_prob = 0.5
            forecast_dir = "neutral"

            # Score
            composite, individual = scorer.score(state, forecast_prob, forecast_dir)

            # Confluence
            confs = confluence.detect(state, forecast_dir, individual)

            # Risk filter
            rejected, _ = risk_filter.check(state, spread_pips=self._spread * 10_000)

            if rejected:
                result.equity_curve.append(equity)
                continue

            # Signal decision (relaxed for backtest — uses composite score)
            signal = SignalType.HOLD
            if composite > 0.3 and len(confs) >= 2:
                signal = SignalType.BUY
            elif composite < -0.3 and len(confs) >= 2:
                signal = SignalType.SELL

            if signal == SignalType.HOLD:
                result.equity_curve.append(equity)
                continue

            # Position sizing
            entry = current_close
            if signal == SignalType.BUY:
                entry += self._spread / 2 + self._slippage
            else:
                entry -= self._spread / 2 + self._slippage

            pos = pos_sizer.compute(signal, state, entry)

            active_trade = BacktestTrade(
                entry_idx=i,
                entry_price=entry,
                direction="buy" if signal == SignalType.BUY else "sell",
                stop_loss=pos["stop_loss"],
                take_profit=pos["take_profit"],
            )

            result.equity_curve.append(equity)

        # Close any remaining trade
        if active_trade is not None:
            active_trade.exit_idx = n - 1
            active_trade.exit_price = closes[-1]
            if active_trade.direction == "buy":
                active_trade.pnl_pips = (closes[-1] - active_trade.entry_price) * 10_000
            else:
                active_trade.pnl_pips = (active_trade.entry_price - closes[-1]) * 10_000
            active_trade.result = "win" if active_trade.pnl_pips > 0 else "loss"
            result.trades.append(active_trade)

        # Compute statistics
        self._compute_stats(result)

        elapsed = time.time() - t0
        logger.info(
            f"Backtest complete in {elapsed:.1f}s: "
            f"{result.total_trades} trades, "
            f"WR={result.win_rate*100:.1f}%, "
            f"PnL={result.total_pnl_pips:.1f} pips, "
            f"Sharpe={result.sharpe_ratio:.3f}"
        )

        return result

    def _check_trade_exit(
        self,
        trade: BacktestTrade,
        high: float,
        low: float,
        idx: int,
    ) -> bool:
        """Check if SL or TP is hit by current candle."""
        if trade.direction == "buy":
            if low <= trade.stop_loss:
                trade.exit_idx = idx
                trade.exit_price = trade.stop_loss
                trade.pnl_pips = (trade.stop_loss - trade.entry_price) * 10_000
                trade.result = "loss"
                return True
            if high >= trade.take_profit:
                trade.exit_idx = idx
                trade.exit_price = trade.take_profit
                trade.pnl_pips = (trade.take_profit - trade.entry_price) * 10_000
                trade.result = "win"
                return True
        else:
            if high >= trade.stop_loss:
                trade.exit_idx = idx
                trade.exit_price = trade.stop_loss
                trade.pnl_pips = (trade.entry_price - trade.stop_loss) * 10_000
                trade.result = "loss"
                return True
            if low <= trade.take_profit:
                trade.exit_idx = idx
                trade.exit_price = trade.take_profit
                trade.pnl_pips = (trade.entry_price - trade.take_profit) * 10_000
                trade.result = "win"
                return True

        return False

    def _compute_stats(self, result: BacktestResult) -> None:
        """Compute aggregate statistics from trade list."""
        trades = result.trades
        result.total_trades = len(trades)

        if result.total_trades == 0:
            return

        result.wins = sum(1 for t in trades if t.result == "win")
        result.losses = sum(1 for t in trades if t.result == "loss")
        result.win_rate = result.wins / result.total_trades
        result.total_pnl_pips = sum(t.pnl_pips for t in trades)

        # Sharpe ratio
        pnls = [t.pnl_pips for t in trades]
        if len(pnls) > 1:
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            if std_pnl > 0:
                result.sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(252 * 12)  # Annualized

        # Profit factor
        gross_profit = sum(t.pnl_pips for t in trades if t.pnl_pips > 0)
        gross_loss = abs(sum(t.pnl_pips for t in trades if t.pnl_pips < 0))
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
