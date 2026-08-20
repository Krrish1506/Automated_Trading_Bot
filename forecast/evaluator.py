"""
Model Evaluator — Classification + Trading Metrics.

Evaluates LSTM model predictions with both standard ML metrics
and simulated trading performance metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation metrics."""
    # Classification
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0

    # Trading
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    cagr: float = 0.0
    total_trades: int = 0
    total_return: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


class Evaluator:
    """Computes classification and trading metrics from model predictions."""

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """Compute standard classification metrics.

        Args:
            y_true: ground truth binary labels
            y_prob: predicted probabilities (0-1)
            threshold: decision threshold
        """
        y_pred = (y_prob >= threshold).astype(np.float64)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # ROC AUC (manual computation to avoid sklearn dependency)
        roc_auc = Evaluator._compute_roc_auc(y_true, y_prob)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        }

    @staticmethod
    def trading_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        closes: np.ndarray,
        threshold_buy: float = 0.55,
        threshold_sell: float = 0.45,
        spread_pips: float = 1.5,
        horizon: int = 12,
    ) -> dict:
        """Simulate trades and compute trading performance metrics.

        Strategy: go long when prob > threshold_buy, short when prob < threshold_sell.
        Hold position for `horizon` bars, then exit.

        Args:
            y_true: ground truth labels
            y_prob: predicted probabilities
            closes: close prices corresponding to each prediction
            threshold_buy: probability threshold for long
            threshold_sell: probability threshold for short
            spread_pips: spread cost in pips
            horizon: bars to hold each trade
        """
        n = len(y_prob)
        spread = spread_pips * 0.0001  # Convert pips to price

        trades = []  # List of (entry_price, exit_price, direction, pnl)
        equity = [1.0]  # Equity curve starting at 1.0
        position_size = 0.02  # Risk 2% per trade

        i = 0
        while i < n - horizon:
            prob = y_prob[i]

            if prob > threshold_buy:
                # Long trade
                entry = closes[i] + spread / 2
                exit_price = closes[min(i + horizon, n - 1)]
                pnl = (exit_price - entry) / entry
                trades.append({
                    "direction": "long",
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                })
                equity.append(equity[-1] * (1 + pnl * position_size / 0.02))
                i += horizon  # Skip ahead
            elif prob < threshold_sell:
                # Short trade
                entry = closes[i] - spread / 2
                exit_price = closes[min(i + horizon, n - 1)]
                pnl = (entry - exit_price) / entry
                trades.append({
                    "direction": "short",
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                })
                equity.append(equity[-1] * (1 + pnl * position_size / 0.02))
                i += horizon
            else:
                equity.append(equity[-1])
                i += 1

        if len(trades) == 0:
            return {
                "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "cagr": 0.0, "total_trades": 0,
                "total_return": 0.0,
            }

        # Compute metrics
        pnls = np.array([t["pnl"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        # Win rate
        win_rate = len(wins) / max(len(pnls), 1)

        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-8
        profit_factor = gross_profit / max(gross_loss, 1e-8)

        # Sharpe ratio (annualized, assuming M5 bars, 252 trading days)
        bars_per_year = 252 * 24 * 12  # M5 bars per year
        avg_return = np.mean(pnls)
        std_return = np.std(pnls) + 1e-8
        trades_per_year = bars_per_year / max(horizon, 1)
        sharpe = (avg_return / std_return) * np.sqrt(min(trades_per_year, len(trades)))

        # Max drawdown
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_dd = float(np.max(drawdown))

        # Total return
        total_return = equity_arr[-1] / equity_arr[0] - 1

        # CAGR (approximate)
        n_bars = len(y_prob)
        years = n_bars / bars_per_year
        if years > 0 and total_return > -1:
            cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        else:
            cagr = 0.0

        return {
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "cagr": float(cagr),
            "total_trades": len(trades),
            "total_return": float(total_return),
        }

    @staticmethod
    def full_evaluation(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        closes: np.ndarray,
        horizon: int = 12,
    ) -> EvalResult:
        """Run full evaluation: classification + trading metrics."""
        cls_metrics = Evaluator.classification_metrics(y_true, y_prob)
        trd_metrics = Evaluator.trading_metrics(
            y_true, y_prob, closes, horizon=horizon
        )

        return EvalResult(
            accuracy=cls_metrics["accuracy"],
            precision=cls_metrics["precision"],
            recall=cls_metrics["recall"],
            f1=cls_metrics["f1"],
            roc_auc=cls_metrics["roc_auc"],
            sharpe=trd_metrics["sharpe"],
            max_drawdown=trd_metrics["max_drawdown"],
            win_rate=trd_metrics["win_rate"],
            profit_factor=trd_metrics["profit_factor"],
            cagr=trd_metrics["cagr"],
            total_trades=trd_metrics["total_trades"],
            total_return=trd_metrics["total_return"],
        )

    @staticmethod
    def _compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute ROC AUC without sklearn (pure NumPy)."""
        pos = y_prob[y_true == 1]
        neg = y_prob[y_true == 0]

        if len(pos) == 0 or len(neg) == 0:
            return 0.5

        # Mann-Whitney U statistic
        n_pos = len(pos)
        n_neg = len(neg)

        # Vectorized: count how often pos > neg
        auc = 0.0
        for p in pos:
            auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)

        return float(auc / (n_pos * n_neg))

    @staticmethod
    def print_report(
        fold_results: list[dict],
        test_result: EvalResult | None = None,
    ) -> None:
        """Print formatted statistical report."""
        print("\n" + "=" * 70)
        print("  LSTM TRAINING REPORT — EUR/USD M5")
        print("=" * 70)

        # Per-fold results
        print("\n--- Walk-Forward Fold Results ---")
        print(f"{'Fold':>4} | {'Acc':>6} | {'Prec':>6} | {'F1':>6} | "
              f"{'AUC':>6} | {'Sharpe':>7} | {'MaxDD':>7} | {'WinR':>6} | {'Trades':>6}")
        print("-" * 70)

        sharpes = []
        accs = []
        for r in fold_results:
            e = r.get("eval", EvalResult())
            print(f"{r.get('fold', 0):>4} | {e.accuracy:>6.3f} | {e.precision:>6.3f} | "
                  f"{e.f1:>6.3f} | {e.roc_auc:>6.3f} | {e.sharpe:>7.2f} | "
                  f"{e.max_drawdown:>7.3f} | {e.win_rate:>6.3f} | {e.total_trades:>6}")
            sharpes.append(e.sharpe)
            accs.append(e.accuracy)

        # Cross-fold statistics
        print("\n--- Cross-Fold Statistics ---")
        print(f"  Mean Sharpe:  {np.mean(sharpes):>8.3f} ± {np.std(sharpes):.3f}")
        print(f"  Mean Accuracy:{np.mean(accs):>8.3f} ± {np.std(accs):.3f}")

        # Final test results
        if test_result:
            print("\n--- Out-of-Sample Test Results ---")
            print(f"  Accuracy:      {test_result.accuracy:.4f}")
            print(f"  Precision:     {test_result.precision:.4f}")
            print(f"  Recall:        {test_result.recall:.4f}")
            print(f"  F1:            {test_result.f1:.4f}")
            print(f"  ROC-AUC:       {test_result.roc_auc:.4f}")
            print(f"  Sharpe:        {test_result.sharpe:.3f}")
            print(f"  Max Drawdown:  {test_result.max_drawdown:.4f}")
            print(f"  Win Rate:      {test_result.win_rate:.4f}")
            print(f"  Profit Factor: {test_result.profit_factor:.3f}")
            print(f"  CAGR:          {test_result.cagr:.4f}")
            print(f"  Total Trades:  {test_result.total_trades}")
            print(f"  Total Return:  {test_result.total_return:.4f}")

        # Model acceptance
        print("\n--- Model Acceptance ---")
        mean_sharpe = np.mean(sharpes) if sharpes else 0
        std_sharpe = np.std(sharpes) if sharpes else 0
        test_sharpe = test_result.sharpe if test_result else 0
        train_test_gap = abs(np.mean(accs) - (test_result.accuracy if test_result else 0))

        rejected = False
        if mean_sharpe < 1.0:
            print(f"  ⚠ CONCERN: Mean fold Sharpe ({mean_sharpe:.2f}) < 1.0")
        if std_sharpe > 0.5:
            print(f"  ⚠ CONCERN: Sharpe std ({std_sharpe:.2f}) > 0.5 — unstable")
            rejected = True
        if train_test_gap > 0.10:
            print(f"  ⚠ CONCERN: Train-test accuracy gap ({train_test_gap:.2f}) > 10% — possible overfit")
            rejected = True

        if rejected:
            print("  ❌ MODEL REJECTED — retrain with different hyperparameters")
        else:
            if mean_sharpe >= 1.0:
                print("  ✅ MODEL ACCEPTED — Sharpe ≥ 1.0, stable across folds")
            else:
                print("  ⚠  MODEL MARGINAL — Sharpe < 1.0 but stable. Consider retraining.")

        print("=" * 70 + "\n")
