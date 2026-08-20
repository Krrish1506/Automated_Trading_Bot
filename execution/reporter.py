"""
Trade Reporter.

Records executed trades, manages session equity, and generates CSV/HTML
reports for the live competition.
"""
import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    trade_id: int
    timestamp: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_inr: float
    return_pct: float
    fib_level: float
    ml_confidence: float
    pattern: str
    confluences: str
    cumulative_equity: float

class TradeReporter:
    """Records trades and generates end-of-session reports."""

    def __init__(self, initial_capital: float = 1000000.0, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trades: List[TradeRecord] = []
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.trade_counter = 0

    def record_trade(self, direction: str, entry_price: float, exit_price: float, 
                     fib_level: float, ml_confidence: float, pattern: str, confluences: list[str]):
        """Log a closed trade."""
        self.trade_counter += 1
        
        # Calculate PnL (Assumes 2% risk sizing applied correctly, simplified calc here)
        # For a full implementation, the execution engine passes the exact PnL.
        # This is a placeholder for the exact PnL from the engine.
        pnl_inr = 0.0 # To be overridden by execution engine actuals if needed
        
        return_pct = (exit_price - entry_price) / entry_price if direction == "BUY" else (entry_price - exit_price) / entry_price
        
        # If we just use percentage return * position size. 
        # But we'll let the engine pass the exact pnl_inr. For now, we expect it to be updated.
        
        # Update capital
        self.current_capital += pnl_inr
        
        record = TradeRecord(
            trade_id=self.trade_counter,
            timestamp=datetime.now().isoformat(),
            symbol="EUR/USD",
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_inr=pnl_inr,
            return_pct=return_pct,
            fib_level=fib_level,
            ml_confidence=ml_confidence,
            pattern=pattern,
            confluences=", ".join(confluences),
            cumulative_equity=self.current_capital
        )
        self.trades.append(record)
        logger.info(f"Recorded Trade #{self.trade_counter}: {direction} P&L: ₹{pnl_inr:.2f}")
        return record

    def update_last_trade_pnl(self, pnl_inr: float):
        """Update P&L of the last recorded trade (used if P&L is calculated after recording)."""
        if self.trades:
            self.current_capital = self.current_capital - self.trades[-1].pnl_inr + pnl_inr
            self.trades[-1].pnl_inr = pnl_inr
            self.trades[-1].cumulative_equity = self.current_capital

    def generate_reports(self) -> dict:
        """Generate CSV, JSON, and HTML reports."""
        if not self.trades:
            logger.info("No trades to report.")
            return {}

        base_path = self.output_dir / f"session_{self.session_id}"
        
        # 1. CSV
        csv_path = base_path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TradeRecord.__dataclass_fields__.keys())
            writer.writeheader()
            for t in self.trades:
                writer.writerow(asdict(t))

        # 2. JSON Summary
        wins = [t for t in self.trades if t.pnl_inr > 0]
        summary = {
            "session_id": self.session_id,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "initial_capital": self.initial_capital,
            "final_capital": self.current_capital,
            "total_pnl": self.current_capital - self.initial_capital,
            "total_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0.0,
        }
        
        json_path = base_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump({"summary": summary, "trades": [asdict(t) for t in self.trades]}, f, indent=2)

        # 3. HTML (Simple structured table)
        html_path = base_path.with_suffix(".html")
        html_content = f"""
        <html>
        <head><style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .win {{ color: green; }}
            .loss {{ color: red; }}
        </style></head>
        <body>
            <h2>AI Trading Session Report: {self.session_id}</h2>
            <p><strong>Total P&L:</strong> ₹{summary['total_pnl']:.2f}</p>
            <p><strong>Win Rate:</strong> {summary['win_rate']*100:.1f}%</p>
            <p><strong>Total Trades:</strong> {summary['total_trades']}</p>
            <table>
                <tr>
                    <th>ID</th><th>Time</th><th>Dir</th><th>Entry</th><th>Exit</th>
                    <th>Fib</th><th>Conf %</th><th>Pattern</th><th>P&L (₹)</th>
                </tr>
        """
        for t in self.trades:
            pnl_class = "win" if t.pnl_inr > 0 else "loss"
            html_content += f"""
                <tr>
                    <td>{t.trade_id}</td>
                    <td>{t.timestamp}</td>
                    <td>{t.direction}</td>
                    <td>{t.entry_price:.5f}</td>
                    <td>{t.exit_price:.5f}</td>
                    <td>{t.fib_level:.3f}</td>
                    <td>{t.ml_confidence:.1f}%</td>
                    <td>{t.pattern}</td>
                    <td class="{pnl_class}">₹{t.pnl_inr:.2f}</td>
                </tr>
            """
        html_content += "</table></body></html>"
        
        with open(html_path, "w") as f:
            f.write(html_content)
            
        logger.info(f"Reports saved to {self.output_dir}/session_{self.session_id}.*")
        
        return {
            "csv": str(csv_path),
            "json": str(json_path),
            "html": str(html_path)
        }
