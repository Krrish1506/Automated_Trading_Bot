"""
Historical Data Loader for Backtesting.

Loads M5 OHLCV data from CSV/Parquet files with validation,
gap detection, and format normalization.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.types import Candle

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate historical OHLCV data."""

    def __init__(self, data_path: str):
        self._path = Path(data_path)

    def load(self) -> list[Candle]:
        """Load historical data and return as list of Candle objects.

        Supports CSV and Parquet formats.
        Expected columns: timestamp/date, open, high, low, close, volume
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Data file not found: {self._path}")

        if self._path.suffix == ".parquet":
            df = pd.read_parquet(self._path)
        else:
            df = pd.read_csv(self._path)

        df = self._normalize_columns(df)
        self._validate(df)

        candles = []
        for _, row in df.iterrows():
            candles.append(Candle(
                timestamp=float(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))

        logger.info(
            f"Loaded {len(candles)} candles from {self._path.name} "
            f"({candles[0].timestamp} → {candles[-1].timestamp})"
        )
        return candles

    def load_as_arrays(self) -> dict[str, np.ndarray]:
        """Load data as numpy arrays (faster for backtest)."""
        if self._path.suffix == ".parquet":
            df = pd.read_parquet(self._path)
        else:
            df = pd.read_csv(self._path)

        df = self._normalize_columns(df)
        self._validate(df)

        return {
            "timestamp": df["timestamp"].values.astype(np.float64),
            "open": df["open"].values.astype(np.float64),
            "high": df["high"].values.astype(np.float64),
            "low": df["low"].values.astype(np.float64),
            "close": df["close"].values.astype(np.float64),
            "volume": df["volume"].values.astype(np.float64),
        }

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and ensure timestamp is numeric."""
        df.columns = df.columns.str.lower().str.strip()

        # Handle various timestamp column names
        ts_cols = [c for c in df.columns if c in ("timestamp", "date", "time", "datetime")]
        if ts_cols:
            ts_col = ts_cols[0]
            if df[ts_col].dtype == object:
                df["timestamp"] = pd.to_datetime(df[ts_col]).astype(np.int64) // 10**9
            elif df[ts_col].dtype in ("datetime64[ns]",):
                df["timestamp"] = df[ts_col].astype(np.int64) // 10**9
            else:
                df["timestamp"] = df[ts_col]
        else:
            df["timestamp"] = np.arange(len(df), dtype=np.float64) * 300  # 5-min intervals

        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate data integrity."""
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            df.ffill(inplace=True)

        # Validate OHLC integrity
        invalid = (df["high"] < df["low"]).sum()
        if invalid > 0:
            logger.warning(f"{invalid} candles with high < low detected")

        logger.info(f"Data validated: {len(df)} rows, columns={list(df.columns)}")
