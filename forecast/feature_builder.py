"""
Feature Builder — Assembles Multi-Factor Input for the Forecast Model.

Transforms candle buffer + indicator state into normalized feature
tensors suitable for LSTM/Transformer input. Uses running statistics
for normalization (zero disk I/O).
"""
from __future__ import annotations

import numpy as np
import torch

from core.candle_buffer import CandleBuffer
from core.types import IndicatorState, TrendDirection


class FeatureBuilder:
    """Builds normalized feature tensors for the AI forecast model.

    Features (12 total):
    0: close (normalized)
    1: open (normalized)
    2: high (normalized)
    3: low (normalized)
    4: ATR (normalized)
    5: RSI (0-100 → 0-1)
    6: MACD histogram (normalized)
    7: volume (normalized)
    8: BOS flag (0/1)
    9: AlphaTrend direction (-1/0/1)
    10: nearest_demand_dist (normalized)
    11: nearest_supply_dist (normalized)

    Normalization uses running mean/std computed from the buffer
    to avoid storing scaler artifacts on disk.
    """

    def __init__(self, sequence_length: int = 30, input_features: int = 12):
        self._seq_len = sequence_length
        self._n_features = input_features

        # Running statistics for normalization
        self._running_mean = np.zeros(4, dtype=np.float64)  # OHLC
        self._running_var = np.ones(4, dtype=np.float64)
        self._count = 0

    def build(
        self,
        buffer: CandleBuffer,
        state: IndicatorState,
    ) -> torch.Tensor | None:
        """Build feature tensor from buffer and indicator state.

        Returns:
            Tensor of shape (1, seq_len, n_features) or None if
            insufficient data.
        """
        if buffer.size < self._seq_len + 1:
            return None

        closes = buffer.get_array("close", self._seq_len)
        opens = buffer.get_array("open", self._seq_len)
        highs = buffer.get_array("high", self._seq_len)
        lows = buffer.get_array("low", self._seq_len)
        volumes = buffer.get_array("volume", self._seq_len)

        # Update running stats from OHLC
        ohlc = np.stack([opens, highs, lows, closes], axis=1)
        self._update_stats(ohlc)

        # Normalize OHLC using running statistics
        ohlc_norm = (ohlc - self._running_mean) / (np.sqrt(self._running_var) + 1e-8)

        # Build feature matrix (seq_len, n_features)
        features = np.zeros((self._seq_len, self._n_features), dtype=np.float64)

        features[:, 0] = ohlc_norm[:, 3]  # close
        features[:, 1] = ohlc_norm[:, 0]  # open
        features[:, 2] = ohlc_norm[:, 1]  # high
        features[:, 3] = ohlc_norm[:, 2]  # low

        # ATR normalized by mean close
        mean_close = np.mean(closes) if np.mean(closes) > 0 else 1.0
        features[:, 4] = state.atr / mean_close if state.atr > 0 else 0.0

        # RSI → 0-1 scale
        features[:, 5] = state.rsi / 100.0

        # MACD histogram normalized
        features[:, 6] = state.macd_histogram / (mean_close * 0.001) if mean_close > 0 else 0.0

        # Volume normalized
        vol_mean = np.mean(volumes)
        vol_std = np.std(volumes) + 1e-8
        features[:, 7] = (volumes - vol_mean) / vol_std

        # BOS flag
        features[:, 8] = 1.0 if state.structure.bos else 0.0

        # AlphaTrend direction
        if state.trend_direction == TrendDirection.BULLISH:
            features[:, 9] = 1.0
        elif state.trend_direction == TrendDirection.BEARISH:
            features[:, 9] = -1.0
        else:
            features[:, 9] = 0.0

        # Zone distances (capped and normalized)
        features[:, 10] = min(state.nearest_demand_dist, 10.0) / 10.0
        features[:, 11] = min(state.nearest_supply_dist, 10.0) / 10.0

        # Convert to tensor
        tensor = torch.tensor(
            features, dtype=torch.float32
        ).unsqueeze(0)  # (1, seq_len, n_features)

        return tensor

    def _update_stats(self, ohlc: np.ndarray) -> None:
        """Update running mean/variance using Welford's algorithm."""
        for row in ohlc:
            self._count += 1
            delta = row - self._running_mean
            self._running_mean += delta / self._count
            delta2 = row - self._running_mean
            self._running_var += (delta * delta2 - self._running_var) / self._count
