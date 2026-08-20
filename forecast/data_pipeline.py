"""
Offline Data Pipeline — Feature Engineering for LSTM Training.

Reads raw OHLCV CSV, computes all indicators bar-by-bar using only
past data (no leakage), and produces a feature matrix + binary labels
ready for walk-forward training.

Usage:
    from forecast.data_pipeline import DataPipeline
    pipeline = DataPipeline("data/eurusd_m5.csv")
    features, labels, timestamps = pipeline.build()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from indicators.holt_winters_rsi import HoltWintersRSI
from indicators.predictive_macd import PredictiveMACD
from indicators.pivot_trend import PivotTrend
from indicators.alpha_trend import AlphaTrend
from indicators.alpha_trend_plus import AlphaTrendPlus
from indicators.smc_engine import SMCEngine
from indicators.hidden_zone import HiddenZoneDetector
from indicators.fibonacci import FibonacciEngine
from indicators.harmonic import HarmonicScanner

logger = logging.getLogger(__name__)

# Feature names in order
FEATURE_NAMES = [
    "close_norm",         # 0
    "open_norm",          # 1
    "high_norm",          # 2
    "low_norm",           # 3
    "atr_norm",           # 4
    "rsi",                # 5
    "hw_rsi",             # 6
    "macd_histogram",     # 7
    "macd_signal",        # 8
    "volume_norm",        # 9
    "bos_flag",           # 10
    "choch_flag",         # 11
    "alpha_trend_dir",    # 12
    "trend_strength",     # 13
    "demand_dist",        # 14
    "supply_dist",        # 15
    "fib_distance",       # 16
    "harmonic_prz",       # 17
    "pivot_break",        # 18
    "momentum_shift",     # 19
]

N_FEATURES = len(FEATURE_NAMES)


class DataPipeline:
    """Builds feature matrix from raw OHLCV data using all indicator modules.

    Guarantees no data leakage: each bar's features use only past data.
    """

    def __init__(
        self,
        csv_path: str = "data/eurusd_m5.csv",
        prediction_horizon: int = 12,
        warmup: int = 200,
        config: dict | None = None,
    ):
        self._csv_path = csv_path
        self._horizon = prediction_horizon
        self._warmup = warmup
        self._cfg = config or {}

    def build(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build features and labels from CSV.

        Returns:
            features: (n_valid, n_features) array
            labels: (n_valid,) binary array
            timestamps: (n_valid,) array
        """
        logger.info(f"Loading data from {self._csv_path}")
        df = pd.read_csv(self._csv_path)
        n = len(df)
        logger.info(f"Loaded {n} bars")

        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)
        timestamps = df["timestamp"].values.astype(np.float64)

        # --- Initialize indicators ---
        ind_cfg = self._cfg.get("indicators", {})

        hw_rsi = HoltWintersRSI(
            rsi_period=ind_cfg.get("rsi_period", 14),
            alpha=ind_cfg.get("hw_alpha", 0.3),
            beta=ind_cfg.get("hw_beta", 0.05),
            gamma=ind_cfg.get("hw_gamma", 0.1),
        )
        macd = PredictiveMACD(
            fast=ind_cfg.get("macd_fast", 12),
            slow=ind_cfg.get("macd_slow", 26),
            signal=ind_cfg.get("macd_signal", 9),
        )
        pivot = PivotTrend(
            swing_window=ind_cfg.get("pivot_swing_window", 5),
        )
        alpha_trend = AlphaTrend(
            atr_period=ind_cfg.get("alpha_trend_atr_period", 14),
            atr_multiplier=ind_cfg.get("alpha_trend_atr_multiplier", 1.0),
        )
        alpha_plus = AlphaTrendPlus(
            atr_period=ind_cfg.get("alpha_plus_atr_period", 14),
            atr_multiplier=ind_cfg.get("alpha_plus_atr_multiplier", 1.5),
            adx_period=ind_cfg.get("alpha_plus_adx_period", 14),
            adx_threshold=ind_cfg.get("alpha_plus_adx_threshold", 20.0),
            volume_surge_mult=ind_cfg.get("alpha_plus_volume_surge_mult", 1.5),
        )
        smc = SMCEngine(
            swing_window=ind_cfg.get("smc_swing_window", 5),
            order_block_lookback=ind_cfg.get("smc_ob_lookback", 10),
        )
        hidden_zone = HiddenZoneDetector(
            vol_cluster_window=ind_cfg.get("hz_vol_cluster_window", 20),
            imbalance_threshold=ind_cfg.get("hz_imbalance_threshold", 1.5),
            zone_max_age=ind_cfg.get("hz_zone_max_age", 100),
        )
        fibonacci = FibonacciEngine()
        harmonic = HarmonicScanner()

        # Import types for enum comparisons
        from core.types import TrendDirection, StructureBreak

        # --- Compute features bar-by-bar with lookback window ---
        logger.info("Computing features (bar-by-bar, no leakage)...")
        lookback = 500  # Fixed lookback window
        feature_matrix = np.full((n, N_FEATURES), np.nan, dtype=np.float64)

        for i in range(self._warmup, n):
            lb = max(0, i - lookback + 1)
            o_s = opens[lb:i + 1]
            h_s = highs[lb:i + 1]
            l_s = lows[lb:i + 1]
            c_s = closes[lb:i + 1]
            v_s = volumes[lb:i + 1]

            # Compute indicators
            try:
                rsi_r = hw_rsi.compute(c_s)
                macd_r = macd.compute(c_s)
                pivot_r = pivot.compute(h_s, l_s, c_s)
                at_r = alpha_trend.compute(h_s, l_s, c_s)
                atp_r = alpha_plus.compute(h_s, l_s, c_s, v_s)
                smc_r = smc.compute(o_s, h_s, l_s, c_s)
                atr = at_r.get("atr", 0.0)
                zone_r = hidden_zone.compute(o_s, h_s, l_s, c_s, v_s, atr)
                fib_r = fibonacci.compute(h_s, l_s, c_s, v_s, atr)
                harm_r = harmonic.compute(h_s, l_s, c_s)
            except Exception as e:
                if i == self._warmup:
                    logger.warning(f"Indicator error at bar {i}: {e}")
                continue

            # Current price for normalization
            price = c_s[-1]
            mean_c = np.mean(c_s[-min(100, len(c_s)):])
            if mean_c <= 0:
                mean_c = 1.0

            # Fill feature row
            row = np.zeros(N_FEATURES, dtype=np.float64)

            # 0-3: OHLC normalized (returns-based for stationarity)
            row[0] = (closes[i] - mean_c) / mean_c
            row[1] = (opens[i] - mean_c) / mean_c
            row[2] = (highs[i] - mean_c) / mean_c
            row[3] = (lows[i] - mean_c) / mean_c

            # 4: ATR normalized by mean close
            row[4] = atr / mean_c if atr > 0 else 0.0

            # 5-6: RSI and HW-RSI (already 0-100, scale to 0-1)
            row[5] = rsi_r.get("rsi", 50.0) / 100.0
            row[6] = rsi_r.get("hw_rsi", 50.0) / 100.0

            # 7-8: MACD histogram and signal normalized
            row[7] = macd_r.get("histogram", 0.0) / (mean_c * 0.001) if mean_c > 0 else 0.0
            row[8] = macd_r.get("signal_line", 0.0) / (mean_c * 0.001) if mean_c > 0 else 0.0

            # 9: Volume (will be z-scored later using train stats)
            row[9] = volumes[i]

            # 10-11: Structure flags (smc_r is a StructureState object)
            row[10] = 1.0 if getattr(smc_r, "bos", False) else 0.0
            row[11] = 1.0 if getattr(smc_r, "choch", False) else 0.0

            # 12: AlphaTrend direction (TrendDirection enum)
            td = at_r.get("trend_direction", None)
            if td is not None:
                if td == TrendDirection.BULLISH:
                    row[12] = 1.0
                elif td == TrendDirection.BEARISH:
                    row[12] = -1.0
                else:
                    row[12] = 0.0

            # 13: Trend strength
            row[13] = min(atp_r.get("trend_strength", 0.0), 1.0)

            # 14-15: Zone distances (capped, normalized by ATR)
            atr_div = max(atr, 1e-8)
            dd = zone_r.get("nearest_demand_dist", float('inf'))
            sd = zone_r.get("nearest_supply_dist", float('inf'))
            row[14] = min(dd / atr_div, 10.0) / 10.0 if dd != float('inf') else 1.0
            row[15] = min(sd / atr_div, 10.0) / 10.0 if sd != float('inf') else 1.0

            # 16: Fibonacci nearest distance (FibonacciLevels object)
            if hasattr(fib_r, 'nearest_fib_dist'):
                row[16] = min(abs(fib_r.nearest_fib_dist) / atr_div, 5.0) / 5.0
            elif isinstance(fib_r, dict):
                row[16] = min(abs(fib_r.get("nearest_fib_dist", 0.0)) / atr_div, 5.0) / 5.0
            else:
                row[16] = 0.5  # neutral

            # 17: Harmonic PRZ proximity
            if isinstance(harm_r, dict):
                active = harm_r.get("active_pattern", None)
            else:
                active = getattr(harm_r, "active_pattern", None)
            row[17] = 1.0 if active is not None else 0.0

            # 18: Pivot break (StructureBreak enum → -1/0/1)
            pb = pivot_r.get("pivot_break", None)
            if pb is not None:
                if pb == StructureBreak.UP:
                    row[18] = 1.0
                elif pb == StructureBreak.DOWN:
                    row[18] = -1.0
                else:
                    row[18] = 0.0

            # 19: Momentum shift (string: "bullish"/"bearish"/"neutral")
            ms = rsi_r.get("momentum_shift", "neutral")
            row[19] = 1.0 if ms == "bullish" else (-1.0 if ms == "bearish" else 0.0)

            feature_matrix[i] = row

            if (i - self._warmup) % 5000 == 0 and i > self._warmup:
                logger.info(f"  Progress: {i}/{n} bars ({100*i/n:.1f}%)")

        # --- Generate labels ---
        logger.info("Generating binary labels (horizon=%d candles)...", self._horizon)
        labels = np.full(n, np.nan, dtype=np.float64)
        for i in range(n - self._horizon):
            labels[i] = 1.0 if closes[i + self._horizon] > closes[i] else 0.0

        # --- Filter valid rows (no NaN in features or label) ---
        valid_mask = ~np.isnan(feature_matrix).any(axis=1) & ~np.isnan(labels)
        valid_idx = np.where(valid_mask)[0]

        features_out = feature_matrix[valid_idx]
        labels_out = labels[valid_idx]
        timestamps_out = timestamps[valid_idx]

        logger.info(
            f"Pipeline complete: {len(features_out)} valid samples "
            f"(from {n} bars, warmup={self._warmup}, horizon={self._horizon})"
        )
        logger.info(
            f"Label distribution: {np.mean(labels_out):.3f} positive "
            f"({int(np.sum(labels_out))}/{len(labels_out)})"
        )

        return features_out, labels_out, timestamps_out

    @staticmethod
    def compute_scaler(
        features: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Compute normalization statistics from training data.

        Returns dict with mean/std per feature for z-score features,
        and passthrough for already-normalized features.
        """
        if feature_names is None:
            feature_names = FEATURE_NAMES

        scaler = {}
        for i, name in enumerate(feature_names):
            col = features[:, i]
            scaler[name] = {
                "mean": float(np.nanmean(col)),
                "std": float(np.nanstd(col) + 1e-8),
                "idx": i,
            }
        return scaler

    @staticmethod
    def apply_scaler(features: np.ndarray, scaler: dict) -> np.ndarray:
        """Apply z-score normalization using precomputed scaler.

        Only z-scores features that need it (OHLC, volume).
        Features already in [0,1] or [-1,1] are left as-is.
        """
        out = features.copy()
        # Z-score these columns
        zscore_cols = ["close_norm", "open_norm", "high_norm", "low_norm",
                       "volume_norm", "macd_histogram", "macd_signal"]

        for name in zscore_cols:
            if name in scaler:
                idx = scaler[name]["idx"]
                mean = scaler[name]["mean"]
                std = scaler[name]["std"]
                out[:, idx] = (out[:, idx] - mean) / std

        return out

    @staticmethod
    def save_scaler(scaler: dict, path: str) -> None:
        """Save scaler to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(scaler, f, indent=2)
        logger.info(f"Scaler saved to {path}")

    @staticmethod
    def load_scaler(path: str) -> dict:
        """Load scaler from JSON."""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_feature_schema(path: str) -> None:
        """Save feature schema for production use."""
        schema = {
            "version": 1,
            "n_features": N_FEATURES,
            "feature_names": FEATURE_NAMES,
            "prediction_horizon": 12,
            "description": "EUR/USD M5 LSTM features — 20 features from 9 indicators",
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info(f"Feature schema saved to {path}")
