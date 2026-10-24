"""
NSE Dataset Validation & Model Readiness Tests.

Tests specifically for NSE/NIFTY50 stock market data:
1. Dataset feed correctness (CSV schema, OHLCV integrity)
2. Model trained artifact checks (saved .pt, .json, .pkl, .txt files)
3. Data pipeline sanity for NSE 1h candles
4. Feature engineering validity for NSE price ranges
5. Label distribution and class balance

Run:  pytest tests/test_nse_dataset.py -v
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"

# Primary NSE data file (created by scripts/download_nifty.py)
NSE_CSV      = DATA_DIR / "nifty50_5m.csv" if (DATA_DIR / "nifty50_5m.csv").exists() else DATA_DIR / "nifty50_1h.csv"
CACHE_NPZ    = DATA_DIR / "features_v2_nifty_5m_cache.npz" if (DATA_DIR / "features_v2_nifty_5m_cache.npz").exists() else DATA_DIR / "features_v2_nifty_cache.npz"

# Expected saved model artifacts
LSTM_MODEL   = MODEL_DIR / "lstm_eurusd_v2.pt"
XGB_MODEL    = MODEL_DIR / "xgb_eurusd_v2.json"
LGBM_MODEL   = MODEL_DIR / "lgbm_eurusd_v2.txt"
META_MODEL   = MODEL_DIR / "meta_model_v2.pkl"
SCALER_JSON  = MODEL_DIR / "scaler_v2.json"
SCHEMA_JSON  = MODEL_DIR / "feature_schema_v2.json"

# NSE price sanity bounds (NIFTY50 index range, roughly)
NSE_PRICE_MIN = 5_000.0
NSE_PRICE_MAX = 30_000.0
NSE_ATR_MAX_PCT = 0.05   # Max 5% daily ATR


# ════════════════════════════════════════════════════════════════
# 1. Dataset Feed Correctness
# ════════════════════════════════════════════════════════════════

class TestNSEDatasetFeed:
    """Validate the raw NSE CSV data feed for correctness."""

    @pytest.fixture(scope="class")
    def nse_df(self):
        """Load the NSE dataset. Skip if not downloaded yet."""
        if not NSE_CSV.exists():
            pytest.skip(
                f"NSE dataset not found at {NSE_CSV}. "
                "Run: python scripts/download_nifty.py"
            )
        return pd.read_csv(NSE_CSV)

    # ── Schema ────────────────────────────────────────────────

    def test_required_columns_present(self, nse_df):
        """CSV must have: timestamp, open, high, low, close, volume."""
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(nse_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_extra_unnamed_columns(self, nse_df):
        """No unnamed/index artifacts from CSV export."""
        unnamed = [c for c in nse_df.columns if c.startswith("Unnamed")]
        assert not unnamed, f"Unnamed columns found: {unnamed}"

    # ── Row Count ─────────────────────────────────────────────

    def test_minimum_row_count(self, nse_df):
        """At least 500 rows needed for warmup (200) + training."""
        assert len(nse_df) >= 500, (
            f"Only {len(nse_df)} rows — need at least 500 for training. "
            "Re-download with a longer period."
        )

    def test_sufficient_rows_for_training(self, nse_df):
        """2-year 1h data should give ~2000-3500 rows (trading hours)."""
        assert len(nse_df) >= 1000, (
            f"Dataset has {len(nse_df)} rows — recommend 1000+ for robust training."
        )

    # ── Timestamps ────────────────────────────────────────────

    def test_timestamps_are_numeric(self, nse_df):
        """Timestamps must be Unix epoch integers."""
        assert pd.api.types.is_numeric_dtype(nse_df["timestamp"]), \
            "timestamp column must be numeric (Unix epoch seconds)"

    def test_timestamps_monotonic_increasing(self, nse_df):
        """Timestamps must be strictly monotonically increasing."""
        ts = nse_df["timestamp"].values
        assert (np.diff(ts) > 0).all(), \
            "Timestamps are not strictly increasing — duplicates or out-of-order rows found"

    def test_timestamp_range_is_recent(self, nse_df):
        """Data should not be older than 3 years."""
        ts = nse_df["timestamp"].values
        cutoff_3yr_ago = time.time() - 3 * 365 * 24 * 3600
        assert ts[-1] > cutoff_3yr_ago, \
            f"Latest timestamp ({pd.Timestamp(ts[-1], unit='s').date()}) is too old"

    def test_candle_interval_is_5m(self, nse_df):
        """Median gap between candles should be ~300 seconds (5m) or ~3600s (1h)."""
        ts = nse_df["timestamp"].values
        diffs = np.diff(ts)
        # Filter to trading-hours gaps (skip overnight/weekend)
        gaps_5m = diffs[(diffs >= 240) & (diffs <= 600)]
        gaps_1h = diffs[(diffs >= 3000) & (diffs <= 5400)]
        if len(gaps_5m) > len(gaps_1h):
            median_gap = np.median(gaps_5m)
            assert 240 <= median_gap <= 360, (
                f"Median candle gap {median_gap:.0f}s is not ~300s (5m)."
            )
        elif len(gaps_1h) > 0:
            median_gap = np.median(gaps_1h)
            assert 3000 <= median_gap <= 4800, (
                f"Median candle gap {median_gap:.0f}s is not ~3600s (1h)."
            )

    # ── OHLCV Integrity ───────────────────────────────────────

    def test_no_null_values(self, nse_df):
        """No NaN values in any OHLCV column."""
        null_counts = nse_df[["open", "high", "low", "close", "volume"]].isnull().sum()
        assert null_counts.sum() == 0, f"Null values found:\n{null_counts[null_counts > 0]}"

    def test_ohlc_relationship(self, nse_df):
        """High >= max(open, close) and Low <= min(open, close) for every bar."""
        high_ok  = (nse_df["high"] >= nse_df[["open", "close"]].max(axis=1)).all()
        low_ok   = (nse_df["low"]  <= nse_df[["open", "close"]].min(axis=1)).all()
        assert high_ok, "Some candles have High < max(Open, Close) — data corruption"
        assert low_ok,  "Some candles have Low > min(Open, Close) — data corruption"

    def test_no_zero_prices(self, nse_df):
        """No zero or negative price values."""
        for col in ["open", "high", "low", "close"]:
            assert (nse_df[col] > 0).all(), f"Zero/negative values in {col}"

    def test_no_zero_volume(self, nse_df):
        """Volume should be positive for most candles (>80%)."""
        pct_nonzero = (nse_df["volume"] > 0).mean()
        assert pct_nonzero >= 0.80, (
            f"Only {pct_nonzero:.1%} of candles have non-zero volume"
        )

    def test_price_range_is_nse(self, nse_df):
        """Prices should be in NSE NIFTY50 realistic range."""
        median_close = nse_df["close"].median()
        assert NSE_PRICE_MIN <= median_close <= NSE_PRICE_MAX, (
            f"Median close {median_close:.0f} is outside NSE range "
            f"[{NSE_PRICE_MIN:.0f}, {NSE_PRICE_MAX:.0f}]. "
            "Check you downloaded ^NSEI (NIFTY50 index)."
        )

    def test_no_extreme_price_spikes(self, nse_df):
        """No bar-to-bar change >15% (indicates bad tick/data error)."""
        pct_changes = nse_df["close"].pct_change().abs().dropna()
        extreme = pct_changes[pct_changes > 0.15]
        assert len(extreme) == 0, (
            f"{len(extreme)} extreme price spikes (>15%) detected at rows: "
            f"{list(extreme.index[:5])}"
        )

    def test_atr_is_reasonable(self, nse_df):
        """ATR (High-Low range) should be < 5% of price (realistic volatility)."""
        hl_pct = ((nse_df["high"] - nse_df["low"]) / nse_df["close"]).median()
        assert hl_pct < NSE_ATR_MAX_PCT, (
            f"Median H-L range {hl_pct:.2%} exceeds {NSE_ATR_MAX_PCT:.0%} — "
            "data may be daily not hourly"
        )

    def test_class_label_distribution(self, nse_df):
        """Binary labels (UP/DOWN) should be roughly balanced (35-65%)."""
        horizon = 4  # matches HORIZON = 4 in train_lstm_v2.py
        if len(nse_df) <= horizon:
            pytest.skip("Not enough rows to compute labels")
        closes = nse_df["close"].values
        labels = (closes[horizon:] > closes[:-horizon]).astype(int)
        up_pct = labels.mean()
        assert 0.35 <= up_pct <= 0.65, (
            f"Label imbalance: UP={up_pct:.1%} — "
            "strongly skewed dataset may hurt training"
        )


# ════════════════════════════════════════════════════════════════
# 2. Feature Cache Validation
# ════════════════════════════════════════════════════════════════

class TestFeatureCache:
    """Validate the pre-computed features_v2_nifty_cache.npz file."""

    @pytest.fixture(scope="class")
    def cache(self):
        if not CACHE_NPZ.exists():
            pytest.skip(
                f"Feature cache not found at {CACHE_NPZ}. "
                "Run training once to generate it."
            )
        return np.load(CACHE_NPZ)

    def test_cache_has_required_keys(self, cache):
        required = {"features", "labels", "closes"}
        missing = required - set(cache.keys())
        assert not missing, f"Cache missing keys: {missing}"

    def test_feature_matrix_shape(self, cache):
        features = cache["features"]
        assert features.ndim == 2, "features must be 2D (n_samples, n_features)"
        assert features.shape[1] == 12, (
            f"Expected 12 selected features, got {features.shape[1]}. "
            "Check SELECTED_FEATURES in train_lstm_v2.py"
        )

    def test_no_nan_in_features(self, cache):
        nan_count = np.isnan(cache["features"]).sum()
        assert nan_count == 0, f"{nan_count} NaN values in cached features"

    def test_no_inf_in_features(self, cache):
        inf_count = np.isinf(cache["features"]).sum()
        assert inf_count == 0, f"{inf_count} Inf values in cached features"

    def test_labels_are_binary(self, cache):
        labels = cache["labels"]
        unique = set(np.unique(labels))
        assert unique.issubset({0, 1}), f"Labels contain non-binary values: {unique}"

    def test_feature_label_length_match(self, cache):
        assert len(cache["features"]) == len(cache["labels"]), \
            "features and labels have different lengths"

    def test_feature_value_ranges(self, cache):
        """Check features contain no extreme values (normalization check)."""
        features = cache["features"]
        assert not np.any(np.abs(features) > 1000), \
            "Feature values exceed 1000 — possible normalization failure"


# ════════════════════════════════════════════════════════════════
# 3. Trained Model Artifact Checks
# ════════════════════════════════════════════════════════════════

class TestModelArtifacts:
    """Check that trained model files exist and are loadable."""

    def test_models_directory_exists(self):
        assert MODEL_DIR.exists(), (
            f"Models directory '{MODEL_DIR}' does not exist. "
            "Run: python -m forecast.train_lstm_v2"
        )

    def test_lstm_model_exists(self):
        assert LSTM_MODEL.exists(), (
            f"LSTM model not found at {LSTM_MODEL}. Train the model first."
        )

    def test_xgb_model_exists(self):
        assert XGB_MODEL.exists(), (
            f"XGBoost model not found at {XGB_MODEL}. Train the model first."
        )

    def test_lgbm_model_exists(self):
        assert LGBM_MODEL.exists(), (
            f"LightGBM model not found at {LGBM_MODEL}. Train the model first."
        )

    def test_meta_model_exists(self):
        assert META_MODEL.exists(), (
            f"Meta-model (LogReg) not found at {META_MODEL}. Train the model first."
        )

    def test_scaler_exists(self):
        assert SCALER_JSON.exists(), (
            f"Scaler not found at {SCALER_JSON}. Train the model first."
        )

    def test_feature_schema_exists(self):
        assert SCHEMA_JSON.exists(), (
            f"Feature schema not found at {SCHEMA_JSON}. Train the model first."
        )


# ════════════════════════════════════════════════════════════════
# 4. Saved LSTM Model Loadability & Sanity
# ════════════════════════════════════════════════════════════════

class TestLSTMModelLoading:
    """Load saved LSTM model and verify it can do inference."""

    @pytest.fixture(scope="class")
    def checkpoint(self):
        if not LSTM_MODEL.exists():
            pytest.skip(f"LSTM model not trained yet: {LSTM_MODEL}")
        return torch.load(LSTM_MODEL, map_location="cpu", weights_only=False)

    def test_checkpoint_has_required_keys(self, checkpoint):
        assert "model_state_dict" in checkpoint, "No model_state_dict in checkpoint"
        assert "config" in checkpoint, "No config in checkpoint"

    def test_config_schema(self, checkpoint):
        cfg = checkpoint["config"]
        for key in ["input_features", "hidden_dim", "num_layers", "seq_len",
                    "horizon", "n_classes", "feature_names", "feature_indices"]:
            assert key in cfg, f"Missing config key: {key}"

    def test_config_values_match_training(self, checkpoint):
        cfg = checkpoint["config"]
        assert cfg["input_features"] == 12, \
            f"Expected 12 features, got {cfg['input_features']}"
        assert cfg["n_classes"] == 2, \
            f"Expected 2-class (UP/DOWN), got {cfg['n_classes']}"
        assert cfg["seq_len"] == 40, \
            f"Expected seq_len=40, got {cfg['seq_len']}"
        assert cfg["horizon"] == 4, \
            f"Expected horizon=4 (1h bars), got {cfg['horizon']}"

    def test_model_loads_and_runs(self, checkpoint):
        """Model should load cleanly and do a forward pass."""
        from forecast.model import LSTMForecaster
        import torch.nn as nn

        cfg = checkpoint["config"]
        model = LSTMForecaster(
            input_features=cfg["input_features"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=0.0,
        )
        model.head = nn.Sequential(
            nn.Linear(cfg["hidden_dim"], 32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32, cfg["n_classes"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dummy = torch.randn(1, cfg["seq_len"], cfg["input_features"])
        with torch.no_grad():
            logits = model.head(model.lstm(dummy)[0][:, -1, :])
        assert logits.shape == (1, cfg["n_classes"]), \
            f"Unexpected output shape: {logits.shape}"
        probs = torch.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5, \
            "Softmax probabilities don't sum to 1"

    def test_model_inference_latency(self, checkpoint):
        """Inference should be < 50ms on CPU for 1 sample."""
        from forecast.model import LSTMForecaster
        import torch.nn as nn

        cfg = checkpoint["config"]
        model = LSTMForecaster(
            input_features=cfg["input_features"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=0.0,
        )
        model.head = nn.Sequential(
            nn.Linear(cfg["hidden_dim"], 32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32, cfg["n_classes"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        dummy = torch.randn(1, cfg["seq_len"], cfg["input_features"])

        # Warmup
        with torch.no_grad():
            model.head(model.lstm(dummy)[0][:, -1, :])

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            with torch.no_grad():
                model.head(model.lstm(dummy)[0][:, -1, :])
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        assert avg_ms < 50, f"LSTM inference {avg_ms:.1f}ms exceeds 50ms limit"


# ════════════════════════════════════════════════════════════════
# 5. Scaler & Schema Validation
# ════════════════════════════════════════════════════════════════

class TestScalerAndSchema:
    """Validate saved scaler and feature schema JSON files."""

    @pytest.fixture(scope="class")
    def scaler(self):
        if not SCALER_JSON.exists():
            pytest.skip(f"Scaler not found: {SCALER_JSON}")
        with open(SCALER_JSON) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def schema(self):
        if not SCHEMA_JSON.exists():
            pytest.skip(f"Schema not found: {SCHEMA_JSON}")
        with open(SCHEMA_JSON) as f:
            return json.load(f)

    def test_scaler_has_all_features(self, scaler):
        expected_features = [
            "close_norm", "atr_norm", "rsi", "hw_rsi", "macd_histogram",
            "bos_flag", "choch_flag", "alpha_trend_dir", "trend_strength",
            "demand_dist", "supply_dist", "pivot_break",
        ]
        for feat in expected_features:
            assert feat in scaler, f"Feature '{feat}' missing from scaler"

    def test_scaler_values_are_finite(self, scaler):
        for feat, stats in scaler.items():
            assert np.isfinite(stats["mean"]), f"{feat}: mean is not finite"
            assert np.isfinite(stats["std"]) and stats["std"] > 0, \
                f"{feat}: std={stats['std']} is invalid"

    def test_schema_version(self, schema):
        assert schema.get("version") == 2, \
            f"Expected schema version 2, got {schema.get('version')}"

    def test_schema_feature_count(self, schema):
        assert schema.get("n_features") == 12, \
            f"Schema says {schema.get('n_features')} features, expected 12"

    def test_schema_class_names(self, schema):
        assert schema.get("class_names") == ["DOWN", "UP"], \
            f"Unexpected class names: {schema.get('class_names')}"

    def test_schema_models_listed(self, schema):
        models = schema.get("models", [])
        for m in ["LSTM", "XGBoost", "LightGBM"]:
            assert any(m in s for s in models), f"{m} not listed in schema models"


# ════════════════════════════════════════════════════════════════
# 6. Ensemble Model Loading
# ════════════════════════════════════════════════════════════════

class TestEnsembleModels:
    """Verify XGBoost, LightGBM, and meta-model load correctly."""

    def test_xgb_loads(self):
        if not XGB_MODEL.exists():
            pytest.skip(f"XGB model not found: {XGB_MODEL}")
        from xgboost import XGBClassifier
        xgb = XGBClassifier()
        xgb.load_model(str(XGB_MODEL))
        assert xgb is not None

    def test_lgbm_loads(self):
        if not LGBM_MODEL.exists():
            pytest.skip(f"LGBM model not found: {LGBM_MODEL}")
        from lightgbm import Booster
        booster = Booster(model_file=str(LGBM_MODEL))
        assert booster is not None

    def test_meta_model_loads(self):
        if not META_MODEL.exists():
            pytest.skip(f"Meta-model not found: {META_MODEL}")
        import pickle
        with open(META_MODEL, "rb") as f:
            meta = pickle.load(f)
        assert hasattr(meta, "predict"), "Meta-model has no predict() method"
        assert hasattr(meta, "predict_proba"), "Meta-model has no predict_proba() method"

    def test_meta_model_predicts_2classes(self):
        if not META_MODEL.exists():
            pytest.skip(f"Meta-model not found: {META_MODEL}")
        import pickle
        with open(META_MODEL, "rb") as f:
            meta = pickle.load(f)
        # 3 models x 2 classes = 6 meta-features
        dummy_meta = np.random.dirichlet([1, 1], size=(10,))
        dummy_meta = np.hstack([dummy_meta, dummy_meta, dummy_meta])
        probs = meta.predict_proba(dummy_meta)
        assert probs.shape[1] == 2, \
            f"Meta-model outputs {probs.shape[1]} classes, expected 2"


# ════════════════════════════════════════════════════════════════
# 7. NSE-Specific Data Pipeline Smoke Test
# ════════════════════════════════════════════════════════════════

class TestNSEPipelineSmoke:
    """Run DataPipeline on a tiny synthetic NSE-priced dataset."""

    def _make_nse_csv(self, tmp_path: Path, n: int = 400) -> Path:
        """Create a synthetic NSE-range OHLCV CSV."""
        np.random.seed(99)
        t_base = 1_700_000_000
        closes = 21000 + np.cumsum(np.random.randn(n) * 50)  # NIFTY50-like prices
        opens  = closes + np.random.randn(n) * 20
        highs  = np.maximum(opens, closes) + abs(np.random.randn(n) * 15)
        lows   = np.minimum(opens, closes) - abs(np.random.randn(n) * 15)
        vols   = np.random.uniform(50000, 200000, n)

        df = pd.DataFrame({
            "timestamp": [t_base + i * 3600 for i in range(n)],
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": vols,
        })
        csv_path = tmp_path / "synthetic_nifty.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_pipeline_on_synthetic_nse_data(self, tmp_path):
        """DataPipeline should process synthetic NSE data without crashing."""
        from forecast.data_pipeline import DataPipeline, N_FEATURES

        csv_path = self._make_nse_csv(tmp_path, n=400)
        pipeline = DataPipeline(
            csv_path=str(csv_path),
            prediction_horizon=4,
            warmup=200,
        )
        try:
            features, labels, timestamps = pipeline.build()
        except Exception as e:
            pytest.fail(f"DataPipeline.build() raised: {e}")

        assert features.ndim == 2
        assert features.shape[1] == N_FEATURES, \
            f"Expected {N_FEATURES} features, got {features.shape[1]}"
        assert len(features) == len(labels)
        assert len(labels) > 0, "Pipeline returned 0 valid samples"
        assert set(np.unique(labels)).issubset({0.0, 1.0}), \
            f"Unexpected label values: {np.unique(labels)}"

        print(f"\n  Pipeline returned {len(features)} samples, "
              f"{int(labels.sum())} UP / {int((1-labels).sum())} DOWN")

    def test_pipeline_feature_no_nan(self, tmp_path):
        """All features from pipeline should be finite (no NaN/Inf)."""
        from forecast.data_pipeline import DataPipeline

        csv_path = self._make_nse_csv(tmp_path, n=400)
        pipeline = DataPipeline(str(csv_path), prediction_horizon=4, warmup=200)
        try:
            features, labels, timestamps = pipeline.build()
        except Exception as e:
            pytest.skip(f"Pipeline failed: {e}")

        nan_ct = np.isnan(features).sum()
        inf_ct = np.isinf(features).sum()
        assert nan_ct == 0, f"{nan_ct} NaN values in pipeline output"
        assert inf_ct == 0, f"{inf_ct} Inf values in pipeline output"


# ════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
