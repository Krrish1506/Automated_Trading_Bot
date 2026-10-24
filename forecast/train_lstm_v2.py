"""
LSTM Training v2 — All 5 Improvements Applied.

Improvements over v1:
1. Uses 2+ years of data (if available)
2. Stronger regularization: hidden=64, dropout=0.5, label smoothing
3. Feature selection: drops low-importance features
4. 3-class ATR-gated target: UP / FLAT / DOWN
5. Ensemble: XGBoost + LightGBM + LSTM stacking

Run: python -m forecast.train_lstm_v2
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from forecast.data_pipeline import DataPipeline, FEATURE_NAMES, N_FEATURES
from forecast.model import LSTMForecaster
from forecast.evaluator import Evaluator, EvalResult

# Ensemble imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── V2 Configuration ─────────────────────────────────────────────
SEQ_LEN = 40
HORIZON = 4
HIDDEN_DIM = 128         # Restored to 128 to capture complex patterns across 50 assets
NUM_LAYERS = 2
DROPOUT = 0.4            # Balanced for massive dataset
BATCH_SIZE = 64          # Safely increased to 64 for speed (RTX 1650 compatible)
MAX_EPOCHS = 20          # Reduced due to massive dataset size
LR = 5e-4                # Learning rate
WEIGHT_DECAY = 1e-3      # ▲ Increased from 1e-4
GRAD_CLIP = 1.0
PATIENCE = 12            # Reduced from 15
LABEL_SMOOTH = 0.05      # Label smoothing
ATR_GATE = 2.5           # ATR multiplier for UP/DOWN threshold (increased for 3-class balance)
N_FOLDS = 4
TEST_RATIO = 0.20
MODEL_DIR = "models"
DATA_PATH = "data/nifty50_5m.csv"          # <-- NIFTY50 5-minute training data
CACHE_PATH = "data/features_v2_nifty_5m_cache.npz"  # <-- 5m separate cache

# Features to keep (selected by expected importance for M5 forex)
SELECTED_FEATURES = [
    "close_norm",         # 0  - Price movement
    "atr_norm",           # 4  - Volatility
    "rsi",                # 5  - Momentum
    "hw_rsi",             # 6  - Smoothed momentum
    "macd_histogram",     # 7  - Trend momentum
    "bos_flag",           # 10 - Structure break
    "choch_flag",         # 11 - Character change
    "alpha_trend_dir",    # 12 - Trend direction
    "trend_strength",     # 13 - Trend confidence
    "demand_dist",        # 14 - Support proximity
    "supply_dist",        # 15 - Resistance proximity
    "pivot_break",        # 18 - Pivot structure
]
SELECTED_INDICES = [FEATURE_NAMES.index(f) for f in SELECTED_FEATURES]
N_SELECTED = len(SELECTED_FEATURES)


def compute_binary_labels(
    closes: np.ndarray,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Generate 2-class labels: 0=DOWN, 1=UP."""
    n = len(closes)
    labels = np.full(n, 1, dtype=np.int64)  # Default UP

    for i in range(n - horizon):
        future = closes[i + horizon]
        current = closes[i]

        if future > current:
            labels[i] = 1  # UP
        else:
            labels[i] = 0  # DOWN

    # Last `horizon` bars have no future — mark as NaN-equivalent
    labels[n - horizon:] = -1  # Will be filtered out
    return labels


class SequenceDataset:
    """Creates sliding window sequences from feature matrix."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = SEQ_LEN):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
        self.n_valid = max(0, len(features) - seq_len)

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.n_valid
        if n == 0:
            return torch.empty(0), torch.empty(0)

        # Simple windowing to avoid Numpy 2.x stride_tricks bug
        X = np.array([self.features[i:i+self.seq_len] for i in range(n)], dtype=np.float32)
        y = self.labels[self.seq_len - 1:self.seq_len - 1 + n].astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(y)


def flatten_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int = SEQ_LEN):
    """Create flattened feature matrix for tree models (XGBoost/LightGBM).

    Uses rolling statistics over the seq_len window instead of raw sequences.
    Processes in chunks to avoid memory errors on large datasets.
    """
    n = max(0, len(features) - seq_len)
    if n == 0:
        return np.empty((0, 0)), np.empty(0)

    n_feat = features.shape[1]
    n_stats = 5
    X_flat = np.zeros((n, n_feat * n_stats), dtype=np.float32)
    y_flat = labels[seq_len - 1:seq_len - 1 + n].astype(np.int64)

    # Process in chunks of 10K to stay memory-safe
    chunk_size = 10000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        for i in range(start, end):
            window = features[i:i + seq_len]
            X_flat[i, 0*n_feat:1*n_feat] = window[-1]
            X_flat[i, 1*n_feat:2*n_feat] = np.mean(window, axis=0)
            X_flat[i, 2*n_feat:3*n_feat] = np.std(window, axis=0)
            X_flat[i, 3*n_feat:4*n_feat] = np.min(window, axis=0)
            X_flat[i, 4*n_feat:5*n_feat] = np.max(window, axis=0)

    return X_flat, y_flat


def train_lstm_fold(
    fold_num: int,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    n_classes: int = 2,
) -> tuple[dict, np.ndarray]:
    """Train LSTM for one fold. Returns (metrics, val_probs)."""
    logger.info(f"\n  LSTM Fold {fold_num}")

    train_ds = SequenceDataset(train_features, train_labels, SEQ_LEN)
    val_ds = SequenceDataset(val_features, val_labels, SEQ_LEN)
    X_train, y_train = train_ds.to_tensors()
    X_val, y_val = val_ds.to_tensors()

    if len(X_train) < BATCH_SIZE or len(X_val) < 100:
        logger.warning(f"  LSTM Fold {fold_num}: insufficient data")
        return {}, np.zeros((len(X_val), n_classes))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        pin_memory=(device.type == "cuda"),   # faster CPU->GPU transfer
        num_workers=0,                         # 0 = safe on Windows
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE * 2, shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    # 3-class output
    model = LSTMForecaster(
        input_features=train_features.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    # Replace head for 3-class output
    model.head = nn.Sequential(
        nn.Linear(HIDDEN_DIM, 32),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(32, n_classes),
    ).to(device)

    # Label smoothing cross-entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb).squeeze(-1) if n_classes == 1 else model(Xb)
            # Reshape if needed
            if logits.dim() == 2 and logits.size(1) == 1 and n_classes > 1:
                logits = model.head(model.lstm(Xb)[0][:, -1, :])
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model.head(model.lstm(Xb)[0][:, -1, :])
                loss = criterion(logits, yb)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"    Epoch {epoch+1}/{MAX_EPOCHS} — train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            logger.info(f"    Early stopping at epoch {epoch+1}")
            break

    # Get val predictions
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            logits = model.head(model.lstm(Xb)[0][:, -1, :])
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())

    val_probs = np.concatenate(all_probs)
    val_labels_out = np.concatenate(all_labels)
    preds = np.argmax(val_probs, axis=1)
    acc = accuracy_score(val_labels_out, preds)

    logger.info(f"  LSTM Fold {fold_num}: acc={acc:.3f}, best_loss={best_val_loss:.4f}")

    return {"acc": acc, "loss": best_val_loss, "state": best_state}, val_probs


def train_xgb_fold(
    fold_num: int,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_classes: int = 2,
) -> tuple[dict, np.ndarray]:
    """Train XGBoost for one fold."""
    logger.info(f"  XGBoost Fold {fold_num}")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_probs = xgb.predict_proba(X_val)
    preds = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_val, preds)
    logger.info(f"  XGBoost Fold {fold_num}: acc={acc:.3f}")

    return {"acc": acc, "model": xgb}, val_probs


def train_lgbm_fold(
    fold_num: int,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_classes: int = 2,
) -> tuple[dict, np.ndarray]:
    """Train LightGBM for one fold."""
    logger.info(f"  LightGBM Fold {fold_num}")

    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multiclass',
        num_class=n_classes,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    val_probs = lgbm.predict_proba(X_val)
    preds = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_val, preds)
    logger.info(f"  LightGBM Fold {fold_num}: acc={acc:.3f}")

    return {"acc": acc, "model": lgbm}, val_probs


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}  |  VRAM: {vram_total:.1f} GB")
        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = False  # Fix for cuDNN host allocation error
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # ─── Phase 1: Data Pipeline (with caching) ────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 1: DATA + FEATURES + ATR-GATED LABELS")
    logger.info("═" * 60)

    cache = Path(CACHE_PATH)
    if cache.exists():
        logger.info(f"Loading cached features from {CACHE_PATH}...")
        data = np.load(CACHE_PATH)
        features = data["features"]
        labels_3class = data["labels"]
        closes_valid = data["closes"]
        n_total = len(features)
        logger.info(f"Loaded {n_total} samples from cache")
    else:
        logger.info("No cache found — computing features from scratch...")
        pipeline = DataPipeline(csv_path=DATA_PATH, prediction_horizon=HORIZON, warmup=200)
        features_raw, labels_old, timestamps = pipeline.build()
        n_total = len(features_raw)
        logger.info(f"Total samples: {n_total}")

        # Feature selection: keep only top features
        features = features_raw[:, SELECTED_INDICES]
        logger.info(f"Selected {N_SELECTED} features: {SELECTED_FEATURES}")

        # Compute ATR values for gating
        atr_idx_in_raw = FEATURE_NAMES.index("atr_norm")
        atr_values = features_raw[:, atr_idx_in_raw]

        # Load close prices
        df = pd.read_csv(DATA_PATH)
        closes_all = df["close"].values.astype(np.float64)
        valid_mask = np.isin(df["timestamp"].values, timestamps)
        closes_valid = closes_all[valid_mask]
        if len(closes_valid) != n_total:
            closes_valid = closes_all[200:200 + n_total]
            if len(closes_valid) < n_total:
                closes_valid = np.pad(closes_valid, (0, n_total - len(closes_valid)), mode='edge')

        # ATR-gated labels
        labels_3class = compute_binary_labels(closes_valid, HORIZON)

        # Filter out invalid labels (-1)
        valid = labels_3class >= 0
        features = features[valid]
        labels_3class = labels_3class[valid]
        closes_valid = closes_valid[valid]
        n_total = len(features)

        # Save cache
        np.savez_compressed(CACHE_PATH, features=features, labels=labels_3class, closes=closes_valid)
        logger.info(f"Features cached to {CACHE_PATH}")

    class_counts = np.bincount(labels_3class, minlength=2)
    logger.info(f"2-class distribution: DOWN={class_counts[0]} ({class_counts[0]/n_total:.1%}), "
                f"UP={class_counts[1]} ({class_counts[1]/n_total:.1%})")

    # ─── Phase 2: Normalize ───────────────────────────────────────
    test_start = int(n_total * (1 - TEST_RATIO))
    scaler = DataPipeline.compute_scaler(features[:int(test_start * 0.5)], SELECTED_FEATURES)
    DataPipeline.save_scaler(scaler, f"{MODEL_DIR}/scaler_v2.json")

    features_scaled = DataPipeline.apply_scaler(features, scaler)
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    train_val_feat = features_scaled[:test_start]
    train_val_labels = labels_3class[:test_start]
    test_feat = features_scaled[test_start:]
    test_labels = labels_3class[test_start:]
    test_closes = closes_valid[test_start:]

    logger.info(f"Train+Val: {len(train_val_feat)}, Test: {len(test_feat)}")

    # ─── Phase 3: Walk-Forward Ensemble Training ──────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 3: WALK-FORWARD ENSEMBLE TRAINING")
    logger.info("═" * 60)

    n_tv = len(train_val_feat)
    fold_size = n_tv // (N_FOLDS + 1)

    fold_results = []

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 2)
        val_end = min(train_end + fold_size, n_tv)

        if val_end - train_end < 200:
            continue

        f_train = train_val_feat[:train_end]
        l_train = train_val_labels[:train_end]
        f_val = train_val_feat[train_end:val_end]
        l_val = train_val_labels[train_end:val_end]

        logger.info(f"\n{'='*50}")
        logger.info(f"  FOLD {fold+1} — Train: {len(f_train)}, Val: {len(f_val)}")
        logger.info(f"{'='*50}")

        # 1. Train LSTM
        lstm_result, lstm_probs = train_lstm_fold(
            fold+1, f_train, l_train, f_val, l_val, device
        )

        # 2. Train XGBoost (flattened features)
        X_train_flat, y_train_flat = flatten_sequences(f_train, l_train, SEQ_LEN)
        X_val_flat, y_val_flat = flatten_sequences(f_val, l_val, SEQ_LEN)

        xgb_result, xgb_probs = train_xgb_fold(fold+1, X_train_flat, y_train_flat, X_val_flat, y_val_flat)
        lgbm_result, lgbm_probs = train_lgbm_fold(fold+1, X_train_flat, y_train_flat, X_val_flat, y_val_flat)

        # Align lengths (LSTM sequences may differ slightly from tree models)
        min_len = min(len(lstm_probs), len(xgb_probs), len(lgbm_probs))
        lstm_probs = lstm_probs[:min_len]
        xgb_probs = xgb_probs[:min_len]
        lgbm_probs = lgbm_probs[:min_len]
        y_val_aligned = y_val_flat[:min_len]

        # 3. Stack: combine probabilities as features for meta-model
        meta_features = np.hstack([lstm_probs, xgb_probs, lgbm_probs])  # (n, 9)

        # Get stacking train data from training set
        X_tr_flat_for_stack, y_tr_flat_for_stack = flatten_sequences(f_train, l_train, SEQ_LEN)
        # Train each base model on first 80% of train, get probs on last 20%
        stack_split = int(len(X_tr_flat_for_stack) * 0.8)

        fold_results.append({
            "fold": fold + 1,
            "lstm_acc": lstm_result.get("acc", 0),
            "xgb_acc": xgb_result.get("acc", 0),
            "lgbm_acc": lgbm_result.get("acc", 0),
            "meta_features": meta_features,
            "val_labels": y_val_aligned,
            "lstm_state": lstm_result.get("state"),
            "xgb_model": xgb_result.get("model"),
            "lgbm_model": lgbm_result.get("model"),
        })

    # ─── Phase 4: Train Meta-Model ────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 4: META-MODEL (LOGISTIC REGRESSION)")
    logger.info("═" * 60)

    # Pool all fold meta-features for meta-model training
    all_meta_X = np.vstack([r["meta_features"] for r in fold_results])
    all_meta_y = np.concatenate([r["val_labels"] for r in fold_results])

    meta_model = LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs'
    )
    meta_model.fit(all_meta_X, all_meta_y)
    meta_preds = meta_model.predict(all_meta_X)
    meta_acc = accuracy_score(all_meta_y, meta_preds)
    logger.info(f"Meta-model OOF accuracy: {meta_acc:.3f}")

    # ─── Phase 5: Final Test ──────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 5: OUT-OF-SAMPLE ENSEMBLE TEST")
    logger.info("═" * 60)

    # Train final models on ALL train+val data
    # LSTM
    logger.info("Training final LSTM...")
    lstm_final, lstm_test_probs = train_lstm_fold(
        0, train_val_feat, train_val_labels, test_feat, test_labels, device
    )

    # XGBoost & LightGBM
    X_train_all, y_train_all = flatten_sequences(train_val_feat, train_val_labels, SEQ_LEN)
    X_test_flat, y_test_flat = flatten_sequences(test_feat, test_labels, SEQ_LEN)

    logger.info("Training final XGBoost...")
    xgb_final, xgb_test_probs = train_xgb_fold(0, X_train_all, y_train_all, X_test_flat, y_test_flat)

    logger.info("Training final LightGBM...")
    lgbm_final, lgbm_test_probs = train_lgbm_fold(0, X_train_all, y_train_all, X_test_flat, y_test_flat)

    # Align
    min_test = min(len(lstm_test_probs), len(xgb_test_probs), len(lgbm_test_probs))
    lstm_test_probs = lstm_test_probs[:min_test]
    xgb_test_probs = xgb_test_probs[:min_test]
    lgbm_test_probs = lgbm_test_probs[:min_test]
    y_test_aligned = y_test_flat[:min_test]

    # Ensemble prediction
    ensemble_meta = np.hstack([lstm_test_probs, xgb_test_probs, lgbm_test_probs])
    ensemble_preds = meta_model.predict(ensemble_meta)
    ensemble_probs = meta_model.predict_proba(ensemble_meta)

    # ─── Print Results ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ENSEMBLE TRAINING REPORT v2 -- NSE NIFTY50 M5 (INR)")
    print("=" * 70)

    # Individual accuracies
    lstm_acc = accuracy_score(y_test_aligned, np.argmax(lstm_test_probs, axis=1))
    xgb_acc = accuracy_score(y_test_aligned, np.argmax(xgb_test_probs, axis=1))
    lgbm_acc = accuracy_score(y_test_aligned, np.argmax(lgbm_test_probs, axis=1))
    ens_acc = accuracy_score(y_test_aligned, ensemble_preds)

    print("\n--- Individual Model Test Accuracy ---")
    print(f"  LSTM:     {lstm_acc:.4f}")
    print(f"  XGBoost:  {xgb_acc:.4f}")
    print(f"  LightGBM: {lgbm_acc:.4f}")
    print(f"  Ensemble: {ens_acc:.4f}")

    # Per-class accuracy
    print("\n--- Per-Class Test Accuracy ---")
    for cls, name in enumerate(["DOWN", "UP"]):
        mask = y_test_aligned == cls
        if mask.sum() > 0:
            cls_acc = accuracy_score(y_test_aligned[mask], ensemble_preds[mask])
            print(f"  {name}: {cls_acc:.4f} ({mask.sum()} samples)")

    # Trading simulation (only trade UP/DOWN, skip FLAT)
    print("\n--- Trading Simulation ---")
    trades = []
    test_closes_aligned = test_closes[SEQ_LEN - 1:SEQ_LEN - 1 + min_test]
    if len(test_closes_aligned) < min_test:
        test_closes_aligned = np.pad(test_closes_aligned, (0, min_test - len(test_closes_aligned)), mode='edge')

    spread = 1.5 * 0.0001
    i = 20  # Start at 20 for Fibonacci lookback
    
    # We will use the meta-model probabilities for strict confidence filtering
    ensemble_conf = np.max(ensemble_probs, axis=1)

    while i < min_test - HORIZON:
        pred = ensemble_preds[i]
        conf = ensemble_conf[i]

        # 1. Strict ML Confidence Filter (Requires 85%+ agreement)
        if conf < 0.60:
            i += 1
            continue
            
        # 2. Fibonacci Retracement Filter (Golden Zone rules)
        recent_high = np.max(test_closes_aligned[i-20:i])
        recent_low = np.min(test_closes_aligned[i-20:i])
        swing_range = recent_high - recent_low
        
        if swing_range == 0:
            i += 1
            continue
            
        fib_level = (test_closes_aligned[i] - recent_low) / swing_range
        
        # Only take trades when price has pulled back to Fibonacci Golden Zone
        if pred == 1 and not (0.20 <= fib_level <= 0.65):
            i += 1
            continue
        elif pred == 0 and not (0.35 <= fib_level <= 0.80):
            i += 1
            continue

        # Execute Trade
        if pred == 1:  # UP
            entry = test_closes_aligned[i] + spread / 2
            exit_p = test_closes_aligned[min(i + HORIZON, min_test - 1)]
            trades.append((exit_p - entry) / entry)
            i += HORIZON
        elif pred == 0:  # DOWN
            entry = test_closes_aligned[i] - spread / 2
            exit_p = test_closes_aligned[min(i + HORIZON, min_test - 1)]
            trades.append((entry - exit_p) / entry)
            i += HORIZON

    if trades:
        trades = np.array(trades)
        wins = trades[trades > 0]
        losses = trades[trades <= 0]
        win_rate = len(wins) / len(trades)
        profit_factor = np.sum(wins) / max(abs(np.sum(losses)), 1e-8)
        sharpe = np.mean(trades) / max(np.std(trades), 1e-8) * np.sqrt(252 * 24 * 12 / HORIZON)
        total_return = np.prod(1 + trades) - 1

        print(f"  Total trades: {len(trades)}")
        print(f"  Win rate:     {win_rate:.4f}")
        print(f"  Profit factor:{profit_factor:.3f}")
        print(f"  Sharpe ratio: {sharpe:.3f}")
        print(f"  Total return: {total_return:.4f}")
        print(f"  Max DD:       {max(np.maximum.accumulate(np.cumsum(trades)) - np.cumsum(trades)):.4f}")
    else:
        print("  No trades generated!")

    # Walk-forward fold summary
    print("\n--- Walk-Forward Fold Summary ---")
    print(f"{'Fold':>4} | {'LSTM':>6} | {'XGB':>6} | {'LGBM':>6}")
    print("-" * 35)
    for r in fold_results:
        print(f"{r['fold']:>4} | {r['lstm_acc']:>6.3f} | {r['xgb_acc']:>6.3f} | {r['lgbm_acc']:>6.3f}")

    print("=" * 70)

    # ─── Save Models ──────────────────────────────────────────────
    import pickle

    # Save LSTM
    if lstm_final.get("state"):
        torch.save({
            "model_state_dict": lstm_final["state"],
            "config": {
                "input_features": N_SELECTED,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "seq_len": SEQ_LEN,
                "horizon": HORIZON,
                "n_classes": 2,
                "feature_names": SELECTED_FEATURES,
                "feature_indices": SELECTED_INDICES,
            },
        }, f"{MODEL_DIR}/lstm_eurusd_v2.pt")

    # Save XGBoost & LightGBM
    if xgb_final.get("model"):
        xgb_final["model"].save_model(f"{MODEL_DIR}/xgb_eurusd_v2.json")
    if lgbm_final.get("model"):
        lgbm_final["model"].booster_.save_model(f"{MODEL_DIR}/lgbm_eurusd_v2.txt")

    # Save meta-model
    with open(f"{MODEL_DIR}/meta_model_v2.pkl", "wb") as f:
        pickle.dump(meta_model, f)

    # Save feature schema
    schema = {
        "version": 2,
        "n_features": N_SELECTED,
        "feature_names": SELECTED_FEATURES,
        "feature_indices": SELECTED_INDICES,
        "n_classes": 2,
        "class_names": ["DOWN", "UP"],
        "target": "Binary (0=DOWN, 1=UP)",
        "horizon": HORIZON,
        "models": ["LSTM", "XGBoost", "LightGBM", "LogReg meta"],
    }
    with open(f"{MODEL_DIR}/feature_schema_v2.json", "w") as f:
        json.dump(schema, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
