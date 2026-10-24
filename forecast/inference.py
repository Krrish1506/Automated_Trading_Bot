"""
Low-Latency Inference Engine for the AI Forecast Model.

Loads model weights into memory at startup and performs sub-5ms
inference using JIT-compiled model. Subscribes to INDICATORS_READY
and publishes FORECAST_READY events.
"""
from __future__ import annotations

import time
import logging
import os

import torch

from core.event_bus import EventBus
from core.candle_buffer import CandleBuffer
from core.events import (
    IndicatorsReadyEvent, ForecastReadyEvent, EventType,
)
from forecast.model import LSTMForecaster, TransformerForecaster
from forecast.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Low-latency model inference engine.

    Loads the forecast model at startup (JIT-compiled for speed)
    and runs inference on each indicator update. Target: < 5ms latency.
    """

    def __init__(
        self,
        event_bus: EventBus,
        candle_buffer: CandleBuffer,
        config: dict,
    ):
        self._bus = event_bus
        self._buffer = candle_buffer
        self._cfg = config.get("forecast", {})

        self._model_type = self._cfg.get("model_type", "lstm")
        self._seq_len = self._cfg.get("sequence_length", 30)
        self._n_features = self._cfg.get("input_features", 12)
        self._hidden_dim = self._cfg.get("hidden_dim", 128)
        self._num_layers = self._cfg.get("num_layers", 2)
        self._dropout = self._cfg.get("dropout", 0.2)
        self._model_path = self._cfg.get("model_path", "models/lstm_eurusd.pt")
        self._buy_threshold = self._cfg.get("probability_threshold_buy", 0.65)
        self._sell_threshold = self._cfg.get("probability_threshold_sell", 0.35)

        self._n_classes = 1
        # Build model
        self._model = self._build_model()
        self._model.eval()

        # Try JIT compilation for speed
        self._jit_model = None

        # Feature builder
        self._feature_builder = FeatureBuilder(
            sequence_length=self._seq_len,
            input_features=self._n_features,
        )

        # Load pretrained weights if available
        self._load_weights()

        # Stats
        self._last_inference_ms: float = 0.0
        self._inference_count: int = 0

        # Subscribe
        self._bus.subscribe(EventType.INDICATORS_READY, self._on_indicators_ready)

    def _build_model(self):
        """Build the forecast model based on config."""
        if self._model_type == "transformer":
            return TransformerForecaster(
                input_features=self._n_features,
                d_model=self._hidden_dim,
                nhead=4,
                num_layers=self._num_layers,
                dropout=self._dropout,
                max_seq_len=self._seq_len,
            )
        else:
            return LSTMForecaster(
                input_features=self._n_features,
                hidden_dim=self._hidden_dim,
                num_layers=self._num_layers,
                dropout=self._dropout,
            )

    def _try_jit_compile(self) -> None:
        """Try to JIT compile the model for faster inference."""
        try:
            dummy = torch.randn(1, self._seq_len, self._n_features)
            self._jit_model = torch.jit.trace(self._model, dummy)
            logger.info("Model JIT-compiled successfully")
        except Exception as exc:
            logger.warning(f"JIT compilation failed, using eager mode: {exc}")
            self._jit_model = None

    def _load_weights(self) -> None:
        """Load pretrained model weights if checkpoint exists."""
        if os.path.exists(self._model_path):
            try:
                checkpoint = torch.load(
                    self._model_path,
                    map_location="cpu",
                    weights_only=False,
                )
                # Handle both old (raw state_dict) and new (full checkpoint) formats
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    cfg = checkpoint.get("config", {})
                    if "n_classes" in cfg and cfg["n_classes"] > 1:
                        import torch.nn as nn
                        self._n_classes = cfg["n_classes"]
                        hidden = cfg.get("hidden_dim", self._hidden_dim)
                        self._model.head = nn.Sequential(
                            nn.Linear(hidden, 32),
                            nn.ReLU(),
                            nn.Dropout(0.0),
                            nn.Linear(32, self._n_classes),
                        )
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(
                        f"Loaded model v2 from {self._model_path} "
                        f"(features={cfg.get('input_features')}, "
                        f"hidden={cfg.get('hidden_dim')}, classes={self._n_classes})"
                    )
                else:
                    self._model.load_state_dict(checkpoint)
                    logger.info(f"Loaded model weights from {self._model_path}")

                # Re-JIT after loading weights
                self._try_jit_compile()
            except Exception as exc:
                logger.warning(f"Failed to load weights: {exc}")
        else:
            logger.warning(
                f"No model checkpoint at {self._model_path}, "
                "using random weights (train required)"
            )

    async def _on_indicators_ready(self, event: IndicatorsReadyEvent) -> None:
        """Run inference when indicators are ready."""
        t0 = time.perf_counter()

        # Build feature tensor
        features = self._feature_builder.build(self._buffer, event.state)
        if features is None:
            return

        # Run inference
        probability = self._predict(features)

        self._last_inference_ms = (time.perf_counter() - t0) * 1000
        self._inference_count += 1

        # Determine direction
        if probability > self._buy_threshold:
            direction = "up"
        elif probability < self._sell_threshold:
            direction = "down"
        else:
            direction = "neutral"

        logger.info(
            f"Forecast: prob={probability:.4f} dir={direction} "
            f"latency={self._last_inference_ms:.2f}ms"
        )

        await self._bus.publish(ForecastReadyEvent(
            timestamp=event.timestamp,
            probability=probability,
            direction=direction,
        ))

    @torch.no_grad()
    def _predict(self, features: torch.Tensor) -> float:
        """Run model inference. Returns UP probability 0-1."""
        model = self._jit_model if self._jit_model is not None else self._model
        logits = model(features)
        if self._n_classes > 1:
            prob = torch.softmax(logits, dim=1)[:, 1]
        else:
            prob = torch.sigmoid(logits)
        return float(prob.item())

    @property
    def stats(self) -> dict:
        return {
            "model_type": self._model_type,
            "inference_count": self._inference_count,
            "last_latency_ms": round(self._last_inference_ms, 2),
            "jit_compiled": self._jit_model is not None,
        }
