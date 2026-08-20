"""
LSTM Forecasting Model — Direction Prediction.

Compact LSTM network for predicting directional probability
10-15 candles ahead. Designed for < 5ms inference on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """LSTM model for directional price prediction.

    Input: sequence of multi-factor feature vectors
    Output: raw logits (use BCEWithLogitsLoss for training,
            torch.sigmoid() for inference probabilities)

    Architecture kept compact for sub-5ms inference:
    - 2 LSTM layers with hidden_dim=128
    - Dropout between layers
    - FC head → logit
    """

    def __init__(
        self,
        input_features: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_features) tensor.

        Returns:
            (batch, 1) logit tensor.
        """
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probabilities (sigmoid applied)."""
        return torch.sigmoid(self.forward(x))


class TransformerForecaster(nn.Module):
    """Transformer-based forecaster (alternative to LSTM).

    Uses positional encoding and multi-head self-attention
    for capturing complex temporal dependencies.
    """

    def __init__(
        self,
        input_features: int = 20,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_features, d_model)

        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.01
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_features) tensor.

        Returns:
            (batch, 1) logit tensor.
        """
        seq_len = x.size(1)
        x = self.input_proj(x) + self.pos_encoding[:, :seq_len, :]
        encoded = self.encoder(x)
        last = encoded[:, -1, :]
        return self.head(last)
