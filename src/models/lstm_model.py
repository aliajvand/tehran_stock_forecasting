"""مدل LSTM برای طبقه‌بندی روند سه‌کلاسه (up/down/neutral)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class LSTMPredictor(nn.Module):
    """
    LSTM دولایه + لایه Dense برای طبقه‌بندی ۳ کلاسه.

    ورودی: ``(batch, seq_len, n_features)``
    خروجی: ``(logits (batch, 3), expected_return (batch, 1))``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        """پارامترها از بخش ``model.lstm`` فایل config می‌آیند."""
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
        self.return_head = nn.Linear(hidden_size, 1)  # بازدهی تخمینی

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """کدگذاری دنباله و برگرداندن بردار حالت آخرین گام زمانی."""
        output, _ = self.lstm(x)
        return self.dropout(self.norm(output[:, -1, :]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """عبور رو به جلو: خروجی logits و بازدهی تخمینی."""
        latent = self.encode(x)
        return self.head(latent), self.return_head(latent).squeeze(-1)


def build_lstm(input_size: int, config: Optional[dict] = None) -> LSTMPredictor:
    """ساخت مدل LSTM از تنظیمات (``model.lstm``)."""
    cfg = config or {}
    return LSTMPredictor(
        input_size=input_size,
        hidden_size=int(cfg.get("hidden_size", 128)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
