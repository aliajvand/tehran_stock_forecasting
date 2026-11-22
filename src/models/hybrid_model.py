"""
مدل هیبرید LSTM + GNN.

شاخه LSTM سری زمانی هر نماد را می‌بیند، شاخه GNN جایگاه همان نماد در گراف جریان
پول بازار را؛ دو نمایش با concat ترکیب و از یک MLP رد می‌شوند.

خروجی: logits سه‌کلاسه + احتمال (softmax) + بازدهی تخمینی.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from src.models.lstm_model import LSTMPredictor


class HybridLSTMGNN(nn.Module):
    """ترکیب شاخه زمانی و شاخه گرافی با لایه fusion."""

    def __init__(
        self,
        seq_input_size: int,
        graph_input_size: int = 0,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        gnn_hidden: int = 64,
        gnn_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        gnn_type: str = "GCN",
    ) -> None:
        """
        پارامترها:
            seq_input_size: تعداد فیچرهای سری زمانی.
            graph_input_size: تعداد فیچرهای رأس گراف (۰ = بدون شاخه گراف).
        """
        super().__init__()
        self.lstm_branch = LSTMPredictor(
            input_size=seq_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.use_graph = graph_input_size > 0
        fusion_dim = lstm_hidden

        if self.use_graph:
            from src.models.gnn_model import GCNPredictor  # import تنبل (وابستگی سنگین)

            self.gnn_branch = GCNPredictor(
                input_size=graph_input_size,
                hidden_size=gnn_hidden,
                num_layers=gnn_layers,
                dropout=dropout,
                num_classes=num_classes,
                conv_type=gnn_type,
            )
            fusion_dim += gnn_hidden

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(fusion_dim // 2, num_classes)
        self.return_head = nn.Linear(fusion_dim // 2, 1)

    def forward(
        self,
        sequences: torch.Tensor,
        graph_x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        node_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        عبور رو به جلو.

        پارامترها:
            sequences: ``(batch, seq_len, n_features)``
            graph_x, edge_index: گراف روز جاری (مشترک بین نمونه‌های batch)
            node_index: ایندکس رأس هر نمونه در گراف ``(batch,)``

        خروجی: (logits، بازدهی تخمینی)
        """
        latent = self.lstm_branch.encode(sequences)

        if self.use_graph and graph_x is not None and edge_index is not None:
            node_emb = self.gnn_branch.encode(graph_x, edge_index, edge_weight)
            if node_index is None:
                pooled = node_emb.mean(dim=0, keepdim=True).expand(latent.size(0), -1)
            else:
                pooled = node_emb.index_select(0, node_index)
            latent = torch.cat([latent, pooled], dim=1)

        fused = self.fusion(latent)
        return self.class_head(fused), self.return_head(fused).squeeze(-1)

    @torch.no_grad()
    def predict(self, sequences: torch.Tensor, **graph_kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        پیش‌بینی نهایی به‌همراه احتمال کلاس‌ها.

        خروجی: ``{"class": ایندکس کلاس, "proba": احتمال‌ها, "expected_return": بازدهی}``
        """
        self.eval()
        logits, expected = self.forward(sequences, **graph_kwargs)
        proba = torch.softmax(logits, dim=1)
        return {"class": proba.argmax(dim=1), "proba": proba, "expected_return": expected}


def build_hybrid(
    seq_input_size: int, graph_input_size: int = 0, config: Optional[dict] = None
) -> HybridLSTMGNN:
    """ساخت مدل هیبرید از تنظیمات بخش ``model``."""
    cfg = config or {}
    lstm_cfg = cfg.get("lstm", {}) or {}
    gnn_cfg = cfg.get("gnn", {}) or {}
    return HybridLSTMGNN(
        seq_input_size=seq_input_size,
        graph_input_size=graph_input_size,
        lstm_hidden=int(lstm_cfg.get("hidden_size", 128)),
        lstm_layers=int(lstm_cfg.get("num_layers", 2)),
        gnn_hidden=int(gnn_cfg.get("hidden_size", 64)),
        gnn_layers=int(gnn_cfg.get("num_layers", 2)),
        dropout=float(lstm_cfg.get("dropout", 0.3)),
        gnn_type=str(gnn_cfg.get("type", "GCN")),
    )
