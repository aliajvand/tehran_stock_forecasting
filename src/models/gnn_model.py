"""مدل GNN (GCN/GAT) روی گراف جریان پول بین نمادها."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:  # pragma: no cover - وابستگی سنگین
    from torch_geometric.nn import GATConv, GCNConv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "برای مدل GNN باید torch-geometric نصب باشد: pip install torch-geometric"
    ) from exc


class GCNPredictor(nn.Module):
    """
    GNN دولایه برای تولید نمایش (embedding) هر نماد و طبقه‌بندی ۳ کلاسه.

    ورودی: ``x (n_nodes, n_features)`` و ``edge_index (2, n_edges)``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        conv_type: str = "GCN",
        heads: int = 4,
    ) -> None:
        """پارامترها از بخش ``model.gnn`` فایل config می‌آیند."""
        super().__init__()
        self.conv_type = str(conv_type).upper()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()

        in_dim = input_size
        for _ in range(max(1, num_layers)):
            if self.conv_type == "GAT":
                conv = GATConv(in_dim, hidden_size, heads=heads, concat=False)
            else:
                conv = GCNConv(in_dim, hidden_size)
            self.convs.append(conv)
            in_dim = hidden_size

        self.output_size = hidden_size
        self.head = nn.Linear(hidden_size, num_classes)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """تولید نمایش گرافی برای هر رأس."""
        hidden = x
        for conv in self.convs:
            if self.conv_type == "GAT":
                hidden = conv(hidden, edge_index)
            else:
                hidden = conv(hidden, edge_index, edge_weight)
            hidden = torch.relu(hidden)
            hidden = self.dropout(hidden)
        return hidden

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """خروجی logits برای هر رأس."""
        return self.head(self.encode(x, edge_index, edge_weight))


def build_gnn(input_size: int, config: Optional[dict] = None) -> GCNPredictor:
    """ساخت مدل GNN از تنظیمات (``model.gnn``)."""
    cfg = config or {}
    return GCNPredictor(
        input_size=input_size,
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
        conv_type=str(cfg.get("type", "GCN")),
        heads=int(cfg.get("heads", 4)),
    )
