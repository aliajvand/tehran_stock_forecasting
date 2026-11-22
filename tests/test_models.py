"""تست مدل‌ها با تنسور جعلی (فاز ۳). در نبود torch، تست‌ها skip می‌شوند."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch نصب نیست")

from src.models.hybrid_model import build_hybrid  # noqa: E402
from src.models.lstm_model import build_lstm  # noqa: E402


def test_lstm_output_shapes() -> None:
    """خروجی LSTM باید (batch,3) و (batch,) باشد."""
    model = build_lstm(12, {"hidden_size": 32, "num_layers": 2, "dropout": 0.1})
    logits, expected = model(torch.randn(8, 20, 12))
    assert logits.shape == (8, 3)
    assert expected.shape == (8,)


def test_hybrid_without_graph() -> None:
    """مدل هیبرید باید بدون شاخه گراف هم کار کند."""
    model = build_hybrid(10, 0, {"lstm": {"hidden_size": 32}, "gnn": {"hidden_size": 16}})
    logits, expected = model(torch.randn(4, 15, 10))
    assert logits.shape == (4, 3)
    assert expected.shape == (4,)


def test_hybrid_predict_probabilities() -> None:
    """احتمال‌ها باید جمعشان ۱ شود."""
    model = build_hybrid(6, 0, {"lstm": {"hidden_size": 16}})
    out = model.predict(torch.randn(5, 12, 6))
    assert out["proba"].shape == (5, 3)
    assert torch.allclose(out["proba"].sum(dim=1), torch.ones(5), atol=1e-5)
    assert out["class"].shape == (5,)


def test_hybrid_with_graph_branch() -> None:
    """شاخه GNN در صورت نصب بودن torch-geometric باید کار کند."""
    pytest.importorskip("torch_geometric", reason="torch-geometric نصب نیست")
    model = build_hybrid(8, 4, {"lstm": {"hidden_size": 16}, "gnn": {"hidden_size": 8, "type": "GCN"}})
    graph_x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    node_index = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    logits, _ = model(torch.randn(4, 10, 8), graph_x, edge_index, None, node_index)
    assert logits.shape == (4, 3)
