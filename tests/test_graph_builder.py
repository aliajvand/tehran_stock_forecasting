"""تست ساخت گراف جریان پول (فاز ۲)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.money_flow import compute_money_flow

nx = pytest.importorskip("networkx", reason="networkx نصب نیست")

from src.features.graph_builder import (  # noqa: E402
    build_dynamic_graphs,
    build_money_flow_graph,
    graph_summary,
)


def test_graph_has_all_symbols(sample_client_type: pd.DataFrame) -> None:
    """هر نماد باید یک رأس داشته باشد."""
    graph = build_money_flow_graph(compute_money_flow(sample_client_type), window_days=5)
    assert set(graph.nodes()) == {"فولاد", "فخوز"}


def test_edge_weights_bounded(sample_client_type: pd.DataFrame) -> None:
    """وزن یال‌ها باید مثبت و حداکثر ۱ باشد."""
    graph = build_money_flow_graph(
        compute_money_flow(sample_client_type), window_days=10, correlation_threshold=0.0
    )
    for _, _, data in graph.edges(data=True):
        assert 0 < data["weight"] <= 1.0


def test_dynamic_graphs_per_day(sample_client_type: pd.DataFrame) -> None:
    """گراف پویا باید برای هر روز خواسته‌شده ساخته شود."""
    flow = compute_money_flow(sample_client_type)
    days = sorted(flow["gregorian_date"].unique())[-3:]
    graphs = build_dynamic_graphs(flow, window_days=5, dates=days)
    assert len(graphs) == 3
    assert all(g.number_of_nodes() == 2 for g in graphs.values())


def test_graph_summary_shape(sample_client_type: pd.DataFrame) -> None:
    """خلاصه گراف باید کلیدهای لازم را داشته باشد."""
    graph = build_money_flow_graph(compute_money_flow(sample_client_type))
    summary = graph_summary(graph)
    assert {"nodes", "edges", "density", "top_hubs"} <= set(summary)


def test_empty_input_returns_empty_graph() -> None:
    """ورودی خالی باید گراف خالی بدهد."""
    assert build_money_flow_graph(pd.DataFrame()).number_of_nodes() == 0
