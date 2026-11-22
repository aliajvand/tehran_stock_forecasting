"""
ساخت گراف جریان سرمایه بین سهام (ماژول کلیدی شاخه GNN).

منطق:
    * رأس‌ها: نمادها (هر رأس بردار ویژگی روزانه دارد).
    * یال‌ها: هم‌حرکتی جریان پول حقیقی بین دو نماد در پنجره اخیر
      (همبستگی خالص ورود پول) + یال ضعیف‌تر برای هم‌صنعتی بودن.
    * وزن یال: قدر مطلق همبستگی (به‌علاوه پاداش هم‌صنعتی).
    * گراف «پویا» است: برای هر روز معاملاتی با پنجره کشویی ساخته می‌شود.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_section, get_symbol_sectors

try:  # pragma: no cover
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]
    logger.warning("networkx نصب نیست؛ ساخت گراف ممکن نیست")


def _flow_matrix(client_df: pd.DataFrame, value_col: str = "net_val_retail") -> pd.DataFrame:
    """
    تبدیل داده حقیقی/حقوقی به ماتریس «تاریخ × نماد» از خالص جریان پول.
    """
    df = client_df.copy()
    if value_col not in df.columns:
        buy = pd.to_numeric(df.get("val_buy_retail", 0), errors="coerce").fillna(0.0)
        sell = pd.to_numeric(df.get("val_sell_retail", 0), errors="coerce").fillna(0.0)
        df[value_col] = buy - sell
    return df.pivot_table(
        index="gregorian_date", columns="symbol", values=value_col, aggfunc="sum"
    ).sort_index()


def build_money_flow_graph(
    client_type_df: pd.DataFrame,
    window_days: int = 5,
    as_of: Optional[int] = None,
    correlation_threshold: Optional[float] = None,
    node_features: Optional[pd.DataFrame] = None,
) -> Any:
    """
    ساخت گراف جریان پول برای یک روز مشخص.

    پارامترها:
        client_type_df: داده حقیقی/حقوقی چند نماد (خروجی ``compute_money_flow``).
        window_days: طول پنجره کشویی برای محاسبه همبستگی.
        as_of: تاریخ عددی ``YYYYMMDD`` مرجع (پیش‌فرض: آخرین تاریخ موجود).
        correlation_threshold: حد آستانه ساخت یال.
        node_features: DataFrame با ایندکس نماد برای ویژگی رأس‌ها.

    خروجی: ``networkx.Graph``
    """
    if nx is None:
        raise ImportError("برای ساخت گراف باید networkx نصب باشد")
    if client_type_df is None or client_type_df.empty:
        logger.warning("build_money_flow_graph: ورودی خالی است")
        return nx.Graph()

    graph_cfg = get_section("features", {}).get("graph", {}) or {}
    threshold = float(
        correlation_threshold if correlation_threshold is not None
        else graph_cfg.get("correlation_threshold", 0.3)
    )
    sector_weight = float(graph_cfg.get("same_sector_weight", 0.5))

    matrix = _flow_matrix(client_type_df)
    if matrix.empty:
        return nx.Graph()

    if as_of is not None:
        matrix = matrix.loc[matrix.index <= as_of]
    window = matrix.tail(max(2, int(window_days)))
    if window.shape[0] < 2 or window.shape[1] < 2:
        logger.debug("پنجره برای ساخت یال کافی نیست: {}", window.shape)

    sectors = get_symbol_sectors()
    graph = nx.Graph()
    for symbol in matrix.columns:
        attrs: Dict[str, Any] = {"sector": sectors.get(symbol, "unknown")}
        if node_features is not None and symbol in node_features.index:
            attrs["x"] = node_features.loc[symbol].to_numpy(dtype=float)
        graph.add_node(symbol, **attrs)

    corr = window.corr(min_periods=2) if window.shape[0] >= 2 else pd.DataFrame()
    symbols: List[str] = list(matrix.columns)
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            weight = 0.0
            if not corr.empty and a in corr.index and b in corr.columns:
                value = corr.loc[a, b]
                if pd.notna(value) and abs(float(value)) >= threshold:
                    weight = abs(float(value))
            if sectors.get(a) and sectors.get(a) == sectors.get(b):
                weight = max(weight, sector_weight)
            if weight > 0:
                graph.add_edge(a, b, weight=round(weight, 4))

    logger.info(
        "گراف جریان پول: {} رأس، {} یال (پنجره {} روز)",
        graph.number_of_nodes(), graph.number_of_edges(), window_days,
    )
    return graph


def build_dynamic_graphs(
    client_type_df: pd.DataFrame, window_days: int = 5, dates: Optional[Sequence[int]] = None
) -> Dict[int, Any]:
    """
    ساخت گراف برای هر روز معاملاتی (گراف پویا).

    خروجی: ``{gregorian_date: networkx.Graph}``
    """
    matrix = _flow_matrix(client_type_df)
    target_dates = list(dates) if dates is not None else list(matrix.index)
    graphs: Dict[int, Any] = {}
    for day in target_dates:
        try:
            graphs[int(day)] = build_money_flow_graph(
                client_type_df, window_days=window_days, as_of=int(day)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ساخت گراف روز {} ناموفق بود: {}", day, exc)
    logger.info("{} گراف روزانه ساخته شد", len(graphs))
    return graphs


def graph_to_pyg(graph: Any, node_order: Optional[Sequence[str]] = None) -> Any:
    """
    تبدیل گراف ``networkx`` به ``torch_geometric.data.Data``.

    ویژگی رأس از خصیصه ``x`` گرفته می‌شود؛ در نبود آن، درجه و مجموع وزن یال‌ها
    به‌عنوان ویژگی پیش‌فرض ساخته می‌شود.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("برای تبدیل گراف باید torch و torch-geometric نصب باشند") from exc

    nodes: List[str] = list(node_order or graph.nodes())
    index = {name: i for i, name in enumerate(nodes)}

    features: List[List[float]] = []
    for name in nodes:
        attrs = graph.nodes[name]
        if "x" in attrs:
            features.append([float(v) for v in np.asarray(attrs["x"]).ravel()])
        else:
            degree = float(graph.degree(name))
            strength = float(sum(d.get("weight", 1.0) for _, _, d in graph.edges(name, data=True)))
            features.append([degree, strength])

    width = max((len(f) for f in features), default=1)
    padded = [f + [0.0] * (width - len(f)) for f in features]

    edge_index: List[List[int]] = [[], []]
    edge_weight: List[float] = []
    for a, b, data in graph.edges(data=True):
        if a not in index or b not in index:
            continue
        weight = float(data.get("weight", 1.0))
        for src, dst in ((index[a], index[b]), (index[b], index[a])):  # گراف بی‌جهت
            edge_index[0].append(src)
            edge_index[1].append(dst)
            edge_weight.append(weight)

    data = Data(
        x=torch.tensor(padded, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float),
    )
    data.symbols = nodes  # type: ignore[attr-defined]
    return data


def graph_node_features(
    graph: Any, feature_df: pd.DataFrame, symbol_col: str = "symbol"
) -> Tuple[List[str], np.ndarray]:
    """
    استخراج ماتریس ویژگی رأس‌ها از یک DataFrame فیچر (یک سطر برای هر نماد).

    خروجی: (ترتیب نمادها، ماتریس numpy)
    """
    nodes = list(graph.nodes())
    frame = feature_df.set_index(symbol_col) if symbol_col in feature_df.columns else feature_df
    numeric = frame.select_dtypes(include="number")
    matrix = numeric.reindex(nodes).fillna(0.0).to_numpy(dtype=float)
    return nodes, matrix


def graph_summary(graph: Any) -> Dict[str, Any]:
    """خلاصه آماری گراف برای نمایش در داشبورد/API."""
    if graph is None or graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "density": 0.0, "top_hubs": []}
    degrees = sorted(dict(graph.degree(weight="weight")).items(), key=lambda kv: kv[1], reverse=True)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 4) if nx is not None else 0.0,
        "top_hubs": [{"symbol": s, "strength": round(float(d), 3)} for s, d in degrees[:5]],
    }
