"""روت‌های بازار: گراف جریان پول و آخرین قیمت‌ها."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src import db as dbm
from src.features.graph_builder import build_money_flow_graph, graph_summary
from src.features.money_flow import compute_money_flow

router = APIRouter(tags=["market"])


@router.get("/graph/money-flow")
def money_flow_graph(window_days: int = 5) -> Dict[str, Any]:
    """گراف جریان پول بازار به‌صورت لیست رأس‌ها و یال‌ها."""
    client = dbm.load_client_type()
    if client.empty:
        raise HTTPException(status_code=503, detail="داده حقیقی/حقوقی موجود نیست")
    try:
        graph = build_money_flow_graph(compute_money_flow(client), window_days=window_days)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "summary": graph_summary(graph),
        "nodes": [{"id": n, "sector": d.get("sector", "")} for n, d in graph.nodes(data=True)],
        "edges": [{"source": a, "target": b, "weight": d.get("weight", 0)}
                  for a, b, d in graph.edges(data=True)],
    }


@router.get("/market/latest")
def latest_prices() -> List[Dict[str, Any]]:
    """آخرین قیمت پایانی هر نماد."""
    df = dbm.read_sql(
        "SELECT i.symbol, MAX(d.gregorian_date) AS gregorian_date, "
        "d.close, d.last, d.volume FROM daily_history d "
        "JOIN instruments i ON i.ins_code = d.ins_code GROUP BY i.symbol"
    )
    return df.to_dict(orient="records") if not df.empty else []
