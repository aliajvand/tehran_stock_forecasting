"""Comprehensive test suite for Tehran Stock Exchange Forecasting Engine FastAPI backend."""
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_endpoint():
    """Verify system health, DB connection, and table row counts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "table_counts" in data
    assert data["total_records_indexed"] > 0
    assert "instruments" in data["table_counts"]

def test_symbols_endpoint():
    """Verify retrieval of liquid instruments with trading history."""
    response = client.get("/symbols?limit=10")
    assert response.status_code == 200
    symbols = response.json()
    assert isinstance(symbols, list)
    assert len(symbols) > 0
    first = symbols[0]
    assert "symbol" in first
    assert "ins_code" in first
    assert "history_days" in first
    assert first["history_days"] > 0

def test_forecast_endpoint_valid_symbol():
    """Verify directional forecast and probability distribution for known symbol."""
    # فولاد is a primary steel ticker
    response = client.post("/forecast", json={"symbol": "فولاد", "horizon_days": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "فولاد"
    assert data["predicted_direction"] in ["up", "down", "neutral"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    probs = data["probabilities"]
    assert "up" in probs and "down" in probs and "neutral" in probs
    assert abs(probs["up"] + probs["down"] + probs["neutral"] - 1.0) < 0.01
    assert "institutional_flow_signal" in data

def test_governance_endpoint():
    """Verify walk-forward cross validation metrics and architecture provenance."""
    response = client.get("/governance")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "tehran_stock_forecasting"
    assert "hybrid_gnn_model" in data
    hybrid = data["hybrid_gnn_model"]
    assert hybrid["walk_forward_splits"] == 5
    assert hybrid["fold5_peak_directional_accuracy"] == 0.6985
    assert "limitations" in data
    assert len(data["limitations"]) > 0
