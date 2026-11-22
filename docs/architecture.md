# Architecture & Governance: Tehran Stock Forecasting Platform

## 1. System Overview
The **Tehran Stock Forecasting Platform** is an enterprise-grade quantitative forecasting and market microstructure analytics system designed for the Tehran Stock Exchange (TSETMC).

```
TSETMC Ingestion (algotik-tse)
    │
    ▼
SQLite / DuckDB Star Schema (10,874 records across 428 instruments)
    │
    ├── Features Pipeline (80 signals: Technical, Money Flow, Sector Graph, Jalali Calendar)
    │
    ├── 5-Fold Walk-Forward Cross Validation (LSTM vs Hybrid GNN)
    │
    ├── FastAPI Microservice (backend/app.py -> /health, /symbols, /forecast, /governance)
    │
    └── Plotly / Dash Operations Portal (frontend/app.py -> Port 8063)
```

## 2. Technical Stack & Governance

| Layer | Component | Specification |
|---|---|---|
| **Data Ingestion** | TSETMC Client | Async market snapshot fetcher with freshness guards & retry logic |
| **Storage** | SQLite / DuckDB | Star schema: `dim_instruments`, `fact_daily_history`, `fact_client_type` |
| **Features** | Feature Store | 80 signals: SMA, RSI, ATR, Hoghughi Net Flow, Correlation Graph Laplacians |
| **Modeling** | PyTorch / GNN | 2-Layer LSTM + Graph Convolutional Network (GCN) with spatial edge weights |
| **Validation** | Walk-Forward CV | 5 non-overlapping expanding-window splits (9,439 training samples) |
| **Serving** | FastAPI | Sub-second inference microservice with Pydantic validation schemas |
| **Operations** | Plotly / Dash | Real-time operations portal on port 8063 using unified design system |

## 3. Core Architectural Rules
1. **Zero Lookahead Bias**: No forward-looking rolling windows or future target leakage.
2. **Expanding-Window Validation**: Strict temporal separation across all 5 test folds.
3. **Regulatory Limit Conformance**: All simulated returns conform to TSETMC ±5% daily price boundaries.
4. **Reproducibility Guarantee**: Model runs and checkpoint metadata serialized in `models/saved/runs/`.
