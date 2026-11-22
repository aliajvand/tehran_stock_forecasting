# Technical Extensions & Advanced Engineering: Tehran Stock Forecasting

## 1. Overview
The baseline project establishes a clean walk-forward forecasting pipeline. The following extensions deepen the mathematical, algorithmic, and engineering architecture of the system.

---

## 2. Implemented Extensions

### Extension A: Hybrid LSTM + Graph Convolutional Network (GNN)
- **Mathematical Formulation**: Financial assets in the Tehran Stock Exchange do not move in isolation. Cross-ownership, supply chain dependencies (e.g. Iron Ore miners supplying Steel producers), and macro factors induce strong co-movements.
- **Implementation**:
  - Spatial Adjacency Matrix $A \in \mathbb{R}^{N \times N}$ based on rolling 60-day return correlations and industry sector groupings.
  - 2-layer Graph Convolutional Network (GCN) coupled with a temporal Recurrent LSTM backbone.
  - Achieved peak directional accuracy of **69.85%** and **0.7024 ROC-AUC** on Fold 5.

### Extension B: Analytical Star Schema & Windowed SQL Marts
- **Implementation**: Engineered SQLite/DuckDB star schema:
  - `dim_instruments`: 428 indexed TSETMC securities with sector codes and ISIN numbers.
  - `fact_daily_history`: 10,874 daily price rows (Open, High, Low, Close, Volume, Value, Trade Count).
  - `fact_client_type`: 10,876 rows of institutional vs retail order flow facts (Haghighi / Hoghughi).
  - `fact_prediction_signals`: Auditable walk-forward inference log with probability distributions.
- **Analytical Queries**:
  - `01_market_regime_volatility.sql`: Calculates 20-day Average True Range (ATR) and regime classification across sectors.
  - `02_institutional_money_flow_momentum.sql`: Rolling 5-day net institutional accumulation ratio.
  - `03_rolling_sharpe_drawdown.sql`: Cumulative portfolio equity curve and drawdown under ±5% price limits.

### Extension C: Production FastAPI Microservice & Dash Operations Portal
- **FastAPI Backend (`backend/app.py`)**:
  - `/health`: DB connection status, 10,874 records verified, model status.
  - `/symbols`: Paginated liquid instruments with latest close and history depth.
  - `/forecast`: Tactical next-day directional inference with probability distribution (Up/Down/Neutral), expected return, and institutional flow signal.
  - `/governance`: Auditable 5-fold walk-forward cross validation metrics and limitations.
- **Dash Operations Dashboard (`frontend/app.py`)**:
  - Deployed on dedicated port 8063 using unified `design_system/` tokens.
  - Interactive Candlestick Chart (Open/High/Low/Close + Volume bars).
  - Institutional vs Retail Order Flow bar charts (Hoghughi vs Haghighi net volume).
  - 5-Fold Walk-Forward Performance Comparison (LSTM vs Hybrid GNN).

---

## 3. Methodological Future Extensions
1. **Order Book Level-2 Microstructure Dynamics**:
   - Model limit order book imbalance (LOBI) at 5 depth levels ($L_1 \dots L_5$) during pre-opening auctions (08:45 - 09:00).
2. **Reinforcement Learning with Asymmetric Price Limits (PPO / SAC)**:
   - Train an execution agent that learns to place limit orders ahead of queue lockups to avoid execution slippage.
