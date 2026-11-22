# Resume Project Summary: Tehran Stock Forecasting Platform

## Bullet Points for Resume (Ali Ajvand - Data Scientist & BI)

- **Tehran Stock Forecasting Platform (Python, PyTorch, FastAPI, Plotly/Dash, SQLite/DuckDB, Algorithmic Trading)**
  - Engineered an end-to-end quantitative forecasting platform indexing 10,874 daily price records and 10,876 institutional order flow facts across 428 TSETMC securities.
  - Implemented 5-fold expanding-window walk-forward cross validation across 9,439 samples, completely eliminating temporal lookahead bias in financial time series.
  - Developed a Hybrid LSTM + Graph Convolutional Network (GCN) modeling cross-stock sector correlations and institutional (Hoghughi) accumulation, achieving **69.85% directional accuracy** and **0.7024 ROC-AUC** on trend holdouts.
  - Built an analytical star schema with windowed SQL marts computing 20-day Average True Range (ATR), institutional accumulation ratios, and drawdown curves under TSE ±5% price limits.
  - Deployed a sub-second FastAPI microservice (`/forecast`, `/symbols`, `/governance`) and an interactive Plotly/Dash operations portal with real-time candlestick charts and order flow analytics.
