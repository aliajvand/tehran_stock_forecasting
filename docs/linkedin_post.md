# LinkedIn Post: Tehran Stock Forecasting Platform

Forecasting emerging equity markets like the Tehran Stock Exchange (TSETMC) is notoriously difficult due to extreme market microstructure constraints:
- Strict daily price fluctuation limits (±5% caps), causing synthetic liquidity lockups ('saf-e kharid' and 'saf-e forush').
- Significant retail herd behavior contrasted with institutional ('Hoghughi') smart-money accumulation.
- Non-I.I.D. temporal sequences where standard cross-validation results in severe lookahead bias.

To address these challenges, I built and productionized the **Tehran Stock Forecasting Platform**:

Key technical and quantitative highlights:
- **Comprehensive Market Ingestion**: Indexed 10,874 daily price records and 10,876 institutional order flow facts across 428 TSETMC securities into an analytical star schema.
- **Multimodal Feature Store**: Engineered 80 quantitative signals spanning momentum indicators (RSI, ATR, Bollinger Bands), net institutional flow ratios, Jalali calendar cycle alignments, and cross-sector correlation graph edges.
- **5-Fold Expanding-Window Walk-Forward CV**: Enforced strict temporal out-of-sample evaluation across 9,439 samples, eliminating lookahead bias.
- **Hybrid LSTM + Graph Convolutional Network**: Reached **69.85% peak directional accuracy** and **0.7024 ROC-AUC** on trend-following holdout regimes by capturing inter-stock industry co-movements.
- **Production Architecture**: Packaged into a low-latency FastAPI microservice (`/forecast`, `/symbols`, `/governance`) and a responsive Plotly/Dash operations portal featuring interactive candlestick charting and order flow tracking.

Repository and evaluation reports: https://github.com/aliajvand/tehran-stock-forecasting

#QuantitativeFinance #MachineLearning #DeepLearning #AlgorithmicTrading #FastAPI #PlotlyDash #Python
