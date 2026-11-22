-- ==============================================================================
-- Tehran Stock Exchange (TSETMC) Market Data Mart & Forecasting Schema
-- Storage: SQLite / DuckDB
-- ==============================================================================

-- 1. Instruments Dimension
CREATE TABLE IF NOT EXISTS dim_instruments (
    ins_code BIGINT PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    name VARCHAR(128) NOT NULL,
    english_name VARCHAR(64),
    company_code VARCHAR(16),
    company_isin VARCHAR(16),
    instrument_isin VARCHAR(16),
    asset_type VARCHAR(32) DEFAULT 'stock',
    market VARCHAR(64),
    sector_code VARCHAR(16),
    industry_group VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_instruments_symbol ON dim_instruments(symbol);
CREATE INDEX IF NOT EXISTS idx_instruments_sector ON dim_instruments(industry_group);

-- 2. Daily Price & Microstructure History Fact
CREATE TABLE IF NOT EXISTS fact_daily_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code BIGINT NOT NULL,
    gregorian_date INTEGER NOT NULL,  -- YYYYMMDD integer format
    jalali_date VARCHAR(10) NOT NULL, -- 1403/06/24
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    last REAL,
    volume BIGINT,
    value REAL,
    trade_count INTEGER,
    FOREIGN KEY (ins_code) REFERENCES dim_instruments(ins_code),
    UNIQUE(ins_code, gregorian_date)
);

CREATE INDEX IF NOT EXISTS idx_history_ins_date ON fact_daily_history(ins_code, gregorian_date);
CREATE INDEX IF NOT EXISTS idx_history_date ON fact_daily_history(gregorian_date);

-- 3. Institutional vs Retail Order Flow Fact (Haghighi / Hoghughi)
CREATE TABLE IF NOT EXISTS fact_client_type (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code BIGINT NOT NULL,
    gregorian_date INTEGER NOT NULL,
    jalali_date VARCHAR(10) NOT NULL,
    n_buy_retail INTEGER,
    n_buy_institutional INTEGER,
    n_sell_retail INTEGER,
    n_sell_institutional INTEGER,
    vol_buy_retail BIGINT,
    vol_buy_institutional BIGINT,
    vol_sell_retail BIGINT,
    vol_sell_institutional BIGINT,
    val_buy_retail REAL,
    val_buy_institutional REAL,
    val_sell_retail REAL,
    val_sell_institutional REAL,
    per_capita_buy_retail REAL,
    per_capita_sell_retail REAL,
    per_capita_buy_institutional REAL,
    per_capita_sell_institutional REAL,
    power_retail REAL,
    power_institutional REAL,
    FOREIGN KEY (ins_code) REFERENCES dim_instruments(ins_code),
    UNIQUE(ins_code, gregorian_date)
);

CREATE INDEX IF NOT EXISTS idx_client_ins_date ON fact_client_type(ins_code, gregorian_date);

-- 4. Walk-Forward Prediction Audit & Signal Log
CREATE TABLE IF NOT EXISTS fact_prediction_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code BIGINT NOT NULL,
    prediction_date INTEGER NOT NULL,
    target_horizon_days INTEGER DEFAULT 1,
    model_version VARCHAR(32) NOT NULL,
    predicted_class VARCHAR(16) NOT NULL, -- 'up', 'down', 'neutral'
    prob_up REAL NOT NULL,
    prob_down REAL NOT NULL,
    prob_neutral REAL NOT NULL,
    expected_return_pct REAL,
    actual_return_pct REAL,
    directional_hit BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ins_code) REFERENCES dim_instruments(ins_code)
);
