-- ==============================================================================
-- Business Question 1: How do market volatility regimes and Average True Range (ATR)
-- shift across Iranian industrial sectors (Metals, Petrochemicals, Automotive)?
-- ==============================================================================

WITH price_deltas AS (
    SELECT 
        h.ins_code,
        i.symbol,
        i.industry_group,
        h.gregorian_date,
        h.close,
        h.high,
        h.low,
        LAG(h.close, 1) OVER (PARTITION BY h.ins_code ORDER BY h.gregorian_date) as prev_close
    FROM daily_history h
    JOIN instruments i ON h.ins_code = i.ins_code
),
true_ranges AS (
    SELECT 
        ins_code,
        symbol,
        industry_group,
        gregorian_date,
        close,
        -- True Range = MAX(high - low, ABS(high - prev_close), ABS(low - prev_close))
        MAX(
            high - low,
            COALESCE(ABS(high - prev_close), high - low),
            COALESCE(ABS(low - prev_close), high - low)
        ) as true_range,
        (close - prev_close) * 1.0 / NULLIF(prev_close, 0) as daily_return
    FROM price_deltas
),
rolling_metrics AS (
    SELECT 
        ins_code,
        symbol,
        industry_group,
        gregorian_date,
        close,
        AVG(true_range) OVER (
            PARTITION BY ins_code 
            ORDER BY gregorian_date 
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) as atr_20d,
        AVG(daily_return) OVER (
            PARTITION BY ins_code 
            ORDER BY gregorian_date 
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) as mean_return_20d
    FROM true_ranges
)
SELECT 
    symbol,
    industry_group,
    gregorian_date,
    ROUND(close, 0) as close_price,
    ROUND(atr_20d, 2) as atr_20d,
    ROUND(mean_return_20d * 100, 2) as rolling_20d_return_pct,
    CASE 
        WHEN atr_20d > 1.5 * AVG(atr_20d) OVER (PARTITION BY industry_group) THEN 'High Volatility Expansion'
        WHEN atr_20d < 0.7 * AVG(atr_20d) OVER (PARTITION BY industry_group) THEN 'Low Volatility Squeeze'
        ELSE 'Normal Regime'
    END as volatility_regime
FROM rolling_metrics
ORDER BY gregorian_date DESC, symbol ASC
LIMIT 100;
