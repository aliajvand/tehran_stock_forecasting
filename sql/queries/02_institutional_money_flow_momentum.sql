-- ==============================================================================
-- Business Question 2: Which tickers exhibit significant institutional (Hoghughi)
-- smart-money accumulation vs retail panic selling prior to major price rallies?
-- ==============================================================================

WITH daily_flows AS (
    SELECT 
        c.ins_code,
        i.symbol,
        i.name,
        c.gregorian_date,
        c.vol_buy_institutional,
        c.vol_sell_institutional,
        c.vol_buy_retail,
        c.vol_sell_retail,
        (COALESCE(c.vol_buy_institutional, 0) - COALESCE(c.vol_sell_institutional, 0)) as net_institutional_vol,
        (COALESCE(c.vol_buy_retail, 0) - COALESCE(c.vol_sell_retail, 0)) as net_retail_vol,
        (COALESCE(c.vol_buy_institutional, 0) + COALESCE(c.vol_buy_retail, 0)) as total_market_volume,
        c.power_institutional,
        c.power_retail
    FROM client_type c
    JOIN instruments i ON c.ins_code = i.ins_code
),
rolling_accumulation AS (
    SELECT 
        ins_code,
        symbol,
        name,
        gregorian_date,
        net_institutional_vol,
        net_retail_vol,
        power_institutional,
        power_retail,
        -- 5-day rolling net institutional volume
        SUM(net_institutional_vol) OVER (
            PARTITION BY ins_code 
            ORDER BY gregorian_date 
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as net_ins_5d,
        -- 5-day rolling total volume
        SUM(total_market_volume) OVER (
            PARTITION BY ins_code 
            ORDER BY gregorian_date 
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as total_vol_5d
    FROM daily_flows
)
SELECT 
    symbol,
    name,
    gregorian_date,
    net_ins_5d,
    total_vol_5d,
    ROUND((net_ins_5d * 100.0) / NULLIF(total_vol_5d, 0), 2) as institutional_accumulation_ratio_pct,
    ROUND(power_institutional, 2) as ins_buyer_power,
    ROUND(power_retail, 2) as retail_buyer_power,
    CASE 
        WHEN (net_ins_5d * 1.0) / NULLIF(total_vol_5d, 0) > 0.15 AND power_institutional > 1.2 THEN 'Strong Institutional Accumulation (Bullish)'
        WHEN (net_ins_5d * 1.0) / NULLIF(total_vol_5d, 0) < -0.15 AND power_retail > 1.2 THEN 'Heavy Institutional Distribution (Bearish)'
        ELSE 'Balanced Order Flow'
    END as smart_money_signal
FROM rolling_accumulation
WHERE total_vol_5d > 0
ORDER BY gregorian_date DESC, institutional_accumulation_ratio_pct DESC
LIMIT 100;
