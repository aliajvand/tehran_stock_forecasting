-- ==============================================================================
-- Business Question 3: What is the cumulative equity curve, maximum drawdown,
-- and annualized Sharpe ratio under realistic TSE trading limits (+/-5%)?
-- ==============================================================================

WITH stock_returns AS (
    SELECT 
        h.ins_code,
        i.symbol,
        h.gregorian_date,
        h.close,
        LAG(h.close, 1) OVER (PARTITION BY h.ins_code ORDER BY h.gregorian_date) as prev_close,
        -- Clamp price returns to TSE regulatory limit bands (-5.0% to +5.0%)
        CASE 
            WHEN ((h.close - LAG(h.close, 1) OVER (PARTITION BY h.ins_code ORDER BY h.gregorian_date)) * 1.0 / NULLIF(LAG(h.close, 1) OVER (PARTITION BY h.ins_code ORDER BY h.gregorian_date), 0)) > 0.05 THEN 0.05
            WHEN ((h.close - LAG(h.close, 1) OVER (PARTITION BY h.ins_code ORDER BY h.gregorian_date)) * 1.0 / NULLIF(LAG(h.close, 1) OVER (PARTITION BY h.ins_code ORDER BY h.gregorian_date), 0)) < -0.05 THEN -0.05
            ELSE ((h.close - LAG(h.close, 1) OVER (PARTITION BY h.ins_code ORDER BY h.gregorian_date)) * 1.0 / NULLIF(LAG(h.close, 1) OVER (PARTITION BY h.ins_code ORDER BY h.gregorian_date), 0))
        END as bounded_daily_return
    FROM daily_history h
    JOIN instruments i ON h.ins_code = i.ins_code
),
portfolio_daily AS (
    SELECT 
        gregorian_date,
        COUNT(DISTINCT ins_code) as active_stocks,
        AVG(bounded_daily_return) as daily_portfolio_return
    FROM stock_returns
    WHERE bounded_daily_return IS NOT NULL
    GROUP BY gregorian_date
),
equity_accumulation AS (
    SELECT 
        gregorian_date,
        active_stocks,
        daily_portfolio_return,
        -- Cumulative portfolio return index starting from 1.0 (base 100%)
        1.0 + SUM(daily_portfolio_return) OVER (ORDER BY gregorian_date) as cumulative_wealth_index
    FROM portfolio_daily
),
peak_and_drawdown AS (
    SELECT 
        gregorian_date,
        daily_portfolio_return,
        cumulative_wealth_index,
        MAX(cumulative_wealth_index) OVER (ORDER BY gregorian_date) as running_peak
    FROM equity_accumulation
)
SELECT 
    gregorian_date,
    ROUND(daily_portfolio_return * 100, 3) as daily_return_pct,
    ROUND(cumulative_wealth_index, 4) as equity_curve,
    ROUND(running_peak, 4) as historical_peak,
    ROUND(((cumulative_wealth_index - running_peak) * 100.0) / running_peak, 2) as drawdown_pct
FROM peak_and_drawdown
ORDER BY gregorian_date DESC
LIMIT 120;
