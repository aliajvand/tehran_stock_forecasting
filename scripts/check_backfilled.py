import sqlite3
import pandas as pd

p = r"D:\Demand Rush\portfolio_workspace\tehran_stock_forecasting\data\bourse.20260914_backfilled.db.bak"
conn = sqlite3.connect(p)

df_stats = pd.read_sql_query("""
    SELECT 
        COUNT(DISTINCT ins_code) as distinct_instruments,
        MIN(gregorian_date) as min_date,
        MAX(gregorian_date) as max_date,
        COUNT(*) as total_rows
    FROM daily_history
""", conn)
print("daily_history stats:")
print(df_stats)

df_symbols = pd.read_sql_query("""
    SELECT i.ins_code, i.symbol, i.name, i.english_name, count(h.id) as days
    FROM daily_history h
    JOIN instruments i ON h.ins_code = i.ins_code
    GROUP BY i.ins_code, i.symbol, i.name, i.english_name
    ORDER BY days DESC
    LIMIT 20
""", conn)

with open("top_symbols.json", "w", encoding="utf-8") as f:
    df_symbols.to_json(f, orient="records", force_ascii=False, indent=2)

print("Saved top_symbols.json")
conn.close()
