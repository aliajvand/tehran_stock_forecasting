import sqlite3
import json
from pathlib import Path

db_path = Path(r"D:\Demand Rush\portfolio_workspace\tehran_stock_forecasting\data\bourse.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cur.fetchall()]

summary = {}
for name in tables:
    cur.execute(f'SELECT count(*) FROM "{name}"')
    cnt = cur.fetchone()[0]
    cur.execute(f'PRAGMA table_info("{name}")')
    cols = [col[1] for col in cur.fetchall()]
    cur.execute(f'SELECT * FROM "{name}" LIMIT 3')
    sample = cur.fetchall()
    summary[name] = {
        "count": cnt,
        "columns": cols,
        "sample_count": len(sample)
    }

conn.close()

with open("data_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Saved data_summary.json successfully.")
for k, v in summary.items():
    print(f"- {k}: {v['count']} rows, cols: {len(v['columns'])}")
