import sqlite3
from pathlib import Path

db_files = [
    Path(r"D:\Demand Rush\portfolio_workspace\tehran_stock_forecasting\data\bourse.db"),
    Path(r"D:\Demand Rush\portfolio_workspace\tehran_stock_forecasting\data\bourse.20260914_backfilled.db.bak"),
    Path(r"D:\Demand Rush\portfolio_workspace\tehran_stock_forecasting\data\bourse.db.bak"),
    Path(r"D:\Demand Rush\portfolio_workspace\tehran_stock_forecasting\data\backups\bourse_20260914_170426.db")
]

for p in db_files:
    if not p.exists():
        continue
    print(f"\n--- {p.name} (size: {p.stat().st_size:,} bytes) ---")
    conn = sqlite3.connect(p)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cur.fetchall()]
    for t in tables:
        try:
            cnt = cur.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            if cnt > 0:
                print(f"  {t}: {cnt} rows")
        except Exception as e:
            print(f"  {t}: error {e}")
    conn.close()
