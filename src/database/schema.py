"""
طراحی و ساخت دیتابیس پروژه
"""
import sqlite3
from pathlib import Path
from loguru import logger

DB_PATH = Path("data/bourse.db")

SCHEMA_SQL = """
-- ============================================================
-- جدول ابزارها
-- ============================================================
CREATE TABLE IF NOT EXISTS instruments (
    ins_code        INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    name            TEXT,
    english_name    TEXT,
    company_code    TEXT,
    company_isin    TEXT,
    instrument_isin TEXT,
    asset_type      TEXT,
    market          TEXT,
    sector_code     TEXT,
    industry_group  TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- جدول تاریخچه روزانه
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code        INTEGER NOT NULL,
    gregorian_date  INTEGER NOT NULL,
    jalali_date     TEXT NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    last            REAL,
    volume          INTEGER,
    value           REAL,
    trade_count     INTEGER,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code),
    UNIQUE(ins_code, gregorian_date)
);

-- ============================================================
-- جدول حقیقی/حقوقی
-- ============================================================
CREATE TABLE IF NOT EXISTS client_type (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code                    INTEGER NOT NULL,
    gregorian_date              INTEGER NOT NULL,
    jalali_date                 TEXT NOT NULL,
    n_buy_retail                INTEGER,
    n_buy_institutional         INTEGER,
    n_sell_retail               INTEGER,
    n_sell_institutional        INTEGER,
    vol_buy_retail              INTEGER,
    vol_buy_institutional       INTEGER,
    vol_sell_retail             INTEGER,
    vol_sell_institutional      INTEGER,
    val_buy_retail              REAL,
    val_buy_institutional       REAL,
    val_sell_retail             REAL,
    val_sell_institutional      REAL,
    per_capita_buy_retail       REAL,
    per_capita_sell_retail      REAL,
    per_capita_buy_institutional  REAL,
    per_capita_sell_institutional REAL,
    power_retail                REAL,
    power_institutional         REAL,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code),
    UNIQUE(ins_code, gregorian_date)
);

-- ============================================================
-- جدول معاملات ریز
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code        INTEGER NOT NULL,
    gregorian_date  INTEGER NOT NULL,
    trade_no        INTEGER,
    time            TEXT,
    timestamp       INTEGER,
    price           REAL,
    volume          INTEGER,
    value           REAL,
    canceled        BOOLEAN DEFAULT 0,
    source          TEXT,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code)
);

-- ============================================================
-- جدول order book
-- ============================================================
CREATE TABLE IF NOT EXISTS order_book (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code        INTEGER NOT NULL,
    snapshot_time   INTEGER NOT NULL,
    level           INTEGER NOT NULL,
    bid_price       REAL,
    bid_volume      INTEGER,
    bid_order_count INTEGER,
    ask_price       REAL,
    ask_volume      INTEGER,
    ask_order_count INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code),
    UNIQUE(ins_code, snapshot_time, level)
);

-- ============================================================
-- جدول صف خرید/فروش
-- ============================================================
CREATE TABLE IF NOT EXISTS queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code        INTEGER NOT NULL,
    snapshot_time   INTEGER NOT NULL,
    side            TEXT NOT NULL,
    price           REAL,
    volume          INTEGER,
    count           INTEGER,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code)
);

-- ============================================================
-- جدول کندل درون‌روزی
-- ============================================================
CREATE TABLE IF NOT EXISTS intraday_candles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code        INTEGER NOT NULL,
    interval        TEXT NOT NULL,
    datetime        TEXT NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    volume          INTEGER,
    trade_count     INTEGER,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code),
    UNIQUE(ins_code, interval, datetime)
);

-- ============================================================
-- جدول snapshot بازار
-- ============================================================
CREATE TABLE IF NOT EXISTS market_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ins_code        INTEGER NOT NULL,
    snapshot_time   INTEGER NOT NULL,
    last            REAL,
    close           REAL,
    volume          INTEGER,
    value           REAL,
    trade_count     INTEGER,
    open            REAL,
    high            REAL,
    low             REAL,
    change_pct      REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code),
    UNIQUE(ins_code, snapshot_time)
);

-- ============================================================
-- جدول شاخص‌ها
-- ============================================================
CREATE TABLE IF NOT EXISTS indices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    index_code      TEXT UNIQUE NOT NULL,
    index_name      TEXT NOT NULL,
    value           REAL,
    change_pct      REAL,
    snapshot_time   INTEGER
);

-- ============================================================
-- جدول اعضای شاخص
-- ============================================================
CREATE TABLE IF NOT EXISTS index_companies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    index_code      TEXT NOT NULL,
    ins_code        INTEGER NOT NULL,
    weight          REAL,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code),
    UNIQUE(index_code, ins_code)
);

-- ============================================================
-- جدول اخبار کدال
-- ============================================================
CREATE TABLE IF NOT EXISTS codal_news (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id         TEXT UNIQUE,
    ins_code        INTEGER,
    symbol          TEXT,
    title           TEXT NOT NULL,
    content         TEXT,
    news_type       TEXT,
    publish_date    TEXT,
    publish_time    TEXT,
    url             TEXT,
    attachment_url  TEXT,
    sentiment       REAL,
    category        TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ins_code) REFERENCES instruments(ins_code)
);

-- ============================================================
-- ایندکس‌ها
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_instruments_symbol      ON instruments(symbol);
CREATE INDEX IF NOT EXISTS idx_instruments_market      ON instruments(market);
CREATE INDEX IF NOT EXISTS idx_instruments_asset_type  ON instruments(asset_type);
CREATE INDEX IF NOT EXISTS idx_instruments_industry    ON instruments(industry_group);

CREATE INDEX IF NOT EXISTS idx_daily_ins_date          ON daily_history(ins_code, gregorian_date);
CREATE INDEX IF NOT EXISTS idx_trades_ins_date         ON trades(ins_code, gregorian_date);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp        ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_client_ins_date         ON client_type(ins_code, gregorian_date);
CREATE INDEX IF NOT EXISTS idx_news_symbol             ON codal_news(symbol);
CREATE INDEX IF NOT EXISTS idx_news_date               ON codal_news(publish_date);
CREATE INDEX IF NOT EXISTS idx_news_ins_code           ON codal_news(ins_code);
CREATE INDEX IF NOT EXISTS idx_snapshot_ins_time ON market_snapshot(ins_code, snapshot_time);
CREATE INDEX IF NOT EXISTS idx_book_ins_time    ON order_book(ins_code, snapshot_time);
"""

def create_database(db_path: Path = DB_PATH, drop_existing: bool = False):
    """
    ساخت دیتابیس و جدول‌ها

    پارامترها:
        db_path: مسیر دیتابیس
        drop_existing: اگه True باشه، جدول‌های قبلی رو پاک می‌کنه
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if drop_existing:
        logger.warning("⚠️ حذف جدول‌های قبلی...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"   🗑️ جدول {table} حذف شد")
        conn.commit()

    executed = 0
    for statement in SCHEMA_SQL.split(";"):
        # 🔧 حذف خطوط کامنت از ابتدای هر دستور
        lines = [
            line for line in statement.split("\n")
            if not line.strip().startswith("--")
        ]
        statement = "\n".join(lines).strip()
        if statement:
            try:
                cursor.execute(statement)
                executed += 1
            except sqlite3.Error as e:
                logger.warning(f"خطا در اجرای: {statement[:60]}... | {e}")

    conn.commit()
    conn.close()
    logger.success(f"✅ دیتابیس ساخته شد: {db_path} | {executed} دستور اجرا شد")
def get_connection(db_path: Path = DB_PATH):
    """دریافت اتصال به دیتابیس"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def list_tables(db_path: Path = DB_PATH):
    """لیست جدول‌های موجود"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
    conn.close()
    return tables


if __name__ == "__main__":
    create_database(drop_existing=True)
    print("\nجدول‌های ساخته‌شده:")
    for t in list_tables():
        print(f"  - {t}")