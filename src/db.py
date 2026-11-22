"""
ابزارهای مشترک کار با دیتابیس SQLite.

⚠️ این ماژول اسکیما را نمی‌سازد و تغییر نمی‌دهد؛ ساخت جدول‌ها فقط توسط
``src/database/schema.py`` انجام می‌شود. اینجا صرفاً اتصال، upsert و
خواندن داده انجام می‌شود.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import pandas as pd
from loguru import logger

from src.config import get_db_path


# -------------------------------------------------------------------- #
# تاریخ: در دیتابیس، gregorian_date عدد صحیح YYYYMMDD است
# -------------------------------------------------------------------- #
def to_int_date(value: Any) -> Optional[int]:
    """تبدیل تاریخ (رشته/datetime) به عدد صحیح ``YYYYMMDD``."""
    ts = pd.to_datetime(value, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return int(ts.strftime("%Y%m%d"))


def from_int_date(value: Any) -> Optional[pd.Timestamp]:
    """تبدیل عدد ``YYYYMMDD`` به ``pandas.Timestamp``."""
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(str(int(value)), format="%Y%m%d", errors="coerce")


def int_date_series(index: Any) -> "pd.Series":
    """تبدیل یک ایندکس/سری تاریخ به سری عددی ``YYYYMMDD``."""
    ts = pd.to_datetime(pd.Series(list(index)), errors="coerce")
    return ts.dt.strftime("%Y%m%d").astype("Int64")


# -------------------------------------------------------------------- #
# اتصال
# -------------------------------------------------------------------- #
def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """اتصال به SQLite با تنظیمات مناسب نوشتن حجمی."""
    path = Path(db_path) if db_path else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    except sqlite3.Error as exc:
        logger.error("اتصال به دیتابیس «{}» ناموفق بود: {}", path, exc)
        raise


def ensure_schema(db_path: Optional[Path] = None, drop_existing: bool = False) -> None:
    """
    فراخوانی تابع مقداردهی اسکیما در ``src/database/schema.py``.

    نام تابع بین نسخه‌ها متفاوت است، بنابراین چند نام رایج امتحان می‌شود.
    تابع اصلی پروژه ما ``create_database`` است که پارامتر ``drop_existing``
    را می‌پذیرد.

    پارامترها:
        db_path: مسیر دیتابیس (پیش‌فرض از config).
        drop_existing: اگر True باشد، جدول‌های قبلی حذف و از نو ساخته می‌شوند.
    """
    path = Path(db_path) if db_path else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from src.database import schema as schema_module
    except ImportError as exc:
        logger.error("ماژول src/database/schema.py قابل import نبود: {}", exc)
        raise

    # ترتیب اهمیت: create_database تابع اصلی پروژه ماست، بقیه fallback
    for name in (
        "create_database",
        "init_db",
        "create_tables",
        "create_schema",
        "initialize_database",
        "main",
    ):
        func = getattr(schema_module, name, None)
        if not callable(func):
            continue
        try:
            # اول با (path, drop_existing) امتحان می‌کنیم
            try:
                func(str(path), drop_existing=drop_existing)
            except TypeError:
                # اگه تابع این امضا رو نداشت، با path تنها امتحان می‌کنیم
                try:
                    func(str(path))
                except TypeError:
                    # آخرین تلاش: بدون هیچ پارامتری
                    func()
            logger.info("اسکیمای دیتابیس با {}() آماده شد", name)
            return
        except Exception as exc:  # noqa: BLE001 - امضای تابع کاربر ناشناخته است
            logger.warning("اجرای {}() ناموفق بود: {}", name, exc)
    logger.warning("تابع مقداردهی اسکیما پیدا نشد؛ فرض می‌کنیم جدول‌ها از قبل وجود دارند")


# -------------------------------------------------------------------- #
# نوشتن
# -------------------------------------------------------------------- #
def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """لیست ستون‌های واقعی یک جدول (برای هم‌راستایی با اسکیمای موجود)."""
    try:
        return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.Error as exc:
        logger.error("خواندن ساختار جدول {} ناموفق بود: {}", table, exc)
        return []


def table_count(conn: sqlite3.Connection, table: str) -> int:
    """تعداد سطرهای یک جدول (۰ در صورت خطا)."""
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return 0


def _clean(value: Any) -> Any:
    """تبدیل NaN/NaT به None و numpy scalar به نوع پایتونی."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def upsert(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    conflict_keys: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> int:
    """
    درج حجمی با ``ON CONFLICT (...) DO UPDATE`` تا اجرای مکرر بی‌خطر باشد.

    خروجی: تعداد سطرهای ارسال‌شده به دیتابیس.
    """
    payload = [tuple(_clean(v) for v in row) for row in rows]
    if not payload:
        return 0

    existing = set(table_columns(conn, table))
    cols = [c for c in columns if not existing or c in existing]
    if len(cols) != len(columns):
        idx = [i for i, c in enumerate(columns) if c in cols]
        payload = [tuple(row[i] for i in idx) for row in payload]
    if not cols:
        logger.error("هیچ ستون معتبری برای درج در {} پیدا نشد", table)
        return 0

    cols_sql = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    conflict_sql = ", ".join(f'"{c}"' for c in conflict_keys)
    updates = [f'"{c}"=excluded."{c}"' for c in cols if c not in conflict_keys]
    if "updated_at" in existing and "updated_at" not in cols:
        updates.append('"updated_at"=CURRENT_TIMESTAMP')
    if not updates:  # همه ستون‌ها کلید هستند
        updates = [f'"{cols[0]}"=excluded."{cols[0]}"']

    sql = (
        f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict_sql}) DO UPDATE SET {', '.join(updates)}"
    )
    try:
        with conn:
            conn.executemany(sql, payload)
        return len(payload)
    except sqlite3.Error as exc:
        logger.error("درج در جدول {} ناموفق بود: {}", table, exc)
        return 0


def upsert_frame(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    conflict_keys: Sequence[str],
) -> int:
    """upsert یک DataFrame؛ فقط ستون‌هایی که در جدول وجود دارند نوشته می‌شوند."""
    if df is None or df.empty:
        return 0
    existing = set(table_columns(conn, table))
    cols = [c for c in df.columns if c in existing] if existing else list(df.columns)
    if not cols:
        logger.warning("ستون‌های DataFrame با جدول {} هم‌خوانی ندارند", table)
        return 0
    return upsert(conn, table, cols, conflict_keys, df[cols].itertuples(index=False, name=None))


# -------------------------------------------------------------------- #
# خواندن
# -------------------------------------------------------------------- #
def read_sql(query: str, params: Sequence[Any] = (), db_path: Optional[Path] = None) -> pd.DataFrame:
    """اجرای یک کوئری SELECT و برگرداندن DataFrame."""
    conn = connect(db_path)
    try:
        return pd.read_sql_query(query, conn, params=tuple(params))
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        logger.error("اجرای کوئری ناموفق بود: {}", exc)
        return pd.DataFrame()
    finally:
        conn.close()


def load_instruments(db_path: Optional[Path] = None) -> pd.DataFrame:
    """خواندن جدول ``instruments``."""
    return read_sql(
        "SELECT ins_code, symbol, name, english_name, market, industry_group, asset_type "
        "FROM instruments",
        db_path=db_path,
    )


def _symbol_filter(symbols: Optional[Sequence[str]]) -> tuple[str, List[Any]]:
    """ساخت شرط SQL برای فیلتر نمادها."""
    if not symbols:
        return "", []
    marks = ", ".join("?" for _ in symbols)
    return f" AND i.symbol IN ({marks})", list(symbols)


def load_daily_history(
    symbols: Optional[Sequence[str]] = None, db_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    خواندن تاریخچه روزانه همراه با نماد.

    خروجی شامل ستون ``date`` (datetime) علاوه بر ``gregorian_date`` عددی است.
    """
    cond, params = _symbol_filter(symbols)
    df = read_sql(
        "SELECT i.symbol, d.ins_code, d.gregorian_date, d.jalali_date, d.open, d.high, d.low, "
        "d.close, d.last, d.volume, d.value, d.trade_count "
        "FROM daily_history d JOIN instruments i ON i.ins_code = d.ins_code "
        "WHERE 1=1" + cond + " ORDER BY i.symbol, d.gregorian_date",
        params,
        db_path=db_path,
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["gregorian_date"].astype(str), format="%Y%m%d", errors="coerce")
    return df


def load_client_type(
    symbols: Optional[Sequence[str]] = None, db_path: Optional[Path] = None
) -> pd.DataFrame:
    """خواندن داده حقیقی/حقوقی همراه با نماد و ستون ``date``."""
    cond, params = _symbol_filter(symbols)
    df = read_sql(
        "SELECT i.symbol, c.* FROM client_type c "
        "JOIN instruments i ON i.ins_code = c.ins_code "
        "WHERE 1=1" + cond + " ORDER BY i.symbol, c.gregorian_date",
        params,
        db_path=db_path,
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["gregorian_date"].astype(str), format="%Y%m%d", errors="coerce")
    return df


def load_news(symbols: Optional[Sequence[str]] = None, db_path: Optional[Path] = None) -> pd.DataFrame:
    """خواندن اخبار کدال از جدول ``codal_news``."""
    if symbols:
        marks = ", ".join("?" for _ in symbols)
        return read_sql(
            f"SELECT * FROM codal_news WHERE symbol IN ({marks}) ORDER BY publish_date DESC",
            list(symbols),
            db_path=db_path,
        )
    return read_sql("SELECT * FROM codal_news ORDER BY publish_date DESC", db_path=db_path)