"""تست backfill با کلاینت جعلی و دیتابیس موقت (فاز ۱)."""

from __future__ import annotations

import sqlite3

from src.data_ingestion.backfill import BackfillRunner, jalali_years_back
from src.data_ingestion.tsetmc_client import TsetmcClient


def _runner(client: TsetmcClient, conn: sqlite3.Connection) -> BackfillRunner:
    """ساخت runner با بازه سه‌ساله."""
    start, end = jalali_years_back(3)
    return BackfillRunner(client, conn, start, end)


def test_instruments_only_stocks(client: TsetmcClient, temp_db: sqlite3.Connection) -> None:
    """فقط سهام باید در instruments درج شود."""
    runner = _runner(client, temp_db)
    stocks = runner.fetch_symbols()
    runner.run_instruments(stocks)
    types = [r[0] for r in temp_db.execute("SELECT DISTINCT asset_type FROM instruments")]
    assert types == ["stock"]
    count = temp_db.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]
    assert count == len(stocks)


def test_history_and_client_type_written(client: TsetmcClient, temp_db: sqlite3.Connection) -> None:
    """تاریخچه و حقیقی/حقوقی باید نوشته شوند و ins_code عددی باشد."""
    runner = _runner(client, temp_db)
    stocks = runner.fetch_symbols()
    runner.run_instruments(stocks)
    targets = client.validate_symbols(["فولاد", "فخوز"], stocks)["frame"]

    assert runner.run_daily_history(targets) > 0
    assert runner.run_client_type(targets) > 0

    row = temp_db.execute(
        "SELECT ins_code, gregorian_date, close, last, trade_count FROM daily_history LIMIT 1"
    ).fetchone()
    assert isinstance(row[0], int)
    assert 19000101 <= row[1] <= 21001231
    assert row[2] is not None and row[4] is not None


def test_upsert_is_idempotent(client: TsetmcClient, temp_db: sqlite3.Connection) -> None:
    """اجرای دوباره نباید سطر تکراری بسازد."""
    runner = _runner(client, temp_db)
    stocks = runner.fetch_symbols()
    runner.run_instruments(stocks)
    targets = client.validate_symbols(["فولاد"], stocks)["frame"]

    runner.run_daily_history(targets)
    first = temp_db.execute("SELECT COUNT(*) FROM daily_history").fetchone()[0]
    runner.run_daily_history(targets)
    second = temp_db.execute("SELECT COUNT(*) FROM daily_history").fetchone()[0]
    assert first == second > 0


def test_report_counts_tables(client: TsetmcClient, temp_db: sqlite3.Connection) -> None:
    """گزارش نهایی باید تعداد سطر هر جدول را بدهد."""
    runner = _runner(client, temp_db)
    stocks = runner.fetch_symbols()
    runner.run_instruments(stocks)
    report = runner.report()
    assert report["instruments"] > 0
    assert set(report) == {"instruments", "daily_history", "client_type"}


def test_date_range_spans_three_years() -> None:
    """بازه پیش‌فرض باید حدود سه سال باشد."""
    start, end = jalali_years_back(3)
    assert start < end
    assert int(end[:4]) - int(start[:4]) == 3
