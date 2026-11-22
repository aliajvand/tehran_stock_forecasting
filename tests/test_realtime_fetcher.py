"""تست fetcher لحظه‌ای با داده جعلی (فاز ۱)."""

from __future__ import annotations

import sqlite3

from src.data_ingestion.backfill import BackfillRunner, jalali_years_back
from src.data_ingestion.realtime_fetcher import RealtimeFetcher
from src.data_ingestion.tsetmc_client import TsetmcClient


def test_fetch_once_writes_snapshot_and_book(
    client: TsetmcClient, temp_db: sqlite3.Connection
) -> None:
    """یک دور دریافت باید در market_snapshot و order_book سطر بنویسد."""
    start, end = jalali_years_back(1)
    runner = BackfillRunner(client, temp_db, start, end)
    runner.run_instruments(runner.fetch_symbols())

    fetcher = RealtimeFetcher(client, temp_db, ["فولاد", "فخوز"], interval_sec=0.0)
    written = fetcher.fetch_once()
    assert written["market_snapshot"] == 2
    assert written["order_book"] == 10  # ۵ سطح × ۲ نماد
    assert temp_db.execute("SELECT COUNT(*) FROM market_snapshot").fetchone()[0] == 2


def test_run_forever_stops_at_max_cycles(
    client: TsetmcClient, temp_db: sqlite3.Connection
) -> None:
    """حلقه باید با max_cycles متوقف شود."""
    fetcher = RealtimeFetcher(client, temp_db, ["فولاد"], interval_sec=0.0)
    assert fetcher.run_forever(max_cycles=2) == 2
