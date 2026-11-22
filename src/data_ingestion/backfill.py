"""
Backfill تاریخی بورس تهران (فاز ۱).

مراحل:
    ۱) لیست ابزارها → جدول ``instruments`` (فقط ``asset_type == 'stock'``)
    ۲) تاریخچه روزانه نمادهای هدف → ``daily_history``
    ۳) حقیقی/حقوقی نمادهای هدف → ``client_type``

اجرا:
    python -m src.data_ingestion.backfill
    python -m src.data_ingestion.backfill --only history
    python -m src.data_ingestion.backfill --years 5 --symbols فولاد فخوز

تست‌پذیری: کلاس ``BackfillRunner`` هر کلاینت دلخواهی (از جمله stub) را می‌پذیرد،
پس می‌توان کل جریان را بدون شبکه و بدون دیتای واقعی اجرا کرد.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger

# اجازه اجرای مستقیم فایل بدون نصب پکیج
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import db as dbm  # noqa: E402
from src.config import (  # noqa: E402
    get_db_path,
    get_section,
    get_target_symbols,
    load_config,
    setup_logging,
)

HISTORY_COLUMNS: Sequence[str] = (
    "ins_code",
    "gregorian_date",
    "jalali_date",
    "open",
    "high",
    "low",
    "close",
    "last",
    "volume",
    "value",
    "trade_count",
)

CLIENT_TYPE_COLUMNS: Sequence[str] = (
    "ins_code",
    "gregorian_date",
    "jalali_date",
    "n_buy_retail",
    "n_buy_institutional",
    "n_sell_retail",
    "n_sell_institutional",
    "vol_buy_retail",
    "vol_buy_institutional",
    "vol_sell_retail",
    "vol_sell_institutional",
    "val_buy_retail",
    "val_buy_institutional",
    "val_sell_retail",
    "val_sell_institutional",
    "per_capita_buy_retail",
    "per_capita_sell_retail",
    "per_capita_buy_institutional",
    "per_capita_sell_institutional",
    "power_retail",
    "power_institutional",
)

INSTRUMENT_COLUMNS: Sequence[str] = (
    "ins_code",
    "symbol",
    "name",
    "english_name",
    "company_code",
    "company_isin",
    "instrument_isin",
    "asset_type",
    "market",
    "industry_group",
)


def jalali_years_back(years: int, end: Optional[date] = None) -> tuple[str, str]:
    """
    بازه تاریخی به فرمت میلادی ``YYYY-MM-DD`` برای ``years`` سال گذشته.

    کتابخانه algotik-tse هم تاریخ شمسی و هم میلادی را می‌پذیرد؛ برای پرهیز از
    ابهام، همیشه میلادی می‌فرستیم.
    """
    last = end or date.today()
    first = last - timedelta(days=int(365.25 * max(1, years)) + 5)
    return first.isoformat(), last.isoformat()


class BackfillRunner:
    """اجرای مرحله‌به‌مرحله backfill روی یک اتصال دیتابیس."""

    def __init__(
        self,
        client: Any,
        conn: sqlite3.Connection,
        start: str,
        end: str,
    ) -> None:
        """
        پارامترها:
            client: نمونه ``TsetmcClient`` یا هر stub با همان متدها.
            conn: اتصال باز SQLite.
            start, end: بازه تاریخی درخواست داده.
        """
        self.client = client
        self.conn = conn
        self.start = start
        self.end = end
        self.stats: Dict[str, int] = {}

    # ---------------------------------------------------------------- #
    def fetch_symbols(self, markets: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
        """دریافت لیست ابزارها و نگه‌داشتن فقط سهام."""
        symbols_df = self.client.get_symbols(**(markets or {"bourse": True}))
        if symbols_df is None or symbols_df.empty:
            logger.error("لیست ابزارها خالی است")
            return pd.DataFrame()
        stocks = self.client.filter_stocks(symbols_df)
        logger.info("{} ابزار از نوع stock از {} ابزار", len(stocks), len(symbols_df))
        return stocks

    def run_instruments(self, stocks: pd.DataFrame) -> int:
        """درج/به‌روزرسانی جدول ``instruments``."""
        if stocks.empty:
            logger.warning("مرحله instruments رد شد (لیست خالی)")
            return 0
        cols = [c for c in INSTRUMENT_COLUMNS if c in stocks.columns]
        count = dbm.upsert(
            self.conn, "instruments", cols, ("ins_code",),
            stocks[cols].itertuples(index=False, name=None),
        )
        self.stats["instruments"] = count
        logger.success("instruments: {} سطر upsert شد", count)
        return count

    def run_daily_history(self, targets: pd.DataFrame) -> int:
        """دریافت و درج تاریخچه روزانه نمادهای هدف."""
        total = 0
        for pos, row in enumerate(targets.itertuples(index=False), start=1):
            logger.info("[{}/{}] تاریخچه «{}»", pos, len(targets), row.symbol)
            df = self.client.get_history(symbol=row.symbol, start=self.start, end=self.end)
            if df is None or df.empty:
                logger.warning("تاریخچه «{}» خالی بود", row.symbol)
                continue
            df = df.copy()
            df["ins_code"] = int(row.ins_code)
            cols = [c for c in HISTORY_COLUMNS if c in df.columns]
            n = dbm.upsert(
                self.conn, "daily_history", cols, ("ins_code", "gregorian_date"),
                df[cols].itertuples(index=False, name=None),
            )
            total += n
            logger.info(
                "«{}»: {} سطر ({} تا {})", row.symbol, n,
                df["gregorian_date"].min(), df["gregorian_date"].max(),
            )
        self.stats["daily_history"] = total
        logger.success("daily_history: مجموع {} سطر upsert شد", total)
        return total

    def run_client_type(self, targets: pd.DataFrame) -> int:
        """دریافت و درج داده حقیقی/حقوقی نمادهای هدف."""
        total = 0
        for pos, row in enumerate(targets.itertuples(index=False), start=1):
            logger.info("[{}/{}] حقیقی/حقوقی «{}»", pos, len(targets), row.symbol)
            df = self.client.get_client_type(symbol=row.symbol, start=self.start, end=self.end)
            if df is None or df.empty:
                logger.warning("حقیقی/حقوقی «{}» خالی بود", row.symbol)
                continue
            df = df.copy()
            df["ins_code"] = int(row.ins_code)
            cols = [c for c in CLIENT_TYPE_COLUMNS if c in df.columns]
            n = dbm.upsert(
                self.conn, "client_type", cols, ("ins_code", "gregorian_date"),
                df[cols].itertuples(index=False, name=None),
            )
            total += n
            logger.info("«{}»: {} سطر حقیقی/حقوقی", row.symbol, n)
        self.stats["client_type"] = total
        logger.success("client_type: مجموع {} سطر upsert شد", total)
        return total

    def report(self) -> Dict[str, int]:
        """گزارش نهایی: تعداد سطرهای واقعی هر جدول در دیتابیس."""
        counts = {
            table: dbm.table_count(self.conn, table)
            for table in ("instruments", "daily_history", "client_type")
        }
        logger.info("──────── گزارش نهایی Backfill ────────")
        for table, count in counts.items():
            logger.info("{:<15}: {:>8} سطر", table, count)
        return counts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """پارس آرگومان‌های خط فرمان (بدون ``input()``)."""
    parser = argparse.ArgumentParser(description="Backfill تاریخی بورس تهران")
    parser.add_argument("--years", type=int, default=None, help="تعداد سال گذشته (پیش‌فرض از config)")
    parser.add_argument("--symbols", nargs="*", default=None, help="محدود کردن به نمادهای مشخص")
    parser.add_argument(
        "--only",
        choices=["all", "instruments", "history", "client_type"],
        default="all",
        help="اجرای فقط یک مرحله",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """نقطه ورود اسکریپت backfill."""
    args = parse_args(argv)
    setup_logging()
    config = load_config()
    data_cfg = get_section("data", {})

    years = int(args.years or data_cfg.get("history_years", 3))
    start, end = jalali_years_back(years)
    db_path = get_db_path()
    logger.info("بازه: {} تا {} | دیتابیس: {}", start, end, db_path)

    from src.data_ingestion.tsetmc_client import TsetmcClient  # import تنبل برای تست‌پذیری

    client = TsetmcClient(
        retries=3,
        retry_delay=1.5,
        request_delay=float(data_cfg.get("rate_limit_delay_sec", 0.2)),
    )

    dbm.ensure_schema(db_path)
    conn = dbm.connect(db_path)
    try:
        runner = BackfillRunner(client, conn, start, end)
        stocks = runner.fetch_symbols(config.get("markets") or {"bourse": True})
        if stocks.empty:
            logger.error("بدون لیست ابزارها ادامه ممکن نیست")
            return 1

        if args.only in ("all", "instruments"):
            runner.run_instruments(stocks)

        wanted: List[str] = list(args.symbols) if args.symbols else get_target_symbols()
        check = client.validate_symbols(wanted, stocks)
        targets = check["frame"]
        if targets.empty:
            logger.error("هیچ نماد هدفی تأیید نشد؛ خروج")
            return 1

        if args.only in ("all", "history"):
            runner.run_daily_history(targets)
        if args.only in ("all", "client_type"):
            runner.run_client_type(targets)

        runner.report()
        logger.success("Backfill تمام شد ({} نماد)", len(targets))
        return 0
    finally:
        conn.close()
        logger.info("اتصال دیتابیس بسته شد")


if __name__ == "__main__":
    raise SystemExit(main())
