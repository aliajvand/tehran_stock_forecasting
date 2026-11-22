"""
Collector بلادرنگ — دریافت snapshot از TSETMC و ذخیره در دیتابیس.

استراتژی:
    * از ``att.get_market_snapshot()`` استفاده می‌کنیم (یک call = همه چیز).
    * فیلتر به ۱۸ نماد هدف با ``InsCode``.
    * فلگ‌های freshness رو چک می‌کنیم قبل از ذخیره.
    * حلقه: هر ۱۰ ثانیه در ساعات بازار.

اجرا:
    python -m src.data_ingestion.realtime_collector
    python -m src.data_ingestion.realtime_collector --once
    python -m src.data_ingestion.realtime_collector --interval 5
"""

from __future__ import annotations

import argparse
import signal
import sqlite3
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import db as dbm  # noqa: E402
from src.config import (  # noqa: E402
    get_db_path,
    get_section,
    get_target_symbols,
    setup_logging,
)


# زمان‌های بازار (به وقت تهران)
MARKET_OPEN = dtime(8, 55)     # کمی قبل از باز شدن
MARKET_CLOSE = dtime(12, 35)   # کمی بعد از بسته شدن
WEEKEND_DAYS = {3, 4}          # پنجشنبه=3، جمعه=4 (پایتون: Mon=0)


def is_market_open(now: Optional[datetime] = None) -> bool:
    """آیا الان بازار بازه (یا نزدیک به باز شدن)؟"""
    now = now or datetime.now()
    if now.weekday() in WEEKEND_DAYS:
        return False
    t = now.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


class RealtimeCollector:
    """حلقه دریافت snapshot لحظه‌ای و ذخیره در دیتابیس."""

    def __init__(
        self,
        interval_sec: float = 10.0,
        target_symbols: Optional[Sequence[str]] = None,
    ) -> None:
        self.interval_sec = float(interval_sec)
        self.target_symbols = set(target_symbols or get_target_symbols())
        self.conn = dbm.connect(get_db_path())
        self._running = False

        # نقشه symbol → ins_code از جدول instruments
        self.symbol_to_ins = self._load_symbol_map()
        self.target_ins_codes = set(self.symbol_to_ins.get(s) for s in self.target_symbols)
        self.target_ins_codes.discard(None)
        logger.info(
            "Collector آماده | نمادهای هدف: {} | ins_codes: {}",
            len(self.target_symbols), len(self.target_ins_codes),
        )

    def _load_symbol_map(self) -> Dict[str, int]:
        """نگاشت symbol → ins_code از جدول instruments."""
        try:
            rows = self.conn.execute(
                "SELECT symbol, ins_code FROM instruments"
            ).fetchall()
            return {str(s): int(c) for s, c in rows if c is not None}
        except sqlite3.Error as exc:
            logger.error("خواندن instruments ناموفق: {}", exc)
            return {}

    # ------------------------------------------------------------------ #
    # دریافت و ذخیره
    # ------------------------------------------------------------------ #
    def fetch_snapshot(self) -> Optional[Dict[str, Any]]:
        """فراخوانی get_market_snapshot با مدیریت خطا."""
        try:
            from algotik_tse import get_market_snapshot
            snap = get_market_snapshot()
            if not isinstance(snap, dict):
                logger.warning("snapshot dict نیست: {}", type(snap))
                return None
            return snap
        except Exception as exc:  # noqa: BLE001
            logger.error("خطا در دریافت snapshot: {}", exc)
            return None

    def save_snapshot(self, snap: Dict[str, Any]) -> Dict[str, int]:
        """ذخیره stocks و order_book در دیتابیس."""
        counts = {"snapshot": 0, "order_book": 0}

        # زمان snapshot
        exchange_time = snap.get("exchange_time")
        if exchange_time is None:
            logger.warning("exchange_time نداریم؛ رد شد")
            return counts
        try:
            snapshot_ts = int(pd.Timestamp(exchange_time).timestamp())
        except Exception:
            snapshot_ts = int(time.time())

        # چک کیفیت
        if snap.get("is_stale", False):
            logger.warning("Snapshot قدیمی است (is_stale=True)")
            return counts

        # ۱. ذخیره stocks
        stocks = snap.get("stocks")
        if isinstance(stocks, pd.DataFrame) and not stocks.empty:
            # 🔧 تبدیل InsCode به عدد (نسخه TSETMC رشته می‌دهد)
            stocks = stocks.copy()
            stocks["InsCode"] = pd.to_numeric(stocks["InsCode"], errors="coerce")
            filtered = stocks[stocks["InsCode"].isin(self.target_ins_codes)]
            logger.debug("stocks: {} → {} (پس از فیلتر)", len(stocks), len(filtered))
            
            if not filtered.empty:
                rows = []
                for _, r in filtered.iterrows():
                    rows.append((
                        int(r["InsCode"]),
                        snapshot_ts,
                        _f(r.get("Last")),
                        _f(r.get("Close")),
                        _i(r.get("Volume")),
                        _f(r.get("Value")),
                        _i(r.get("TradeCount")),
                        _f(r.get("Open")),
                        _f(r.get("High")),
                        _f(r.get("Low")),
                        _f(r.get("ChangePct")),
                    ))
                n = dbm.upsert(
                    self.conn, "market_snapshot",
                    ("ins_code", "snapshot_time", "last", "close", "volume",
                    "value", "trade_count", "open", "high", "low", "change_pct"),
                    ("ins_code", "snapshot_time"),
                    rows,
                )
                counts["snapshot"] = n

        # ۲. ذخیره order_book
        book = snap.get("order_book")
        if isinstance(book, pd.DataFrame) and not book.empty:
            book = book.copy()
            book["InsCode"] = pd.to_numeric(book["InsCode"], errors="coerce")
            filtered_b = book[book["InsCode"].isin(self.target_ins_codes)]
            logger.debug("order_book: {} → {} (پس از فیلتر)", len(book), len(filtered_b))
            
            if not filtered_b.empty:
                rows_b = []
                for _, r in filtered_b.iterrows():
                    rows_b.append((
                        int(r["InsCode"]),
                        snapshot_ts,
                        int(r["Level"]) if pd.notna(r["Level"]) else None,
                        _f(r.get("BidPrice")),
                        _i(r.get("BidVolume")),
                        _f(r.get("AskPrice")),
                        _i(r.get("AskVolume")),
                        _i(r.get("BidOrderCount")),
                        _i(r.get("AskOrderCount")),
                    ))
                n_b = dbm.upsert(
                    self.conn, "order_book",
                    ("ins_code", "snapshot_time", "level",
                    "bid_price", "bid_volume", "ask_price", "ask_volume",
                    "bid_order_count", "ask_order_count"),
                    ("ins_code", "snapshot_time", "level"),
                    rows_b,
                )
                counts["order_book"] = n_b

        return counts
    # ------------------------------------------------------------------ #
    def run_forever(self, max_cycles: Optional[int] = None) -> int:
        """حلقه اصلی Collector."""
        self._running = True
        cycles = 0
        logger.info("🚀 Collector شروع | بازه: {}s | بازار: {} - {}",
                    self.interval_sec, MARKET_OPEN, MARKET_CLOSE)

        while self._running:
            started = time.monotonic()
            now = datetime.now()

            if not is_market_open(now):
                if cycles == 0:
                    logger.info("⏸️ بازار بسته ({}) — تا ۶۰ ثانیه دیگر چک می‌کنم",
                                now.strftime("%a %H:%M"))
                time.sleep(60)
                continue

            snap = self.fetch_snapshot()
            if snap is not None:
                counts = self.save_snapshot(snap)
                age = snap.get("snapshot_age_seconds", -1)
                logger.info(
                    "📸 {} | snapshot: {} | order_book: {} | age: {:.0f}s",
                    now.strftime("%H:%M:%S"),
                    counts["snapshot"], counts["order_book"], age or 0,
                )

            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break

            elapsed = time.monotonic() - started
            time.sleep(max(0.0, self.interval_sec - elapsed))

        logger.info("Collector خاتمه یافت ({} دور)", cycles)
        return cycles

    def stop(self, signum: Optional[int] = None, frame: Optional[FrameType] = None) -> None:
        """توقف تمیز."""
        logger.warning("⏹️ درخواست توقف")
        self._running = False

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------- #
# کمکی‌ها
# ---------------------------------------------------------------------- #
def _f(v: Any) -> Optional[float]:
    """تبدیل ایمن به float."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> Optional[int]:
    """تبدیل ایمن به int."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------- #
# ورود
# ---------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collector بلادرنگ TSETMC")
    parser.add_argument("--once", action="store_true", help="فقط یک دور اجرا کن")
    parser.add_argument("--interval", type=float, default=None,
                        help="فاصله بین دورها (ثانیه)")
    parser.add_argument("--force", action="store_true",
                        help="بدون توجه به ساعت بازار اجرا کن (برای تست)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()

    data_cfg = get_section("data", {})
    interval = float(args.interval or data_cfg.get("ingestion_interval_sec", 10))

    # 🔧 پچ محلی: is_market_open رو توی همین ماژول override کن
    if args.force:
        global is_market_open
        is_market_open = lambda now=None: True   # noqa: E731
        logger.warning("⚠️ حالت --force فعال شد؛ چک ساعت بازار غیرفعال است")

    collector = RealtimeCollector(interval_sec=interval)
    signal.signal(signal.SIGINT, collector.stop)
    signal.signal(signal.SIGTERM, collector.stop)

    try:
        collector.run_forever(max_cycles=1 if args.once else None)
        return 0
    finally:
        collector.close()


if __name__ == "__main__":
    raise SystemExit(main())