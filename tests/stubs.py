"""
ماژول جعلی ``algotik_tse`` برای تست بدون شبکه.

ساختار ستون‌ها عیناً مطابق نسخه نصب‌شده کاربر است تا تست‌ها واقعاً قرارداد
داده را بسنجند (ایندکس = نماد فارسی، ``instrument_id`` = InsCode و ...).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS: List[str] = [
    "فولاد", "فخوز", "ذوب", "شپنا", "شتران", "تاپیکو", "وبملت", "وتجارت",
    "وبصادر", "خودرو", "خساپا", "دارو", "تیپیکو", "همراه", "اخابر",
    "وپارس", "سرمایه", "غبشهر",
]

CLIENT_TYPE_COLUMNS: List[str] = [
    "N_buy_retail", "N_buy_institutional", "N_sell_retail", "N_sell_institutional",
    "Vol_buy_retail", "Vol_buy_institutional", "Vol_sell_retail", "Vol_sell_institutional",
    "Val_buy_retail", "Val_buy_institutional", "Val_sell_retail", "Val_sell_institutional",
    "Per_capita_buy_retail", "Per_capita_sell_retail", "Per_capita_buy_institutional",
    "Per_capita_sell_institutional", "Power_retail", "Power_institutional",
]


class FakeAlgotik:
    """جایگزین ماژول ``algotik_tse`` با همان امضای توابع."""

    def __init__(self, symbols: Sequence[str] | None = None, rows: int = 200, seed: int = 7) -> None:
        """پارامترها: لیست نمادها، تعداد روز داده و seed تکرارپذیری."""
        self.symbols = list(symbols or DEFAULT_SYMBOLS)
        self.rows = int(rows)
        self.rng = np.random.default_rng(seed)
        self.calls: Dict[str, int] = {}

    def _track(self, name: str) -> None:
        """شمارش فراخوانی‌ها (برای assert در تست‌ها)."""
        self.calls[name] = self.calls.get(name, 0) + 1

    def _dates(self) -> pd.DatetimeIndex:
        """ایندکس تاریخ روزهای کاری."""
        index = pd.bdate_range(end=pd.Timestamp("2025-09-01"), periods=self.rows)
        return pd.DatetimeIndex(index, name="Date")

    # -------------------------------------------------------------- #
    def get_symbols(self, **kwargs: Any) -> pd.DataFrame:
        """لیست ابزارها؛ یک ETF هم اضافه می‌شود تا فیلتر stock تست شود."""
        self._track("get_symbols")
        rows: List[Dict[str, Any]] = []
        names = self.symbols + ["اهرم"]
        for i, symbol in enumerate(names):
            rows.append(
                {
                    "name": f"شرکت {symbol}",
                    "instrument_isin": f"IRO1{i:07d}",
                    "english_name": f"COMPANY{i}",
                    "company_code": f"CD{i:03d}",
                    "company_isin": f"IRC{i:07d}",
                    "market": "بازار اول (تابلوی اصلی) بورس",
                    "industry_group": "فلزات اساسی",
                    "asset_type": "stock" if symbol != "اهرم" else "fund",
                    "instrument_id": 59612098290740355 + i,
                }
            )
        return pd.DataFrame(rows, index=pd.Index(names))

    def get_history(
        self,
        symbol: str,
        start: str,
        end: str,
        output_type: str = "full",
        date_format: str = "both",
        auto_adjust: bool = True,
        progress: bool = False,
    ) -> pd.DataFrame:
        """تاریخچه روزانه جعلی با گشت تصادفی + یک NaN برای تست پاک‌سازی."""
        self._track("get_history")
        index = self._dates()
        n = len(index)
        close = 1000 + np.cumsum(self.rng.normal(0, 12, n))
        frame = pd.DataFrame(
            {
                "Open": close * (1 + self.rng.normal(0, 0.004, n)),
                "High": close * (1 + abs(self.rng.normal(0, 0.008, n))),
                "Low": close * (1 - abs(self.rng.normal(0, 0.008, n))),
                "Close": close,
                "Final": close * (1 + self.rng.normal(0, 0.002, n)),
                "Volume": self.rng.integers(10_000, 5_000_000, n),
                "No.": self.rng.integers(100, 9_000, n),
                "Value": close * self.rng.integers(10_000, 900_000, n),
                "J-Date": ["1404-06-10"] * n,
                "Weekday": ["Saturday"] * n,
                "Weekday_fa": ["شنبه"] * n,
                "Ticker": [f"شرکت {symbol}"] * n,
            },
            index=index,
        )
        frame.iloc[5, frame.columns.get_loc("Open")] = np.nan
        return frame

    def get_client_type(
        self,
        symbol: str,
        start: str,
        end: str,
        output_type: str = "full",
        date_format: str = "both",
        progress: bool = False,
    ) -> pd.DataFrame:
        """داده حقیقی/حقوقی جعلی با همه ۱۸ ستون عددی."""
        self._track("get_client_type")
        index = self._dates()
        n = len(index)
        data = {col: self.rng.random(n) * 1e7 for col in CLIENT_TYPE_COLUMNS}
        frame = pd.DataFrame(data, index=index)
        frame["Power_retail"] = self.rng.random(n) * 3
        frame["Power_institutional"] = self.rng.random(n) * 3
        frame["J-Date"] = "1404-06-10"
        frame["Weekday"] = "Saturday"
        frame["Weekday_fa"] = "شنبه"
        frame["Ticker"] = f"شرکت {symbol}"
        return frame

    def get_order_book(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """پنج سطح سفارش جعلی."""
        self._track("get_order_book")
        return pd.DataFrame(
            {
                "buy_price": [1000, 999, 998, 997, 996],
                "buy_volume": [500, 400, 300, 200, 100],
                "sell_price": [1001, 1002, 1003, 1004, 1005],
                "sell_volume": [450, 350, 250, 150, 50],
                "last": [1000.5] * 5,
                "volume": [100_000] * 5,
            }
        )

    def get_queue(self, symbol: str, side: str = "both", **kwargs: Any) -> pd.DataFrame:
        """صف خرید/فروش جعلی."""
        self._track("get_queue")
        return pd.DataFrame({"side": ["buy", "sell"], "volume": [1000, 2000]})

    def get_trades(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """معاملات ریز جعلی."""
        self._track("get_trades")
        return pd.DataFrame({"time": ["09:00", "09:01"], "price": [1000, 1001], "volume": [10, 20]})

    def get_intraday(self, symbol: str, interval: str = "5min", **kwargs: Any) -> pd.DataFrame:
        """کندل درون‌روزی جعلی."""
        self._track("get_intraday")
        return pd.DataFrame({"open": [1000], "close": [1002], "volume": [5000]})

    def get_info(self, symbol: str, **kwargs: Any) -> Dict[str, Any]:
        """اطلاعات پایه جعلی."""
        self._track("get_info")
        return {"symbol": symbol, "eps": 500, "pe": 6.2}


class FlakyAlgotik(FakeAlgotik):
    """نسخه‌ای که چند بار اول خطا می‌دهد تا مکانیزم retry تست شود."""

    def __init__(self, fail_times: int = 2, **kwargs: Any) -> None:
        """پارامترها: تعداد شکست اولیه."""
        super().__init__(**kwargs)
        self.fail_times = int(fail_times)
        self.attempts = 0

    def get_history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """شکست کنترل‌شده در چند تلاش اول."""
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ConnectionError("شبکه در دسترس نیست (تست)")
        return super().get_history(*args, **kwargs)


SCHEMA_SQL: List[str] = [
    """CREATE TABLE IF NOT EXISTS instruments (
        ins_code INTEGER PRIMARY KEY, symbol TEXT NOT NULL, name TEXT, english_name TEXT,
        company_code TEXT, company_isin TEXT, instrument_isin TEXT, asset_type TEXT,
        market TEXT, sector_code TEXT, industry_group TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS daily_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ins_code INTEGER, gregorian_date INTEGER,
        jalali_date TEXT, open REAL, high REAL, low REAL, close REAL, last REAL,
        volume INTEGER, value REAL, trade_count INTEGER,
        UNIQUE(ins_code, gregorian_date))""",
    """CREATE TABLE IF NOT EXISTS client_type (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ins_code INTEGER, gregorian_date INTEGER,
        jalali_date TEXT, n_buy_retail REAL, n_buy_institutional REAL, n_sell_retail REAL,
        n_sell_institutional REAL, vol_buy_retail REAL, vol_buy_institutional REAL,
        vol_sell_retail REAL, vol_sell_institutional REAL, val_buy_retail REAL,
        val_buy_institutional REAL, val_sell_retail REAL, val_sell_institutional REAL,
        per_capita_buy_retail REAL, per_capita_sell_retail REAL,
        per_capita_buy_institutional REAL, per_capita_sell_institutional REAL,
        power_retail REAL, power_institutional REAL,
        UNIQUE(ins_code, gregorian_date))""",
    """CREATE TABLE IF NOT EXISTS order_book (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ins_code INTEGER, gregorian_date INTEGER,
        snapshot_time TEXT, level INTEGER, buy_price REAL, buy_volume REAL,
        sell_price REAL, sell_volume REAL)""",
    """CREATE TABLE IF NOT EXISTS market_snapshot (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ins_code INTEGER, gregorian_date INTEGER,
        snapshot_time TEXT, last_price REAL, volume REAL)""",
    """CREATE TABLE IF NOT EXISTS codal_news (
        id INTEGER PRIMARY KEY AUTOINCREMENT, news_id TEXT UNIQUE, ins_code INTEGER,
        symbol TEXT, title TEXT, content TEXT, news_type TEXT, publish_date TEXT,
        publish_time TEXT, url TEXT, attachment_url TEXT, sentiment REAL, category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
]


def create_test_schema(db_path: str) -> None:
    """
    ساخت اسکیمای معادل برای تست.

    ⚠️ فقط برای تست است؛ اسکیمای واقعی پروژه در ``src/database/schema.py`` است و
    این تابع آن فایل را نمی‌خواند و تغییر نمی‌دهد.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        for statement in SCHEMA_SQL:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()
