"""
کلاینت انتزاعی روی کتابخانه ``algotik-tse``.

همه تماس‌ها با ``algotik_tse`` فقط از این کلاس عبور می‌کنند تا:
    * نام ستون‌ها یک‌بار برای همیشه به اسکیمای دیتابیس نگاشت شود،
    * retry با تأخیر فزاینده و rate-limit در یک نقطه مدیریت شود،
    * لایه‌های بالاتر به تغییرات نسخه کتابخانه وابسته نباشند.

⚠️ ستون‌های نسخه فعلی کتابخانه: خروجی ``get_symbols`` ایندکسش نماد فارسی است و
``instrument_id`` همان InsCode است. ستون‌های ``InsCode``/``Symbol``/``Name`` وجود ندارند.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

try:  # pragma: no cover - وابستگی خارجی
    import algotik_tse as att
except ImportError:  # pragma: no cover
    att = None  # type: ignore[assignment]
    logger.warning("algotik-tse نصب نیست؛ فقط حالت تست با ماژول تزریقی کار می‌کند")


# نگاشت خروجی get_history به ستون‌های جدول daily_history
# نکته: در نسخه فعلی، Close = قیمت پایانی و Final = آخرین قیمت معامله‌شده
_HISTORY_MAP: Dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Final": "last",
    "Volume": "volume",
    "Value": "value",
    "No.": "trade_count",
    "J-Date": "jalali_date",
}

_HISTORY_COLS: Tuple[str, ...] = (
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

# ستون‌های حقیقی/حقوقی (نام‌ها با lower() به ستون‌های DB تبدیل می‌شوند)
_CLIENT_TYPE_COLS: Tuple[str, ...] = (
    "N_buy_retail",
    "N_buy_institutional",
    "N_sell_retail",
    "N_sell_institutional",
    "Vol_buy_retail",
    "Vol_buy_institutional",
    "Vol_sell_retail",
    "Vol_sell_institutional",
    "Val_buy_retail",
    "Val_buy_institutional",
    "Val_sell_retail",
    "Val_sell_institutional",
    "Per_capita_buy_retail",
    "Per_capita_sell_retail",
    "Per_capita_buy_institutional",
    "Per_capita_sell_institutional",
    "Power_retail",
    "Power_institutional",
)

_SYMBOL_SOURCE_COLS: Tuple[str, ...] = (
    "name",
    "instrument_isin",
    "english_name",
    "company_code",
    "company_isin",
    "market",
    "industry_group",
    "asset_type",
    "instrument_id",
)


def _to_int_date(value: Any) -> Optional[int]:
    """تبدیل تاریخ میلادی به عدد ``YYYYMMDD`` (فرمت ستون gregorian_date)."""
    ts = pd.to_datetime(value, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return int(ts.strftime("%Y%m%d"))


class TsetmcClient:
    """پوشش‌دهنده امن روی توابع ``algotik_tse`` با retry، rate-limit و لاگ."""

    def __init__(
        self,
        retries: int = 3,
        retry_delay: float = 1.5,
        request_delay: float = 0.2,
        module: Any = None,
    ) -> None:
        """
        پارامترها:
            retries: تعداد تلاش (پیش‌فرض ۳).
            retry_delay: تأخیر پایه بین تلاش‌ها؛ به‌صورت فزاینده ضرب می‌شود.
            request_delay: مکث اجباری بعد از هر درخواست (رعایت rate limit).
            module: تزریق ماژول جایگزین برای تست با stub (پیش‌فرض ``algotik_tse``).
        """
        self.retries = max(1, int(retries))
        self.retry_delay = float(retry_delay)
        self.request_delay = float(request_delay)
        resolved = module if module is not None else att
        if resolved is None:
            raise ImportError(
                "کتابخانه algotik-tse نصب نیست. نصب: pip install algotik-tse"
            )
        self.att: Any = resolved

    # ------------------------------------------------------------------ #
    # ابزارهای داخلی
    # ------------------------------------------------------------------ #
    def _call(self, func_name: str, label: str, **kwargs: Any) -> Optional[Any]:
        """
        اجرای یک تابع کتابخانه با retry و تأخیر فزاینده.

        در صورت شکست کامل ``None`` برمی‌گرداند تا حلقه backfill با یک نماد
        خراب متوقف نشود.
        """
        func: Optional[Callable[..., Any]] = getattr(self.att, func_name, None)
        if not callable(func):
            logger.error("تابع {} در نسخه نصب‌شده algotik-tse وجود ندارد", func_name)
            return None

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            try:
                result = func(**kwargs)
                time.sleep(self.request_delay)  # رعایت rate limit
                return result
            except TypeError as exc:
                # آرگومان اضافه‌ای که نسخه کاربر پشتیبانی نمی‌کند را حذف می‌کنیم
                dropped = False
                for opt in ("progress", "auto_adjust", "date_format", "output_type"):
                    if opt in kwargs and opt in str(exc):
                        kwargs.pop(opt)
                        dropped = True
                        break
                last_error = exc
                if dropped:
                    logger.warning("آرگومان پشتیبانی‌نشده در {} حذف شد و دوباره تلاش می‌کنیم", label)
                    continue
                logger.warning("خطای آرگومان در {}: {}", label, exc)
                break
            except Exception as exc:  # noqa: BLE001 - خطای شبکه/پارس متنوع است
                last_error = exc
                wait = self.retry_delay * attempt
                logger.warning(
                    "خطا در {} (تلاش {}/{}): {} | {:.1f} ثانیه صبر",
                    label, attempt, self.retries, exc, wait,
                )
                time.sleep(wait)
        logger.error("دریافت {} ناموفق بود: {}", label, last_error)
        return None

    @staticmethod
    def _empty(df: Any) -> bool:
        """بررسی خالی بودن خروجی (None یا DataFrame تهی)."""
        return df is None or not isinstance(df, pd.DataFrame) or df.empty

    # ------------------------------------------------------------------ #
    # لیست ابزارها
    # ------------------------------------------------------------------ #
    def get_symbols(
        self,
        bourse: bool = True,
        farabourse: bool = False,
        payeh: bool = False,
        haghe_taqadom: bool = False,
        sandogh: bool = False,
        bonds: bool = False,
        options: bool = False,
        mortgage: bool = False,
        commodity: bool = False,
        energy: bool = False,
    ) -> pd.DataFrame:
        """
        لیست ابزارهای بازار با نام ستون‌های یکسان‌شده.

        خروجی: ``ins_code (int), symbol, name, english_name, company_code,
        company_isin, instrument_isin, asset_type, market, industry_group``
        """
        raw = self._call(
            "get_symbols",
            "لیست ابزارها",
            bourse=bourse,
            farabourse=farabourse,
            payeh=payeh,
            haghe_taqadom=haghe_taqadom,
            sandogh=sandogh,
            bonds=bonds,
            options=options,
            mortgage=mortgage,
            commodity=commodity,
            energy=energy,
        )
        if self._empty(raw):
            logger.error("لیست ابزارها خالی برگشت")
            return pd.DataFrame()

        df = raw.copy()
        df.index = df.index.astype(str).str.strip()  # ایندکس = نماد فارسی
        df = df.rename_axis("symbol").reset_index()

        missing = [c for c in _SYMBOL_SOURCE_COLS if c not in df.columns]
        if missing:
            logger.warning("ستون‌های زیر در خروجی get_symbols نبودند: {}", missing)
            for col in missing:
                df[col] = None

        out = pd.DataFrame(
            {
                "ins_code": pd.to_numeric(df["instrument_id"], errors="coerce").astype("Int64"),
                "symbol": df["symbol"],
                "name": df["name"],
                "english_name": df["english_name"],
                "company_code": df["company_code"],
                "company_isin": df["company_isin"],
                "instrument_isin": df["instrument_isin"],
                "asset_type": df["asset_type"],
                "market": df["market"],
                "industry_group": df["industry_group"],
            }
        )
        out = out[out["ins_code"].notna()].reset_index(drop=True)
        logger.info("تعداد {} ابزار دریافت شد", len(out))
        return out

    @staticmethod
    def filter_stocks(symbols_df: pd.DataFrame) -> pd.DataFrame:
        """فیلتر کردن فقط ابزارهای ``asset_type == 'stock'``."""
        if symbols_df.empty or "asset_type" not in symbols_df.columns:
            return symbols_df
        mask = symbols_df["asset_type"].astype(str).str.strip().str.lower() == "stock"
        return symbols_df[mask].reset_index(drop=True)

    def validate_symbols(
        self, symbols_list: Sequence[str], symbols_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        بررسی وجود نمادهای هدف در لیست ابزارها.

        خروجی: ``{"available": [...], "missing": [...], "frame": DataFrame}``
        """
        if symbols_df is None:
            symbols_df = self.filter_stocks(self.get_symbols())
        wanted = [str(s).strip() for s in symbols_list]
        if symbols_df is None or symbols_df.empty:
            return {"available": [], "missing": wanted, "frame": pd.DataFrame()}

        found = symbols_df[symbols_df["symbol"].isin(wanted)].copy().reset_index(drop=True)
        available = [s for s in wanted if s in set(found["symbol"])]
        missing = [s for s in wanted if s not in set(found["symbol"])]
        if missing:
            logger.warning("نمادهای پیدا‌نشده: {}", "، ".join(missing))
        logger.info("{} نماد از {} نماد هدف تأیید شد", len(available), len(wanted))
        return {"available": available, "missing": missing, "frame": found}

    # ------------------------------------------------------------------ #
    # تاریخچه روزانه
    # ------------------------------------------------------------------ #
    def get_history(self, symbol: str, start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
        """
        تاریخچه روزانه یک نماد با ستون‌های هم‌نام جدول ``daily_history``.

        پارامترها:
            symbol: نماد فارسی (مثل «فولاد»).
            start, end: تاریخ شروع/پایان (شمسی مثل ``1402-06-24`` یا میلادی).
            auto_adjust: تعدیل قیمت‌ها بابت افزایش سرمایه و سود نقدی.
        """
        raw = self._call(
            "get_history",
            f"تاریخچه «{symbol}»",
            symbol=symbol,
            start=start,
            end=end,
            output_type="full",
            date_format="both",
            auto_adjust=auto_adjust,
            progress=False,
        )
        if self._empty(raw):
            return pd.DataFrame()

        df = raw.copy().rename(columns=_HISTORY_MAP)
        df["gregorian_date"] = [_to_int_date(v) for v in df.index]
        for col in _HISTORY_COLS:
            if col not in df.columns:
                df[col] = None
        out = df[list(_HISTORY_COLS)].reset_index(drop=True)
        return out[out["gregorian_date"].notna()].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # حقیقی / حقوقی
    # ------------------------------------------------------------------ #
    def get_client_type(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """داده حقیقی/حقوقی یک نماد با ستون‌های هم‌نام جدول ``client_type``."""
        raw = self._call(
            "get_client_type",
            f"حقیقی/حقوقی «{symbol}»",
            symbol=symbol,
            start=start,
            end=end,
            output_type="full",
            date_format="both",
            progress=False,
        )
        if self._empty(raw):
            return pd.DataFrame()

        df = raw.copy()
        df = df.rename(columns={c: c.lower() for c in _CLIENT_TYPE_COLS if c in df.columns})
        if "J-Date" in df.columns:
            df = df.rename(columns={"J-Date": "jalali_date"})
        df["gregorian_date"] = [_to_int_date(v) for v in df.index]

        cols = ["gregorian_date", "jalali_date"] + [c.lower() for c in _CLIENT_TYPE_COLS]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        out = df[cols].reset_index(drop=True)
        return out[out["gregorian_date"].notna()].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # داده لحظه‌ای / ریز
    # ------------------------------------------------------------------ #
    def get_order_book(self, symbol: str) -> pd.DataFrame:
        """پنج سطح سفارش (عرضه/تقاضا) یک نماد."""
        raw = self._call("get_order_book", f"عرضه‌وتقاضای «{symbol}»", symbol=symbol)
        return pd.DataFrame() if self._empty(raw) else raw.copy()

    def get_queue(self, symbol: str, side: str = "both") -> pd.DataFrame:
        """صف خرید/فروش یک نماد."""
        raw = self._call("get_queue", f"صف «{symbol}»", symbol=symbol, side=side)
        return pd.DataFrame() if self._empty(raw) else raw.copy()

    def get_trades(self, symbol: str) -> pd.DataFrame:
        """معاملات ریز (tick) یک نماد."""
        raw = self._call("get_trades", f"معاملات ریز «{symbol}»", symbol=symbol)
        return pd.DataFrame() if self._empty(raw) else raw.copy()

    def get_intraday(self, symbol: str, interval: str = "5min") -> pd.DataFrame:
        """کندل‌های درون‌روزی یک نماد."""
        raw = self._call(
            "get_intraday", f"کندل درون‌روزی «{symbol}»", symbol=symbol, interval=interval
        )
        return pd.DataFrame() if self._empty(raw) else raw.copy()

    def get_info(self, symbol: str) -> Optional[Any]:
        """اطلاعات پایه شرکت (ساختار خروجی dict یا DataFrame است)."""
        return self._call("get_info", f"اطلاعات «{symbol}»", symbol=symbol)

    def get_shareholders(self, symbol: str) -> pd.DataFrame:
        """سهامداران عمده یک نماد."""
        raw = self._call("get_shareholders", f"سهامداران «{symbol}»", symbol=symbol)
        return pd.DataFrame() if self._empty(raw) else raw.copy()

    def list_indices(self) -> Optional[Any]:
        """لیست شاخص‌های بازار."""
        return self._call("list_indices", "لیست شاخص‌ها")

    def get_index_companies(self, index_name: str) -> pd.DataFrame:
        """اعضای یک شاخص."""
        raw = self._call("get_index_companies", f"اعضای شاخص «{index_name}»", index_name=index_name)
        return pd.DataFrame() if self._empty(raw) else raw.copy()

    def get_currency(self) -> Optional[Any]:
        """قیمت دلار، طلا و ... (ساختار خروجی به نسخه کتابخانه بستگی دارد)."""
        return self._call("get_currency", "قیمت ارز و طلا")

    def get_market_snapshot(self, symbols: Sequence[str]) -> List[Dict[str, Any]]:
        """
        snapshot لحظه‌ای چند نماد به شکل «لیست dictهای تخت».

        ⚠️ خروجی توابع لحظه‌ای کتابخانه dict تودرتو است، پس هرگز نباید مستقیم به
        ``pd.DataFrame(...)`` داده شود؛ اینجا اول تخت‌سازی می‌شود.
        """
        rows: List[Dict[str, Any]] = []
        for symbol in symbols:
            book = self.get_order_book(symbol)
            if book.empty:
                continue
            flat: Dict[str, Any] = {"symbol": symbol}
            numeric = book.select_dtypes(include="number")
            for col in numeric.columns:
                try:
                    flat[str(col).strip().lower().replace(" ", "_")] = float(numeric[col].iloc[0])
                except (IndexError, TypeError, ValueError):
                    continue
            rows.append(flat)
        return rows
