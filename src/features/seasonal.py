"""
فیچرهای فصلی و تقویمی (تقویم شمسی + ماه‌های قمری + تجزیه فصلی + SARIMA).

نکته: ``jdatetime`` برای تاریخ شمسی و ``statsmodels`` برای تجزیه فصلی و SARIMA
استفاده می‌شود؛ اگر نصب نباشند، تابع‌ها با هشدار از کار می‌گذرند و کرش نمی‌کنند.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_section

try:  # pragma: no cover
    import jdatetime
except ImportError:  # pragma: no cover
    jdatetime = None  # type: ignore[assignment]
    logger.warning("jdatetime نصب نیست؛ فیچرهای تقویم شمسی محدود می‌شوند")


# فصل‌های شمسی: بهار ۱ | تابستان ۲ | پاییز ۳ | زمستان ۴
def _jalali_parts(value: pd.Timestamp) -> Tuple[int, int, int]:
    """تبدیل تاریخ میلادی به (سال، ماه، روز) شمسی."""
    if jdatetime is None or pd.isna(value):
        return (0, 0, 0)
    jd = jdatetime.date.fromgregorian(date=pd.Timestamp(value).date())
    return (jd.year, jd.month, jd.day)


def _hijri_month(value: pd.Timestamp) -> int:
    """
    تخمین ماه قمری با تقویم جدولی (برای شناسایی رمضان و محرم).

    دقت ±۱ روز است که برای فیچرسازی کافی است.
    """
    if pd.isna(value):
        return 0
    ts = pd.Timestamp(value)
    # شماره روز جولیان
    a = (14 - ts.month) // 12
    y = ts.year + 4800 - a
    m = ts.month + 12 * a - 3
    jdn = ts.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    days = jdn - 1948440 + 10632
    n = (days - 1) // 10631
    days = days - 10631 * n + 354
    j = ((10985 - days) // 5316) * ((50 * days) // 17719) + (days // 5670) * ((43 * days) // 15238)
    days = (
        days
        - ((30 - j) // 15) * ((17719 * j) // 50)
        - (j // 16) * ((15238 * j) // 43)
        + 29
    )
    month = (24 * days) // 709
    return int(min(12, max(1, month)))


def add_calendar_features(df: pd.DataFrame, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    افزودن فیچرهای تقویمی: ماه شمسی، فصل، هفتهٔ ماه، روز هفته، رمضان و محرم.

    ستون تاریخ: ``date`` (datetime) یا ``gregorian_date`` عددی ``YYYYMMDD``.
    """
    if df is None or df.empty:
        logger.warning("add_calendar_features: ورودی خالی است")
        return pd.DataFrame()

    out = df.copy()
    if date_column and date_column in out.columns:
        dates = pd.to_datetime(out[date_column], errors="coerce")
    elif "date" in out.columns:
        dates = pd.to_datetime(out["date"], errors="coerce")
    elif "gregorian_date" in out.columns:
        dates = pd.to_datetime(out["gregorian_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        raise ValueError("ستون تاریخ (date یا gregorian_date) پیدا نشد")

    parts = [_jalali_parts(v) for v in dates]
    out["jalali_year"] = [p[0] for p in parts]
    out["jalali_month"] = [p[1] for p in parts]
    out["jalali_day"] = [p[2] for p in parts]
    out["jalali_season"] = [(p[1] - 1) // 3 + 1 if p[1] else 0 for p in parts]
    out["week_of_month"] = [(p[2] - 1) // 7 + 1 if p[2] else 0 for p in parts]
    out["day_of_week"] = dates.dt.weekday          # شنبه در ایران = ۵ در استاندارد پایتون
    out["is_month_end"] = [1 if p[2] >= 27 else 0 for p in parts]
    out["is_year_end"] = [1 if p[1] == 12 else 0 for p in parts]  # اسفند

    hijri = [_hijri_month(v) for v in dates]
    out["hijri_month"] = hijri
    out["is_ramadan"] = [1 if m == 9 else 0 for m in hijri]
    out["is_muharram"] = [1 if m == 1 else 0 for m in hijri]

    # کدگذاری دوره‌ای برای ماه (تا مدل فاصله اسفند-فروردین را بفهمد)
    month = pd.Series(out["jalali_month"], index=out.index).replace(0, np.nan)
    out["month_sin"] = np.sin(2 * np.pi * month / 12).fillna(0.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12).fillna(0.0)
    logger.info("فیچرهای تقویمی اضافه شد ({} سطر)", len(out))
    return out


def decompose_seasonal(
    series: pd.Series, period: Optional[int] = None, model: str = "additive"
) -> pd.DataFrame:
    """
    تجزیه فصلی سری زمانی با ``statsmodels.seasonal_decompose``.

    خروجی: DataFrame با ستون‌های ``trend, seasonal, resid``.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    per = int(period or get_section("features", {}).get("seasonal_period", 22))
    if len(clean) < 2 * per:
        logger.warning("داده برای تجزیه فصلی کافی نیست ({} < {})", len(clean), 2 * per)
        return pd.DataFrame(index=series.index, columns=["trend", "seasonal", "resid"], dtype=float)
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        result = seasonal_decompose(clean.to_numpy(), period=per, model=model, extrapolate_trend="freq")
        return pd.DataFrame(
            {"trend": result.trend, "seasonal": result.seasonal, "resid": result.resid},
            index=clean.index,
        ).reindex(series.index)
    except Exception as exc:  # noqa: BLE001
        logger.error("تجزیه فصلی ناموفق بود: {}", exc)
        return pd.DataFrame(index=series.index, columns=["trend", "seasonal", "resid"], dtype=float)


def add_seasonal_decomposition(
    df: pd.DataFrame, column: str = "close", period: Optional[int] = None
) -> pd.DataFrame:
    """افزودن ستون‌های ``seasonal_trend/seasonal_seasonal/seasonal_resid`` به هر نماد."""
    if df is None or df.empty or column not in df.columns:
        return df if df is not None else pd.DataFrame()

    def _one(group: pd.DataFrame) -> pd.DataFrame:
        """تجزیه فصلی یک نماد."""
        g = group.copy()
        comp = decompose_seasonal(g[column], period=period)
        g["seasonal_trend"] = comp["trend"].to_numpy()
        g["seasonal_seasonal"] = comp["seasonal"].to_numpy()
        g["seasonal_resid"] = comp["resid"].to_numpy()
        return g

    if "symbol" in df.columns:
        parts = [_one(g) for _, g in df.groupby("symbol", sort=False)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _one(df).reset_index(drop=True)


def add_sarima_forecast(
    df: pd.DataFrame,
    column: str = "close",
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 5),
    steps: int = 1,
) -> pd.DataFrame:
    """
    افزودن پیش‌بینی SARIMA به‌عنوان فیچر کمکی.

    برای هر نماد، مدل روی کل سری برازش می‌شود و مقدار برازش‌شده (in-sample) به
    همراه پیش‌بینی گام بعد در ستون ``sarima_pred`` و انحراف آن در
    ``sarima_gap`` قرار می‌گیرد.
    """
    if df is None or df.empty or column not in df.columns:
        return df if df is not None else pd.DataFrame()

    def _one(group: pd.DataFrame) -> pd.DataFrame:
        """برازش SARIMA روی یک نماد."""
        g = group.copy()
        series = pd.to_numeric(g[column], errors="coerce").ffill().dropna()
        g["sarima_pred"] = np.nan
        g["sarima_gap"] = np.nan
        if len(series) < 60:
            logger.debug("داده SARIMA کافی نیست ({} سطر)", len(series))
            return g
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            model = SARIMAX(
                series.to_numpy(),
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            in_sample = pd.Series(fitted.fittedvalues, index=series.index)
            g.loc[in_sample.index, "sarima_pred"] = in_sample
            g["sarima_gap"] = (g[column] - g["sarima_pred"]) / g[column].replace(0, np.nan)
            forecast = float(fitted.forecast(steps=steps)[-1])
            logger.debug("SARIMA پیش‌بینی گام بعد: {:.2f}", forecast)
        except Exception as exc:  # noqa: BLE001
            logger.warning("برازش SARIMA ناموفق بود: {}", exc)
        return g

    if "symbol" in df.columns:
        parts = [_one(g) for _, g in df.groupby("symbol", sort=False)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _one(df).reset_index(drop=True)


def compute_all_seasonal(df: pd.DataFrame, with_sarima: bool = False) -> pd.DataFrame:
    """اجرای همه فیچرهای فصلی/تقویمی به‌صورت یکجا."""
    out = add_calendar_features(df)
    out = add_seasonal_decomposition(out)
    if with_sarima:
        out = add_sarima_forecast(out)
    return out
