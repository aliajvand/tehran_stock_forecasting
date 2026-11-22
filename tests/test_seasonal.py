"""تست فیچرهای فصلی و تقویمی (فاز ۲)."""

from __future__ import annotations

import pandas as pd

from src.features.seasonal import add_calendar_features, decompose_seasonal


def test_calendar_features_added(sample_history: pd.DataFrame) -> None:
    """ستون‌های تقویمی باید ساخته شوند."""
    out = add_calendar_features(sample_history)
    for col in ("jalali_month", "jalali_season", "week_of_month", "day_of_week",
                "is_ramadan", "is_muharram", "month_sin"):
        assert col in out.columns
    assert out["jalali_season"].between(0, 4).all()
    assert out["week_of_month"].between(0, 5).all()


def test_hijri_month_detection() -> None:
    """ماه قمری باید در بازه ۱ تا ۱۲ باشد."""
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=400, freq="D"), "close": 1.0})
    out = add_calendar_features(frame)
    assert out["hijri_month"].between(1, 12).all()
    assert out["is_ramadan"].sum() > 0  # در بازه ۴۰۰ روزه حتماً رمضان هست


def test_decompose_seasonal_columns(sample_history: pd.DataFrame) -> None:
    """تجزیه فصلی باید سه ستون برگرداند."""
    series = sample_history[sample_history["symbol"] == "فولاد"]["close"].reset_index(drop=True)
    out = decompose_seasonal(series, period=22)
    assert list(out.columns) == ["trend", "seasonal", "resid"]
    assert len(out) == len(series)


def test_decompose_short_series_is_safe() -> None:
    """سری کوتاه نباید کرش کند."""
    out = decompose_seasonal(pd.Series([1.0, 2.0, 3.0]), period=22)
    assert out.isna().all().all()
