"""تست پاک‌سازی داده (فاز ۲)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.cleaner import clean_client_type, clean_history, detect_outliers


def test_clean_history_removes_zero_volume(sample_history: pd.DataFrame) -> None:
    """روزهای با حجم صفر باید حذف شوند."""
    df = sample_history.copy()
    df.loc[df.index[:3], "volume"] = 0
    out = clean_history(df, adjust=False, drop_outliers=False)
    assert (out["volume"] > 0).all()
    assert len(out) < len(df)


def test_clean_history_fills_nan(sample_history: pd.DataFrame) -> None:
    """NaN موجود در ستون open باید پر شود."""
    assert sample_history["open"].isna().any()
    out = clean_history(sample_history, adjust=False, drop_outliers=False)
    assert not out["open"].isna().any()


def test_detect_outliers_flags_spike(sample_history: pd.DataFrame) -> None:
    """جهش شدید قیمت باید به‌عنوان پرت علامت بخورد."""
    df = sample_history[sample_history["symbol"] == "فولاد"].reset_index(drop=True).copy()
    df.loc[10, "close"] = float(df["close"].max() * 50)
    mask = detect_outliers(df, method="iqr")
    assert bool(mask.iloc[10])


def test_clean_client_type_adds_shares(sample_client_type: pd.DataFrame) -> None:
    """ستون‌های نسبتی باید ساخته شوند و NaN نداشته باشند."""
    out = clean_client_type(sample_client_type)
    assert {"retail_buy_share", "retail_sell_share"} <= set(out.columns)
    assert out["retail_buy_share"].between(0, 1).all()
    assert not out.filter(like="vol_").isna().any().any()


def test_clean_history_handles_empty_input() -> None:
    """ورودی خالی نباید استثنا بدهد."""
    assert clean_history(pd.DataFrame()).empty
    assert clean_client_type(pd.DataFrame()).empty
