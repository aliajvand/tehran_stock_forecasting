"""تست فیچرهای تکنیکال (فاز ۲)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.technical import atr, compute_all_technical, feature_columns, rsi, stochastic


def test_rsi_bounds() -> None:
    """RSI باید بین ۰ و ۱۰۰ بماند."""
    series = pd.Series(np.linspace(100, 200, 120) + np.random.default_rng(3).normal(0, 3, 120))
    values = rsi(series, 14)
    assert values.between(0, 100).all()


def test_rsi_all_up_is_high() -> None:
    """در روند صعودی پیوسته، RSI باید بالای ۷۰ باشد."""
    series = pd.Series(np.arange(1, 100, dtype=float))
    assert rsi(series, 14).iloc[-1] > 70


def test_compute_all_technical_adds_columns(sample_history: pd.DataFrame) -> None:
    """فیچرهای اصلی باید ساخته شوند."""
    out = compute_all_technical(sample_history)
    for col in ("rsi", "macd", "macd_signal", "bb_upper", "atr", "stoch_k", "return_1d"):
        assert col in out.columns
    assert len(out) == len(sample_history)


def test_features_computed_per_symbol(sample_history: pd.DataFrame) -> None:
    """فیچرها نباید بین نمادها قاطی شوند (اولین سطر هر نماد بازدهی NaN دارد)."""
    out = compute_all_technical(sample_history)
    firsts = out.groupby("symbol")["return_1d"].nth(0)
    assert firsts.isna().all()


def test_atr_and_stochastic_positive(sample_history: pd.DataFrame) -> None:
    """ATR نامنفی و %K بین ۰ و ۱۰۰ است."""
    df = sample_history[sample_history["symbol"] == "فولاد"]
    values = atr(df["high"], df["low"], df["close"], 14).dropna()
    assert (values >= 0).all()
    stoch = stochastic(df["high"], df["low"], df["close"]).dropna()
    assert stoch["stoch_k"].between(-1, 101).all()


def test_feature_columns_excludes_keys(sample_history: pd.DataFrame) -> None:
    """ستون‌های کلید/شناسه نباید فیچر شمرده شوند."""
    out = compute_all_technical(sample_history)
    cols = feature_columns(out)
    assert "ins_code" not in cols and "gregorian_date" not in cols
    assert "rsi" in cols
