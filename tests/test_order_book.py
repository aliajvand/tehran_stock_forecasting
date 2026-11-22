"""تست فیچرهای دفتر سفارش (فاز ۲)."""

from __future__ import annotations

import pandas as pd

from src.features.order_book import compute_order_book_features


def _book() -> pd.DataFrame:
    """دفتر سفارش نمونه با ۵ سطح."""
    return pd.DataFrame(
        {
            "ins_code": [1000] * 5,
            "gregorian_date": [20250101] * 5,
            "level": [1, 2, 3, 4, 5],
            "buy_price": [1000, 999, 998, 997, 996],
            "buy_volume": [500, 400, 300, 200, 100],
            "sell_price": [1001, 1002, 1003, 1004, 1005],
            "sell_volume": [450, 350, 250, 150, 50],
        }
    )


def test_order_book_features() -> None:
    """فیچرهای فشار و عدم توازن باید محاسبه شوند."""
    out = compute_order_book_features(_book())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["buy_volume_total"] == 1500
    assert -1 <= row["ob_imbalance"] <= 1
    assert row["ob_pressure"] > 1  # تقاضا بیشتر از عرضه
    assert row["ob_spread_pct"] > 0


def test_empty_book_is_safe() -> None:
    """ورودی خالی نباید کرش کند."""
    assert compute_order_book_features(pd.DataFrame()).empty
