"""
فیچرهای عرضه و تقاضا (order book) از جدول ``order_book``.

این فیچرها برای پیش‌بینی روزانه نقش کمکی دارند: میانگین روزانه فشار خرید/فروش
و شدت صف‌ها به‌عنوان ستون‌های اضافی به دیتاست روزانه اضافه می‌شوند.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src import db as dbm


def compute_order_book_features(book_df: pd.DataFrame) -> pd.DataFrame:
    """
    فیچرهای هر snapshot: فشار خرید، عدم توازن سفارش‌ها و شیب دفتر سفارش.

    ورودی: DataFrame با ستون‌های
    ``ins_code, gregorian_date, level, buy_price, buy_volume, sell_price, sell_volume``
    """
    if book_df is None or book_df.empty:
        logger.warning("compute_order_book_features: ورودی خالی است")
        return pd.DataFrame()

    df = book_df.copy()
    for col in ("buy_price", "buy_volume", "sell_price", "sell_volume"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    keys = [c for c in ("ins_code", "gregorian_date") if c in df.columns]
    if not keys:
        raise ValueError("ستون‌های ins_code و gregorian_date لازم است")

    grouped = df.groupby(keys, sort=False)
    out = grouped.agg(
        buy_volume_total=("buy_volume", "sum"),
        sell_volume_total=("sell_volume", "sum"),
        best_buy_price=("buy_price", "max"),
        best_sell_price=("sell_price", "min"),
        levels=("level", "count") if "level" in df.columns else ("buy_volume", "count"),
    ).reset_index()

    total = out["buy_volume_total"] + out["sell_volume_total"]
    out["ob_imbalance"] = ((out["buy_volume_total"] - out["sell_volume_total"]) /
                           total.replace(0, np.nan)).fillna(0.0)
    out["ob_pressure"] = (out["buy_volume_total"] /
                          out["sell_volume_total"].replace(0, np.nan)).replace(
                              [np.inf, -np.inf], np.nan).fillna(1.0)
    mid = (out["best_buy_price"] + out["best_sell_price"]) / 2
    out["ob_spread_pct"] = ((out["best_sell_price"] - out["best_buy_price"]) /
                            mid.replace(0, np.nan)).fillna(0.0)
    logger.info("فیچرهای دفتر سفارش ساخته شد ({} سطر)", len(out))
    return out


def load_daily_order_book_features(db_path: Optional[str] = None) -> pd.DataFrame:
    """خواندن دفتر سفارش از دیتابیس و تبدیل به فیچرهای روزانه (میانگین روز)."""
    raw = dbm.read_sql("SELECT * FROM order_book", db_path=db_path)
    if raw.empty:
        logger.warning("جدول order_book خالی است")
        return pd.DataFrame()

    # 🔧 هم‌راست‌سازی ستون‌های جدول order_book با ورودی تابع compute_order_book_features
    # (bid/ask در DB ↔ buy/sell در فیچرساز) + استخراج gregorian_date از snapshot_time.
    df = raw.rename(
        columns={
            "bid_price": "buy_price",
            "bid_volume": "buy_volume",
            "ask_price": "sell_price",
            "ask_volume": "sell_volume",
        }
    )
    if "snapshot_time" in df.columns and "gregorian_date" not in df.columns:
        df["gregorian_date"] = (
            pd.to_datetime(df["snapshot_time"], unit="s", errors="coerce")
            .dt.strftime("%Y%m%d")
            .astype("Int64")
        )
    df = df.dropna(subset=["gregorian_date"])

    feats = compute_order_book_features(df)
    if feats.empty:
        return feats
    numeric = [c for c in feats.columns if c.startswith("ob_")]
    return feats.groupby(["ins_code", "gregorian_date"], as_index=False)[numeric].mean()
