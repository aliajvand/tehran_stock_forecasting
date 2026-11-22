"""
ساخت دیتاست آموزش از دیتابیس: فیچر + برچسب + دنباله‌های زمانی.

جریان: ``load_feature_table`` → ``add_labels`` → ``build_sequences``

برچسب‌گذاری: بازدهی آینده در افق ``prediction.horizon_days`` محاسبه و با آستانه
``prediction.threshold_pct`` به سه کلاس تبدیل می‌شود:
    0 = down | 1 = neutral | 2 = up
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src import db as dbm
from src.config import get_prediction_config, get_section
from src.features.money_flow import compute_all_money_flow
from src.features.seasonal import compute_all_seasonal
from src.features.technical import compute_all_technical, feature_columns
from src.preprocessing.cleaner import clean_client_type, clean_history

CLASS_NAMES: Tuple[str, str, str] = ("down", "neutral", "up")
LABEL_MAP: Dict[str, int] = {"down": 0, "neutral": 1, "up": 2}


def load_feature_table(
    symbols: Optional[Sequence[str]] = None,
    with_sentiment: Optional[bool] = None,
    with_order_book: Optional[bool] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    ساخت جدول فیچر روزانه برای همه نمادها.

    ترکیب: تاریخچه پاک‌شده + فیچر تکنیکال + فصلی/تقویمی + جریان پول
    (+ احساسات اخبار و دفتر سفارش در صورت فعال بودن در config).
    """
    feats_cfg = get_section("features", {})
    history = dbm.load_daily_history(symbols, db_path=db_path)
    if history.empty:
        logger.error("جدول daily_history خالی است؛ اول backfill را اجرا کنید")
        return pd.DataFrame()

    table = clean_history(history)
    if feats_cfg.get("technical", True):
        table = compute_all_technical(table)
    if feats_cfg.get("seasonal", True):
        table = compute_all_seasonal(table)

    if feats_cfg.get("money_flow", True):
        client = dbm.load_client_type(symbols, db_path=db_path)
        if not client.empty:
            flow = compute_all_money_flow(clean_client_type(client))
            keep = ["symbol", "gregorian_date"] + [
                c for c in flow.columns
                if c.startswith(("net_", "power_", "retail_", "per_capita_buy_ratio"))
                or c.endswith("_imbalance")
            ]
            table = table.merge(
                flow[list(dict.fromkeys(keep))], on=["symbol", "gregorian_date"], how="left"
            )
        else:
            logger.warning("داده حقیقی/حقوقی موجود نیست؛ فیچر جریان پول رد شد")

    if with_sentiment if with_sentiment is not None else feats_cfg.get("news_sentiment", False):
        try:
            from src.features.news_sentiment import build_daily_sentiment_features

            sentiment = build_daily_sentiment_features()
            if not sentiment.empty:
                table = table.merge(sentiment, on=["symbol", "gregorian_date"], how="left")
                for col in ("news_count", "sentiment_mean", "sentiment_sum", "sentiment_decayed"):
                    if col in table.columns:
                        table[col] = table[col].fillna(0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("افزودن فیچر احساسات ناموفق بود: {}", exc)

    if with_order_book if with_order_book is not None else feats_cfg.get("order_book", False):
        try:
            from src.features.order_book import load_daily_order_book_features

            book = load_daily_order_book_features(db_path=db_path)
            if not book.empty:
                table = table.merge(book, on=["ins_code", "gregorian_date"], how="left")
        except Exception as exc:  # noqa: BLE001
            logger.warning("افزودن فیچر دفتر سفارش ناموفق بود: {}", exc)

    table = table.sort_values(["symbol", "gregorian_date"]).reset_index(drop=True)
    logger.info("جدول فیچر: {} سطر × {} ستون", len(table), len(table.columns))
    return table


def add_labels(
    df: pd.DataFrame, horizon_days: Optional[int] = None, threshold_pct: Optional[float] = None
) -> pd.DataFrame:
    """
    افزودن ستون‌های ``future_return`` و ``target`` (۰ down، ۱ neutral، ۲ up).

    سطرهای پایانی هر نماد که آینده‌شان معلوم نیست حذف می‌شوند.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    pred_cfg = get_prediction_config()
    horizon = int(horizon_days or pred_cfg.get("horizon_days", 1))
    threshold = float(threshold_pct if threshold_pct is not None else pred_cfg.get("threshold_pct", 1.0)) / 100.0

    out = df.copy()
    price = pd.to_numeric(out["close"], errors="coerce")
    if "symbol" in out.columns:
        out["future_return"] = (
            price.groupby(out["symbol"]).shift(-horizon) / price - 1
        )
    else:
        out["future_return"] = price.shift(-horizon) / price - 1
    # قیمت صفر → انفینیتی؛ باید قبل از برچسب‌گذاری ناپاک شود
    out["future_return"] = out["future_return"].replace([np.inf, -np.inf], np.nan)

    out["target"] = np.select(
        [out["future_return"] > threshold, out["future_return"] < -threshold],
        [LABEL_MAP["up"], LABEL_MAP["down"]],
        default=LABEL_MAP["neutral"],
    )
    out.loc[out["future_return"].isna(), "target"] = np.nan
    out = out.dropna(subset=["future_return", "target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)

    dist = out["target"].value_counts(normalize=True).sort_index()
    logger.info(
        "برچسب‌ها (افق {} روز، آستانه {:.1%}): down={:.1%} neutral={:.1%} up={:.1%}",
        horizon, threshold,
        float(dist.get(0, 0.0)), float(dist.get(1, 0.0)), float(dist.get(2, 0.0)),
    )
    return out


def build_sequences(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    sequence_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    تبدیل جدول فیچر به دنباله‌های ``(نمونه, seq_len, n_features)``.

    خروجی: dict با کلیدهای ``X, y, symbols, dates, feature_cols``
    """
    if df is None or df.empty:
        return {"X": np.empty((0, 0, 0)), "y": np.empty(0), "symbols": np.empty(0), "dates": np.empty(0)}

    seq_len = int(sequence_length or get_section("features", {}).get("sequence_length", 60))
    cols: List[str] = list(feature_cols or feature_columns(df))
    if not cols:
        raise ValueError("هیچ ستون فیچری برای ساخت دنباله پیدا نشد")

    frames: List[np.ndarray] = []
    labels: List[int] = []
    syms: List[str] = []
    dates: List[int] = []

    grouped = df.groupby("symbol", sort=False) if "symbol" in df.columns else [("all", df)]
    for symbol, group in grouped:
        g = group.sort_values("gregorian_date")
        matrix = (
            g[cols].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        targets = g["target"].to_numpy() if "target" in g.columns else np.zeros(len(g))
        day = g["gregorian_date"].to_numpy()
        if len(g) <= seq_len:
            logger.debug("نماد «{}» داده کافی برای دنباله ندارد ({} سطر)", symbol, len(g))
            continue
        for end in range(seq_len, len(g)):
            frames.append(matrix[end - seq_len:end])
            labels.append(int(targets[end]))
            syms.append(str(symbol))
            dates.append(int(day[end]))

    if not frames:
        logger.error("هیچ دنباله‌ای ساخته نشد (داده کم است)")
        return {"X": np.empty((0, seq_len, len(cols))), "y": np.empty(0),
                "symbols": np.empty(0), "dates": np.empty(0), "feature_cols": np.array(cols)}

    out = {
        "X": np.stack(frames).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "symbols": np.asarray(syms),
        "dates": np.asarray(dates, dtype=np.int64),
        "feature_cols": np.asarray(cols),
    }
    logger.info("دنباله‌ها: X={} y={}", out["X"].shape, out["y"].shape)
    return out


def walk_forward_splits(
    dates: np.ndarray, n_splits: int = 5, val_ratio: float = 0.2
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    تولید بازه‌های Walk-Forward بر اساس ترتیب زمانی (بدون نشت اطلاعات).

    خروجی: لیستی از (ایندکس آموزش، ایندکس اعتبارسنجی)
    """
    order = np.argsort(dates, kind="stable")
    total = len(order)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if total < 10 or n_splits < 1:
        cut = int(total * (1 - val_ratio))
        return [(order[:cut], order[cut:])] if cut > 0 else []

    fold = total // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = fold * i
        val_end = min(total, train_end + fold)
        if val_end <= train_end:
            break
        splits.append((order[:train_end], order[train_end:val_end]))
    logger.info("{} بازه Walk-Forward ساخته شد", len(splits))
    return splits


def standardize(train_x: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    استانداردسازی z-score با آماره‌های مجموعه آموزش (جلوگیری از نشت اطلاعات).
    """
    if train_x.size == 0:
        return (train_x, *others)
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    scale = lambda arr: ((arr - mean) / std).astype(np.float32) if arr.size else arr  # noqa: E731
    return (scale(train_x), *(scale(o) for o in others))
