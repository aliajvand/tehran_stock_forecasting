"""تست ساخت دیتاست، برچسب و بازه‌های Walk-Forward (فاز ۳)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.technical import compute_all_technical
from src.training.dataset import (
    add_labels,
    build_sequences,
    standardize,
    walk_forward_splits,
)


def test_labels_three_classes(sample_history: pd.DataFrame) -> None:
    """برچسب‌ها باید در {۰,۱,۲} باشند و آینده‌نگر حذف شود."""
    out = add_labels(sample_history, horizon_days=1, threshold_pct=1.0)
    assert set(out["target"].unique()) <= {0, 1, 2}
    assert not out["future_return"].isna().any()
    assert len(out) < len(sample_history)


def test_label_threshold_logic() -> None:
    """قیمت صعودی قوی باید کلاس up (۲) بگیرد."""
    frame = pd.DataFrame(
        {
            "symbol": ["فولاد"] * 5,
            "gregorian_date": [20250101, 20250102, 20250103, 20250104, 20250105],
            "close": [100.0, 105.0, 110.0, 115.0, 120.0],
        }
    )
    out = add_labels(frame, horizon_days=1, threshold_pct=1.0)
    assert (out["target"] == 2).all()


def test_build_sequences_shapes(sample_history: pd.DataFrame) -> None:
    """شکل X باید (نمونه, seq_len, فیچر) باشد."""
    table = add_labels(compute_all_technical(sample_history))
    data = build_sequences(table, sequence_length=30)
    assert data["X"].ndim == 3
    assert data["X"].shape[1] == 30
    assert len(data["y"]) == data["X"].shape[0]
    assert not np.isnan(data["X"]).any()


def test_walk_forward_is_time_ordered() -> None:
    """آموزش هر بازه باید قبل از اعتبارسنجی آن باشد."""
    dates = np.arange(20250101, 20250101 + 300)
    splits = walk_forward_splits(dates, n_splits=4)
    assert len(splits) == 4
    for train_idx, val_idx in splits:
        assert dates[train_idx].max() <= dates[val_idx].min()


def test_standardize_uses_train_stats() -> None:
    """میانگین مجموعه آموزش بعد از استانداردسازی نزدیک صفر است."""
    train = np.random.default_rng(0).normal(5, 2, (50, 10, 4)).astype(np.float32)
    val = np.random.default_rng(1).normal(5, 2, (10, 10, 4)).astype(np.float32)
    train_s, val_s = standardize(train, val)
    assert abs(float(train_s.mean())) < 1e-5
    assert val_s.shape == val.shape
