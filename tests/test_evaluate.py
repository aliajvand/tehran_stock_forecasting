"""تست معیارهای ارزیابی (فاز ۳)."""

from __future__ import annotations

import numpy as np

from src.training.evaluate import (
    aggregate_folds,
    compute_metrics,
    confusion,
    directional_accuracy,
    format_report,
)


def test_perfect_prediction() -> None:
    """پیش‌بینی کامل باید دقت ۱۰۰٪ بدهد."""
    y = np.array([0, 1, 2, 0, 2])
    metrics = compute_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["directional_accuracy"] == 1.0


def test_directional_accuracy_ignores_neutral() -> None:
    """نمونه‌های neutral واقعی در دقت جهت‌دهی شمرده نمی‌شوند."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 2])
    assert directional_accuracy(y_true, y_pred) == 1.0


def test_confusion_matrix_shape() -> None:
    """ماتریس درهم‌ریختگی باید ۳×۳ باشد."""
    matrix = confusion(np.array([0, 1, 2]), np.array([0, 2, 2]))
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 3


def test_report_is_readable() -> None:
    """گزارش متنی باید معیارهای اصلی را داشته باشد."""
    text = format_report(compute_metrics(np.array([0, 2, 1]), np.array([0, 2, 1])))
    assert "Directional Accuracy" in text
    assert "ماتریس درهم‌ریختگی" in text


def test_aggregate_folds() -> None:
    """میانگین بازه‌ها باید محاسبه شود."""
    summary = aggregate_folds([
        {"accuracy": 0.6, "directional_accuracy": 0.7},
        {"accuracy": 0.8, "directional_accuracy": 0.75},
    ])
    assert summary["folds"] == 2
    assert abs(summary["accuracy_mean"] - 0.7) < 1e-9
