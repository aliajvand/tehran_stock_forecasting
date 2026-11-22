"""
ارزیابی مدل: دقت جهت‌دهی، F1، AUC-ROC و ماتریس درهم‌ریختگی.

گزارش‌ها در ``logs/evaluation/`` به‌صورت JSON و متن ذخیره می‌شوند.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from loguru import logger

from src.config import get_section, resolve_path

CLASS_NAMES: tuple[str, str, str] = ("down", "neutral", "up")


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    دقت جهت‌دهی: نسبت پیش‌بینی درست جهت، فقط روی نمونه‌های غیر-neutral واقعی.

    این معیار اصلی پروژه است (هدف: بالای ۷۰٪).
    """
    mask = np.asarray(y_true) != 1
    if not mask.any():
        return 0.0
    return float((np.asarray(y_true)[mask] == np.asarray(y_pred)[mask]).mean())


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """ماتریس درهم‌ریختگی (بدون وابستگی اجباری به sklearn)."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            matrix[true, pred] += 1
    return matrix


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    محاسبه همه معیارها.

    خروجی: dict شامل ``accuracy, directional_accuracy, f1_macro, f1_per_class,
    auc_roc, confusion_matrix, support``
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.size == 0:
        return {"accuracy": 0.0, "directional_accuracy": 0.0, "f1_macro": 0.0, "support": 0}

    metrics: Dict[str, Any] = {
        "support": int(y_true.size),
        "accuracy": float((y_true == y_pred).mean()),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "confusion_matrix": confusion(y_true, y_pred).tolist(),
        "class_names": list(CLASS_NAMES),
    }

    try:
        from sklearn.metrics import f1_score, roc_auc_score

        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_per_class"] = [
            float(v) for v in f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        ]
        if y_proba is not None and len(np.unique(y_true)) > 1:
            proba = np.asarray(y_proba, dtype=float)
            metrics["auc_roc"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=[0, 1, 2])
            )
    except ImportError:
        logger.warning("scikit-learn نصب نیست؛ F1 و AUC محاسبه نشد")
    except Exception as exc:  # noqa: BLE001 - مثلاً کلاس غایب در y_true
        logger.warning("محاسبه F1/AUC ناموفق بود: {}", exc)

    return metrics


def format_report(metrics: Dict[str, Any], title: str = "ارزیابی مدل") -> str:
    """ساخت گزارش متنی خوانا از معیارها."""
    lines = [f"── {title} ──"]
    lines.append(f"تعداد نمونه         : {metrics.get('support', 0)}")
    lines.append(f"Accuracy            : {metrics.get('accuracy', 0):.2%}")
    lines.append(f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}")
    if "f1_macro" in metrics:
        lines.append(f"F1 (macro)          : {metrics['f1_macro']:.4f}")
    if "auc_roc" in metrics:
        lines.append(f"AUC-ROC (ovr)       : {metrics['auc_roc']:.4f}")
    if "confusion_matrix" in metrics:
        lines.append("ماتریس درهم‌ریختگی (سطر=واقعی، ستون=پیش‌بینی):")
        lines.append("            " + "  ".join(f"{c:>9}" for c in CLASS_NAMES))
        for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
            lines.append(f"{name:>11} " + "  ".join(f"{v:>9}" for v in row))
    return "\n".join(lines)


def save_report(
    metrics: Dict[str, Any], name: str = "evaluation", extra: Optional[Dict[str, Any]] = None
) -> Path:
    """ذخیره گزارش JSON + متن در ``logs/evaluation/``."""
    reports_dir = resolve_path(str(get_section("paths").get("reports", "logs/evaluation")))
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"generated_at": stamp, "metrics": metrics, **(extra or {})}

    json_path = reports_dir / f"{name}_{stamp}.json"
    text_path = reports_dir / f"{name}_{stamp}.txt"
    try:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        text_path.write_text(format_report(metrics), encoding="utf-8")
        logger.success("گزارش ارزیابی ذخیره شد: {}", json_path)
    except OSError as exc:
        logger.error("ذخیره گزارش ناموفق بود: {}", exc)
    return json_path


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    name: str = "evaluation",
    save: bool = True,
) -> Dict[str, Any]:
    """محاسبه معیارها + لاگ گزارش + ذخیره اختیاری."""
    metrics = compute_metrics(y_true, y_pred, y_proba)
    logger.info("\n{}", format_report(metrics))
    if save:
        save_report(metrics, name=name)
    return metrics


def aggregate_folds(fold_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """میانگین و انحراف معیار نتایج بازه‌های Walk-Forward."""
    if not fold_metrics:
        return {}
    keys = ("accuracy", "directional_accuracy", "f1_macro", "auc_roc")
    summary: Dict[str, Any] = {"folds": len(fold_metrics)}
    for key in keys:
        values = [float(m[key]) for m in fold_metrics if key in m]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary
