"""
انسمبل: ترکیب مدل هیبرید (LSTM+GNN) با XGBoost و Prophet.

روش‌ها:
    * ``soft_voting``: میانگین وزنی احتمال کلاس‌ها
    * ``stacking``: رگرسیون لجستیک روی احتمال‌های اعضا (متا-مدل)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger


class XGBMember:
    """عضو XGBoost که روی نمای تخت‌شده دنباله‌ها آموزش می‌بیند."""

    def __init__(self, num_classes: int = 3, **params: Any) -> None:
        """پارامترها به ``XGBClassifier`` پاس داده می‌شوند."""
        self.num_classes = num_classes
        self.params: Dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "objective": "multi:softprob",
            "num_class": num_classes,
            **params,
        }
        self.model: Optional[Any] = None

    @staticmethod
    def flatten(x: np.ndarray) -> np.ndarray:
        """
        تخت‌سازی دنباله برای مدل درختی: آماره‌های آخرین گام + میانگین/انحراف پنجره.
        """
        if x.ndim == 2:
            return x
        last = x[:, -1, :]
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        delta = x[:, -1, :] - x[:, 0, :]
        return np.concatenate([last, mean, std, delta], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "XGBMember":
        """آموزش مدل."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("برای عضو XGBoost باید xgboost نصب باشد") from exc
        self.model = XGBClassifier(**self.params)
        self.model.fit(self.flatten(x), y)
        logger.info("XGBoost آموزش دید ({} نمونه)", len(y))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """احتمال کلاس‌ها."""
        if self.model is None:
            raise RuntimeError("مدل XGBoost آموزش ندیده است")
        return np.asarray(self.model.predict_proba(self.flatten(x)), dtype=float)


class ProphetMember:
    """
    عضو Prophet: روند قیمت را پیش‌بینی و به احتمال کلاس تبدیل می‌کند.

    Prophet یک مدل تک‌سری است، پس برای هر نماد جدا برازش می‌شود.
    """

    def __init__(self, threshold_pct: float = 1.0) -> None:
        """پارامترها: آستانه درصدی تعیین کلاس."""
        self.threshold = float(threshold_pct) / 100.0
        self.models: Dict[str, Any] = {}

    def fit(self, history: pd.DataFrame) -> "ProphetMember":
        """
        برازش Prophet برای هر نماد.

        ورودی: DataFrame با ستون‌های ``symbol, date, close``
        """
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError("برای عضو Prophet باید prophet نصب باشد") from exc

        for symbol, group in history.groupby("symbol"):
            frame = pd.DataFrame(
                {"ds": pd.to_datetime(group["date"]), "y": pd.to_numeric(group["close"], errors="coerce")}
            ).dropna()
            if len(frame) < 60:
                continue
            try:
                model = Prophet(daily_seasonality=False, weekly_seasonality=True)
                model.fit(frame)
                self.models[str(symbol)] = model
            except Exception as exc:  # noqa: BLE001
                logger.warning("برازش Prophet برای «{}» ناموفق بود: {}", symbol, exc)
        logger.info("Prophet برای {} نماد آموزش دید", len(self.models))
        return self

    def predict_proba(self, symbols: Sequence[str], horizon: int = 1) -> np.ndarray:
        """تبدیل پیش‌بینی روند به احتمال سه‌کلاسه (down/neutral/up)."""
        out: List[List[float]] = []
        for symbol in symbols:
            model = self.models.get(str(symbol))
            if model is None:
                out.append([1 / 3, 1 / 3, 1 / 3])
                continue
            try:
                future = model.make_future_dataframe(periods=horizon)
                forecast = model.predict(future)
                change = float(forecast["yhat"].iloc[-1] / forecast["yhat"].iloc[-1 - horizon] - 1)
            except Exception:  # noqa: BLE001
                out.append([1 / 3, 1 / 3, 1 / 3])
                continue
            if change > self.threshold:
                out.append([0.15, 0.25, 0.60])
            elif change < -self.threshold:
                out.append([0.60, 0.25, 0.15])
            else:
                out.append([0.25, 0.50, 0.25])
        return np.asarray(out, dtype=float)


class EnsemblePredictor:
    """ترکیب چند مجموعه احتمال با رأی‌گیری نرم یا stacking."""

    def __init__(self, weights: Optional[Sequence[float]] = None, method: str = "soft_voting") -> None:
        """
        پارامترها:
            weights: وزن اعضا (نرمال‌سازی می‌شود).
            method: ``soft_voting`` یا ``stacking``.
        """
        self.weights = list(weights) if weights else None
        self.method = method
        self.meta: Optional[Any] = None

    def soft_voting(self, probas: Sequence[np.ndarray]) -> np.ndarray:
        """میانگین وزنی احتمال‌ها."""
        stack = np.stack([np.asarray(p, dtype=float) for p in probas])
        weights = np.asarray(self.weights or [1.0] * len(stack), dtype=float)[: len(stack)]
        weights = weights / weights.sum()
        return np.tensordot(weights, stack, axes=(0, 0))

    def fit_stacking(self, probas: Sequence[np.ndarray], y: np.ndarray) -> "EnsemblePredictor":
        """آموزش متا-مدل رگرسیون لجستیک روی احتمال‌های اعضا."""
        from sklearn.linear_model import LogisticRegression

        features = np.concatenate([np.asarray(p, dtype=float) for p in probas], axis=1)
        self.meta = LogisticRegression(max_iter=1000, multi_class="auto").fit(features, y)
        self.method = "stacking"
        logger.info("متا-مدل stacking آموزش دید ({} نمونه)", len(y))
        return self

    def predict_proba(self, probas: Sequence[np.ndarray]) -> np.ndarray:
        """احتمال نهایی انسمبل."""
        if self.method == "stacking" and self.meta is not None:
            features = np.concatenate([np.asarray(p, dtype=float) for p in probas], axis=1)
            return np.asarray(self.meta.predict_proba(features), dtype=float)
        return self.soft_voting(probas)

    def predict(self, probas: Sequence[np.ndarray]) -> np.ndarray:
        """کلاس نهایی (۰ down، ۱ neutral، ۲ up)."""
        return self.predict_proba(probas).argmax(axis=1)

    def save(self, path: Path) -> None:
        """ذخیره متا-مدل و وزن‌ها."""
        try:
            import joblib

            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"weights": self.weights, "method": self.method, "meta": self.meta}, path)
            logger.info("انسمبل ذخیره شد: {}", path)
        except Exception as exc:  # noqa: BLE001
            logger.error("ذخیره انسمبل ناموفق بود: {}", exc)
