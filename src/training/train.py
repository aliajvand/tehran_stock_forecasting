"""
آموزش مدل با Walk-Forward Validation و Early Stopping.

اجرا:
    python -m src.training.train
    python -m src.training.train --model lstm --epochs 20
    python -m src.training.train --model hybrid --with-graph

خروجی‌ها:
    * وزن بهترین مدل هر بازه در ``models/saved/``
    * لاگ متریک‌ها به سبک MLflow (JSON در ``models/saved/runs/``)
    * گزارش ارزیابی در ``logs/evaluation/``
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

from src.config import get_section, resolve_path, setup_logging
from src.training.dataset import (
    add_labels,
    build_sequences,
    load_feature_table,
    standardize,
    walk_forward_splits,
)
from src.training.evaluate import aggregate_folds, evaluate_predictions


def set_seed(seed: int = 42) -> None:
    """تثبیت seed برای تکرارپذیری."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(preference: str = "auto") -> Any:
    """انتخاب دستگاه اجرا (cuda در صورت وجود)."""
    import torch

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunLogger:
    """لاگ متریک‌ها به سبک MLflow (فایل JSON ساده، بدون وابستگی اضافه)."""

    def __init__(self, name: str = "run") -> None:
        """ساخت پوشه اجرا با مهر زمانی."""
        base = resolve_path(str(get_section("paths").get("models", "models/saved"))) / "runs"
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = base / f"{name}_{stamp}.json"
        self.payload: Dict[str, Any] = {"name": name, "started_at": stamp, "params": {}, "folds": []}

    def log_params(self, **params: Any) -> None:
        """ثبت هایپرپارامترها."""
        self.payload["params"].update({k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()})

    def log_fold(self, fold: int, metrics: Dict[str, Any]) -> None:
        """ثبت متریک یک بازه."""
        self.payload["folds"].append({"fold": fold, **metrics})

    def finish(self, summary: Dict[str, Any]) -> Path:
        """نوشتن فایل نهایی اجرا."""
        self.payload["summary"] = summary
        self.payload["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            self.path.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.success("لاگ اجرا ذخیره شد: {}", self.path)
        except OSError as exc:
            logger.error("ذخیره لاگ اجرا ناموفق بود: {}", exc)
        return self.path


def _train_torch_fold(
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    train_cfg: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    آموزش یک بازه با Early Stopping.

    خروجی: dict شامل پیش‌بینی‌های اعتبارسنجی و بهترین loss.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = resolve_device(str(train_cfg.get("device", "auto")))
    model = model.to(device)

    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=True)
    val_tensor = torch.from_numpy(val_x).to(device)
    val_target = torch.from_numpy(val_y).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, int(train_cfg.get("epochs", 100)) + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
        train_loss = total / max(1, len(dataset))

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_tensor)
            val_loss = float(criterion(val_logits, val_target).item())

        if epoch % 5 == 0 or epoch == 1:
            logger.info("epoch {:>3} | train_loss={:.4f} | val_loss={:.4f}", epoch, train_loss, val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping در epoch {}", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_path)
                logger.info("وزن مدل ذخیره شد: {}", save_path)
            except OSError as exc:
                logger.error("ذخیره مدل ناموفق بود: {}", exc)

    model.eval()
    with torch.no_grad():
        logits, expected = model(val_tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return {
        "y_pred": proba.argmax(axis=1),
        "y_proba": proba,
        "expected_return": expected.cpu().numpy(),
        "best_val_loss": best_loss,
    }


def train(
    model_type: str = "hybrid",
    symbols: Optional[Sequence[str]] = None,
    epochs: Optional[int] = None,
    splits: Optional[int] = None,
    with_graph: bool = False,
) -> Dict[str, Any]:
    """
    آموزش کامل با Walk-Forward.

    پارامترها:
        model_type: ``lstm`` | ``hybrid`` | ``xgboost``
        with_graph: فعال کردن شاخه GNN مدل هیبرید.

    خروجی: خلاصه متریک‌های همه بازه‌ها.
    """
    model_cfg = get_section("model", {})
    train_cfg = dict(model_cfg.get("training", {}) or {})
    if epochs:
        train_cfg["epochs"] = int(epochs)
    set_seed(int(train_cfg.get("seed", 42)))

    table = load_feature_table(symbols)
    if table.empty:
        logger.error("جدول فیچر خالی است؛ آموزش ممکن نیست")
        return {}
    labeled = add_labels(table)
    data = build_sequences(labeled)
    if data["X"].size == 0:
        logger.error("دنباله‌ای برای آموزش وجود ندارد")
        return {}

    run = RunLogger(name=f"train_{model_type}")
    run.log_params(model=model_type, samples=int(data["X"].shape[0]),
                   features=int(data["X"].shape[-1]), **train_cfg)

    folds = walk_forward_splits(data["dates"], n_splits=int(splits or train_cfg.get("walk_forward_splits", 5)))
    if not folds:
        logger.error("بازه Walk-Forward ساخته نشد (داده کم است)")
        return {}

    models_dir = resolve_path(str(get_section("paths").get("models", "models/saved")))
    fold_metrics: List[Dict[str, Any]] = []

    for index, (train_idx, val_idx) in enumerate(folds, start=1):
        logger.info("──── بازه {}/{} | آموزش={} اعتبارسنجی={} ────",
                    index, len(folds), len(train_idx), len(val_idx))
        train_x, val_x = standardize(data["X"][train_idx], data["X"][val_idx])
        train_y, val_y = data["y"][train_idx], data["y"][val_idx]

        if model_type == "xgboost":
            from src.models.ensemble import XGBMember

            member = XGBMember().fit(train_x, train_y)
            proba = member.predict_proba(val_x)
            result = {"y_pred": proba.argmax(axis=1), "y_proba": proba}
        else:
            if model_type == "lstm":
                from src.models.lstm_model import build_lstm

                model = build_lstm(train_x.shape[-1], model_cfg.get("lstm", {}))
            else:
                from src.models.hybrid_model import build_hybrid

                graph_size = train_x.shape[-1] if with_graph else 0
                model = build_hybrid(train_x.shape[-1], graph_size, model_cfg)
            result = _train_torch_fold(
                model, train_x, train_y, val_x, val_y, train_cfg,
                save_path=models_dir / f"{model_type}_fold{index}.pt",
            )

        metrics = evaluate_predictions(
            val_y, result["y_pred"], result.get("y_proba"),
            name=f"{model_type}_fold{index}", save=True,
        )
        run.log_fold(index, metrics)
        fold_metrics.append(metrics)

    summary = aggregate_folds(fold_metrics)
    run.finish(summary)
    logger.success(
        "میانگین دقت جهت‌دهی روی {} بازه: {:.2%}",
        summary.get("folds", 0), summary.get("directional_accuracy_mean", 0.0),
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """نقطه ورود آموزش."""
    parser = argparse.ArgumentParser(description="آموزش مدل پیش‌بینی بورس")
    parser.add_argument("--model", choices=["lstm", "hybrid", "xgboost"], default="hybrid")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--with-graph", action="store_true", help="فعال کردن شاخه GNN")
    args = parser.parse_args(argv)

    setup_logging()
    summary = train(
        model_type=args.model,
        symbols=args.symbols,
        epochs=args.epochs,
        splits=args.splits,
        with_graph=args.with_graph,
    )
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
