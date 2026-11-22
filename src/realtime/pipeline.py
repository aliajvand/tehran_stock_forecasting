"""
خط لوله لحظه‌ای: دریافت → فیچرسازی → پیش‌بینی → ذخیره.

از ``asyncio`` استفاده می‌شود و با SIGINT/SIGTERM به‌صورت تمیز خاتمه می‌یابد.
کلاس ``Predictor`` جدا است تا API و داشبورد هم از همان مسیر استنتاج کنند.

اجرا:
    python -m src.realtime.pipeline
    python -m src.realtime.pipeline --once
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_prediction_config, get_section, resolve_path, setup_logging
from src.training.dataset import CLASS_NAMES, add_labels, build_sequences, load_feature_table


class Predictor:
    """بارگذاری مدل ذخیره‌شده و تولید سیگنال روز جاری."""

    def __init__(self, model_path: Optional[Path] = None, model_type: str = "hybrid") -> None:
        """
        پارامترها:
            model_path: مسیر فایل وزن (پیش‌فرض: آخرین فایل در ``models/saved``).
            model_type: ``hybrid`` یا ``lstm``.
        """
        self.model_type = model_type
        self.models_dir = resolve_path(str(get_section("paths").get("models", "models/saved")))
        self.model_path = model_path or self._latest_checkpoint()
        self.model: Optional[Any] = None
        self.feature_cols: List[str] = []

    def _latest_checkpoint(self) -> Optional[Path]:
        """یافتن آخرین وزن ذخیره‌شده."""
        files = sorted(self.models_dir.glob(f"{self.model_type}_fold*.pt"))
        return files[-1] if files else None

    def _load(self, input_size: int) -> Optional[Any]:
        """ساخت مدل و بارگذاری وزن‌ها (در صورت وجود)."""
        try:
            import torch

            model_cfg = get_section("model", {})
            if self.model_type == "lstm":
                from src.models.lstm_model import build_lstm

                model = build_lstm(input_size, model_cfg.get("lstm", {}))
            else:
                from src.models.hybrid_model import build_hybrid

                model = build_hybrid(input_size, 0, model_cfg)
            if self.model_path and Path(self.model_path).exists():
                state = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(state)
                logger.info("مدل بارگذاری شد: {}", self.model_path)
            else:
                logger.warning("وزن مدل پیدا نشد؛ پیش‌بینی با مدل آموزش‌ندیده انجام می‌شود")
            model.eval()
            return model
        except ImportError as exc:
            logger.error("torch در دسترس نیست؛ پیش‌بینی ممکن نیست: {}", exc)
            return None

    def predict_latest(self, symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        پیش‌بینی برای آخرین روز موجود هر نماد.

        خروجی: ``symbol, gregorian_date, close, signal, probability, expected_return``
        """
        table = load_feature_table(symbols)
        if table.empty:
            return pd.DataFrame()
        labeled = add_labels(table)
        data = build_sequences(labeled)
        if data["X"].size == 0:
            logger.error("داده کافی برای ساخت دنباله وجود ندارد")
            return pd.DataFrame()

        # آخرین دنباله هر نماد
        frame = pd.DataFrame({"symbol": data["symbols"], "date": data["dates"]})
        last_idx = frame.groupby("symbol")["date"].idxmax().to_numpy()
        x = data["X"][last_idx]

        if self.model is None:
            self.model = self._load(x.shape[-1])
        if self.model is None:
            return pd.DataFrame()

        import torch

        flat = x.reshape(-1, x.shape[-1])
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std[std == 0] = 1.0
        scaled = ((x - mean) / std).astype(np.float32)

        with torch.no_grad():
            logits, expected = self.model(torch.from_numpy(scaled))
            proba = torch.softmax(logits, dim=1).numpy()

        closes = labeled.groupby("symbol")["close"].last()
        out = pd.DataFrame(
            {
                "symbol": frame.loc[last_idx, "symbol"].to_numpy(),
                "gregorian_date": frame.loc[last_idx, "date"].to_numpy(),
                "signal": [CLASS_NAMES[i] for i in proba.argmax(axis=1)],
                "probability": proba.max(axis=1).round(4),
                "expected_return": np.asarray(expected).round(4),
            }
        )
        out["close"] = out["symbol"].map(closes)
        out["horizon_days"] = int(get_prediction_config().get("horizon_days", 1))
        logger.info("سیگنال برای {} نماد تولید شد", len(out))
        return out


class RealtimePipeline:
    """حلقه غیرهمزمان دریافت داده و تولید سیگنال."""

    def __init__(self, interval_sec: Optional[float] = None, symbols: Optional[Sequence[str]] = None) -> None:
        """پارامترها از بخش ``data`` فایل config خوانده می‌شود."""
        data_cfg = get_section("data", {})
        self.interval_sec = float(interval_sec or data_cfg.get("ingestion_interval_sec", 10))
        self.symbols = list(symbols) if symbols else None
        self.predictor = Predictor()
        self._stop_event: Optional[asyncio.Event] = None
        self.latest: pd.DataFrame = pd.DataFrame()

    async def _fetch_snapshot(self) -> int:
        """دریافت یک دور snapshot در thread جدا (تابع شبکه blocking است)."""
        def _job() -> int:
            from src import db as dbm
            from src.config import get_db_path, get_target_symbols
            from src.data_ingestion.realtime_fetcher import RealtimeFetcher
            from src.data_ingestion.tsetmc_client import TsetmcClient

            conn = dbm.connect(get_db_path())
            try:
                fetcher = RealtimeFetcher(
                    TsetmcClient(), conn, self.symbols or get_target_symbols(), self.interval_sec
                )
                written = fetcher.fetch_once()
                return int(sum(written.values()))
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_job)
        except Exception as exc:  # noqa: BLE001 - حلقه نباید بمیرد
            logger.error("دریافت snapshot ناموفق بود: {}", exc)
            return 0

    async def _predict(self) -> pd.DataFrame:
        """اجرای پیش‌بینی در thread جدا (محاسبات سنگین)."""
        try:
            return await asyncio.to_thread(self.predictor.predict_latest, self.symbols)
        except Exception as exc:  # noqa: BLE001
            logger.error("پیش‌بینی ناموفق بود: {}", exc)
            return pd.DataFrame()

    async def run_cycle(self) -> Dict[str, Any]:
        """یک دور کامل خط لوله."""
        rows = await self._fetch_snapshot()
        signals = await self._predict()
        if not signals.empty:
            self.latest = signals
            self._persist(signals)
        return {"snapshot_rows": rows, "signals": len(signals)}

    def _persist(self, signals: pd.DataFrame) -> None:
        """ذخیره سیگنال‌های روز در فایل CSV پوشه processed."""
        try:
            out_dir = resolve_path(str(get_section("paths").get("processed", "data/processed")))
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "latest_signals.csv"
            signals.to_csv(path, index=False, encoding="utf-8-sig")
            logger.debug("سیگنال‌ها ذخیره شد: {}", path)
        except OSError as exc:
            logger.error("ذخیره سیگنال‌ها ناموفق بود: {}", exc)

    async def run_forever(self, max_cycles: Optional[int] = None) -> int:
        """حلقه اصلی با خاتمه تمیز."""
        self._stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._stop_event.set)

        cycles = 0
        while not self._stop_event.is_set():
            result = await self.run_cycle()
            cycles += 1
            logger.info("دور {} → {}", cycles, result)
            if max_cycles is not None and cycles >= max_cycles:
                break
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_sec)
        logger.info("خط لوله بعد از {} دور تمیز خاتمه یافت", cycles)
        return cycles


def main(argv: Optional[Sequence[str]] = None) -> int:
    """نقطه ورود خط لوله لحظه‌ای."""
    parser = argparse.ArgumentParser(description="خط لوله لحظه‌ای پیش‌بینی")
    parser.add_argument("--once", action="store_true", help="فقط یک دور")
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    pipeline = RealtimePipeline(symbols=args.symbols)
    asyncio.run(pipeline.run_forever(max_cycles=1 if args.once else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
