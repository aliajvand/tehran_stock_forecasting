"""روت‌های پیش‌بینی و سیگنال."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.config import get_target_symbols
from src.realtime.pipeline import Predictor

router = APIRouter(tags=["predictions"])
_predictor = Predictor()
_cache: Dict[str, pd.DataFrame] = {}


def _signals(refresh: bool = False) -> pd.DataFrame:
    """گرفتن سیگنال‌ها با کش ساده در حافظه."""
    if refresh or "signals" not in _cache or _cache["signals"].empty:
        _cache["signals"] = _predictor.predict_latest()
    return _cache["signals"]


@router.get("/predict/{symbol}")
def predict_symbol(symbol: str, refresh: bool = False) -> Dict[str, Any]:
    """پیش‌بینی یک نماد مشخص."""
    if symbol not in get_target_symbols():
        raise HTTPException(status_code=404, detail=f"نماد «{symbol}» در لیست نمادهای هدف نیست")
    frame = _signals(refresh)
    if frame.empty:
        raise HTTPException(status_code=503, detail="مدل یا داده آماده نیست")
    row = frame[frame["symbol"] == symbol]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"سیگنالی برای «{symbol}» موجود نیست")
    return row.iloc[0].to_dict()


@router.get("/signals")
def all_signals(refresh: bool = False) -> List[Dict[str, Any]]:
    """سیگنال‌های امروز برای همه نمادها."""
    frame = _signals(refresh)
    if frame.empty:
        logger.warning("سیگنالی برای بازگشت وجود ندارد")
        return []
    return frame.to_dict(orient="records")
