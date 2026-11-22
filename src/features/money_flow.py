"""
فیچرهای جریان پول از داده حقیقی/حقوقی (جدول ``client_type``).

مفاهیم:
    * خالص ورود پول حقیقی = ارزش خرید حقیقی − ارزش فروش حقیقی
    * عدم توازن (imbalance) = خالص / کل
    * قدرت خریدار (power) = سرانه خرید / سرانه فروش
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_section


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """تقسیم ایمن (تقسیم بر صفر → NaN → صفر)."""
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_money_flow(client_df: pd.DataFrame, windows: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """
    خالص جریان پول حقیقی/حقوقی + میانگین متحرک آن.

    ورودی: خروجی ``db.load_client_type`` یا ``cleaner.clean_client_type``.
    """
    if client_df is None or client_df.empty:
        logger.warning("compute_money_flow: ورودی خالی است")
        return pd.DataFrame()

    wins = list(windows or get_section("features", {}).get("money_flow_windows", [1, 5, 20]))
    out = client_df.copy()
    sort_cols = [c for c in ("symbol", "gregorian_date") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)

    for col in [c for c in out.columns if c.startswith(("n_", "vol_", "val_", "per_capita_", "power_"))]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["net_val_retail"] = out.get("val_buy_retail", 0) - out.get("val_sell_retail", 0)
    out["net_val_institutional"] = out.get("val_buy_institutional", 0) - out.get("val_sell_institutional", 0)
    out["net_vol_retail"] = out.get("vol_buy_retail", 0) - out.get("vol_sell_retail", 0)
    out["net_vol_institutional"] = out.get("vol_buy_institutional", 0) - out.get("vol_sell_institutional", 0)

    total_val = (
        out.get("val_buy_retail", 0)
        + out.get("val_buy_institutional", 0)
        + out.get("val_sell_retail", 0)
        + out.get("val_sell_institutional", 0)
    )
    out["net_val_retail_ratio"] = _safe_div(out["net_val_retail"], total_val)
    out["net_val_inst_ratio"] = _safe_div(out["net_val_institutional"], total_val)

    def _rolling(group: pd.DataFrame) -> pd.DataFrame:
        """میانگین متحرک جریان پول یک نماد."""
        g = group.copy()
        for w in wins:
            w = int(w)
            g[f"net_val_retail_ma{w}"] = g["net_val_retail"].rolling(w, min_periods=1).mean()
            g[f"net_val_inst_ma{w}"] = g["net_val_institutional"].rolling(w, min_periods=1).mean()
            g[f"net_val_retail_sum{w}"] = g["net_val_retail"].rolling(w, min_periods=1).sum()
        return g

    if "symbol" in out.columns:
        parts = [_rolling(g) for _, g in out.groupby("symbol", sort=False)]
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        out = _rolling(out).reset_index(drop=True)

    logger.info("فیچرهای جریان پول ساخته شد ({} سطر)", len(out))
    return out


def compute_imbalance(client_df: pd.DataFrame) -> pd.DataFrame:
    """عدم توازن خرید/فروش در تعداد، حجم و ارزش."""
    if client_df is None or client_df.empty:
        return pd.DataFrame()

    out = client_df.copy()
    for prefix in ("n", "vol", "val"):
        buy = out.get(f"{prefix}_buy_retail", 0) + out.get(f"{prefix}_buy_institutional", 0)
        sell = out.get(f"{prefix}_sell_retail", 0) + out.get(f"{prefix}_sell_institutional", 0)
        out[f"{prefix}_imbalance"] = _safe_div(buy - sell, buy + sell)
    out["retail_inst_imbalance"] = _safe_div(
        out.get("net_val_retail", out.get("val_buy_retail", 0) - out.get("val_sell_retail", 0)),
        (out.get("val_buy_retail", 0) + out.get("val_sell_retail", 0)),
    )
    return out


def compute_power_features(client_df: pd.DataFrame) -> pd.DataFrame:
    """
    فیچرهای قدرت معامله‌گر از ``power_retail`` / ``power_institutional`` و سرانه‌ها.
    """
    if client_df is None or client_df.empty:
        return pd.DataFrame()

    out = client_df.copy()
    out["power_ratio"] = _safe_div(
        pd.to_numeric(out.get("power_retail", 0), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("power_institutional", 1), errors="coerce").fillna(0.0),
    )
    out["per_capita_buy_ratio"] = _safe_div(
        pd.to_numeric(out.get("per_capita_buy_retail", 0), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("per_capita_sell_retail", 1), errors="coerce").fillna(0.0),
    )

    def _trend(group: pd.DataFrame) -> pd.DataFrame:
        """روند ۵ روزه قدرت خریدار حقیقی."""
        g = group.copy()
        g["power_retail_ma5"] = pd.to_numeric(g.get("power_retail", 0), errors="coerce").rolling(
            5, min_periods=1
        ).mean()
        g["power_retail_trend"] = g["power_retail_ma5"].diff()
        return g

    if "symbol" in out.columns:
        parts = [_trend(g) for _, g in out.groupby("symbol", sort=False)]
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        out = _trend(out).reset_index(drop=True)
    return out


def compute_all_money_flow(client_df: pd.DataFrame) -> pd.DataFrame:
    """اجرای زنجیره کامل فیچرهای جریان پول."""
    out = compute_money_flow(client_df)
    out = compute_imbalance(out)
    out = compute_power_features(out)
    return out
