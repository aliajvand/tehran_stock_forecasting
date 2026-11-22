"""تست فیچرهای جریان پول (فاز ۲)."""

from __future__ import annotations

import pandas as pd

from src.features.money_flow import (
    compute_all_money_flow,
    compute_imbalance,
    compute_money_flow,
    compute_power_features,
)


def test_money_flow_net_values(sample_client_type: pd.DataFrame) -> None:
    """خالص جریان پول باید برابر خرید منهای فروش باشد."""
    out = compute_money_flow(sample_client_type)
    expected = out["val_buy_retail"] - out["val_sell_retail"]
    assert (out["net_val_retail"] - expected).abs().max() < 1e-6
    assert "net_val_retail_ma5" in out.columns


def test_imbalance_in_range(sample_client_type: pd.DataFrame) -> None:
    """عدم توازن باید بین -۱ و ۱ باشد."""
    out = compute_imbalance(compute_money_flow(sample_client_type))
    for col in ("n_imbalance", "vol_imbalance", "val_imbalance"):
        assert out[col].between(-1, 1).all()


def test_power_features(sample_client_type: pd.DataFrame) -> None:
    """فیچرهای قدرت خریدار باید ساخته شوند."""
    out = compute_power_features(sample_client_type)
    assert {"power_ratio", "power_retail_ma5", "power_retail_trend"} <= set(out.columns)
    assert not out["power_ratio"].isna().any()


def test_all_money_flow_no_nan(sample_client_type: pd.DataFrame) -> None:
    """زنجیره کامل نباید NaN در ستون‌های کلیدی بگذارد."""
    out = compute_all_money_flow(sample_client_type)
    assert not out["net_val_retail_ratio"].isna().any()
    assert len(out) == len(sample_client_type)
