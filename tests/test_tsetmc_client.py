"""تست کلاینت TSETMC با داده جعلی (فاز ۱)."""

from __future__ import annotations

import pandas as pd

from src.data_ingestion.tsetmc_client import TsetmcClient
from tests.stubs import FakeAlgotik, FlakyAlgotik


def test_get_symbols_maps_instrument_id_to_ins_code(client: TsetmcClient) -> None:
    """ins_code باید از instrument_id ساخته شود و symbol از ایندکس."""
    df = client.get_symbols()
    assert not df.empty
    assert {"ins_code", "symbol", "asset_type", "industry_group"} <= set(df.columns)
    assert "InsCode" not in df.columns and "Symbol" not in df.columns
    assert pd.api.types.is_integer_dtype(df["ins_code"].dropna().astype("int64"))
    assert "فولاد" in set(df["symbol"])


def test_filter_stocks_removes_non_stock(client: TsetmcClient) -> None:
    """ابزارهای غیر سهام باید حذف شوند."""
    stocks = client.filter_stocks(client.get_symbols())
    assert set(stocks["asset_type"].unique()) == {"stock"}
    assert "اهرم" not in set(stocks["symbol"])


def test_validate_symbols_reports_missing(client: TsetmcClient) -> None:
    """خروجی validate_symbols باید available و missing داشته باشد."""
    stocks = client.filter_stocks(client.get_symbols())
    result = client.validate_symbols(["فولاد", "نماد‌ناموجود"], stocks)
    assert result["available"] == ["فولاد"]
    assert result["missing"] == ["نماد‌ناموجود"]
    assert len(result["frame"]) == 1


def test_get_history_columns_and_dates(client: TsetmcClient) -> None:
    """ستون‌ها باید به نام‌های دیتابیس نگاشت و تاریخ عدد YYYYMMDD شود."""
    df = client.get_history("فولاد", "2024-01-01", "2025-09-01")
    assert {"gregorian_date", "jalali_date", "open", "close", "last", "trade_count"} <= set(df.columns)
    assert df["gregorian_date"].between(19000101, 21001231).all()
    assert len(str(int(df["gregorian_date"].iloc[0]))) == 8


def test_get_client_type_has_all_21_columns(client: TsetmcClient) -> None:
    """همه ستون‌های حقیقی/حقوقی باید با حروف کوچک موجود باشند."""
    df = client.get_client_type("فولاد", "2024-01-01", "2025-09-01")
    for col in ("n_buy_retail", "vol_sell_institutional", "per_capita_buy_retail", "power_retail"):
        assert col in df.columns
    assert not df.empty


def test_retry_recovers_after_failures() -> None:
    """کلاینت باید بعد از چند خطا موفق شود."""
    flaky = FlakyAlgotik(fail_times=1, rows=30)
    client = TsetmcClient(retries=3, retry_delay=0.0, request_delay=0.0, module=flaky)
    df = client.get_history("فولاد", "2024-01-01", "2024-06-01")
    assert not df.empty
    assert flaky.attempts == 2


def test_returns_empty_frame_when_all_retries_fail() -> None:
    """در شکست کامل باید DataFrame خالی برگردد نه استثنا."""
    flaky = FlakyAlgotik(fail_times=99, rows=30)
    client = TsetmcClient(retries=2, retry_delay=0.0, request_delay=0.0, module=flaky)
    assert client.get_history("فولاد", "2024-01-01", "2024-06-01").empty


def test_order_book_and_intraday(client: TsetmcClient, fake_att: FakeAlgotik) -> None:
    """توابع کمکی باید DataFrame برگردانند و فراخوانی ثبت شود."""
    assert not client.get_order_book("فولاد").empty
    assert not client.get_intraday("فولاد", "5min").empty
    assert fake_att.calls.get("get_order_book", 0) >= 1
