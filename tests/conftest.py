"""تنظیمات مشترک pytest: دیتابیس موقت، کلاینت جعلی و داده نمونه."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import db as dbm  # noqa: E402
from src.data_ingestion.tsetmc_client import TsetmcClient  # noqa: E402
from tests.stubs import FakeAlgotik, create_test_schema  # noqa: E402


@pytest.fixture()
def fake_att() -> FakeAlgotik:
    """ماژول جعلی algotik-tse."""
    return FakeAlgotik(rows=180)


@pytest.fixture()
def client(fake_att: FakeAlgotik) -> TsetmcClient:
    """کلاینت با ماژول تزریق‌شده و بدون تأخیر شبکه."""
    return TsetmcClient(retries=2, retry_delay=0.0, request_delay=0.0, module=fake_att)


@pytest.fixture()
def temp_db(tmp_path: Path) -> Iterator[Any]:
    """دیتابیس SQLite موقت با اسکیمای تست."""
    path = tmp_path / "test_bourse.db"
    create_test_schema(str(path))
    conn = dbm.connect(path)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def sample_history(client: TsetmcClient) -> pd.DataFrame:
    """تاریخچه نمونه دو نماد با ستون‌های دیتابیس."""
    frames = []
    for i, symbol in enumerate(("فولاد", "فخوز")):
        df = client.get_history(symbol, "2024-01-01", "2025-09-01")
        df["symbol"] = symbol
        df["ins_code"] = 1000 + i
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["gregorian_date"].astype(str), format="%Y%m%d")
    return out


@pytest.fixture()
def sample_client_type(client: TsetmcClient) -> pd.DataFrame:
    """داده حقیقی/حقوقی نمونه دو نماد."""
    frames = []
    for i, symbol in enumerate(("فولاد", "فخوز")):
        df = client.get_client_type(symbol, "2024-01-01", "2025-09-01")
        df["symbol"] = symbol
        df["ins_code"] = 1000 + i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
