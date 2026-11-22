"""تست شبیه‌ساز بک‌تست (فاز ۴)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.simulator import Simulator


def _signals() -> pd.DataFrame:
    """سیگنال ساده: خرید در روز اول، فروش در روز آخر با روند صعودی."""
    days = [20250101, 20250102, 20250103, 20250104, 20250105]
    prices = [100.0, 105.0, 110.0, 115.0, 120.0]
    signal = ["up", "neutral", "neutral", "neutral", "down"]
    return pd.DataFrame(
        {"symbol": ["فولاد"] * 5, "gregorian_date": days, "close": prices, "signal": signal}
    )


def test_profitable_trade() -> None:
    """روند صعودی باید سود بدهد و یک معامله ثبت شود."""
    result = Simulator(initial_capital=1_000_000, commission=0.0037, slippage=0.001).run(_signals())
    assert len(result.trades) == 1
    assert result.metrics["total_return"] > 0.10
    assert result.metrics["win_rate"] == 1.0


def test_commission_reduces_return() -> None:
    """کارمزد بالاتر باید بازدهی را کم کند."""
    low = Simulator(initial_capital=1_000_000, commission=0.0, slippage=0.0).run(_signals())
    high = Simulator(initial_capital=1_000_000, commission=0.02, slippage=0.01).run(_signals())
    assert high.metrics["total_return"] < low.metrics["total_return"]


def test_metrics_keys_and_equity_curve() -> None:
    """همه معیارهای کلیدی باید موجود باشند."""
    result = Simulator(initial_capital=1_000_000).run(_signals())
    assert {"total_return", "sharpe", "max_drawdown", "win_rate", "trades"} <= set(result.metrics)
    assert len(result.equity_curve) == 5
    assert (result.metrics["max_drawdown"] <= 0)


def test_numeric_signals_accepted() -> None:
    """سیگنال عددی (۰/۱/۲) هم باید پذیرفته شود."""
    frame = _signals()
    frame["signal"] = [2, 1, 1, 1, 0]
    result = Simulator(initial_capital=1_000_000).run(frame)
    assert len(result.trades) == 1


def test_html_report_created(tmp_path, monkeypatch) -> None:
    """گزارش HTML باید ساخته شود."""
    import src.backtesting.simulator as sim

    monkeypatch.setattr(sim, "resolve_path", lambda *a, **k: tmp_path)
    simulator = Simulator(initial_capital=1_000_000)
    path = simulator.report_html(simulator.run(_signals()), name="test_report")
    assert path.exists()
    assert "گزارش بک‌تست" in path.read_text(encoding="utf-8")
