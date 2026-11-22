"""
شبیه‌ساز معاملات (بک‌تست) با کارمزد و اسلیپیج.

معیارها: بازدهی کل، Sharpe، حداکثر افت سرمایه (MaxDD)، Win Rate، تعداد معاملات.
گزارش خروجی: HTML (و در صورت نصب بودن weasyprint، PDF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_section, resolve_path


@dataclass
class Trade:
    """یک معامله کامل (ورود و خروج)."""

    symbol: str
    entry_date: int
    entry_price: float
    exit_date: Optional[int] = None
    exit_price: Optional[float] = None
    shares: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0


@dataclass
class BacktestResult:
    """نتیجه کامل بک‌تست."""

    equity_curve: pd.DataFrame
    trades: List[Trade] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class Simulator:
    """
    شبیه‌ساز ساده «سیگنال‌محور».

    قاعده: سیگنال ``up`` = ورود، سیگنال ``down`` = خروج، ``neutral`` = نگه‌داشتن.
    خرید در قیمت پایانی روز سیگنال + اسلیپیج، با کارمزد دوطرفه.
    """

    def __init__(
        self,
        initial_capital: Optional[float] = None,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        max_positions: Optional[int] = None,
    ) -> None:
        """پارامترها پیش‌فرض از بخش ``backtesting`` فایل config خوانده می‌شوند."""
        cfg = get_section("backtesting", {})
        self.initial_capital = float(initial_capital or cfg.get("initial_capital", 100_000_000))
        self.commission = float(commission if commission is not None else cfg.get("commission", 0.0037))
        self.slippage = float(slippage if slippage is not None else cfg.get("slippage", 0.001))
        self.max_positions = int(max_positions or cfg.get("max_positions", 5))

    # ---------------------------------------------------------------- #
    def run(self, signals: pd.DataFrame) -> BacktestResult:
        """
        اجرای بک‌تست.

        ورودی ``signals``: ستون‌های ``symbol, gregorian_date, close, signal``
        که ``signal`` یکی از ``up/down/neutral`` یا ۰/۱/۲ است.
        """
        required = {"symbol", "gregorian_date", "close", "signal"}
        missing = required - set(signals.columns)
        if missing:
            raise ValueError(f"ستون‌های لازم موجود نیست: {sorted(missing)}")
        if signals.empty:
            logger.warning("سیگنالی برای بک‌تست وجود ندارد")
            return BacktestResult(pd.DataFrame(columns=["date", "equity"]))

        data = signals.copy()
        data["signal"] = data["signal"].map(
            lambda v: {0: "down", 1: "neutral", 2: "up"}.get(v, v) if not isinstance(v, str) else v
        )
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data = data.dropna(subset=["close"]).sort_values(["gregorian_date", "symbol"])

        cash = self.initial_capital
        positions: Dict[str, Trade] = {}
        trades: List[Trade] = []
        curve: List[Dict[str, Any]] = []

        for day, group in data.groupby("gregorian_date", sort=True):
            prices = dict(zip(group["symbol"], group["close"]))

            day_signals = dict(zip(group["symbol"], group["signal"].astype(str)))

            # ابتدا خروج‌ها (آزادسازی نقدینگی برای ورودهای همان روز)
            for symbol in list(positions):
                if day_signals.get(symbol) == "down":
                    trade = positions.pop(symbol)
                    exit_price = float(prices.get(symbol, trade.entry_price)) * (1 - self.slippage)
                    proceeds = trade.shares * exit_price * (1 - self.commission)
                    cash += proceeds
                    trade.exit_date, trade.exit_price = int(day), exit_price
                    trade.pnl = proceeds - trade.shares * trade.entry_price * (1 + self.commission)
                    trade.return_pct = trade.pnl / max(1e-9, trade.shares * trade.entry_price)
                    trades.append(trade)

            # سپس ورودها
            free_slots = max(0, self.max_positions - len(positions))
            buys = [s for s, sig in day_signals.items() if sig == "up" and s not in positions]
            buys = buys[:free_slots]
            # سرمایه بین خریدهای واقعی همان روز تقسیم می‌شود (نه بین اسلات‌های خالی)
            per_trade = cash / max(1, len(buys))
            for symbol in buys:
                price = float(prices.get(symbol, 0.0)) * (1 + self.slippage)
                if price <= 0:
                    continue
                budget = per_trade
                shares = budget / (price * (1 + self.commission))
                if shares <= 0:
                    continue
                cost = shares * price * (1 + self.commission)
                cash -= cost
                positions[symbol] = Trade(symbol=symbol, entry_date=int(day), entry_price=price, shares=shares)

            market_value = sum(t.shares * float(prices.get(s, t.entry_price)) for s, t in positions.items())
            curve.append({"date": int(day), "equity": cash + market_value, "cash": cash,
                          "positions": len(positions)})

        # بستن پوزیشن‌های باز در آخرین قیمت
        if positions:
            last_prices = data.groupby("symbol")["close"].last().to_dict()
            last_day = int(data["gregorian_date"].max())
            for symbol, trade in positions.items():
                exit_price = float(last_prices.get(symbol, trade.entry_price)) * (1 - self.slippage)
                proceeds = trade.shares * exit_price * (1 - self.commission)
                cash += proceeds
                trade.exit_date, trade.exit_price = last_day, exit_price
                trade.pnl = proceeds - trade.shares * trade.entry_price * (1 + self.commission)
                trade.return_pct = trade.pnl / max(1e-9, trade.shares * trade.entry_price)
                trades.append(trade)
            if curve:
                curve[-1]["equity"] = cash

        equity = pd.DataFrame(curve)
        metrics = self.compute_metrics(equity, trades)
        logger.success(
            "بک‌تست: بازدهی {:.2%} | Sharpe {:.2f} | MaxDD {:.2%} | WinRate {:.1%} | {} معامله",
            metrics.get("total_return", 0), metrics.get("sharpe", 0),
            metrics.get("max_drawdown", 0), metrics.get("win_rate", 0), len(trades),
        )
        return BacktestResult(equity_curve=equity, trades=trades, metrics=metrics)

    # ---------------------------------------------------------------- #
    def compute_metrics(self, equity: pd.DataFrame, trades: List[Trade]) -> Dict[str, float]:
        """محاسبه معیارهای عملکرد از منحنی سرمایه و لیست معاملات."""
        if equity.empty:
            return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
                    "trades": 0, "final_equity": self.initial_capital}

        series = pd.to_numeric(equity["equity"], errors="coerce").ffill()
        returns = series.pct_change().dropna()
        peak = series.cummax()
        drawdown = (series - peak) / peak.replace(0, np.nan)
        wins = [t for t in trades if t.pnl > 0]

        sharpe = 0.0
        if len(returns) > 2 and returns.std(ddof=0) > 0:
            sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252))

        return {
            "total_return": float(series.iloc[-1] / self.initial_capital - 1),
            "final_equity": float(series.iloc[-1]),
            "sharpe": sharpe,
            "max_drawdown": float(drawdown.min() if len(drawdown) else 0.0),
            "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
            "trades": len(trades),
            "avg_trade_return": float(np.mean([t.return_pct for t in trades])) if trades else 0.0,
        }

    # ---------------------------------------------------------------- #
    def report_html(self, result: BacktestResult, name: str = "backtest") -> Path:
        """
        ساخت گزارش HTML (نمودار سرمایه با plotly در صورت نصب بودن).

        خروجی: مسیر فایل ساخته‌شده در ``logs/evaluation/``.
        """
        reports = resolve_path(str(get_section("paths").get("reports", "logs/evaluation")))
        reports.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = reports / f"{name}_{stamp}.html"

        rows = "".join(
            f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" for k, v in result.metrics.items()
        )
        chart = "<p>برای نمودار، plotly را نصب کنید.</p>"
        if not result.equity_curve.empty:
            try:
                import plotly.graph_objects as go

                dates = pd.to_datetime(result.equity_curve["date"].astype(str), format="%Y%m%d")
                figure = go.Figure(
                    go.Scatter(x=dates, y=result.equity_curve["equity"], mode="lines", name="سرمایه")
                )
                figure.update_layout(title="منحنی سرمایه", template="plotly_white")
                chart = figure.to_html(full_html=False, include_plotlyjs="cdn")
            except ImportError:
                pass

        trade_rows = "".join(
            f"<tr><td>{t.symbol}</td><td>{t.entry_date}</td><td>{t.exit_date}</td>"
            f"<td>{t.return_pct:.2%}</td><td>{t.pnl:,.0f}</td></tr>"
            for t in result.trades[:100]
        )
        html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<title>گزارش بک‌تست</title>
<style>body{{font-family:Tahoma,sans-serif;margin:24px;background:#fafafa}}
table{{border-collapse:collapse;margin:16px 0;background:#fff}}
td,th{{border:1px solid #ddd;padding:6px 12px;text-align:right}}
th{{background:#f0f0f0}}h1,h2{{color:#222}}</style></head>
<body><h1>گزارش بک‌تست</h1><p>تاریخ تولید: {stamp}</p>
<h2>معیارها</h2><table><tr><th>معیار</th><th>مقدار</th></tr>{rows}</table>
<h2>منحنی سرمایه</h2>{chart}
<h2>معاملات (حداکثر ۱۰۰ مورد)</h2>
<table><tr><th>نماد</th><th>ورود</th><th>خروج</th><th>بازدهی</th><th>سود/زیان</th></tr>{trade_rows}</table>
</body></html>"""
        try:
            path.write_text(html, encoding="utf-8")
            logger.success("گزارش بک‌تست: {}", path)
        except OSError as exc:
            logger.error("نوشتن گزارش ناموفق بود: {}", exc)
        return path
