"""
داشبورد Streamlit سیستم پیش‌بینی بورس تهران.

اجرا:
    streamlit run dashboard/app.py

صفحات:
    ۱) نمای کلی بازار
    ۲) پیش‌بینی هر نماد
    ۳) گراف جریان پول
    ۴) اخبار کدال با تحلیل احساسات
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# اجازه اجرای streamlit از ریشه پروژه
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from src import db as dbm  # noqa: E402
from src.config import get_target_symbols, setup_logging  # noqa: E402

setup_logging()
st.set_page_config(page_title="پیش‌بینی بورس تهران", page_icon="📈", layout="wide")
st.markdown(
    "<style>body,.stApp{direction:rtl;font-family:Tahoma,sans-serif}</style>",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def load_history(symbols: List[str]) -> pd.DataFrame:
    """خواندن تاریخچه روزانه (با کش ۵ دقیقه‌ای)."""
    return dbm.load_daily_history(symbols)


@st.cache_data(ttl=300)
def load_flow(symbols: List[str]) -> pd.DataFrame:
    """خواندن داده حقیقی/حقوقی."""
    return dbm.load_client_type(symbols)


@st.cache_data(ttl=300)
def load_news_df() -> pd.DataFrame:
    """خواندن اخبار کدال."""
    return dbm.load_news()


@st.cache_data(ttl=120)
def load_signals() -> pd.DataFrame:
    """خواندن سیگنال‌های ذخیره‌شده خط لوله لحظه‌ای."""
    path = _ROOT / "data" / "processed" / "latest_signals.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except OSError:
            return pd.DataFrame()
    return pd.DataFrame()


def page_market_overview(symbols: List[str]) -> None:
    """صفحه ۱: نمای کلی بازار."""
    st.header("📊 نمای کلی بازار")
    history = load_history(symbols)
    if history.empty:
        st.warning("داده‌ای در دیتابیس نیست. اول `python -m src.data_ingestion.backfill` را اجرا کنید.")
        return

    latest = history.sort_values("gregorian_date").groupby("symbol").tail(2)
    rows: List[Dict[str, Any]] = []
    for symbol, group in latest.groupby("symbol"):
        if len(group) < 2:
            continue
        prev, now = group.iloc[0], group.iloc[1]
        change = (now["close"] / prev["close"] - 1) if prev["close"] else 0.0
        rows.append({"نماد": symbol, "پایانی": now["close"], "تغییر": change, "حجم": now["volume"]})
    table = pd.DataFrame(rows)

    if not table.empty:
        cols = st.columns(4)
        cols[0].metric("تعداد نماد", len(table))
        cols[1].metric("مثبت", int((table["تغییر"] > 0).sum()))
        cols[2].metric("منفی", int((table["تغییر"] < 0).sum()))
        cols[3].metric("میانگین تغییر", f"{table['تغییر'].mean():.2%}")
        st.dataframe(
            table.style.format({"پایانی": "{:,.0f}", "تغییر": "{:.2%}", "حجم": "{:,.0f}"}),
            use_container_width=True,
        )

    signals = load_signals()
    if not signals.empty:
        st.subheader("سیگنال‌های اخیر مدل")
        st.dataframe(signals, use_container_width=True)


def page_symbol_prediction(symbols: List[str]) -> None:
    """صفحه ۲: پیش‌بینی هر نماد."""
    st.header("🔮 پیش‌بینی نماد")
    symbol = st.selectbox("نماد", symbols)
    history = load_history([symbol])
    if history.empty:
        st.warning("داده‌ای برای این نماد نیست.")
        return

    try:
        import plotly.graph_objects as go

        figure = go.Figure(
            go.Candlestick(
                x=history["date"], open=history["open"], high=history["high"],
                low=history["low"], close=history["close"], name=symbol,
            )
        )
        figure.update_layout(template="plotly_white", height=480, xaxis_rangeslider_visible=False)
        st.plotly_chart(figure, use_container_width=True)
    except ImportError:
        st.line_chart(history.set_index("date")["close"])

    signals = load_signals()
    row = signals[signals["symbol"] == symbol] if not signals.empty else pd.DataFrame()
    if not row.empty:
        item = row.iloc[0]
        cols = st.columns(3)
        cols[0].metric("سیگنال", str(item.get("signal", "-")))
        cols[1].metric("احتمال", f"{float(item.get('probability', 0)):.1%}")
        cols[2].metric("بازدهی تخمینی", f"{float(item.get('expected_return', 0)):.2%}")
    else:
        st.info("سیگنالی ذخیره نشده. `python -m src.realtime.pipeline --once` را اجرا کنید.")


def page_money_flow_graph(symbols: List[str]) -> None:
    """صفحه ۳: گراف جریان پول."""
    st.header("🕸️ گراف جریان پول")
    window = st.slider("پنجره (روز)", min_value=2, max_value=30, value=5)
    client = load_flow(symbols)
    if client.empty:
        st.warning("داده حقیقی/حقوقی موجود نیست.")
        return

    try:
        from src.features.graph_builder import build_money_flow_graph, graph_summary
        from src.features.money_flow import compute_money_flow

        graph = build_money_flow_graph(compute_money_flow(client), window_days=window)
    except ImportError as exc:
        st.error(f"برای این صفحه networkx لازم است: {exc}")
        return

    summary = graph_summary(graph)
    cols = st.columns(3)
    cols[0].metric("رأس", summary["nodes"])
    cols[1].metric("یال", summary["edges"])
    cols[2].metric("چگالی", summary["density"])
    st.subheader("قطب‌های جریان پول")
    st.dataframe(pd.DataFrame(summary["top_hubs"]), use_container_width=True)
    st.subheader("یال‌ها")
    edges = pd.DataFrame(
        [{"از": a, "به": b, "وزن": d.get("weight", 0)} for a, b, d in graph.edges(data=True)]
    )
    st.dataframe(edges.sort_values("وزن", ascending=False), use_container_width=True)


def page_news() -> None:
    """صفحه ۴: اخبار کدال با احساسات."""
    st.header("📰 اخبار کدال")
    news = load_news_df()
    if news.empty:
        st.warning("خبری ذخیره نشده. `python -m src.features.news_sentiment` را اجرا کنید.")
        return

    categories = ["همه"] + sorted(news["category"].dropna().unique().tolist())
    choice = st.selectbox("دسته احساسات", categories)
    view = news if choice == "همه" else news[news["category"] == choice]
    cols = st.columns(3)
    cols[0].metric("تعداد خبر", len(view))
    cols[1].metric("میانگین احساسات", f"{pd.to_numeric(view['sentiment'], errors='coerce').mean():.2f}")
    cols[2].metric("مثبت", int((view["category"] == "positive").sum()))
    st.dataframe(
        view[["symbol", "title", "publish_date", "sentiment", "category"]].head(200),
        use_container_width=True,
    )


def main() -> None:
    """نقطه ورود داشبورد."""
    st.sidebar.title("پیش‌بینی بورس تهران")
    page = st.sidebar.radio(
        "صفحه",
        ["نمای کلی بازار", "پیش‌بینی نماد", "گراف جریان پول", "اخبار کدال"],
    )
    symbols = get_target_symbols()
    if page == "نمای کلی بازار":
        page_market_overview(symbols)
    elif page == "پیش‌بینی نماد":
        page_symbol_prediction(symbols)
    elif page == "گراف جریان پول":
        page_money_flow_graph(symbols)
    else:
        page_news()
    st.sidebar.caption("داده: TSETMC + کدال | مدل: LSTM + GNN")


main()
