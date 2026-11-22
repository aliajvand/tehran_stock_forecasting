"""
اخبار کدال + تحلیل احساسات فارسی.

جریان کار:
    ۱) ``CodalScraper`` اطلاعیه‌های هر نماد را از سرویس جست‌وجوی کدال می‌گیرد
       (اگر پکیج ``codalpy`` نصب باشد، اول از آن استفاده می‌شود).
    ۲) ``SentimentAnalyzer`` متن را با ``hazm`` نرمال می‌کند و با یک مدل
       HuggingFace فارسی امتیاز می‌دهد؛ در نبود مدل، از واژگان کلیدی
       (rule-based) استفاده می‌شود تا خط لوله هرگز قطع نشود.
    ۳) نتیجه در جدول ``codal_news`` ذخیره می‌شود و فیچر روزانه ساخته می‌شود.

اجرا:
    python -m src.features.news_sentiment --symbols فولاد --pages 1
"""

from __future__ import annotations

import argparse
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger

from src import db as dbm
from src.config import get_db_path, get_section, get_target_symbols, setup_logging

# واژگان جایگزین برای زمانی که مدل ترنسفورمر در دسترس نیست
_POSITIVE_WORDS: tuple[str, ...] = (
    "افزایش", "سود", "رشد", "سودآور", "تعدیل مثبت", "افزایش سرمایه", "قرارداد",
    "پیش‌بینی سود", "بهبود", "صعود", "فروش بیشتر", "درآمد", "توسعه", "موافقت",
)
_NEGATIVE_WORDS: tuple[str, ...] = (
    "کاهش", "زیان", "تعدیل منفی", "توقف نماد", "تعلیق", "ابطال", "جریمه", "بدهی",
    "افت", "شکایت", "لغو", "خطر", "کاهش سرمایه", "توقف تولید", "تحریم",
)


def _normalize(text: str) -> str:
    """نرمال‌سازی متن فارسی با hazm (در نبود hazm، پاک‌سازی ساده)."""
    clean = " ".join(str(text or "").split())
    try:
        from hazm import Normalizer

        return Normalizer().normalize(clean)
    except ImportError:
        return clean.replace("ي", "ی").replace("ك", "ک")


class SentimentAnalyzer:
    """تحلیل احساسات متن فارسی با مدل HuggingFace و fallback واژگانی."""

    def __init__(self, model_name: Optional[str] = None, use_transformer: bool = True) -> None:
        """
        پارامترها:
            model_name: نام مدل HuggingFace (پیش‌فرض از ``config.news.sentiment_model``).
            use_transformer: اگر False باشد، فقط روش واژگانی اجرا می‌شود.
        """
        news_cfg = get_section("news", {})
        self.model_name = model_name or str(news_cfg.get("sentiment_model", ""))
        self._pipeline: Optional[Any] = None
        if use_transformer and self.model_name:
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        """بارگذاری تنبل مدل ترنسفورمر (در صورت شکست، fallback فعال می‌ماند)."""
        try:
            from transformers import pipeline

            self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
            logger.info("مدل احساسات «{}» بارگذاری شد", self.model_name)
        except Exception as exc:  # noqa: BLE001 - نبود مدل/اینترنت/حافظه
            logger.warning("بارگذاری مدل احساسات ناموفق بود؛ روش واژگانی فعال شد ({})", exc)
            self._pipeline = None

    def _lexicon_score(self, text: str) -> float:
        """امتیاز احساسات بر اساس شمارش واژگان مثبت/منفی در بازه [-1, 1]."""
        pos = sum(text.count(w) for w in _POSITIVE_WORDS)
        neg = sum(text.count(w) for w in _NEGATIVE_WORDS)
        if pos + neg == 0:
            return 0.0
        return round((pos - neg) / (pos + neg), 3)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        تحلیل احساسات یک متن.

        خروجی: ``{"score": float در [-1,1], "label": positive|negative|neutral,
        "method": transformer|lexicon}``
        """
        clean = _normalize(text)
        if not clean:
            return {"score": 0.0, "label": "neutral", "method": "empty"}

        if self._pipeline is not None:
            try:
                result = self._pipeline(clean[:512])[0]
                label = str(result.get("label", "")).lower()
                score = float(result.get("score", 0.0))
                signed = score if any(k in label for k in ("pos", "happy", "1")) else -score
                return {
                    "score": round(signed, 3),
                    "label": "positive" if signed > 0.15 else "negative" if signed < -0.15 else "neutral",
                    "method": "transformer",
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("خطای مدل احساسات؛ fallback واژگانی: {}", exc)

        score = self._lexicon_score(clean)
        return {
            "score": score,
            "label": "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral",
            "method": "lexicon",
        }


class CodalScraper:
    """دریافت اطلاعیه‌های کدال برای یک نماد."""

    def __init__(self, base_url: Optional[str] = None, request_delay: Optional[float] = None) -> None:
        """پارامترها از بخش ``news`` فایل config خوانده می‌شود."""
        news_cfg = get_section("news", {})
        self.base_url = str(base_url or news_cfg.get("codal_base_url", "https://search.codal.ir")).rstrip("/")
        self.request_delay = float(request_delay or news_cfg.get("request_delay_sec", 1.0))

    def _via_codalpy(self, symbol: str) -> List[Dict[str, Any]]:
        """تلاش برای استفاده از پکیج ``codalpy`` (اگر نصب باشد)."""
        try:
            import codalpy  # type: ignore[import-not-found]
        except ImportError:
            return []
        try:
            raw = codalpy.get_letters(symbol)  # type: ignore[attr-defined]
            frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            rows: List[Dict[str, Any]] = []
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "title": str(row.get("title", "")),
                        "publish_date": str(row.get("publish_date", row.get("date", ""))),
                        "publish_time": str(row.get("publish_time", "")),
                        "url": str(row.get("url", "")),
                        "news_type": str(row.get("letter_type", "")),
                    }
                )
            logger.info("codalpy: {} اطلاعیه برای «{}»", len(rows), symbol)
            return rows
        except Exception as exc:  # noqa: BLE001
            logger.warning("codalpy برای «{}» جواب نداد: {}", symbol, exc)
            return []

    def fetch(self, symbol: str, pages: int = 1) -> List[Dict[str, Any]]:
        """
        دریافت لیست اطلاعیه‌های یک نماد.

        ابتدا ``codalpy`` و در صورت نبود آن، جست‌وجوی HTTP کدال استفاده می‌شود.
        """
        rows = self._via_codalpy(symbol)
        if rows:
            return rows

        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("requests/beautifulsoup4 نصب نیست؛ اسکرپ کدال ممکن نیست")
            return []

        collected: List[Dict[str, Any]] = []
        for page in range(1, max(1, pages) + 1):
            params = {"Symbol": symbol, "PageNumber": page, "search": "true"}
            try:
                resp = requests.get(f"{self.base_url}/", params=params, timeout=20)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001 - خطای شبکه/تایم‌اوت
                logger.warning("درخواست کدال برای «{}» صفحه {} ناموفق: {}", symbol, page, exc)
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            rows_html = soup.select("table tbody tr")
            if not rows_html:
                logger.debug("صفحه {} کدال برای «{}» ردیفی نداشت", page, symbol)
                break
            for tr in rows_html:
                cells = [c.get_text(strip=True) for c in tr.find_all("td")]
                link = tr.find("a", href=True)
                if len(cells) < 3:
                    continue
                collected.append(
                    {
                        "title": cells[2] if len(cells) > 2 else cells[-1],
                        "publish_date": cells[-2] if len(cells) > 3 else "",
                        "publish_time": cells[-1] if len(cells) > 3 else "",
                        "url": str(link["href"]) if link else "",
                        "news_type": cells[1] if len(cells) > 1 else "",
                    }
                )
            time.sleep(self.request_delay)  # رعایت ادب شبکه
        logger.info("کدال: {} اطلاعیه برای «{}»", len(collected), symbol)
        return collected


def _news_id(symbol: str, title: str, publish_date: str, url: str) -> str:
    """ساخت شناسه یکتا و پایدار برای هر خبر (برای UNIQUE بودن news_id)."""
    raw = f"{symbol}|{title}|{publish_date}|{url}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]


def collect_and_store_news(
    symbols: Optional[Sequence[str]] = None,
    pages: Optional[int] = None,
    analyzer: Optional[SentimentAnalyzer] = None,
    scraper: Optional[CodalScraper] = None,
    conn: Optional[Any] = None,
) -> int:
    """
    دریافت اخبار، تحلیل احساسات و ذخیره در جدول ``codal_news``.

    خروجی: تعداد سطرهای ذخیره‌شده.
    """
    targets = list(symbols or get_target_symbols())
    news_cfg = get_section("news", {})
    page_count = int(pages or news_cfg.get("max_pages", 3))
    scraper = scraper or CodalScraper()
    analyzer = analyzer or SentimentAnalyzer()

    own_connection = conn is None
    conn = conn if conn is not None else dbm.connect(get_db_path())
    try:
        codes = {str(s): int(c) for s, c in
                 conn.execute("SELECT symbol, ins_code FROM instruments").fetchall()}
        total = 0
        for symbol in targets:
            items = scraper.fetch(symbol, pages=page_count)
            if not items:
                continue
            rows: List[Dict[str, Any]] = []
            for item in items:
                text = f"{item.get('title', '')} {item.get('content', '')}"
                verdict = analyzer.analyze_sentiment(text)
                rows.append(
                    {
                        "news_id": _news_id(symbol, item.get("title", ""),
                                            item.get("publish_date", ""), item.get("url", "")),
                        "ins_code": codes.get(symbol),
                        "symbol": symbol,
                        "title": _normalize(item.get("title", "")),
                        "content": _normalize(item.get("content", "")),
                        "news_type": item.get("news_type", ""),
                        "publish_date": item.get("publish_date", ""),
                        "publish_time": item.get("publish_time", ""),
                        "url": item.get("url", ""),
                        "attachment_url": item.get("attachment_url", ""),
                        "sentiment": verdict["score"],
                        "category": verdict["label"],
                    }
                )
            total += dbm.upsert_frame(conn, "codal_news", pd.DataFrame(rows), ("news_id",))
        logger.success("codal_news: {} خبر ذخیره شد", total)
        return total
    finally:
        if own_connection:
            conn.close()


def build_daily_sentiment_features(
    news_df: Optional[pd.DataFrame] = None, decay_days: int = 3
) -> pd.DataFrame:
    """
    تبدیل اخبار به فیچر روزانه هر نماد.

    خروجی: ``symbol, gregorian_date, news_count, sentiment_mean, sentiment_sum,
    sentiment_decayed``
    """
    df = news_df if news_df is not None else dbm.load_news()
    if df is None or df.empty:
        logger.warning("خبری برای ساخت فیچر احساسات وجود ندارد")
        return pd.DataFrame(columns=["symbol", "gregorian_date", "news_count", "sentiment_mean"])

    out = df.copy()
    dates = pd.to_datetime(out["publish_date"], errors="coerce")
    out["gregorian_date"] = dates.dt.strftime("%Y%m%d").astype("Int64")
    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["gregorian_date"])

    agg = out.groupby(["symbol", "gregorian_date"], as_index=False).agg(
        news_count=("sentiment", "size"),
        sentiment_mean=("sentiment", "mean"),
        sentiment_sum=("sentiment", "sum"),
    )
    agg = agg.sort_values(["symbol", "gregorian_date"])
    agg["sentiment_decayed"] = agg.groupby("symbol")["sentiment_mean"].transform(
        lambda s: s.ewm(halflife=max(1, decay_days), adjust=False).mean()
    )
    logger.info("فیچر احساسات روزانه برای {} سطر ساخته شد", len(agg))
    return agg


def main(argv: Optional[Sequence[str]] = None) -> int:
    """نقطه ورود: دریافت و ذخیره اخبار کدال."""
    parser = argparse.ArgumentParser(description="دریافت اخبار کدال و تحلیل احساسات")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--pages", type=int, default=None)
    parser.add_argument("--no-transformer", action="store_true", help="فقط روش واژگانی")
    args = parser.parse_args(argv)

    setup_logging()
    analyzer = SentimentAnalyzer(use_transformer=not args.no_transformer)
    collect_and_store_news(symbols=args.symbols, pages=args.pages, analyzer=analyzer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
