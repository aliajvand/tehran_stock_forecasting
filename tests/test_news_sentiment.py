"""تست تحلیل احساسات و فیچر خبری (فاز ۲/۳)."""

from __future__ import annotations

import pandas as pd

from src.features.news_sentiment import (
    SentimentAnalyzer,
    build_daily_sentiment_features,
    collect_and_store_news,
)


class FakeScraper:
    """اسکرپر جعلی کدال."""

    def fetch(self, symbol: str, pages: int = 1) -> list[dict]:
        """دو خبر ثابت برمی‌گرداند."""
        return [
            {"title": "افزایش سود خالص و رشد فروش", "publish_date": "2025-08-20",
             "publish_time": "10:00", "url": "http://x/1", "news_type": "اطلاعیه"},
            {"title": "زیان انباشته و توقف نماد", "publish_date": "2025-08-21",
             "publish_time": "11:00", "url": "http://x/2", "news_type": "اطلاعیه"},
        ]


def test_lexicon_sentiment_direction() -> None:
    """متن مثبت و منفی باید برچسب درست بگیرند."""
    analyzer = SentimentAnalyzer(use_transformer=False)
    assert analyzer.analyze_sentiment("افزایش سود و رشد درآمد")["label"] == "positive"
    assert analyzer.analyze_sentiment("زیان و توقف نماد و جریمه")["label"] == "negative"
    assert analyzer.analyze_sentiment("مجمع عمومی برگزار شد")["label"] == "neutral"


def test_store_news_and_features(temp_db) -> None:
    """اخبار باید در codal_news ذخیره و به فیچر روزانه تبدیل شوند."""
    temp_db.execute(
        "INSERT OR REPLACE INTO instruments (ins_code, symbol, asset_type) VALUES (1000,'فولاد','stock')"
    )
    temp_db.commit()

    count = collect_and_store_news(
        symbols=["فولاد"], pages=1,
        analyzer=SentimentAnalyzer(use_transformer=False), scraper=FakeScraper(),
        conn=temp_db,
    )
    assert count == 2

    news = pd.read_sql_query("SELECT * FROM codal_news", temp_db)
    assert len(news) == 2
    features = build_daily_sentiment_features(news)
    assert {"symbol", "gregorian_date", "sentiment_mean", "sentiment_decayed"} <= set(features.columns)
    assert len(features) == 2


def test_news_id_is_stable() -> None:
    """اجرای دوباره روی خبر یکسان نباید سطر جدید بسازد (news_id پایدار است)."""
    from src.features.news_sentiment import _news_id

    first = _news_id("فولاد", "تیتر", "2025-01-01", "http://x")
    second = _news_id("فولاد", "تیتر", "2025-01-01", "http://x")
    assert first == second and len(first) == 32
