# سیستم پیش‌بینی هوشمند بورس تهران

پیش‌بینی روند روزانه سهام بورس تهران با **LSTM + GNN + تحلیل احساسات اخبار کدال**،
از دریافت داده تا API و داشبورد. هدف: **دقت جهت‌دهی بالای ۷۰٪**.

---

## ۱) نصب (Windows 10/11 + Anaconda Python 3.11)

```bat
cd C:\Users\programmer\Desktop\trade
conda activate base
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

> **torch / torch-geometric:** اگر نصب از `requirements.txt` خطا داد، اول torch را
> طبق دستور رسمی <https://pytorch.org/get-started/locally/> نصب کنید و بعد بقیه را.
> بدون torch هم فاز ۱ و ۲ (دریافت داده و فیچرسازی) کامل کار می‌کند.

---

## ۲) اجرا به ترتیب فازها

```bat
:: فاز ۱ — پر کردن دیتابیس از ۳ سال گذشته
python -m src.data_ingestion.backfill
python -m src.data_ingestion.backfill --only history --symbols فولاد فخوز
python -m src.data_ingestion.realtime_fetcher --once

:: فاز ۲ — اخبار کدال + احساسات (فیچرها خودکار در زمان آموزش ساخته می‌شوند)
python -m src.features.news_sentiment --pages 1

:: فاز ۳ — آموزش با Walk-Forward
python -m src.training.train --model lstm --epochs 30
python -m src.training.train --model hybrid --with-graph
python -m src.training.train --model xgboost

:: فاز ۴ — پیش‌بینی لحظه‌ای و API
python -m src.realtime.pipeline --once
uvicorn api.main:app --port 8000

:: فاز ۵ — داشبورد
streamlit run dashboard/app.py
```

بررسی سریع دیتابیس:

```bat
python -c "import sqlite3;c=sqlite3.connect('data/bourse.db');print([ (t,c.execute('select count(*) from '+t).fetchone()[0]) for t in ('instruments','daily_history','client_type')])"
```

---

## ۳) تست

```bat
pytest -q                      :: همه تست‌ها با داده جعلی (بدون شبکه)
pytest tests/test_backfill.py -v
```

تست‌ها از `tests/stubs.py` استفاده می‌کنند: یک `algotik_tse` جعلی با **دقیقاً**
ساختار ستون‌های نسخه واقعی. تست‌های وابسته به torch در نبود آن `skip` می‌شوند.

---

## ۴) ساختار پروژه

```
trade/
├── data/            bourse.db + raw/processed/features
├── src/
│   ├── config.py         بارگذاری YAML + لاگ + مسیرها
│   ├── db.py             اتصال، upsert، خواندن (اسکیما را نمی‌سازد)
│   ├── database/         schema.py ⚠️ فایل خودتان، دست‌نخورده
│   ├── data_ingestion/   tsetmc_client, backfill, realtime_fetcher
│   ├── preprocessing/    cleaner
│   ├── features/         technical, seasonal, money_flow, order_book,
│   │                     news_sentiment, graph_builder
│   ├── models/           lstm, gnn, hybrid, ensemble
│   ├── training/         dataset, train (Walk-Forward), evaluate
│   ├── backtesting/      simulator
│   └── realtime/         pipeline (asyncio) + کلاس Predictor
├── api/             FastAPI (main.py + routes/)
├── dashboard/       Streamlit (۴ صفحه)
├── configs/         config.yaml, symbols.yaml
├── docs/            architecture.md, scope.md
├── tests/           تست‌های stub همه ماژول‌ها
└── requirements.txt
```

---

## ۵) نکات حیاتی `algotik-tse` (نسخه فعلی)

`att.get_symbols(...)` → ایندکس = **نماد فارسی**، ستون‌ها:
`name, instrument_isin, english_name, company_code, company_isin, market,
industry_group, asset_type, instrument_id`

🔴 **ستون‌های `InsCode`، `Symbol` و `Name` وجود ندارند.** `instrument_id` همان
InsCode است و در `TsetmcClient` به `ins_code` نگاشت می‌شود.

### نگاشت ستون‌ها به دیتابیس

| algotik-tse | دیتابیس | توضیح |
|---|---|---|
| index (`Date`) | `gregorian_date` | عدد صحیح `YYYYMMDD` |
| `J-Date` | `jalali_date` | مثل `1403-02-19` |
| `Close` | `close` | قیمت پایانی |
| `Final` | `last` | آخرین قیمت |
| `No.` | `trade_count` | تعداد معاملات |
| `N_buy_retail` … | `n_buy_retail` … | همان نام با حروف کوچک |

`att.search_symbol()` در این نسخه **وجود ندارد** و استفاده نشده است.
خروجی توابع لحظه‌ای dict تودرتو است و هرگز مستقیم به `pd.DataFrame(...)` داده
نمی‌شود (تخت‌سازی در `TsetmcClient.get_market_snapshot`).

---

## ۶) تنظیمات

همه چیز از `configs/` خوانده می‌شود، هیچ مقدار hard-code شده‌ای در کد نیست:
- `symbols.yaml` — نمادها به تفکیک صنعت + افق و آستانه پیش‌بینی
- `config.yaml` — بازه داده، rate limit، فیچرها، هایپرپارامترهای مدل، بک‌تست، لاگ، مسیرها

---

## ۷) خروجی‌ها

| مسیر | محتوا |
|---|---|
| `data/bourse.db` | همه داده خام و پردازش‌شده |
| `models/saved/*.pt` | وزن بهترین مدل هر بازه Walk-Forward |
| `models/saved/runs/*.json` | لاگ اجرا به سبک MLflow (پارامتر + متریک هر بازه) |
| `logs/evaluation/*` | گزارش ارزیابی و گزارش HTML بک‌تست |
| `data/processed/latest_signals.csv` | سیگنال‌های آخرین اجرای خط لوله |
| `logs/app.log` | لاگ چرخشی loguru |

---

## ۸) قواعد کدنویسی رعایت‌شده
Type hints روی همه توابع · docstring فارسی · `loguru` به‌جای `print` ·
`try/except` روی هر I/O · `pathlib.Path` · تنظیمات از YAML · بدون `input()` ·
`time.sleep` بین درخواست‌ها · PEP 8.

## ۹) نقشه راه بعدی
تیون هایپرپارامتر با `optuna` · افزودن دیتای تیک برای گراف واقعی جریان پول ·
مدل چندافقی (۱/۲/۳ روز به‌صورت هم‌زمان) · اپلیکیشن موبایل روی همین API.
