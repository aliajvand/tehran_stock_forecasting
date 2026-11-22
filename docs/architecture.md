# معماری سیستم

## نمای کلی

```
TSETMC (algotik-tse)        کدال (HTTP/codalpy)
        │                          │
        ▼                          ▼
 data_ingestion/            features/news_sentiment.py
 ├─ tsetmc_client.py  ◄── تنها نقطه تماس با کتابخانه
 ├─ backfill.py       ── ۳ سال تاریخچه  ─┐
 └─ realtime_fetcher.py ─ snapshot ۱۰ثانیه│
                                          ▼
                            SQLite: data/bourse.db
                     (instruments, daily_history, client_type,
                      order_book, market_snapshot, codal_news, ...)
                                          │
                            preprocessing/cleaner.py
                                          │
        ┌─────────────────────────────────┼─────────────────────────────┐
        ▼                 ▼               ▼               ▼             ▼
   technical.py      seasonal.py    money_flow.py   order_book.py  news_sentiment.py
        └─────────────────┴───────────────┴───────────────┴─────────────┘
                                          │
                             training/dataset.py  ──►  graph_builder.py
                        (فیچر + برچسب + دنباله)        (گراف پویا → PyG)
                                          │
                                models/hybrid_model.py
                        LSTM branch ──┐            ┌── GNN branch
                                      └── fusion ──┘
                                          │
                       ┌──────────────────┼──────────────────┐
                       ▼                  ▼                  ▼
              training/train.py   backtesting/simulator   realtime/pipeline.py
              (Walk-Forward)      (کارمزد/اسلیپیج)        (asyncio loop)
                                          │
                                 ┌────────┴────────┐
                                 ▼                 ▼
                           api/main.py       dashboard/app.py
                          (FastAPI)          (Streamlit)
```

## لایه‌ها

### ۱) دریافت داده (`src/data_ingestion/`)
- `tsetmc_client.py` — تنها جایی که `algotik_tse` صدا زده می‌شود. retry با تأخیر
  فزاینده، `time.sleep` برای rate limit، و **نگاشت ستون‌ها** به اسکیمای دیتابیس.
  ماژول کتابخانه قابل تزریق است (`module=`) تا تست با stub انجام شود.
- `backfill.py` — `BackfillRunner` سه مرحله را جدا اجرا می‌کند؛ همه درج‌ها
  `ON CONFLICT DO UPDATE` هستند پس اجرای مکرر بی‌خطر است.
- `realtime_fetcher.py` — حلقه snapshot هر ۱۰ ثانیه با خاتمه تمیز.

### ۲) پاک‌سازی (`src/preprocessing/cleaner.py`)
حذف تکراری، پر کردن شکاف تا ۵ روز، حذف روزهای بی‌حجم، تعدیل قیمت، حذف پرت با IQR.

### ۳) مهندسی ویژگی (`src/features/`)
| ماژول | خروجی |
|---|---|
| `technical.py` | RSI، MACD، بولینگر، SMA/EMA، ATR، استوکاستیک، بازدهی و نوسان |
| `seasonal.py` | ماه/فصل شمسی، هفته ماه، روز هفته، رمضان/محرم، تجزیه فصلی، SARIMA |
| `money_flow.py` | خالص ورود پول حقیقی/حقوقی، عدم توازن، قدرت خریدار |
| `order_book.py` | فشار خرید، عدم توازن سفارش، اسپرد |
| `news_sentiment.py` | امتیاز احساسات هر خبر + میانگین روزانه با میرایی نمایی |
| `graph_builder.py` | گراف روزانه `networkx` → `torch_geometric.data.Data` |

### ۴) مدل‌سازی (`src/models/`)
- `lstm_model.py` — LSTM دولایه، دو سر خروجی: کلاس + بازدهی تخمینی.
- `gnn_model.py` — GCN/GAT دولایه روی گراف جریان پول.
- `hybrid_model.py` — concat نمایش LSTM و GNN → MLP → ۳ کلاس + بازدهی.
- `ensemble.py` — رأی‌گیری نرم یا stacking بین هیبرید، XGBoost و Prophet.

### ۵) آموزش و ارزیابی (`src/training/`)
- `dataset.py` — ساخت جدول فیچر، برچسب‌گذاری سه‌کلاسه، دنباله‌سازی، بازه‌های
  Walk-Forward و استانداردسازی با آماره‌های **فقط مجموعه آموزش**.
- `train.py` — Walk-Forward + Early Stopping + ذخیره وزن هر بازه + لاگ JSON
  به سبک MLflow در `models/saved/runs/`.
- `evaluate.py` — Directional Accuracy (معیار اصلی)، F1، AUC-ROC، ماتریس
  درهم‌ریختگی؛ گزارش در `logs/evaluation/`.

### ۶) بک‌تست و لحظه‌ای
- `backtesting/simulator.py` — سیگنال‌محور با کارمزد و اسلیپیج، خروجی HTML.
- `realtime/pipeline.py` — حلقه `asyncio`: دریافت → فیچر → پیش‌بینی → ذخیره،
  به‌همراه کلاس `Predictor` که API و داشبورد هم از آن استفاده می‌کنند.

### ۷) ارائه
- `api/main.py` + `api/routes/` — FastAPI: `/health`, `/predict/{symbol}`,
  `/signals`, `/graph/money-flow`, `/market/latest`.
- `dashboard/app.py` — Streamlit با ۴ صفحه (بازار، پیش‌بینی، گراف، اخبار).

## ماژول‌های مشترک
- `src/config.py` — بارگذاری `configs/*.yaml` با کش + راه‌اندازی loguru + مسیرها.
- `src/db.py` — اتصال SQLite، `upsert` عمومی، خواندن جدول‌ها، تبدیل تاریخ
  `YYYYMMDD` ⇄ `datetime`. **اسکیما را نمی‌سازد**؛ فقط `src/database/schema.py`
  مسئول ساخت جدول‌هاست.

## قرارداد داده (مهم)
- `gregorian_date` در دیتابیس **عدد صحیح `YYYYMMDD`** است.
- `ins_code` عدد صحیح و برگرفته از `instrument_id` کتابخانه است.
- نگاشت قیمت: `Close → close` (پایانی) و `Final → last` (آخرین)، `No. → trade_count`.
- کلید یکتا: `instruments(ins_code)`، `daily_history(ins_code, gregorian_date)`،
  `client_type(ins_code, gregorian_date)`، `codal_news(news_id)`.
