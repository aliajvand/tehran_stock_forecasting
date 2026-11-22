"""
بک‌اند FastAPI سیستم پیش‌بینی بورس.

اجرا:
    uvicorn api.main:app --reload --port 8000

مسیرها:
    GET /health              بررسی سلامت سرویس و دیتابیس
    GET /predict/{symbol}    پیش‌بینی یک نماد
    GET /signals             سیگنال‌های امروز
    GET /graph/money-flow    گراف جریان پول
    GET /market/latest       آخرین قیمت‌ها
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from loguru import logger

from src import db as dbm
from src.config import get_db_path, get_section, load_config, setup_logging

setup_logging()
_project = get_section("project", {"name": "Bourse Predictor", "version": "0.1.0"})

app = FastAPI(
    title=str(_project.get("name", "Bourse Predictor")),
    version=str(_project.get("version", "0.1.0")),
    description="API پیش‌بینی روند سهام بورس تهران (LSTM + GNN + تحلیل احساسات)",
)

from api.routes.market import router as market_router  # noqa: E402
from api.routes.predictions import router as predictions_router  # noqa: E402

app.include_router(predictions_router)
app.include_router(market_router)


@app.get("/health", tags=["system"])
def health() -> Dict[str, Any]:
    """بررسی سلامت سرویس: اتصال دیتابیس و تعداد سطرهای جدول‌های کلیدی."""
    status: Dict[str, Any] = {
        "status": "ok",
        "project": _project.get("name"),
        "version": _project.get("version"),
        "database": str(get_db_path()),
    }
    try:
        conn = dbm.connect()
        try:
            status["tables"] = {
                table: dbm.table_count(conn, table)
                for table in ("instruments", "daily_history", "client_type", "codal_news")
            }
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("بررسی سلامت دیتابیس ناموفق بود: {}", exc)
        status["status"] = "degraded"
        status["error"] = str(exc)
    return status


@app.get("/", tags=["system"])
def root() -> Dict[str, Any]:
    """اطلاعات پایه سرویس."""
    return {"name": _project.get("name"), "docs": "/docs", "health": "/health"}


def main() -> int:
    """اجرای سرور با uvicorn از روی تنظیمات config."""
    import uvicorn

    api_cfg = load_config().get("api", {}) or {}
    uvicorn.run(
        "api.main:app",
        host=str(api_cfg.get("host", "127.0.0.1")),
        port=int(api_cfg.get("port", 8000)),
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
