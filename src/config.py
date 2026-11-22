"""
بارگذاری تنظیمات پروژه از ``configs/*.yaml`` و راه‌اندازی لاگ.

هیچ مقداری در کد hard-code نمی‌شود؛ همه ماژول‌ها تنظیمات را از این ماژول می‌گیرند.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

# ریشه پروژه = پوشه‌ای که src در آن قرار دارد
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

_LOGGING_READY = False


def load_yaml(path: Path) -> Dict[str, Any]:
    """خواندن یک فایل YAML و برگرداندن dictionary (با مدیریت خطا)."""
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.error("فایل تنظیمات پیدا نشد: {}", path)
        raise
    except yaml.YAMLError as exc:
        logger.error("خطای نگارشی در فایل YAML {}: {}", path, exc)
        raise
    if not isinstance(data, dict):
        raise ValueError(f"ساختار {path} باید dictionary باشد")
    return data


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """تنظیمات اصلی پروژه (``configs/config.yaml``) با کش."""
    return load_yaml(CONFIG_DIR / "config.yaml")


@lru_cache(maxsize=1)
def load_symbols() -> Dict[str, Any]:
    """فایل نمادها (``configs/symbols.yaml``) با کش."""
    return load_yaml(CONFIG_DIR / "symbols.yaml")


def get_section(name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """گرفتن یک بخش از config (مثل ``data`` یا ``model``)."""
    value = load_config().get(name)
    return value if isinstance(value, dict) else (default or {})


def resolve_path(relative: str, create_parent: bool = False) -> Path:
    """تبدیل مسیر نسبی موجود در config به مسیر مطلق داخل پروژه."""
    path = Path(relative)
    full = path if path.is_absolute() else PROJECT_ROOT / path
    if create_parent:
        full.parent.mkdir(parents=True, exist_ok=True)
    return full


def get_path(key: str, default: str) -> Path:
    """گرفتن یکی از مسیرهای بخش ``paths`` در config."""
    return resolve_path(str(get_section("paths").get(key, default)))


def get_db_path() -> Path:
    """مسیر مطلق فایل دیتابیس SQLite."""
    return resolve_path(str(get_section("paths").get("db", "data/bourse.db")), create_parent=True)


def get_target_symbols(symbols_cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """لیست تخت و بدون تکرار همه نمادهای هدف از ``symbols.yaml``."""
    cfg = symbols_cfg or load_symbols()
    out: List[str] = []
    for names in (cfg.get("symbols") or {}).values():
        for name in names or []:
            if name:
                out.append(str(name).strip())
    return list(dict.fromkeys(out))


def get_symbol_sectors(symbols_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """نگاشت «نماد → گروه صنعتی» برای استفاده در ساخت گراف."""
    cfg = symbols_cfg or load_symbols()
    out: Dict[str, str] = {}
    for sector, names in (cfg.get("symbols") or {}).items():
        for name in names or []:
            if name:
                out[str(name).strip()] = str(sector)
    return out


def get_prediction_config(symbols_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """تنظیمات پیش‌بینی (افق، کلاس‌ها، آستانه درصدی)."""
    cfg = symbols_cfg or load_symbols()
    pred = cfg.get("prediction") or {}
    pred.setdefault("horizon_days", 1)
    pred.setdefault("classes", ["up", "down", "neutral"])
    pred.setdefault("threshold_pct", 1.0)
    return pred


def setup_logging(force: bool = False, log_file: Optional[str] = None) -> None:
    """
    راه‌اندازی loguru روی کنسول + فایل (فقط یک‌بار در طول اجرا).

    پارامترها:
        force: اگر True باشد، تنظیمات قبلی بازنویسی می‌شود.
        log_file: مسیر دلخواه فایل لاگ (پیش‌فرض از config).
    """
    global _LOGGING_READY
    if _LOGGING_READY and not force:
        return

    log_cfg = get_section("logging", {"level": "INFO", "file": "logs/app.log"})
    level = str(log_cfg.get("level", "INFO")).upper()
    target = resolve_path(str(log_file or log_cfg.get("file", "logs/app.log")), create_parent=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    try:
        logger.add(
            target,
            level=level,
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
            enqueue=True,
        )
    except OSError as exc:  # pragma: no cover - مثلاً مسیر بدون دسترسی نوشتن
        logger.warning("نوشتن فایل لاگ ممکن نشد ({}): {}", target, exc)
    _LOGGING_READY = True
