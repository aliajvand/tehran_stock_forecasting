"""
تست واقعی: Backfill ۲ نماد × ۱۰ روز از TSETMC واقعی
هدف: ببینیم آیا کد Claude با دیتای واقعی کار می‌کنه یا نه
"""
import sys
from pathlib import Path
from datetime import datetime
import jdatetime

# اضافه کردن مسیر پروژه
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
import sqlite3


def create_test_db():
    """ساخت دیتابیس تستی"""
    test_db = Path("data/test_bourse.db")
    test_db.parent.mkdir(parents=True, exist_ok=True)
    
    # اسکیمای مشابه اصلی
    schema_sql = """
    CREATE TABLE IF NOT EXISTS instruments (
        ins_code INTEGER PRIMARY KEY,
        symbol TEXT NOT NULL,
        name TEXT,
        english_name TEXT,
        company_code TEXT,
        company_isin TEXT,
        instrument_isin TEXT,
        asset_type TEXT,
        market TEXT,
        sector_code TEXT,
        industry_group TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS daily_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ins_code INTEGER NOT NULL,
        gregorian_date INTEGER NOT NULL,
        jalali_date TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, last REAL,
        volume INTEGER, value REAL, trade_count INTEGER,
        UNIQUE(ins_code, gregorian_date)
    );
    
    CREATE TABLE IF NOT EXISTS client_type (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ins_code INTEGER NOT NULL,
        gregorian_date INTEGER NOT NULL,
        jalali_date TEXT NOT NULL,
        n_buy_retail INTEGER, n_buy_institutional INTEGER,
        n_sell_retail INTEGER, n_sell_institutional INTEGER,
        vol_buy_retail INTEGER, vol_buy_institutional INTEGER,
        vol_sell_retail INTEGER, vol_sell_institutional INTEGER,
        val_buy_retail REAL, val_buy_institutional REAL,
        val_sell_retail REAL, val_sell_institutional REAL,
        per_capita_buy_retail REAL, per_capita_sell_retail REAL,
        per_capita_buy_institutional REAL, per_capita_sell_institutional REAL,
        power_retail REAL, power_institutional REAL,
        UNIQUE(ins_code, gregorian_date)
    );
    """
    
    conn = sqlite3.connect(test_db)
    for stmt in schema_sql.split(";"):
        if stmt.strip():
            conn.execute(stmt)
    conn.commit()
    return conn


def test_real_fetch():
    """تست واقعی دریافت دیتا"""
    logger.info("=" * 60)
    logger.info("🧪 تست واقعی TSETMC — ۲ نماد، ۱۰ روز")
    logger.info("=" * 60)
    
    # ۱. اول ببینیم algotik_tse چطور import میشه
    try:
        import algotik_tse as att
        logger.success(f"✅ algotik_tse import شد")
    except ImportError as e:
        logger.error(f"❌ algotik_tse نصب نیست: {e}")
        logger.info("💡 pip install algotik-tse")
        return
    
    # ۲. تست get_symbols با دیتای واقعی
    logger.info("\n📋 تست get_symbols...")
    try:
        symbols_df = att.get_symbols(bourse=True, farabourse=False, payeh=False,
                                      haghe_taqadom=False, sandogh=False, bonds=False,
                                      options=False, mortgage=False, commodity=False,
                                      energy=False)
        logger.success(f"✅ {len(symbols_df)} نماد دریافت شد")
        logger.info(f"   ستون‌ها: {symbols_df.columns.tolist()}")
        
        # چک کردن اینکه نمادهای هدف ما هستن
        target = ["فولاد", "شپنا"]
        available = symbols_df.index.tolist()
        
        for sym in target:
            if sym in available:
                row = symbols_df.loc[sym]
                logger.success(f"   ✅ {sym} پیدا شد | ins_code={row['instrument_id']} | نوع={row['asset_type']}")
            else:
                logger.warning(f"   ⚠️ {sym} توی لیست نیست!")
        
    except Exception as e:
        logger.error(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ۳. تست get_history برای یک نماد
    logger.info("\n📈 تست get_history برای فولاد...")
    try:
        df = att.get_history(
            symbol="فولاد",
            start="1404-06-01",
            end="1404-06-15",
            output_type="full",
            date_format="both",
            auto_adjust=True,
            progress=False
        )
        
        logger.success(f"✅ {len(df)} ردیف دریافت شد")
        logger.info(f"   ایندکس: {type(df.index).__name__}")
        logger.info(f"   ستون‌ها: {df.columns.tolist()}")
        logger.info(f"   نمونه:\n{df.head(3)}")
        
    except Exception as e:
        logger.error(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()
    
    # ۴. تست get_client_type
    logger.info("\n👥 تست get_client_type برای فولاد...")
    try:
        df = att.get_client_type(
            symbol="فولاد",
            start="1404-06-01",
            end="1404-06-15",
            output_type="full",
            date_format="both",
            progress=False
        )
        
        logger.success(f"✅ {len(df)} ردیف دریافت شد")
        logger.info(f"   ستون‌ها: {df.columns.tolist()}")
        
    except Exception as e:
        logger.error(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_real_fetch()