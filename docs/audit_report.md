# Audit Report — Tehran Stock Exchange Predictor

**Date**: 2026-09-14  
**Auditor**: Automated pipeline audit + notebook verification

---

## Executive Summary

7 bugs fixed (5 original + 2 new). Full pipeline test notebook passes all 13 cells on a live execution. All 62 existing unit tests pass. Full 5-fold production training completed for LSTM and Hybrid on 18 symbols (9,439 sequences).

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.13 |
| algotik-tse | 1.8.0 |
| PyTorch | 2.14.0+cpu |
| pandas | 2.3.3 |
| numpy | 2.3.5 |

---

## Original Bugs (1–5): Status

| # | Description | Status |
|---|------------|--------|
| 1 | Comment handling in `schema.py` `create_database` (`--` chunks from `;` split) | ✅ Fixed in current code (line filtering at `schema.py:260-264`) |
| 2 | UNIQUE constraints on `market_snapshot` and `order_book` | ✅ Fixed (`schema.py:114`, `schema.py:167`) |
| 3 | InsCode string-to-numeric in `realtime_collector.py` `save_snapshot` | ✅ Fixed (`realtime_collector.py:132` — `pd.to_numeric(..., errors="coerce")`) |
| 4 | `--force` patch for `is_market_open` using `global` | ✅ Fixed (`realtime_collector.py:287-289` — correct module-level `global` override) |
| 5 | Symbol `سرمایه` not found in TSETMC | ✅ Fixed — replaced with `شستا` (سرمایه‌گذاری تامین اجتماعی) in `configs/symbols.yaml:35` |

---

## New Bugs Found and Fixed

### Bug 6 — Order Book Feature Pipeline Column Mismatch (LATENT)

**File**: `src/features/order_book.py` — `load_daily_order_book_features()`

**Root cause**: DB `order_book` table uses columns `bid_price`, `bid_volume`, `ask_price`, `ask_volume`, `snapshot_time`, but the feature function expected `buy_price`, `buy_volume`, `sell_price`, `sell_volume`, `gregorian_date`. Additionally, it attempted to group by `gregorian_date` which doesn't exist in the raw query output.

**Impact**: Silent failure caught by `try/except` in `dataset.py:load_feature_table` — order book features always silently skipped once data exists in the table.

**Fix** (`order_book.py:62-84`):
- Renames `bid/ask → buy/sell` before passing to `compute_order_book_features`
- Derives `gregorian_date` (YYYYMMDD int) from `snapshot_time` (epoch seconds)
- Drops rows with missing date

**Status**: ✅ Fixed, backed up as `order_book.py.bak`, tested with synthetic DB insertion

---

### Bug 7 — Non-finite Values (Inf) in Feature Matrix Causing NaN Training

**File**: `src/features/technical.py`, `src/training/dataset.py`

**Root cause**: `pct_change()` on close price produces `Inf` when a prior close is exactly `0.0` (a rare data artifact from price adjustments). These inf values survive `fillna(0.0)` (which only targets NaN) and corrupt `standardize()`, producing NaN in model inputs from fold 2 onward in walk-forward training.

**Evidence**: 2 zero-close rows → 52 inf cells in feature table → 510 inf cells in sequence matrix → `val_loss=nan` for all folds past fold 1.

**Fixes applied** (3 layers):
1. `technical.py:113-115` — `pct_change().replace([np.inf, -np.inf], np.nan)` on return_1d/5d/20d
2. `dataset.py:165-171` — `build_sequences` matrix pipeline: added `.replace([np.inf, -np.inf], np.nan)` before ffill/fillna  
3. `dataset.py:119` — `add_labels`: clamp `future_return` inf → NaN before label assignment

**Status**: ✅ Fixed, backed up as `technical.py.bak` and `dataset.py.bak`, verified zero inf/nan in X matrix, verified finite standardized tensors

---

### Bug 8 — Zero/Non-positive Close Prices (Hardening)

**File**: `src/preprocessing/cleaner.py` — `_clean_group()`

**Root cause**: Two rows in TSETMC history had `close = 0` (price-adjustment artifact). While Bug 7 sanitized `inf` downstream, filtering at the source is the correct defense so no anomalous price rows enter feature computation at all.

**Fix** (`cleaner.py:126-127`): Added `g = g[g["close"] > 0]` right after `dropna` on close, before volume/outlier/adjustment logic.

**Status**: ✅ Fixed, backed up as `cleaner.py.bak`, verified `0` zero-close rows and `0` non-finite values in the sequence matrix

---

## Full Pipeline Notebook Test

**File**: `notebooks/00_full_pipeline_test.ipynb`  
**Cells**: 1 markdown + 13 code cells  
**Result**: ✅ All 13 cells executed without error

| Cell | Description | Result |
|------|------------|--------|
| 1 | Setup & environment | algotik-tse 1.8.0, torch 2.14.0 |
| 2 | DB schema health | 11 tables present, 428 instruments |
| 3 | TSETMC client + symbol validation | All 18 targets available (including شستا) |
| 4 | Instruments sync | 428 instruments upserted |
| 5 | Backfill history + client_type | 1643 history + 1647 client rows (3 symbols, ~1 year) |
| 6 | Data QA | 0% NaN in core OHLCV columns |
| 7 | Feature table | 1591 rows × 85 columns (technical + seasonal + money_flow) |
| 8 | Labels | 3-class: down=29.5%, neutral=34.6%, up=35.9% |
| 9 | Sequences + splits | X=(1408,60,80), 3 folds |
| 10 | Train (LSTM, 5 epochs) | val_loss: 1.28→1.15→1.04, dir accuracy: 35.3% mean |
| 11 | Signal generation (WF manual) | 1056 signals across 3 folds |
| 12 | Backtest (Simulator) | Trades executed, metrics computed, HTML report saved |
| 13 | Realtime predict_latest | 3 predictions produced for latest available day |

---

## Smoke Test Model Metrics (LSTM, 5 epochs, 3 folds)

| Metric | Value |
|--------|-------|
| accuracy_mean | 0.38 |
| accuracy_std | 0.09 |
| directional_accuracy_mean | 0.35 |
| directional_accuracy_std | 0.22 |

> Low accuracy expected in smoke test (5 epochs, 3 symbols only). Target 70%+ requires full 100-epoch training on all symbols.

---

## Full Production Training (18 Symbols)

Run after backfilling all target symbols. See [training_results.md](training_results.md) for full per-fold table.

| Model | Accuracy | Dir. Acc | F1-macro | AUC-ROC | Runtime |
|-------|----------|----------|----------|---------|---------|
| LSTM (5-fold) | 0.475 | 0.403 | 0.378 | 0.613 | 14 min |
| Hybrid (5-fold) | 0.451 | 0.373 | 0.361 | 0.608 | 6 min |

- **Dataset**: 10,568 clean rows → 9,439 sequences (60×80)
- **Weights**: `models/saved/{lstm,hybrid}_fold1..5.pt`
- Folds 4-5 reach 62-70% directional accuracy & AUC 0.62-0.70; folds 1-3 weaker (early-stop plateau)
- GNN branch not wired into training loop (sequences only); `--with-graph` adds dead parameters — documented in training_results.md

---

## Database State (Post-Backfill)

| Table | Rows |
|-------|------|
| instruments | 428 |
| daily_history | 10,874 (18 symbols, ~600 rows each) |
| client_type | 10,876 (18 symbols) |
| market_snapshot | 0 |
| order_book | 0 |
| codal_news | 0 |

**Backups**:  
- `data/bourse.db.bak` (3.3 MB) — before any changes  
- `data/bourse.20260914_backfilled.db.bak` (3.5 MB) — after 18-symbol backfill

---

## Files Modified

| File | Change |
|------|--------|
| `configs/symbols.yaml` | Bug 5: `سرمایه` → `شستا` |
| `src/features/order_book.py` | Bug 6: column rename + gregorian_date derivation |
| `src/features/technical.py` | Bug 7: inf → NaN on return_* columns |
| `src/training/dataset.py` | Bug 7: inf sanitization in build_sequences + add_labels |
| `src/preprocessing/cleaner.py` | Bug 8: `close > 0` filter in _clean_group |

**Backup files created**:  
- `data/bourse.db.bak`  
- `data/bourse.20260914_backfilled.db.bak`  
- `src/features/order_book.py.bak`  
- `src/features/technical.py.bak`  
- `src/training/dataset.py.bak`  
- `src/preprocessing/cleaner.py.bak`

---

**Docs created**:  
- `docs/audit_report.md` (this file)  
- `docs/training_results.md` (per-fold metrics + analysis)  
- `notebooks/00_full_pipeline_test.ipynb` (13-cell pipeline verification)

---

## Algorithmic Column Mapping Audit

All mappings between `algotik-tse` output and DB/code verified live:

| Library Function | Key Columns | Verified |
|-----------------|-------------|----------|
| `get_symbols()` | index=persian name, `instrument_id`=InsCode | ✅ |
| `get_history()` | Open/High/Low/Close/Final/Volume/Value/No./J-Date | ✅ |
| `get_client_type()` | N_buy_retail, Vol_buy_retail, Val_buy_retail, Power_retail, Per_capita_* | ✅ |
| `get_market_snapshot()` stocks | InsCode(string→numeric), Last/Close/Volume/Value/TradeCount/Open/High/Low/ChangePct | ✅ |
| `get_market_snapshot()` order_book | InsCode/Level/BidPrice/AskPrice/BidVolume/AskVolume/BidOrderCount/AskOrderCount | ✅ |

---

## Warnings (Not Bugs)

| Warning | Module | Cause |
|---------|--------|-------|
| `news_sentiment` skipped | `dataset.py:80` | No data in `codal_news` table (network scraping not run) |
| `order_book` skipped | `dataset.py:90` | Empty `order_book` table (no market hours data collected) |
| GNN branch unused in training | `train.py` | `model(...)` receives sequences only; graph tensors never fed |
| Fold-1 LSTM directional collapse | `train.py` | Best-weight selection by CE loss, not directional accuracy |
| Old evaluation JSONs in `logs/evaluation/` | — | Generated from first run with NaN; harmless, can be deleted |

---

## Recommendations

1. **Model selection metric**: switch best-epoch selection in `_train_torch_fold` from raw `val_loss` to directional accuracy or F1 — fold 1 LSTM otherwise picks a near-random-directional model.
2. **Graph integration**: wire `graph_builder` adjacency + node features into the hybrid training loop (currently `--with-graph` only adds dead parameters).
3. **Order book features**: run `RealtimeCollector` during TSE hours (07:45–12:45 Teheran, Saturday–Wednesday) to populate `market_snapshot` and `order_book` tables.
4. **News sentiment**: run `python -m src.features.news_sentiment --pages 2` to populate `codal_news` for sentiment features.
