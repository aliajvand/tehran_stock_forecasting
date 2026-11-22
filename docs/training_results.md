# Training Results — LSTM & Hybrid (5-Fold Walk-Forward)

**Date**: 2026-09-14  
**Dataset**: 18 symbols, 10,568 clean history rows → 9,439 sequences (60×80)  
**Relevant period**: ~3 years (2023-09 → 2026-09)

## Per-Fold Metrics

| Model | Fold | Accuracy | Dir. Acc | F1-macro | AUC-ROC |
|-------|------|----------|-------------|----------|---------|
| LSTM  | 1    | 0.481    | 0.017       | 0.237    | 0.526   |
| LSTM  | 2    | 0.415    | 0.345       | 0.415    | 0.635   |
| LSTM  | 3    | 0.393    | 0.355       | 0.393    | 0.587   |
| LSTM  | 4    | 0.490    | 0.621       | 0.379    | 0.620   |
| LSTM  | 5    | 0.596    | 0.678       | 0.465    | 0.697   |
| **LSTM mean** | — | **0.475** | **0.403** | **0.378** | **0.613** |
| Hybrid | 1    | 0.458    | 0.130       | 0.312    | 0.577   |
| Hybrid | 2    | 0.315    | 0.121       | 0.246    | 0.572   |
| Hybrid | 3    | 0.363    | 0.290       | 0.336    | 0.564   |
| Hybrid | 4    | 0.499    | 0.627       | 0.416    | 0.624   |
| Hybrid | 5    | 0.619    | 0.699       | 0.497    | 0.702   |
| **Hybrid mean** | — | **0.451** | **0.373** | **0.361** | **0.608** |

## Training Configuration

| Param | Value |
|-------|-------|
| model.lstm | hidden 128, layers 2, dropout 0.3 |
| epochs | 100 (early stopping at patience 10) |
| batch_size | 64 |
| walk_forward_splits | 5 |
| lr / seed | 0.001 / 42 |
| device | auto (CPU) |

## Runtime

| Model | Wall time |
|-------|-----------|
| LSTM (5 folds) | 14 min |
| Hybrid (5 folds) | 6 min |

## Key Observations

1. **Second half of folds much stronger**: folds 4-5 reach 62-70% directional accuracy and AUC 0.62-0.70, vs ~12-35% for folds 1-3. Best models use the most training history.
2. **Fold 1 LSTM collapse**: best weight selected by CE val-loss performed near-random on direction (1.7%). Model selection by raw loss is not aligned with directional accuracy — consider early stopping on directional-accuracy or F1 instead.
3. **Class behavior**: model frequently predicts the majority `neutral` class in early folds (fold 1 confusion matrix: 742/1573 → neutral), inflating accuracy while suppressing directional accuracy.
4. **Hybrid GNN branch was not exercised** — the training loop passes only sequences to `model()`; graph tensors are never fed. `--with-graph` adds dead parameters. Wiring `graph_builder` adjacency into training is a future enhancement.

## Artifacts

- Weights: `models/saved/lstm_fold1..5.pt`, `models/saved/hybrid_fold1..5.pt`
- Eval reports: `logs/evaluation/{lstm,hybrid}_fold*_20260914_16*.json`
- Run logs: `models/saved/train_{lstm,hybrid}_20260914_16*.json`