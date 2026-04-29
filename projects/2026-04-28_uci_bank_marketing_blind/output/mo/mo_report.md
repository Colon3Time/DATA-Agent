Mo Model Report - Phase 2: Tune
==================================================
Problem Type: Classification
Phase: 2 (Tune - Random Forest hyperparameter search)
Date: 2026-04-29
Input: C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_uci_bank_marketing_blind\output\finn\engineered_data.csv
Target: y
Rows: 41,188
Features: 1

Tuning Setup:
- Algorithm: Random Forest
- Search: RandomizedSearchCV
- Combinations: 30
- CV: 5-fold StratifiedKFold
- Scoring: f1_weighted

Tuned vs Default:
| Algorithm | CV Mean | CV Std | Train F1 | Test F1 | Gap | Test AUC | Time(s) |
|-----------|---------|--------|----------|---------|-----|----------|---------|
| Random Forest baseline | 0.8661 | 0.0016 | 0.8754 | 0.8706 | 0.0048 | 0.7903 | 3.93 |
| Random Forest tuned | 0.8669 | 0.0018 | 0.8755 | 0.8703 | 0.0052 | 0.7901 | 99.82 |

Winner: Random Forest baseline
Test F1 Improvement: -0.0003 (-0.03%)

Best Params:
```json
{
  "bootstrap": true,
  "class_weight": null,
  "max_depth": null,
  "max_features": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "n_estimators": 300
}
```

Top Feature Importance:
| Feature | Importance |
|---------|------------|
| euribor3m | 1.0000 |

VALIDATION
==============================
Overfitting Check: train_test_gap=0.0048
Leakage Check: no forbidden ID/target-derived columns detected in engineered data
LOOP_BACK_TO_FINN: NO
DL_ESCALATE: NO
Reason: Random Forest remains stable and classical ML is sufficient for this tabular dataset.

Business Recommendation:
Use the tuned model only if the marginal improvement justifies added complexity.
If improvement is below 1%, keep the baseline Random Forest for simpler operations.
