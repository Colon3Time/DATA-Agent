Mo Model Report — Phase 2: Hyperparameter Tuning
==================================================
Input file: C:\Users\Amorntep\DAta-agent\projects\2026-04-28_uci_bank_marketing_blind\output\finn\engineered_data.csv
Phase: 2 (Tune - RandomizedSearchCV on RandomForest)
Search iterations: 30

Baseline Results (RandomForest default):
- CV F1: 0.8661 ± 0.0016
- Test F1: 0.8706
- Test AUC: 0.7902668610547667
- Train F1: 0.8754

Tuned Results (RandomForest best):
- Best CV F1: 0.8672
- Test F1: 0.8702
- Test AUC: 0.7904092616396999
- Train F1: 0.8755
- Improvement over baseline: -0.04%

Best Hyperparameters:
{
  "n_estimators": 200,
  "min_samples_split": 10,
  "min_samples_leaf": 2,
  "max_features": "sqrt",
  "max_depth": null,
  "class_weight": null,
  "bootstrap": true
}

Overfitting Check:
- Train F1: 0.8755
- Test F1: 0.8702
- Gap: 0.0053
- Status: OK: Gap (0.0053) - acceptable

ALGORITHM_RATIONALE
===================
Best Algorithm: RandomForest (Tuned)
Why This Algorithm:
  - ข้อมูล: tabular with mixed numerical/categorical features, banking data
  - Theory: ensemble of decision trees - handles non-linearity, feature interactions
  - vs others: robust to outliers, provides feature importance

NEXT_STEP: DONE
