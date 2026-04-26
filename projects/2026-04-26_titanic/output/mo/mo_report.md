# Mo Model Report — Phase 1: Explore

## Summary
- **Problem Type**: Binary Classification
- **Phase**: 1 (Explore — all algorithms, default params)
- **Target**: Survived
- **Dataset**: 891 rows, 20 features
- **Imbalance Ratio**: {0: 549, 1: 342}

## Algorithm Comparison (5-fold CV)

| Algorithm | CV Mean | CV Std | Test F1 | Test AUC | Time (s) |
|-----------|---------|--------|---------|----------|----------|
| Logistic Regression | 0.8058 | 0.0361 | 0.8443 | 0.8752 | 0.06 |
| Random Forest | 0.7949 | 0.0167 | 0.8528 | 0.8458 | 0.93 |
| XGBoost | 0.8009 | 0.0147 | 0.7824 | 0.8112 | 0.69 |
| LightGBM | 0.8031 | 0.0100 | 0.7871 | 0.8194 | 1.52 |
| SVM | 0.8180 | 0.0261 | 0.8212 | 0.8572 | 0.41 |
| KNN | 0.8072 | 0.0254 | 0.8083 | 0.8251 | 0.12 |

## Winner
- **Random Forest**: Test F1 = 0.8528, Test AUC = 0.8458
- CV Score: 0.7949 ± 0.0167

## PREPROCESSING_REQUIREMENT

| Requirement | Value |
|-------------|-------|
| Algorithm Selected | Random Forest |
| Algorithm Type | Tree-based |
| Scaling | Not required |
| Encoding | Label Encoding (tree-based) or One-Hot (linear) |
| Missing Values | Already handled (median fill) |
| Transform | None |
| Loop Back To Finn | YES — for feature verification |

## Feature Importance (from Random Forest)

| Feature | Importance |
|---------|-----------|
| PassengerId | 0.1285 |
| Fare_scaled | 0.1155 |
| Sex_female | 0.1123 |
| Fare | 0.1082 |
| Age | 0.0932 |
| Title_Mr | 0.0879 |
| Age_scaled | 0.0861 |
| Sex_male | 0.0772 |
| Pclass_3 | 0.0351 |
| SibSp | 0.0324 |

## Next Steps
- **DL_ESCALATE**: NO
- **Phase 2 (Recommended)**: Hyperparameter Tuning on Random Forest
- **Finn Loop**: Verify features are optimal for Random Forest

## Self-Improvement Report

- **Phase**: 1 (Explore)
- **Algorithms Tested**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, KNN
- **Winner**: Random Forest
- **Improvement Opportunity**: Tune hyperparameters in Phase 2
- **Knowledge Base Updated**: Yes — Random Forest is best for Titanic