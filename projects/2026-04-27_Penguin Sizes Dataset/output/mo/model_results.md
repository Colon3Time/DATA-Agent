# Mo Model Report — Phase 1: Explore

## Overview
- **Project:** Palmer Penguins Species Classification
- **Phase:** 1 (Explore — all algorithms, default params)
- **Target:** species
- **Classes:** ['Adelie', 'Chinstrap', 'Gentoo']
- **Train/Test:** 275/69

## Algorithm Comparison (5-fold CV)

| Algorithm | CV Score (mean) | CV Std | Test Accuracy | Test F1 | Precision | Recall | Time (s) |
|-----------|----------------|--------|--------------|---------|-----------|--------|----------|
| Logistic Regression | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.05 |
| Random Forest | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.75 |
| KNN | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.63 |
| XGBoost | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.35 |
| LightGBM | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.21 |
| SVM | 0.9964 | 0.0073 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.05 |

## Winner
**Logistic Regression** — CV: 1.0000, Test F1: 1.0000

## PREPROCESSING_REQUIREMENT
- **Algorithm Selected:** Logistic Regression
- **Scaling:** StandardScaler applied ✓
- **Encoding:** LabelEncoder for target ✓
- **Loop Back To Finn:** NO
- **Reason:** Finn already provided clean engineered data with proper encoding. No additional preprocessing needed.

## Business Recommendation
The Logistic Regression model shows strong performance with F1 score of 1.0000. This model can reliably classify penguin species based on the available features.
