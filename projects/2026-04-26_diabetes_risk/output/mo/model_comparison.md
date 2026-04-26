# Model Comparison Detail

Generated: 2026-04-26 11:34:38

## Data Information
- Total samples: 767
- Features: 8
- Target: Outcome
- Target distribution:
  - 0: 500 (65.2%)
  - 1: 267 (34.8%)

## Model Performance Summary

| Model | CV F1 (mean) | CV Std | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|-------------|--------|---------------|----------------|-------------|---------|----------|
| XGBoost | 0.744 | 0.0416 | 0.7208 | 0.7126 | 0.7208 | 0.7141 | 0.79 |
| KNN | 0.7372 | 0.0408 | 0.7143 | 0.7081 | 0.7143 | 0.71 | 0.7419 |
| LightGBM | 0.7445 | 0.0426 | 0.7143 | 0.7065 | 0.7143 | 0.7084 | 0.778 |
| SVM | 0.7775 | 0.0352 | 0.7143 | 0.7051 | 0.7143 | 0.7065 | 0.7613 |
| Random Forest | 0.7618 | 0.0369 | 0.7078 | 0.6989 | 0.7078 | 0.7008 | 0.7817 |
| Logistic Regression | 0.7821 | 0.0483 | 0.6818 | 0.6714 | 0.6818 | 0.6742 | 0.7883 |

## Best Model: Random Forest
- Test F1 Score: 0.7141
- CV F1 Score: 0.7770
- Best Parameters: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 10}
- Improvement from default: 1.90%

## Business Recommendation
- Recommended model: Random Forest
- Model is ready for deployment
- Expected F1 performance: 71.41%
- Easy to interpret

## Self-Improvement Report

Phase ที่ผ่าน: 1-2
Algorithm ที่ชนะ: Random Forest
Tuning improvement: 1.90%
วิธีใหม่ที่พบ: ไม่พบ
Knowledge Base: ไม่มีการเปลี่ยนแปลง