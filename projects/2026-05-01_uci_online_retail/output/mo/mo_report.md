# Mo Model Report

Input: `projects\2026-05-01_uci_online_retail\output\finn\engineered_data.csv`
Manifest: `projects\2026-05-01_uci_online_retail\output\finn\finn_feature_manifest.json`
Rows: 5,878

## Random Split Validation

### Target `monetary`
- Task: regression
- Features used: 5
- Excluded: Customer ID, avg_order_value, avg_unit_price, clv_proxy, first_purchase, grain, is_churned_180d, is_high_value, last_purchase, monetary, recency_days, total_quantity

| Model | CV | Test Metric | Secondary |
|---|---:|---:|---|
| Ridge | 0.2900 | 0.4034 | RMSE=14804.48, MAE=2036.50 |
| RandomForest | 0.2112 | 0.5575 | RMSE=12749.77, MAE=1750.82 |

### Target `is_churned_180d`
- Task: classification
- Features used: 8
- Excluded: Customer ID, clv_proxy, first_purchase, grain, is_churned_180d, is_high_value, last_purchase, monetary, recency_days

| Model | CV | Test Metric | Secondary |
|---|---:|---:|---|
| LogisticRegression | 0.6917 | 0.6810 | ROC-AUC=0.8042, PR-AUC=0.6874 |
| RandomForest | 0.7109 | 0.6974 | ROC-AUC=0.8154, PR-AUC=0.6893 |
| GradientBoosting | 0.7071 | 0.7003 | ROC-AUC=0.8177, PR-AUC=0.6954 |

### Target `is_high_value`
- Task: classification
- Features used: 8
- Excluded: Customer ID, avg_order_value, clv_proxy, first_purchase, grain, is_churned_180d, is_high_value, last_purchase, monetary

| Model | CV | Test Metric | Secondary |
|---|---:|---:|---|
| LogisticRegression | 0.8425 | 0.8489 | ROC-AUC=0.9852, PR-AUC=0.9490 |
| RandomForest | 0.8996 | 0.9075 | ROC-AUC=0.9922, PR-AUC=0.9678 |
| GradientBoosting | 0.9013 | 0.9045 | ROC-AUC=0.9925, PR-AUC=0.9731 |

## OOT Validation
- `is_churned_90d` ROC-AUC=0.7447, PR-AUC=0.7845, F1=0.7861
