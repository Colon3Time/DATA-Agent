DANA_REPORT
===========
DATA_QUALITY_AUDIT
=================
raw_shape: 259,546 x 11
clean_shape: 259,546 x 12
target_column: Value
missing_before: 148
missing_after: 148
rows_removed_duplicates: 0
is_outlier_count: 0
role_counts: id=0, date=2, label=1, numeric=0, categorical=8
label_column: Value
id_columns: none
date_columns: Year, Source_Year
missing_handling: preserve raw semantics; normalize blanks to NA; limited imputation only where safe
outlier_strategy: robust z-score on numeric columns + conservative flagging
train_only_safeguards: none applied at cleaning stage
bias_impact: duplicate rows removed only; no target-aware transforms applied
feature_usage_guard: id-like columns should be excluded from numeric model features; label should be excluded from feature calculations when supervision is present
downstream_warnings: preserve flags; confirm target leakage before modeling
DATASET_RISK_REGISTER
Source credibility: Medium
License/usage: unknown from input file
Business fit: depends on domain — validate with downstream agents
Target suitability: Value
Recency/deployment fit: depends on source data
Leakage risks: target and date-derived fields must be reviewed downstream
Bias/coverage risks: check for duplicates, missing values, and outliers
Data dictionary: derived from input file headers
Verdict: Use with caveats
