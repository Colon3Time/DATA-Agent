DANA_REPORT
===========
DATA_QUALITY_AUDIT
=================
raw_shape: 105 x 4
clean_shape: 105 x 7
target_column: Value
missing_before: 0
missing_after: 0
rows_removed_duplicates: 0
is_outlier_count: 0
role_counts: id=0, date=1, label=1, numeric=0, categorical=2
label_column: Value
id_columns: none
date_columns: Year
missing_handling: preserve raw semantics; normalize blanks to NA; limited imputation only where safe
outlier_strategy: robust z-score on numeric columns + conservative flagging
train_only_safeguards: none applied at cleaning stage
bias_impact: duplicate rows removed only; no target-aware transforms applied
feature_usage_guard: id-like columns should be excluded from numeric model features; label should be excluded from feature calculations when supervision is present
downstream_warnings: cancellations/returns are preserved and flagged; confirm target leakage before modeling
DATASET_RISK_REGISTER
Source credibility: Medium
License/usage: unknown from input file
Business fit: High for retail behavior analysis
Target suitability: Value
Recency/deployment fit: depends on source workbook
Leakage risks: target and date-derived fields must be reviewed downstream
Bias/coverage risks: anonymous customers / cancellation rows / missing descriptions
Data dictionary: partial from workbook headers
Verdict: Use with caveats
