DANA_REPORT
===========
DATA_QUALITY_AUDIT
=================
raw_shape: 1,067,371 x 8
clean_shape: 1,021,426 x 11
target_column: unknown
missing_before: 247,389
missing_after: 234,594
rows_removed_duplicates: 45,945
is_outlier_count: 139,581
missing_handling: preserve raw semantics; normalize blanks to NA; limited imputation only where safe
outlier_strategy: robust z-score on numeric columns + conservative flagging
train_only_safeguards: none applied at cleaning stage
bias_impact: duplicate rows removed only; no target-aware transforms applied
downstream_warnings: cancellations/returns are preserved and flagged; confirm target leakage before modeling
DATASET_RISK_REGISTER
Source credibility: Medium
License/usage: unknown from input file
Business fit: High for retail behavior analysis
Target suitability: ambiguous
Recency/deployment fit: depends on source workbook
Leakage risks: target and date-derived fields must be reviewed downstream
Bias/coverage risks: anonymous customers / cancellation rows / missing descriptions
Data dictionary: partial from workbook headers
Verdict: Use with caveats
