FINN_REPORT
===========

FEATURE_GOVERNANCE
=================
feature_lineage: derived from eddie_output.csv
prediction_time_availability: customer-level RFM features only
leakage_controls: drop raw ids/date/label-like fields before modeling
train_only_transforms: aggregation only; no target-aware transforms
temporal/OOT support columns: recency_days is computed relative to latest observed date
actionability: hand off customer-level features to Mo
warnings: confirm target ownership from Scout and use column_roles.json when available
target column : unknown
selected features : recency_days, frequency, monetary, row_count, avg_revenue, return_rate, outlier_rate
engineered_columns: 9