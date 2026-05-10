FINN_REPORT
===========

FEATURE_GOVERNANCE
==================
feature_lineage: derived from eddie_output.csv
feature_mode: generic supervised feature table
prediction_time_availability: row-level columns only; target is retained only for Mo training
leakage_controls: high-cardinality id-like columns excluded; target excluded from feature list
train_only_transforms: encoding/imputation recipe is deterministic and does not use target statistics
target column : Value
problem_type : regression
excluded columns : none
selected features : num__Year, num__is_return, num__is_cancellation, num__is_outlier, num__revenue, cat__Country_Thailand, cat__Indicator_Export_Growth_pct, cat__Indicator_Foreign_Reserve_USD_billion, cat__Indicator_GDP_Growth_pct, cat__Indicator_GDP_USD_billion, cat__Indicator_Inflation_pct, cat__Indicator_Population_million, cat__Indicator_Unemployment_pct
engineered_columns: 14
handoff_status: ready for Mo