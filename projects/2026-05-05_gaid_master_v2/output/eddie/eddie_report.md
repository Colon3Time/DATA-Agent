EDDIE_REPORT
============

PIPELINE_SPEC
=============
problem_type: regression
target_column: Value
recommended_model: baseline + tree model comparison
preprocessing: preserve row-level transaction columns, normalize customer id aliases, carry outlier flags forward
key_features: Value, Source_Year, Year

BUSINESS_EDA_FRAME
==================
business question: what behavior patterns in scout_output.csv should be preserved for modeling and business follow-up?
decision owner: analytics / modeling team
target kpi: Value
strongest evidence: quick scan of the cleaned table with 11 columns
causality status: correlational only
temporal/leakage risk: date and customer identifiers must be handled carefully downstream
imbalance/skew risk: verify class balance or heavy-tailed drivers before modeling
validation strategy: use Finn + Mo validation after this bridge report

BUSINESS_EDA_BRIEF
==================
Insight: cleaned transaction data still preserves the main transaction structure from scout_output.csv.
Evidence: 259,546 rows and 11 columns after cleaning.
Business hypothesis: revenue/retention signals can be derived from transaction and customer history.
Follow-up question: which fields should be excluded or encoded before Finn builds features?
Next handoff: iris_eda / Finn
Risk / caveat: exploratory framing only; do not replace Scout target ownership.
Confidence: Medium