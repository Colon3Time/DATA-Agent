Iris EDA Bridge Report
=======================

BUSINESS_EDA_BRIEF
==================
Insight: Highest numeric missingness: collateral_value_pct_loan (54.5%).
Evidence: quick bridge scan from iris_eda_output.csv and column profiling.
Business hypothesis: the strongest operational leverage is likely in the most incomplete or most skewed drivers.
Follow-up question: which of the flagged columns should Finn keep, derive, or exclude before modeling?
Next handoff: Finn
Risk / caveat: this is exploratory and should not overwrite Scout target ownership.
Confidence: Medium

Observations
------------
Rows: 770
Columns: 82
Columns with the most missingness start with collateral_value_pct_loan; this may be the best follow-up area for Finn/Mo.

Top findings:
- Highest numeric missingness: collateral_value_pct_loan (54.5%).
- Strong numeric relationship detected between foreign_ownership_pct and private_domestic_ownership_pct.