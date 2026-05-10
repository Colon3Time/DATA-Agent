Iris EDA Bridge Report
=======================

BUSINESS_EDA_BRIEF
==================
Insight: Highest numeric missingness: Customer_ID (22.9%).
Evidence: quick bridge scan from iris_eda_output.csv and column profiling.
Business hypothesis: the strongest operational leverage is likely in the most incomplete or most skewed drivers.
Follow-up question: which of the flagged columns should Finn keep, derive, or exclude before modeling?
Next handoff: Finn
Risk / caveat: this is exploratory and should not overwrite Scout target ownership.
Confidence: Medium

Observations
------------
Rows: 1,021,426
Columns: 13
Columns with the most missingness start with Customer_ID; this may be the best follow-up area for Finn/Mo.

Top findings:
- Highest numeric missingness: Customer_ID (22.9%).
- Strong numeric relationship detected between Customer_ID and Customer ID.