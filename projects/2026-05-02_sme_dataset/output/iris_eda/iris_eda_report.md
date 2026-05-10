Iris EDA Bridge Report
=======================

BUSINESS_EDA_BRIEF
==================
Insight: Highest numeric missingness: TSIC-2_dg (0.0%).
Evidence: quick bridge scan from iris_eda_output.csv and column profiling.
Business hypothesis: the strongest operational leverage is likely in the most incomplete or most skewed drivers.
Follow-up question: which of the flagged columns should Finn keep, derive, or exclude before modeling?
Next handoff: Finn
Risk / caveat: this is exploratory and should not overwrite Scout target ownership.
Confidence: Medium

Observations
------------
Rows: 8,316
Columns: 15
Columns with the most missingness start with TSIC-2_dg; this may be the best follow-up area for Finn/Mo.

Top findings:
- Highest numeric missingness: TSIC-2_dg (0.0%).
- Strong numeric relationship detected between TSIC-2_dg and TSIC-5dg.