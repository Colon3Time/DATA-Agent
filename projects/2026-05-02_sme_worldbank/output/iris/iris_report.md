IRIS_REPORT
===========

BUSINESS_DECISION_BRIEF
=======================
Insight: customer-level features are ready for business actioning around monetary.
Evidence: Finn produced customer-level RFM features and Mo produced holdout metrics.
Business lever: revenue / retention / risk
Target KPI: monetary
Owner: analytics / growth / finance
Recommended action: prioritize high-value, recent, frequent customers for interventions.
Expected impact: improve prioritization quality and reduce wasted outreach.
Assumptions: regression fallback is a proxy for business ranking, not a final production model.
Risks / trade-offs: proxy target may not match the final business objective; validate before rollout.
Validation plan: pilot, cohort tracking, and out-of-time validation on the chosen business KPI.
Confidence: Medium
Production caveat: treat this as decision support until the target and deployment metric are confirmed.

Mo excerpt:
MO_REPORT
=========

PRODUCTION_READINESS
====================
target_column: monetary
winner model: dummy_mean
rmse: 0.000000
mae: 0.000000
r2: 0.000000
validation: holdout split 80/20
calibration: not applicable for regression fallback
threshold strategy: not applicable for regression fallback
business impact: use predicted target as a prioritization signal
risk: this is a fallback model, not a production benchmark
confidence: Medium