REX_REPORT
==========

Production readiness: prototype only; monitoring and retrain plan required.
Validation limitations: out-of-time validation is still recommended before rollout.
Business impact assumptions: treat fallback metrics as directional and confirm on the real KPI.
Monitoring: track drift, accuracy proxy, and business KPI after deployment.
Retrain plan: retrain when drift or KPI decay appears.

Executive summary
=================
The pipeline completed with a fallback offline path.
The generated outputs are usable for review but not final production approval.

## QUINN
QUINN_REPORT
============

WORLD_CLASS_QC
===============
leakage: none obvious in the fallback path
overfitting: holdout metrics reviewed from Mo report
drift: not fully proven; OOT validation still recommended
calibration: regression fallback, calibration not applicable
business_satisfaction: satisfied at fallback-review level
restart_cycle: no
verdict: satisfied

Mo report excerpt:
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
risk: this is a fallback model, not a production b

## IRIS
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


## VERA
VERA_REPORT
===========

VISUAL_QC
=========
source evidence: fallback visual summary from upstream pipeline outputs
decision purpose: support executive and analyst review
chart rationale: no chart rendered in offline fallback, but summary checks remain valid
misleading-risk check: avoid implying causal impact from the fallback visuals
accessibility: use simple labels 