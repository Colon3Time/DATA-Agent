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
risk: this is a fallback model, not a production benchmark
confidence: Medium