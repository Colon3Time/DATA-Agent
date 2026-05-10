REX_REPORT
==========

Production readiness: prototype only; monitoring and retrain plan required.
Validation limitations: time-based / out-of-time validation limitation remains unless explicitly proven.
Business impact assumptions: ROI and cost impact require domain assumptions and pilot evidence.
Monitoring: track drift, prediction quality proxy, source coverage, and target KPI after deployment.
Retrain plan: retrain when drift, KPI decay, or source schema changes appear.

Executive summary
=================
The pipeline produced a target-aligned analytical benchmark and QC summary.
The result is suitable for review, not final production approval.

## QUINN
QUINN_REPORT
============

WORLD_CLASS_QC
===============
target_alignment: pass
leakage: warn - no obvious report-level leakage token, code-level audit still required
overfitting: warn - builtin holdout benchmark is limited
drift: warn - OOT validation not proven
calibration: warn - probability calibration not proven
business_satisfaction: satisfied
restart_cycle: no
verdict: satisfied
Restart From: none

Mo report excerpt:
MO_REPORT
=========

PRODUCTION_READINESS
====================
problem_type: regression
target_column: Value
winner model: numpy_ridge_alpha_1
rmse_log_scale: 0.175815
mae_log_scale: 0.138608
r2_log_scale: 0.992766
target_trained: log1p(Value)
target_transform: log1p(Value)
inverse_transform: np.expm1
target_transform_note: clipped 7 negative Value rows to 0 before log1p
target_transform_note: hard override for target Value
rmse_original_scale: 24.874606
mae_original

## IRIS
IRIS_REPORT
===========

BUSINESS_DECISION_BRIEF
=======================
Insight: model evidence is available for target `Value`.
Evidence: Mo benchmark selected `numpy_ridge_alpha_1` for a `regression` task.
Business lever: data governance, analytical prioritization, and risk review
Target KPI: Value
Owner: analytics / data product team
Recommended action: review class balance, top error modes, and source coverage before operational rollout.
Expected impact: improves confidence in downstream decisions by tying actions to the declared Scout target.
Assumptions: builtin benchmark is a minimum viable analytical check, not final model selection.
Risks / trade-offs: source metadata may encode collection-year patterns rather than deployable business behavior.
Validation plan: out-of-time validation if a real time axis exists, plus holdout review and domain sign-off.
Confidence: Medium
Product

## VERA
VERA_REPORT
===========

VISUAL_QC
=========
source evidence: Mo predictions for target `Value`
decision purpose: support model review and executive explanation
chart rationale: generated 0 target-aligned diagnostic chart(s)
misleading-risk check: charts show association/error patterns only, not causal impact
accessibility: simple titles, direct labels, high-contrast color scale
charts: none

fallback_chart: charts/01_data_overview.png


## MO
MO_REPORT
=========

PRODUCTION_READINESS
====================
problem_type: regression
target_column: Value
winner model: numpy_ridge_alpha_1
rmse_log_scale: 0.175815
mae_log_scale: 0.138608
r2_log_scale: 0.992766
target_t