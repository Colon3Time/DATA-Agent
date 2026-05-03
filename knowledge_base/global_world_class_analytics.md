## [PROVEN] World-Class Analytics Default

Every agent must treat world-class analytics as the default standard, even when the user does not explicitly say it.

Core expectations:
- prove the dataset is credible, licensed, relevant, recent enough, documented, and fit for the target decision
- preserve raw meaning and target integrity
- frame EDA around the business question, KPI, owner, effect size, causality limits, and validation risk
- enforce feature lineage, prediction-time availability, leakage prevention, train-only transforms, and OOT support
- report imbalance-aware metrics, PR-AUC, positive-class metrics, threshold economics, calibration, and time-based validation
- make visuals traceable, non-misleading, accessible, sourced, caveated, and decision-oriented
- gate the cycle for leakage, overfitting, drift, calibration, business satisfaction, and production readiness
- separate executive-ready outputs from production-ready status and never state ROI without assumptions

Large data protocol:
- If the dataset is large or wide, profile a small sample first.
- Prefer chunked reads, streaming, sparse transforms, or incremental learners when memory is tight.
- Avoid full pairwise correlation, exhaustive one-hot encoding, and O(n^2) operations unless there is a clear reason.
- Record the sample fraction, row count processed, memory risk, and any approximation used.

Role isolation:
- `role`, `set_role`, agent name, pipeline stage, and dispatch labels are governance metadata, not predictive features.
- Keep role metadata in logs, reports, and audit blocks only.
- If a role-like field is truly predictive, document why it is safe, available at prediction time, and not a leakage proxy.

Method performance reporting:
- Any time an agent compares methods, the report must include the candidates, the metric used, the score for each candidate, and why the winner won.
- Do not end with a bare summary like "best method was X" without the comparison table or ranked list.
- If only one method was possible, say so explicitly and explain the constraint.
- If a method was rejected, record the reason: lower score, leakage risk, instability, cost, memory, or business mismatch.
- Prefer concrete numbers over adjectives. Summaries are fine only after the numbers are shown.

Required block names that should be preserved in reports when relevant:
DATASET_RISK_REGISTER, DATA_QUALITY_AUDIT, BUSINESS_EDA_FRAME, FEATURE_GOVERNANCE, PRODUCTION_READINESS, PATTERN_VALIDITY, BUSINESS_DECISION_BRIEF, VISUAL_QC, WORLD_CLASS_QC.
