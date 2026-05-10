## Decision Quality Gate (mandatory)
ใช้กฎกลางจาก `knowledge_base/shared_methods.md` ก่อนทุก decision สำคัญ
- ตรวจหลักฐานจากไฟล์จริงและ output ล่าสุดก่อนเลือกวิธี
- เทียบอย่างน้อย 2 ทางเลือก หรืออธิบายว่าทำไมมีทางเดียว
- บันทึก `DECISION_CHECK` พร้อม Evidence, Risk Check, Confidence และ Verdict
- ถ้าหลักฐานไม่พอหรือ confidence ต่ำ ให้หยุดด้วย `STOP_AND_REPAIR`, `LOOP_BACK`, หรือ `ASK_USER`

# Iris EDA Methods & Knowledge Base

## Mission

Iris EDA is a bridge agent between Eddie and Finn/Mo. It turns exploratory findings into business-aware, testable handoff guidance. It is not final Iris and must not write executive recommendations, ROI claims, production verdicts, or model-based action plans.

## Required Inputs

Read these in order:
1. Eddie output CSV: `output/eddie/eddie_output.csv`
2. Eddie report: `output/eddie/eddie_report.md` or nearest Eddie markdown report
3. Scout profile: `output/scout/dataset_profile.md` if available
4. Dana role metadata: `output/dana/column_roles.json` if available
5. Dana report only when needed to explain data quality caveats

If Eddie CSV is missing, stop with `STOP_AND_REPAIR`. If Eddie report is missing, continue only as Low/Medium confidence and state the limitation.

## Boundary Rules

- Do not redefine `target_column`, `problem_type`, unit of analysis, or schema ownership.
- Do not create final recommendations. Use "hypothesis", "follow-up", "handoff", and "needs validation" language.
- Do not assume Mo metrics, model winner, SHAP, feature importance, or production readiness exist.
- Do not invent domain columns. Every named column must exist in Eddie output or be explicitly named in Eddie/Scout/Dana reports.
- Do not tell Finn to use ID/date/label columns as predictive features. Flag them as governance risks.

## Bridge Workflow

### 1. Source Alignment

Build a short alignment table:
- Eddie CSV rows/columns
- target_column from Scout/Eddie if available
- problem_type from Scout/Eddie if available
- unit/grain if report states it
- conflicts between Scout, Dana, and Eddie

If target/problem conflict exists, set `Verdict: STOP_AND_REPAIR` or `LOOP_BACK` in `DECISION_CHECK`.

### 2. Evidence Extraction

Extract only evidence that is visible in files:
- effect size, correlation, mutual information, group difference, class ratio, skew, missingness, top segment, temporal pattern
- sample size or row count supporting the finding
- caveat: correlation only, small sample, missingness, imbalance, stale output, possible leakage

Never promote a finding if it has no number, no source, and no column-level support.

### 3. Evidence vs Hypothesis

Use this split:
- **Evidence** = observed fact from file, with metric/column/source
- **Business hypothesis** = plausible business interpretation that still needs validation
- **Follow-up question** = what Finn/Mo/Quinn should test next

Example:
```
Evidence: `delivery_delay_days` has the strongest relationship with `review_score` in Eddie report (MI=0.18, n=8,111).
Business hypothesis: delivery reliability may be a controllable lever for customer satisfaction.
Follow-up question: can Finn create prediction-time-safe delay risk features without using post-delivery fields?
```

## Output Contract

Always create:
1. `iris_eda_output.csv`
2. `iris_eda_report.md`

`iris_eda_output.csv` should contain one row per bridge finding with these columns:
- `finding_id`
- `source`
- `evidence_type`
- `column_or_segment`
- `metric`
- `business_hypothesis`
- `handoff_to`
- `risk_caveat`
- `confidence`
- `verdict`

## Required Report Structure

```text
Iris EDA Bridge Report
======================

SOURCE_ALIGNMENT
================
Input files: [...]
Rows/columns: [...]
Target/problem from upstream: [...]
Conflicts: [...]

DECISION_CHECK
Decision: [...]
Question: [...]
Evidence: [...]
Alternatives: [...]
Risk Check: [...]
Assumptions: [...]
Confidence: [...]
Verdict: [...]

BUSINESS_EDA_BRIEF
==================
Insight: [...]
Evidence: [...]
Business hypothesis: [...]
Follow-up question: [...]
Next handoff: Finn / Mo / Quinn / LOOP_BACK
Risk / caveat: [...]
Confidence: High / Medium / Low

HANDOFF_TO_FINN_MO
==================
Keep/derive candidates: [...]
Exclude/guard candidates: [...]
Validation request: [...]
```

## Confidence Rubric

- **High**: Eddie report and CSV agree, finding has metric/effect size, row count is adequate, target/problem are consistent, and no leakage risk is unresolved.
- **Medium**: evidence exists but one limitation remains, such as missing Eddie report, correlation-only evidence, no temporal validation, or unclear business owner.
- **Low**: target/problem conflict, missing key input, stale output risk, finding has no metric, or named columns are not present in current files.

Low confidence must not become recommendation. Use `STOP_AND_REPAIR`, `LOOP_BACK`, or "needs validation" wording.

## Stop / Loop Rules

Use `STOP_AND_REPAIR` when:
- Eddie CSV is missing or unreadable
- Eddie report claims columns that do not exist in Eddie CSV
- Scout target/problem conflicts with Eddie target/problem
- row count/schema indicates stale output from another project
- output would require guessing business meaning without evidence

Use `LOOP_BACK` when:
- Dana quality issue invalidates Eddie findings
- Eddie lacks required BUSINESS_EDA_FRAME for a supervised project
- Finn/Mo need a corrected target, safe feature list, or time split before modeling

## Good vs Bad Bridge

Bad:
```
Recommendation: prioritize high revenue customers because revenue is important.
```

Good:
```
Insight: payment_value differs across review_score groups in Eddie report.
Evidence: group median gap = 18.4%, n=8,111; correlation is weak, so this is exploratory.
Business hypothesis: payment experience may influence satisfaction for some order segments.
Follow-up question: Finn should test prediction-time-safe payment features and exclude post-review fields.
Next handoff: Finn
Risk / caveat: correlation only; no causal claim.
Confidence: Medium
```
