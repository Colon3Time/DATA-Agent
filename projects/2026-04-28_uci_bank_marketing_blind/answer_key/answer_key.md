# Answer Key - UCI Bank Marketing Blind Benchmark

This file is for post-run judging only. Agents must not read this during the pipeline.

## Source
- Dataset: UCI Machine Learning Repository - Bank Marketing.
- Recommended raw file: `bank-additional-full.csv`.
- The CSV uses semicolon delimiter.

## Expected Raw Data
- Rows: 41,188
- Columns: 21
- Target: `y`
- Positive class: `yes`
- Positive count: 4,640
- Negative count: 36,548
- Positive rate: 0.1127
- Problem type: binary classification

## Expected Columns
`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `y`

## Known Missing-Like Values
The dataset encodes missing categorical values as `unknown`, not true nulls:
- `default`: 8,597
- `education`: 1,731
- `housing`: 990
- `loan`: 990
- `job`: 330
- `marital`: 80

## Leakage / Timing Trap
- `duration` is highly predictive but is known only after the call has occurred.
- For a realistic pre-call campaign-scoring model, `duration` must be excluded before modeling.
- A report may mention `duration` as an explanatory post-call diagnostic, but it must not be a deployed pre-call model feature.

## Data Cleaning Expectations
- Preserve all rows unless a strong documented reason exists.
- Do not treat the target `y` as an outlier column.
- Do not convert `unknown` blindly to numeric labels without documenting the meaning.
- Keep target distribution stable after cleaning.
- If multiple files are present, prefer `bank-additional-full.csv` over smaller samples.

## EDA Expectations
Must identify:
- Class imbalance around 11.3% positive.
- Business context: bank term-deposit subscription campaign.
- `duration` is a special leakage/timing variable.
- Previous campaign outcome, contact month, prior contacts, and macroeconomic variables are plausible drivers.
- Accuracy alone is misleading due to imbalance.

## Modeling Expectations
Acceptable models include logistic regression, tree ensembles, gradient boosting, XGBoost/LightGBM, or calibrated models.

Required:
- Stratified split or stratified CV.
- Exclude `duration` for realistic deployment.
- Report positive-class metrics and ROC-AUC/PR-AUC.
- Do not accept near-perfect scores without leakage investigation.

## Business Insight Expectations
A good final report should say:
- The task is campaign prioritization for term-deposit subscription.
- Use the score before calls to prioritize call lists, not after calls.
- Optimize for a business tradeoff such as recall at manageable call volume or precision for high-cost channels.
- Do not operationalize recommendations based on `duration` because it is not known before the call.
- Segment recommendations may include prior successful contact, contact channel, month, and macroeconomic context.

## Critical Fail Conditions
- Target is not `y`.
- `duration` is used as a model feature for the main predictive model.
- Target `y` is used as a feature, outlier flag basis, or transformed away.
- Report claims success while QC failed.
- Model metrics are near perfect and no leakage warning is raised.
- The system reads this answer key during agent execution.
