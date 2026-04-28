# Client Brief - UCI Bank Marketing Blind Benchmark

## Business Context
We are a bank planning outbound term-deposit marketing campaigns. The business wants a model that can score clients before or at campaign planning time, so marketing teams can prioritize likely subscribers and reduce wasted calls.

Source dataset: UCI Machine Learning Repository, Bank Marketing dataset.

## Dataset Handling
- The raw input may be zipped and may contain multiple CSV files.
- Prefer the full `bank-additional-full.csv` dataset when available.
- CSV delimiter may be semicolon (`;`), so detect the delimiter instead of assuming comma.

## Prediction Target
- Target column: `y`
- Positive class: `yes`
- Problem type: binary classification
- Business meaning: whether the client subscribes to a term deposit.

## Timing And Leakage Policy
The model should be usable before the phone call result is known.

Do not use features that are only known after the call is completed or that directly encode the outcome. In particular:
- `duration` is post-call information and must not be used for a realistic pre-call predictive model.
- Do not use target encoding unless it is fitted out-of-fold inside cross-validation or inside the train split only.
- Do not use row identifiers or keys as model features if any are present.

## Evaluation Requirements
- Use stratified splitting or stratified cross-validation.
- Report metrics beyond accuracy, including positive-class recall/precision/F1 and ROC-AUC or PR-AUC.
- The positive class is imbalanced, so accuracy alone is not sufficient.
- If F1/AUC/accuracy is near perfect, treat it as likely leakage and loop back for investigation.

## Expected Deliverables
- Cleaned dataset and cleaning report.
- EDA report that discusses class imbalance and important relationships.
- Feature engineering report that explicitly states excluded leakage columns.
- Model comparison report with credible metrics.
- QC report that checks target consistency, leakage, overfitting, and report/CSV consistency.
- Business recommendations for campaign targeting.
