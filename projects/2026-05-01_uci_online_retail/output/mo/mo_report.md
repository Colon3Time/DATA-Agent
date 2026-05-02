# Mo Baseline Modeling Report

Input: C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\finn\engineered_data.csv
Rows: 5,878
Base feature count: 10

## Results

| task       | model          |    roc_auc |     pr_auc |         f1 |   n_test |    mae |         r2 |
|:-----------|:---------------|-----------:|-----------:|-----------:|---------:|-------:|-----------:|
| churn      | logistic_churn |   0.806069 |   0.688001 |   0.689249 |     1470 |  nan   | nan        |
| churn      | rf_churn       |   0.816536 |   0.693368 |   0.706497 |     1470 |  nan   | nan        |
| high_value | rf_high_value  |   0.99027  |   0.95896  |   0.889246 |     1470 |  nan   | nan        |
| clv_proxy  | rf_regressor   | nan        | nan        | nan        |     1470 | 1188.6 |   0.523831 |

## Validation Caveat

These are baseline models on Finn's customer-level analytical table. Target-derived direct leakage was removed per task (`recency_days` for churn; `monetary`/`avg_order_value` for high-value and CLV proxy). The churn and CLV targets are still derived from full historical data for pipeline validation. Production-grade modeling must rebuild labels with a cutoff date and compute features only before that cutoff.

MODEL_GOVERNANCE
================
grain: customer
leakage_status: acceptable for baseline, not final deployment
required_next_step: time-cutoff validation before business claim
