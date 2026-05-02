# Executive Report - UCI Online Retail Analytics

## Executive Summary

The project now has a full working analytics pipeline from Scout through Rex using the UCI Online Retail transaction data. The key correction was enforcing grain-specific analytics: transactions for descriptive sales, customers for RFM/churn/CLV, invoices for basket analysis, and product-month for inventory.

## Key Findings

- Total valid-sales revenue: 20,972,978.64
- Top revenue country: United Kingdom (85.2% share)
- Peak revenue month: 2011-11
- Customer-level model table: 5,878 known customers
- Champions segment size: 1,294 customers
- Baseline churn label rate: 40.8%

## Model Results

| task       | model          |    roc_auc |     pr_auc |         f1 |   n_test |    mae |         r2 |
|:-----------|:---------------|-----------:|-----------:|-----------:|---------:|-------:|-----------:|
| churn      | logistic_churn |   0.806069 |   0.688001 |   0.689249 |     1470 |  nan   | nan        |
| churn      | rf_churn       |   0.816536 |   0.693368 |   0.706497 |     1470 |  nan   | nan        |
| high_value | rf_high_value  |   0.99027  |   0.95896  |   0.889246 |     1470 |  nan   | nan        |
| clv_proxy  | rf_regressor   | nan        | nan        | nan        |     1470 | 1188.6 |   0.523831 |

## Business Recommendations

1. Prioritize UK retention campaigns because revenue concentration is high.
2. Use RFM segments for marketing actions: Champions/Loyal for retention, Potential for cross-sell, At Risk/Lost for win-back.
3. Use product-month demand tables for inventory planning rather than transaction rows.
4. Treat current Mo results as baseline validation only; final churn/CLV claims require time-cutoff labels.

## Deliverables

- Dana clean table: `output/dana/dana_output.csv`
- Eddie EDA: `output/eddie/eddie_report.md`
- Finn features: `output/finn/engineered_data.csv`
- Iris segmentation: `output/iris/rfm_segments.csv`
- Mo model results: `output/mo/model_results.md`
- Vera charts: `output/vera/charts/`

FINAL_STATUS
============
status: complete_with_modeling_caveat
main_caveat: production-grade churn/CLV requires time-cutoff validation
