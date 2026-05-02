# Finn Feature Engineering Report - UCI Online Retail

Input: C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\eddie\eddie_output.csv

## Outputs

- `engineered_data.csv`: customer-level model table, 5,878 rows
- `finn_output.csv`: same customer-level table for pipeline compatibility
- `invoice_basket_features.csv`: invoice-level basket table, 40,083 rows
- `product_month_features.csv`: product-month inventory table, 67,062 rows

## Grain Contract

- Customer grain: RFM, CLV proxy, churn target candidates
- Invoice grain: market basket readiness
- Product-month grain: inventory/demand optimization
- Transaction grain is not used for churn/CLV modeling

## Target Columns Created

- `is_churned_180d`: 1 if recency > 180 days at snapshot 2011-12-10
- `clv_proxy`: historical customer monetary value
- `is_high_value`: top 20% monetary customers

## Leakage Notes

These are analytical targets from the full historical dataset. For production modeling, Mo must use a time cutoff and rebuild features only from transactions before the cutoff. This file is acceptable for baseline experimentation and pipeline validation, not final deployment claims.

## Feature Governance

- UNKNOWN_CUSTOMER excluded from customer-level table
- Returns/cancellations excluded from valid sales features
- InvoiceDate parsed as real calendar dates, not Excel serial nanoseconds
- No row-level transaction table is handed to Mo for churn/CLV

## Suggested Next Steps

- Iris: run RFM segmentation from `engineered_data.csv` and basket analysis from `invoice_basket_features.csv`
- Mo: train baseline churn/high-value classifiers on `engineered_data.csv`, with leakage caveat clearly reported

FEATURE_GOVERNANCE
==================
selected_feature_count: 17
primary_grain: customer
target_columns: is_churned_180d, is_high_value, clv_proxy
train_only_transforms: required in Mo for production-grade validation
