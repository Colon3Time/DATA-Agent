# Finn Feature Engineering Report - UCI Online Retail

Input: projects\2026-05-01_uci_online_retail\output\eddie\eddie_output.csv

## Outputs

- `engineered_data.csv`: full-history customer table, 5,878 rows
- `engineered_data_oot.csv`: time-cutoff table at 2011-09-30 with 90-day churn label, 5,430 rows
- `finn_feature_manifest.json`: target-specific feature exclusions for Mo
- `invoice_basket_features.csv`: invoice-level basket table, 40,083 rows
- `product_month_features.csv`: product-month inventory table, 67,062 rows

## Leakage Controls

- `engineered_data.csv` keeps targets for reference, but Mo must follow `finn_feature_manifest.json`.
- Monetary-derived columns are explicitly listed: clv_proxy, avg_order_value, is_high_value.
- OOT validation table uses features before 2011-09-30 and labels activity in the next 90 days.

FEATURE_GOVERNANCE
==================
selected_feature_count: 17
primary_grain: customer
target_columns: monetary, is_churned_180d, is_high_value, clv_proxy
train_only_transforms: enforced downstream by Mo manifest
