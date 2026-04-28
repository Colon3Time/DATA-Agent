Dana Cleaning Report
===================

Input: C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\scout\scout_output.csv

Before: 2237 rows, 21 columns
After:  2237 rows, 22 columns

Missing Values:
- age: 3.5% missing -> median imputation
- avg_order_value: 2.5% missing -> median imputation
- late_delivery_rate: 3.0% missing -> median imputation
- product_variety_score: 2.0% missing -> median imputation
- competitor_price_index: 1.8% missing -> median imputation
- account_note_post_period: 89.5% missing -> median imputation
- customer_id: 0% missing -> no action needed
- signup_date: 0% missing -> no action needed
- region: 0% missing -> no action needed
- plan_type: 0% missing -> no action needed
- acquisition_channel: 0% missing -> no action needed
- account_tenure_days: 0% missing -> no action needed
- orders_last_90d: 0% missing -> no action needed
- app_sessions_last_30d: 0% missing -> no action needed
- support_tickets_90d: 0% missing -> no action needed
- avg_delivery_delay_hours: 0% missing -> no action needed
- discount_ratio: 0% missing -> no action needed
- return_rate_90d: 0% missing -> no action needed
- days_since_last_order: 0% missing -> no action needed
- account_status_30d: 0% missing -> no action needed
- post_period_refund_flag: 0% missing -> no action needed

Zero-as-Missing Conversion:

Outlier Detection:
- Likely Error: None
- Likely Real (flagged, preserved):
  - age: 9 rows -> is_outlier=1
  - account_tenure_days: 69 rows -> is_outlier=1
  - avg_order_value: 84 rows -> is_outlier=1
  - orders_last_90d: 3 rows -> is_outlier=1
  - app_sessions_last_30d: 5 rows -> is_outlier=1
  - support_tickets_90d: 126 rows -> is_outlier=1
  - late_delivery_rate: 44 rows -> is_outlier=1
  - avg_delivery_delay_hours: 77 rows -> is_outlier=1
  - discount_ratio: 42 rows -> is_outlier=1
  - product_variety_score: 13 rows -> is_outlier=1
  - return_rate_90d: 72 rows -> is_outlier=1
  - days_since_last_order: 62 rows -> is_outlier=1
  - account_status_30d: 291 rows -> is_outlier=1
- Uncertain (flagged):
  - row 291: age (97.0) -> age=97.00 (IQR bounds: 8.50-68.50)
  - row 304: age (97.0) -> age=97.00 (IQR bounds: 8.50-68.50)
  - row 925: age (104.0) -> age=104.00 (IQR bounds: 8.50-68.50)
  - row 1333: age (104.0) -> age=104.00 (IQR bounds: 8.50-68.50)
  - row 1404: age (104.0) -> age=104.00 (IQR bounds: 8.50-68.50)
- outlier_flags.csv: 902 rows (details in report)
  Columns: row_index, column_name, value, verdict, reason

Data Quality Score:
- Completeness: Before 95.1% -> After 100.0%
- Validity: Before 100.0% -> After 100.0%
- Overall: Before 97.6% -> After 100.0%

Column Stats (Before -> After):
- age: mean 38.38->38.37, std 11.46->11.25
- account_tenure_days: mean 276.47->276.50, std 178.03->177.98
- avg_order_value: mean 61.38->61.14, std 65.24->64.43
- late_delivery_rate: mean 0.18->0.17, std 0.12->0.12
- product_variety_score: mean 62.09->62.08, std 18.37->18.19

New Method Found:
None

Zero-as-Missing Conversion:
- competitor_price_index: 0 zeros found