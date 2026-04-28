Scout Dataset Brief
===================
Dataset: PulseCart Customer Behavior (raw)
Source: pulsecart_raw.csv (local input)
License: ไม่ระบุ (internal dataset)
Size: 2,237 rows x 21 columns
Format: CSV

Columns Summary:
- customer_id: str — unique=2204, missing=0.0%
- signup_date: str — unique=587, missing=0.0%
- region: str — unique=7, missing=0.0%
- plan_type: str — unique=7, missing=0.0%
- acquisition_channel: str — unique=5, missing=0.0%
- age: float64 — unique=61, missing=3.5%
- account_tenure_days: int64 — unique=643, missing=0.0%
- avg_order_value: float64 — unique=1870, missing=2.5%
- orders_last_90d: int64 — unique=19, missing=0.0%
- app_sessions_last_30d: int64 — unique=23, missing=0.0%
- support_tickets_90d: int64 — unique=6, missing=0.0%
- late_delivery_rate: float64 — unique=465, missing=3.0%
- avg_delivery_delay_hours: float64 — unique=1598, missing=0.0%
- discount_ratio: float64 — unique=540, missing=0.0%
- product_variety_score: float64 — unique=679, missing=2.0%
- return_rate_90d: float64 — unique=273, missing=0.0%
- competitor_price_index: float64 — unique=515, missing=1.8%
- days_since_last_order: float64 — unique=628, missing=0.0%
- account_status_30d: int64 — unique=2, missing=0.0%
- post_period_refund_flag: str — unique=2, missing=0.0%
- account_note_post_period: str — unique=4, missing=89.5%

Known Issues:
- Missing columns: {"account_note_post_period": 89.45, "age": 3.53, "late_delivery_rate": 2.95, "avg_order_value": 2.5, "product_variety_score": 1.97}

[DATASET_PROFILE block]
DATASET_PROFILE
===============
rows         : 2,237
cols         : 21
dtypes       : numeric=14, categorical=7, datetime=0
missing      : {"account_note_post_period": 89.45, "age": 3.53, "late_delivery_rate": 2.95, "avg_order_value": 2.5, "product_variety_score": 1.97}
target_column: account_status_30d
problem_type : classification
class_dist   : {"0": 0.8699, "1": 0.1301}
imbalance_ratio: 6.69
recommended_scaling: StandardScaler