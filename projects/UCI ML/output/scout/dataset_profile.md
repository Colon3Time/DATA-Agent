DATASET_PROFILE
===============
rows: 1067371
cols: 8
problem_type: regression
target: Quantity
target_description: ปริมาณการสั่งซื้อต่อรายการ — ใช้สำหรับ demand forecasting และ customer value analysis
imbalance: no_target or regression
missing_cols: {"Customer ID": 22.77, "Description": 0.41}
key_features: ['Quantity', 'Price', 'Customer ID']
data_types: numeric=4, categorical=4, datetime=1, id=2, other=-3
date_cols: ['InvoiceDate']
id_cols: ['StockCode', 'Description']
size_mb: 283.84
recommended_scaling: StandardScaler
business_level: medium (descriptive + behavioral)
dispatch_recommendation: Dana: descriptive analytics (quantity based), Dana+Mo: behavioral analytics (customer patterns)

DATASET_RISK_REGISTER
=====================
Source credibility: High — UCI Machine Learning Repository, curated benchmark dataset
License/usage: Allowed — public domain for academic/commercial use
Business fit: High — Online Retail transaction data, directly applicable to e-commerce analytics
Target suitability: Clear — Quantity: ปริมาณการสั่งซื้อต่อรายการ — ใช้สำหรับ demand forecasting และ customer value analysis
Recency/deployment fit: Dataset is historical (2010-2011) — limited for current trend detection but valid for pattern learning
Leakage risks: None — no future information, no post-outcome columns
Bias/coverage risks: Single retailer from UK — may not generalize to other markets
Data dictionary: Available (UCI provides column descriptions)
Verdict: Use — suitable for multi-level analytics (descriptive → behavioral → predictive)