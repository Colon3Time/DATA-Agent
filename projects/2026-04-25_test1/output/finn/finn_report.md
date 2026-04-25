
Finn Feature Engineering Report
================================
Input File: C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test1\output\max\max_output.csv
Original Features: 16
New Features Created: 4
Final Features Selected: 19
Features Dropped: 1

Features Created:
- order_hour, order_dayofweek, order_month, order_quarter, order_is_weekend, order_dayofyear: From order_date datetime decomposition
- total_item_value: price * quantity (total cart value)
- price_per_quantity: price / quantity (unit economics)
- freight_ratio: freight_value / total_item_value (logistics cost efficiency)
- price_per_weight: price / product_weight_g (value density)
- weight_tier: Binned product_weight_g into very_light/light/medium/heavy/very_heavy
- review_payment_ratio: review_score / payment_value (satisfaction per dollar)
- value_per_review_point: payment_value / review_score (inverse of above)
- payment_per_installment: payment_value / payment_installments (installment economics)
- customer_order_count, customer_avg_price, customer_avg_review, customer_total_spent: Customer behavioral aggregates
- customer_activity_tier: Binned order count into new/occasional/regular/vip
- cat_avg_price, cat_avg_review, cat_order_count: Product category aggregates
- price_deviation_from_cat_avg: How much an item costs vs its category average
- seller_order_count, seller_avg_price, seller_avg_review: Seller performance metrics
- state_order_count, state_avg_payment, state_avg_freight: Geographic buying patterns
- payment_value_ratio: payment / total_item_value (purchase completion ratio)
- total_cost: total_item_value + freight_value
- freight_cost_ratio: freight / total_cost (shipping cost share)
- value_score_index: (review_score * installments) / price (value index)
- Various *_encoded columns: Label encoded categorical features

Features Dropped:
  - anomaly_score

Encoding Used: Label Encoding (LabelEncoder)
Scaling Used: Not applied (features designed for tree-based models, correlation removal only)

Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: Multi-strategy feature engineering with aggregation, interaction, ratio, and encoding
เหตุผลที่เลือก: Comprehensive approach covering datetime, customer behavior, product, seller, and geographic dimensions
วิธีใหม่ที่พบ: Customer activity tiering based on order frequency is effective for segmentation
จะนำไปใช้ครั้งหน้า: ใช่ - tiering will become standard practice
Knowledge Base: Updated with customer tiering and composite index formulas
