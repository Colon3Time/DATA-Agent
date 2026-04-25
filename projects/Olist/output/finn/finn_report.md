# Finn Feature Engineering Report

## Overview
- **Input**: C:\Users\Amorntep\DATA-Agent\projects\olist\output\max\max_output.csv
- **Original Features**: 22
- **New Features Created**: 38
- **Final Features**: 59
- **Rows**: 98666

## Features Created

### Datetime Features (32)
- Basic datetime components (hour, day, week, month, quarter) from all datetime columns
- `is_weekend`: Weekend indicator for each datetime column
- `is_first_day`/`is_last_day`: Month boundary indicators

### Time Difference Features (8)
- `approval_time_hours`: Time from purchase to approval
- `delivery_time_days`: Total delivery time from purchase
- `processing_time_hours`: Time from approval to carrier handoff
- `carrier_delivery_time_days`: Time with carrier
- `delivery_vs_estimated_days`: Days difference from estimated delivery

### Customer Behavior Features (1)
- `customer_order_count`: Total orders per customer
- `customer_total_spend`: Total spend per customer
- `customer_avg_spend`: Average order value per customer
- `customer_spend_std`: Spend variability per customer
- `customer_tenure_days`: Days between first and last order
- `customer_activity_tier`: Customer activity classification (new/occasional/active/highly_active)

### Product Features (0)
- `price_per_weight`: Price normalized by weight
- `price_per_length`: Price normalized by length
- `product_volume_cm3`: Product volume from dimensions
- `review_score_binned`: Review score categorized (poor/good/excellent)

### Interaction Features (0)
- `installment_value`: Payment value per installment
- `review_score_x_price`: Review score weighted by price
- `freight_to_price_ratio`: Freight cost relative to price
- `weight_payment_ratio`: Weight per unit payment

### Encoded Features (2)
- One-Hot Encoding for low cardinality (≤5 unique values)
- Label Encoding for medium cardinality (6-50 unique values)
- High cardinality columns (>50 unique values) left as-is

## Features Dropped
Total: 1
- Highly correlated features (>0.95 correlation)
- avg_item_price

## Encoding Used
- One-Hot Encoding: For categorical columns with ≤5 unique values
- Label Encoding: For categorical columns with 6-50 unique values
- Decision based on cardinality to balance information retention vs dimensionality

## Missing Value Handling
- **Numeric columns**: Median for skewed, Mean for normal distribution
- **Categorical columns**: Filled with 'unknown'

## Feature Selection Method
- Target: payment_value
- Method: SelectKBest (f_classif for continuous target, mutual_info_classif for categorical)
- Selected 12 features with positive importance scores

## Self-Improvement Report

### Method Used
Comprehensive feature engineering pipeline covering:
1. Datetime decomposition and time differences
2. Customer aggregation features
3. Product ratio features
4. Interaction features
5. Statistical encoding
6. Feature selection via SelectKBest

### Why These Methods
- **Datetime features**: Essential for time-series patterns - hour affects behavior, weekday/weekend affects sales
- **Time differences**: Critical for delivery performance analysis and prediction
- **Customer aggregation**: Captures user behavior patterns - active vs occasional customers behave differently
- **Product ratios**: Normalize features to make them comparable across product categories
- **Interaction features**: Models complex relationships between attributes (e.g., quality vs price)

### New Methods Discovered
All techniques used were from existing Knowledge Base. No new methods discovered.

### Improvement for Next Time
Could add:
- Polynomial features for non-linear relationships
- Target encoding for high cardinality categorical variables
- RFE (Recursive Feature Elimination) as alternative selection method
- Weight of Evidence (WoE) encoding for categorical variables with target relationship

### Knowledge Base Update
No update needed - all methods were already documented in Knowledge Base.
