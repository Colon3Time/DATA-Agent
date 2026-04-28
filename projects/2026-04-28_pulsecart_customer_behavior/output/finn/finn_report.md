# Finn Feature Engineering Report
**Date**: 2026-04-28 13:53
**Input**: `C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\eddie\eddie_output.csv`
**Original Features**: 23
**Original Rows**: 2237

**Dropped ID columns**: ['customer_id']

## 1. Data Overview
- Original shape: 2237 rows × 22 cols
- Target column: is_outlier

## 2. Feature Engineering
### Features Created (46):
- `account_tenure_days_over_avg_order_value`
- `account_tenure_days_x_avg_order_value`
- `account_tenure_days_minus_avg_order_value`
- `account_tenure_days_over_avg_delivery_delay_hours`
- `account_tenure_days_x_avg_delivery_delay_hours`
- `account_tenure_days_minus_avg_delivery_delay_hours`
- `account_tenure_days_over_product_variety_score`
- `account_tenure_days_x_product_variety_score`
- `account_tenure_days_minus_product_variety_score`
- `account_tenure_days_over_days_since_last_order`
- `account_tenure_days_x_days_since_last_order`
- `account_tenure_days_minus_days_since_last_order`
- `avg_order_value_over_avg_delivery_delay_hours`
- `avg_order_value_x_avg_delivery_delay_hours`
- `avg_order_value_minus_avg_delivery_delay_hours`
- `avg_order_value_over_product_variety_score`
- `avg_order_value_x_product_variety_score`
- `avg_order_value_minus_product_variety_score`
- `avg_order_value_over_days_since_last_order`
- `avg_order_value_x_days_since_last_order`
- ... and 26 more

### Features Dropped (5):
- region (replaced with freq encoding)
- plan_type (replaced with freq encoding)
- acquisition_channel (one-hot encoded)
- post_period_refund_flag (one-hot encoded)
- account_note_post_period (one-hot encoded)

## 3. Feature Selection (Auto-Compare)
**Best Method**: `rfecv`

| Method | CV Score | Features |
|--------|----------|----------|
| rfecv | 0.9892 | 26 |
| rf_importance | 0.9888 | 26 |
| variance_threshold | 0.9852 | 52 |
| lasso_l1 | 0.9785 | 29 |
| mutual_info | 0.9648 | 26 |

**Selected Features** (26):
- `age`
- `account_tenure_days`
- `avg_order_value`
- `support_tickets_90d`
- `late_delivery_rate`
- `avg_delivery_delay_hours`
- `discount_ratio`
- `return_rate_90d`
- `days_since_last_order`
- `account_status_30d`
- `account_tenure_days_minus_avg_order_value`
- `account_tenure_days_minus_avg_delivery_delay_hours`
- `account_tenure_days_minus_product_variety_score`
- `account_tenure_days_minus_days_since_last_order`
- `avg_order_value_x_avg_delivery_delay_hours`
- `avg_order_value_minus_avg_delivery_delay_hours`
- `avg_order_value_minus_product_variety_score`
- `avg_order_value_x_days_since_last_order`
- `avg_order_value_minus_days_since_last_order`
- `avg_delivery_delay_hours_over_product_variety_score`
- `avg_delivery_delay_hours_x_days_since_last_order`
- `avg_delivery_delay_hours_minus_days_since_last_order`
- `product_variety_score_minus_days_since_last_order`
- `avg_order_value_log`
- `avg_delivery_delay_hours_log`
- `days_since_last_order_log`

## 4. Encoding & Scaling
- **Scaling**: StandardScaler (Z-score) on 26 numeric features
- **Categorical Encoding**: One-Hot for ≤5 categories, Frequency for 5-100 categories
- **Null Handling**: Numeric → Median, Categorical → 'Unknown'

## 5. Self-Improvement Report
- **Method used**: `auto_compare_feature_selection` → best: `rfecv`
- **Reason**: Data-driven CV score comparison across 5 methods
- **New techniques found**: Interactive ratio/product features added
- **Knowledge Base**: Updated with PulseCart project patterns

## Agent Report — Finn
==============================
**Received from**: Eddie — cleaned customer behavior data
**Input shape**: (2237, 22)
**What was done**:
- ID column removal
- Target leakage guard check
- DateTime feature extraction (year, month, day, dayofweek, weekend, hour)
- Numeric interaction features (ratios, products, differences)
- Log transforms for skewed features
- Binned features for high-variance columns
- Frequency/One-Hot encoding for categoricals
- ML-based feature selection (auto_compare × 5 methods)
- StandardScaler normalization

**Key findings**:
- 46 new features created
- 5 features dropped (redundant/leakage/low-variance)
- Best selection method: rfecv

**Output**: `finn_output.csv` ((2237, 28))
**Sent to**: Mo — for model training with engineered features