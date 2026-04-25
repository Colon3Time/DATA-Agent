# Patterns Found

Generated: 2026-04-25 19:42

## Dataset Overview

- Rows: 600
- Columns: 16
- Numeric features: 5
- Categorical features: 8

## Anomalies Detected

- **Anomalies count**: 30 (5.0%)
- **Top anomalous rows**:
       unit_price  quantity  discount_pct  total_amount  return_flag
288     27393.0      13.0          0.15    181740.375            1
349     19017.0      11.0          0.00    181740.375            1
183     27393.0       7.0          0.05    181740.375            1
502     27393.0      10.0          0.10    181740.375            1
565       301.0       1.0          0.20       241.000            1

## Correlations

- **unit_price** ↔ **total_amount**: r = 0.826

## Clusters

- Optimal clusters: 3
- Silhouette Score: 0.401
- Davies-Bouldin Index: 1.038

### Cluster 0 (404 rows)
- Key features:
  - total_amount: 18295.00
  - unit_price: 3205.56
  - quantity: 6.96

### Cluster 1 (149 rows)
- Key features:
  - total_amount: 134250.50
  - unit_price: 19891.47
  - quantity: 8.85

### Cluster 2 (47 rows)
- Key features:
  - total_amount: 47674.41
  - unit_price: 6475.56
  - quantity: 7.81

## Key Patterns

### Pattern 1: Data Distribution

- The dataset contains 600 records with 5 numeric features
- **unit_price**: mean=7605.38, std=8861.18
- **quantity**: mean=7.50, std=3.86
- **discount_pct**: mean=0.06, std=0.06

### Pattern 2: Anomaly Insights

- **unit_price**: Anomaly mean=11900.55 vs Normal mean=7379.32 (+61.3% difference)
- **quantity**: Anomaly mean=7.97 vs Normal mean=7.47 (+6.6% difference)
- **discount_pct**: Anomaly mean=0.10 vs Normal mean=0.06 (+75.1% difference)

## Business Implications

1. **Anomaly Patterns**: Identified outliers that may represent fraud, errors, or rare events
2. **Cluster Segments**: Natural groupings found in data using K-Means clustering
3. **Correlation Insights**: Feature relationships that can inform feature engineering

## Recommended Actions

1. Investigate anomaly rows for potential data quality issues
2. Use cluster segments for targeted analysis or personalization
3. Consider correlation pairs for dimensionality reduction
