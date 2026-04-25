# Max Data Mining Report
========================================

**Dataset**: C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test1\input\sales_data_500.csv
**Rows**: 15, **Columns**: 16

## Data Quality Overview
- Missing values handled: 0

## Correlation Analysis
- **UnitPrice - TotalAmount**: r = 0.719

## Anomaly Detection Results
- **Anomalies detected**: 1 (6.7%)
- **Method**: Isolation Forest detected outliers in numeric features

## Clustering Results
- **Algorithm**: K-Means with Elbow Method
- **Number of clusters**: 4
- **Silhouette score**: 0.328
- **Features used**: OrderID, Quantity, UnitPrice, TotalAmount

### Cluster Profiles
**Cluster 0** (33.3% of customers):
  - OrderID: mean=1003.6, median=1004.0
  - Quantity: mean=3.0, median=3.0
  - UnitPrice: mean=71.99, median=49.99
  - TotalAmount: mean=147.97, median=149.95

**Cluster 1** (6.7% of customers):
  - OrderID: mean=1011.0, median=1011.0
  - Quantity: mean=6.0, median=6.0
  - UnitPrice: mean=199.99, median=199.99
  - TotalAmount: mean=1199.94, median=1199.94

**Cluster 2** (26.7% of customers):
  - OrderID: mean=1008.25, median=1008.0
  - Quantity: mean=8.5, median=8.5
  - UnitPrice: mean=16.37, median=12.74
  - TotalAmount: mean=134.56, median=104.2

**Cluster 3** (33.3% of customers):
  - OrderID: mean=1011.6, median=1012.0
  - Quantity: mean=2.4, median=2.0
  - UnitPrice: mean=31.09, median=39.99
  - TotalAmount: mean=66.38, median=62.0

## Patterns Found
### Pattern 1: Key Correlation
- **Description**: Strong correlation: UnitPrice - TotalAmount (r=0.719)
- **Evidence**: From correlation analysis of 4 numeric features
- **Business Implication**: These metrics move together - can be used for cross-selling or prediction
- **Recommended Action**: Use this relationship for predictive modeling

## Summary & Recommendations
1. **Customer Segmentation**: Identified customer clusters with distinct characteristics
2. **Anomaly Detection**: Found unusual patterns for further investigation
3. **Behavioral Patterns**: Discovered key customer behavior correlations
4. **Business Actions**: Recommendations provided for each pattern