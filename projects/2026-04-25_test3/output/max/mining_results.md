# Max Data Mining Report
============================================================

**Generated**: 2026-04-25 19:42:12

## Techniques Used

- Isolation Forest (Anomaly Detection)
- Pearson Correlation (Feature Relationships)
- K-Means Clustering (Segmentation)
- Elbow Method (Optimal Cluster Selection)
- Data Profiling & Statistical Analysis

## Patterns Found

### Pattern 1: Anomaly Segments

- **Evidence**: 30 rows (5.0%) identified as anomalies
- **Detection Method**: Isolation Forest (contamination=0.05)
- **Importance**: These rows deviate significantly from normal patterns

### Pattern 2: Feature Clusters

- **Evidence**: 3 distinct clusters found
- **Quality**: Silhouette=0.401, Davies-Bouldin=1.038
- **Importance**: Data naturally groups into 3 segments

### Pattern 3: Correlation Patterns

- **Evidence**: 1 significant correlation pairs found
  - unit_price ↔ total_amount: r=0.826
- **Importance**: Indicates interdependencies between features

## Anomalies Detected

- Total anomalies: 30
- Threshold: Top 5% most anomalous

## Clusters Found

### Cluster 0 (n=404)
- **Characteristics**: total_amount=18295.00, unit_price=3205.56, quantity=6.96
- **Size**: 67.3% of data

### Cluster 1 (n=149)
- **Characteristics**: total_amount=134250.50, unit_price=19891.47, quantity=8.85
- **Size**: 24.8% of data

### Cluster 2 (n=47)
- **Characteristics**: total_amount=47674.41, unit_price=6475.56, quantity=7.81
- **Size**: 7.8% of data

## Business Implication

### What These Patterns Mean

1. **Customer/Data Segments**: The clusters represent distinct groups that may require different strategies
2. **Risk/Anomaly Detection**: Flagged anomalies warrant investigation for data quality or fraud
3. **Feature Relationships**: Correlations can guide feature selection and engineering

### Key Relationships

- **unit_price** and **total_amount** have a positive correlation (0.83)

