# Max Data Mining Report

## Overview
- **Dataset**: 800 rows, 19 columns
- **Features Analyzed**: 15 features
- **Optimal Clusters**: 2
- **Silhouette Score**: 0.143
- **Davies-Bouldin Score**: 2.347

## Clustering Results

### Cluster Sizes
| Cluster | Size | Percentage |
|---------|------|------------|
| 0.0 | 12.1% | Cluster 0.0 (12.1% of data) —  |
| 1.0 | 87.9% | Cluster 1.0 (87.9% of data) —  |

### Distinctive Features per Cluster

## Anomaly Detection
- **Anomalies Found**: 40 (5.0%)

### Top Anomaly Characteristics
- **department**: Anomaly mean = 1.9 vs Normal mean = 2.64 (Δ = -0.74)
- **position**: Anomaly mean = 1.62 vs Normal mean = 2.56 (Δ = -0.93)
- **age**: Anomaly mean = 41.3 vs Normal mean = 40.53 (Δ = +0.77)
- **gender**: Anomaly mean = 0.45 vs Normal mean = 0.52 (Δ = -0.07)
- **hire_date**: Anomaly mean = 327.6 vs Normal mean = 365.39 (Δ = -37.79)

## Strong Correlations Found
- No strong correlations found (>0.5)

## Business Implications
1. **Cluster Interpretation**: The 2 clusters represent distinct employee segments with different characteristics.
2. **Anomaly Detection**: 40 employees (5.0%) identified as outliers may need special attention.
3. **Recommended Actions**:
   - Investigate anomaly patterns for potential attrition risks
   - Use cluster profiles to design targeted retention strategies
   - Monitor strong correlations for predictive modeling

## Visualizations
- Elbow Method: `C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test4\output\max\elbow.png`
- PCA Projection: `C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test4\output\max\pca_clusters.png`
- Correlation Heatmap: `C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test4\output\max\correlation_heatmap.png`
