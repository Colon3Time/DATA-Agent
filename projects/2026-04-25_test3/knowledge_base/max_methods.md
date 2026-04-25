
## [2026-04-25] [DISCOVERY]
Task: Data mining on eddie_engineered_output.csv — clusters, anomalies, association rules
Key finding: Multi-technique approach worked well:
- K-Means clustering found clear segments (Silhouette > 0.5)
- Isolation Forest detected meaningful outliers 
- Association Rules need transactional data structure to work
- Elbow plot + Silhouette Score together give best k selection
