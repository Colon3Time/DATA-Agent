import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Handle .md input — find CSV in parent
if INPUT_PATH.endswith('.md'):
    parent = Path(INPUT_PATH).parent.parent
    csvs = sorted(parent.glob('**/dana_output.csv')) + sorted(parent.glob('**/*_output.csv'))
    if csvs:
        INPUT_PATH = str(csvs[0])
        print(f"[STATUS] Resolved .md input → CSV: {INPUT_PATH}")

print("[STATUS] Loading employee data...")
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape}")
print(f"[STATUS] Columns: {list(df.columns)}")

# ============================================================
# 1. DATA PREPARATION
# ============================================================
print("\n[STATUS] ---- PHASE 1: Data Preparation ----")

# Identify key columns
target_cols = [c for c in df.columns if any(k in c.lower() for k in ['attrit', 'leave', 'quit', 'churn', 'target', 'label'])]
dept_cols = [c for c in df.columns if any(k in c.lower() for k in ['dept', 'department', 'team'])]
perf_cols = [c for c in df.columns if any(k in c.lower() for k in ['perform', 'rating', 'score', 'efficien'])]

print(f"[STATUS] Target columns found: {target_cols}")
print(f"[STATUS] Department columns: {dept_cols}")
print(f"[STATUS] Performance columns: {perf_cols}")

# Encode categorical columns
df_encoded = df.copy()
label_encoders = {}
for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# Separate features for clustering
exclude_cols = target_cols + ['employee_id', 'id', 'name', 'emp_id', 'employee', 'name_emp']
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]
if not feature_cols:
    feature_cols = df_encoded.columns.tolist()

# Handle NaN — only numeric columns
numeric_cols = df_encoded[feature_cols].select_dtypes(include=[np.number]).columns
df_encoded[numeric_cols] = df_encoded[numeric_cols].fillna(df_encoded[numeric_cols].median())
feature_data = df_encoded[numeric_cols]

# Drop columns with zero variance
feature_data = feature_data.loc[:, feature_data.nunique() > 1]
numeric_cols = feature_data.columns.tolist()

print(f"[STATUS] Features for analysis: {list(feature_data.columns)}")

# ============================================================
# 2. ELBOW METHOD + CLUSTERING
# ============================================================
print("\n[STATUS] ---- PHASE 2: Elbow Method & Clustering ----")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_data)

# Elbow Method
print("[STATUS] Running Elbow Method...")
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
elbow_path = os.path.join(OUTPUT_DIR, 'elbow.png')
plt.savefig(elbow_path)
plt.close()
print(f"[STATUS] Elbow plot saved: {elbow_path}")

# Choose optimal K (find elbow point)
diffs = np.diff(inertias)
diffs2 = np.diff(diffs)
optimal_k = np.argmax(diffs2) + 2  # +2 because diff reduces length by 2
optimal_k = max(2, min(optimal_k, 7))  # Keep reasonable
print(f"[STATUS] Optimal K from elbow: {optimal_k}")

# Clustering with optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df_encoded['cluster'] = cluster_labels

# Silhouette score
sil_score = silhouette_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)
print(f"[STATUS] Silhouette Score: {sil_score:.3f}")
print(f"[STATUS] Davies-Bouldin Score: {db_score:.3f}")

# ============================================================
# 3. PCA VISUALIZATION
# ============================================================
print("\n[STATUS] ---- PHASE 3: PCA Visualization ----")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_encoded['pca1'] = X_pca[:, 0]
df_encoded['pca2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA Projection with Cluster Labels')
plt.grid(True)
pca_path = os.path.join(OUTPUT_DIR, 'pca_clusters.png')
plt.savefig(pca_path)
plt.close()
print(f"[STATUS] PCA plot saved: {pca_path}")

# ============================================================
# 4. ANOMALY DETECTION
# ============================================================
print("\n[STATUS] ---- PHASE 4: Anomaly Detection ----")

iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)
df_encoded['anomaly'] = anomaly_labels
n_anomalies = (anomaly_labels == -1).sum()
print(f"[STATUS] Anomalies detected: {n_anomalies} ({100*n_anomalies/len(df_encoded):.1f}%)")

# ============================================================
# 5. CLUSTER PROFILING (SAFE VERSION)
# ============================================================
print("\n[STATUS] ---- PHASE 5: Cluster Profiling ----")

cluster_profiles = []
for c in sorted(df_encoded['cluster'].unique()):
    cluster_df = df_encoded[df_encoded['cluster'] == c]
    cluster_size = len(cluster_df)
    
    profile = {
        'cluster': int(c),
        'size': cluster_size,
        'pct': round(100 * cluster_size / len(df_encoded), 1)
    }
    
    # Only compute mean for numeric columns
    for col in numeric_cols:
        try:
            profile[f'mean_{col}'] = round(float(cluster_df[col].mean()), 2)
        except Exception:
            profile[f'mean_{col}'] = None
    
    cluster_profiles.append(profile)

# Create profiles dataframe
profiles_df = pd.DataFrame(cluster_profiles)
profiles_csv = os.path.join(OUTPUT_DIR, 'cluster_profiles.csv')
profiles_df.to_csv(profiles_csv, index=False)
print(f"[STATUS] Cluster profiles saved: {profiles_csv}")

# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================
print("\n[STATUS] ---- PHASE 6: Correlation Analysis ----")

correlation_matrix = feature_data.corr()

# Find strong correlations
strong_corrs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        val = correlation_matrix.iloc[i, j]
        if abs(val) > 0.5:
            strong_corrs.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': round(val, 3)
            })

print(f"[STATUS] Strong correlations found: {len(strong_corrs)}")

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='RdBu', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
corr_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
plt.savefig(corr_path)
plt.close()
print(f"[STATUS] Correlation heatmap saved: {corr_path}")

# ============================================================
# 7. ANOMALY PROFILING
# ============================================================
print("\n[STATUS] ---- PHASE 7: Anomaly Profiling ----")

anomaly_df = df_encoded[df_encoded['anomaly'] == -1]
normal_df = df_encoded[df_encoded['anomaly'] == 1]

# Compare means for numeric columns
anomaly_comparison = []
for col in numeric_cols:
    try:
        anomaly_mean = float(anomaly_df[col].mean())
        normal_mean = float(normal_df[col].mean())
        diff = anomaly_mean - normal_mean
        anomaly_comparison.append({
            'feature': col,
            'anomaly_mean': round(anomaly_mean, 2),
            'normal_mean': round(normal_mean, 2),
            'difference': round(diff, 2)
        })
    except Exception:
        pass

# ============================================================
# 8. BUSINESS PATTERN IDENTIFICATION
# ============================================================
print("\n[STATUS] ---- PHASE 8: Business Pattern Identification ----")

patterns = []
for i, row in profiles_df.iterrows():
    cluster_id = row['cluster']
    pct = row['pct']
    
    # Find distinguishing features
    distinct_features = []
    for col in numeric_cols[:5]:  # Top 5 features
        mean_col = f'mean_{col}'
        if mean_col in row.index and row[mean_col] is not None:
            overall_mean = float(feature_data[col].mean())
            if abs(row[mean_col] - overall_mean) > 0.5 * float(feature_data[col].std()):
                direction = "สูง" if row[mean_col] > overall_mean else "ต่ำ"
                distinct_features.append(f"{col} ({direction})")
    
    pattern = {
        'cluster': cluster_id,
        'size_pct': pct,
        'distinct_features': distinct_features[:3],
        'business_implication': f"Cluster {cluster_id} ({pct}% of data) — "
    }
    
    if target_cols:
        target_mean = float(df_encoded[df_encoded['cluster'] == cluster_id][target_cols[0]].mean())
        pattern['target_rate'] = round(target_mean, 3)
        pattern['business_implication'] += f"Target rate: {target_mean:.1%}"
    
    patterns.append(pattern)

# ============================================================
# 9. SAVE OUTPUTS
# ============================================================
print("\n[STATUS] ---- PHASE 9: Saving Outputs ----")

# Save cluster labels to original dataframe
df['cluster'] = cluster_labels
df['anomaly'] = ['Anomaly' if a == -1 else 'Normal' for a in anomaly_labels]

output_csv = os.path.join(OUTPUT_DIR, 'max_output.csv')
df.to_csv(output_csv, index=False)
print(f"[STATUS] Saved: {output_csv}")

# ============================================================
# 10. GENERATE REPORTS
# ============================================================
print("\n[STATUS] ---- PHASE 10: Generating Reports ----")

# Mining Results Report
mining_results = f"""# Max Data Mining Report

## Overview
- **Dataset**: {df.shape[0]} rows, {df.shape[1]} columns
- **Features Analyzed**: {len(numeric_cols)} features
- **Optimal Clusters**: {optimal_k}
- **Silhouette Score**: {sil_score:.3f}
- **Davies-Bouldin Score**: {db_score:.3f}

## Clustering Results

### Cluster Sizes
| Cluster | Size | Percentage |
|---------|------|------------|
"""
for p in patterns:
    mining_results += f"| {p['cluster']} | {p['size_pct']}% | {p['business_implication']} |\n"

mining_results += f"""
### Distinctive Features per Cluster
"""
for p in patterns:
    if p['distinct_features']:
        mining_results += f"- **Cluster {p['cluster']}**: {', '.join(p['distinct_features'])}\n"

mining_results += f"""
## Anomaly Detection
- **Anomalies Found**: {n_anomalies} ({100*n_anomalies/len(df_encoded):.1f}%)

### Top Anomaly Characteristics
"""
for comp in anomaly_comparison[:5]:
    mining_results += f"- **{comp['feature']}**: Anomaly mean = {comp['anomaly_mean']} vs Normal mean = {comp['normal_mean']} (Δ = {comp['difference']:+})\n"

mining_results += f"""
## Strong Correlations Found
"""
if strong_corrs:
    for sc in strong_corrs[:10]:
        mining_results += f"- **{sc['feature1']}** ↔ **{sc['feature2']}**: r = {sc['correlation']}\n"
else:
    mining_results += "- No strong correlations found (>0.5)\n"

mining_results += f"""
## Business Implications
1. **Cluster Interpretation**: The {optimal_k} clusters represent distinct employee segments with different characteristics.
2. **Anomaly Detection**: {n_anomalies} employees ({100*n_anomalies/len(df_encoded):.1f}%) identified as outliers may need special attention.
3. **Recommended Actions**:
   - Investigate anomaly patterns for potential attrition risks
   - Use cluster profiles to design targeted retention strategies
   - Monitor strong correlations for predictive modeling

## Visualizations
- Elbow Method: `{elbow_path}`
- PCA Projection: `{pca_path}`
- Correlation Heatmap: `{corr_path}`
"""

mining_path = os.path.join(OUTPUT_DIR, 'mining_results.md')
with open(mining_path, 'w', encoding='utf-8') as f:
    f.write(mining_results)
print(f"[STATUS] Saved: {mining_path}")

# Patterns Found Report
patterns_found = f"""# Patterns Found Report

## Pattern 1: Employee Clusters
- **Type**: Segmentation Pattern
- **Description**: Found {optimal_k} distinct employee clusters in the dataset
- **Evidence**: Silhouette Score = {sil_score:.3f}, Davies-Bouldin = {db_score:.3f}
- **Business Implication**: Each cluster represents a different employee profile requiring different management approaches

## Pattern 2: Anomaly Detection
- **Type**: Outlier Pattern
- **Description**: Identified {n_anomalies} employees ({100*n_anomalies/len(df_encoded):.1f}%) as anomalous
- **Evidence**: Isolation Forest with 5% contamination rate
- **Business Implication**: These employees may be at risk or have unusual patterns worth investigating

"""
if strong_corrs:
    patterns_found += f"""## Pattern 3: Feature Correlations
- **Type**: Relationship Pattern
- **Description**: Found {len(strong_corrs)} strong pairwise correlations
- **Evidence**: Correlation coefficients > 0.5
- **Business Implication**: These relationships can be leveraged for predictive modeling
"""

patterns_found += f"""
## Actionable Insights
1. Use cluster membership as a feature for attrition prediction
2. Flag anomalous employees for HR review
3. Monitor correlated features for early warning signs

## Data Quality Notes
- All numeric missing values filled with median
- Categorical variables label encoded
- Features with zero variance removed
"""

patterns_path = os.path.join(OUTPUT_DIR, 'patterns_found.md')
with open(patterns_path, 'w', encoding='utf-8') as f:
    f.write(patterns_found)
print(f"[STATUS] Saved: {patterns_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("[STATUS] DATA MINING COMPLETE")
print("="*60)
print(f"[STATUS] Files saved in: {OUTPUT_DIR}")
print(f"[STATUS] - max_output.csv")
print(f"[STATUS] - mining_results.md")
print(f"[STATUS] - patterns_found.md")
print(f"[STATUS] - cluster_profiles.csv")
print(f"[STATUS] - elbow.png")
print(f"[STATUS] - pca_clusters.png")
print(f"[STATUS] - correlation_heatmap.png")