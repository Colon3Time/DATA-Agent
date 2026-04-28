import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# รับ Argument
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--input',      default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH  = args.input
OUTPUT_DIR  = Path(args.output_dir) if args.output_dir else Path('.')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# โหลดข้อมูล
# ============================================================
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ============================================================
# 1. Data Profiling & Cleaning
# ============================================================
print('\n=== DATA PROFILING ===')

# Drop duplicated columns
df = df.loc[:, ~df.columns.duplicated()]

# แยก column type
numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'[STATUS] Numeric: {numeric_cols}')
print(f'[STATUS] Categorical: {categorical_cols}')

# Handle missing values
numeric_missing = df[numeric_cols].isnull().sum()
if numeric_missing.sum() > 0:
    print(f'[STATUS] Missing numeric values: {dict(numeric_missing[numeric_missing>0])}')
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

# ============================================================
# 2. Outlier Detection (Isolation Forest)
# ============================================================
print('\n=== ANOMALY DETECTION ===')
from sklearn.ensemble import IsolationForest

if len(numeric_cols) >= 2:
    X_num = df[numeric_cols].copy()
    
    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    
    df['is_anomaly'] = iso.fit_predict(X_scaled)
    df['is_anomaly'] = df['is_anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    df['anomaly_scores'] = iso.score_samples(X_scaled)
    
    n_anomalies = (df['is_anomaly'] == 'Anomaly').sum()
    print(f'[STATUS] Anomalies found: {n_anomalies} / {len(df)} ({100*n_anomalies/len(df):.1f}%)')
    
    top_anomalies = df.nsmallest(10, 'anomaly_scores')[numeric_cols + ['is_anomaly', 'anomaly_scores']]
    print('[STATUS] Top anomalies:')
    print(top_anomalies.head(5).to_string())
else:
    print('[STATUS] Not enough numeric columns for anomaly detection')
    df['is_anomaly'] = 'Normal'
    df['anomaly_scores'] = 0

# ============================================================
# 3. Correlation Analysis
# ============================================================
print('\n=== CORRELATION ANALYSIS ===')
if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr()
    
    # หาคู่ที่มี correlation สูง
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.5:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
    
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f'[STATUS] High correlation pairs (|r| > 0.5):')
    for pair in high_corr_pairs[:10]:
        print(f'  {pair[0]} vs {pair[1]}: r = {pair[2]:.3f}')
else:
    print('[STATUS] Not enough numeric columns for correlation analysis')
    high_corr_pairs = []

# ============================================================
# 4. Clustering (K-Means with Elbow)
# ============================================================
print('\n=== CLUSTERING ===')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

if len(numeric_cols) >= 2:
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
    
    # Elbow Method
    inertias = []
    K_range = range(2, min(11, len(df)))
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    # หา optimal k ด้วย elbow (difference method)
    if len(inertias) >= 3:
        diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        diff_diffs = [abs(diffs[i] - diffs[i+1]) for i in range(len(diffs)-1)]
        optimal_k = K_range[diff_diffs.index(max(diff_diffs)) + 1] if diff_diffs else K_range[0]
    else:
        optimal_k = K_range[0]
    
    print(f'[STATUS] Optimal K (elbow): {optimal_k}')
    
    # Fit K-Means
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = km_final.fit_predict(X_scaled)
    
    # Evaluate
    sil_score = silhouette_score(X_scaled, df['cluster'])
    db_score = davies_bouldin_score(X_scaled, df['cluster'])
    print(f'[STATUS] Silhouette Score: {sil_score:.3f}, Davies-Bouldin: {db_score:.3f}')
    
    # Cluster profiles
    cluster_profiles = df.groupby('cluster')[numeric_cols].mean()
    cluster_counts = df['cluster'].value_counts().sort_index()
    print(f'[STATUS] Cluster sizes: {dict(cluster_counts)}')
    print('[STATUS] Cluster profiles (mean):')
    for c in range(optimal_k):
        print(f'  Cluster {c} ({cluster_counts[c]} rows):')
        top_features = cluster_profiles.loc[c].sort_values(ascending=False).head(5)
        for feat, val in top_features.items():
            print(f'    {feat}: {val:.2f}')
else:
    print('[STATUS] Not enough numeric columns for clustering')
    df['cluster'] = 0

# ============================================================
# 5. Pattern Detection — Categorical Relationships
# ============================================================
print('\n=== PATTERN DETECTION ===')
patterns = []

# 5a. Time-based patterns (if datetime columns exist)
date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'day' in c.lower()]
print(f'[STATUS] Date/time columns found: {date_cols}')

# 5b. Categorical vs numeric analysis
if len(categorical_cols) > 0 and len(numeric_cols) > 0:
    for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical cols
        if df[cat_col].nunique() <= 20 and df[cat_col].nunique() > 1:
            cat_stats = df.groupby(cat_col)[numeric_cols].mean()
            print(f'[STATUS] Pattern found: {cat_col} vs numeric features')
            print(cat_stats.round(2).to_string())

# 5c. Detect patterns between categorical columns
if len(categorical_cols) >= 2:
    for i in range(len(categorical_cols)):
        for j in range(i+1, len(categorical_cols)):
            col1, col2 = categorical_cols[i], categorical_cols[j]
            if df[col1].nunique() <= 10 and df[col2].nunique() <= 10:
                crosstab = pd.crosstab(df[col1], df[col2], normalize='index')
                print(f'[STATUS] Cross-tab: {col1} vs {col2}')
                print(crosstab.round(3).to_string())

# ============================================================
# 6. Association Rule Mining (if applicable)
# ============================================================
print('\n=== ASSOCIATION RULES ===')
basket_cols = [c for c in df.columns if 'item' in c.lower() or 'product' in c.lower() or 'category' in c.lower() or 'basket' in c.lower()]

if len(basket_cols) >= 1:
    print(f'[STATUS] Potential basket columns: {basket_cols}')
    # Check if data has transaction-like structure
    for col in basket_cols:
        unique_vals = df[col].nunique()
        print(f'  {col}: {unique_vals} unique values')
else:
    print('[STATUS] No basket/transaction columns found')

# ============================================================
# 7. Sequential Pattern Mining (if timestamp exists)
# ============================================================
timestamp_cols = [c for c in df.columns if 'timestamp' in c.lower() or 'datetime' in c.lower()]
if timestamp_cols:
    print(f'[STATUS] Timestamp columns: {timestamp_cols}')
    try:
        df[timestamp_cols[0]] = pd.to_datetime(df[timestamp_cols[0]])
        # Extract time features
        df['hour'] = df[timestamp_cols[0]].dt.hour
        df['dayofweek'] = df[timestamp_cols[0]].dt.dayofweek
        print('[STATUS] Time features extracted')
    except:
        print('[STATUS] Could not parse timestamp')

# ============================================================
# 8. Generate Reports
# ============================================================
print('\n=== GENERATING REPORTS ===')

# 8a. patterns_found.md
patterns_file = OUTPUT_DIR / 'patterns_found.md'
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

with open(patterns_file, 'w', encoding='utf-8') as f:
    f.write('# Patterns Found\n\n')
    f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    
    f.write('## Dataset Overview\n\n')
    f.write(f'- Rows: {len(df)}\n')
    f.write(f'- Columns: {len(df.columns)}\n')
    f.write(f'- Numeric features: {len(numeric_cols)}\n')
    f.write(f'- Categorical features: {len(categorical_cols)}\n\n')
    
    f.write('## Anomalies Detected\n\n')
    if 'is_anomaly' in df.columns:
        n_anom = (df['is_anomaly'] == 'Anomaly').sum()
        f.write(f'- **Anomalies count**: {n_anom} ({100*n_anom/len(df):.1f}%)\n')
        if n_anom > 0:
            f.write('- **Top anomalous rows**:\n')
            top_anom_rows = df[df['is_anomaly'] == 'Anomaly'].nsmallest(5, 'anomaly_scores')
            f.write(f'  {top_anom_rows[numeric_cols[:5]].to_string()}\n')
    f.write('\n')
    
    f.write('## Correlations\n\n')
    if high_corr_pairs:
        for pair in high_corr_pairs[:5]:
            f.write(f'- **{pair[0]}** ↔ **{pair[1]}**: r = {pair[2]:.3f}\n')
    else:
        f.write('- No strong correlations found\n')
    f.write('\n')
    
    f.write('## Clusters\n\n')
    if 'cluster' in df.columns:
        f.write(f'- Optimal clusters: {optimal_k}\n')
        f.write(f'- Silhouette Score: {sil_score:.3f}\n')
        f.write(f'- Davies-Bouldin Index: {db_score:.3f}\n\n')
        for c in range(optimal_k):
            f.write(f'### Cluster {c} ({cluster_counts[c]} rows)\n')
            f.write(f'- Key features:\n')
            top_feats = cluster_profiles.loc[c].sort_values(ascending=False).head(3)
            for feat, val in top_feats.items():
                f.write(f'  - {feat}: {val:.2f}\n')
            f.write('\n')
    
    f.write('## Key Patterns\n\n')
    f.write('### Pattern 1: Data Distribution\n\n')
    f.write(f'- The dataset contains {len(df)} records with {len(numeric_cols)} numeric features\n')
    for col in numeric_cols[:3]:
        f.write(f'- **{col}**: mean={df[col].mean():.2f}, std={df[col].std():.2f}\n')
    f.write('\n')
    
    f.write('### Pattern 2: Anomaly Insights\n\n')
    if 'is_anomaly' in df.columns and n_anom > 0:
        anom_df = df[df['is_anomaly'] == 'Anomaly']
        for col in numeric_cols[:3]:
            anom_mean = anom_df[col].mean()
            normal_mean = df[df['is_anomaly'] == 'Normal'][col].mean()
            diff_pct = ((anom_mean - normal_mean) / normal_mean) * 100 if normal_mean != 0 else 0
            f.write(f'- **{col}**: Anomaly mean={anom_mean:.2f} vs Normal mean={normal_mean:.2f} ({diff_pct:+.1f}% difference)\n')
    f.write('\n')
    
    f.write('## Business Implications\n\n')
    f.write('1. **Anomaly Patterns**: Identified outliers that may represent fraud, errors, or rare events\n')
    f.write('2. **Cluster Segments**: Natural groupings found in data using K-Means clustering\n')
    f.write('3. **Correlation Insights**: Feature relationships that can inform feature engineering\n\n')
    
    f.write('## Recommended Actions\n\n')
    f.write('1. Investigate anomaly rows for potential data quality issues\n')
    f.write('2. Use cluster segments for targeted analysis or personalization\n')
    f.write('3. Consider correlation pairs for dimensionality reduction\n')

print(f'[STATUS] Saved: {patterns_file}')

# 8b. mining_results.md
mining_file = OUTPUT_DIR / 'mining_results.md'
with open(mining_file, 'w', encoding='utf-8') as f:
    f.write('# Max Data Mining Report\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    
    f.write('## Techniques Used\n\n')
    f.write('- Isolation Forest (Anomaly Detection)\n')
    f.write('- Pearson Correlation (Feature Relationships)\n')
    f.write('- K-Means Clustering (Segmentation)\n')
    f.write('- Elbow Method (Optimal Cluster Selection)\n')
    f.write('- Data Profiling & Statistical Analysis\n\n')
    
    f.write('## Patterns Found\n\n')
    
    f.write('### Pattern 1: Anomaly Segments\n\n')
    if 'is_anomaly' in df.columns:
        n_anom = (df['is_anomaly'] == 'Anomaly').sum()
        f.write(f'- **Evidence**: {n_anom} rows ({100*n_anom/len(df):.1f}%) identified as anomalies\n')
        f.write(f'- **Detection Method**: Isolation Forest (contamination=0.05)\n')
        f.write(f'- **Importance**: These rows deviate significantly from normal patterns\n\n')
    else:
        f.write('- **Evidence**: No anomalies detected\n')
        f.write('- **Importance**: Data appears homogeneous\n\n')
    
    f.write('### Pattern 2: Feature Clusters\n\n')
    if 'cluster' in df.columns:
        f.write(f'- **Evidence**: {optimal_k} distinct clusters found\n')
        f.write(f'- **Quality**: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}\n')
        f.write(f'- **Importance**: Data naturally groups into {optimal_k} segments\n\n')
    
    f.write('### Pattern 3: Correlation Patterns\n\n')
    if high_corr_pairs:
        f.write(f'- **Evidence**: {len(high_corr_pairs)} significant correlation pairs found\n')
        for pair in high_corr_pairs[:3]:
            f.write(f'  - {pair[0]} ↔ {pair[1]}: r={pair[2]:.3f}\n')
    else:
        f.write('- **Evidence**: No strong linear correlations\n')
    f.write(f'- **Importance**: Indicates interdependencies between features\n\n')
    
    f.write('## Anomalies Detected\n\n')
    if 'is_anomaly' in df.columns:
        f.write(f'- Total anomalies: {n_anom}\n')
        f.write(f'- Threshold: Top 5% most anomalous\n')
    f.write('\n')
    
    f.write('## Clusters Found\n\n')
    if 'cluster' in df.columns:
        for c in range(optimal_k):
            f.write(f'### Cluster {c} (n={cluster_counts[c]})\n')
            f.write(f'- **Characteristics**: ')
            top_3 = cluster_profiles.loc[c].sort_values(ascending=False).head(3)
            chars = ', '.join([f'{feat}={val:.2f}' for feat, val in top_3.items()])
            f.write(f'{chars}\n')
            f.write(f'- **Size**: {100*cluster_counts[c]/len(df):.1f}% of data\n\n')
    
    f.write('## Business Implication\n\n')
    f.write('### What These Patterns Mean\n\n')
    f.write('1. **Customer/Data Segments**: The clusters represent distinct groups that may require different strategies\n')
    f.write('2. **Risk/Anomaly Detection**: Flagged anomalies warrant investigation for data quality or fraud\n')
    f.write('3. **Feature Relationships**: Correlations can guide feature selection and engineering\n\n')
    
    if high_corr_pairs:
        f.write('### Key Relationships\n\n')
        for pair in high_corr_pairs[:3]:
            direction = 'positive' if pair[2] > 0 else 'negative'
            f.write(f'- **{pair[0]}** and **{pair[1]}** have a {direction} correlation ({pair[2]:.2f})\n')
        f.write('\n')

print(f'[STATUS] Saved: {mining_file}')

# ============================================================
# 9. Self-Improvement Report
# ============================================================
improvement_file = OUTPUT_DIR / 'improvement_report.md'
with open(improvement_file, 'w', encoding='utf-8') as f:
    f.write('# Self-Improvement Report\n\n')
    f.write(f'**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    
    f.write('## Methods Used\n\n')
    f.write('1. **Isolation Forest** — Anomaly detection on all numeric features\n')
    f.write('2. **K-Means with Elbow** — Automatic optimal cluster selection\n')
    f.write('3. **Correlation Matrix** — Pearson correlation for feature relationships\n')
    f.write('4. **Statistical Profiling** — Mean, std, distribution analysis\n\n')
    
    f.write('## Selection Rationale\n\n')
    f.write(f'- Isolation Forest: Handles high-dimensional data well, contamination={0.05}\n')
    f.write('- K-Means: Scalable, interpretable clusters with elbow method\n')
    f.write('- Correlation: Simple and effective for initial feature analysis\n\n')
    
    f.write('## New Methods Discovered\n\n')
    f.write('- None new — all methods from knowledge base applied\n\n')
    
    f.write('## Knowledge Base Update\n\n')
    f.write('- No changes needed — current methods sufficient\n')

print(f'[STATUS] Saved: {improvement_file}')

# ============================================================
# 10. Save output CSV
# ============================================================
output_csv = OUTPUT_DIR / 'max_output.csv'
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# ============================================================
# SUMMARY
# ============================================================
print('\n' + '='*60)
print('MAX DATA MINING COMPLETE')
print('='*60)
print(f'Input: {INPUT_PATH}')
print(f'Outputs in: {OUTPUT_DIR}')
print(f'  - max_output.csv')
print(f'  - patterns_found.md')
print(f'  - mining_results.md')
print(f'  - improvement_report.md')
print(f'Clusters: {optimal_k if "cluster" in df.columns and "optimal_k" in dir() else "N/A"}')
print(f'Anomalies: {n_anom if "n_anom" in dir() else "N/A"}')
print('='*60)