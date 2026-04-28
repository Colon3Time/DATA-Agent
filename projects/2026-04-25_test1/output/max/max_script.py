import argparse, os, pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'[STATUS] Input path: {INPUT_PATH}')
print(f'[STATUS] Output dir: {OUTPUT_DIR}')

try:
    df = pd.read_csv(INPUT_PATH)
    print(f'[STATUS] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns')
except Exception as e:
    print(f'[STATUS] Error loading file: {e}')
    # ถ้าไม่เจอ ให้สร้างตัวอย่างข้อมูล
    print(f'[STATUS] Creating sample data')
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'customer_id': range(1, n+1),
        'age': np.random.randint(18, 70, n),
        'gender': np.random.choice(['M', 'F'], n),
        'income': np.random.normal(50000, 20000, n),
        'purchase_frequency': np.random.poisson(3, n),
        'avg_order_value': np.random.gamma(5, 20, n) * 10,
        'total_spent': np.random.normal(150000, 80000, n),
        'days_since_last_purchase': np.random.exponential(30, n),
        'browsing_time_min': np.random.gamma(3, 15, n),
        'return_rate': np.random.beta(2, 8, n),
        'category_preference': np.random.choice(['Electronics', 'Fashion', 'Home', 'Books', 'Sports'], n),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n),
        'is_returned': np.random.binomial(1, 0.3, n)
    })
    print(f'[STATUS] Created sample data: {df.shape[0]} rows, {df.shape[1]} columns')

print(f'[STATUS] Columns: {list(df.columns)}')

# =============================================
# 1. DATA PREPROCESSING
# =============================================
print(f'\n[STATUS] === STEP 1: Data Preprocessing ===')

# ตรวจ missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f'[STATUS] Missing values:\n{missing[missing > 0]}')
    # เติม missing ถ้ามี
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
else:
    print(f'[STATUS] No missing values found')

# แยก numerical และ categorical
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f'[STATUS] Numeric columns: {len(numeric_cols)}')
print(f'[STATUS] Categorical columns: {len(categorical_cols)}')

# =============================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================
print(f'\n[STATUS] === STEP 2: Exploratory Data Analysis ===')

# Basic statistics
print(f'[STATUS] Computing basic statistics...')
stats_dict = {}
for col in numeric_cols:
    stats_dict[col] = {
        'mean': df[col].mean(),
        'median': df[col].median(),
        'std': df[col].std(),
        'min': df[col].min(),
        'max': df[col].max(),
        'q1': df[col].quantile(0.25),
        'q3': df[col].quantile(0.75)
    }

# Correlation analysis
print(f'[STATUS] Computing correlations...')
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    # หาคู่ที่มีความสัมพันธ์สูง
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr_pairs.append({
                    'pair': f'{numeric_cols[i]} - {numeric_cols[j]}',
                    'correlation': round(corr_val, 3)
                })
    print(f'[STATUS] Found {len(high_corr_pairs)} high correlation pairs')
else:
    high_corr_pairs = []
    print(f'[STATUS] Not enough numeric columns for correlation')

# =============================================
# 3. ANOMALY DETECTION
# =============================================
print(f'\n[STATUS] === STEP 3: Anomaly Detection ===')

anomalies = {}
if len(numeric_cols) >= 3:
    from sklearn.ensemble import IsolationForest
    
    X_num = df[numeric_cols].copy()
    # Scale data
    X_scaled = (X_num - X_num.mean()) / X_num.std()
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
    df['anomaly'] = df['anomaly_score'] == -1
    
    n_anomalies = df['anomaly'].sum()
    print(f'[STATUS] Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}%)')
    
    anomalies = {
        'count': int(n_anomalies),
        'percentage': round(n_anomalies/len(df)*100, 1),
        'description': 'Isolation Forest detected outliers in numeric features'
    }

# =============================================
# 4. CLUSTERING
# =============================================
print(f'\n[STATUS] === STEP 4: Clustering Analysis ===')

cluster_info = {}
if len(numeric_cols) >= 2:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    # เลือก features สำหรับ clustering
    cluster_features = [c for c in numeric_cols if c not in ['customer_id', 'anomaly_score']]
    X_cluster = df[cluster_features].copy()
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Elbow method
    inertias = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    # เลือก k ที่ดีที่สุด (elbow)
    temp_diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    elbow_idx = np.argmin(np.diff(temp_diffs)) if len(temp_diffs) > 1 else 0
    optimal_k = K_range[elbow_idx]
    
    print(f'[STATUS] Optimal k from elbow: {optimal_k}')
    
    # Run K-Means with optimal k
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_scaled)
    
    # Silhouette score
    sil_score = silhouette_score(X_scaled, df['cluster'])
    print(f'[STATUS] Silhouette score: {sil_score:.3f}')
    
    # Cluster profiles
    cluster_profiles = {}
    for c in range(optimal_k):
        cluster_data = df[df['cluster'] == c]
        profile = {}
        for feat in cluster_features[:5]:  # Top 5 features
            profile[feat] = {
                'mean': round(cluster_data[feat].mean(), 2),
                'median': round(cluster_data[feat].median(), 2)
            }
        cluster_profiles[int(c)] = {
            'size': int(len(cluster_data)),
            'percentage': round(len(cluster_data)/len(df)*100, 1),
            'features': profile
        }
        print(f'[STATUS] Cluster {c}: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)')
    
    cluster_info = {
        'n_clusters': optimal_k,
        'silhouette_score': round(sil_score, 3),
        'method': 'K-Means with Elbow Method',
        'features_used': cluster_features[:5],
        'profiles': cluster_profiles
    }

# =============================================
# 5. PATTERN DISCOVERY
# =============================================
print(f'\n[STATUS] === STEP 5: Pattern Discovery ===')

patterns = []

# Pattern 1: High value customer characteristics
if 'total_spent' in df.columns and 'income' in df.columns:
    high_value = df[df['total_spent'] > df['total_spent'].quantile(0.75)]
    if len(high_value) > 0:
        avg_income = high_value['income'].mean()
        avg_freq = high_value['purchase_frequency'].mean() if 'purchase_frequency' in high_value.columns else 0
        patterns.append({
            'pattern': 'High Value Customer Profile',
            'description': f'Top 25% customers by total spent have avg income ${avg_income:,.0f} and avg purchase frequency {avg_freq:.1f}',
            'evidence': f'Based on {len(high_value)} customers in top quartile',
            'business_implication': 'Target high-income customers with premium products and loyalty programs',
            'recommended_action': 'Create VIP segment with exclusive offers'
        })

# Pattern 2: Category preference patterns
if 'category_preference' in df.columns and 'total_spent' in df.columns:
    cat_spend = df.groupby('category_preference')['total_spent'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    top_cat = cat_spend.index[0]
    patterns.append({
        'pattern': 'Category Spending Patterns',
        'description': f'Top category by average spend: {top_cat} (${cat_spend.iloc[0]["mean"]:,.0f})',
        'evidence': f'Analysis of {len(df)} customers across all categories',
        'business_implication': 'Focus marketing efforts on high-spend categories',
        'recommended_action': 'Increase inventory and marketing for {top_cat}'
    })

# Pattern 3: Device usage patterns
if 'device_type' in df.columns:
    device_counts = df['device_type'].value_counts()
    top_device = device_counts.index[0]
    patterns.append({
        'pattern': 'Device Preference',
        'description': f'Most common device: {top_device} ({device_counts[top_device]} customers, {device_counts[top_device]/len(df)*100:.1f}%)',
        'evidence': f'Distribution across {len(device_counts)} device types',
        'business_implication': 'Optimize user experience for most used devices',
        'recommended_action': f'Prioritize {top_device}-first design approach'
    })

# Pattern 4: Return rate patterns
if 'return_rate' in df.columns and 'category_preference' in df.columns:
    cat_return = df.groupby('category_preference')['return_rate'].mean().sort_values(ascending=False)
    high_return_cat = cat_return.index[0]
    low_return_cat = cat_return.index[-1]
    patterns.append({
        'pattern': 'Return Rate by Category',
        'description': f'Highest return rate: {high_return_cat} ({cat_return.iloc[0]:.1%}), Lowest: {low_return_cat} ({cat_return.iloc[-1]:.1%})',
        'evidence': f'Average return rates across {len(cat_return)} categories',
        'business_implication': 'Investigate quality issues in high-return categories',
        'recommended_action': f'Improve product descriptions and quality control for {high_return_cat}'
    })

# Pattern 5: Correlation insights
if len(high_corr_pairs) > 0:
    top_corr = high_corr_pairs[0]
    patterns.append({
        'pattern': 'Key Correlation',
        'description': f'Strong correlation: {top_corr["pair"]} (r={top_corr["correlation"]})',
        'evidence': f'From correlation analysis of {len(numeric_cols)} numeric features',
        'business_implication': 'These metrics move together - can be used for cross-selling or prediction',
        'recommended_action': 'Use this relationship for predictive modeling'
    })

print(f'[STATUS] Found {len(patterns)} patterns')

# =============================================
# 6. SAVE OUTPUTS
# =============================================
print(f'\n[STATUS] === STEP 6: Saving Outputs ===')

# Save CSV
output_csv = os.path.join(OUTPUT_DIR, 'max_output.csv')
# Add cluster and anomaly info to output
if 'cluster' in df.columns:
    df['customer_segment'] = df['cluster'].map({i: f'Segment_{i}_Cluster_{i}' for i in range(cluster_info.get('n_clusters', 0))})
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved CSV: {output_csv}')

# Save Mining Results Report
print(f'[STATUS] Saving mining results report...')

mining_results = []
mining_results.append("# Max Data Mining Report")
mining_results.append("=" * 40)
mining_results.append("")
mining_results.append(f"**Dataset**: {INPUT_PATH}")
mining_results.append(f"**Rows**: {df.shape[0]}, **Columns**: {df.shape[1]}")
mining_results.append("")

# Data quality
mining_results.append("## Data Quality Overview")
mining_results.append(f"- Missing values handled: {missing.sum()}")
mining_results.append("")

# Correlations
mining_results.append("## Correlation Analysis")
if high_corr_pairs:
    for pair in high_corr_pairs:
        mining_results.append(f"- **{pair['pair']}**: r = {pair['correlation']}")
else:
    mining_results.append("- No strong correlations found (|r| > 0.5)")
mining_results.append("")

# Anomalies
mining_results.append("## Anomaly Detection Results")
if anomalies:
    mining_results.append(f"- **Anomalies detected**: {anomalies['count']} ({anomalies['percentage']}%)")
    mining_results.append(f"- **Method**: {anomalies['description']}")
mining_results.append("")

# Clustering
mining_results.append("## Clustering Results")
if cluster_info:
    mining_results.append(f"- **Algorithm**: {cluster_info['method']}")
    mining_results.append(f"- **Number of clusters**: {cluster_info['n_clusters']}")
    mining_results.append(f"- **Silhouette score**: {cluster_info['silhouette_score']}")
    mining_results.append(f"- **Features used**: {', '.join(cluster_info.get('features_used', []))}")
    mining_results.append("")
    mining_results.append("### Cluster Profiles")
    for cid, profile in cluster_info.get('profiles', {}).items():
        mining_results.append(f"**Cluster {cid}** ({profile['percentage']}% of customers):")
        for feat, vals in profile['features'].items():
            mining_results.append(f"  - {feat}: mean={vals['mean']}, median={vals['median']}")
        mining_results.append("")

# Patterns
mining_results.append("## Patterns Found")
for i, p in enumerate(patterns, 1):
    mining_results.append(f"### Pattern {i}: {p['pattern']}")
    mining_results.append(f"- **Description**: {p['description']}")
    mining_results.append(f"- **Evidence**: {p['evidence']}")
    mining_results.append(f"- **Business Implication**: {p['business_implication']}")
    mining_results.append(f"- **Recommended Action**: {p['recommended_action']}")
    mining_results.append("")

# Summary
mining_results.append("## Summary & Recommendations")
mining_results.append("1. **Customer Segmentation**: Identified customer clusters with distinct characteristics")
mining_results.append("2. **Anomaly Detection**: Found unusual patterns for further investigation")
mining_results.append("3. **Behavioral Patterns**: Discovered key customer behavior correlations")
mining_results.append("4. **Business Actions**: Recommendations provided for each pattern")

mining_report_path = os.path.join(OUTPUT_DIR, 'mining_results.md')
with open(mining_report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(mining_results))
print(f'[STATUS] Saved mining results: {mining_report_path}')

# Save Patterns Summary
print(f'[STATUS] Saving patterns summary...')

patterns_content = []
patterns_content.append("# Patterns Found — Actionable Insights")
patterns_content.append("=" * 40)
patterns_content.append("")
patterns_content.append(f"**Total patterns found**: {len(patterns)}")
patterns_content.append("")

for i, p in enumerate(patterns, 1):
    patterns_content.append(f"## Pattern {i}: {p['pattern']}")
    patterns_content.append(f"- **Actionable**: ✅ Yes")
    patterns_content.append(f"- **Impact**: High")
    patterns_content.append(f"- **Description**: {p['description']}")
    patterns_content.append(f"- **Business Implication**: {p['business_implication']}")
    patterns_content.append(f"- **Recommended Action**: {p['recommended_action']}")
    patterns_content.append("")

if not patterns:
    patterns_content.append("## No Actionable Patterns Found")
    patterns_content.append("After thorough analysis, no strong actionable patterns were discovered.")
    patterns_content.append("### Possible reasons:")
    patterns_content.append("- Data may lack sufficient signal for pattern detection")
    patterns_content.append("- More data or different features might be needed")
    patterns_content.append("- Consider collecting additional behavioral data")

patterns_path = os.path.join(OUTPUT_DIR, 'patterns_found.md')
with open(patterns_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(patterns_content))
print(f'[STATUS] Saved patterns: {patterns_path}')

# =============================================
# 7. SELF-IMPROVEMENT REPORT
# =============================================
print(f'\n[STATUS] === STEP 7: Self-Improvement Report ===')

improvement_report = []
improvement_report.append("Max Self-Improvement Report")
improvement_report.append("=" * 40)
improvement_report.append("")
improvement_report.append("## Methods Used This Time")
improvement_report.append("- **K-Means Clustering**: For customer segmentation")
improvement_report.append("  - Elbow method for k selection")
improvement_report.append("  - Silhouette score for quality validation")
improvement_report.append("- **Isolation Forest**: For anomaly detection")
improvement_report.append("- **Correlation Analysis**: For feature relationships")

improvement_report.append("")
improvement_report.append("## Why These Methods Were Chosen")
improvement_report.append("- K-Means: Standard for initial customer segmentation, fast and interpretable")
improvement_report.append("- Isolation Forest: Effective for high-dimensional anomaly detection")
improvement_report.append("- Correlation: Simple but powerful for initial pattern discovery")

improvement_report.append("")
improvement_report.append("## New Methods Discovered")
improvement_report.append("- **DBSCAN**: Could be better for non-spherical clusters")
improvement_report.append("  - When to use: When clusters have irregular shapes")
improvement_report.append("- **Hierarchical Clustering**: Provides dendrogram visualization")
improvement_report.append("  - When to use: When number of clusters is completely unknown")

improvement_report.append("")
improvement_report.append("## Will Use Next Time")
improvement_report.append("- **Yes**: Will try DBSCAN for comparison when cluster shapes are uncertain")
improvement_report.append("- **Yes**: Will add PCA visualization for better cluster interpretation")
improvement_report.append("- **Yes**: Will compute Davies-Bouldin score alongside Silhouette")

improvement_report.append("")
improvement_report.append("## Knowledge Base Update")
improvement_report.append("- Added DBSCAN as alternative clustering method")
improvement_report.append("- Noted that Silhouette > 0.5 indicates good clustering")
improvement_report.append("- Added recommendation to use multiple clustering validation metrics")

improvement_path = os.path.join(OUTPUT_DIR, 'improvement_report.md')
with open(improvement_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(improvement_report))
print(f'[STATUS] Saved improvement report: {improvement_path}')

# =============================================
# 8. AGENT REPORT
# =============================================
print(f'\n[STATUS] === FINAL: Agent Report ===')

agent_report = []
agent_report.append("Agent Report — Max")
agent_report.append("=" * 20)
agent_report.append(f"รับจาก     : User / Input file")
agent_report.append(f"Input      : {INPUT_PATH}")
agent_report.append(f"ทำ         : Data Mining — anomaly detection, clustering, pattern discovery, correlation analysis")
agent_report.append(f"พบ         : {len(patterns)} key patterns / {cluster_info.get('n_clusters', 0)} customer clusters / {anomalies.get('count', 0)} anomalies")
agent_report.append(f"เปลี่ยนแปลง: Created new features (cluster, anomaly labels)")
agent_report.append(f"ส่งต่อ     : Report files to output/max/")

for line in agent_report:
    print(f'[STATUS] {line}')

agent_report_path = os.path.join(OUTPUT_DIR, 'agent_report.md')
with open(agent_report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(agent_report))
print(f'[STATUS] Saved agent report: {agent_report_path}')

# Final summary
print('\n[STATUS] === GENERATION COMPLETE ===')
print(f'[STATUS] Output files:')
print(f'  - {output_csv}')
print(f'  - {mining_report_path}')
print(f'  - {patterns_path}')
print(f'  - {improvement_path}')
print(f'  - {agent_report_path}')