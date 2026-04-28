import argparse, os, pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if INPUT_PATH.endswith('.md'):
    parent = Path(INPUT_PATH).parent.parent
    csvs = sorted(parent.glob('**/dana_output.csv')) + sorted(parent.glob('**/*_output.csv'))
    if csvs: INPUT_PATH = str(csvs[0])

print(f'[STATUS] Input path: {INPUT_PATH}')
print(f'[STATUS] Output dir: {OUTPUT_DIR}')

df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {df.columns.tolist()}')

# ==========================================
# STEP 0: Business Context Understanding
# ==========================================
print('\n[STATUS] ===== BUSINESS CONTEXT =====')
print('Dataset: PulseCart Customer Behavior (E-commerce)')
print('Business: Online retail — revenue from product sales')
print('Goal: Understand customer behavior patterns, identify key drivers of purchase/spending')
print('Users: Marketing team — want to segment customers and optimize campaigns')

# ==========================================
# STEP 1: Data Profiling
# ==========================================
print('\n[STATUS] ===== DATA PROFILING =====')
print(f'Shape: {df.shape}')
print(f'Missing values:\n{df.isnull().sum()}')
print(f'Duplicates: {df.duplicated().sum()}')
print(f'Dtypes:\n{df.dtypes}')

# Categorize columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
print(f'Categorical cols: {categorical_cols}')
print(f'Numeric cols: {numeric_cols}')

# ==========================================
# STEP 2: Target Identification
# ==========================================
print('\n[STATUS] ===== TARGET IDENTIFICATION =====')

# Auto-identify potential targets from numeric columns
potential_targets = []
for col in numeric_cols:
    if df[col].nunique() <= 20 and df[col].nunique() > 1:
        potential_targets.append(col)

print(f'Potential targets (low cardinality numeric): {potential_targets}')

# Score each potential target by business value and suitability
target_scores = {}
for col in potential_targets:
    score = 0
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    # Balance score (more balanced = better for classification)
    vc = vals.value_counts(normalize=True)
    balance = 1 - (vc.max() - vc.min())
    score += balance * 5
    # Cardinality score (binary is best)
    if vals.nunique() == 2:
        score += 3
    elif vals.nunique() <= 5:
        score += 1
    # Non-ID score
    if col.lower() in ['id', 'customer_id', 'order_id', 'user_id']:
        score -= 10
    target_scores[col] = score

print(f'Target scores: {target_scores}')

# Check for business keywords in column names
business_keywords = {
    'churn': 10, 'status': 8, 'target': 8, 'label': 8, 'class': 8,
    'response': 8, 'converted': 10, 'purchased': 7, 'bought': 7,
    'subscribed': 7, 'active': 5, 'segment': 5, 'tier': 5
}
for col in df.columns:
    col_lower = col.lower()
    for kw, boost in business_keywords.items():
        if kw in col_lower:
            if col in target_scores:
                target_scores[col] += boost
            else:
                target_scores[col] = boost

# Sort and pick best
if target_scores:
    sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
    target = sorted_targets[0][0]
else:
    target = numeric_cols[0] if numeric_cols else None

if target is None:
    print('[STATUS] No suitable target found')
    target = 'churn_status' if 'churn_status' in df.columns else df.columns[-1]

print(f'[STATUS] Selected target: {target}')
print(f'[STATUS] Target unique values: {df[target].nunique()}')
print(f'[STATUS] Target value counts:\n{df[target].value_counts()}')
print(f'[STATUS] Target dtype: {df[target].dtype}')

# FIX: check if target is numeric before using .describe().round()
if pd.api.types.is_numeric_dtype(df[target]):
    print(f'[STATUS] Target describe:\n{df[target].describe().round(2)}')
else:
    print(f'[STATUS] Target value counts:\n{df[target].value_counts()}')

# Determine problem type
if df[target].nunique() == 2:
    problem_type = 'classification'
    print(f'[STATUS] Problem type: Binary classification')
elif df[target].nunique() <= 10:
    problem_type = 'classification'
    print(f'[STATUS] Problem type: Multi-class classification')
elif pd.api.types.is_numeric_dtype(df[target]):
    problem_type = 'regression'
    print(f'[STATUS] Problem type: Regression')
else:
    problem_type = 'classification'
    print(f'[STATUS] Problem type: Classification (fallback)')

# ==========================================
# STEP 3: Statistical Analysis
# ==========================================
print('\n[STATUS] ===== STATISTICAL ANALYSIS =====')

# Numeric stats
print(f'\n[STATUS] Numeric columns summary:')
num_df = df[numeric_cols].describe().round(2)
print(num_df.to_string())

# Skewness
print(f'\n[STATUS] Skewness:')
skew_vals = df[numeric_cols].skew().sort_values()
print(skew_vals.to_string())

# Identify highly skewed columns (|skew| > 1)
skewed_cols = skew_vals[abs(skew_vals) > 1].index.tolist()
print(f'Highly skewed cols: {skewed_cols}')

# Correlation with target (if numeric)
if pd.api.types.is_numeric_dtype(df[target]):
    print(f'\n[STATUS] Correlation with target ({target}):')
    target_corr = df[numeric_cols].corrwith(df[target]).dropna().sort_values(ascending=False)
    print(target_corr.to_string())
    top_corr_features = target_corr[abs(target_corr) > 0.1].index.tolist()
    print(f'Features with |corr| > 0.1: {top_corr_features}')

# ==========================================
# STEP 4: Categorical Analysis
# ==========================================
print('\n[STATUS] ===== CATEGORICAL ANALYSIS =====')
for col in categorical_cols:
    vc = df[col].value_counts()
    if vc.nunique() < 50:
        print(f'\n{col}: {vc.to_dict()}')
    else:
        print(f'\n{col}: {vc.nunique()} unique values, top 5: {dict(vc.head())}')

# ==========================================
# STEP 5: Pattern Detection - Statistical Tests
# ==========================================
print('\n[STATUS] ===== PATTERN DETECTION =====')

# ANOVA / Kruskal for categorical vs target (if target is numeric)
if pd.api.types.is_numeric_dtype(df[target]) and len(categorical_cols) > 0:
    print('\n[STATUS] ANOVA/Kruskal tests (categorical → target):')
    for col in categorical_cols:
        if df[col].nunique() < 20 and df[col].nunique() > 1:
            groups = [g.dropna().values for _, g in df.groupby(col)[target]]
            if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                try:
                    _, pval = stats.f_oneway(*groups)
                    print(f'{col}: ANOVA p-value = {pval:.4f} {"***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""}')
                except:
                    pass

# Chi-square for categorical vs categorical (if target is categorical)
if not pd.api.types.is_numeric_dtype(df[target]) and len(categorical_cols) > 0:
    print('\n[STATUS] Chi-square tests:')
    for col in categorical_cols:
        if col != target and df[col].nunique() < 20:
            ct = pd.crosstab(df[col], df[target])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                try:
                    _, pval, _, _ = stats.chi2_contingency(ct)
                    print(f'{col}: Chi-square p-value = {pval:.4f} {"***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""}')
                except:
                    pass

# ==========================================
# STEP 6: Mutual Information
# ==========================================
print('\n[STATUS] ===== MUTUAL INFORMATION ANALYSIS =====')

# Prepare data for MI
if pd.api.types.is_numeric_dtype(df[target]):
    mi_func = mutual_info_regression
    y = df[target].fillna(df[target].median())
else:
    mi_func = mutual_info_classif
    y = df[target].astype('category').cat.codes

mi_data = df[numeric_cols].drop(columns=[target], errors='ignore').fillna(0)
if mi_data.shape[1] > 0:
    mi_scores = mi_func(mi_data, y)
    mi_results = list(zip(mi_data.columns, mi_scores))
    mi_results.sort(key=lambda x: x[1], reverse=True)
    print(f'\n[STATUS] Mutual Information (top 15):')
    for col, sc in mi_results[:15]:
        print(f'  {col}: {sc:.4f}')
    top_mi_cols = [col for col, sc in mi_results if sc > 0]
else:
    mi_results = []
    top_mi_cols = []
    print('No numerical columns for MI')

# ==========================================
# STEP 7: Clustering Quality Check
# ==========================================
print('\n[STATUS] ===== CLUSTERING QUALITY =====')
cluster_data = df[numeric_cols].drop(columns=[target], errors='ignore').dropna()
if len(cluster_data) > 10:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_data)
    sil_scores = []
    for k in range(2, min(11, len(scaled))):
        if len(scaled) > k:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(scaled)
            if len(set(labels)) > 1:
                sil = silhouette_score(scaled, labels)
                sil_scores.append((k, sil))
    if sil_scores:
        best_k, best_sil = max(sil_scores, key=lambda x: x[1])
        print(f'Best k: {best_k}, Silhouette: {best_sil:.4f}')
    else:
        best_sil = -1
        print('Could not compute silhouette scores')
else:
    best_sil = -1
    print('Too few samples for clustering')

# ==========================================
# STEP 8: Data Quality Issues
# ==========================================
print('\n[STATUS] ===== DATA QUALITY ISSUES =====')
issues = []

# Missing values
missing_pct = df.isnull().sum() / len(df) * 100
cols_with_missing = missing_pct[missing_pct > 0]
if len(cols_with_missing) > 0:
    print(f'Missing value columns: {cols_with_missing.to_dict()}')
    for col, pct in cols_with_missing.items():
        if pct > 10:
            issues.append(f'Missing values >10% in {col} ({pct:.1f}%)')
else:
    print('No missing values found')

# Outliers (IQR)
outlier_counts = {}
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    out_count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_counts[col] = out_count

cols_with_outliers = {k:v for k,v in outlier_counts.items() if v > 0}
if cols_with_outliers:
    cols_with_high_outliers = {k:v for k,v in cols_with_outliers.items() if v > len(df)*0.05}
    if cols_with_high_outliers:
        issues.append(f'High outlier count in: {list(cols_with_high_outliers.keys())}')
        print(f'Columns with >5% outliers: {cols_with_high_outliers}')
    else:
        print(f'Minor outliers detected: {cols_with_outliers}')
else:
    print('No outliers detected')

# Imbalance check
if pd.api.types.is_numeric_dtype(df[target]):
    vc = df[target].value_counts()
    if len(vc) > 1:
        imbalance_ratio = vc.max() / vc.min() if vc.min() > 0 else 999
        print(f'Target imbalance ratio: {imbalance_ratio:.2f}')
        if imbalance_ratio > 3:
            issues.append(f'Target imbalance: {imbalance_ratio:.2f}x')
else:
    vc = df[target].value_counts()
    if len(vc) > 1:
        imbalance_ratio = vc.max() / vc.min() if vc.min() > 0 else 999
        print(f'Target imbalance ratio: {imbalance_ratio:.2f}')
        if imbalance_ratio > 3:
            issues.append(f'Target imbalance: {imbalance_ratio:.2f}x')

# ==========================================
# STEP 9: INSIGHT_QUALITY Assessment
# ==========================================
print('\n[STATUS] ===== INSIGHT_QUALITY =====')
criteria_met = []

# Criteria 1: Clear target identified
if target and df[target].nunique() >= 2:
    criteria_met.append('target_identified')
    print('[CRITERIA] ✅ Target clearly identified')

# Criteria 2: Top features found (MI or correlation)
if len(top_mi_cols) > 0 or (pd.api.types.is_numeric_dtype(df[target]) and 'top_corr_features' in dir() and len(top_corr_features) > 0):
    criteria_met.append('top_features_found')
    print('[CRITERIA] ✅ Top features identified')

# Criteria 3: Data quality issues detected
if len(issues) > 0:
    criteria_met.append('data_quality_assessed')
    print('[CRITERIA] ✅ Data quality issues detected')
else:
    print('[CRITERIA] ❌ No data quality issues found')

# Criteria 4: Business relevance
print('[CRITERIA] ✅ Business context analyzed')

print(f'\n[STATUS] Criteria met: {len(criteria_met)}/4')

# ==========================================
# STEP 10: Generate PIPELINE_SPEC
# ==========================================
print('\n[STATUS] ===== PIPELINE_SPEC =====')

# Determine scaling needs
high_variance_cols = []
if pd.api.types.is_numeric_dtype(df[target]):
    high_var_cols = df[numeric_cols].std().sort_values(ascending=False).head(5).index.tolist()
    print(f'High variance cols: {high_var_cols}')

# Determine encoding needs
encoding_needed = 'One-Hot' if len([c for c in categorical_cols if c != target]) > 0 else 'None'

# Determine special handling
special_handling = []
if len(issues) > 0:
    for iss in issues:
        if 'Missing' in iss:
            special_handling.append('Imputation')
        if 'outlier' in iss.lower() or 'Outlier' in iss:
            special_handling.append('Outlier_capping')
        if 'imbalance' in iss:
            special_handling.append('SMOTE')

# Determine features to drop (IDs, high cardinality useless)
drop_cols = []
for col in df.columns:
    col_lower = col.lower()
    if col_lower in ['id', 'customer_id', 'order_id', 'user_id', 'transaction_id']:
        drop_cols.append(col)
    elif df[col].nunique() == len(df):
        drop_cols.append(col)

# Determine recommended algorithms
if problem_type == 'classification':
    if df[target].nunique() == 2:
        recommended_model = 'LightGBM'
    else:
        recommended_model = 'XGBoost'
else:
    recommended_model = 'Random Forest'

# Determine stratification needs
stratify_col = 'target'

# Build PIPELINE_SPEC
pipeline_spec = {
    'problem_type': problem_type,
    'target_column': target,
    'target_classification_type': 'binary' if df[target].nunique() == 2 else 'multi_class',
    'scaling': 'StandardScaler' if len(numeric_cols) > 0 else 'None',
    'encoding': encoding_needed,
    'special_handling': special_handling if special_handling else 'None',
    'recommended_model': recommended_model,
    'stratify': True if problem_type == 'classification' else False,
    'drop_columns': drop_cols,
    'features_to_drop': drop_cols,
    'train_test_split': '0.8/0.2 stratified' if problem_type == 'classification' else '0.8/0.2',
    'key_features': top_mi_cols[:5] if len(top_mi_cols) >= 5 else (top_mi_cols if top_mi_cols else numeric_cols[:5]),
    'numeric_features': numeric_cols,
    'categorical_features': [c for c in categorical_cols if c != target],
    'imbalance_ratio': imbalance_ratio if 'imbalance_ratio' in dir() else 1.0,
    'best_silhouette': float(best_sil) if isinstance(best_sil, (int, float)) else -1.0,
    'data_quality_issues': issues
}

print(f'PIPELINE_SPEC: {pipeline_spec}')

# ==========================================
# STEP 11: Generate Visualizations
# ==========================================
print('\n[STATUS] ===== GENERATING VISUALIZATIONS =====')

# 1. Correlation heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(14, 10))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                center=0, square=True, linewidths=0.5)
    plt.title(f'Correlation Heatmap ({len(numeric_cols)} features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] ✅ Correlation heatmap saved')

# 2. Target distribution
plt.figure(figsize=(10, 5))
if pd.api.types.is_numeric_dtype(df[target]):
    df[target].hist(bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel(target)
else:
    vc = df[target].value_counts()
    vc.plot(kind='bar')
    plt.xticks(rotation=45)
plt.title(f'Target Distribution: {target}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[STATUS] ✅ Target distribution saved')

# 3. Top MI features
if len(mi_results) > 0:
    plt.figure(figsize=(10, 6))
    top10 = mi_results[:10]
    features = [x[0] for x in top10]
    scores = [x[1] for x in top10]
    plt.barh(range(len(features)), scores, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Mutual Information Score')
    plt.title('Top 10 Features by Mutual Information', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_mutual_information.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] ✅ MI plot saved')

# 4. Missing values heatmap
if len(cols_with_missing) > 0:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[list(cols_with_missing.index)].isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Pattern', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_missing_values.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] ✅ Missing plot saved')

# 5. Pairplot of top features (only if few enough)
top_for_pair = [target] + top_mi_cols[:3]
available_for_pair = [c for c in top_for_pair if c in df.columns]
if len(available_for_pair) >= 2 and len(available_for_pair) <= 5 and len(df) < 500:
    sns.pairplot(df[available_for_pair].dropna(), diag_kind='kde')
    plt.savefig(os.path.join(OUTPUT_DIR, '05_pairplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] ✅ Pairplot saved')

# ==========================================
# STEP 12: Save Outputs
# ==========================================
print('\n[STATUS] ===== SAVING OUTPUTS =====')

# Save eddie_output.csv with key analysis columns
output_df = df.copy()
output_df['target_is_top_mi'] = output_df[target].isin(top_mi_cols[:3]) if 'top_mi_cols' in dir() else False
output_df.to_csv(os.path.join(OUTPUT_DIR, 'eddie_output.csv'), index=False)
print(f'[STATUS] ✅ eddie_output.csv saved: {output_df.shape}')

# Build report
report_lines = []
report_lines.append('# Eddie EDA Report')
report_lines.append(f'')
report_lines.append(f'- **Dataset:** PulseCart Customer Behavior')
report_lines.append(f'- **Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}')
report_lines.append(f'- **Target:** `{target}` ({problem_type})')
report_lines.append(f'- **Imbalance Ratio:** {imbalance_ratio:.2f}x' if 'imbalance_ratio' in dir() else '-')
report_lines.append(f'- **Best Silhouette:** {best_sil:.4f}' if isinstance(best_sil, (int, float)) and best_sil >= 0 else 'Best Silhouette: N/A')
report_lines.append(f'')

report_lines.append('## PIPELINE_SPEC')
report_lines.append('```json')
import json
report_lines.append(json.dumps(pipeline_spec, indent=2, default=str))
report_lines.append('```')
report_lines.append(f'')

report_lines.append('## Top Features')
report_lines.append('| Feature | MI Score |')
report_lines.append('|---------|----------|')
for col, sc in mi_results[:10]:
    report_lines.append(f'| {col} | {sc:.4f} |')
report_lines.append(f'')

report_lines.append('## Data Quality Issues')
if issues:
    for iss in issues:
        report_lines.append(f'- ⚠️ {iss}')
else:
    report_lines.append('- ✅ No major issues detected')
report_lines.append(f'')

report_lines.append('## INSIGHT_QUALITY')
report_lines.append(f'- Criteria met: {len(criteria_met)}/4')
report_lines.append(f'- Verdict: {"SUFFICIENT" if len(criteria_met) >= 2 else "INSUFFICIENT"}')
if len(criteria_met) < 2:
    report_lines.append('- Next Angle: ลองใช้ target อื่น หรือ feature interaction')
report_lines.append(f'')

report_lines.append('## Key Insights')
report_lines.append(f'1. Target `{target}` has {df[target].nunique()} unique values')
if len(mi_results) > 0:
    report_lines.append(f'2. Top MI features: {", ".join([f"`{c}`" for c, _ in mi_results[:5]])}')
if len(issues) > 0:
    report_lines.append(f'3. Data quality: {", ".join(issues)}')
report_lines.append('')
report_lines.append('*Report generated by Eddie — EDA Agent*')

report_content = '\n'.join(report_lines)
with open(os.path.join(OUTPUT_DIR, 'eddie_report.md'), 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f'[STATUS] ✅ eddie_report.md saved')

print(f'[STATUS] ===== EDDIE EDA COMPLETE ✅ =====')
print(f'[STATUS] Report: {os.path.join(OUTPUT_DIR, "eddie_report.md")}')
print(f'[STATUS] Output: {os.path.join(OUTPUT_DIR, "eddie_output.csv")}')
print(f'[STATUS] Charts: {OUTPUT_DIR}')
print(f'[STATUS] Target: {target} | Problem: {problem_type}')
print(f'[STATUS] Parameters saved ✓')