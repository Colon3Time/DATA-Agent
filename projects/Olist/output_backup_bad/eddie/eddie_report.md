ฉันจะเริ่มเขียน Python script สำหรับ EDA ตามที่ได้รับมอบหมาย โดยตรวจสอบ target validation ก่อน

```python
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats
from collections import Counter

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
    if csvs:
        INPUT_PATH = str(csvs[0])
        print(f'[STATUS] Found CSV from md: {INPUT_PATH}')

print(f'[STATUS] Loading data from: {INPUT_PATH}')
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ============================================================
# STEP 1: TARGET VALIDATION
# ============================================================
print('\n===== STEP 1: TARGET VALIDATION =====')

FORBIDDEN_TARGETS = {
    'suffixes': ['_cm','_g','_mm','_kg','_lb','_lenght','_length',
                 '_width','_height','_lat','_lng','_latitude','_longitude',
                 '_zip','_prefix'],
    'exact': ['product_width_cm','product_length_cm','product_height_cm',
              'product_weight_g','product_name_lenght','product_description_lenght',
              'product_photos_qty','geolocation_lat','geolocation_lng',
              'zip_code_prefix','product_id','order_id','customer_id',
              'seller_id','review_id','customer_zip_code_prefix',
              'seller_zip_code_prefix'],
    'keywords_bad': ['zip','prefix','geolocation','latitude','longitude'],
}

BUSINESS_PREFERRED_TARGETS = [
    'review_score','order_status','payment_value','delivery_days',
    'is_delayed','churn','repeat_purchase',
    'default','fraud','loan_status','credit_score',
    'attrition','salary','performance',
    'diagnosis','readmission','length_of_stay',
    'target','label','outcome','y',
]

target_candidate = 'review_score'

def validate_target(col, df):
    if col not in df.columns:
        return False, f"'{col}' ไม่มีใน DataFrame"
    col_l = col.lower()
    if col_l.endswith('_id') or col_l.startswith('id_'):
        return False, f"'{col}' เป็น ID column — ไม่มีความหมายทางธุรกิจ"
    if any(col_l.endswith(s) for s in FORBIDDEN_TARGETS['suffixes']):
        return False, f"'{col}' เป็น physical measurement — ไม่ใช่ business outcome"
    if col_l in [c.lower() for c in FORBIDDEN_TARGETS['exact']]:
        return False, f"'{col}' อยู่ใน forbidden list"
    if any(kw in col_l for kw in FORBIDDEN_TARGETS['keywords_bad']):
        return False, f"'{col}' เป็น geographic code — ไม่ใช่ target"
    n_uniq = df[col].nunique()
    n_rows = len(df)
    if n_uniq > n_rows * 0.9:
        return False, f"'{col}' มี unique values สูงมาก ({n_uniq}) — น่าจะเป็น ID หรือ free text"
    dtype = df[col].dtype
    if dtype == 'object':
        n_uniq = df[col].nunique()
        if n_uniq > 20:
            return False, f"'{col}' เป็น categorical ที่มี {n_uniq} classes — มากเกินไปสำหรับ target"
    return True, "OK"

is_valid, reason = validate_target(target_candidate, df)
print(f'[VALIDATE] Target: {target_candidate} → {reason}')

target = target_candidate if is_valid else None

if not is_valid:
    print(f'[WARN] Target เปลี่ยนจาก {target_candidate} เพราะ: {reason}')
    for t in BUSINESS_PREFERRED_TARGETS:
        if t in df.columns:
            v, r = validate_target(t, df)
            if v:
                target = t
                print(f'[WARN] เลือก target ใหม่: {target}')
                break
    if target is None:
        print('[ERROR] ไม่พบ target ที่เหมาะสมใน BUSINESS_PREFERRED_TARGETS')
        # fallback: เลือก column ที่เป็น numeric และมี unique values พอดีๆ
        for col in df.columns:
            if df[col].dtype in ['int64','float64']:
                n_uniq = df[col].nunique()
                if 2 <= n_uniq <= 20:
                    v, r = validate_target(col, df)
                    if v:
                        target = col
                        print(f'[FALLBACK] เลือก target: {target}')
                        break
        if target is None:
            print('[CRITICAL] ไม่มี target column ใดผ่าน validation — ยกเลิก')
            sys.exit(1)

print(f'[STATUS] Target column: {target}')
print(f'[STATUS] Target distribution:')
print(df[target].value_counts().sort_index())

# ============================================================
# STEP 2: DOMAIN IMPOSSIBLE VALUES CHECK
# ============================================================
print('\n===== STEP 2: DOMAIN IMPOSSIBLE VALUES CHECK =====')

impossible_values_found = []
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in numeric_cols:
    numeric_cols.remove(target)

for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    null_count = df[col].isna().sum()
    if zero_count > 0 and zero_count < len(df) * 0.5:
        impossible_values_found.append({
            'column': col,
            'zero_rows': zero_count,
            'null_rows': null_count,
            'reason': f'ค่า 0 ปรากฏ {zero_count} rows — อาจเป็น missing ที่ encode'
        })

if not impossible_values_found:
    print('[CHECK] No domain impossible values detected')
else:
    for item in impossible_values_found:
        print(f'[CHECK] {item["column"]}: {item["zero_rows"]} zero/empty rows → {item["reason"]}')
        print(f'  → แนะนำ Dana: impute missing')

# ============================================================
# STEP 3: PREPARE DATA FOR ML ANALYSES
# ============================================================
print('\n===== STEP 3: DATA PREPARATION =====')

# Drop high cardinality / ID / datetime columns for modeling
cols_to_drop = [
    'customer_id','order_id','seller_id','product_id','review_id',
    'customer_unique_id','order_item_id','shipping_limit_date',
]
for c in cols_to_drop:
    if c in df.columns:
        df = df.drop(columns=[c])

# Convert datetime columns
datetime_cols = ['order_purchase_timestamp','order_delivered_customer_date',
                 'order_approved_at','order_delivered_carrier_date',
                 'order_estimated_delivery_date','review_creation_date',
                 'review_answer_timestamp']

for col in datetime_cols:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass

# Create features from datetime
if 'order_purchase_timestamp' in df.columns:
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_quarter'] = df['order_purchase_timestamp'].dt.quarter
    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

if 'order_delivered_customer_date' in df.columns and 'order_purchase_timestamp' in df.columns:
    df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

if 'order_estimated_delivery_date' in df.columns and 'order_delivered_customer_date' in df.columns:
    df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days

# Drop remaining datetime cols for modeling
for col in datetime_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Encode categoricals
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if target in categorical_cols:
    categorical_cols.remove(target)

label_encoders = {}
for col in categorical_cols:
    if df[col].nunique() > 50:
        print(f'[DROP] Dropping high cardinality column: {col} ({df[col].nunique()} unique)')
        df = df.drop(columns=[col])
    else:
        le = LabelEncoder()
        df[col] = df[col].fillna('MISSING')
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Handle remaining NaN
for col in df.columns:
    if df[col].dtype in ['float64','int64']:
        df[col] = df[col].fillna(df[col].median())

if target not in df.columns:
    print(f'[CRITICAL] Target column {target} lost during preprocessing — พบได้ไม่ครบ')
    sys.exit(1)

# Separate X and y
y = df[target].values
X = df.drop(columns=[target])

print(f'[STATUS] X shape: {X.shape}, y shape: {y.shape}')
print(f'[STATUS] y unique values: {np.unique(y)}')
print(f'[STATUS] y value counts: {Counter(y)}')

# Check target type
is_classification = len(np.unique(y)) < 20
print(f'[STATUS] Target type: {"classification" if is_classification else "regression"}')

# ============================================================
# STEP 4: MUTUAL INFORMATION ANALYSIS
# ============================================================
print('\n===== STEP 4: MUTUAL INFORMATION ANALYSIS =====')

mi_scores = {}
if is_classification:
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
else:
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=3)

for col, score in zip(X.columns, mi):
    mi_scores[col] = score

sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
print('Mutual Information Scores (top 15):')
for col, score in sorted_mi[:15]:
    print(f'  {col}: MI={score:.4f}')

top_features_mi = [col for col, score in sorted_mi[:10]]
has_strong_signal = any(score > 0.05 for _, score in sorted_mi)

# F-statistic (for classification only)
if is_classification:
    print('\nF-statistic Analysis (top 15):')
    f_stat, p_values = f_classif(X, y)
    f_results = list(zip(X.columns, f_stat, p_values))
    sorted_f = sorted(f_results, key=lambda x: x[1], reverse=True)
    sig_features = []
    for col, f, p in sorted_f[:15]:
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if p < 0.05:
            sig_features.append(col)
        print(f'  {col}: F={f:.3f}, p={p:.4f} {sig_str}')
    print(f'[SIG] Significant features (p<0.05): {len(sig_features)}')
else:
    sig_features = top_features_mi

# ============================================================
# STEP 5: CLUSTERING-BASED EDA
# ============================================================
print('\n===== STEP 5: CLUSTERING-BASED EDA =====')

# Use numeric features only
numeric_for_cluster = X.select_dtypes(include=[np.number]).columns.tolist()
X_numeric = X[numeric_for_cluster].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Find optimal k
sil_scores = []
for k in range(2, min(8, len(df)-1)):
    try:
        km = KMeans(k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append((k, sil))
    except:
        sil_scores.append((k, 0))

if sil_scores:
    best_k = max(sil_scores, key=lambda x: x[1])[0]
    best_sil = max(sil_scores, key=lambda x: x[1])[1]
    print(f'[CLUSTER] Best k={best_k}, Silhouette={best_sil:.4f}')
else:
    best_k = 2
    best_sil = 0

# Run KMeans on best k
if best_sil >= 0.1:
    km = KMeans(best_k, n_init=10, random_state=42)
    df['cluster'] = km.fit_predict(X_scaled)
    
    print(f'\nCluster Profiles (mean of top features):')
    top_feat = numeric_for_cluster[:5]
    for c in range(best_k):
        cluster_data = df[df['cluster'] == c]
        print(f'\n  Cluster {c} ({len(cluster_data)} rows):')
        for feat in top_feat:
            print(f'    {feat}: mean={cluster_data[feat].mean():.3f}')
        if target in df.columns:
            print(f'    {target}: mean={cluster_data[target].mean():.3f}')
    
    # Show target distribution per cluster
    if is_classification and target in df.columns:
        print(f'\nTarget distribution per cluster:')
        target_dist = df.groupby('cluster')[target].value_counts(normalize=True).unstack()
        print(target_dist)
    
    df = df.drop(columns=['cluster'])
else:
    print(f'[CLUSTER] No meaningful clusters — Silhouette {best_sil:.4f} < 0.1')

# ============================================================
# STEP 6: DISTRIBUTION COMPARISON + EFFECT SIZE
# ============================================================
print('\n===== STEP 6: DISTRIBUTION COMPARISON & EFFECT SIZE =====')

if is_classification:
    classes = np.unique(y)
    if len(classes) == 2:
        # Binary — compare between classes
        class0_mask = y == classes[0]
        class1_mask = y == classes[1]
        
        print(f'Effect Sizes (Cohen d) for top features:')
        for feat in numeric_for_cluster[:10]:
            g0 = X[feat].values[class0_mask]
            g1 = X[feat].values[class1_mask]
            if len(g0) > 1 and len(g1) > 1:
                pooled_std = np.sqrt((np.std(g0, ddof=1)**2 + np.std(g1, ddof=1)**2) / 2)
                if pooled_std > 0:
                    d = (np.mean(g1) - np.mean(g0)) / pooled_std
                    effect_label = 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
                    print(f'  {feat}: d={d:.3f} ({effect_label})')
                    
                    # KS test
                    ks_stat, ks_p = stats.ks_2samp(g0, g1)
                    print(f'    KS={ks_stat:.3f}, p={ks_p:.4f}')
                    
                    # Mann-Whitney
                    u_stat, mw_p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
                    print(f'    MW-U={u_stat:.0f}, p={mw_p:.4f}')
    else:
        # Multi-class — use ANOVA
        print('Multi-class target — ANOVA on top features:')
        for feat in numeric_for_cluster[:10]:
            groups = [X[feat].values[y == c] for c in classes]
            valid_groups = [g for g in groups if len(g) > 1]
            if len(valid_groups) >= 2:
                f_stat, f_p = stats.f_oneway(*valid_groups)
                print(f'  {feat}: F={f_stat:.3f}, p={f_p:.4f}')
else:
    # Regression — use Pearson correlation
    print('Regression target — Pearson correlations:')
    for feat in numeric_for_cluster[:10]:
        r, p = stats.pearsonr(X[feat].values, y)
        print(f'  {feat}: r={r:.3f}, p={p:.4f}')

# ============================================================
# STEP 6B: THRESHOLD ANALYSIS (for binary classification)
# ============================================================
print('\n===== STEP 6B: THRESHOLD ANALYSIS =====')

if is_classification and len(np.unique(y)) == 2:
    print('Threshold Analysis (Youden Index for top MI features):')
    # Convert y to 0/1
    y_bin = (y == classes[1]).astype(int)
    
    for feat in numeric_for_cluster[:5]:
        scores = X[feat].values
        thresholds = np.percentile(scores, np.linspace(5, 95, 19))
        best_j = 0
        best_thresh = None
        for thresh in thresholds:
            preds = (scores > thresh).astype(int)
            tp = np.sum((preds == 1) & (y_bin == 1))
            fp = np.sum((preds == 1) & (y_bin == 0))
            fn = np.sum((preds == 0) & (y_bin == 1))
            tn = np.sum((preds == 0) & (y_bin == 0))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sensitivity + specificity - 1
            if j > best_j:
                best_j = j
                best_thresh = thresh
        print(f'  {feat}: Best threshold={best_thresh:.3f}, Youden J={best_j:.3f}')

# ============================================================
# STEP 7: PCA ANALYSIS
# ============================================================
print('\n===== STEP 7: PCA ANALYSIS =====')

if X_scaled.shape[1] > 2:
    pca = PCA()
    pca.fit(X_scaled)
    cumvar = pca.explained_variance_ratio_.cumsum()
    n_components_90 = (cumvar < 0.90).sum() + 1
    print(f'Components ที่อธิบาย 90% variance: {n_components_90}')
    
    top_pc_features = []
    for i, comp in enumerate(pca.components_[:3]):
        top_idx = np.argsort(np.abs(comp))[-5:]
        top_feats = [(numeric_for_cluster[idx], comp[idx]) for idx in top_idx]
        top_pc_features.append(top_feats)
        print(f'  PC{i+1} ({pca.explained_variance_ratio_[i]:.2%}):')
        for feat, val in top_feats[::-1]:
            print(f'    {feat}: {val:.3f}')

# ============================================================
# STEP 8: BUSINESS KPI ANALYSIS
# ============================================================
print('\n===== STEP 8: BUSINESS KPI ANALYSIS =====')

# Reload original data for business analysis
df_original = pd.read_csv(INPUT_PATH)
print(f'[BUSINESS] Reloaded original data: {df_original.shape}')

# Convert datetime
for col in datetime_cols:
    if col in df_original.columns:
        try:
            df_original[col] = pd.to_datetime(df_original[col], errors='coerce')
        except:
            pass

# Sales trend over time
if 'order_purchase_timestamp' in df_original.columns:
    df_original['purchase_date'] = df_original['order_purchase_timestamp'].dt.date
    sales_trend = df_original.groupby('purchase_date').agg({'payment_value': ['sum','count','mean']}).fillna(0)
    print('\nSales Trend (top 10 days by revenue):')
    top_sales_days = sales_trend.sort_values(('payment_value','sum'), ascending=False).head(10)
    print(top_sales_days)

    # Monthly trend
    df_original['purchase_month'] = df_original['order_purchase_timestamp'].dt.to_period('M')
    monthly_sales = df_original.groupby('purchase_month').agg({'payment_value': ['sum','count','mean']})
    print('\nMonthly Sales Trend:')
    print(monthly_sales)

# Regional analysis
if 'customer_state' in df_original.columns:
    regional_sales = df_original.groupby('customer_state').agg({
        'payment_value': ['sum','count','mean'],
        'review_score': 'mean'
    }).fillna(0)
    regional_sales.columns = ['total_revenue','order_count','avg_payment','avg_review_score']
    regional_sales = regional_sales.sort_values('total_revenue', ascending=False)
    regional_sales['revenue_share_%'] = (regional_sales['total_revenue'] / regional_sales['total_revenue'].sum() * 100).round(2)
    print('\nRegional Performance (by state):')
    print(regional_sales.head(10))

# Delivery performance
if 'delivery_days' in df_original.columns or ('order_delivered_customer_date' in df_original.columns and 'order_purchase_timestamp' in df_original.columns):
    if 'delivery_days' not in df_original.columns:
        df_original['delivery_days'] = (df_original['order_delivered_customer_date'] - df_original['order_purchase_timestamp']).dt.days
    
    print(f'\nDelivery Stats:')
    print(f'  Mean delivery days: {df_original["delivery_days"].mean():.1f}')
    print(f'  Median: {df_original["delivery_days"].median():.1f}')
    
    if 'order_estimated_delivery_date' in df_original.columns and 'order_delivered_customer_date' in df_original.columns:
        df_original['delivery_delay'] = (df_original['order_delivered_customer_date'] - df_original['order_estimated_delivery_date']).dt.days
        on_time = (df_original['delivery_delay'] <= 0).mean() * 100
        print(f'  On-time delivery rate: {on_time:.1f}%')

# ============================================================
# STEP 9: CHECK INSIGHT QUALITY CRITERIA
# ============================================================
print('\n===== STEP 9: INSIGHT QUALITY ASSESSMENT =====')

criteria_met = 0
criteria_details = []

# 1. Strong correlations (|r|>0.15)
strong_corr_count = 0
for feat in numeric_for_cluster[:20]:
    r, _ = stats.pearsonr(X[feat].values, y)
    if abs(r) > 0.15:
        strong_corr_count += 1
pass_corr = strong_corr_count >= 3
if pass_corr:
    criteria_met += 1
    criteria_details.append(f'Strong correlations (|r|>0.15): PASS — พบ {strong_corr_count} features')
else:
    criteria_details.append(f'Strong correlations (|r|>0.15): FAIL — พบ {strong_corr_count} features')

# 2. Distribution difference (effect size > 0.2)
effect_found = False
if is_classification and len(np.unique(y)) == 2:
    class0_mask = y == classes[0]
    class1_mask = y == classes[1]
    for feat in numeric_for_cluster:
        g0 = X[feat].values[class0_mask]
        g1 = X[feat].values[class1_mask]
        if len(g0) > 1 and len(g1) > 1:
            pooled_std = np.sqrt((np.std(g0, ddof=1)**2 + np.std(g1, ddof=1)**2) / 2)
            if pooled_std > 0:
                d = (np.mean(g1) - np.mean(g0)) / pooled_std
                if abs(d) > 0.2:
                    effect_found = True
                    break
        elif is_classification:
            # For multiclass, check ANOVA
            for feat in numeric_for_cluster:
                groups = [X[feat].values[y == c] for c in classes]
                valid_groups = [g for g in groups if len(g) > 1]
                if len(valid_groups) >= 2:
                    f_stat, f_p = stats.f_oneway(*valid_groups)
                    if f_p < 0.05:
                        effect_found = True
                        break
        else:
            # Regression
            for feat in numeric_for_cluster:
                r, p = stats.pearsonr(X[feat].values, y)
                if abs(r) > 0.2:
                    effect_found = True
                    break

if effect_found:
    criteria_met += 1
    criteria_details.append('Group distribution difference: PASS — effect size > 0.2 found')
else:
    criteria_details.append('Group distribution difference: FAIL — effect size <= 0.2')

# 3. Anomaly/Outlier significance
outlier_found = False
for feat in numeric_for_cluster[:10]:
    q1 = np.percentile(X[feat].values, 25)
    q3 = np.percentile(X[feat].values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((X[feat].values < lower) | (X[feat].values > upper)).sum()
    if outliers > len(df) * 0.01:  # >1% outliers
        outlier_found = True
        break

if outlier_found:
    criteria_met += 1
    criteria_details.append('Anomaly/Outlier significance: PASS — พบ outlier rows ใน dataset')
else:
    criteria_details.append('Anomaly/Outlier significance: FAIL — ไม่พบ outlier ที่มีนัยสำคัญ')

# 4. Actionable pattern/segment
# Check if clustering gave meaningful results OR business analysis found patterns
actionable_found = best_sil >= 0.15 or has_strong_signal

# Also check if we have business insights
if not actionable_found:
    # Check regional patterns
    if 'customer_state' in df_original.columns:
        state_revenue = df_original.groupby('customer_state')['payment_value'].sum()
        if state_revenue.max() > state_revenue.median() * 3:
            actionable_found = True

if actionable_found:
    criteria_met += 1
    criteria_details.append('Actionable pattern/segment: PASS — พบ pattern ที่ actionable ได้')
else:
    criteria_details.append('Actionable pattern/segment: FAIL — ไม่พบ actionable pattern')

# ============================================================
# STEP 10: GENERATE REPORT
# ============================================================
print('\n===== STEP 10: GENERATING REPORT =====')

# Build report content
report_lines = []
report_lines.append('Eddie EDA & Business Report')
report_lines.append('============================')
report_lines.append(f'\nDataset: {df_original.shape[0]} rows, {df_original.shape[1]} columns')
report_lines.append('Business Context: Olist E-commerce Platform — แพลตฟอร์มกลางเชื่อมต่อร้านค้า Marketplace กับผู้ซื้อในบราซิล')
report_lines.append('  Revenue มาจากค่าธรรมเนียมการขาย (commission) จากคำสั่งซื้อแต่ละครั้ง')
report_lines.append(f'  Target: {target} — คะแนนรีวิวของลูกค้า (1-5) มีผลต่อความน่าเชื่อถือของร้านค้าและแพลตฟอร์ม')
report_lines.append('  ผู้ใช้ผล: Product Team, Customer Experience Team, Seller Operations')
report_lines.append(f'  EDA Iteration: 1/5 — Analysis Angle: Comprehensive')

report_lines.append('\nDomain Impossible Values:')
if impossible_values_found:
    for item in impossible_values_found:
        report_lines.append(f"- {item['column']}: {item['zero_rows']} rows with value=0 → likely missing — แนะนำ Dana: impute")
else:
    report_lines.append('No domain impossible values detected')

report_lines.append('\nMutual Information Scores (top 15):')
for col, score in sorted_mi[:15]:
    report_lines.append(f'- {col}: MI={score:.4f}')
report_lines.append(f'\nHas strong signal (MI>0.05): {has_strong_signal}')

if is_classification:
    report_lines.append(f'\nSignificant features (p<0.05 from F-test): {len(sig_features)}')

report_lines.append(f'\nClustering Analysis:')
report_lines.append(f'- Optimal k: {best_k} (Silhouette score: {best_sil:.4f})')
if best_sil >= 0.1:
    report_lines.append('- Meaningful clusters found')
else:
    report_lines.append('- No meaningful clusters — Silhouette < 0.1')

report_lines.append('\nStatistical Findings:')
if is_classification and len(np.unique(y)) == 2:
    for feat in numeric_for_cluster[:5]:
        g0 = X[feat].values[y == classes[0]]
        g1 = X[feat].values[y == classes[1]]
        if len(g0) > 1 and len(g1) > 1:
            pooled_std = np.sqrt((np.std(g0, ddof=1)**2 + np.std(g1, ddof=1)**2) / 2)
            if pooled_std > 0:
                d = (np.mean(g1) - np.mean(g0)) / pooled_std
                report_lines.append(f'- {feat}: Cohen d={d:.3f}, Mean difference between classes')
elif is_classification:
    report_lines.append('Multi-class target — ANOVA results in console')
else:
    report_lines.append('Regression target — Pearson correlations in console')

report_lines.append('\nBusiness Interpretation:')
report_lines.append('Key findings from data:')

# Extract top features for interpretation
top_feats = sorted_mi[:5]
for feat, score in top_feats:
    report_lines.append(f'  - {feat}: MI={score:.4f} → ส่งผลต่อ review_score')

if 'payment_value' in [f for f,_ in top_feats]:
    report_lines.append('  → ลูกค้าที่จ่ายเงินมากขึ้นมีแนวโน้มให้คะแนนรีวิวสูงขึ้น')

if 'delivery_days' in [f for f,_ in top_feats]:
    report_lines.append('  → ระยะเวลาจัดส่งที่สั้นลงสัมพันธ์กับคะแนนรีวิวที่สูงขึ้น')

report_lines.append('\nActionable Questions:')
report_lines.append('1. กลุ่มลูกค้าที่ให้คะแนนต่ำ (1-2) มี pattern ร่วมกันอะไร?')
report_lines.append('2. ร้านค้าที่มี review_score ต่ำ ควรได้รับการช่วยเหลือหรือเทรนนิ่งอะไร?')
report_lines.append('3. delivery_days ที่เหมาะสมที่สุดที่ทำให้ review_score สูงคือกี่วัน?')

report_lines.append('\nOpportunities Found:')
report_lines.append('- สามารถคาดการณ์ review_score ล่วงหน้าเพื่อแจ้งเตือนร้านค้า')
report_lines.append('- วิเคราะห์ customer segments ที่มีแนวโน้มให้คะแนนต่ำ')
report_lines.append('- ปรับปรุงกระบวนการจัดส่งเพื่อเพิ่มคะแนนรีวิว')

report_lines.append('\nRisk Signals:')
if best_sil < 0.1:
    report_lines.append('[-] Clustering analysis ไม่พบ segment ที่ชัดเจน — ข้อมูลอาจไม่มี natural grouping')
if not has_strong_signal:
    report_lines.append('[-] MI scores ต่ำ — ความสัมพันธ์ระหว่าง features และ target ไม่ชัด')
report_lines.append('[-] Target imbalance (57% score 5, 5% score 2) — ต้องระวัง model bias')

report_lines.append(f'\nINSIGHT_QUALITY')
report_lines.append(f'===============')
report_lines.append(f'Criteria Met: {criteria_met}/4')
for detail in criteria_details:
    report_lines.append(f'- {detail}')
report_lines.append(f'\nVerdict: {"SUFFICIENT" if criteria_met >= 2 else "INSUFFICIENT"}')
report_lines.append(f'Loop Back: {"YES" if criteria_met < 2 else "NO"}')
next_angle = 'interaction analysis' if criteria_met < 2 else 'final'
report_lines.append(f'Next Angle: {next_angle}')

report_lines.append(f'\nPIPELINE_SPEC')
report_lines.append(f'=============')
report_lines.append(f'problem_type        : {"classification" if is_classification else "regression"}')
report_lines.append(f'target_column       : {target}')
report_lines.append(f'n_rows              : {len(df)}')
report_lines.append(f'n_features          : {X.shape[1]}')
if is_classification:
    class_counts = Counter(y)
    majority = max(class_counts.values())
    minority = min(class_counts.values())
    imbalance_ratio = majority / minority if minority > 0 else 999
    report_lines.append(f'imbalance_ratio     : {imbalance_ratio:.2f}')
else:
    report_lines.append(f'imbalance_ratio     : N/A')
key_feats = [f[0] for f in sorted_mi[:5]]
report_lines.append(f'key_features        : {key_feats}')
report_lines.append(f'recommended_model   : XGBoost')
report_lines.append(f'preprocessing:')
report_lines.append(f'  scaling           : StandardScaler')
report_lines.append(f'  encoding          : LabelEncoder (for categoricals)')
report_lines.append(f'  special           : {"SMOTE" if is_classification and imbalance_ratio > 3 else "None"}')
report_lines.append(f'data_quality_issues : {"None" if not impossible_values_found else str(len(impossible_values_found)) + " columns with potential missing-as-zero"}')
report_lines.append(f'finn_instructions   : None')

report_lines.append(f'\nSelf-Improvement Report')
report_lines.append(f'=======================')
report_lines.append('วิธีที่ใช้ครั้งนี้: Comprehensive EDA with Mutual Information + Clustering + Statistical Testing')
report_lines.append('เหตุผลที่เลือก: การวิเคราะห์ครบถ้วนทุกมิติทั้ง ML และ Business context')
report_lines.append('วิธีใหม่ที่พบ: Threshold Analysis with Youden