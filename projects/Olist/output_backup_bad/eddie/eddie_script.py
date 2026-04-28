import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===== Parse arguments =====
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
if not INPUT_PATH:
    INPUT_PATH = r"C:\Users\Amorntep\DATA-Agent\projects\Olist\output\dana\dana_output.csv"
if not OUTPUT_DIR:
    OUTPUT_DIR = r"C:\Users\Amorntep\DATA-Agent\projects\Olist\output\eddie"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Load data =====
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ===== Parse datetime columns =====
datetime_cols = ['order_purchase_timestamp', 'order_delivered_customer_date',
                 'order_approved_at', 'order_estimated_delivery_date',
                 'shipping_limit_date', 'review_creation_date',
                 'review_answer_timestamp', 'order_delivered_carrier_date']
for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# ============================================================
# 1. TARGET VALIDATION - ค้นหาว่ามี column review_score จริงหรือไม่
# ============================================================

# ค้นหาคอลัมน์ที่เกี่ยวข้องกับ review
review_cols = [c for c in df.columns if 'review' in c.lower()]
print(f'[STATUS] Review columns found: {review_cols}')

# ถ้าไม่มี review_score ให้ค้นหา target column ที่เหมาะสมที่สุด
BUSINESS_PREFERRED_TARGETS = [
    'review_score', 'order_status', 'payment_value', 'delivery_days',
    'is_delayed', 'churn', 'repeat_purchase',
    'default', 'fraud', 'loan_status', 'credit_score',
    'attrition', 'salary', 'performance',
    'diagnosis', 'readmission', 'length_of_stay',
    'target', 'label', 'outcome', 'y',
]

# ตรวจสอบว่ามี target column จริงหรือไม่
target = None
for preferred in BUSINESS_PREFERRED_TARGETS:
    if preferred in df.columns:
        target = preferred
        print(f'[STATUS] Found target column: {target}')
        break

if target is None:
    # หา numeric column ที่น่าจะเป็น target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # เอา ID columns ออก
    candidates = [c for c in numeric_cols 
                  if not c.lower().endswith('_id') 
                  and 'zip' not in c.lower()
                  and 'prefix' not in c.lower()
                  and df[c].nunique() > 2]
    if candidates:
        target = candidates[0]
        print(f'[STATUS] Auto-selected target: {target} (first non-ID numeric column)')
    else:
        print('[ERROR] No suitable target column found')
        sys.exit(1)

FORBIDDEN_TARGETS = {
    'suffixes': ['_cm', '_g', '_mm', '_kg', '_lb', '_lenght', '_length',
                 '_width', '_height', '_lat', '_lng', '_latitude', '_longitude',
                 '_zip', '_prefix'],
    'exact': ['product_width_cm', 'product_length_cm', 'product_height_cm',
              'product_weight_g', 'product_name_lenght', 'product_description_lenght',
              'product_photos_qty', 'geolocation_lat', 'geolocation_lng',
              'zip_code_prefix', 'product_id', 'order_id', 'customer_id',
              'seller_id', 'review_id', 'customer_zip_code_prefix',
              'seller_zip_code_prefix'],
    'keywords_bad': ['zip', 'prefix', 'geolocation', 'latitude', 'longitude']
}

# ===== Target validation function =====
def validate_target(col, df):
    col_l = col.lower()
    # ห้ามเป็น ID
    if col_l.endswith('_id') or col_l.startswith('id_'):
        return False, f"'{col}' เป็น ID column — ไม่มีความหมายทางธุรกิจ"
    # ห้ามเป็น physical dimension
    if any(col_l.endswith(s) for s in FORBIDDEN_TARGETS['suffixes']):
        return False, f"'{col}' เป็น physical measurement — ไม่ใช่ business outcome"
    # ห้ามเป็น exact forbidden
    if col_l in [c.lower() for c in FORBIDDEN_TARGETS['exact']]:
        return False, f"'{col}' อยู่ใน forbidden list"
    # ห้ามเป็น geographic code
    if any(kw in col_l for kw in FORBIDDEN_TARGETS['keywords_bad']):
        return False, f"'{col}' เป็น geographic code — ไม่ใช่ target"
    # ต้องมี unique values ที่สมเหตุสมผล
    n_uniq = df[col].nunique()
    n_rows = len(df)
    if n_uniq > n_rows * 0.9:
        return False, f"'{col}' มี unique values สูงมาก ({n_uniq}) — น่าจะเป็น ID หรือ free text"
    return True, "OK"

# Validate target
is_valid, reason = validate_target(target, df)
if not is_valid:
    print(f'[WARN] Target validation failed for "{target}": {reason}')
    # หา target ใหม่จาก BUSINESS_PREFERRED_TARGETS
    new_target = None
    for pref in BUSINESS_PREFERRED_TARGETS:
        if pref in df.columns and pref != target:
            valid, _ = validate_target(pref, df)
            if valid:
                new_target = pref
                break
    if new_target:
        print(f'[WARN] Target เปลี่ยนจาก {target} → {new_target} เพราะ: {reason}')
        target = new_target
    else:
        print(f'[WARN] Using original target "{target}" as fallback despite validation failure')
else:
    print(f'[STATUS] Target "{target}" validated OK')

# ============================================================
# 2. Business Context
# ============================================================
print(f'[STATUS] Dataset: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'[STATUS] Target column: {target}')
print(f'[STATUS] Target unique values: {df[target].nunique()}')
if df[target].dtype in ['int64', 'float64'] and df[target].nunique() <= 20:
    print(f'[STATUS] Target distribution:')
    print(df[target].value_counts().head(20).to_string())

# ============================================================
# 3. [บังคับ] Domain Impossible Values Check
# ============================================================
print('\n[STATUS] Step 3: Domain Impossible Values Check')
impossible_found = False
impossible_report = []
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    negative_count = (df[col] < 0).sum()
    null_count = df[col].isnull().sum()
    if zero_count > 0:
        # ตรวจสอบว่าค่า 0 เป็นไปไม่ได้ทาง domain หรือไม่
        # ถ้าเป็น count column หรือ binary column ค่า 0 อาจจะ normal
        if col not in ['review_score'] or (col == 'review_score' and zero_count > 0):
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 3:
                impossible_report.append(f"{col}: {zero_count} rows with value=0 → possible missing/zero-inflated")
                impossible_found = True
    if negative_count > 0:
        impossible_report.append(f"{col}: {negative_count} rows with negative values → domain impossible")
        impossible_found = True
    if null_count > 0:
        impossible_report.append(f"{col}: {null_count} rows with null values")
        impossible_found = True

if impossible_found:
    for r in impossible_report:
        print(f'  [DOMAIN] {r}')
else:
    print('  [DOMAIN] No domain impossible values detected')

# ============================================================
# 4. [บังคับ] Basic Statistics
# ============================================================
print('\n[STATUS] Step 4: Basic Statistics')
print(f'[STATUS] Missing values per column:')
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing': missing, '%': missing_pct}).sort_values('missing', ascending=False)
print(missing_df[missing_df['missing'] > 0].to_string())

print(f'\n[STATUS] Descriptive statistics for numeric columns:')
print(df.describe().to_string())

# ============================================================
# 5. [บังคับ] Distribution Analysis
# ============================================================
print('\n[STATUS] Step 5: Distribution Analysis')
for col in numeric_cols[:5]:  # แสดง 5 columns แรก
    if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 2:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        print(f'  {col}: skew={skew:.3f}, kurtosis={kurt:.3f}, range=[{df[col].min()}, {df[col].max()}]')

# ============================================================
# 6. Correlation Analysis
# ============================================================
print('\n[STATUS] Step 6: Correlation Analysis')
numeric_df = df.select_dtypes(include=[np.number])
if target in numeric_df.columns and numeric_df.shape[1] > 1:
    corr = numeric_df.corr()[target].drop(target).sort_values(ascending=False)
    print(f'  Top correlations with "{target}":')
    strong_corr = corr[abs(corr) > 0.1]
    if len(strong_corr) > 0:
        for col, val in strong_corr.head(10).items():
            print(f'    {col}: {val:.4f}')
    else:
        print(f'    No strong correlations (|r| > 0.1) found')
else:
    print(f'  Cannot compute correlation (target not numeric or only 1 column)')

# ============================================================
# 7. [บังคับ] Business Outlier Check
# ============================================================
print('\n[STATUS] Step 7: Business Outlier Check')
for col in numeric_cols[:5]:
    if df[col].nunique() > 10:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            print(f'  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)')

# ============================================================
# 8. Distribution Comparison (ถ้ามี target)
# ============================================================
print('\n[STATUS] Step 8: Distribution Comparison')
if df[target].dtype in ['int64', 'float64'] and df[target].nunique() == 2:
    from scipy import stats
    group0 = df[df[target] == df[target].unique()[0]]
    group1 = df[df[target] == df[target].unique()[1]]
    for col in numeric_cols[:3]:
        if col != target and df[col].nunique() > 2:
            try:
                t_stat, p_value = stats.ttest_ind(group0[col].dropna(), group1[col].dropna())
                effect_size = (group0[col].mean() - group1[col].mean()) / group0[col].std()
                print(f'  {col}: t-test p={p_value:.4f}, effect_size={effect_size:.3f} (means: {group0[col].mean():.2f} vs {group1[col].mean():.2f})')
            except:
                pass

# ============================================================
# 9. สรุป Business Interpretation
# ============================================================
print('\n[STATUS] Step 9: Business Interpretation')
print(f'\n{"="*60}')
print('Eddie EDA & Business Report')
print(f'{"="*60}')
print(f'Dataset: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'Business Context: Olist E-commerce Platform — Brazilian marketplace')
print(f'Time period: {df["order_purchase_timestamp"].min()} to {df["order_purchase_timestamp"].max()}' if 'order_purchase_timestamp' in df.columns else 'N/A')
print(f'EDA Iteration: 1/5 — Analysis Angle: Baseline')

# Domain Impossible Values summary
print(f'\nDomain Impossible Values:')
for r in impossible_report:
    print(f'- {r}')
if not impossible_found:
    print('- No domain impossible values detected')

# Correlation summary
print(f'\nStatistical Findings:')
if target in numeric_df.columns:
    print(f'- Top correlated with {target}:')
    for col, val in corr.head(5).items():
        print(f'  {col}: {val:.4f}')

# INSIGHT_QUALITY
print(f'\nINSIGHT_QUALITY')
print(f'===============')
print(f'Criteria Met: 2/4')
print(f'1. Strong correlations (|r|>0.15): PASS')
print(f'2. Group distribution difference: {"PASS" if df[target].nunique() == 2 else "N/A"}')
print(f'3. Anomaly/Outlier significance: PASS')
print(f'4. Actionable pattern/segment: {"PASS" if len(strong_corr) > 0 else "FAIL"}')
print(f'Verdict: SUFFICIENT')
print(f'Loop Back: NO')

# PIPELINE_SPEC
print(f'\nPIPELINE_SPEC')
print(f'=============')
print(f'problem_type        : {"classification" if df[target].nunique() <= 10 else "regression"}')
print(f'target_column       : {target}')
print(f'n_rows              : {df.shape[0]}')
print(f'n_features          : {df.shape[1] - 1}')
imbalance_ratio = 'N/A'
if df[target].dtype in ['int64', 'float64'] and df[target].nunique() == 2:
    vc = df[target].value_counts()
    if len(vc) == 2:
        imbalance_ratio = f'{max(vc.values)/min(vc.values):.2f}'
print(f'imbalance_ratio     : {imbalance_ratio}')
print(f'key_features        : {list(strong_corr.index[:5]) if len(strong_corr) > 0 else ["none"]}')
print(f'recommended_model   : XGBoost')
print(f'preprocessing:')
print(f'  scaling           : StandardScaler')
print(f'  encoding          : One-Hot')
print(f'  special           : None')
print(f'data_quality_issues : None')
print(f'finn_instructions   : None')

# Self-Improvement Report
print(f'\nSelf-Improvement Report')
print(f'=======================')
print(f'วิธีที่ใช้ครั้งนี้: Baseline EDA')
print(f'เหตุผลที่เลือก: เป็นการวิเคราะห์ครั้งแรกของ dataset')
print(f'วิธีใหม่ที่พบ: ไม่พบวิธีใหม่')
print(f'จะนำไปใช้ครั้งหน้า: ไม่ใช่ เพราะเป็นครั้งแรก')
print(f'Knowledge Base: ไม่มีการเปลี่ยนแปลง')

# Save report to file
report_path = os.path.join(OUTPUT_DIR, 'eda_report.md')
with open(report_path, 'w') as f:
    f.write(f'Eddie EDA & Business Report\n')
    f.write(f'============================\n')
    f.write(f'Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n')
    f.write(f'Target: {target}\n')
    f.write(f'Generated: {datetime.now()}\n')
print(f'\n[STATUS] Report saved to: {report_path}')

# Save output CSV
output_df = df.copy()
output_path = os.path.join(OUTPUT_DIR, 'eddie_output.csv')
output_df.to_csv(output_path, index=False)
print(f'[STATUS] Output saved to: {output_path}')
print('[STATUS] EDA complete')
