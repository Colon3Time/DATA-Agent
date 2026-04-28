import argparse
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1: Load data ──
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()
INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir

if not INPUT_PATH or not os.path.exists(INPUT_PATH):
    print(f'[ERROR] --input required and must exist: {INPUT_PATH}')
    import sys; sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_PATH, sep=';')
print(f'[STATUS] Loaded: {df.shape} from {INPUT_PATH}')
df_original = df.copy()

# ── STEP 2: ตรวจสอบ missing values ──
missing_before = df.isnull().sum().sum()
print(f'[STATUS] Total missing values: {missing_before}')
for col in df.columns:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        print(f'  [STATUS] {col}: {n_miss} missing ({n_miss/len(df)*100:.2f}%)')
if missing_before == 0:
    print('[STATUS] Scout รายงานถูกต้อง: ไม่มี missing values ใน dataset นี้')

# ── STEP 3: ตรวจสอบ data types ──
print('[STATUS] Checking data types...')
dtype_report = []
for col in df.columns:
    orig_dtype = df[col].dtype
    unique_vals = df[col].nunique()
    sample_vals = df[col].dropna().unique()[:5]
    dtype_report.append({
        'column': col,
        'dtype': str(orig_dtype),
        'n_unique': unique_vals,
        'samples': sample_vals
    })
    print(f'  [STATUS] {col}: {orig_dtype}, {unique_vals} unique')

# ── STEP 4: แก้ไข data types — categorical columns ──
# bank-additional-full.csv มี column: age, job, marital, education, default, housing, loan,
# contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp.var.rate,
# cons.price.idx, cons.conf.idx, euribor3m, nr.employed, y

# categorical columns ตาม domain knowledge
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
            'contact', 'month', 'day_of_week', 'poutcome', 'y']

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
        print(f'[STATUS] {col}: converted to category')

# numeric columns — ยืนยันว่าเป็น numeric
num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # ตรวจสอบ coercion (ควรไม่มีเพราะ Scout บอกว่าไม่มี missing)
        coerced = df[col].isnull().sum() - df_original[col].isnull().sum()
        if coerced > 0:
            print(f'[WARN] {col}: {coerced} values coerced to NaN during conversion')

print('[STATUS] Data type conversion complete')

# ── STEP 5: Outlier Detection — IQR + Isolation Forest ──
# กำหนด domain bounds สำหรับ UCI Bank Marketing
# age: 18-100 (ปกติคนทำงาน 18-65 แต่มี retiree ได้)
# campaign: 1-50 (ส่วนใหญ่ 1-10, มากเกินไปคือ bot)
# pdays: -1 to 999 (-1 = ไม่เคยติดต่อ, 999 = นานมาก)
# previous: 0-50
# duration: 0-5000 (วินาที, ปกติ 100-1000)
# numeric socio-economic: emp.var.rate (employment variation rate) ~ -3..3
# cons.price.idx (consumer price index) ~ 92..94
# cons.conf.idx (consumer confidence index) ~ -50..-30
# euribor3m ~ 0..5
# nr.employed ~ 4900..5200

DOMAIN_MIN = {
    'age': 17, 'duration': 0, 'campaign': 1, 'pdays': -1, 'previous': 0,
    'emp.var.rate': -5.0, 'cons.price.idx': 90.0, 'cons.conf.idx': -60.0,
    'euribor3m': 0.0, 'nr.employed': 4800.0
}
DOMAIN_MAX = {
    'age': 100, 'duration': 5000, 'campaign': 50, 'pdays': 999, 'previous': 50,
    'emp.var.rate': 5.0, 'cons.price.idx': 97.0, 'cons.conf.idx': -10.0,
    'euribor3m': 6.0, 'nr.employed': 5400.0
}

feat_cols = [c for c in num_cols if c in df.columns]
outlier_records = []

# IQR-based detection
for col in feat_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo_b = q1 - 1.5 * iqr
    hi_b = q3 + 1.5 * iqr
    domain_lo = DOMAIN_MIN.get(col, -np.inf)
    domain_hi = DOMAIN_MAX.get(col, np.inf)
    
    mask = (df[col] < lo_b) | (df[col] > hi_b)
    outlier_indices = df.index[mask]
    
    for idx in outlier_indices:
        val = df.loc[idx, col]
        # ตรวจสอบ domain violation ก่อน
        if val < domain_lo or val > domain_hi:
            verdict = 'Likely Error'
            action = 'capped'
            # cap ที่ domain bound
            if val < domain_lo:
                df.loc[idx, col] = domain_lo
            else:
                df.loc[idx, col] = domain_hi
        else:
            # IQR outlier แต่อยู่ใน domain ที่เป็นไปได้ — ตรวจสอบ severity
            deviation = abs(val - df[col].median()) / iqr if iqr > 0 else 0
            if deviation > 10:  # extreme outlier (10x IQR)
                verdict = 'Likely Error'
                action = 'capped'
                df.loc[idx, col] = df[col].median()
            else:
                verdict = 'Likely Real'
                action = 'flagged'
        outlier_records.append({
            'row_index': idx,
            'column_name': col,
            'value': float(val),
            'verdict': verdict,
            'reason': f'{col}={val:.2f}, IQR=[{lo_b:.2f},{hi_b:.2f}], domain=[{domain_lo},{domain_hi}]',
            'action': action
        })

# Isolation Forest — multivariate outlier detection
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_mask = iso.fit_predict(df[feat_cols]) == -1

for idx in df.index[iso_mask]:
    if not any(r['row_index'] == idx for r in outlier_records):
        # ดึงค่าที่ผิดปกติมากที่สุด
        val_str = '; '.join([f'{c}={df.loc[idx,c]:.2f}' for c in feat_cols])
        outlier_records.append({
            'row_index': idx,
            'column_name': 'multivariate',
            'value': None,
            'verdict': 'Uncertain',
            'reason': f'Isolation Forest anomaly — values: {val_str}',
            'action': 'flagged'
        })

# สร้าง is_outlier column
df['is_outlier'] = 0
for r in outlier_records:
    if r['verdict'] != 'Likely Error':
        df.loc[r['row_index'], 'is_outlier'] = 1

print(f'[STATUS] Outlier detection complete: {len(outlier_records)} records found')
for verdict in ['Likely Error', 'Likely Real', 'Uncertain']:
    count = sum(1 for r in outlier_records if r['verdict'] == verdict)
    print(f'  [STATUS] {verdict}: {count}')

# ── STEP 6: ตรวจสอบ cardinality ของ categorical columns ──
print('[STATUS] Checking categorical column cardinality...')
cardinality_report = []
for col in cat_cols:
    if col in df.columns:
        unique_vals = df[col].nunique()
        val_counts = df[col].value_counts()
        most_common = val_counts.index[0] if len(val_counts) > 0 else None
        least_common = val_counts.index[-1] if len(val_counts) > 0 else None
        least_common_count = val_counts.iloc[-1] if len(val_counts) > 0 else 0
        cardinality_report.append({
            'column': col,
            'cardinality': unique_vals,
            'most_common': str(most_common),
            'least_common': str(least_common),
            'least_common_count': int(least_common_count)
        })
        print(f'  [STATUS] {col}: {unique_vals} unique values')
        # แสดง rare categories (< 1%)
        threshold = len(df) * 0.01
        rare = val_counts[val_counts < threshold]
        if len(rare) > 0:
            rare_str = ', '.join([f'{v}({c})' for v, c in rare.items()])
            print(f'    Rare categories (<1%): {rare_str}')
        
        # ตรวจสอบ 'unknown' labels
        if 'unknown' in val_counts.index:
            n_unknown = val_counts['unknown']
            pct_unknown = n_unknown / len(df) * 100
            print(f'    [STATUS] {col}: {n_unknown} unknown ({pct_unknown:.2f}%)')

# ── STEP 7: Data Quality Score ──
n = len(df)
missing_after = df.drop(columns=['is_outlier']).isnull().sum().sum()
likely_error_count = sum(1 for r in outlier_records if r['verdict'] == 'Likely Error')

completeness_before = (1 - df_original.isnull().sum().sum() / (n * len(df_original.columns))) * 100
completeness_after = (1 - missing_after / (n * (len(df.columns) - 1))) * 100

validity_before = 100.0  # ยังไม่มีการแก้ไข
validity_after = (1 - likely_error_count / n) * 100

overall_before = 0.5 * completeness_before + 0.5 * validity_before
overall_after = 0.5 * completeness_after + 0.5 * validity_after

print(f'[STATUS] Quality: {overall_before:.1f}% -> {overall_after:.1f}%')

# ── STEP 8: Save outputs ──
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# Save outlier flags
flags_path = os.path.join(OUTPUT_DIR, 'outlier_flags.csv')
flags_df = pd.DataFrame(outlier_records)
flags_df.to_csv(flags_path, index=False)
print(f'[STATUS] Saved: {flags_path}')

# ── STEP 9: Save report ──
report_lines = []
report_lines.append('Dana Cleaning Report')
report_lines.append('====================')
report_lines.append(f'Before: {len(df_original)} rows, {len(df_original.columns)} columns')
report_lines.append(f'After:  {len(df)} rows, {len(df.columns)} columns')
report_lines.append('')
report_lines.append('Missing Values:')
if missing_before == 0:
    report_lines.append('- No missing values detected (confirmed Scout report)')
else:
    for col in df_original.columns:
        n_miss = df_original[col].isnull().sum()
        if n_miss > 0:
            report_lines.append(f'- {col}: {n_miss}/{n} ({n_miss/n*100:.2f}%) — no action needed (0 detected)')
report_lines.append('')

report_lines.append('Data Types Conversion:')
report_lines.append('- Categorical columns converted: ' + ', '.join(cat_cols))
report_lines.append('- Numeric columns verified: ' + ', '.join(num_cols))
report_lines.append('')

report_lines.append('Categorical Column Cardinality:')
for item in cardinality_report:
    report_lines.append(f'- {item["column"]}: {item["cardinality"]} unique, most common="{item["most_common"]}", least common="{item["least_common"]}" ({item["least_common_count"]} rows)')
report_lines.append('')

report_lines.append('Outlier Detection:')
report_lines.append(f'- Method: Isolation Forest (contamination=0.05) + IQR (1.5x)')
report_lines.append(f'- Likely Error (แก้ไขแล้ว):')
likely_errors = [r for r in outlier_records if r['verdict'] == 'Likely Error']
if likely_errors:
    for r in likely_errors[:5]:  # แสดง 5 อันแรก
        report_lines.append(f'  - row {r["row_index"]}: {r["column_name"]}={r["value"]:.2f} -> capped')
    if len(likely_errors) > 5:
        report_lines.append(f'  - ... and {len(likely_errors)-5} more')
else:
    report_lines.append(f'  - None')
report_lines.append(f'- Likely Real (เก็บไว้ + flagged):')
likely_reals = [r for r in outlier_records if r['verdict'] == 'Likely Real']
if likely_reals:
    for r in likely_reals[:5]:
        report_lines.append(f'  - row {r["row_index"]}: {r["column_name"]}={r["value"]:.2f}')
    if len(likely_reals) > 5:
        report_lines.append(f'  - ... and {len(likely_reals)-5} more')
else:
    report_lines.append(f'  - None')
report_lines.append(f'- Uncertain (Isolation Forest):')
uncertains = [r for r in outlier_records if r['verdict'] == 'Uncertain']
if uncertains:
    report_lines.append(f'  - {len(uncertains)} rows flagged by Isolation Forest')
else:
    report_lines.append(f'  - None')
report_lines.append(f'- outlier_flags.csv: {len(outlier_records)} rows total')
report_lines.append('')

report_lines.append('Data Quality Score:')
report_lines.append(f'- Completeness: Before {completeness_before:.1f}% -> After {completeness_after:.1f}%')
report_lines.append(f'- Validity: Before {validity_before:.1f}% -> After {validity_after:.1f}%')
report_lines.append(f'- Overall: Before {overall_before:.1f}% -> After {overall_after:.1f}%')
report_lines.append('')

report_lines.append('Column Stats (Before -> After):')
for col in feat_cols + ['age']:
    before_mean = df_original[col].mean()
    after_mean = df[col].mean()
    before_std = df_original[col].std()
    after_std = df[col].std()
    report_lines.append(f'- {col}: mean {before_mean:.2f} -> {after_mean:.2f}, std {before_std:.2f} -> {after_std:.2f}')
report_lines.append('')

report_lines.append('New Method Found: None')

report_text = '\n'.join(report_lines)
with open(os.path.join(OUTPUT_DIR, 'dana_report.md'), 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f'[STATUS] Saved: {os.path.join(OUTPUT_DIR, "dana_report.md")}')

print(f'[STATUS] Dana cleaning complete — all files saved to {OUTPUT_DIR}')