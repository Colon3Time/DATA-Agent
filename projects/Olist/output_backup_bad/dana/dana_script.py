import argparse
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1: Load data ──
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not INPUT_PATH or not os.path.exists(INPUT_PATH):
    print(f'[ERROR] --input required and must exist: {INPUT_PATH}')
    import sys; sys.exit(1)

df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape} from {INPUT_PATH}')
df_original = df.copy()

# ── STEP 2: Pre-Cleaning Audit ──
print('\n[STATUS] Pre-Cleaning Audit:')
print(f'  Shape: {df.shape}')
print(f'  Columns: {len(df.columns)}')

# Safe column check - only print if review_score exists
if 'review_score' in df.columns:
    print(f'  Target (review_score): {df["review_score"].dtype}')
    print(f'  Target distribution:')
    print(f'    {df["review_score"].value_counts().sort_index().to_dict()}')
else:
    print('  Note: review_score column not found in dataset')

# Check missing values
missing_report = df.isnull().sum()
missing_pct = (missing_report / len(df) * 100).round(2)
print(f'\n[STATUS] Missing values (top 10):')
missing_nonzero = missing_report[missing_report > 0].sort_values(ascending=False)
if len(missing_nonzero) > 0:
    for col in missing_nonzero.head(10).index:
        print(f'  {col}: {missing_report[col]} ({missing_pct[col]}%)')
else:
    print('  No missing values detected')

# Check cardinality for known categorical columns
known_cat_cols = ['product_category_name', 'customer_city', 'customer_state', 'seller_id', 'product_id', 
                  'customer_id', 'order_id', 'review_id', 'product_category_name_english']
print(f'\n[STATUS] Cardinality checks:')
for col in known_cat_cols:
    if col in df.columns:
        print(f'  {col} cardinality: {df[col].nunique()}')

# ── STEP 3: Identify and handle potential leakage columns ──
print('\n[STATUS] Checking potential leakage columns...')
LEAKAGE_COLS = ['order_delivered_customer_date', 'order_estimated_delivery_date', 'order_approved_at',
                'shipping_limit_date', 'order_delivered_carrier_date']
existing_leakage = [c for c in LEAKAGE_COLS if c in df.columns]

if existing_leakage:
    print(f'  Found leakage columns: {existing_leakage}')
    for col in existing_leakage:
        if col in df.columns:
            print(f'  Dropping: {col}')
            df.drop(columns=[col], inplace=True)
else:
    print('  No leakage columns detected')

# ── STEP 4: Convert date columns to datetime features ──
print('\n[STATUS] Processing date columns...')

# Find all date-like columns
date_cols = [c for c in df.columns if any(x in c.lower() for x in ['_date', '_at', 'timestamp', 'delivery'])]

# Remove leakage cols already dropped
date_cols = [c for c in date_cols if c in df.columns]

if 'order_purchase_timestamp' in df.columns:
    print(f'  Processing order_purchase_timestamp...')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
    # Extract useful date features
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    print('  Date features extracted: purchase_year, purchase_month, purchase_dayofweek, purchase_hour')

# Process other date columns safely
for col in date_cols:
    if col != 'order_purchase_timestamp' and col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f'  Converted {col} to datetime')
        except Exception as e:
            print(f'  [WARN] Could not convert {col}: {e}')

# ── STEP 5: Zero-as-missing for medical/inventory columns ──
print('\n[STATUS] Checking for zero-as-missing patterns...')
ZERO_INVALID_COLS = []
for c in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
          'freight_value', 'price']:
    if c in df.columns:
        zero_count = (df[c] == 0).sum()
        if zero_count > 0:
            ZERO_INVALID_COLS.append(c)
            df[c] = df[c].replace(0, np.nan)
            print(f'  {c}: {zero_count} zeros -> NaN')

if not ZERO_INVALID_COLS:
    print('  No zero-as-invalid patterns detected')

# ── STEP 6: Separate numeric and categorical columns ──
print('\n[STATUS] Separating column types...')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove is_outlier if exists from previous run
if 'is_outlier' in num_cols:
    num_cols.remove('is_outlier')

# Remove date-related numeric columns that might cause issues
date_numeric_cols = ['purchase_year', 'purchase_month', 'purchase_dayofweek', 'purchase_hour']
date_numeric_cols = [c for c in date_numeric_cols if c in num_cols]

print(f'  Numeric columns: {len(num_cols)}')
print(f'  Categorical columns: {len(cat_cols)}')

# ── STEP 7: KNN Imputation for numeric columns ──
print('\n[STATUS] KNN Imputation...')
num_cols_for_impute = [c for c in num_cols if c not in date_numeric_cols]  # Exclude date features

if len(num_cols_for_impute) > 0:
    missing_sum = df[num_cols_for_impute].isnull().sum().sum()
    if missing_sum > 0:
        print(f'  Total missing values to impute: {missing_sum}')
        try:
            imputer = KNNImputer(n_neighbors=min(5, len(df)-1))
            df[num_cols_for_impute] = pd.DataFrame(
                imputer.fit_transform(df[num_cols_for_impute]),
                columns=num_cols_for_impute,
                index=df.index
            )
            print(f'  KNN Imputation complete')
        except Exception as e:
            print(f'  [WARN] KNN failed: {e} - using median imputation')
            for col in num_cols_for_impute:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
    else:
        print('  No missing values in numeric columns')
else:
    print('  No numeric columns for imputation')

# ── STEP 8: Post-imputation clip for physical constraints ──
print('\n[STATUS] Post-imputation domain clip...')
DOMAIN_MIN = {
    'product_weight_g': 0,
    'product_length_cm': 0,
    'product_height_cm': 0,
    'product_width_cm': 0,
    'freight_value': 0,
    'price': 0,
    'review_score': 1
}
DOMAIN_MAX = {
    'product_weight_g': 50000,
    'product_length_cm': 200,
    'product_height_cm': 200,
    'product_width_cm': 200,
    'freight_value': 1000,
    'price': 20000,
    'review_score': 5
}

for col, lo in DOMAIN_MIN.items():
    if col in df.columns:
        df[col] = df[col].clip(lower=lo)
        print(f'  {col}: clipped lower={lo}')
for col, hi in DOMAIN_MAX.items():
    if col in df.columns:
        df[col] = df[col].clip(upper=hi)
        print(f'  {col}: clipped upper={hi}')

# ── STEP 9: Outlier Detection (IQR + Isolation Forest) ──
print('\n[STATUS] Outlier Detection...')
outlier_records = []

# Feature columns for outlier detection (exclude date features and id columns)
feat_cols = [c for c in num_cols_for_impute 
             if c not in ['review_score', 'purchase_year', 'purchase_month'] 
             and not any(x in c for x in ['id', 'ID', '_id'])]

if len(feat_cols) > 0:
    # IQR Method
    for col in feat_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo_b, hi_b = q1 - 1.5*iqr, q3 + 1.5*iqr
        domain_lo = DOMAIN_MIN.get(col, -np.inf)
        domain_hi = DOMAIN_MAX.get(col, np.inf)
        
        outlier_mask = (df[col] < lo_b) | (df[col] > hi_b)
        for idx in df[outlier_mask].index:
            val = df.loc[idx, col]
            if val < domain_lo or val > domain_hi:
                verdict, action = 'Likely Error', 'capped'
                df.loc[idx, col] = df[col].median()
                print(f'  [CAPPED] {col} row {idx}: {val:.2f} -> {df[col].median():.2f} (Likely Error)')
            else:
                verdict, action = 'Likely Real', 'flagged'
            outlier_records.append({
                'row_index': idx,
                'column_name': col,
                'value': round(val, 4),
                'verdict': verdict,
                'reason': f'{col}={val:.4f} (IQR outlier, bounds=[{lo_b:.2f}, {hi_b:.2f}])',
                'action': action
            })

    # Isolation Forest
    if len(feat_cols) >= 2 and len(df) > 5:
        try:
            iso = IsolationForest(contamination=min(0.05, 0.1), random_state=42)
            iso_mask = iso.fit_predict(df[feat_cols]) == -1
            for idx in df.index[iso_mask]:
                if not any(r['row_index'] == idx for r in outlier_records):
                    outlier_records.append({
                        'row_index': idx,
                        'column_name': 'multivariate',
                        'value': None,
                        'verdict': 'Uncertain',
                        'reason': 'Isolation Forest anomaly (multivariate)',
                        'action': 'flagged'
                    })
        except Exception as e:
            print(f'  [WARN] Isolation Forest failed: {e}')

# Add is_outlier flag
df['is_outlier'] = 0
for r in outlier_records:
    if r['verdict'] != 'Likely Error':
        df.loc[r['row_index'], 'is_outlier'] = 1

# Save outlier flags
flags_df = pd.DataFrame(outlier_records)
flags_path = os.path.join(OUTPUT_DIR, 'outlier_flags.csv')
flags_df.to_csv(flags_path, index=False)
print(f'  Outlier flags saved: {len(flags_df)} records')

# ── STEP 10: Data Quality Score ──
print('\n[STATUS] Calculating Data Quality Score...')
n = len(df)
total_cells_before = n * len(df_original.columns)
total_cells_after = n * len(df.columns)

missing_before = df_original.isnull().sum().sum()
missing_after = df.isnull().sum().sum()

likely_error_count = sum(1 for r in outlier_records if r['verdict'] == 'Likely Error')

completeness_before = (1 - missing_before / total_cells_before) * 100
completeness_after = (1 - missing_after / total_cells_after) * 100

validity_before = (1 - missing_before / n) * 100  # proxy: missing cells per row
validity_after = (1 - likely_error_count / n) * 100

overall_before = 0.5 * completeness_before + 0.5 * validity_before
overall_after = 0.5 * completeness_after + 0.5 * validity_after

print(f'  Before: Completeness={completeness_before:.2f}%, Validity={validity_before:.2f}%, Overall={overall_before:.2f}%')
print(f'  After: Completeness={completeness_after:.2f}%, Validity={validity_after:.2f}%, Overall={overall_after:.2f}%')

# ── STEP 11: Generate Report ──
print('\n[STATUS] Generating report...')
report_lines = []
report_lines.append('Dana Cleaning Report')
report_lines.append('=' * 50)
report_lines.append(f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
report_lines.append(f'')
report_lines.append(f'Before: {df_original.shape[0]} rows, {df_original.shape[1]} columns')
report_lines.append(f'After: {df.shape[0]} rows, {df.shape[1]} columns')
report_lines.append(f'')
report_lines.append('Missing Values:')
report_lines.append('-' * 30)
missing_after_report = df.isnull().sum()
missing_after_report = missing_after_report[missing_after_report > 0]
if len(missing_after_report) > 0:
    for col in missing_after_report.index:
        pct = (missing_after_report[col] / n * 100).round(2)
        report_lines.append(f'  - {col}: {missing_after_report[col]} ({pct}%) remaining')
else:
    report_lines.append('  No missing values detected')

report_lines.append(f'')
report_lines.append('Outlier Detection:')
report_lines.append('-' * 30)
report_lines.append(f'  Method: Isolation Forest + IQR (1.5x)')

error_records = [r for r in outlier_records if r['verdict'] == 'Likely Error']
real_records = [r for r in outlier_records if r['verdict'] == 'Likely Real']
uncertain_records = [r for r in outlier_records if r['verdict'] == 'Uncertain']

if error_records:
    report_lines.append(f'  Likely Error (fixed): {len(error_records)} rows')
    for r in error_records[:5]:  # Top 5
        report_lines.append(f'    - row {r["row_index"]}: {r["column_name"]}={r["value"]} (capped to median)')
    if len(error_records) > 5:
        report_lines.append(f'    ... and {len(error_records)-5} more')
else:
    report_lines.append('  Likely Error (fixed): None')

if real_records:
    report_lines.append(f'  Likely Real (flagged): {len(real_records)} rows')
    for r in real_records[:5]:
        report_lines.append(f'    - row {r["row_index"]}: {r["column_name"]}={r["value"]} (is_outlier=1)')
    if len(real_records) > 5:
        report_lines.append(f'    ... and {len(real_records)-5} more')
else:
    report_lines.append('  Likely Real (flagged): None')

if uncertain_records:
    report_lines.append(f'  Uncertain (flagged): {len(uncertain_records)} rows')
    for r in uncertain_records[:3]:
        report_lines.append(f'    - row {r["row_index"]}: {r["reason"]}')
    if len(uncertain_records) > 3:
        report_lines.append(f'    ... and {len(uncertain_records)-3} more')
else:
    report_lines.append('  Uncertain (flagged): None')

report_lines.append(f'')
report_lines.append('Data Quality Score:')
report_lines.append('-' * 30)
report_lines.append(f'  Completeness: {completeness_before:.2f}% -> {completeness_after:.2f}%')
report_lines.append(f'  Validity: {validity_before:.2f}% -> {validity_after:.2f}%')
report_lines.append(f'  Overall: {overall_before:.2f}% -> {overall_after:.2f}%')
report_lines.append(f'')
report_lines.append('Column Stats (Before -> After):')
report_lines.append('-' * 30)
for col in feat_cols[:5]:
    if col in df_original.columns and col in df.columns:
        before_mean = df_original[col].mean()
        after_mean = df[col].mean()
        before_std = df_original[col].std()
        after_std = df[col].std()
        report_lines.append(f'  {col}: mean {before_mean:.2f}->{after_mean:.2f}, std {before_std:.2f}->{after_std:.2f}')

report_lines.append(f'')
report_lines.append('New Method Found: None')

report_content = '\n'.join(report_lines)
report_path = os.path.join(OUTPUT_DIR, 'dana_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f'  Report saved: {report_path}')

# ── STEP 12: Save cleaned data ──
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')
print(f'[STATUS] Cleaning complete!')

# ── Agent Report ──
print('\n' + '='*60)
print('Agent Report — Dana')
print('='*60)
print(f'Input      : {INPUT_PATH}')
print(f'Output dir : {OUTPUT_DIR}')
print(f'Rows       : {df_original.shape[0]} -> {df.shape[0]}')
print(f'Columns    : {df_original.shape[1]} -> {df.shape[1]}')
print(f'New cols   : purchase_year, purchase_month, purchase_dayofweek, purchase_hour, is_outlier')
print(f'Outliers   : {len(outlier_records)} (Error={len(error_records)}, Real={len(real_records)}, Uncertain={len(uncertain_records)})')
print(f'Quality    : {overall_before:.2f}% -> {overall_after:.2f}%')
print('='*60)

# ── Self-Improvement Report ──
print('\n[STATUS] Self-Improvement Report:')
print('  - Added safe column existence checks to prevent KeyError')
print('  - Domain bounds set for e-commerce data instead of medical')
print('  - Excluded date features from KNN imputation to avoid contamination')
print('  - Outlier detection handles case sensitivity in column names')
