import argparse
import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import IsolationForest
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir

if not INPUT_PATH or not os.path.exists(INPUT_PATH):
    print(f'[ERROR] --input required and must exist: {INPUT_PATH}')
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('-' * 60)
print('[STATUS] Starting Dana — Olist Data Cleaner')
print('-' * 60)

# ── STEP 1: Load SQLite database ──
conn = sqlite3.connect(INPUT_PATH)
print(f'[STATUS] Connected to SQLite DB: {INPUT_PATH}')

# Load all 9 tables
customers = pd.read_sql_query("SELECT * FROM customers", conn)
geolocation = pd.read_sql_query("SELECT * FROM geolocation", conn)
orders = pd.read_sql_query("SELECT * FROM orders", conn)
order_items = pd.read_sql_query("SELECT * FROM order_items", conn)
payments = pd.read_sql_query("SELECT * FROM order_payments", conn)
reviews = pd.read_sql_query("SELECT * FROM order_reviews", conn)
products = pd.read_sql_query("SELECT * FROM products", conn)
sellers = pd.read_sql_query("SELECT * FROM sellers", conn)
category_translation = pd.read_sql_query("SELECT * FROM product_category_name_translation", conn)
conn.close()
print('[STATUS] All 9 tables loaded')

# ── STEP 2: Aggregate geolocation (avg lat/long per zip_code_prefix) ──
# Drop duplicates first — geolocation has many duplicate rows per zip
geolocation_agg = geolocation.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()
geolocation_agg.columns = ['zip_code_prefix', 'avg_lat', 'avg_lng']
print(f'[STATUS] Geolocation aggregated: {len(geolocation_agg)} unique zip codes')

# ── STEP 3: Join all tables into one DataFrame ──

# Join orders → order_items → payments → reviews
base = orders.merge(order_items, on='order_id', how='left')
base = base.merge(payments, on='order_id', how='left')
base = base.merge(reviews, on='order_id', how='left')

# Join products
base = base.merge(products, on='product_id', how='left')

# Join sellers
base = base.merge(sellers, on='seller_id', how='left')

# Join customers
base = base.merge(customers, on='customer_id', how='left')

# Join geolocation for customer zip
# customers have customer_zip_code_prefix — map to avg lat/lng
zip_lat_map = dict(zip(geolocation_agg['zip_code_prefix'], geolocation_agg['avg_lat']))
zip_lng_map = dict(zip(geolocation_agg['zip_code_prefix'], geolocation_agg['avg_lng']))
base['customer_avg_lat'] = base['customer_zip_code_prefix'].map(zip_lat_map)
base['customer_avg_lng'] = base['customer_zip_code_prefix'].map(zip_lng_map)

# Map seller zip to avg lat/lng
seller_zip_col = 'seller_zip_code_prefix'
if seller_zip_col in base.columns:
    base['seller_avg_lat'] = base[seller_zip_col].map(zip_lat_map)
    base['seller_avg_lng'] = base[seller_zip_col].map(zip_lng_map)

# Join category translation
base = base.merge(category_translation, on='product_category_name', how='left')

df_original = base.copy()
print(f'[STATUS] Joined DataFrame: {base.shape}')

# ── STEP 4: Drop high-missing columns and rows ──

# Drop review_comment_title (88% missing)
if 'review_comment_title' in base.columns:
    base = base.drop(columns=['review_comment_title'])
    print('[STATUS] Dropped: review_comment_title (88% missing)')

# Drop columns that are all NaN or IDs that don't help cleaning
cols_to_drop = []
for col in base.columns:
    missing_pct = base[col].isnull().mean()
    if missing_pct > 0.85:
        cols_to_drop.append(col)
        print(f'[STATUS] Dropped: {col} ({missing_pct:.0%} missing)')

if cols_to_drop:
    base = base.drop(columns=cols_to_drop)

# Drop rows where order_id is null (no order data)
base = base.dropna(subset=['order_id'])
print(f'[STATUS] After drop: {base.shape}')

# ── STEP 5: Feature selection for numeric columns ──
numeric_cols = base.select_dtypes(include=[np.number]).columns.tolist()

# Filter to columns that have at least some missing values worth imputing
impute_cols = [c for c in numeric_cols if base[c].isnull().sum() > 0]
print(f'[STATUS] Columns needing imputation: {impute_cols}')

# ── STEP 6: KNN Imputation ──
if impute_cols:
    # Ensure all columns are numeric
    for col in impute_cols:
        base[col] = pd.to_numeric(base[col], errors='coerce')

    imputer = KNNImputer(n_neighbors=5)
    imputed_array = imputer.fit_transform(base[impute_cols])
    base[impute_cols] = pd.DataFrame(imputed_array, columns=impute_cols, index=base.index)
    
    # Clip to non-negative (safe for all numeric columns)
    for col in impute_cols:
        base[col] = base[col].clip(lower=0)
    
    print(f'[STATUS] KNN Imputation complete for {len(impute_cols)} columns')

# ── STEP 7: Outlier Detection ──
feat_cols = [c for c in numeric_cols if c in base.columns and c not in ['is_outlier']]
outlier_records = []

# IQR Detection
for col in feat_cols:
    q1 = base[col].quantile(0.25)
    q3 = base[col].quantile(0.75)
    iqr = q3 - q1
    lo_b = q1 - 1.5 * iqr
    hi_b = q3 + 1.5 * iqr
    
    outliers_in_col = base[(base[col] < lo_b) | (base[col] > hi_b)].index
    for idx in outliers_in_col:
        val = base.loc[idx, col]
        # Determine if likely error or real extreme
        if val < 0:
            verdict = 'Likely Error'
            action = 'flagged'
        elif val > hi_b * 3:  # Extreme beyond 3x IQR
            verdict = 'Likely Error'
            action = 'flagged'
        else:
            verdict = 'Likely Real'
            action = 'flagged'
        
        outlier_records.append({
            'row_index': idx,
            'column_name': col,
            'value': val,
            'verdict': verdict,
            'reason': f'IQR outlier: {col}={val:.4f} (bounds: {lo_b:.4f}, {hi_b:.4f})',
            'action': action
        })

# Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
iso_array = base[feat_cols].fillna(base[feat_cols].median()).values
iso_mask = iso.fit_predict(iso_array) == -1

for idx in base.index[iso_mask]:
    if not any(r['row_index'] == idx for r in outlier_records):
        outlier_records.append({
            'row_index': idx,
            'column_name': 'multivariate',
            'value': None,
            'verdict': 'Uncertain',
            'reason': 'Isolation Forest anomaly',
            'action': 'flagged'
        })

print(f'[STATUS] Outlier detection: {len(outlier_records)} outliers found')

# Add is_outlier flag
base['is_outlier'] = 0
for r in outlier_records:
    base.loc[r['row_index'], 'is_outlier'] = 1

# ── STEP 8: Save outputs ──
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
base.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# Save outlier_flags.csv
flags_df = pd.DataFrame(outlier_records)
if len(flags_df) > 0:
    flags_csv = os.path.join(OUTPUT_DIR, 'outlier_flags.csv')
    flags_df.to_csv(flags_csv, index=False)
    print(f'[STATUS] Saved: {flags_csv}')
else:
    print('[STATUS] No outliers — skipping outlier_flags.csv')

# ── STEP 9: Data Quality Score ──
n_before = len(df_original)
n_after = len(base)
missing_before = df_original.isnull().sum().sum()
missing_after = base.isnull().sum().sum()

completeness_before = (1 - missing_before / (n_before * len(df_original.columns))) * 100 if n_before > 0 else 100
completeness_after = (1 - missing_after / (n_after * len(base.columns))) * 100 if n_after > 0 else 100

likely_errors = sum(1 for r in outlier_records if r['verdict'] == 'Likely Error')
validity_before = 100
validity_after = (1 - likely_errors / max(n_after, 1)) * 100

overall_before = 0.5 * completeness_before + 0.5 * validity_before
overall_after = 0.5 * completeness_after + 0.5 * validity_after

print(f'[STATUS] Quality: {overall_before:.1f}% -> {overall_after:.1f}%')

# ── STEP 10: Generate Report ──
report_lines = []
report_lines.append('Dana Cleaning Report')
report_lines.append('=' * 60)
report_lines.append('')
report_lines.append(f'Before: {n_before} rows, {len(df_original.columns)} columns')
report_lines.append(f'After: {n_after} rows, {len(base.columns)} columns')
report_lines.append('')
report_lines.append('Missing Values:')
report_lines.append('-' * 40)

dropped_high_missing = [c for c in df_original.columns if c not in base.columns]
for col in dropped_high_missing:
    report_lines.append(f'- {col}: Dropped (>85% missing)')

missing_before_cols = df_original.isnull().sum()
missing_after_cols = base.isnull().sum()
for col in base.columns:
    mb = missing_before_cols.get(col, 0)
    ma = missing_after_cols.get(col, 0)
    if mb > 0:
        report_lines.append(f'- {col}: {mb} missing -> {ma} (KNN Imputation)')
    if mb == 0:
        pass  # No missing

if len(dropped_high_missing) == 0:
    report_lines.append('No columns dropped due to missing >85%')

report_lines.append('')
report_lines.append('Outlier Detection:')
report_lines.append('-' * 40)
report_lines.append(f'Method: Isolation Forest (contamination=0.05) + IQR (1.5x)')
report_lines.append(f'Total outliers found: {len(outlier_records)}')

likely_errors_out = [r for r in outlier_records if r['verdict'] == 'Likely Error']
likely_real = [r for r in outlier_records if r['verdict'] == 'Likely Real']
uncertain = [r for r in outlier_records if r['verdict'] == 'Uncertain']

report_lines.append(f'- Likely Error (flagged): {len(likely_errors_out)} rows')
if likely_errors_out:
    for r in likely_errors_out[:5]:
        report_lines.append(f'  * row {r["row_index"]}: {r["column_name"]}={r["value"]:.2f} ({r["reason"]})')
    if len(likely_errors_out) > 5:
        report_lines.append(f'  * ... and {len(likely_errors_out)-5} more')

report_lines.append(f'- Likely Real (flagged): {len(likely_real)} rows')
if likely_real:
    for r in likely_real[:5]:
        report_lines.append(f'  * row {r["row_index"]}: {r["column_name"]}={r["value"]:.2f} ({r["reason"]})')
    if len(likely_real) > 5:
        report_lines.append(f'  * ... and {len(likely_real)-5} more')

report_lines.append(f'- Uncertain: {len(uncertain)} rows')
if uncertain:
    for r in uncertain[:5]:
        report_lines.append(f'  * row {r["row_index"]}: {r["column_name"]} ({r["reason"]})')

if len(outlier_records) == 0:
    report_lines.append('None — data is clean')

report_lines.append('')
report_lines.append(f'outlier_flags.csv: {len(outlier_records)} rows')
report_lines.append('')
report_lines.append('Data Quality Score:')
report_lines.append('-' * 40)
report_lines.append(f'Completeness: Before {completeness_before:.1f}% -> After {completeness_after:.1f}%')
report_lines.append(f'Validity: Before {validity_before:.1f}% -> After {validity_after:.1f}%')
report_lines.append(f'Overall: Before {overall_before:.1f}% -> After {overall_after:.1f}%')
report_lines.append('')
report_lines.append('Column Stats (Before -> After):')
report_lines.append('-' * 40)

for col in numeric_cols[:10]:
    before_mean = df_original[col].mean() if col in df_original.columns else 0
    after_mean = base[col].mean() if col in base.columns else 0
    before_std = df_original[col].std() if col in df_original.columns else 0
    after_std = base[col].std() if col in base.columns else 0
    report_lines.append(f'- {col}: mean {before_mean:.2f} -> {after_mean:.2f}, std {before_std:.2f} -> {after_std:.2f}')

report_lines.append('')
report_lines.append('New Method Found: None')

report_content = '\n'.join(report_lines)
report_path = os.path.join(OUTPUT_DIR, 'dana_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f'[STATUS] Saved: {report_path}')

# ── Agent Report ──
print('')
print('Agent Report — Dana')
print('=' * 60)
print(f'  รับจาก     : User / File path')
print(f'  Input      : {INPUT_PATH} (SQLite, 9 tables)')
print(f'  ทำ         : Load 9 tables -> join -> aggregate geolocation -> drop high-missing columns -> KNN imputation -> outlier detection')
print(f'  พบ         : {n_after} rows after cleaning, {len(outlier_records)} outliers detected')
print(f'  เปลี่ยนแปลง: review_comment_title dropped, numeric missing imputed via KNN, outlier flags added')
print(f'  ส่งต่อ     : dana_output.csv -> next agent (Eddie/Finn/Mo)')

print('-' * 60)
print('[STATUS] DONE')
