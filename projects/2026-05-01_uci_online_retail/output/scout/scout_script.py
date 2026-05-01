import argparse, os, json, sys
import pandas as pd
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir

# ถ้า input path ว่างหรือเป็น .md ให้ค้นหา CSV
if not INPUT_PATH or INPUT_PATH.endswith('.md'):
    base = Path(OUTPUT_DIR).parent.parent
    csvs = sorted(base.glob('input/**/scout_output.csv')) + sorted(base.glob('input/**/*.csv'))
    if csvs:
        INPUT_PATH = str(csvs[0])
        print(f'[STATUS] Auto-detected input: {INPUT_PATH}')
    else:
        # fallback path จาก task
        INPUT_PATH = r"C:\Users\Amorntep\DATA-Agent\projects\UCI ML\input\scout_output - scout_output.csv"
        print(f'[STATUS] Fallback input path: {INPUT_PATH}')

if not os.path.exists(INPUT_PATH):
    print(f'[ERROR] Input file not found: {INPUT_PATH}')
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD DATA ==========
df = pd.read_csv(INPUT_PATH, encoding='utf-8', low_memory=False)
print(f'[STATUS] Loaded: {df.shape}')

# ========== BASIC PROFILE ==========
n_rows, n_cols = df.shape
print(f'[STATUS] rows={n_rows:,}, cols={n_cols}')

dtype_counts = {}
for dtype in df.dtypes.unique():
    count = (df.dtypes == dtype).sum()
    dtype_name = str(dtype)
    dtype_counts[dtype_name] = count

n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object', 'category', 'str']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]

print(f'[STATUS] dtypes: numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}')

# ========== COLUMNS DETAIL ==========
col_summary = []
for col in df.columns:
    dtype = str(df[col].dtype)
    n_uniq = df[col].nunique()
    miss_pct = round(df[col].isna().mean() * 100, 2)
    sample_vals = df[col].dropna().unique()[:3].tolist()
    col_summary.append({
        'name': col,
        'dtype': dtype,
        'n_unique': n_uniq,
        'missing_pct': miss_pct,
        'sample_values': [str(v) for v in sample_vals]
    })

# ========== DATE/TIME COLUMNS ==========
date_cols = []
for col in df.columns:
    col_l = col.lower()
    if any(kw in col_l for kw in ['date', 'time', 'timestamp', 'day', 'month', 'year', 'hour']):
        date_cols.append(col)
        print(f'[STATUS] Date/time candidate: {col}')

# ========== BUSINESS KEY CANDIDATES ==========
key_cols = []
for col in df.columns:
    col_l = col.lower()
    if col_l.endswith('_id') or col_l.startswith('id_') or col_l == 'id' or 'key' in col_l:
        key_cols.append(col)
        print(f'[STATUS] Key column: {col}')

# ========== TARGET CANDIDATES (auto-detect) ==========
FORBIDDEN_TARGET_SUFFIXES = [
    '_cm', '_g', '_mm', '_kg', '_lb',
    '_length', '_lenght', '_width', '_height',
    '_lat', '_lng', '_latitude', '_longitude',
    '_zip', '_prefix', '_code',
]
FORBIDDEN_TARGET_KEYWORDS = [
    'zip_code', 'zip_prefix', 'geolocation', 'latitude', 'longitude',
    'product_id', 'order_id', 'customer_id', 'seller_id', 'review_id',
    'product_name_lenght', 'product_description_lenght',
    'product_weight_g', 'product_length_cm', 'product_height_cm',
    'product_width_cm', 'product_photos_qty',
]

def is_forbidden_target(col):
    col_l = col.lower()
    if col_l in [k.lower() for k in FORBIDDEN_TARGET_KEYWORDS]:
        return True
    if any(col_l.endswith(s) for s in FORBIDDEN_TARGET_SUFFIXES):
        return True
    if col_l.endswith('_id') or col_l.startswith('id_'):
        return True
    return False

BUSINESS_TARGET_KEYWORDS = [
    "review_score", "order_status", "payment_value", "freight_value",
    "delivery_days", "delay", "churn",
    "target", "label", "survived", "fraud", "default", "outcome",
    "result", "response", "converted", "clicked", "bought",
    "cancelled", "returned", "status", "class",
]

target_col = None
for kw in BUSINESS_TARGET_KEYWORDS:
    for col in df.columns:
        if col.lower() == kw or col.lower().startswith(kw):
            if not is_forbidden_target(col):
                target_col = col
                print(f'[STATUS] Target selected (business keyword): {target_col}')
                break
    if target_col:
        break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            target_col = col
            print(f'[STATUS] Target selected (binary column): {target_col}')
            break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10:
            target_col = col
            print(f'[STATUS] Target selected (categorical): {target_col}')
            break

if not target_col:
    for col in reversed(df.columns):
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            target_col = col
            print(f'[STATUS] Target selected (numeric low-cardinality): {target_col}')
            break

if not target_col:
    print(f'[WARN] ไม่พบ target column ที่เหมาะสม')

# ========== PROBLEM TYPE DETECTION ==========
problem_type = "unknown"
imbalance = None
class_dist = {}

if target_col:
    n_uniq = df[target_col].nunique()
    if n_uniq <= 20:
        problem_type = "classification"
        vc = df[target_col].value_counts(normalize=True).round(4)
        class_dist = vc.to_dict()
        majority = vc.max()
        minority = vc.min()
        imbalance = round(majority / minority, 2) if minority > 0 else None
    else:
        date_col_names = df.select_dtypes(include=['datetime', 'object']).columns
        has_date = any('date' in c.lower() or 'time' in c.lower() for c in date_col_names)
        problem_type = "time_series" if has_date else "regression"
elif df.select_dtypes(include='number').shape[1] >= 2:
    problem_type = "clustering"

# ========== RECOMMENDED PREPROCESSING ==========
if problem_type in ("classification", "regression"):
    scaling = "StandardScaler" if n_numeric > 0 else "None"
elif problem_type == "time_series":
    scaling = "MinMaxScaler"
else:
    scaling = "StandardScaler"

# ========== MISSING PROFILE ==========
miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(5).round(2).to_dict()

# ========== WRITE PROFILE ==========
profile_lines = [
    "DATASET_PROFILE",
    "===============",
    f"rows         : {n_rows:,}",
    f"cols         : {n_cols}",
    f"dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}",
    f"missing      : {json.dumps(top_miss, ensure_ascii=False)}",
    f"target_column: {target_col or 'unknown'}",
    f"problem_type : {problem_type}",
]
if class_dist:
    profile_lines.append(f"class_dist   : {json.dumps({str(k): v for k, v in list(class_dist.items())[:6]})}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {scaling}")

profile_text = "\n".join(profile_lines)
print(profile_text)

profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f'[STATUS] Profile saved: {profile_path}')

# ========== SAVE CSV (pass-through) ==========
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False)
print(f'[STATUS] Saved: {out_csv}')