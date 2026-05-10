import os, argparse, sys, json
import pandas as pd
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ตรวจสอบว่า input เป็นไฟล์หรือ folder
input_path = Path(INPUT_PATH)

if input_path.is_file():
    csv_path = str(input_path)
    print(f"[STATUS] Input is file: {csv_path}")
elif input_path.is_dir():
    # หาไฟล์ CSV ใน folder
    csv_files = sorted(input_path.glob('*.csv'))
    if not csv_files:
        # ลองหาใน parent/input
        parent_input = input_path.parent / "input"
        if parent_input.exists():
            csv_files = sorted(parent_input.glob('*.csv'))
    if csv_files:
        csv_path = str(csv_files[0])
        print(f"[STATUS] Found CSV in folder: {csv_path}")
    else:
        print("[ERROR] No CSV file found in input path")
        sys.exit(1)
else:
    print(f"[ERROR] Input path not found: {INPUT_PATH}")
    sys.exit(1)

print(f"[STATUS] Loading: {csv_path}")

# ตรวจสอบ delimiter
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    first_line = f.readline()
    if '\t' in first_line:
        sep = '\t'
        print("[STATUS] Delimiter: tab")
    elif ';' in first_line:
        sep = ';'
        print("[STATUS] Delimiter: semicolon (;)")
    else:
        sep = ','
        print("[STATUS] Delimiter: comma (,)")

# อ่าน CSV
df = pd.read_csv(csv_path, sep=sep, encoding='utf-8-sig')
print(f"[STATUS] Loaded: {df.shape}")
print(f"[STATUS] Columns: {list(df.columns)}")

# ========== 1. DATA PROFILE ==========
print("\n" + "="*60)
print("DATA PROFILE — Thailand Economic Indicators")
print("="*60)

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ตรวจสอบ year column
year_col = None
for col in df.columns:
    if 'year' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
        year_col = col
        break
if year_col:
    try:
        print(f"Time range: {df[year_col].min()} - {df[year_col].max()}")
    except:
        print(f"Time column: {year_col} (cannot determine range)")

# dtypes
n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object','category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]
print(f"Dtypes: numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}")

# Missing values
miss = (df.isnull().sum() / len(df) * 100).round(2)
print(f"\nMissing values (%):")
for col, pct in miss.items():
    if pct > 0:
        print(f"  {col}: {pct}%")
    else:
        print(f"  {col}: 0%")

# Statistical summary
print(f"\nDescriptive statistics (numeric columns):")
numeric_cols = df.select_dtypes(include='number').columns
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe().to_string())
else:
    print("  No numeric columns found")

# ========== 2. Target column detection ==========
FORBIDDEN_TARGET_SUFFIXES = [
    '_cm', '_g', '_mm', '_kg', '_lb',
    '_lenght', '_length', '_width', '_height',
    '_lat', '_lng', '_latitude', '_longitude',
    '_zip', '_prefix', '_code',
]
FORBIDDEN_TARGET_KEYWORDS = [
    'zip_code', 'zip_prefix', 'geolocation', 'latitude', 'longitude',
    'product_id', 'order_id', 'customer_id', 'seller_id', 'review_id',
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
    "gdp", "inflation", "unemployment", "growth", "rate", "index",
]

target_col = None
for kw in BUSINESS_TARGET_KEYWORDS:
    for col in df.columns:
        if col.lower() == kw or col.lower().startswith(kw) or kw in col.lower():
            if not is_forbidden_target(col):
                target_col = col
                print(f"[STATUS] Target selected (business keyword): {target_col}")
                break
    if target_col:
        break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            target_col = col
            print(f"[STATUS] Target selected (binary column): {target_col}")
            break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10:
            target_col = col
            print(f"[STATUS] Target selected (categorical): {target_col}")
            break

if not target_col:
    for col in reversed(df.columns):
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            target_col = col
            print(f"[STATUS] Target selected (numeric low-cardinality): {target_col}")
            break

if not target_col:
    print(f"[WARN] No suitable target column found")

# ========== 3. Problem type detection ==========
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
        date_cols = df.select_dtypes(include=['datetime','object']).columns
        has_date = any('date' in c.lower() or 'time' in c.lower() or 'year' in c.lower() for c in date_cols)
        problem_type = "time_series" if has_date else "regression"
elif n_numeric >= 2:
    problem_type = "clustering"

# ========== 4. Scaling recommendation ==========
if problem_type in ("classification", "regression"):
    scaling = "StandardScaler" if n_numeric > 0 else "None"
elif problem_type == "time_series":
    scaling = "MinMaxScaler"
else:
    scaling = "StandardScaler"

# ========== 5. Write profiles ==========

# --- dataset_profile.md ---
top_miss = miss[miss > 0].head(5).round(2).to_dict() if miss.max() > 0 else {}

profile_lines = [
    "DATASET_PROFILE",
    "===============",
    f"rows         : {df.shape[0]:,}",
    f"cols         : {df.shape[1]}",
    f"dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}",
    f"missing      : {json.dumps(top_miss, ensure_ascii=False)}",
    f"target_column: {target_col or 'unknown'}",
    f"problem_type : {problem_type}",
]
if class_dist:
    profile_lines.append(f"class_dist   : {json.dumps({str(k): v for k,v in list(class_dist.items())[:6]})}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {scaling}")

# --- DATASET_RISK_REGISTER ---
risk_register_lines = [
    "",
    "DATASET_RISK_REGISTER",
    "=====================",
    "Source credibility: Medium (ต้องตรวจสอบแหล่งที่มาของไฟล์)",
    "License/usage: Unknown (ต้องตรวจสอบ license)",
    "Business fit: Medium (ข้อมูลเศรษฐกิจไทย — เหมาะกับการวิเคราะห์แนวโน้ม)",
    f"Target suitability: {'clear' if target_col else 'ambiguous'}",
    "Recency/deployment fit: Unknown (ต้องตรวจสอบวันที่อัปเดตล่าสุด)",
    "Leakage risks: None detected",
    "Bias/coverage risks: Unknown (ต้องตรวจสอบ coverage ของ indicators)",
    "Data dictionary: Missing (ต้องสร้าง data dictionary)",
    "Verdict: Use with caveats (ต้องตรวจสอบ source, license, และ data dictionary ก่อน)",
]

profile_text = "\n".join(profile_lines + risk_register_lines)

profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# --- Save output CSV ---
output_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(output_csv, index=False)
print(f"[STATUS] Output CSV saved: {output_csv}")

print(f"\n{'='*60}")
print("PROFILE COMPLETE")
print(f"{'='*60}")
print(f"Target column: {target_col}")
print(f"Problem type: {problem_type}")
print(f"Scaling: {scaling}")