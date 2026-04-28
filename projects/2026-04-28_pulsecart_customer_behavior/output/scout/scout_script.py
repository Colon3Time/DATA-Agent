import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\input\pulsecart_raw.csv"
OUTPUT_DIR = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\scout"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ตรวจสอบว่าไฟล์มีจริง
if not os.path.exists(INPUT_PATH):
    print(f"[ERROR] ไม่พบไฟล์: {INPUT_PATH}")
    input_dir = Path(INPUT_PATH).parent
    files = sorted(input_dir.glob("*.*"))
    print(f"[STATUS] ไฟล์ใน input/: {[f.name for f in files]}")
    exit(1)

df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape}")
print(f"[STATUS] Columns ({len(df.columns)}): {list(df.columns)}")
print(f"[STATUS] Dtypes:\n{df.dtypes}")
print(f"[STATUS] First 5 rows:\n{df.head(5).to_string()}")

# --- dtypes breakdown ---
n_numeric  = df.select_dtypes(include='number').shape[1]
n_cat      = df.select_dtypes(include=['object','category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]
print(f"[STATUS] dtypes: numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}")

# --- missing ---
miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(5).round(2).to_dict()
print(f"[STATUS] Top missing: {top_miss}")

# --- unique values per column ---
uniqs = {c: df[c].nunique() for c in df.columns}
print(f"[STATUS] Unique values: {uniqs}")

# --- sample values (first 8 cols) ---
sample_vals = {}
for c in df.columns[:8]:
    vals = df[c].dropna().unique()[:5]
    sample_vals[c] = list(vals)
print(f"[STATUS] Sample values: {sample_vals}")

# --- basic stats for numeric ---
num_cols = df.select_dtypes(include='number').columns
if len(num_cols) > 0:
    print(f"[STATUS] Numeric stats:\n{df[num_cols].describe().to_string()}")

# ===== Auto-Profiling =====

# --- คอลัมน์ที่ห้ามเป็น target เด็ดขาด ---
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

# Business target keywords (เรียงตามความสำคัญ)
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
    print(f"[WARN] ไม่พบ target column ที่เหมาะสม")

# --- problem type detection ---
problem_type = "unknown"
imbalance    = None
class_dist   = {}
if target_col:
    n_uniq = df[target_col].nunique()
    if n_uniq <= 20:
        problem_type = "classification"
        vc = df[target_col].value_counts(normalize=True).round(4)
        class_dist = vc.to_dict()
        majority   = vc.max()
        minority   = vc.min()
        imbalance  = round(majority / minority, 2) if minority > 0 else None
    else:
        date_cols = df.select_dtypes(include=['datetime','object']).columns
        has_date  = any('date' in c.lower() or 'time' in c.lower() for c in date_cols)
        problem_type = "time_series" if has_date else "regression"
elif df.select_dtypes(include='number').shape[1] >= 2:
    problem_type = "clustering"

# --- recommended scaling ---
if problem_type in ("classification","regression"):
    scaling = "StandardScaler" if n_numeric > 0 else "None"
elif problem_type == "time_series":
    scaling = "MinMaxScaler"
else:
    scaling = "StandardScaler"

# --- write DATASET_PROFILE ---
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
    # แปลง keys เป็น string เพื่อ json serialize ได้
    class_dist_str = {str(k): v for k, v in list(class_dist.items())[:6]}
    profile_lines.append(f"class_dist   : {json.dumps(class_dist_str)}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {scaling}")

profile_text = "\n".join(profile_lines)
print(profile_text)

profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# --- save pass-through CSV ---
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False)
print(f"[STATUS] Saved: {out_csv}")

# --- สร้าง Dataset Brief ---
brief_lines = [
    "Scout Dataset Brief",
    "===================",
    f"Dataset: PulseCart Customer Behavior (raw)",
    f"Source: pulsecart_raw.csv (local input)",
    f"License: ไม่ระบุ (internal dataset)",
    f"Size: {df.shape[0]:,} rows x {df.shape[1]} columns",
    f"Format: CSV",
    "",
    "Columns Summary:"
]
for col in df.columns:
    dtype = str(df[col].dtype)
    uniq = df[col].nunique()
    miss_pct = round(df[col].isnull().mean() * 100, 1)
    brief_lines.append(f"- {col}: {dtype} — unique={uniq}, missing={miss_pct}%")

brief_lines.extend([
    "",
    "Known Issues:",
    f"- Missing columns: {json.dumps(top_miss) if top_miss else 'none'}",
    "",
    f"[DATASET_PROFILE block]",
    profile_text
])

brief_text = "\n".join(brief_lines)
brief_path = os.path.join(OUTPUT_DIR, "scout_report.md")
with open(brief_path, "w", encoding="utf-8") as f:
    f.write(brief_text)
print(f"[STATUS] Brief saved: {brief_path}")

# --- สร้าง Agent Report ---
agent_report_lines = [
    "Agent Report — Scout",
    "=====================",
    f"รับจาก     : User (โดยตรง)",
    f"Input      : {INPUT_PATH}",
    f"ทำ         : 1) โหลด pulsecart_raw.csv",
    f"             2) auto-profile dataset (dtypes, missing, stats)",
    f"             3) detect target column: {target_col or 'unknown'}",
    f"             4) detect problem type: {problem_type}",
    f"             5) สร้าง DATASET_PROFILE + Dataset Brief",
    f"พบ         : 1) dataset มี {df.shape[0]:,} rows x {df.shape[1]} cols",
    f"             2) target column ที่ตรงกับ business keyword มากที่สุด: {target_col or 'ไม่พบ (ต้องใช้ unknown ไปก่อน)'}",
    f"             3) missing cols: {list(top_miss.keys()) if top_miss else 'none'}",
    f"เปลี่ยนแปลง: raw CSV → profiled dataset พร้อมใช้งาน",
    f"ส่งต่อ     : Anna — ไฟล์ dataset_profile.md + scout_report.md",
    "",
    "Self-Improvement Report",
    "=======================",
    "วิธีที่ใช้ครั้งนี้: โหลด raw CSV, auto-profile ด้วย script",
    "เหตุผลที่เลือก: dataset อยู่ใน local input folder",
    "วิธีใหม่ที่พบ: -",
    "จะนำไปใช้ครั้งหน้า: ใช่",
    "Knowledge Base: ไม่มีการเปลี่ยนแปลง"
]

agent_report_text = "\n".join(agent_report_lines)
agent_report_path = os.path.join(OUTPUT_DIR, "agent_report.txt")
with open(agent_report_path, "w", encoding="utf-8") as f:
    f.write(agent_report_text)
print(f"[STATUS] Agent report saved: {agent_report_path}")

print("[STATUS] Done! พร้อมส่งต่อให้ Anna")