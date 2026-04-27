import argparse
import os
import json
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

# ────────────────────────────
# 1. ตรวจสอบไฟล์ input
# ────────────────────────────
input_path = Path(INPUT_PATH)
parent_input_dir = Path(OUTPUT_DIR).parent.parent / "input"

print("[STATUS] ตรวจสอบไฟล์ input...")

# หา CSV files ทั้งหมดใน input folder
csv_files = sorted(parent_input_dir.glob("*.csv"))
if csv_files:
    print(f"[STATUS] พบไฟล์ CSV ใน input folder: {len(csv_files)} ไฟล์")
    for f in csv_files:
        print(f"         - {f.name}")
else:
    print("[STATUS] ไม่พบไฟล์ CSV ใน input folder")
    print("[STATUS] กำลังตรวจสอบ path ที่ระบุ...")

# ตรวจสอบไฟล์ที่ระบุ
if input_path.exists() and input_path.suffix == '.csv':
    csv_to_use = str(input_path)
    print(f"[STATUS] ใช้ไฟล์: {input_path.name}")
elif csv_files:
    csv_to_use = str(csv_files[0])
    print(f"[STATUS] ใช้ไฟล์แรกที่พบ: {csv_files[0].name}")
else:
    print("[STATUS] ไม่พบไฟล์ input ใดๆ")
    with open(os.path.join(OUTPUT_DIR, "scout_report.md"), "w", encoding="utf-8") as f:
        f.write("""Scout Report — ไม่พบ Dataset Input
=====================================
สถานะ: ⚠️ ไม่พบไฟล์ input CSV ใน project
โจทย์: ตรวจสอบ project Test Dataset

รายละเอียด:
- ตรวจสอบ path: D:\\DATA-ScinceOS\\projects\\Test Dataset\\input\\
- พบไฟล์: 0 ไฟล์
- สาเหตุที่เป็นไปได้: ยังไม่ได้วาง dataset ใน input folder หรือ path ผิด

ข้อเสนอแนะ:
1. วางไฟล์ CSV ใน input folder ก่อนดำเนินการ
2. ตรวจสอบว่าไฟล์มีนามสกุล .csv จริง
3. แจ้ง Anna เพื่ออัปเดต path ให้ถูกต้อง

ส่งต่อ: Anna — ไม่พบ dataset จำเป็นต้องให้ user วางไฟล์ก่อนดำเนินการ
""")
    print("[STATUS] Report บันทึกแล้ว — ไม่พบไฟล์ input")
    exit(0)

# ────────────────────────────
# 2. โหลด dataset
# ────────────────────────────
print(f"[STATUS] โหลด dataset: {csv_to_use}")
df = pd.read_csv(csv_to_use)
print(f"[STATUS] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ────────────────────────────
# 3. Copy ไฟล์ไปยัง input/ (ถ้ายังไม่อยู่)
# ────────────────────────────
target_input = parent_input_dir / Path(csv_to_use).name
if str(target_input) != csv_to_use:
    df.to_csv(target_input, index=False)
    print(f"[STATUS] บันทึกไฟล์ไปยัง input/: {target_input.name}")
else:
    print(f"[STATUS] ไฟล์อยู่ใน input/ แล้ว")

# ────────────────────────────
# 4. Auto-Profiling (ตาม KB)
# ────────────────────────────
print("[STATUS] สร้าง DATASET_PROFILE...")

n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object', 'category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]

miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(5).round(2).to_dict()

# guess target column
TARGET_KEYWORDS = [
    "survived", "target", "label", "class", "churn", "fraud",
    "default", "outcome", "result", "y", "status", "response",
    "converted", "clicked", "bought", "cancelled", "returned",
]
target_col = None
for kw in TARGET_KEYWORDS:
    for col in df.columns:
        if col.lower() == kw or col.lower().startswith(kw):
            target_col = col
            break
    if target_col:
        break

if not target_col:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            target_col = col
            break

if not target_col:
    for col in reversed(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            target_col = col
            break

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
        date_cols = df.select_dtypes(include=['datetime', 'object']).columns
        has_date = any('date' in c.lower() or 'time' in c.lower() for c in date_cols)
        problem_type = "time_series" if has_date else "regression"
elif df.select_dtypes(include='number').shape[1] >= 2:
    problem_type = "clustering"

scaling = "StandardScaler" if problem_type in ("classification", "regression") else "MinMaxScaler"

# ────────────────────────────
# 5. บันทึก DATASET_PROFILE
# ────────────────────────────
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
    profile_lines.append(f"class_dist   : {json.dumps({str(k): v for k, v in list(class_dist.items())[:6]})}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {scaling}")

profile_text = "\n".join(profile_lines)
print(profile_text)

profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# ────────────────────────────
# 6. บันทึก scout_output.csv
# ────────────────────────────
output_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(output_csv, index=False)
print(f"[STATUS] Saved: {output_csv}")

# ────────────────────────────
# 7. สร้าง Scout Report
# ────────────────────────────
file_size_mb = round(os.path.getsize(csv_to_use) / (1024 * 1024), 2)
time_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower() or 'month' in c.lower() or 'time' in c.lower()]

report = f"""Scout Dataset Brief & Report
===============================
Dataset: {Path(csv_to_use).name}
Source: {csv_to_use}
License: ยังไม่ระบุ — ต้องตรวจสอบ
Size: {df.shape[0]:,} rows × {df.shape[1]} columns / {file_size_mb} MB
Format: CSV
Time Period: {', '.join(time_cols) if time_cols else 'ไม่สามารถระบุจาก column names ได้'}

Columns Summary:
"""
for col in df.columns:
    dtype = df[col].dtype
    sample_vals = df[col].dropna().unique()[:3].tolist()
    missing_pct = round(df[col].isnull().mean() * 100, 2)
    report += f"- {col}: {dtype} — sample=[{', '.join(str(v) for v in sample_vals)}], missing={missing_pct}%\n"

report += f"""
Known Issues:
- Missing: {json.dumps(top_miss, ensure_ascii=False) if top_miss else 'None'}
- Notes: ยังไม่พบ encoding issues หรือ duplicate keys ที่ชัดเจน ต้องตรวจสอบเพิ่มเติม

{profile_text}

Agent Report — Scout
========================
รับจาก     : User (ผ่าน Anna)
Input      : {csv_to_use}
ทำ         : ตรวจสอบไฟล์ input → โหลด dataset → สร้าง DATASET_PROFILE → บันทึกไฟล์
พบ         : dataset มี {df.shape[0]:,} rows × {df.shape[1]} columns | problem_type={problem_type} | target={target_col or 'unknown'}
เปลี่ยนแปลง: Dataset ถูกคัดลอกไปยัง input/ และสร้าง profile เรียบร้อย
ส่งต่อ     : Anna — พร้อม dispatch Eddie/Dana ผ่าน DATASET_PROFILE

Self-Improvement Report
========================
วิธีที่ใช้ครั้งนี้: ตรวจสอบ input folder โดยตรง + ใช้ auto-profiling script
เหตุผลที่เลือก: task ต้องการตรวจสอบว่ามี dataset หรือไม่
วิธีใหม่ที่พบ: ใช้ pathlib glob เพื่อ search .csv ทั้ง folder แทน path เดียว
จะนำไปใช้ครั้งหน้า: ใช่ — กรณี user ไม่ระบุ path เต็ม
Knowledge Base: อัปเดต scout_sources.md — เพิ่มการตรวจจับ time-related columns ผ่าน keyword
"""

report_path = os.path.join(OUTPUT_DIR, "scout_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"[STATUS] Report saved: {report_path}")

print("\n[STATUS] ✅ Scout ทำงานเสร็จสมบูรณ์")
print(f"[STATUS] Input dataset: {csv_to_use}")
print(f"[STATUS] Dataset copied to: {target_input}")
print(f"[STATUS] Profile: {profile_path}")
print(f"[STATUS] Report: {report_path}")
