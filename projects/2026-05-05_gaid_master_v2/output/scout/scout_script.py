import argparse, os, sys, json
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

# หาไฟล์ทั้งหมดใน input/
input_dir = Path(INPUT_PATH).parent if INPUT_PATH.endswith('.csv') else Path(INPUT_PATH)
csv_files = sorted(input_dir.glob('*.csv'))
sqlite_files = sorted(input_dir.glob('*.sqlite')) + sorted(input_dir.glob('*.db'))
all_files = csv_files + sqlite_files

print(f"[STATUS] พบไฟล์ทั้งหมด {len(all_files)} ไฟล์ใน {input_dir}")
for f in all_files:
    sz = os.path.getsize(f)
    print(f"  - {f.name} ({sz/1024:.1f} KB)")

# เลือกไฟล์หลัก
if all_files:
    file_to_read = all_files[0]  # GAID_MASTER_V2_COMPILATION_FINAL.csv
else:
    file_to_read = Path(INPUT_PATH)

print(f"[STATUS] เลือกอ่านไฟล์: {file_to_read.name}")

# ================ 1. โหลดข้อมูล ================
if file_to_read.suffix in ('.sqlite', '.db'):
    import sqlite3
    conn = sqlite3.connect(str(file_to_read))
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
    print(f"[STATUS] SQLite tables: {tables}")
    df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", conn) if tables else pd.DataFrame()
else:
    df = pd.read_csv(file_to_read, low_memory=False)

print(f"[STATUS] โหลดข้อมูลสำเร็จ: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ================ 2. Basic Profile ================
profile_lines = []
profile_lines.append(f"# Raw Data Profile — {file_to_read.name}")
profile_lines.append(f"")
profile_lines.append(f"## 1. ภาพรวมไฟล์ทั้งหมดใน input/")
profile_lines.append(f"")
profile_lines.append(f"| ไฟล์ | ขนาด | ชนิด |")
profile_lines.append(f"|------|------|------|")
for f in all_files:
    sz = os.path.getsize(f)
    ext = f.suffix
    profile_lines.append(f"| {f.name} | {sz/1024:.1f} KB | {ext} |")

profile_lines.append(f"")
profile_lines.append(f"## 2. ภาพรวม Dataset หลัก")
profile_lines.append(f"")
profile_lines.append(f"- **ชื่อไฟล์**: {file_to_read.name}")
line_nrows = f"- **จำนวนแถว**: {df.shape[0]:,}"
profile_lines.append(line_nrows)
line_ncols = f"- **จำนวนคอลัมน์**: {df.shape[1]}"
profile_lines.append(line_ncols)
mem_mb = df.memory_usage(deep=True).sum()/1024/1024
line_mem = f"- **ขนาดหน่วยความจำ**: {mem_mb:.1f} MB"
profile_lines.append(line_mem)
profile_lines.append(f"")

# ================ 3. Column Details ================
profile_lines.append(f"## 3. รายละเอียดแต่ละคอลัมน์")
profile_lines.append(f"")
profile_lines.append(f"| ลำดับ | คอลัมน์ | dtype | Missing | Missing % | Unique | Sample Values |")
profile_lines.append(f"|-------|--------|-------|--------|-----------|-------|--------------|")

for i, col in enumerate(df.columns):
    dtype = str(df[col].dtype)
    miss = int(df[col].isnull().sum())
    miss_pct = round(miss / len(df) * 100, 2)
    uniq = df[col].nunique()
    sample_vals = list(df[col].dropna().unique()[:5])
    sample_str = ", ".join([str(v) for v in sample_vals])
    profile_lines.append(f"| {i+1} | {col} | {dtype} | {miss} | {miss_pct}% | {uniq} | {sample_str} |")

profile_lines.append(f"")

# ================ 4. Missing Value Analysis ================
profile_lines.append(f"## 4. การวิเคราะห์ Missing Values")
profile_lines.append(f"")
miss_df = df.isnull().sum().reset_index()
miss_df.columns = ['column', 'missing_count']
miss_df['missing_pct'] = round(miss_df['missing_count'] / len(df) * 100, 2)
miss_df = miss_df[miss_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
if len(miss_df) > 0:
    for _, row in miss_df.iterrows():
        profile_lines.append(f"- **{row['column']}**: {row['missing_count']:,} แถว ({row['missing_pct']}%)")
else:
    profile_lines.append(f"- ไม่มี missing values ใน dataset นี้")
profile_lines.append(f"")

# ================ 5. Numeric Columns Summary ================
profile_lines.append(f"## 5. สรุปคอลัมน์ตัวเลข")
profile_lines.append(f"")
num_cols = df.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    desc = df[num_cols].describe().transpose()
    profile_lines.append(f"| คอลัมน์ | count | mean | std | min | 25% | 50% | 75% | max |")
    profile_lines.append(f"|--------|-------|------|-----|-----|-----|-----|-----|-----|")
    for col in num_cols:
        r = desc.loc[col]
        line = f"| {col} | {r['count']:.0f} | {r['mean']:.2f} | {r['std']:.2f} | {r['min']:.2f} | {r['25%']:.2f} | {r['50%']:.2f} | {r['75%']:.2f} | {r['max']:.2f} |"
        profile_lines.append(line)
else:
    profile_lines.append(f"- ไม่มีคอลัมน์ตัวเลขใน dataset")
profile_lines.append(f"")

# ================ 6. Categorical Columns Summary ================
profile_lines.append(f"## 6. สรุปคอลัมน์ประเภท (Categorical/Object)")
profile_lines.append(f"")
cat_cols = df.select_dtypes(include=['object', 'category']).columns
if len(cat_cols) > 0:
    for col in cat_cols[:10]:  # แสดงแค่ 10 คอลัมน์แรก
        vc = df[col].value_counts().head(5)
        top_vals = ", ".join([f"{k} ({v})" for k, v in vc.items()])
        profile_lines.append(f"- **{col}**: {df[col].nunique()} unique | 5 อันดับแรก: {top_vals}")
else:
    profile_lines.append(f"- ไม่มีคอลัมน์ประเภท object หรือ category")
profile_lines.append(f"")

# ================ 7. Target Column Detection ================
profile_lines.append(f"## 7. Target Column Detection")
profile_lines.append(f"")

# ห้ามเป็น target
FORBIDDEN_TARGET_SUFFIXES = [
    '_cm', '_g', '_mm', '_kg', '_lb',
    '_lenght', '_length', '_width', '_height',
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
                break
    if target_col:
        break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            target_col = col
            break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10:
            target_col = col
            break

if not target_col:
    for col in reversed(df.columns):
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            target_col = col
            break

if target_col:
    profile_lines.append(f"- **Target ที่แนะนำ**: `{target_col}`")
    n_uniq = df[target_col].nunique()
    if n_uniq <= 20:
        vc = df[target_col].value_counts(normalize=True).round(4)
        profile_lines.append(f"- **Problem type**: classification ({n_uniq} classes)")
        dist_str = ", ".join([f"{k}: {v:.1%}" for k, v in vc.items()])
        profile_lines.append(f"- **Class distribution**: {dist_str}")
        if n_uniq == 2:
            maj = vc.max()
            min_v = vc.min()
            imbalance_ratio = round(maj / min_v, 2) if min_v > 0 else None
            if imbalance_ratio and imbalance_ratio > 1.5:
                profile_lines.append(f"- **Imbalance ratio**: {imbalance_ratio} (imbalanced)")
            else:
                profile_lines.append(f"- **Imbalance ratio**: {imbalance_ratio} (balanced)")
    else:
        profile_lines.append(f"- **Problem type**: regression ({n_uniq} unique values)")
else:
    profile_lines.append(f"- **ไม่พบ Target column ที่เหมาะสม** — ต้องระบุด้วยตนเอง")

profile_lines.append(f"")

# ================ 8. Dataset Risk Register ================
profile_lines.append(f"## 8. Dataset Risk Register")
profile_lines.append(f"")
profile_lines.append(f"| หัวข้อ | รายละเอียด |")
profile_lines.append(f"|-------|-----------|")
profile_lines.append(f"| Source credibility | ยังไม่ได้ตรวจสอบแหล่งที่มา — ต้องตรวจสอบเพิ่มเติม |")
profile_lines.append(f"| License/usage | ยังไม่ทราบ license — ต้องตรวจสอบก่อนใช้งาน |")
profile_lines.append(f"| Business fit | รอการประเมินจาก task description |")
profile_lines.append(f"| Target suitability | {'พบ target column: ' + target_col if target_col else 'ยังไม่พบ target column'} |")
profile_lines.append(f"| Data dictionary | {'มี' if any('dictionary' in f.name.lower() or 'description' in f.name.lower() for f in all_files) else 'ยังไม่พบ data dictionary ใน input/'} |")
profile_lines.append(f"| Missing rate overall | {round(df.isnull().mean().mean() * 100, 2)}% |")

# ตรวจสอบว่าไฟล์อื่นๆ ที่เป็น data dictionary มีหรือไม่
dict_files = [f for f in all_files if 'dictionary' in f.name.lower() or 'description' in f.name.lower() or 'codebook' in f.name.lower()]
if dict_files:
    profile_lines.append(f"| Data dictionary files | {', '.join([f.name for f in dict_files])} |")

profile_lines.append(f"")

# ================ เขียนไฟล์ ================
profile_text = "\n".join(profile_lines)

profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# save CSV (pass-through)
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False)
print(f"[STATUS] Saved: {out_csv}")

print(f"[STATUS] ทำงานเสร็จสมบูรณ์")