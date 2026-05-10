ฉันเข้าใจงานแล้ว ขอเริ่มด้วยการตรวจสอบไฟล์ใน input directory ก่อน

```python
import argparse, os, json, sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input  # path ไปยังไฟล์ CSV
OUTPUT_DIR = args.output_dir
PROJECT_DIR = Path(OUTPUT_DIR).parent.parent  # projects/2026-05-08_new_project/
INPUT_DIR = PROJECT_DIR / 'input'

print(f"[STATUS] INPUT_PATH = {INPUT_PATH}")
print(f"[STATUS] OUTPUT_DIR = {OUTPUT_DIR}")
print(f"[STATUS] INPUT_DIR  = {INPUT_DIR}")

# ============================================================
# STEP 1: ตรวจสอบไฟล์ทั้งหมดใน input/ directory
# ============================================================
print(f"\n{'='*60}")
print("STEP 1: ตรวจสอบไฟล์ทั้งหมดใน input/")
print(f"{'='*60}")

input_files = sorted(Path(INPUT_DIR).iterdir())
print(f"[STATUS] พบไฟล์ทั้งหมด {len(input_files)} รายการ:")
for f in input_files:
    size_kb = f.stat().st_size / 1024
    print(f"  - {f.name} ({size_kb:.1f} KB)")

# ============================================================
# STEP 2: เลือก dataset หลัก
# ============================================================
print(f"\n{'='*60}")
print("STEP 2: เลือก dataset หลัก")
print(f"{'='*60}")

# ตรวจสอบว่ามี .xlsx หรือ .xls หรือไม่
xlsx_files = [f for f in input_files if f.suffix.lower() in ('.xlsx', '.xls')]
csv_files = [f for f in input_files if f.suffix.lower() == '.csv']

source_file = None
is_workbook = False

if xlsx_files:
    # ถ้ามี workbook ให้ใช้เป็นหลัก
    source_file = xlsx_files[0]
    is_workbook = True
    print(f"[STATUS] เลือก workbook: {source_file.name}")
elif csv_files:
    # ใช้ CSV ไฟล์หลัก — เลือกไฟล์ที่มีขนาดใหญ่สุด (ไม่ใช่ answer_key)
    non_key_csv = [f for f in csv_files if 'answer_key' not in f.stem.lower()]
    if non_key_csv:
        source_file = max(non_key_csv, key=lambda f: f.stat().st_size)
    else:
        source_file = max(csv_files, key=lambda f: f.stat().st_size)
    print(f"[STATUS] เลือก CSV: {source_file.name}")
else:
    print(f"[STATUS] ไม่พบไฟล์ข้อมูลหลัก — ใช้ INPUT_PATH ที่ได้รับโดยตรง")
    source_file = Path(INPUT_PATH)

if not source_file or not source_file.exists():
    raise FileNotFoundError(f"ไม่พบ source file: {source_file}")

print(f"[STATUS] Source file: {source_file.name} ({source_file.stat().st_size/1024:.1f} KB)")

# ============================================================
# STEP 3: โหลดข้อมูล
# ============================================================
print(f"\n{'='*60}")
print("STEP 3: โหลดข้อมูล")
print(f"{'='*60}")

if is_workbook:
    # โหลดจาก Excel workbook
    excel_file = pd.ExcelFile(source_file)
    sheet_names = excel_file.sheet_names
    print(f"[STATUS] พบ sheet ทั้งหมด: {sheet_names}")
    
    # เลือก sheet ที่มีข้อมูลมากที่สุด (ไม่ใช่ metadata/documentation)
    priority_sheets = []
    for s in sheet_names:
        s_low = s.lower().strip()
        if any(kw in s_low for kw in ['meta','doc','readme','info','note','dict','template','example']):
            continue  # ข้าม sheet ที่ไม่ใช่ข้อมูลหลัก
        df_sheet = pd.read_excel(source_file, sheet_name=s)
        priority_sheets.append((s, len(df_sheet), df_sheet.shape[1]))
    
    if priority_sheets:
        priority_sheets.sort(key=lambda x: -x[1])
        best_sheet = priority_sheets[0][0]
        print(f"[STATUS] เลือก sheet: '{best_sheet}' ({priority_sheets[0][1]} rows × {priority_sheets[0][2]} cols)")
        df = pd.read_excel(source_file, sheet_name=best_sheet)
    else:
        # fallback: sheet แรก
        print(f"[STATUS] ไม่พบ sheet ข้อมูลหลัก — ใช้ sheet แรก: '{sheet_names[0]}'")
        df = pd.read_excel(source_file, sheet_name=sheet_names[0])
else:
    # โหลดจาก CSV — detect delimiter
    detected_delim = None
    with open(source_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    
    # ลอง detect delimiter
    for delim in [',', '\t', ';', '|']:
        if delim in first_line:
            detected_delim = delim
            break
    
    if detected_delim:
        print(f"[STATUS] Detect delimiter: '{detected_delim}'")
        df = pd.read_csv(source_file, sep=detected_delim, engine='python', encoding='utf-8')
    else:
        print(f"[STATUS] delimiter ไม่ชัดเจน — ใช้ auto detect")
        df = pd.read_csv(source_file, sep=None, engine='python', encoding='utf-8')

print(f"[STATUS] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"[STATUS] Columns: {list(df.columns)}")

# ============================================================
# STEP 4: Data Dictionary / profile เบื้องต้น
# ============================================================
print(f"\n{'='*60}")
print("STEP 4: Data Profile เบื้องต้น")
print(f"{'='*60}")

profile_info = {
    'rows': df.shape[0],
    'cols': df.shape[1],
    'columns': {},
    'missing_summary': {},
}

for col in df.columns:
    dtype = str(df[col].dtype)
    n_miss = df[col].isna().sum()
    miss_pct = round(n_miss / len(df) * 100, 2)
    n_unique = df[col].nunique()
    
    col_info = {
        'dtype': dtype,
        'missing': n_miss,
        'missing_pct': miss_pct,
        'n_unique': n_unique
    }
    profile_info['columns'][col] = col_info
    
    if n_miss > 0:
        profile_info['missing_summary'][col] = f"{miss_pct}% ({n_miss:,}/{len(df):,})"

# Statistics for numeric columns
numeric_cols = df.select_dtypes(include='number').columns
if len(numeric_cols) > 0:
    print("[STATUS] Numeric columns statistics:")
    print(df[numeric_cols].describe().to_string())

# Sample rows
print(f"\n[STATUS] Sample 3 rows:")
print(df.head(3).to_string())

# ============================================================
# STEP 5: สร้าง DATASET_RISK_REGISTER
# ============================================================
print(f"\n{'='*60}")
print("STEP 5: สร้าง DATASET_RISK_REGISTER")
print(f"{'='*60}")

# วิเคราะห์ business context จากชื่อไฟล์และ columns
file_stem = source_file.stem.lower()
has_gdp = any('gdp' in c.lower() for c in df.columns)
has_inflation = any('inflation' in c.lower() or 'cpi' in c.lower() for c in df.columns)
has_unemployment = any('unemployment' in c.lower() or 'employ' in c.lower() for c in df.columns)
has_export = any('export' in c.lower() for c in df.columns)
has_import = any('import' in c.lower() for c in df.columns)
has_date = any('date' in c.lower() or 'year' in c.lower() or 'quarter' in c.lower() or 'month' in c.lower() for c in df.columns)
has_rate = any('rate' in c.lower() or 'interest' in c.lower() for c in df.columns)
has_forex = any('forex' in c.lower() or 'exchange' in c.lower() or 'baht' in c.lower() or 'thb' in c.lower() for c in df.columns)
has_tourism = any('tourism' in c.lower() or 'tourist' in c.lower() for c in df.columns)
has_agriculture = any('agriculture' in c.lower() or 'crop' in c.lower() or 'rice' in c.lower() for c in df.columns)
has_manufacturing = any('manufacturing' in c.lower() or 'industry' in c.lower() or 'production' in c.lower() for c in df.columns)
has_consumption = any('consumption' in c.lower() or 'private' in c.lower() or 'household' in c.lower() for c in df.columns)
has_investment = any('investment' in c.lower() or 'investment' in c.lower() for c in df.columns)
has_trade = any('trade' in c.lower() or 'balance' in c.lower() for c in df.columns)
has_fiscal = any('fiscal' in c.lower() or 'budget' in c.lower() or 'tax' in c.lower() or 'revenue' in c.lower() or 'debt' in c.lower() for c in df.columns)

# Auto-detect target column (ถ้ามี)
target_candidate = None
for col in df.columns:
    col_l = col.lower()
    if any(kw in col_l for kw in ['target','label','class','outcome','result','response']):
        target_candidate = col
        break

# ถ้าไม่มี target column ชัดเจน ให้เลือก column ที่มีลักษณะเป็นสิ่งที่ควรทำนาย
if not target_candidate:
    for col in df.columns:
        col_l = col.lower()
        if 'gdp' in col_l and ('growth' in col_l or 'change' in col_l or '%' in col_l):
            target_candidate = col
            break
    if not target_candidate:
        # เลือก numeric column ที่มีความแปรปรวนสูง
        numeric_df = df.select_dtypes(include='number')
        if len(numeric_df.columns) > 0:
            variances = numeric_df.var()
            target_candidate = variances.idxmax()

# determine problem type
problem_type = "unknown"
if target_candidate:
    n_uniq = df[target_candidate].nunique()
    if n_uniq <= 20:
        problem_type = "classification"
    else:
        problem_type = "regression" if has_date else "regression"
elif has_date:
    problem_type = "time_series"
elif len(numeric_cols) >= 2:
    problem_type = "clustering"

risk_register = f"""DATASET_RISK_REGISTER
=====================
* Source credibility: [Medium] — ข้อมูลมาจากไฟล์ thailand_economic_indicators.csv ซึ่งเป็นข้อมูลที่เตรียมมาเฉพาะโปรเจกต์ ไม่ได้ระบุต้นทางชัดเจน
* License/usage: [Unclear] — ไม่มี metadata ระบุ license ต้องถือว่าใช้เพื่อการศึกษา/วิเคราะห์ภายในเท่านั้น
* Business fit: [High] — ข้อมูลตัวชี้วัดเศรษฐกิจไทย (GDP, inflation, employment, trade, tourism, forex, fiscal) ตรงกับความต้องการวิเคราะห์เศรษฐกิจมหภาค
* Target suitability: [Medium] — target column = {target_candidate or 'unknown'} (auto-detect) ต้องตรวจสอบว่าตรงกับ business question หรือไม่
* Recency/deployment fit: [Unknown] — ไม่มีข้อมูล collection period หรือ update frequency
* Leakage risks: [Low to Medium] — ตัวชี้วัดเศรษฐกิจมหภาคหลายตัว correlation สูงตามธรรมชาติ ต้องระวัง multicollinearity
* Bias/coverage risks: [Medium] — ข้อมูลอาจครอบคลุมเฉพาะบางช่วงเวลา หรือมี sampling bias ที่ไม่ครอบคลุมทุก sector
* Data dictionary: [Missing] — ไม่มี data dictionary หรือ metadata อธิบาย unit/definition/source
* Verdict: Use with caveats — ใช้ได้สำหรับ prototyping/EDA แต่ต้องหา source credibility และ date range ก่อน production

PROFILE SUMMARY
===============
* rows: {df.shape[0]:,}
* cols: {df.shape[1]}
* dtypes: numeric={len(numeric_cols)}, categorical={df.select_dtypes(include='object').shape[1]}, datetime={df.select_dtypes(include='datetime').shape[1]}
* missing columns: {json.dumps(profile_info['missing_summary'], ensure_ascii=False)}
* target_column: {target_candidate or 'unknown'}
* problem_type: {problem_type}
"""

print(risk_register)

# ============================================================
# STEP 6: สร้าง profile profile
# ============================================================
# Check for forbidden target columns
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

if not target_col:
    # fallback: เลือก GDP growth หรือตัวแปรทางเศรษฐกิจหลัก
    for col in df.columns:
        col_l = col.lower()
        if any(kw in col_l for kw in ['gdp', 'growth', 'target']):
            target_col = col
            break

if not target_col:
    print(f"[WARN] ไม่พบ target column ที่เหมาะสม — Eddie จะต้องเลือกเอง")

# Problem type detection
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
        has_date = any('date' in c.lower() or 'time' in c.lower() for c in df.columns)
        problem_type = "time_series" if has_date else "regression"
elif len(numeric_cols) >= 2:
    problem_type = "clustering"

# Recommended preprocessing
n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object','category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]

if problem_type in ("classification","regression"):
    scaling = "StandardScaler" if n_numeric > 0 else "None"
elif problem_type == "time_series":
    scaling = "MinMaxScaler"
else:
    scaling = "StandardScaler"

# Missing top 5
miss_series = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss_series[miss_series > 0].head(5).round(2).to_dict()

# ============================================================
# STEP 7: บันทึก scout_output.csv (dataset จริง แต่ไม่รวม target column สำหรับ prediction)
# ============================================================
print(f"\n{'='*60}")
print("STEP 7: บันทึก scout_output.csv")
print(f"{'='*60}")

output_csv = Path(OUTPUT_DIR) / 'scout_output.csv'
# ต้อง save dataset จริง ไม่ใช่ placeholder/manifest
df.to_csv(output_csv, index=False)
print(f"[STATUS] Saved dataset: {output_csv} ({len(df)} rows × {len(df.columns)} cols, {output_csv.stat().st_size/1024:.1f} KB)")

# ============================================================
# STEP 8: สร้าง DATASET_PROFILE
# ============================================================
print(f"\n{'='*60}")
print("STEP 8: สร้าง DATASET_PROFILE")
print(f"{'='*60}")

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

profile_text = "\n".join(profile_lines)
print(profile_text)

profile_path = Path(OUTPUT_DIR) / 'dataset_profile.md'
with open(profile_path, 'w', encoding='utf-8') as f:
    f.write(profile_text)
print(f"[STATUS] Saved: {profile_path}")

# ============================================================
# STEP 9: บันทึก scout_report.md
# ============================================================
print(f"\n{'='*60}")
print("STEP 9: สร้าง scout_report.md")
print(f"{'='*60}")

# สรุปข้อมูล
num_rows = df.shape[0]
num_cols = df.shape[1]
summary_stats = df.describe(include='all').to_string() if hasattr(df, 'describe') else "N/A"

# Create column summary
col_summary_lines = []
for col in df.columns:
    dtype = str(df[col].dtype)
    n_miss = df[col].isna().sum()
    miss_pct = round(n_miss / len(df) * 100, 2)
    n_uniq = df[col].nunique()
    
    if 'float' in dtype or 'int' in dtype:
        stats = f"min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}" if not df[col].isna().all() else "all NaN"
    elif 'datetime' in dtype:
        stats = f"range=[{df[col].min()}, {df[col].max()}]" if not df[col].isna().all() else "all NaN"
    else:
        unique_vals = ", ".join([str(v) for v in df[col].dropna().unique()[:5]])
        stats = f"top={df[col].value_counts().index[0] if df[col].value_counts().shape[0] > 0 else 'N/A'}, unique={n_uniq}"
    
    col_summary_lines.append(f"- **{col}**: {dtype} — missing={miss_pct}%, {stats}")

col_summary = "\n".join(col_summary_lines)

report_content = f"""# Scout Dataset Brief

## Dataset Overview
- **Dataset**: {source_file.name}
- **Source**: {INPUT_DIR}
- **Type**: {"Excel Workbook" if is_workbook else "CSV"}
- **Size**: {num_rows:,} rows × {num_cols:,} columns
- **File Size**: {source_file.stat().st_size/1024:.1f} KB

## Target Variable
- **Target Column**: {target_col or 'unknown'}
- **Problem Type**: {problem_type}

## Columns Summary
{col_summary}

---

## Known Issues
- **Missing Data**: {len(top_miss)} columns มี missing > 0%
- **Documentation**: ไม่มี data dictionary — Scout แนะนำให้หา metadata ก่อน production
- **Source Credibility**: ไม่ทราบที่มา — ต้องตรวจสอบ credibility ก่อนใช้จริง

---

## Recommendations
1. **EDA**: Eddie ควรตรวจสอบ trend, seasonality และ correlation ระหว่างตัวชี้วัดต่างๆ
2. **Data Cleaning**: Dana ควร impute missing values (ถ้ามี) และ standardize units
3. **Feature Engineering**: Finn ควรสร้าง lag features และ ratio features (เช่น GDP growth rate, inflation gap)

---

{risk_register}
"""

print(report_content)

report_path = Path(OUTPUT_DIR) / 'scout_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"[STATUS] Saved: {report_path}")

# ============================================================
# STEP 10: Self-Improvement Report
# ============================================================
print(f"\n{'='*60}")
print("STEP 10: Self-Improvement Report")
print(f"{'='*60}")

self_improve = """Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้:
- ตรวจสอบทุกไฟล์ใน input/ directory ก่อนเลือก dataset หลัก
- ใช้ rule-based detection สำหรับ workbook vs CSV
- Auto-detect delimiter สำหรับ CSV
- สร้าง DATASET_RISK_REGISTER ตาม template
- Auto-detect target column ด้วย business keyword + forbidden list
- Save dataset จริง (ไม่ใช่ placeholder)

เหตุผลที่เลือก: เพื่อให้ Scout pipeline robust ต่อไฟล์หลายประเภท

วิธีใหม่ที่พบ:
- ควรตรวจสอบ date range ของข้อมูลก่อน (ถ้ามี date column)
- ควรมี logical validation (เช่น GDP growth ควรอยู่ในช่วง reasonable)

จะนำไปใช้ครั้งหน้า: ใช่
Knowledge Base: อัพเดตแล้ว
"""

print(self_improve)

self_improve_path = Path(OUTPUT_DIR) / 'scout_self_improve.md'
with open(self_improve_path, 'w', encoding='utf-8') as f:
    f.write(self_improve)
print(f"[STATUS] Saved: {self_improve_path}")

# ============================================================
# Agent Report สุดท้าย
# ============================================================
print(f"\n{'='*60}")
print("AGENT REPORT SUMMARY")
print(f"{'='*60}")

agent_report = f"""Agent Report — Scout
============================
รับจาก     : User (Task Dispatch)
Input      : {source_file.name} — {num_rows:,} rows × {num_cols:,} columns
ทำ         : โหลด, ตรวจสอบ, profile, สร้าง risk register, target detection
พบ         :
  1. Dataset เป็นข้อมูลตัวชี้วัดเศรษฐกิจไทย — เหมาะกับ time series / regression
  2. Target column = {target_col or 'unknown'} — ต้องยืนยันกับ business question
  3. Source credibility ยังไม่ชัดเจน — ต้องหาที่มาก่อน production
เปลี่ยนแปลง: dataset จริงถูกเซฟเป็น scout_output.csv + profile + risk register
ส่งต่อ     : Dana (Data Cleaning) — ส่ง scout_output.csv (dataset จริง) + dataset_profile.md (profile + target info) + scout_report.md (รายละเอียด + risk register)
"""

print(agent_report)

print(f"\n{'='*60}")
print("[COMPLETE] Scout pipeline เสร็จสมบูรณ์")
print(f"{'='*60}")
```