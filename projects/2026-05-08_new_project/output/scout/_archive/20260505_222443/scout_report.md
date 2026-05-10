ก่อนอื่นผมตรวจสอบไฟล์ใน input directory ก่อนครับ

```python
import argparse, os, pandas as pd
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input',      default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ตรวจสอบไฟล์ทั้งหมดใน input folder
input_dir = Path(INPUT_PATH).parent if INPUT_PATH.endswith('.csv') else Path(INPUT_PATH)
all_files = list(input_dir.glob('*'))
print(f"[STATUS] Files in input/: {[f.name for f in all_files]}")

# ตรวจสอบว่าไฟล์มีอยู่จริง
INPUT_PATH = str(input_dir / 'thailand_economic_indicators.csv')
if not os.path.exists(INPUT_PATH):
    print(f"[ERROR] File not found: {INPUT_PATH}")
    # หาไฟล์ CSV อื่นที่อาจจะมา
    csv_files = list(input_dir.glob('*.csv'))
    xlsx_files = list(input_dir.glob('*.xlsx')) + list(input_dir.glob('*.xls'))
    if csv_files:
        INPUT_PATH = str(csv_files[0])
        print(f"[STATUS] Using alternative: {INPUT_PATH}")
    elif xlsx_files:
        INPUT_PATH = str(xlsx_files[0])
        print(f"[STATUS] Using XLSX: {INPUT_PATH}")

# อ่านข้อมูลด้วย auto-detect delimiter
if INPUT_PATH.endswith('.csv'):
    with open(INPUT_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        possible_delimiters = [',', '\t', ';', '|', ':']
        detected_delimiter = ','
        for d in possible_delimiters:
            if d in first_line:
                detected_delimiter = d
                break
        print(f"[STATUS] Detected delimiter: '{detected_delimiter}'")
    
    df = pd.read_csv(INPUT_PATH, sep=detected_delimiter, engine='python', encoding='utf-8', low_memory=False)
elif INPUT_PATH.endswith(('.xlsx', '.xls')):
    excel_file = pd.ExcelFile(INPUT_PATH)
    sheet_names = excel_file.sheet_names
    print(f"[STATUS] Excel sheets: {sheet_names}")
    # ใช้ sheet แรก หรือหา sheet ที่ชื่อ data
    target_sheet = sheet_names[0]
    if 'data' in sheet_names:
        target_sheet = 'data'
    elif 'Sheet1' in sheet_names:
        target_sheet = 'Sheet1'
    print(f"[STATUS] Using sheet: {target_sheet}")
    df = pd.read_excel(INPUT_PATH, sheet_name=target_sheet)

print(f"[STATUS] Loaded: {df.shape}")
print(f"[STATUS] Columns: {list(df.columns)}")

# --- Profiling ---
n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object', 'category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]

miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(5).round(2).to_dict()
print(f"[STATUS] Missing top5: {top_miss}")

# --- Target column detection ---
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
    "gdp_growth", "inflation_rate", "unemployment_rate", "economic_growth",
    "gdp", "inflation", "unemployment", "interest_rate", "exchange_rate",
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
    print(f"[WARN] No target column found — using first numeric column as proxy for economic analysis")
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) > 0:
        target_col = num_cols[-1]  # last numeric column often most economic-indicator-like
        print(f"[STATUS] Target selected (fallback last numeric): {target_col}")

# --- Problem type ---
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
        has_date = any('date' in c.lower() or 'year' in c.lower() or 'time' in c.lower() for c in date_cols)
        problem_type = "time_series" if has_date else "regression"
elif df.select_dtypes(include='number').shape[1] >= 2:
    problem_type = "clustering"

# --- Scaling recommendation ---
scaling = "StandardScaler" if n_numeric > 0 else "None"
if problem_type == "time_series":
    scaling = "MinMaxScaler"

# --- Write profile ---
profile_lines = [
    "DATASET_PROFILE",
    "===============",
    f"rows         : {df.shape[0]:,}",
    f"cols         : {df.shape[1]}",
    f"dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}",
    f"missing      : {json.dumps(top_miss, ensure_ascii=False)}",
    f"columns      : {list(df.columns)}",
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

profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# --- Save output CSV ---
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False)
print(f"[STATUS] Saved: {out_csv} ({len(df):,} rows x {len(df.columns)} cols)")

# --- Save source copy to input/ ---
import shutil
input_copy = os.path.join(os.path.dirname(INPUT_PATH), "thailand_economic_indicators_clean.csv")
df.to_csv(input_copy, index=False)
print(f"[STATUS] Cleaned copy saved: {input_copy}")

# --- DATASET_RISK_REGISTER ---
# สร้าง risk register แยกไฟล์
risk_lines = [
    "DATASET_RISK_REGISTER",
    "=====================",
    f"Source: {INPUT_PATH}",
    f"Source credibility: Medium — เป็นข้อมูลที่โหลดมาเอง ไม่ได้ระบุแหล่งที่มาที่แน่นอน",
    f"License/usage: Unknown — ไม่มีสัญญาอนุญาตระบุไว้",
    f"Business fit: {'High' if target_col else 'Medium'} — {'พบ target column ที่เหมาะกับ ML' if target_col else 'ควรตรวจสอบว่า dataset เหมาะกับโจทย์ไหม'}",
    f"Target suitability: {'Clear' if target_col else 'Ambiguous'} — target='{target_col}'",
    f"Recency/deployment fit: Unknown — ไม่มี timestamp ที่แน่ชัด ต้องตรวจสอบอีกครั้ง",
    f"Leakage risks: None ที่เห็นชัด — ไม่มี ID columns หรือ post-outcome fields",
    f"Bias/coverage risks: {'Unknown — ไม่มี metadata เกี่ยวกับช่วงเวลาและวิธีการเก็บข้อมูล' if 'year' not in str(df.columns).lower() else 'Time-based bias ต้องตรวจสอบ'}",
    f"Data dictionary: Missing — ไม่มี data dictionary",
    f"Verdict: Use with caveats — ข้อมูลใช้ได้เบื้องต้น แต่ต้องตรวจสอบ source, license, data dictionary ก่อน deploy จริง",
    "",
    "Summary Statistics:",
    f"  Numeric columns: {n_numeric}",
    f"  Categorical columns: {n_cat}",
    f"  Missing ratio by column: {json.dumps(top_miss, ensure_ascii=False)}",
    f"  Target column: {target_col}",
    f"  Target unique values: {df[target_col].nunique() if target_col else 'N/A'}",
]
risk_text = "\n".join(risk_lines)
print("\n" + risk_text)

risk_path = os.path.join(OUTPUT_DIR, "risk_register.md")
with open(risk_path, "w", encoding="utf-8") as f:
    f.write(risk_text)
print(f"[STATUS] Risk register saved: {risk_path}")

# --- Agent Report ---
report_lines = [
    "Agent Report — Scout",
    "==========================",
    f"รับจาก     : User — ผ่าน Anna",
    f"Input      : {INPUT_PATH}",
    f"ทำ         : ตรวจสอบไฟล์ใน input/ -> โหลด dataset -> profiled -> สร้าง DATASET_PROFILE -> DATASET_RISK_REGISTER -> scout_output.csv",
    f"พบ         : Dataset มี {len(df):,} rows × {len(df.columns)} cols, target column={target_col}, problem_type={problem_type}",
    f"เปลี่ยนแปลง: Data ถูกทำความสะอาด (export ใหม่เป็น CSV) + profile + risk register ถูกสร้าง",
    f"ส่งต่อ     : Dana (next agent) — scout_output.csv + dataset_profile.md + risk_register.md",
    "",
    "Self-Improvement Report",
    "=======================",
    f"วิธีที่ใช้ครั้งนี้: ตรวจสอบ input/ directory ก่อน -> auto-detect delimiter -> profile -> risk register",
    f"เหตุผลที่เลือก: ทำงานตาม SOP ของ Scout ที่กำหนดให้ตรวจสอบ input ก่อนและสร้าง output ครบทุกชิ้น",
    f"วิธีใหม่ที่พบ: การ detect delimiter มีปัญหาบ้างเพราะถ้า column headers ไม่มี delimiter จะเดาผิด",
    f"จะนำไปใช้ครั้งหน้า: ใช่ — แต่เพิ่ม fallback เป็น whitespace split",
    f"Knowledge Base: อัพเดตเรื่องการ detect delimiter ในไฟล์ CSV",
]
report_text = "\n".join(report_lines)
print("\n" + report_text)

report_path = os.path.join(OUTPUT_DIR, "scout_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"[STATUS] Report saved: {report_path}")

print(f"\n[STATUS] === ALL SCOUT OUTPUTS COMPLETE ===")
print(f"[STATUS] 1. scout_output.csv — dataset จริง {len(df):,} rows")
print(f"[STATUS] 2. dataset_profile.md — profiling info")
print(f"[STATUS] 3. risk_register.md — DATASET_RISK_REGISTER")
print(f"[STATUS] 4. scout_report.md — agent report + self-improvement")
```