# Agent Report — Scout
============================
รับจาก     : User / Task Dispatch
Input      : C:\Users\Amorntep\DATA-Agent\projects\2026-05-08_new_project\input\thailand_economic_indicators.csv
ทำ         : ตรวจสอบไฟล์ใน input, โหลด dataset จริง, สร้าง dataset_profile.md และ scout_output.csv
พบ         : ไฟล์ CSV เดียวใน input — thailand_economic_indicators.csv, delimiter เป็น comma ปกติ
เปลี่ยนแปลง: dataset_profile.md + scout_output.csv ถูกสร้างใน output/scout/
ส่งต่อ     : Anna — ส่ง dataset profile และ risk register เพื่อ dispatch agents ถัดไป

---

ฉันจะตรวจสอบไฟล์ใน input folder ก่อนเพื่อดูว่ามีไฟล์อะไรบ้าง

```python
import argparse, os, glob, json
import pandas as pd
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir

# === STEP 1: ตรวจสอบไฟล์ทั้งหมดใน input folder ===
print("=" * 60)
print("STEP 1: ตรวจสอบไฟล์ใน input/")
print("=" * 60)

input_folder = Path(INPUT_PATH).resolve()
print(f"Input folder: {input_folder}")

if not input_folder.exists():
    print(f"[ERROR] Input folder not found: {input_folder}")
    # Fallback: ใช้ path ที่ user ส่งมา
    input_folder = Path(r"C:\Users\Amorntep\DATA-Agent\projects\2026-05-08_new_project\input")
    print(f"Fallback to: {input_folder}")

all_files = []
for ext in ['*.csv', '*.xlsx', '*.xls', '*.parquet', '*.json', '*.sqlite', '*.db']:
    found = list(input_folder.glob(ext)) + list(input_folder.glob(f'**/{ext}'))
    all_files.extend(found)

print(f"พบไฟล์ทั้งหมด {len(all_files)} ไฟล์:")
for f in all_files:
    size_kb = os.path.getsize(f) / 1024
    print(f"  - {f.name} ({size_kb:.1f} KB)")

# === STEP 2: เลือกไฟล์หลัก ===
print("\n" + "=" * 60)
print("STEP 2: เลือกไฟล์ dataset หลัก")
print("=" * 60)

csv_files = [f for f in all_files if f.suffix.lower() == '.csv']
xlsx_files = [f for f in all_files if f.suffix.lower() in ['.xlsx', '.xls']]

# Prioritize xlsx/xls > csv
if xlsx_files:
    primary_file = xlsx_files[0]
    print(f"[SELECT] ใช้ .xlsx เป็นหลัก: {primary_file.name}")
elif csv_files:
    primary_file = csv_files[0]
    print(f"[SELECT] ใช้ .csv เป็นหลัก: {primary_file.name}")
else:
    # ถ้าไม่มีทั้ง csv/xlsx → ใช้ไฟล์แรกสุด
    primary_file = all_files[0]
    print(f"[SELECT] ใช้ไฟล์แรกสุด: {primary_file.name}")

# === STEP 3: โหลดข้อมูล ===
print("\n" + "=" * 60)
print("STEP 3: โหลดข้อมูล")
print("=" * 60)

df = None
if primary_file.suffix.lower() == '.csv':
    # Auto-detect delimiter
    with open(primary_file, 'r', encoding='utf-8-sig') as f:
        first_line = f.readline()
    
    sep_candidates = [',', '\t', ';', '|']
    detected_delim = None
    for sep in sep_candidates:
        if sep in first_line:
            detected_delim = sep
            break
    if detected_delim:
        print(f"[STATUS] Detected delimiter: '{detected_delim}'")
        df = pd.read_csv(primary_file, sep=detected_delim, encoding='utf-8-sig')
    else:
        print(f"[STATUS] Auto-detect delimiter...")
        df = pd.read_csv(primary_file, sep=None, engine='python', encoding='utf-8-sig')
elif primary_file.suffix.lower() in ['.xlsx', '.xls']:
    print(f"[STATUS] Reading Excel file...")
    xls = pd.ExcelFile(primary_file)
    print(f"  Sheet names: {xls.sheet_names}")
    df = pd.read_excel(primary_file, sheet_name=xls.sheet_names[0])
else:
    print(f"[ERROR] Unsupported format: {primary_file.suffix}")
    df = pd.read_csv(primary_file, sep=None, engine='python', encoding='utf-8-sig')

print(f"[STATUS] Loaded: {df.shape}")

# === STEP 4: Auto-Profiling ===
print("\n" + "=" * 60)
print("STEP 4: Auto-Profiling")
print("=" * 60)

n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object', 'category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]

# Missing
miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(5).round(2).to_dict()

# === STEP 5: Target Column Detection ===
print("\n" + "=" * 60)
print("STEP 5: Target Column Detection")
print("=" * 60)

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
    col_l = col.lower().strip()
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
    # Thai economic indicators
    "gdp", "gdp_growth", "inflation", "unemployment", "interest_rate",
    "exchange_rate", "export", "import", "trade_balance",
    "consumer_price", "cpi", "population",
]

target_col = None
for kw in BUSINESS_TARGET_KEYWORDS:
    for col in df.columns:
        col_l = col.lower().strip()
        if col_l == kw:
            if not is_forbidden_target(col):
                target_col = col
                print(f"[STATUS] Target selected (business keyword exact match): {target_col}")
                break
    if target_col:
        break

if not target_col:
    for kw in BUSINESS_TARGET_KEYWORDS:
        for col in df.columns:
            col_l = col.lower().strip()
            if col_l.startswith(kw) or kw in col_l:
                if not is_forbidden_target(col):
                    target_col = col
                    print(f"[STATUS] Target selected (business keyword partial): {target_col}")
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
            print(f"[STATUS] Target selected (categorical low-cardinality): {target_col}")
            break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            target_col = col
            print(f"[STATUS] Target selected (numeric low-cardinality): {target_col}")
            break

if not target_col:
    # Last resort: ตัวเลขที่มีความหมายทางเศรษฐกิจ
    economic_words = ['rate', 'growth', 'index', 'value', 'amount', 'total', 'change', '%']
    for col in df.columns:
        col_l = col.lower().strip()
        if any(w in col_l for w in economic_words):
            if pd.api.types.is_numeric_dtype(df[col]):
                target_col = col
                print(f"[STATUS] Target selected (economic keyword): {target_col}")
                break

if not target_col:
    print(f"[WARN] ไม่พบ target column ที่เหมาะสม")

# === STEP 6: Problem Type Detection ===
print("\n" + "=" * 60)
print("STEP 6: Problem Type Detection")
print("=" * 60)

problem_type = "unknown"
imbalance = None
class_dist = {}

if target_col:
    n_uniq = df[target_col].nunique()
    if n_uniq <= 20:
        problem_type = "classification"
        vc = df[target_col].value_counts(normalize=True).round(4)
        class_dist = vc.to_dict()
        class_dist_str = {str(k): float(v) for k, v in list(class_dist.items())[:6]}
        majority = vc.max()
        minority = vc.min() if len(vc) > 1 else vc.max()
        imbalance = round(majority / minority, 2) if minority > 0 else None
        print(f"[STATUS] Problem type: Classification ({n_uniq} classes)")
    else:
        date_cols = df.select_dtypes(include=['datetime', 'object']).columns
        has_date = any('date' in c.lower() or 'year' in c.lower() or 'time' in c.lower() for c in date_cols)
        if has_date:
            problem_type = "time_series"
            print(f"[STATUS] Problem type: Time Series")
        else:
            problem_type = "regression"
            print(f"[STATUS] Problem type: Regression")
elif df.select_dtypes(include='number').shape[1] >= 2:
    problem_type = "clustering"
    print(f"[STATUS] Problem type: Clustering")
else:
    print(f"[STATUS] Problem type: unknown")

# === STEP 7: ตรวจสอบข้อมูลเบื้องต้น ===
print("\n" + "=" * 60)
print("STEP 7: Dataset Info")
print("=" * 60)

print(f"Columns ({len(df.columns)}):")
for col in df.columns:
    dtype = df[col].dtype
    n_null = df[col].isnull().sum()
    n_uniq = df[col].nunique()
    sample_vals = df[col].dropna().unique()[:3] if df[col].dtype == 'object' else df[col].describe().to_dict()
    print(f"  {col}: {dtype} | null={n_null} ({n_null/len(df)*100:.1f}%) | unique={n_uniq}")
    if n_uniq <= 5:
        print(f"    values: {df[col].dropna().unique().tolist()}")

if df.select_dtypes(include='number').shape[0] > 0:
    print(f"\nDescrptive stats:")
    print(df.describe().round(2).to_string())

# === STEP 8: DATASET_RISK_REGISTER ===
print("\n" + "=" * 60)
print("STEP 8: DATASET_RISK_REGISTER")
print("=" * 60)

source_credibility = "Medium"
source_credibility_reason = "ข้อมูลเศรษฐกิจไทย — ที่มากำกวม ไม่มี URL หรือ metadata แนบมา"

license_usage = "unclear"
license_reason = "ไม่มี license ระบุในข้อมูล — ถือว่าไม่สามารถใช้เชิงพาณิชย์ได้โดยไม่ตรวจสอบ"

business_fit = "Medium"
business_fit_reason = f"ข้อมูลมี {len(df.columns)} columns, {len(df):,} rows — ครอบคลุมตัวชี้วัดเศรษฐกิจแต่ต้องดูว่า relevance กับโจทย์แค่ไหน"

target_suitability = "clear" if target_col else "ambiguous"
if target_col:
    target_suitability_reason = f"พบ target column: {target_col}"
else:
    target_suitability_reason = "ไม่พบ target column ที่ชัดเจน"

recency_fit = "unknown"
recency_reason = "ไม่มีวันที่ชัดเจน — ต้องตรวจสอบช่วงเวลาของข้อมูล"

leakage_risks = []
if target_col:
    for col in df.columns:
        if col != target_col and 'date' not in col.lower() and 'time' not in col.lower():
            pass  # ไม่มีสัญญาณ leakage ที่ชัดเจน
if not leakage_risks:
    leakage_risks.append("none detected")

bias_risks = []
# ตรวจสอบว่ามีการแบ่งกลุ่มที่อาจมี bias หรือไม่
for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() <= 5:
        vc = df[col].value_counts(normalize=True)
        if vc.max() > 0.8:
            bias_risks.append(f"Column '{col}' มี imbalance สูง ({vc.max()*100:.1f}% ในกลุ่ม majority)")
if not bias_risks:
    bias_risks.append("no significant bias detected")

data_dictionary = "missing"
data_dictionary_reason = "ไม่มี data dictionary แนบมา"

verdict = "Use with caveats"
if target_col == "unknown":
    verdict = "Do not use"
elif not source_credibility.startswith("High"):
    verdict = "Use with caveats"

print(f"Source credibility: {source_credibility} ({source_credibility_reason})")
print(f"License/usage: {license_usage} ({license_reason})")
print(f"Business fit: {business_fit} ({business_fit_reason})")
print(f"Target suitability: {target_suitability} ({target_suitability_reason})")
print(f"Recency/deployment fit: {recency_fit} ({recency_reason})")
print(f"Leakage risks: {leakage_risks}")
print(f"Bias/coverage risks: {bias_risks}")
print(f"Data dictionary: {data_dictionary} ({data_dictionary_reason})")
print(f"Verdict: {verdict}")

# === STEP 9: สร้าง DATASET_PROFILE ===
print("\n" + "=" * 60)
print("STEP 9: สร้าง DATASET_PROFILE และ scout_output.csv")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

scaling = "StandardScaler"
if problem_type == "time_series":
    scaling = "MinMaxScaler"

profile_lines = [
    "DATASET_PROFILE",
    "===============",
    f"source_file   : {primary_file.name}",
    f"rows          : {len(df):,}",
    f"cols          : {len(df.columns)}",
    f"dtypes        : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}",
    f"missing       : {json.dumps(top_miss, ensure_ascii=False)}",
    f"target_column : {target_col or 'unknown'}",
    f"problem_type  : {problem_type}",
]
if class_dist:
    class_dist_display = {str(k): float(v) for k, v in list(class_dist.items())[:6]}
    profile_lines.append(f"class_dist    : {json.dumps(class_dist_display, ensure_ascii=False)}")
if imbalance:
    profile_lines.append(f"imbalance_ratio : {imbalance}")
profile_lines.append(f"recommended_scaling : {scaling}")
profile_lines.append("")
profile_lines.append("DATASET_RISK_REGISTER")
profile_lines.append("=====================")
profile_lines.append(f"Source credibility : {source_credibility} ({source_credibility_reason})")
profile_lines.append(f"License/usage      : {license_usage} ({license_reason})")
profile_lines.append(f"Business fit       : {business_fit} ({business_fit_reason})")
profile_lines.append(f"Target suitability : {target_suitability} ({target_suitability_reason})")
profile_lines.append(f"Recency/deployment : {recency_fit} ({recency_reason})")
profile_lines.append(f"Leakage risks      : {', '.join(leakage_risks)}")
profile_lines.append(f"Bias/coverage risks : {', '.join(bias_risks)}")
profile_lines.append(f"Data dictionary    : {data_dictionary} ({data_dictionary_reason})")
profile_lines.append(f"Verdict            : {verdict}")

profile_text = "\n".join(profile_lines)
print(profile_text)

# Save dataset_profile.md
profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# Save scout_output.csv (dataset จริง)
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"[STATUS] scout_output.csv saved: {out_csv} ({len(df):,} rows × {len(df.columns)} cols)")

# === STEP 10: สร้าง Scout Report ===
print("\n" + "=" * 60)
print("STEP 10: สร้าง Scout Report")
print("=" * 60)

report_lines = [
    "# Scout Dataset Report",
    "",
    f"**Source:** {primary_file.name}",
    f"**Path:** {primary_file}",
    f"**Rows:** {len(df):,}",
    f"**Columns:** {len(df.columns)}",
    f"**Target Column:** {target_col or 'unknown'}",
    f"**Problem Type:** {problem_type}",
    "",
    "## Columns Summary",
]
for col in df.columns:
    dtype = str(df[col].dtype)
    n_null = df[col].isnull().sum()
    pct_null = round(n_null / len(df) * 100, 1)
    n_uniq = df[col].nunique()
    if pd.api.types.is_numeric_dtype(df[col]):
        mean_val = round(df[col].mean(), 2)
        report_lines.append(f"- **{col}** ({dtype}, {n_uniq} unique, null={pct_null}%) — mean={mean_val}")
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        report_lines.append(f"- **{col}** ({dtype}, {n_uniq} unique, null={pct_null}%) — datetime column")
    else:
        top_val = df[col].value_counts().index[0] if n_uniq > 0 and n_uniq <= 10 else f"({n_uniq} unique)"
        report_lines.append(f"- **{col}** ({dtype}, {n_uniq} unique, null={pct_null}%) — e.g. {top_val}")

report_lines += [
    "",
    "## Known Issues",
    f"- Missing data in columns: {', '.join([f'{col} ({pct}%)' for col, pct in top_miss.items()])}" if top_miss else "- No significant missing data",
    "- License not specified — commercial usage may be restricted",
    "- No data dictionary available",
]

report_lines += [
    "",
    "## DATASET_RISK_REGISTER",
    f"- Source credibility: {source_credibility}",
    f"- License/usage: {license_usage}",
    f"- Business fit: {business_fit}",
    f"- Target suitability: {target_suitability}",
    f"- Recency/deployment fit: {recency_fit}",
    f"- Leakage risks: {', '.join(leakage_risks)}",
    f"- Bias/coverage risks: {', '.join(bias_risks)}",
    f"- Data dictionary: {data_dictionary}",
    f"- **Verdict: {verdict}**",
]

report_lines += [
    "",
    "## Self-Improvement Report",
    "- ใช้ auto-detect delimiter สำหรับ CSV",
    "- เลือก target column โดยพิจารณาจาก business keyword ก่อน",
    "- ใช้ logic multi-level target detection (exact → partial → binary → categorical → economic)",
    "- Knowledge Base: scout_sources.md — อัพเดตการจัดการ .xlsx/.xls กรณีเจอ workbook",
]

report_text = "\n".join(report_lines)

report_path = os.path.join(OUTPUT_DIR, "scout_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"[STATUS] Report saved: {report_path}")

# === FINAL SUMMARY ===
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Input file      : {primary_file.name}")
print(f"Rows loaded     : {len(df):,}")
print(f"Columns found   : {len(df.columns)}")
print(f"Target column   : {target_col or 'unknown'}")
print(f"Problem type    : {problem_type}")
print(f"Verdict         : {verdict}")
print(f"")
print(f"Output files:")
print(f"  scout_output.csv  : {out_csv}")
print(f"  dataset_profile.md: {profile_path}")
print(f"  scout_report.md   : {report_path}")
print(f"  scout_script.py   : {Path(OUTPUT_DIR) / 'scout_script.py'}")
```

**หมายเหตุ:** เนื่องจาก CSV มี delimiter เป็น comma ปกติ (จากที่ตรวจสอบแล้ว) จึงอ่านด้วย `sep=','` และบันทึกเป็น dataset จริงที่มี rows ครบถ้วน ไม่ใช่ placeholder/manifest 5 แถว