```python
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
n_cat = df.select_dtypes(include=['object', 'category']).shape[1]
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
    "quantity", "sales", "revenue", "price", "amount",
    "rating", "score", "sentiment", "category",
]

target_candidates = []
for col in df.columns:
    if is_forbidden_target(col):
        continue
    score = 0
    col_l = col.lower()
    
    # business keyword match
    for kw in BUSINESS_TARGET_KEYWORDS:
        if kw in col_l or col_l == kw:
            score += 3
            break
    
    # binary 0/1
    if pd.api.types.is_numeric_dtype(df[col]):
        uniq = set(df[col].dropna().unique())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            score += 2
    
    # categorical low cardinality (2-10)
    if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10:
        score += 1
    
    if score > 0:
        target_candidates.append({
            'column': col,
            'score': score,
            'n_unique': df[col].nunique(),
            'missing_pct': round(df[col].isna().mean() * 100, 2),
            'dtype': str(df[col].dtype),
            'reason': f'business_keyword={3 if any(kw in col_l for kw in BUSINESS_TARGET_KEYWORDS) else 0}, binary={2 if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0,1,0.0,1.0}) else 0}, cat={1 if df[col].dtype=="object" and 2<=df[col].nunique()<=10 else 0}'
        })

target_candidates.sort(key=lambda x: x['score'], reverse=True)

# ========== MISSING ANALYSIS ==========
miss_df = (df.isnull().mean() * 100).sort_values(ascending=False)
miss_cols_highest = miss_df[miss_df > 0].head(10).round(2).to_dict()

# ========== DATA QUALITY RISKS ==========
quality_risks = []

# Check duplicates
dupe_rows = df.duplicated().sum()
if dupe_rows > 0:
    quality_risks.append(f"[DUPLICATE] {dupe_rows:,} duplicated rows ({dupe_rows/n_rows*100:.1f}%)")

# Check high missing
for col, pct in miss_cols_highest.items():
    if pct > 50:
        quality_risks.append(f"[HIGH_MISSING] {col}: {pct}% missing")
    elif pct > 20:
        quality_risks.append(f"[MED_MISSING] {col}: {pct}% missing")

# Check id columns uniqueness
for col in key_cols:
    n_uniq = df[col].nunique()
    if n_uniq < n_rows * 0.5:
        quality_risks.append(f"[LOW_CARD_KEY] {col}: only {n_uniq:,} unique / {n_rows:,} rows (not unique)")

# Check numeric outliers
for col in df.select_dtypes(include='number').columns:
    if df[col].nunique() > 2:  # skip binary
        q99 = df[col].quantile(0.99)
        q01 = df[col].quantile(0.01)
        iqr = q99 - q01
        if iqr == 0:
            continue
        beyond_3iqr = ((df[col] < q01 - 3*iqr) | (df[col] > q99 + 3*iqr)).sum()
        if beyond_3iqr > n_rows * 0.01:
            quality_risks.append(f"[OUTLIER] {col}: {beyond_3iqr:,} values beyond 3xIQR ({beyond_3iqr/n_rows*100:.1f}%)")

# ========== PROBLEM TYPE DETECTION ==========
problem_type = "unknown"
imbalance = None
class_dist = {}

if target_candidates:
    best_target = target_candidates[0]['column']
    n_uniq_target = df[best_target].nunique()
    if n_uniq_target <= 20:
        problem_type = "classification"
        vc = df[best_target].value_counts(normalize=True).round(4)
        class_dist = vc.to_dict()
        majority = vc.max()
        minority = vc.min()
        imbalance = round(majority / minority, 2) if minority > 0 else None
    elif date_cols:
        problem_type = "time_series"
    else:
        problem_type = "regression"
else:
    best_target = "unknown"

# ========== DATASET_RISK_REGISTER ==========
risk_register = f"""DATASET_RISK_REGISTER
=====================
Source credibility: Medium — UCI ML Repository (benchmark source, no explicit collection methodology)
License/usage: Research/educational use allowed — verify commercial terms
Business fit: High — Online Retail dataset maps directly to e-commerce analytics (customer behavior, sales patterns)
Target suitability: {best_target if best_target != 'unknown' else 'Need manual review'}"""
if target_candidates:
    risk_register += f"\n  - Primary: {target_candidates[0]['column']} (score={target_candidates[0]['score']}, {target_candidates[0]['n_unique']} unique)"
if len(target_candidates) > 1:
    risk_register += f"\n  - Alternative: {target_candidates[1]['column']} (score={target_candidates[1]['score']})"
risk_register += f"""
Recency/deployment fit: Unknown — UCI dataset collection date not specified; may be historical
Leakage risks: Possible if InvoiceDate/OrderDate used as feature for prediction of same-time targets
Bias/coverage risks: Single-country/retailer dataset; may not generalize to other markets or B2B
Data dictionary: [Available from UCI source page — check documentation]
Verdict: Use with caveats — verify timeliness and target definition before ML"""

print("\n[DATASET_RISK_REGISTER]")
print(risk_register)

# ========== WRITE DATASET_PROFILE ==========
profile_lines = [
    "DATASET_PROFILE",
    "===============",
    f"rows         : {n_rows:,}",
    f"cols         : {n_cols}",
    f"dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}",
    f"dtype_detail : {json.dumps(dtype_counts)}",
    f"missing_top  : {json.dumps(miss_cols_highest, ensure_ascii=False)}",
    f"duplicates   : {dupe_rows:,} ({dupe_rows/n_rows*100:.2f}%)",
    f"key_columns  : {key_cols}",
    f"date_cols    : {date_cols}",
    f"target_column: {best_target}",
    f"problem_type : {problem_type}",
]
if class_dist:
    profile_lines.append(f"class_dist   : {json.dumps({str(k): round(v,4) for k,v in list(class_dist.items())[:6]})}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {'StandardScaler' if n_numeric > 0 else 'None'}")
profile_lines.append("")
profile_lines.append(f"Schema ({n_cols} columns):")
for cs in col_summary:
    profile_lines.append(f"  {cs['name']}: {cs['dtype']} | unique={cs['n_unique']:,} | miss={cs['missing_pct']}% | sample={cs['sample_values']}")
profile_lines.append("")
profile_lines.append(f"Target Candidates (sorted by score):")
for tc in target_candidates:
    profile_lines.append(f"  {tc['column']}: score={tc['score']}, unique={tc['n_unique']:,}, miss={tc['missing_pct']}%, dtype={tc['dtype']}")
profile_lines.append("")
profile_lines.append(f"Data Quality Risks:")
if quality_risks:
    for risk in quality_risks:
        profile_lines.append(f"  {risk}")
else:
    profile_lines.append("  No significant risks detected")
profile_lines.append("")
profile_lines.append(risk_register)

profile_text = "\n".join(profile_lines)
profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# ========== WRITE SCOUT_REPORT ==========
report_lines = [
    "Scout Dataset Brief",
    "===================",
    f"Dataset: UCI Online Retail (e-commerce transactions)",
    f"Source: {INPUT_PATH}",
    f"License: Research/educational (UCI ML Repository)",
    f"Size: {n_rows:,} rows × {n_cols} columns",
    f"Format: CSV",
    f"Time Period: Unknown (UCI benchmark dataset)",
    "",
    "Columns Summary by Role:",
    "- Business keys:", 
]
for k in key_cols[:5]:
    report_lines.append(f"    * {k}")
report_lines.append("- Date/Time columns:")
for d in date_cols[:5]:
    report_lines.append(f"    * {d}")
report_lines.append("- Target candidates (ranked):")
for tc in target_candidates[:5]:
    report_lines.append(f"    * [{tc['score']}] {tc['column']} ({tc['dtype']}, {tc['n_unique']} unique, miss={tc['missing_pct']}%)")
report_lines.append("")
report_lines.append("Known Issues & Risks:")
for r in quality_risks:
    report_lines.append(f"  - {r}")
report_lines.append("")
report_lines.append(risk_register)
report_lines.append("")
report_lines.append("Dispatch Recommendation:")
if problem_type in ("classification", "regression"):
    report_lines.append("  ✅ เหมาะ dispatch Dana (data pipeline) + Eddie (ML modeling)")
    report_lines.append(f"  - Dana: data cleaning, join, feature engineering, target column={best_target}")
    report_lines.append(f"  - Eddie: model building, problem_type={problem_type}, balance_check={imbalance}")
elif problem_type == "time_series":
    report_lines.append("  ✅ เหมาะ Dana (time-series preprocessing) + Mo (forecasting)")
else:
    report_lines.append("  ⚠️  ต้องตรวจสอบเพิ่ม — dispatch Dana เพื่อ data quality check ก่อน")
report_lines.append("")
if dupe_rows > 0:
    report_lines.append(f"  ⚠️ Duplicate rows: {dupe_rows:,} — Dana ต้อง handle duplicates ก่อน modeling")
if any('HIGH_MISSING' in r for r in quality_risks):
    report_lines.append("  ⚠️ High missing columns detected — Dana ต้อง impute หรือ drop ก่อน Eddie")

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, "scout_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"[STATUS] Report saved: {report_path}")

# ========== SAVE OUTPUT CSV (pass-through) ==========
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False, encoding='utf-8')
print(f"[STATUS] Output CSV saved: {out_csv}")

# ========== SCRIPT SELF-SAVE ==========
script_path = os.path.join(OUTPUT_DIR, "scout_script.py")
with open(script_path, "w", encoding="utf-8") as f:
    f.write(open(__file__).read())
print(f"[STATUS] Script saved: {script_path}")

# ========== FINAL SUMMARY ==========
print(f"""
[STATUS] === SCOUT COMPLETE ===
[STATUS] Input       : {INPUT_PATH}
[STATUS] Rows        : {n_rows:,}
[STATUS] Cols        : {n_cols}
[STATUS] Target      : {best_target} (problem={problem_type})
[STATUS] Profile     : {profile_path}
[STATUS] Report      : {report_path}
[STATUS] Output CSV  : {out_csv}
[STATUS] Risks       : {len(quality_risks)} items
[STATUS] Dispatch    : {'✅ Ready for Dana → Eddie' if problem_type in ('classification','regression') else '✅ Ready for Dana → Mo' if problem_type == 'time_series' else '⚠️ Needs review'}
""")

# ========== AGENT REPORT ==========
agent_report = f"""Agent Report — Scout
============================
รับจาก     : User (existing dataset ดึงจาก project UCI ML/input/)
Input      : {INPUT_PATH} ({n_rows:,} rows × {n_cols:,} cols)
ทำ         : 
  - Profiling: schema, dtypes, missing, duplicates, outliers
  - Target detection: ranked candidate targets by business relevance
  - Risk assessment: quality risks, leakage, bias, source credibility
  - Dispatch recommendation: pipeline readiness for Dana/Eddie
พบ         : 
  1. Target column: {best_target} (score={target_candidates[0]['score'] if target_candidates else 'N/A'})
  2. Problem type: {problem_type}
  3. Risks: {len(quality_risks)} items found ({', '.join(quality_risks[:3])})
เปลี่ยนแปลง : 
  - Dataset профилирован для pipeline dispatch
  - DATASET_RISK_REGISTER added — source credibility Medium (UCI, benchmark)
ส่งต่อ     : Anna — Profile + Report + Risk Register + Dispatch Recommendation
"""
print(agent_report)
```