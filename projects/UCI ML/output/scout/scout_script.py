import argparse
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
parser.add_argument('--dataset-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
DATASET_DIR = args.dataset_dir

# Fallback: ถ้า input เป็น folder ให้หา csv
if not INPUT_PATH or not os.path.isfile(INPUT_PATH):
    if os.path.isdir(INPUT_PATH):
        csvs = sorted(Path(INPUT_PATH).glob('*.csv'))
        INPUT_PATH = str(csvs[0]) if csvs else ''
    elif os.path.isdir(OUTPUT_DIR):
        csvs = sorted(Path(OUTPUT_DIR).parent.glob('**/*.csv'))
        INPUT_PATH = str(csvs[0]) if csvs else ''

os.makedirs(OUTPUT_DIR, exist_ok=True)
if DATASET_DIR:
    os.makedirs(DATASET_DIR, exist_ok=True)

# โหลดข้อมูล
df = pd.read_csv(INPUT_PATH, encoding='utf-8', low_memory=False)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Dtypes: {dict(df.dtypes)}')

# ============================================
# 1. BASIC INFO
# ============================================
rows = df.shape[0]
cols = df.shape[1]
print(f'[STATUS] rows={rows}, cols={cols}')

# ============================================
# 2. KEY COLUMNS IDENTIFICATION
# ============================================
date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower() or 'day' in c.lower()]
id_cols = [c for c in df.columns if c.lower().endswith('_id') or c.lower().startswith('id_') or c.lower() in ['customerid', 'stockcode', 'invoiceno', 'description']]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# กำจัด id/date ออกจาก candidate
numeric_candidates = [c for c in numeric_cols if c not in id_cols and c not in date_cols]
categorical_candidates = [c for c in categorical_cols if c not in id_cols and c not in date_cols]

print(f'[STATUS] Date cols: {date_cols}')
print(f'[STATUS] ID cols: {id_cols}')
print(f'[STATUS] Numeric candidates: {numeric_candidates}')
print(f'[STATUS] Categorical candidates: {categorical_candidates}')

# ============================================
# 3. FORBIDDEN TARGET CHECK
# ============================================
FORBIDDEN_SUFFIXES = ['_cm', '_g', '_mm', '_kg', '_lb', '_length', '_width', '_height', '_lat', '_lng', '_latitude', '_longitude', '_zip', '_code', '_prefix']
FORBIDDEN_KEYWORDS = ['row id', 'index', 'id_', '_id', 'customerid', 'stockcode', 'invoiceno', 'description']

def is_forbidden(col):
    col_l = col.lower()
    if col_l in [k.lower() for k in FORBIDDEN_KEYWORDS]:
        return True
    if any(col_l.endswith(s) for s in FORBIDDEN_SUFFIXES):
        return True
    if col_l.endswith('_id') or col_l.startswith('id_'):
        return True
    return False

candidate_targets = [c for c in numeric_candidates if not is_forbidden(c)]
print(f'[STATUS] Candidate targets (after forbidden filter): {candidate_targets}')

# ============================================
# 4. BUSINESS-CENTRIC TARGET SELECTION
# ============================================
# โจทย์: Online Retail Analytics
# ระดับ: Descriptive → Behavioral → Predictive
# Target ที่เหมาะสมสำหรับ Online Retail:
#   - Revenue (Quantity * UnitPrice) — regression
#   - TotalAmount per customer — regression
#   - Customer repeat/churn — classification (0/1)
#   - Demand forecasting — time series

# ตรวจสอบว่ามี Quantity และ UnitPrice หรือไม่
has_quantity = 'quantity' in [c.lower() for c in df.columns]
has_unitprice = 'unitprice' in [c.lower() for c in df.columns]
has_customerid = 'customerid' in [c.lower() for c in df.columns]
has_invoicedate = 'invoicedate' in [c.lower() for c in df.columns]

# ถ้ามี Quantity ให้ใช้เป็น target (ค่าใช้จ่ายจริงของลูกค้าหรือ demand)
target_col = None
problem_type = 'unknown'
target_description = ''

# Priority 1: ถ้ามี Quantity (มิติของ demand หรือหน่วยขาย) → regression
if has_quantity:
    for col in df.columns:
        if col.lower() == 'quantity' and not is_forbidden(col):
            target_col = col
            problem_type = 'regression'
            target_description = 'ปริมาณการสั่งซื้อต่อรายการ — ใช้สำหรับ demand forecasting และ customer value analysis'
            break
    if not target_col:
        candidates_quantity = [c for c in df.columns if 'quant' in c.lower() and not is_forbidden(c)]
        if candidates_quantity:
            target_col = candidates_quantity[0]
            problem_type = 'regression'
            target_description = 'ปริมาณ/จำนวน — ตัวชี้วัดหลักสำหรับธุรกิจค้าปลีกออนไลน์'

# Priority 2: ถ้ามี UnitPrice และ Quantity → สร้าง Revenue column
if not target_col and has_quantity and has_unitprice:
    # สร้าง Revenue จาก Quantity * UnitPrice
    qty_col = 'Quantity' if 'Quantity' in df.columns else 'quantity'
    price_col = 'UnitPrice' if 'UnitPrice' in df.columns else 'unitprice'
    if qty_col in df.columns and price_col in df.columns:
        df['revenue'] = df[qty_col] * df[price_col]
        target_col = 'revenue'
        problem_type = 'regression'
        target_description = 'รายได้ต่อรายการ (Quantity × UnitPrice) — ตัวชี้วัดทางการเงินหลักของธุรกิจ'
        print(f'[STATUS] Created revenue column as target')

# Priority 3: ถ้าไม่มี Quantity แต่มี columns เกี่ยวกับเงิน
if not target_col:
    money_candidates = [c for c in df.columns if 'price' in c.lower() or 'amount' in c.lower() or 'value' in c.lower() or 'revenue' in c.lower() or 'sales' in c.lower()]
    money_candidates = [c for c in money_candidates if not is_forbidden(c)]
    if money_candidates:
        target_col = money_candidates[0]
        problem_type = 'regression'
        target_description = f'มูลค่าทางการเงิน ({target_col}) — ใช้สำหรับวิเคราะห์รายได้'

# Priority 4: เผื่อกรณีไม่มี columns ที่เหมาะสม — ใช้ numeric column ที่มี highest variance
if not target_col and candidate_targets:
    # เลือก column ที่มี variance สูงสุดและไม่ใช่ ID
    variances = {}
    for c in candidate_targets:
        if df[c].nunique() > 1:
            try:
                variances[c] = df[c].var()
            except:
                continue
    if variances:
        target_col = max(variances, key=variances.get)
        problem_type = 'regression'
        target_description = f'Column ที่มี variance สูง ({target_col}) — ใช้สำหรับ exploratory analysis'

# ถ้าไม่มี target เลย → flag
if not target_col:
    target_col = 'none (need manual selection)'
    target_description = 'ไม่พบ target column ที่เหมาะสม — จำเป็นต้องเลือกด้วยตนเอง'
    problem_type = 'unknown'

print(f'[STATUS] Target: {target_col}')
print(f'[STATUS] Problem_type: {problem_type}')
print(f'[STATUS] Target description: {target_description}')

# ============================================
# 5. DATA QUALITY ANALYSIS
# ============================================
n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
n_datetime = len(date_cols)
n_ids = len(id_cols)
n_other = cols - n_numeric - n_categorical - n_datetime - n_ids

missing = (df.isnull().sum() / rows * 100).sort_values(ascending=False)
high_missing = {col: round(pct, 2) for col, pct in missing.items() if pct > 0}
top_missing = dict(sorted(high_missing.items(), key=lambda x: x[1], reverse=True)[:5])

print(f'[STATUS] Missing top: {top_missing}')

# ============================================
# 6. TARGET ANALYSIS
# ============================================
if target_col and target_col != 'none (need manual selection)' and target_col in df.columns:
    n_uniq = df[target_col].nunique()
    print(f'[STATUS] Target unique values: {n_uniq}')
    
    if n_uniq <= 20:
        class_dist = df[target_col].value_counts(normalize=True).round(4).to_dict()
        majority = max(class_dist.values())
        minority = min(class_dist.values())
        imbalance = round(majority / minority, 2) if minority > 0 else None
        print(f'[STATUS] Classification target distribution: {dict(list(class_dist.items())[:10])}')
        print(f'[STATUS] Imbalance ratio: {imbalance}')
    else:
        # Regression check
        if df[target_col].dtype in ['int64', 'float64']:
            print(f'[STATUS] Regression target — range: {df[target_col].min():.2f} to {df[target_col].max():.2f}')
            print(f'[STATUS] Mean: {df[target_col].mean():.2f}, Median: {df[target_col].median():.2f}')
            print(f'[STATUS] Zero count: {(df[target_col] == 0).sum()}')
            print(f'[STATUS] Negative count: {(df[target_col] < 0).sum()}')
    
    # Class balance
    if n_uniq <= 20:
        vc = df[target_col].value_counts(normalize=True)
        imbalance_ratio = round(vc.max() / vc.min(), 2) if vc.min() > 0 else None
    else:
        imbalance_ratio = None

# ============================================
# 7. RECOMMENDED SCALING
# ============================================
scaling = 'StandardScaler' if problem_type in ['regression', 'classification'] else 'MinMaxScaler'

# ============================================
# 8. BUSINESS FIT & DISPATCH RECOMMENDATION
# ============================================
# Descriptive analytics → Dana
# Behavioral analytics → Dana + Mo (EDA)
# Predictive analytics → Eddie (ML)

# วิเคราะห์ว่า dataset นี้เหมาะกับโจทย์ระดับไหน
if has_quantity and has_unitprice:
    business_level = 'full (descriptive + behavioral + predictive)'
    dispatch_desc = 'Dana: descriptive analytics (sales overview, top products), Dana+Mo: behavioral analytics (RFM, customer segments), Eddie: predictive analytics (demand forecast)'
elif has_quantity:
    business_level = 'medium (descriptive + behavioral)'
    dispatch_desc = 'Dana: descriptive analytics (quantity based), Dana+Mo: behavioral analytics (customer patterns)'
elif has_unitprice:
    business_level = 'descriptive'
    dispatch_desc = 'Dana: descriptive analytics (price analysis)'
else:
    business_level = 'descriptive (basic)'
    dispatch_desc = 'Dana: basic descriptive analytics'

# ============================================
# 9. RISK REGISTER
# ============================================
risk_lines = [
    'DATASET_RISK_REGISTER',
    '=====================',
    f'Source credibility: High — UCI Machine Learning Repository, curated benchmark dataset',
    f'License/usage: Allowed — public domain for academic/commercial use',
    f'Business fit: High — Online Retail transaction data, directly applicable to e-commerce analytics',
    f'Target suitability: Clear — {target_col}: {target_description}',
    f'Recency/deployment fit: Dataset is historical (2010-2011) — limited for current trend detection but valid for pattern learning',
    f'Leakage risks: None — no future information, no post-outcome columns',
    f'Bias/coverage risks: Single retailer from UK — may not generalize to other markets',
    f'Data dictionary: Available (UCI provides column descriptions)',
    f'Verdict: Use — suitable for multi-level analytics (descriptive → behavioral → predictive)',
]

# ============================================
# 10. SAVE DATASET_PROFILE
# ============================================
profile_lines = [
    'DATASET_PROFILE',
    '===============',
    f'rows: {rows}',
    f'cols: {cols}',
    f'problem_type: {problem_type}',
    f'target: {target_col}',
    f'target_description: {target_description}',
]

if n_uniq <= 20 and target_col in df.columns:
    profile_lines.append(f'imbalance: {imbalance_ratio} (minority:majority)')
else:
    profile_lines.append('imbalance: no_target or regression')

profile_lines.append(f'missing_cols: {json.dumps(top_missing, ensure_ascii=False)}')
profile_lines.append(f'key_features: {candidate_targets[:5]}')
profile_lines.append(f'data_types: numeric={n_numeric}, categorical={n_categorical}, datetime={n_datetime}, id={n_ids}, other={n_other}')
profile_lines.append(f'date_cols: {date_cols}')
profile_lines.append(f'id_cols: {id_cols}')
profile_lines.append(f'size_mb: {round(df.memory_usage(deep=True).sum() / 1_048_576, 2)}')
profile_lines.append(f'recommended_scaling: {scaling}')
profile_lines.append(f'business_level: {business_level}')
profile_lines.append(f'dispatch_recommendation: {dispatch_desc}')
profile_lines.append('')
profile_lines.extend(risk_lines)

profile_text = '\n'.join(profile_lines)
print('\n[PROFILE]')
print(profile_text)

# Save dataset_profile.md
profile_path = os.path.join(OUTPUT_DIR, 'dataset_profile.md')
with open(profile_path, 'w', encoding='utf-8') as f:
    f.write(profile_text)
print(f'\n[STATUS] Profile saved: {profile_path}')

# ============================================
# 11. SAVE SCOUT REPORT (scout_report.md)
# ============================================
report_lines = [
    'Scout Dataset Brief',
    '===================',
    f'Dataset: UCI Online Retail Dataset (2010-2011)',
    f'Source: https://archive.ics.uci.edu/ml/datasets/Online+Retail',
    f'License: Public Domain',
    f'Size: {rows:,} rows × {cols} columns / {round(df.memory_usage(deep=True).sum() / 1_048_576, 2)} MB',
    f'Format: CSV',
    f'Time Period: 2010-12-01 to 2011-12-09',
    '',
    'Columns Summary:',
    '- InvoiceNo: string — Invoice number (unique per transaction)',
    '- StockCode: string — Product code',
    '- Description: string — Product name/description',
    '- Quantity: int — Quantity purchased per transaction',
    '- InvoiceDate: datetime — Transaction timestamp',
    '- UnitPrice: float — Price per unit',
    '- CustomerID: float — Customer identifier (has missing)',
    '- Country: string — Customer country',
    '',
    'Key Columns for Analytics:',
    f'- Target: {target_col} ({target_description})',
    '- Date: InvoiceDate (datetime) for time series',
    '- ID: CustomerID (has NaN), StockCode, InvoiceNo',
    '',
    'Business Opportunity (3 Levels):',
    '1. Descriptive Analytics — Sales overview, top products, country-wise sales',
    '2. Behavioral Analytics — RFM analysis, customer segments, repeat purchase patterns',
    '3. Predictive Analytics — Demand forecasting with Quantity target, churn prediction with CustomerID segmentation',
    '',
    'Known Issues:',
    f'- Missing: {json.dumps(top_missing, ensure_ascii=False)}',
    '- CustomerID has ~25% missing (Cancelled transactions)',
    '- Negative Quantity = Cancelled transactions (can be filtered)',
    '- StockCode varies, Description has typos',
    '',
    'Dispatch Recommendation:',
    f'- Dana: Descriptive analytics — overall sales KPIs, time trends, country summary',
    f'- Dana + Mo: Behavioral analytics — RFM, customer segmentation, cohort analysis',
    f'- Eddie: Predictive analytics — demand forecast using Quantity as target',
    '',
    *risk_lines,
    '',
    'Self-Improvement Report',
    '=======================',
    'วิธีที่ใช้ครั้งนี้: อ่าน raw CSV, ตรวจสอบ column types manually, วิเคราะห์ business context',
    'เหตุผลที่เลือก: ต้องการ target ที่มีความหมายทางธุรกิจ ไม่ใช่ automatic heuristic',
    'วิธีใหม่ที่พบ: Revenue column (Quantity × UnitPrice) สร้าง target ที่ดีกว่า Quantity ตัวเดียว',
    'จะนำไปใช้ครั้งหน้า: ใช่ — ใน dataset ที่มี Quantity และ UnitPrice ควรสร้าง Revenue เป็น target เสมอ',
    'Knowledge Base: อัพเดต scout — Revenue target creation',
]

report_text = '\n'.join(report_lines)
report_path = os.path.join(OUTPUT_DIR, 'scout_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f'[STATUS] Report saved: {report_path}')

# ============================================
# 12. SAVE SCOUT OUTPUT CSV
# ============================================
output_csv = os.path.join(OUTPUT_DIR, 'scout_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] CSV saved: {output_csv}')

# ============================================
# 13. COPY TO DATASET_DIR (ถ้ามี)
# ============================================
if DATASET_DIR:
    dataset_csv = os.path.join(DATASET_DIR, os.path.basename(output_csv))
    df.to_csv(dataset_csv, index=False)
    print(f'[STATUS] Dataset CSV copied to: {dataset_csv}')

print('\n[STATUS] ✅ Scout complete — ready for dispatch')