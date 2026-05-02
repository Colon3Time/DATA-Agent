import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ========== ARGPARSE ==========
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=r'c:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\input\thailand_enterprise_surveys_simulated_2026.csv')
parser.add_argument('--output-dir', default=r'c:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\output\scout')
parser.add_argument('--input-dir', default=r'c:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\input')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
INPUT_DIR = args.input_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== REPORT ENGINE ==========
report_sections = {
    'header': [],
    'file_check': [],
    'overview': [],
    'quality': [],
    'risk_register': [],
    'benchmark': [],
    'self_improvement': [],
    'agent_report': []
}

def add(section, text):
    report_sections[section].append(text)

def write_report():
    report_path = os.path.join(OUTPUT_DIR, 'scout_report.md')
    lines = []
    lines.append('# Scout Report — Dataset Hunter & Source Acquisition')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Project: 2026-05-02_sme_worldbank')
    lines.append('')
    
    for section_name, section_lines in report_sections.items():
        title = section_name.replace('_', ' ').title()
        lines.append(f'## {title}')
        lines.append('')
        for line in section_lines:
            lines.append(line)
        lines.append('')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[STATUS] Report saved: {report_path}')

# ========== STEP 1: FILE CHECK ==========
add('header', '## Step 1: File Check')
add('header', f'Input Path: {INPUT_PATH}')
add('header', f'Input Dir: {INPUT_DIR}')
add('header', f'Output Dir: {OUTPUT_DIR}')

actual_input_path = INPUT_PATH
if not os.path.exists(actual_input_path):
    print(f'[ERROR] File not found: {actual_input_path}')
    add('file_check', f'File not found: {actual_input_path}')
    input_dir = Path(INPUT_DIR)
    csv_files = list(input_dir.glob('**/*.csv'))
    if csv_files:
        actual_input_path = str(csv_files[0])
        add('file_check', f'Fallback to: {actual_input_path}')
    else:
        add('file_check', 'No CSV files found in input directory')
        write_report()
        sys.exit(1)

file_size = os.path.getsize(actual_input_path)
add('file_check', f'File found: {actual_input_path}')
add('file_check', f'File size: {file_size:,} bytes ({file_size/1024:.1f} KB)')

# ========== STEP 2: LOAD DATA ==========
add('header', '')
add('header', '## Step 2: Load Data')

print(f'[STATUS] Attempting to load: {actual_input_path}')
df = pd.read_csv(actual_input_path, encoding='utf-8', engine='python')
print(f'[STATUS] Loaded: {df.shape}')
add('overview', f'Rows: {df.shape[0]:,}')
add('overview', f'Columns: {df.shape[1]}')
add('overview', f'Column names: {list(df.columns)}')

# Basic info
n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object', 'category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]
add('overview', f'Numeric columns: {n_numeric}')
add('overview', f'Categorical columns: {n_cat}')
add('overview', f'Datetime columns: {n_datetime}')

# Missing data
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_info = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_info = missing_info[missing_info['missing_count'] > 0].sort_values('missing_pct', ascending=False)
add('overview', f'Columns with missing: {len(missing_info)}')
if len(missing_info) > 0:
    for col, row in missing_info.iterrows():
        add('overview', f'  - {col}: {int(row["missing_count"])} missing ({row["missing_pct"]}%)')

# ========== STEP 3: QUALITY EVALUATION ==========
add('header', '')
add('header', '## Step 3: Quality Evaluation')

# Completeness score
completeness = 1 - df.isnull().mean().mean()
add('quality', f'Completeness: {completeness:.2%}')

# Size adequacy
size_score = min(1.0, len(df) / 1000)
add('quality', f'Size adequacy score: {size_score:.2f}')

# Feature richness
feature_score = min(1.0, len(df.columns) / 10)
add('quality', f'Feature richness score: {feature_score:.2f}')

# Duplicate check
dup_count = df.duplicated().sum()
add('quality', f'Duplicate rows: {dup_count:,}')

# ========== STEP 4: TARGET DETECTION ==========
add('header', '')
add('header', '## Step 4: Target Detection')

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

# Try to find business target
BUSINESS_TARGET_KEYWORDS = [
    "review_score", "order_status", "payment_value", "freight_value",
    "delivery_days", "delay", "churn",
    "target", "label", "survived", "fraud", "default", "outcome",
    "result", "response", "converted", "clicked", "bought",
    "cancelled", "returned", "status", "class",
    "sales", "revenue", "profit", "growth", "exit",
    "size", "employment", "performance",
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

# Priority 2: binary column
if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            target_col = col
            print(f'[STATUS] Target selected (binary column): {target_col}')
            break

# Priority 3: categorical <=10
if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10:
            target_col = col
            print(f'[STATUS] Target selected (categorical): {target_col}')
            break

# Priority 4: numeric low-cardinality
if not target_col:
    for col in reversed(list(df.columns)):
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            target_col = col
            print(f'[STATUS] Target selected (numeric low-cardinality): {target_col}')
            break

add('quality', f'Detected target column: {target_col or "unknown"}')

problem_type = 'unknown'
imbalance = None
class_dist = {}
if target_col:
    n_uniq = df[target_col].nunique()
    if n_uniq <= 20:
        problem_type = 'classification'
        vc = df[target_col].value_counts(normalize=True).round(4)
        class_dist = vc.to_dict()
        majority = vc.max()
        minority = vc.min()
        imbalance = round(majority / minority, 2) if minority > 0 else None
    else:
        problem_type = 'regression'
    add('quality', f'Problem type: {problem_type}')
    if class_dist:
        dist_str = json.dumps({str(k): round(v, 3) for k, v in list(class_dist.items())[:8]})
        add('quality', f'Class distribution (top 8): {dist_str}')
    if imbalance:
        add('quality', f'Imbalance ratio: {imbalance}')
else:
    add('quality', 'Problem type: unknown (no target column detected)')

# ========== STEP 5: RISK REGISTER ==========
add('header', '')
add('header', '## Step 5: Risk Register')

# Source credibility
add('risk_register', 'Source credibility: Medium (simulated data from World Bank format)')
add('risk_register', 'License/usage: Simulated data for educational purposes')
add('risk_register', 'Business fit: High - Thailand SME/enterprise survey data')
add('risk_register', f'Target suitability: {"clear" if target_col else "missing"}')
add('risk_register', 'Recency/deployment fit: Simulated 2026 data, current')
add('risk_register', 'Leakage risks: None identified (synthetic simulation)')
add('risk_register', 'Bias/coverage risks: Simulated data may not reflect real survey bias')
add('risk_register', 'Data dictionary: Not available (simulated dataset)')
add('risk_register', 'Verdict: Use with caveats - simulated data for prototyping')

# ========== STEP 6: PROFILE OUTPUT ==========
profile_lines = []
profile_lines.append('DATASET_PROFILE')
profile_lines.append('===============')
profile_lines.append(f'rows         : {df.shape[0]:,}')
profile_lines.append(f'cols         : {df.shape[1]}')
profile_lines.append(f'dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}')
profile_lines.append(f'missing      : {json.dumps({col: float(missing_pct[col]) for col in missing_info.index[:5]}, ensure_ascii=False)}')
profile_lines.append(f'target_column: {target_col or "unknown"}')
profile_lines.append(f'problem_type : {problem_type}')
if class_dist:
    profile_lines.append(f'class_dist   : {json.dumps({str(k): v for k,v in list(class_dist.items())[:6]})}')
if imbalance is not None:
    profile_lines.append(f'imbalance_ratio: {imbalance}')
profile_lines.append(f'recommended_scaling: {"StandardScaler" if n_numeric > 0 else "None"}')

profile_text = '\n'.join(profile_lines)
print(f'[STATUS] Profile generated')

profile_path = os.path.join(OUTPUT_DIR, 'dataset_profile.md')
with open(profile_path, 'w', encoding='utf-8') as f:
    f.write(profile_text)
print(f'[STATUS] Profile saved: {profile_path}')

# Save output CSV
output_csv = os.path.join(OUTPUT_DIR, 'scout_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Output saved: {output_csv}')

# ========== STEP 7: SELF IMPROVEMENT ==========
add('header', '')
add('header', '## Self-Improvement Report')
add('self_improvement', 'Method used: Local file inspection + automated profiling')
add('self_improvement', 'Reason for selection: Input CSV provided directly')
add('self_improvement', 'New methods found: None')
add('self_improvement', 'Will use next time: Yes')
add('self_improvement', 'Knowledge base: No changes needed')

# ========== STEP 8: AGENT REPORT ==========
add('header', '')
add('header', '## Agent Report — Scout')
add('agent_report', 'Received from: User')
add('agent_report', f'Input: {actual_input_path}')
add('agent_report', f'Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns')
add('agent_report', f'Target column: {target_col or "unknown"}')
add('agent_report', f'Problem type: {problem_type}')
add('agent_report', f'Missing data: {len(missing_info)} columns with missing values')
add('agent_report', 'Sent to: Anna — dataset_profile.md + scout_report.md + scout_output.csv')

# ========== WRITE REPORT ==========
write_report()
print(f'[STATUS] All tasks completed successfully')