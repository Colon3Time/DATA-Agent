import argparse
import os
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

print(f'[STATUS] INPUT_PATH: {INPUT_PATH}')
print(f'[STATUS] OUTPUT_DIR: {OUTPUT_DIR}')

# ------------------------------------------------------------
# 1. LOAD VERA OUTPUT
# ------------------------------------------------------------
df_vera = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded vera_output.csv: {df_vera.shape}')

# ------------------------------------------------------------
# 2. LOAD ALL AGENT OUTPUTS
# ------------------------------------------------------------
project_root = Path(INPUT_PATH).parent.parent
output_dir = project_root / 'output'

agent_outputs = {}
agent_order = ['dana', 'iris', 'vera', 'rex', 'quinn']
for agent in agent_order:
    files = list((output_dir / agent).glob('*_output*'))
    if files:
        f = str(files[0])
        df = pd.read_csv(f)
        agent_outputs[agent] = {'file': f, 'df': df}
        print(f'[STATUS] Loaded {agent}_output: {df.shape}')
    else:
        print(f'[WARN] No output found for {agent}')

# ------------------------------------------------------------
# 3. QUALITY CHECKS (12 checks)
# ------------------------------------------------------------
qc_results = []
qc_errors = []

# --- Check 1: Vera CSV shape ---
expected_rows = 5
expected_cols = 19
passed = df_vera.shape[0] == expected_rows and df_vera.shape[1] == expected_cols
qc_results.append({
    'check': '1_vera_shape',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'Expected ({expected_rows},{expected_cols}) → Got {df_vera.shape}'
})
if not passed:
    qc_errors.append('Vera CSV shape mismatch')

# --- Check 2: Missing values in Vera ---
total_cells = df_vera.size
missing_cells = int(df_vera.isnull().sum().sum())
pct_missing = (missing_cells / total_cells * 100) if total_cells > 0 else 0
passed = missing_cells == 0
qc_results.append({
    'check': '2_vera_missing_values',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'Missing cells: {missing_cells}/{total_cells} ({pct_missing:.1f}%)'
})
if not passed:
    qc_errors.append(f'Vera has {missing_cells} missing values')

# --- Check 3: Numeric columns valid ---
numeric_cols = df_vera.select_dtypes(include=[np.number]).columns
passed = len(numeric_cols) >= 3
qc_results.append({
    'check': '3_vera_numeric_columns',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'Numeric columns: {list(numeric_cols)} (count={len(numeric_cols)})'
})
if not passed:
    qc_errors.append('Vera has too few numeric columns')

# --- Check 4: No infinite values ---
has_inf = False
for col in df_vera.select_dtypes(include=[np.number]).columns:
    if df_vera[col].isnull().any() or np.isinf(df_vera[col]).any():
        has_inf = True
        break
passed = not has_inf
qc_results.append({
    'check': '4_vera_no_infinite',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'Has infinite or NaN values: {has_inf}'
})
if not passed:
    qc_errors.append('Vera has infinite or NaN values')

# --- Check 5: Data types consistency ---
passed = all(df_vera.dtypes.apply(lambda x: x in ['int64', 'float64', 'object', 'bool']))
qc_results.append({
    'check': '5_vera_data_types',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'Data types: {dict(df_vera.dtypes)}'
})
if not passed:
    qc_errors.append('Vera has unexpected data types')

# --- Check 6: No duplicate rows ---
dupes = df_vera.duplicated().sum()
passed = dupes == 0
qc_results.append({
    'check': '6_vera_no_duplicates',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'Duplicate rows: {dupes}'
})
if not passed:
    qc_errors.append(f'Vera has {dupes} duplicate rows')

# --- Check 7: Column names clean ---
passed = all(' ' not in col for col in df_vera.columns)
qc_results.append({
    'check': '7_vera_column_names_clean',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'Columns: {list(df_vera.columns)}'
})
if not passed:
    qc_errors.append('Vera has spaces in column names')

# --- Check 8: Numeric ranges valid (no negative where unexpected) ---
numeric_ranges_ok = True
for col in df_vera.select_dtypes(include=[np.number]).columns:
    if (df_vera[col] < 0).any():
        numeric_ranges_ok = False
        break
passed = numeric_ranges_ok
qc_results.append({
    'check': '8_vera_numeric_ranges',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'All numeric columns non-negative: {numeric_ranges_ok}'
})
if not passed:
    qc_errors.append('Vera has negative numeric values')

# --- Check 9: Object columns not empty ---
object_cols = df_vera.select_dtypes(include=['object']).columns
empty_strings = 0
for col in object_cols:
    empty_strings += (df_vera[col].astype(str).str.strip() == '').sum()
passed = empty_strings == 0
qc_results.append({
    'check': '9_vera_no_empty_strings',
    'status': 'PASS' if passed else 'FAIL',
    'detail': f'Empty strings in object columns: {empty_strings}'
})
if not passed:
    qc_errors.append(f'Vera has {empty_strings} empty strings')

# --- Check 10: Cross-agent consistency (Dana key exists) ---
if 'dana' in agent_outputs:
    dana_df = agent_outputs['dana']['df']
    common_cols = set(df_vera.columns) & set(dana_df.columns)
    passed = len(common_cols) >= 1
    qc_results.append({
        'check': '10_cross_agent_consistency',
        'status': 'PASS' if passed else 'FAIL',
        'detail': f'Common columns with Dana: {common_cols}'
    })
    if not passed:
        qc_errors.append('No common columns with Dana output')
else:
    qc_results.append({
        'check': '10_cross_agent_consistency',
        'status': 'SKIP',
        'detail': 'Dana output not found'
    })

# --- Check 11: Iris output exists ---
if 'iris' in agent_outputs:
    iris_df = agent_outputs['iris']['df']
    passed = iris_df.shape[0] > 0
    qc_results.append({
        'check': '11_iris_output_exists',
        'status': 'PASS' if passed else 'FAIL',
        'detail': f'Iris output shape: {iris_df.shape}'
    })
    if not passed:
        qc_errors.append('Iris output is empty')
else:
    qc_results.append({
        'check': '11_iris_output_exists',
        'status': 'SKIP',
        'detail': 'Iris output not found'
    })

# --- Check 12: Rex output exists ---
if 'rex' in agent_outputs:
    rex_df = agent_outputs['rex']['df']
    passed = rex_df.shape[0] > 0
    qc_results.append({
        'check': '12_rex_output_exists',
        'status': 'PASS' if passed else 'FAIL',
        'detail': f'Rex output shape: {rex_df.shape}'
    })
    if not passed:
        qc_errors.append('Rex output is empty')
else:
    qc_results.append({
        'check': '12_rex_output_exists',
        'status': 'SKIP',
        'detail': 'Rex output not found'
    })

# ------------------------------------------------------------
# 4. SAVE QC RESULTS
# ------------------------------------------------------------
qc_df = pd.DataFrame(qc_results)
qc_df['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

output_csv = os.path.join(OUTPUT_DIR, 'quinn_qc_results.csv')
qc_df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved QC results: {output_csv}')

# ------------------------------------------------------------
# 5. SUMMARY
# ------------------------------------------------------------
passed_count = len(qc_df[qc_df['status'] == 'PASS'])
failed_count = len(qc_df[qc_df['status'] == 'FAIL'])
skipped_count = len(qc_df[qc_df['status'] == 'SKIP'])

print(f'\n[STATUS] QC Summary: {passed_count} PASS, {failed_count} FAIL, {skipped_count} SKIP')
if qc_errors:
    print(f'[ERRORS] Issues found: {len(qc_errors)}')
    for err in qc_errors:
        print(f'  - {err}')
else:
    print('[STATUS] All checks passed!')

# ------------------------------------------------------------
# 6. SELF-IMPROVEMENT REPORT
# ------------------------------------------------------------
si_report = f"""Agent Report — Quinn
============================
รับจาก     : Vera
Input      : {INPUT_PATH}
ทำ         : Quality Check — ตรวจสอบคุณภาพ Vera output และ cross-agent consistency
พบ         : {passed_count}/{len(qc_results)} checks ผ่าน, {failed_count} FAIL, {skipped_count} SKIP
เปลี่ยนแปลง: ไม่มีการเปลี่ยนแปลงข้อมูล (เป็นการตรวจสอบเท่านั้น)
ส่งต่อ     : Rex — quinn_qc_results.csv

Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: Multi-agent QC framework
เหตุผลที่เลือก: ครอบคลุมทั้ง data integrity (missing, duplicates, types), format standards (column names, empty strings), และ cross-agent consistency
วิธีใหม่ที่พบ: ไม่พบวิธีใหม่
จะนำไปใช้ครั้งหน้า: ใช่ — framework นี้ใช้ได้ดีสำหรับงาน EDA QC ทั่วไป
Knowledge Base: ไม่มีการเปลี่ยนแปลง
"""

# Save report
report_path = os.path.join(OUTPUT_DIR, 'qc_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(si_report)
print(f'[STATUS] Saved QC report: {report_path}')

# Save self-improvement too
si_path = os.path.join(OUTPUT_DIR, 'self_improvement_report.md')
with open(si_path, 'w', encoding='utf-8') as f:
    f.write(si_report)
print(f'[STATUS] Saved self-improvement report: {si_path}')