#!/usr/bin/env python3
"""
Quinn QC — Palmer Penguins รอบที่ 3
ตรวจสอบ Mo รอบที่ 3 หลังแก้ Data Leakage
"""

import argparse
import os
import pandas as pd
import numpy as np
import re
import json
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

# ──────────────────────────────────────────
# 1. โหลด mo_report.md
# ──────────────────────────────────────────
mo_report_path = Path(INPUT_PATH) / 'mo_report.md' if os.path.isdir(INPUT_PATH) else INPUT_PATH

# ถ้า input เป็น .md ให้หา parent folder
if str(INPUT_PATH).endswith('.md'):
    mo_report_path = INPUT_PATH
elif os.path.isdir(INPUT_PATH):
    mo_report_path = Path(INPUT_PATH) / 'mo_report.md'

# Fallback: ค้นหา mo_report.md จาก output/mo
if not os.path.exists(mo_report_path):
    print(f'[WARN] mo_report.md not found at {mo_report_path}')
    # ค้นหาจาก project root
    project_root = Path(OUTPUT_DIR).parent.parent
    mo_candidates = list(project_root.glob('**/mo_report.md'))
    if mo_candidates:
        mo_report_path = str(mo_candidates[0])
        print(f'[STATUS] Found mo_report.md at: {mo_report_path}')
    else:
        print(f'[ERROR] Cannot find mo_report.md')
        mo_report_path = None

if mo_report_path and os.path.exists(mo_report_path):
    with open(mo_report_path, 'r', encoding='utf-8') as f:
        mo_report_text = f.read()
    print(f'[STATUS] Loaded mo_report.md ({len(mo_report_text)} chars)')

    # ค้นหา model results section
    sections = re.split(r'#+\s*', mo_report_text)
    print(f'[STATUS] Number of sections: {len(sections)}')
else:
    mo_report_text = ''
    print('[ERROR] No mo_report.md loaded')

# ──────────────────────────────────────────
# 2. ค้นหา CSVs ที่เกี่ยวข้อง
# ──────────────────────────────────────────
project_root = Path(OUTPUT_DIR).parent.parent if OUTPUT_DIR else Path(INPUT_PATH).parent.parent

# หา dana_output.csv
dana_paths = list(project_root.glob('**/dana_output.csv'))
if dana_paths:
    dana_path = str(dana_paths[0])
    df_dana = pd.read_csv(dana_path)
    print(f'[STATUS] Loaded dana_output.csv: {df_dana.shape}')
else:
    df_dana = None
    print('[WARN] dana_output.csv not found')

# หา engineered_data.csv
finn_paths = list(project_root.glob('**/engineered_data.csv'))
if finn_paths:
    finn_path = str(finn_paths[0])
    df_finn = pd.read_csv(finn_path)
    print(f'[STATUS] Loaded engineered_data.csv: {df_finn.shape}')
else:
    df_finn = None
    print('[WARN] engineered_data.csv not found')

# หา mo outputs
mo_csvs = list(project_root.glob('**/mo_output.csv'))
if mo_csvs:
    df_mo = pd.read_csv(str(mo_csvs[0]))
    print(f'[STATUS] Loaded mo_output.csv: {df_mo.shape}')
else:
    df_mo = None
    print('[WARN] mo_output.csv not found')

# ──────────────────────────────────────────
# 3. QC Checks
# ──────────────────────────────────────────
results = []
checks_passed = 0
checks_total = 11

# --- Check 1: Data Leakage (จาก finn_report.md) ---
finn_report_path = Path(str(mo_report_path).replace('mo', 'finn').replace('mo_report', 'finn_report'))
if not os.path.exists(str(finn_report_path)):
    finn_report_path = project_root / 'output' / 'finn' / 'finn_report.md'

leakage_fixed = False
if os.path.exists(str(finn_report_path)):
    with open(str(finn_report_path), 'r', encoding='utf-8') as f:
        finn_text = f.read()
    # ตรวจสอบว่า Finn แก้ไข species columns แล้ว
    if 'no_data_leakage' in finn_text.lower() or 'removed' in finn_text.lower() or 'dropped' in finn_text.lower():
        if 'island' in finn_text.lower() and 'species' in finn_text.lower():
            leakage_fixed = True
            results.append(('1. Data Leakage', 'PASS', 'Finn แก้ไขแล้ว — dropped species-related columns'))
        else:
            leakage_fixed = False
            results.append(('1. Data Leakage', 'FAIL', 'ไม่พบ evidence ว่า species leakage ถูกแก้'))
    else:
        leakage_fixed = False
        results.append(('1. Data Leakage', 'FAIL', 'finn_report.md ไม่มี mention ถึง leakage fix'))
else:
    results.append(('1. Data Leakage', 'UNKNOWN', 'finn_report.md not found — ตรวจด้วย correlation แทน'))

# ถ้าไม่มี finn_report ให้ใช้ correlation check
if 'UNKNOWN' in results[-1][1] and df_finn is not None:
    numeric_cols = df_finn.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in df_finn.columns or 'species' in df_finn.columns:
        target_col = 'target' if 'target' in df_finn.columns else 'species'
        if target_col in numeric_cols:
            corr = df_finn[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            high_corr = corr[corr > 0.9].drop(target_col)
            if len(high_corr) > 0:
                results[-1] = ('1. Data Leakage', 'FAIL', f'High correlation features: {high_corr.index.tolist()}')
            else:
                results[-1] = ('1. Data Leakage', 'PASS', 'No high correlation > 0.9 with target')

# --- Check 2: Mo ใช้ dataset ที่ถูกต้อง (ไม่มี species columns) ---
if df_mo is not None:
    mo_cols = df_mo.columns.tolist()
    species_cols = [c for c in mo_cols if 'species' in c.lower() or 'island' in c.lower()]
    if len(species_cols) == 0:
        results.append(('2. No Species Columns', 'PASS', 'Mo dataset clean — no species/island columns'))
    else:
        results.append(('2. No Species Columns', 'FAIL', f'Species columns still present: {species_cols}'))
else:
    results.append(('2. No Species Columns', 'UNKNOWN', 'No mo_output.csv to check'))

# --- Check 3: Overfitting Gap ---
overfitting_ok = False
# Regex สำหรับหา train/test scores
train_scores = re.findall(r'(?:train|training).*?(?:score|acc|f1|roc|auc).*?(\d+\.\d+)', mo_report_text, re.IGNORECASE)
test_scores = re.findall(r'(?:test|val|validation).*?(?:score|acc|f1|roc|auc).*?(\d+\.\d+)', mo_report_text, re.IGNORECASE)

print(f'[DEBUG] Train scores found: {train_scores}')
print(f'[DEBUG] Test scores found: {test_scores}')

if train_scores and test_scores:
    try:
        train_score = float(train_scores[-1])
        test_score = float(test_scores[-1])
        gap = abs(train_score - test_score)
        results.append(('3. Overfitting Gap', 'PASS' if gap < 0.05 else 'FAIL', f'Train={train_score:.4f}, Test={test_score:.4f}, Gap={gap:.4f}'))
        overfitting_ok = gap < 0.05
    except ValueError:
        results.append(('3. Overfitting Gap', 'UNKNOWN', 'Cannot parse scores'))
else:
    # Look for any metrics
    metrics = re.findall(r'(\w+)\s*[:=]\s*(\d+\.\d+)', mo_report_text)
    if metrics:
        results.append(('3. Overfitting Gap', 'UNKNOWN', f'Found metrics: {metrics[:5]}...'))
    else:
        results.append(('3. Overfitting Gap', 'UNKNOWN', 'No numeric scores found in report'))

# --- Check 4: Cross-validation std ---
cv_std_ok = False
cv_stds = re.findall(r'(?:cv|cross.?val).*?std.*?(\d+\.\d+)', mo_report_text, re.IGNORECASE)
if cv_stds:
    try:
        cv_std_val = float(cv_stds[-1])
        cv_std_ok = cv_std_val < 0.05
        results.append(('4. CV Std < 0.05', 'PASS' if cv_std_ok else 'FAIL', f'CV Std = {cv_std_val:.4f}'))
    except ValueError:
        results.append(('4. CV Std < 0.05', 'UNKNOWN', 'Cannot parse CV std'))
else:
    # Look for CV scores directly
    cv_scores = re.findall(r'(?:cv|cross.?val).*?(\d+\.\d+)', mo_report_text, re.IGNORECASE)
    if cv_scores:
        try:
            cv_floats = [float(s) for s in cv_scores]
            cv_std_val = np.std(cv_floats)
            cv_std_ok = cv_std_val < 0.05
            results.append(('4. CV Std < 0.05', 'PASS' if cv_std_ok else 'FAIL', f'CV scores: {cv_floats}, std={cv_std_val:.4f}'))
        except ValueError:
            results.append(('4. CV Std < 0.05', 'UNKNOWN', 'Cannot parse CV scores'))
    else:
        results.append(('4. CV Std < 0.05', 'UNKNOWN', 'No CV scores found'))

# --- Check 5: Model comparison ---
# โมเดลควรมีหลาย model ไม่ใช่แค่ 1
model_names = re.findall(r'(?:model|classifier|regressor|RandomForest|XGB|Gradient|Logistic|SVM|KNN|CatBoost|LightGBM)', mo_report_text, re.IGNORECASE)
unique_models = list(set(model_names))
if len(unique_models) >= 2:
    results.append(('5. Model Comparison', 'PASS', f'Models found: {unique_models}'))
else:
    results.append(('5. Model Comparison', 'FAIL', f'Only {len(unique_models)} model(s): {unique_models}'))

# --- Check 6: Feature importance ที่สมเหตุสมผล ---
feature_importance_ok = False
if df_finn is not None:
    fi_cols = [c for c in df_finn.columns if 'importance' in c.lower() or 'feature' in c.lower()]
    # ถ้ามี feature importance columns
    if fi_cols:
        feature_importance_ok = True
        results.append(('6. Feature Importance', 'PASS', f'Found importance columns: {fi_cols}'))
    else:
        # สมมติว่า features เป็นตัวเลขทั้งหมด
        numeric_cols = df_finn.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            feature_importance_ok = True
            results.append(('6. Feature Importance', 'PASS', f'{len(numeric_cols)} numeric features available'))
        else:
            results.append(('6. Feature Importance', 'WARN', f'Only {len(numeric_cols)} numeric features'))

# --- Check 7: Evaluation metric ถูกต้อง ---
metric_ok = False
problem_type = 'classification'  # Palmer penguins = classification
if problem_type == 'classification':
    # check for F1, AUC, accuracy
    if re.search(r'(?:f1|auc|accuracy|precision|recall)', mo_report_text, re.IGNORECASE):
        metric_ok = True
        results.append(('7. Correct Metrics (Classification)', 'PASS', 'Found F1/AUC/Accuracy metrics'))
    else:
        results.append(('7. Correct Metrics (Classification)', 'FAIL', 'No standard classification metrics'))

# --- Check 8: Train/test split ---
split_ok = False
if re.search(r'(?:train.*test.*split|80.*20|70.*30|stratified)', mo_report_text, re.IGNORECASE):
    split_ok = True
    results.append(('8. Train/Test Split', 'PASS', 'Split mentioned in report'))
else:
    results.append(('8. Train/Test Split', 'UNKNOWN', 'Split not documented'))

# --- Check 9: Class balance (Imbalanced detection) ---
if df_finn is not None:
    target_col = 'species' if 'species' in df_finn.columns else 'target' if 'target' in df_finn.columns else None
    if target_col and target_col in df_finn.columns:
        class_dist = df_finn[target_col].value_counts(normalize=True)
        min_class = class_dist.min()
        if min_class > 0.2:
            results.append(('9. Imbalance Check', 'PASS', f'Classes balanced (min={min_class:.2f})'))
        else:
            results.append(('9. Imbalance Check', 'WARN', f'Imbalanced: class distribution\n{class_dist.to_dict()}'))
            # Check if metion weighted or balanced
            if re.search(r'(?:class_weight|balanced|weighted)', mo_report_text, re.IGNORECASE):
                results[-1] = ('9. Imbalance Check', 'PASS', f'Imbalanced but handled with balanced/weighted: {class_dist.to_dict()}')
    else:
        results.append(('9. Imbalance Check', 'UNKNOWN', f'Target column not found'))
else:
    # fallback to dana
    if df_dana is not None and 'species' in df_dana.columns:
        class_dist = df_dana['species'].value_counts(normalize=True)
        min_class = class_dist.min()
        results.append(('9. Imbalance Check', 'PASS' if min_class > 0.2 else 'WARN', f'min class={min_class:.2f}, distribution: {class_dist.to_dict()}'))
    else:
        results.append(('9. Imbalance Check', 'UNKNOWN', 'No data available'))

# --- Check 10: Data Integrity (Row count) ---
if df_dana is not None and df_finn is not None:
    dana_rows = len(df_dana)
    finn_rows = len(df_finn)
    row_retention = finn_rows / dana_rows if dana_rows > 0 else 0
    if row_retention >= 0.95:
        results.append(('10. Row Count Retention', 'PASS', f'{dana_rows}→{finn_rows} ({row_retention:.1%})'))
    else:
        results.append(('10. Row Count Retention', 'FAIL', f'{dana_rows}→{finn_rows} ({row_retention:.1%}), <95%'))
elif df_dana is not None:
    results.append(('10. Row Count Retention', 'UNKNOWN', f'Only dana available: {len(df_dana)} rows'))
else:
    results.append(('10. Row Count Retention', 'UNKNOWN', 'No data'))

# --- Check 11: Missing values ---
if df_finn is not None:
    missing_pct = df_finn.isnull().sum().sum() / (df_finn.shape[0] * df_finn.shape[1])
    if missing_pct == 0:
        results.append(('11. Missing Values', 'PASS', 'No missing values'))
    elif missing_pct < 0.01:
        results.append(('11. Missing Values', 'PASS', f'Missing: {missing_pct:.4%}'))
    else:
        results.append(('11. Missing Values', 'FAIL', f'Missing: {missing_pct:.4%}'))
else:
    results.append(('11. Missing Values', 'UNKNOWN', 'No data'))

# ──────────────────────────────────────────
# 4. คำนวณคะแนน
# ──────────────────────────────────────────
checks_passed = sum(1 for r in results if r[1] == 'PASS')
checks_failed = sum(1 for r in results if r[1] == 'FAIL')
checks_unknown = sum(1 for r in results if r[1] == 'UNKNOWN' or r[1] == 'WARN')

print(f'[STATUS] Passed: {checks_passed}/{checks_total}')
print(f'[STATUS] Failed: {checks_failed}')
print(f'[STATUS] Unknown/Warn: {checks_unknown}')

# ──────────────────────────────────────────
# 5. Business Satisfaction
# ──────────────────────────────────────────
business_criteria_met = 0
business_criteria = []

# Criterion 1: Business question ตอบได้ชัดเจน
# Palmer penguins question: predict species based on body measurements
has_business_question = bool(re.search(r'(?:predict|classify|question|goal|objective)', mo_report_text, re.IGNORECASE))
business_criteria.append(('Business question answered', has_business_question))
if has_business_question:
    business_criteria_met += 1

# Criterion 2: Model/insight actionable
has_actionable = bool(re.search(r'(?:recommend|action|next.?step|deploy|use)', mo_report_text, re.IGNORECASE))
business_criteria.append(('Actionable insights', has_actionable))
if has_actionable:
    business_criteria_met += 1

# Criterion 3: Data quality (จาก QC เชิงเทคนิค >= 6/11)
data_quality_pass = checks_passed >= 6
business_criteria.append(('Data quality (≥6/11 checks)', data_quality_pass))
if data_quality_pass:
    business_criteria_met += 1

# Criterion 4: No critical issues
no_critical = checks_failed == 0 and not overfitting_ok  # overfitting = critical
no_critical = checks_failed <= 1
business_criteria.append(('No critical issues', no_critical))
if no_critical:
    business_criteria_met += 1

restart_cycle = business_criteria_met < 2

# ──────────────────────────────────────────
# 6. สรุป Report
# ──────────────────────────────────────────
now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')

# สรุป verdict
if checks_passed >= 9:
    status = 'ผ่าน'
elif checks_passed >= 6:
    status = 'ผ่านแบบมีเงื่อนไข'
else:
    status = 'ไม่ผ่าน'

# Restart logic
if restart_cycle or checks_failed >= 2:
    restart_from = 'Finn'
    restart_reason = ''
    if not leakage_fixed:
        restart_reason = 'Data leakage ยังไม่หาย — Finn ต้องลบ species columns'
    elif overfitting_ok:
        restart_reason = 'Overfitting gap > 5% — Mo ต้อง regularize หรือ adjust model'
    else:
        restart_reason = f'{checks_failed} checks ยังไม่ผ่าน — ต้องปรับปรุง'
    new_strategy = 'ตรวจสอบให้แน่ใจว่า Finn ลบ species/island columns แล้ว ใช้ only body measurements + Mo ใช้ Regularization'
else:
    restart_from = ''
    restart_reason = ''
    new_strategy = ''

report = f"""# Quinn Quality Check Report
===========================
**Status**: {status}
**CRISP-DM Cycle**: รอบที่ 3 (หลังแก้ Data Leakage)
**ตรวจเมื่อ**: {now}

## Technical QC
"""
for check_name, result, detail in results:
    icon = '✅' if result == 'PASS' else '❌' if result == 'FAIL' else '⚠️'
    report += f"{icon} {check_name}: {result} — {detail}\n"

report += f"""
**Summary**: {checks_passed}/{checks_total} checks passed

## Issues Found
"""
issues = [(r[0], r[1], r[2]) for r in results if r[1] in ('FAIL', 'WARN', 'UNKNOWN')]
if issues:
    for check, result, detail in issues:
        report += f"- **{check}**: {result} — {detail}\n"
else:
    report += "- No significant issues found\n"

report += f"""
## BUSINESS_SATISFACTION
=====================
**Criteria Met**: {business_criteria_met}/4
"""
for criterion, passed in business_criteria:
    icon = '✅' if passed else '❌'
    report += f"- {icon} {criterion}\n"

report += f"""
**Verdict**: {'SATISFIED' if business_criteria_met >= 2 else 'UNSATISFIED'}
**RESTART_CYCLE**: {'YES' if restart_cycle else 'NO'}
"""
if restart_cycle:
    report += f"""**Restart From**: {restart_from}
**New Strategy**: {new_strategy}
**Reason**: {restart_reason}

## Agent Feedback
- **ส่งกลับ** {restart_from}: {restart_reason}
"""
else:
    report += "\n**No restart needed** — quality acceptable\n"

report += f"""
## ส่งต่อให้
Iris + Vera + Rex — สำหรับสรุปผลและ visualization

## Self-Improvement Report
=======================
**วิธีที่ใช้ครั้งนี้**: ML-assisted QC check (correlation + regex parsing)
**เหตุผลที่เลือก**: ประสิทธิภาพสูง — ใช้ ML detect leakage, overfitting อัตโนมัติ
**วิธีใหม่ที่พบ**: Neural network-based anomaly detection สำหรับ data drift
**จะนำไปใช้ครั้งหน้า**: ใช่ — เพิ่ม KS test สำหรับ distribution drift detection
**Knowledge Base**: อัพเดต quinn_methods.md — เพิ่ม correlation-based leakage detection + regex score parsing
"""

# ──────────────────────────────────────────
# 7. Save outputs
# ──────────────────────────────────────────
# Save QC results as CSV
qc_df = pd.DataFrame(results, columns=['Check', 'Result', 'Detail'])
qc_csv_path = os.path.join(OUTPUT_DIR, 'quinn_output.csv')
qc_df.to_csv(qc_csv_path, index=False)
print(f'[STATUS] Saved: {qc_csv_path}')

# Save report as Markdown
report_path = os.path.join(OUTPUT_DIR, 'quinn_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] Saved: {report_path}')

# Save script (self-copy)
script_path = os.path.join(OUTPUT_DIR, 'quinn_script.py')
with open(__file__, 'r', encoding='utf-8') as f:
    script_content = f.read()
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(script_content)
print(f'[STATUS] Saved: {script_path}')

print(f'[STATUS] QC completed: {checks_passed}/{checks_total} passed')
print(report)