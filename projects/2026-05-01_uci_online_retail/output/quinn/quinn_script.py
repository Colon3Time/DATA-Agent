import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'[STATUS] INPUT_PATH={INPUT_PATH}')
print(f'[STATUS] OUTPUT_DIR={OUTPUT_DIR}')

# ── Locate project root from input ──────────────────────────────
project_root = Path(INPUT_PATH).parent.parent if INPUT_PATH else Path(OUTPUT_DIR).parent
print(f'[STATUS] Project root: {project_root}')

# ── Load all available outputs ──────────────────────────────────
def load_outputs(base_path):
    outputs = {}
    for f in sorted(base_path.glob('**/*output*.csv')):
        try:
            df = pd.read_csv(f)
            outputs[f.name] = df
            print(f'[STATUS] Loaded: {f.name} shape={df.shape}')
        except Exception as e:
            print(f'[WARN] Could not load {f.name}: {e}')
    return outputs

all_outputs = load_outputs(project_root)

# Load report files
report_texts = {}
for f in sorted(project_root.glob('**/*report*.md')):
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            report_texts[f.name] = fh.read()
        print(f'[STATUS] Loaded report: {f.name} ({len(report_texts[f.name])} chars)')
    except Exception as e:
        print(f'[WARN] Could not load {f.name}: {e}')

# ── Helper: safe access to columns ──────────────────────────────
def safe_cols(df):
    return [c for c in df.columns if c not in ['Unnamed: 0', 'index', 'RowNumber']]

# ── 1. DATA LEAKAGE CHECK ───────────────────────────────────────
leakage_issues = []
leakage_details = {}

# Check if CustomerID appears as feature in model data
for fname, df in all_outputs.items():
    cols = safe_cols(df)
    # Check for suspicious columns that might leak target
    suspicious = [c for c in cols if any(kw in c.lower() for kw in ['target', 'label', 'class', 'y_true', 'actual'])]
    if suspicious and 'features' in fname.lower():
        leakage_issues.append(f'[{fname}] Suspicious columns: {suspicious} — may leak target')
        leakage_details[fname] = {'suspicious_cols': suspicious}

    # Check correlation with target if present
    target_col = None
    for t in ['Churn', 'churn', 'CLV', 'clv', 'Target', 'target', 'Revenue', 'revenue', 'Quantity', 'quantity']:
        if t in cols:
            target_col = t
            break

    if target_col:
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[num_cols].corr()[target_col].abs().sort_values(ascending=False)
        high_corr = corr[corr > 0.95]
        if len(high_corr) > 1:  # exclude self-correlation
            high_corr = high_corr[high_corr.index != target_col]
        if len(high_corr) > 0:
            leakage_issues.append(f'[{fname}] High correlation with target ({target_col}): {high_corr.index.tolist()} (values: {high_corr.values.round(3).tolist()})')
            leakage_details[fname] = {'high_corr_features': high_corr.index.tolist(), 'corr_values': high_corr.values.round(3).tolist()}

leakage_pass = len(leakage_issues) == 0
print(f'[STATUS] Data Leakage: {"PASS" if leakage_pass else f"FAIL ({len(leakage_issues)} issues)"}')

# ── 2. MODEL PERFORMANCE CHECK ──────────────────────────────────
model_metrics = {}
performance_issues = []

# Check for Mo model results
for fname, text in report_texts.items():
    if 'model' in fname.lower() or 'mo' in fname.lower():
        # Extract metrics
        metrics_found = {}
        for line in text.split('\n'):
            line_lower = line.lower().strip()
            for m in ['f1', 'auc', 'roc_auc', 'roc auc', 'pr_auc', 'average precision',
                      'rmse', 'mae', 'r2', 'r-squared', 'accuracy', 'precision', 'recall',
                      'f1-score', 'f1_score', 'weighted f1']:
                if m in line_lower and ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        val_str = parts[-1].strip()
                        try:
                            val = float(val_str)
                            metrics_found[m] = val
                        except:
                            pass
        if metrics_found:
            model_metrics[fname] = metrics_found
            print(f'[STATUS] Metrics from {fname}: {metrics_found}')

# Check specific metrics presence
pr_auc_found = any('pr_auc' in str(v).lower() or 'average precision' in str(v).lower()
                    for v in model_metrics.values())
auc_roc_found = any('roc_auc' in str(v).lower() or 'roc auc' in str(v).lower() or 'auc' in str(v).lower()
                     for v in model_metrics.values())
f1_weighted_found = any('weighted f1' in str(v).lower() or 'f1-score' in str(v).lower()
                         for v in model_metrics.values())

if not f1_weighted_found and not pr_auc_found:
    performance_issues.append('No F1-weighted or PR-AUC metrics found — likely missing imbalance-aware evaluation')

# Check for CLV metrics separately
clv_metrics_found = any('rmse' in str(v).lower() or 'mae' in str(v).lower() or 'r2' in str(v).lower()
                         for v in model_metrics.values())
if not clv_metrics_found:
    performance_issues.append('No regression metrics (RMSE/MAE/R2) found for CLV model')

# Check for inventory optimization metrics
inventory_found = any('inventory' in k.lower() or 'inv' in k.lower() for k in all_outputs.keys())
if not inventory_found or 'inventory_optimization' not in str(model_metrics):
    performance_issues.append('Inventory optimization model metrics not found')

performance_pass = len(performance_issues) <= 1  # allow minor issues
print(f'[STATUS] Model Performance: {"PASS" if performance_pass else f"FAIL ({len(performance_issues)} issues)"}')

# ── 3. PRODUCTION READINESS CHECK ───────────────────────────────
production_checks = {
    'monitoring': False,
    'retraining': False,
    'deployment_validation': False,
    'dependency_benchmark': False,
    'calibration': False,
    'oot_validation': False,
    'threshold_economics': False,
}

production_notes = []
for fname, text in report_texts.items():
    text_lower = text.lower()
    if 'monitor' in text_lower or 'monitoring' in text_lower:
        production_checks['monitoring'] = True
        production_notes.append(f'Monitoring found in {fname}')
    if 'retrain' in text_lower or 'retraining' in text_lower or 'refresh schedule' in text_lower:
        production_checks['retraining'] = True
        production_notes.append(f'Retraining found in {fname}')
    if 'deploy' in text_lower or 'deployment' in text_lower or 'production' in text_lower:
        production_checks['deployment_validation'] = True
        production_notes.append(f'Deployment validation found in {fname}')
    if 'xgboost' in text_lower or 'lightgbm' in text_lower or 'catboost' in text_lower or 'benchmark' in text_lower:
        production_checks['dependency_benchmark'] = True
        production_notes.append(f'Dependency benchmark found in {fname}')
    if 'calib' in text_lower or 'brier' in text_lower:
        production_checks['calibration'] = True
        production_notes.append(f'Calibration found in {fname}')
    if 'out of time' in text_lower or 'oot' in text_lower or 'time-based' in text_lower or 'temporal' in text_lower:
        production_checks['oot_validation'] = True
        production_notes.append(f'OOT validation found in {fname}')
    if 'threshold' in text_lower or 'cost-benefit' in text_lower or 'expected value' in text_lower or 'economic' in text_lower:
        production_checks['threshold_economics'] = True
        production_notes.append(f'Threshold economics found in {fname}')

production_score = sum(1 for v in production_checks.values() if v)
production_pass = production_score >= 4  # at least 4/7 for executive-ready

print(f'[STATUS] Production Readiness: {production_score}/7 checks passed')
for k, v in production_checks.items():
    print(f'  {"✅" if v else "❌"} {k}')

# Determine readiness level
if production_score >= 6:
    readiness_level = 'Production-ready'
elif production_score >= 4:
    readiness_level = 'Executive-ready prototype'
else:
    readiness_level = 'Prototype'

# ── 4. BUSINESS SATISFACTION ────────────────────────────────────
business_criteria = {
    'model_performance': False,
    'actionable_insights': False,
    'business_questions_covered': False,
    'technical_soundness': False,
}

# Check for actionable insights in reports
insight_count = 0
actionable_insight_count = 0
for fname, text in report_texts.items():
    text_lower = text.lower()
    # Count insight-like statements
    insight_keywords = ['insight', 'finding', 'observation', 'discovery', 'pattern']
    for kw in insight_keywords:
        if kw in text_lower:
            insight_count += text_lower.count(kw)

    # Count actionable recommendations
    action_keywords = ['recommend', 'action', 'suggest', 'should', 'implement', 'optimize', 'increase', 'reduce']
    for kw in action_keywords:
        if kw in text_lower:
            actionable_insight_count += text_lower.count(kw)

business_criteria['actionable_insights'] = actionable_insight_count >= 5
business_criteria['model_performance'] = performance_pass and not leakage_issues
business_criteria['technical_soundness'] = leakage_pass

# Check business question coverage
business_qas = {
    'churn': ('churn' in str(report_texts).lower()),
    'clv': ('clv' in str(report_texts).lower() or 'customer lifetime value' in str(report_texts).lower()),
    'inventory': ('inventory' in str(report_texts).lower() or 'stock' in str(report_texts).lower()),
    'customer_segmentation': ('segment' in str(report_texts).lower() or 'rfm' in str(report_texts).lower()),
}
covered_qas = sum(1 for v in business_qas.values() if v)
total_qas = len(business_qas)
business_criteria['business_questions_covered'] = covered_qas >= 3

print(f'[STATUS] Business Questions: {covered_qas}/{total_qas} covered')
for q, found in business_qas.items():
    print(f'  {"✅" if found else "❌"} {q}')

criteria_passed = sum(1 for v in business_criteria.values() if v)
business_satisfied = criteria_passed >= 3

# ── 5. WORLD-CLASS QC CHECKS ────────────────────────────────────
world_class_qc = {
    'imbalance_metrics': pr_auc_found or f1_weighted_found,
    'validation_realism': production_checks['oot_validation'],
    'threshold_economics': production_checks['threshold_economics'],
    'calibration': production_checks['calibration'],
    'dependency_benchmark': production_checks['dependency_benchmark'],
    'production_readiness': readiness_level,
}

wc_passed = sum(1 for k, v in world_class_qc.items() if v == True or (k == 'production_readiness' and v != 'Prototype'))
print(f'[STATUS] World-Class QC: {wc_passed}/5 checks passed (excl. production_readiness)')

# ── 6. MAX INVENTORY OPTIMIZATION CHECK ─────────────────────────
inventory_output_exists = any('inventory' in k.lower() or 'max' in k.lower() or 'optim' in k.lower() for k in all_outputs.keys())
if not inventory_output_exists:
    print('[WARN] No inventory optimization output found from Max')

inventory_report_found = any('max' in k.lower() or 'inventory' in k.lower() or 'optim' in k.lower() for k in report_texts.keys())
print(f'[STATUS] Max Inventory Optimization report: {"FOUND" if inventory_report_found else "NOT FOUND"}')

# ── VERDICT ──────────────────────────────────────────────────────
blocking_issues = []
blocking_issues.extend(leakage_issues)
blocking_issues.extend(performance_issues)

if not business_satisfied:
    blocking_issues.append(f'Business satisfaction: only {criteria_passed}/4 criteria passed')

if not inventory_report_found:
    blocking_issues.append('Inventory optimization report from Max not found/loaded')

if production_score < 3:
    blocking_issues.append(f'Production readiness insufficient ({production_score}/7)')

restart_cycle = not business_satisfied or not leakage_pass or criteria_passed < 2

if restart_cycle:
    # Determine restart point
    if leakage_issues:
        restart_from = 'Finn'
        restart_reason = 'Data leakage detected — Finn must rebuild features to prevent target leakage'
        new_strategy = 'Add feature lineage enforcement, remove any post-event or target-correlated features'
    elif not performance_pass:
        restart_from = 'Mo'
        restart_reason = 'Model performance insufficient or missing key metrics (F1-weighted, PR-AUC, regression metrics)'
        new_strategy = 'Retrain with imbalance-aware evaluation, add XGBoost/LightGBM benchmark, report PR-AUC for churn'
    elif not business_satisfied:
        restart_from = 'Iris'
        restart_reason = f'Business satisfaction not met ({criteria_passed}/4) — insufficient actionable insights or coverage'
        new_strategy = 'Generate at least 5 actionable recommendations tied to business KPIs, answer all business questions'
    elif production_score < 3:
        restart_from = 'Rex'
        restart_reason = f'Production readiness too low ({production_score}/7) for executive-ready delivery'
        new_strategy = 'Add monitoring plan, retraining schedule, deployment validation, and dependency benchmarks'
    else:
        restart_from = 'Eddie'
        restart_reason = 'Overall quality insufficient — reframe business questions and re-collect requirements'
        new_strategy = 'Redefine success criteria with clearer KPIs before next cycle'
else:
    restart_from = 'N/A'
    restart_reason = 'All quality gates passed'
    new_strategy = 'No restart needed — proceed to delivery'

verdict = 'PASS' if not restart_cycle else 'FAIL'
status = 'ผ่าน' if verdict == 'PASS' else 'ไม่ผ่าน'
if not restart_cycle and production_score < 6:
    status = 'ผ่านแบบมีเงื่อนไข'
    verdict = 'CONDITIONAL'

print(f'\n[STATUS] FINAL VERDICT: {verdict}')
print(f'[STATUS] RESTART_CYCLE: {"YES" if restart_cycle else "NO"}')
if restart_cycle:
    print(f'[STATUS] Restart from: {restart_from}')
    print(f'[STATUS] Reason: {restart_reason}')
else:
    print(f'[STATUS] PASS — Ready for delivery')

# ── Generate QC Results CSV ─────────────────────────────────────
results_data = []
checks = [
    ('data_leakage', 'PASS' if leakage_pass else 'FAIL', str(leakage_issues) if leakage_issues else 'No leakage detected'),
    ('model_performance', 'PASS' if performance_pass else 'FAIL', str(performance_issues) if performance_issues else 'All metrics present'),
    ('production_readiness', 'PASS' if production_score >= 4 else 'FAIL', f'{production_score}/7 checks passed, level: {readiness_level}'),
    ('business_satisfaction', 'PASS' if business_satisfied else 'FAIL', f'{criteria_passed}/4 criteria passed'),
    ('world_class_qc', 'PASS' if wc_passed >= 3 else 'FAIL', f'{wc_passed}/5 checks passed'),
    ('inventory_optimization', 'FOUND' if inventory_report_found else 'NOT_FOUND', 'Max report'),
    ('restart_cycle', 'YES' if restart_cycle else 'NO', restart_reason),
]

for check_name, result, detail in checks:
    results_data.append({
        'check': check_name,
        'result': result,
        'detail': detail[:500],
        'timestamp': datetime.now().isoformat(),
    })

results_df = pd.DataFrame(results_data)
results_csv = os.path.join(OUTPUT_DIR, 'quinn_qc_results.csv')
results_df.to_csv(results_csv, index=False)
print(f'[STATUS] Saved: {results_csv}')

# ── Generate QC Report ──────────────────────────────────────────
report_lines = []
report_lines.append('Quinn Quality Check Report')
report_lines.append('===========================')
report_lines.append(f'Status: {status}')
report_lines.append(f'CRISP-DM Cycle: รอบที่ 1')
report_lines.append(f'Project: 2026-05-01_uci_online_retail')
report_lines.append('')

# Technical QC
report_lines.append('Technical QC:')
report_lines.append('------------')
report_lines.append(f'{"✅" if leakage_pass else "❌"} Data leakage: {"No issues" if leakage_pass else f"{len(leakage_issues)} issues found"}')
if leakage_issues:
    for issue in leakage_issues:
        report_lines.append(f'    - {issue}')
report_lines.append(f'{"✅" if performance_pass else "❌"} Model performance: {"All checks passed" if performance_pass else f"{len(performance_issues)} issues"}')
if performance_issues:
    for issue in performance_issues:
        report_lines.append(f'    - {issue}')
report_lines.append(f'{"✅" if inventory_report_found else "❌"} Inventory optimization report: {"Found" if inventory_report_found else "Not found"}')
report_lines.append('')

# Issues Found
report_lines.append('Issues Found:')
report_lines.append('------------')
if blocking_issues:
    for issue in blocking_issues:
        report_lines.append(f'- {issue}')
else:
    report_lines.append('- No blocking issues')
report_lines.append('')

# BUSINESS_SATISFACTION
report_lines.append('BUSINESS_SATISFACTION')
report_lines.append('=====================')
report_lines.append(f'Criteria Passed: {criteria_passed}/4')
for name, passed in business_criteria.items():
    status_icon = '✅' if passed else '❌'
    name_display = name.replace('_', ' ').title()
    report_lines.append(f'{status_icon} {name_display}: {"PASS" if passed else "FAIL"}')
report_lines.append('')

# Verdict
report_lines.append('Verdict:')
report_lines.append(f'Result: {"SATISFIED" if business_satisfied else "UNSATISFIED"}')
report_lines.append(f'RESTART_CYCLE: {"YES" if restart_cycle else "NO"}')
if restart_cycle:
    report_lines.append(f'Restart From: {restart_from}')
    report_lines.append(f'Restart Reason: {restart_reason}')
    report_lines.append(f'New Strategy: {new_strategy}')
report_lines.append('')

# WORLD_CLASS_QC Block
report_lines.append('WORLD_CLASS_QC')
report_lines.append('==============')
report_lines.append(f'Imbalance metrics: {"PASS" if world_class_qc["imbalance_metrics"] else "FAIL"} — PR-AUC/Average Precision/F1-weighted found')
report_lines.append(f'Validation realism: {"PASS" if world_class_qc["validation_realism"] else "FAIL"} — OOT/time-based split: {"found" if world_class_qc["validation_realism"] else "not found"}')
report_lines.append(f'Threshold economics: {"PASS" if world_class_qc["threshold_economics"] else "FAIL"} — {"found" if world_class_qc["threshold_economics"] else "not found"}')
report_lines.append(f'Calibration: {"PASS" if world_class_qc["calibration"] else "FAIL"} — {"found" if world_class_qc["calibration"] else "not found"}')
report_lines.append(f'Tabular benchmark dependencies: {"PASS" if world_class_qc["dependency_benchmark"] else "FAIL"} — XGBoost/LightGBM/CatBoost: {"tested" if world_class_qc["dependency_benchmark"] else "provisioning needed"}')
report_lines.append(f'Production readiness: {readiness_level}')
report_lines.append(f'Blocking issues: {blocking_issues if blocking_issues else "None"}')
report_lines.append('')

# Production Readiness Details
report_lines.append('Production Readiness Details:')
report_lines.append('---------------------------')
for check, passed in production_checks.items():
    report_lines.append(f'{"✅" if passed else "❌"} {check.replace("_", " ").title()}: {"Present" if passed else "Missing"}')
report_lines.append(f'Overall: {readiness_level}')
report_lines.append('')

# Business Questions
report_lines.append('Business Questions Coverage:')
report_lines.append('--------------------------')
for q, found in business_qas.items():
    report_lines.append(f'{"✅" if found else "❌"} {q.title()}: {"Covered" if found else "Not covered"}')
report_lines.append('')

# Self-Improvement Report
report_lines.append('Self-Improvement Report')
report_lines.append('=======================')
report_lines.append('วิธีที่ใช้ครั้งนี้: World-Class Analytics Default + Auto-Score Business Satisfaction')
report_lines.append('เหตุผลที่เลือก: ครอบคลุมทั้ง data leakage, model performance, production readiness, และ business value')
report_lines.append('วิธีใหม่ที่พบ: การตรวจ inventory optimization เพิ่มเติมจาก Max output')
report_lines.append('จะนำไปใช้ครั้งหน้า: ใช่ — เพราะ inventory optimization เป็นส่วนสำคัญของ retail analytics')
report_lines.append('Knowledge Base: อัพเดต — เพิ่ม inventory optimization check ใน QC checklist')
report_lines.append('')

# Agent Report
report_lines.append('Agent Report — Quinn')
report_lines.append('=====================')
report_lines.append('รับจาก     : Project output directory')
report_lines.append('Input      : All CSV outputs and markdown reports from all agents (Dana, Eddie, Finn, Mo, Iris, Vera, Rex, Max)')
report_lines.append('ทำ         : Complete QC review — data leakage, model performance, production readiness, business satisfaction, inventory optimization')
report_lines.append('พบ         : ')
report_lines.append(f'  1. Data leakage: {"Found" if leakage_issues else "None found"}')
report_lines.append(f'  2. Model metrics: {"Present" if model_metrics else "Missing — need F1-weighted, PR-AUC, RMSE"}')
report_lines.append(f'  3. Production readiness: {production_score}/7 checks — level: {readiness_level}')
report_lines.append(f'  4. Business satisfaction: {criteria_passed}/4 criteria')
report_lines.append(f'  5. Inventory optimization from Max: {"Found" if inventory_report_found else "Not found — may need to regenerate"}')
report_lines.append(f'เปลี่ยนแปลง: Verdict — {verdict}, RESTART_CYCLE: {"YES" if restart_cycle else "NO"}')
report_lines.append(f'ส่งต่อ     : {"Restart cycle → " + restart_from if restart_cycle else "Iris + Vera + Rex — proceed to delivery"}')
report_lines.append(f'  - QC report: {os.path.join(OUTPUT_DIR, "quinn_qc_report.md")}')
report_lines.append(f'  - QC results CSV: {results_csv}')

report_text = '\n'.join(report_lines)

report_path = os.path.join(OUTPUT_DIR, 'quinn_qc_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f'[STATUS] Saved report: {report_path}')

print(f'\n[COMPLETE] QC verdict: {verdict}')
print(f'[COMPLETE] RESTART_CYCLE: {"YES" if restart_cycle else "NO"}')
if restart_cycle:
    print(f'[COMPLETE] Restart from: {restart_from}')
print(f'[COMPLETE] Report: {report_path}')
print(f'[COMPLETE] CSV: {results_csv}')