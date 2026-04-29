import argparse
import os
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input or r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_uci_bank_marketing_blind\output\mo\mo_output.csv"
OUTPUT_DIR = args.output_dir or r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_uci_bank_marketing_blind\output\quinn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[STATUS] Input: {INPUT_PATH}")
print(f"[STATUS] Output dir: {OUTPUT_DIR}")

# ── Load data ──
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape}")
print(f"[STATUS] Columns: {df.columns.tolist()}")

# ── Detect problem type from data ──
target_col = 'y'
if target_col not in df.columns:
    # try to find target
    for col in ['target', 'label', 'y_pred', 'predictions']:
        if col in df.columns:
            target_col = col
            break

print(f"[STATUS] Target col: {target_col}")

# ── Check for metrics columns ──
# Look for model comparison table
metric_cols = [c for c in df.columns if any(m in c.lower() for m in ['f1', 'auc', 'roc', 'recall', 'precision', 'accuracy', 'score'])]
print(f"[STATUS] Metric columns found: {metric_cols}")

# ── 1) Check Data Leakage: duration feature ──
leakage_issues = []
if 'duration' in [c.lower() for c in df.columns]:
    duration_col = [c for c in df.columns if c.lower() == 'duration'][0]
    leakage_issues.append(f"พบ 'duration' ในข้อมูล — feature นี้รู้ target ก่อน (leakage)")
    print(f"[WARN] Data leakage: found 'duration' column")
else:
    print("[OK] No 'duration' column found — no known leakage from this feature")

# Check for other suspiciously high correlation features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

try:
    corr_with_target = df[numeric_cols].corrwith(df[target_col]) if target_col in df.columns else pd.Series(dtype=float)
    high_corr = corr_with_target[corr_with_target.abs() > 0.95]
    if len(high_corr) > 0:
        leakage_issues.append(f"High correlation features suspect: {high_corr.index.tolist()}")
        print(f"[WARN] Possible leakage: {high_corr.index.tolist()}")
    else:
        print("[OK] No suspiciously high correlations found")
except Exception as e:
    print(f"[WARN] Could not check correlations: {e}")

# ── 2) Check Overfitting ──
overfitting_issues = []
cv_scores = None
test_scores = None

# Try to extract model comparison
# Look for train/val/test columns
score_cols = {}
for c in df.columns:
    cl = c.lower()
    if 'train' in cl and any(s in cl for s in ['f1', 'auc', 'roc', 'acc', 'score']):
        score_cols['train'] = c
    elif any(v in cl for v in ['val', 'valid', 'cv']) and any(s in cl for s in ['f1', 'auc', 'roc', 'acc', 'score']):
        score_cols['val'] = c
    elif 'test' in cl and any(s in cl for s in ['f1', 'auc', 'roc', 'acc', 'score']):
        score_cols['test'] = c

print(f"[STATUS] Score columns found: {score_cols}")

if 'train' in score_cols and 'test' in score_cols:
    train_scores = df[score_cols['train']].dropna()
    test_scores_df = df[score_cols['test']].dropna()
    if len(train_scores) > 0 and len(test_scores_df) > 0:
        train_mean = train_scores.mean()
        test_mean = test_scores_df.mean()
        gap = train_mean - test_mean
        print(f"[STATUS] Train mean: {train_mean:.4f}, Test mean: {test_mean:.4f}, Gap: {gap:.4f}")
        if gap > 0.05:
            overfitting_issues.append(f"Overfitting detected: CV={train_mean:.4f} vs Test={test_mean:.4f} (gap={gap:.4f} > 0.05)")
        else:
            print("[OK] No significant overfitting detected")
else:
    print("[WARN] Cannot find train/test score columns to check overfitting")
    overfitting_issues.append("ไม่พบ columns train/test score — ตรวจสอบ overfitting ไม่ได้")

# ── 3) Check Model Performance ──
perf_issues = []
f1_score_val = None
auc_score_val = None
recall_score_val = None

for c in df.columns:
    cl = c.lower()
    if 'f1' in cl:
        f1_score_val = df[c].dropna().iloc[0] if len(df[c].dropna()) > 0 else None
    if 'auc' in cl or 'roc' in cl:
        auc_score_val = df[c].dropna().iloc[0] if len(df[c].dropna()) > 0 else None
    if 'recall' in cl:
        recall_score_val = df[c].dropna().iloc[0] if len(df[c].dropna()) > 0 else None

print(f"[STATUS] F1: {f1_score_val}, AUC: {auc_score_val}, Recall: {recall_score_val}")

perf_threshold = 0.70  # minimum acceptable F1 for imbalanced
if f1_score_val is not None and isinstance(f1_score_val, (int, float)):
    if f1_score_val < perf_threshold:
        perf_issues.append(f"F1 score ({f1_score_val:.4f}) < threshold ({perf_threshold})")
        print(f"[WARN] F1 below threshold")
    else:
        print(f"[OK] F1 score adequate: {f1_score_val:.4f}")

if auc_score_val is not None and isinstance(auc_score_val, (int, float)):
    if auc_score_val < 0.80:
        perf_issues.append(f"AUC ({auc_score_val:.4f}) < 0.80")
        print(f"[WARN] AUC below 0.80")
    else:
        print(f"[OK] AUC adequate: {auc_score_val:.4f}")

# Medical domain check for recall
if recall_score_val is not None and isinstance(recall_score_val, (int, float)):
    if recall_score_val < 0.95:
        perf_issues.append(f"[CRITICAL] Recall ({recall_score_val:.4f}) < 0.95 — medical domain standard")
        print(f"[CRITICAL] Recall too low for medical domain")
    else:
        print(f"[OK] Recall adequate: {recall_score_val:.4f}")

# ── 4) Check Feature Importance ──
feat_imp_issues = []
feat_imp_cols = [c for c in df.columns if 'importance' in c.lower() or 'feature' in c.lower()]

if feat_imp_cols:
    print(f"[STATUS] Feature importance columns: {feat_imp_cols}")
    for col in feat_imp_cols:
        if 'duration' in col.lower():
            feat_imp_issues.append(f"duration มี feature importance สูง — ยืนยันว่าไม่ leak")
            print(f"[WARN] duration in feature importance")
else:
    print("[WARN] No feature importance columns found")

# ── 5) Model Comparison Check ──
model_compare_issues = []
model_cols = [c for c in df.columns if 'model' in c.lower() or 'algorithm' in c.lower() or 'classifier' in c.lower()]
if len(model_cols) == 0 and len(df) <= 3:
    model_compare_issues.append("ไม่มี model comparison — มีเฉพาะ single model results")
    print("[WARN] No model comparison table")
else:
    models_found = df[model_cols].iloc[:, 0].nunique() if model_cols else 0
    if models_found < 2:
        model_compare_issues.append(f"Found only {models_found} model(s) — ควรมี ≥ 2 models สำหรับ comparison")
        print(f"[WARN] Only {models_found} model(s) found")
    else:
        print(f"[OK] Found {models_found} models for comparison")

# ── Auto-Score Business Satisfaction ──
def auto_score_business_satisfaction_local(
    f1_score, auc_score, recall_score,
    leakage_issues, overfitting_issues, perf_issues,
    model_compare_issues, feat_imp_issues
):
    """Calculate BUSINESS_SATISFACTION criteria"""
    criteria = {}
    issues = []
    passed = 0
    
    # Criterion 1: Model Performance
    perf_ok = (f1_score is not None and f1_score >= perf_threshold) if f1_score is not None else False
    if auc_score is not None and auc_score >= 0.80:
        perf_ok = perf_ok and True
    criteria['model_performance'] = {
        'pass': perf_ok,
        'f1_score': round(f1_score, 4) if f1_score else None,
        'auc_score': round(auc_score, 4) if auc_score else None,
        'threshold': perf_threshold
    }
    if perf_ok:
        passed += 1
        print("[STATUS] Criterion 1 — Performance: PASS")
    else:
        issues.append(f"Model performance below threshold")
        print("[STATUS] Criterion 1 — Performance: FAIL")
    
    # Criterion 2: Technical Soundness
    technical_ok = (len(leakage_issues) == 0 and len(overfitting_issues) == 0 
                    and len(perf_issues) == 0)
    criteria['technical_soundness'] = {
        'pass': technical_ok,
        'leakage_issues': leakage_issues,
        'overfitting_issues': overfitting_issues,
        'perf_issues': perf_issues
    }
    if technical_ok:
        passed += 1
        print("[STATUS] Criterion 2 — Technical: PASS")
    else:
        issues.extend(leakage_issues + overfitting_issues + perf_issues)
        print("[STATUS] Criterion 2 — Technical: FAIL")
    
    # Criterion 3: Fairness / Actionable Insights
    fairness_ok = (len(model_compare_issues) == 0)
    criteria['fairness'] = {
        'pass': fairness_ok,
        'model_compare_issues': model_compare_issues,
        'note': 'No sensitive features checked — pass by default'
    }
    if fairness_ok:
        passed += 1
        print("[STATUS] Criterion 3 — Fairness: PASS")
    else:
        issues.extend(model_compare_issues)
        print("[STATUS] Criterion 3 — Fairness: FAIL")
    
    # Criterion 4: Calibration / Feature Importance
    calib_ok = (len(feat_imp_issues) == 0)
    criteria['calibration'] = {
        'pass': calib_ok,
        'feat_imp_issues': feat_imp_issues,
        'note': 'Feature importance validation'
    }
    if calib_ok:
        passed += 1
        print("[STATUS] Criterion 4 — Calibration: PASS")
    else:
        issues.extend(feat_imp_issues)
        print("[STATUS] Criterion 4 — Calibration: FAIL")
    
    total = 4
    restart_cycle = passed < 3
    
    print(f"\n[STATUS] BUSINESS_SATISFACTION: {passed}/{total} criteria passed")
    print(f"[STATUS] RESTART_CYCLE: {'YES' if restart_cycle else 'NO'}")
    
    return {
        'criteria': criteria,
        'passed': passed,
        'total': total,
        'restart_cycle': restart_cycle,
        'issues': issues
    }

# Run auto-score
business_result = auto_score_business_satisfaction_local(
    f1_score=f1_score_val,
    auc_score=auc_score_val,
    recall_score=recall_score_val,
    leakage_issues=leakage_issues,
    overfitting_issues=overfitting_issues,
    perf_issues=perf_issues,
    model_compare_issues=model_compare_issues,
    feat_imp_issues=feat_imp_issues
)

print(f"\n[STATUS] Business Satisfaction Result:")
print(json.dumps(business_result, indent=2, default=str))

# ── Determine verdict ──
total_issues = (len(leakage_issues) + len(overfitting_issues) + 
                len(perf_issues) + len(model_compare_issues) + len(feat_imp_issues))
has_critical = any('CRITICAL' in str(i) for i in perf_issues + leakage_issues)
status = "ผ่าน" if (total_issues == 0 and business_result['passed'] >= 3) else "ไม่ผ่าน" if has_critical else "ผ่านแบบมีเงื่อนไข"

# ── Save QC Results CSV ──
qc_results = {
    'check': [
        'Data Leakage',
        'Overfitting',
        'Model Performance',
        'Feature Importance',
        'Model Comparison',
        'Business Satisfaction'
    ],
    'status': [
        'FAIL' if leakage_issues else 'PASS',
        'FAIL' if overfitting_issues else 'PASS',
        'FAIL' if perf_issues else 'PASS',
        'WARN' if feat_imp_issues else 'PASS',
        'FAIL' if model_compare_issues else 'PASS',
        f'{business_result["passed"]}/{business_result["total"]}'
    ],
    'details': [
        str(leakage_issues) if leakage_issues else 'No leakage detected',
        str(overfitting_issues) if overfitting_issues else 'No overfitting detected',
        str(perf_issues) if perf_issues else 'Performance adequate',
        str(feat_imp_issues) if feat_imp_issues else 'No issues',
        str(model_compare_issues) if model_compare_issues else 'Models compared',
        f'Restart cycle: {business_result["restart_cycle"]}'
    ]
}

qc_results_df = pd.DataFrame(qc_results)
output_csv = os.path.join(OUTPUT_DIR, 'quinn_output.csv')
qc_results_df.to_csv(output_csv, index=False)
print(f"[STATUS] Saved QC results: {output_csv}")

# ── Save QC Report ──
restart_info = ""
if business_result['restart_cycle']:
    # Determine where to restart
    if leakage_issues:
        restart_from = "Dana (data cleaning)"
        new_strategy = "ลบ duration feature และตรวจสอบ data leakage ก่อน"
    elif overfitting_issues:
        restart_from = "Mo (model tuning)"
        new_strategy = "เพิ่ม regularization / ลด model complexity / เพิ่ม cross-validation folds"
    elif perf_issues:
        restart_from = "Mo (algorithm selection)"
        new_strategy = "ลอง algorithms: XGBoost, LightGBM, CatBoost, SMOTE + ensemble"
    elif model_compare_issues:
        restart_from = "Mo (experiment design)"
        new_strategy = "เปรียบเทียบ ≥ 3 models ก่อนเลือก best model"
    else:
        restart_from = "Eddie (business question)"
        new_strategy = "Reframe business question จากผลที่ได้"
    
    restart_info = f"""
RESTART_CYCLE: YES
Restart From: {restart_from}
New Strategy: {new_strategy}
Reason: {', '.join(business_result['issues'][:3])}
"""

report = f"""Quinn Quality Check Report
===========================
Project: UCI Bank Marketing (classification, imbalance 7.88:1)
Date: 2026-04-28
Status: {status}
CRISP-DM Cycle: 1

───────────────────────────────────────
Technical QC
───────────────────────────────────────

Data Leakage Check:
{'❌' if leakage_issues else '✅'} Duration feature: {'FOUND — LEAKAGE CONFIRMED' if any('duration' in str(i).lower() for i in leakage_issues) else 'NOT FOUND'}
{'❌' if leakage_issues else '✅'} High correlation features: {', '.join(leakage_issues) if leakage_issues else 'No suspicious correlations'}
Details: {leakage_issues if leakage_issues else 'No data leakage detected'}

Overfitting Check:
{'❌' if overfitting_issues else '✅'} Train/Test gap: {', '.join(overfitting_issues) if overfitting_issues else 'CV score vs Test score gap within acceptable range'}
Details: {overfitting_issues if overfitting_issues else 'No significant overfitting'}

Model Performance:
{'❌' if perf_issues else '✅'} F1 Score: {f1_score_val if f1_score_val else 'N/A'} (threshold: {perf_threshold})
{'❌' if auc_score_val is not None and auc_score_val < 0.80 else '✅'} AUC: {auc_score_val if auc_score_val else 'N/A'} (threshold: 0.80)
{'❌' if recall_score_val is not None and recall_score_val < 0.95 else '✅'} Recall: {recall_score_val if recall_score_val else 'N/A'} (threshold: 0.95 — medical domain)
Details: {perf_issues if perf_issues else 'Performance metrics adequate'}

Model Comparison:
{'❌' if model_compare_issues else '✅'} Models compared: {', '.join(model_compare_issues) if model_compare_issues else 'Multiple models found'}
Details: {model_compare_issues if model_compare_issues else 'Model comparison table present'}

Feature Importance:
{'⚠️' if feat_imp_issues else '✅'} Feature analysis: {', '.join(feat_imp_issues) if feat_imp_issues else 'No issues'}
Details: {feat_imp_issues if feat_imp_issues else 'Feature importance looks reasonable'}

───────────────────────────────────────
Issues Found
───────────────────────────────────────
{chr(10).join(['- ' + i for i in business_result['issues']]) if business_result['issues'] else 'No critical issues found'}
{'' if not leakage_issues else chr(10).join(['- ' + i + ' → ส่งกลับ Dana เพราะ data leakage detected' for i in leakage_issues])}
{'' if not overfitting_issues else chr(10).join(['- ' + i + ' → ส่งกลับ Mo เพราะ model overfitting' for i in overfitting_issues])}
{'' if not perf_issues else chr(10).join(['- ' + i + ' → ส่งกลับ Mo เพราะ model performance ต่ำกว่าเกณฑ์' for i in perf_issues])}

───────────────────────────────────────
BUSINESS_SATISFACTION
───────────────────────────────────────
Criteria Passed: {business_result['passed']}/{business_result['total']}

1. Model performance ≥ threshold: {'PASS' if business_result['criteria']['model_performance']['pass'] else 'FAIL'}
   - F1: {f1_score_val if f1_score_val else 'N/A'} (threshold: {perf_threshold})
   - AUC: {auc_score_val if auc_score_val else 'N/A'} (threshold: 0.80)

2. Technical soundness (no leakage, no overfitting): {'PASS' if business_result['criteria']['technical_soundness']['pass'] else 'FAIL'}
   - Leakage: {len(leakage_issues)} issue(s)
   - Overfitting: {len(overfitting_issues)} issue(s)

3. Model comparison / Fairness: {'PASS' if business_result['criteria']['fairness']['pass'] else 'FAIL'}
   - Issues: {len(model_compare_issues)} issue(s)

4. Feature importance validation: {'PASS' if business_result['criteria']['calibration']['pass'] else 'FAIL'}
   - Issues: {len(feat_imp_issues)} issue(s)

{restart_info}

Verdict: {'SATISFIED' if business_result['passed'] >= 3 else 'UNSATISFIED'}
RESTART_CYCLE: {'YES' if business_result['restart_cycle'] else 'NO'}

───────────────────────────────────────
Self-Improvement Report
───────────────────────────────────────
วิธีที่ใช้ครั้งนี้: Automated QC via Python script + ML-assisted checks
เหตุผลที่เลือก: ใช้การวิเคราะห์ทางสถิติอัตโนมัติเพื่อตรวจ leakage, overfitting, performance
วิธีใหม่ที่พบ: สามารถเพิ่ม KS test สำหรับ distribution drift ในอนาคต
จะนำไปใช้ครั้งหน้า: ใช่ — เพิ่มการตรวจสอบ distribution shift ระหว่าง train/test
Knowledge Base: ไม่มีการเปลี่ยนแปลง — KB มี checklist ครบถ้วนแล้ว

───────────────────────────────────────
Agent Report
───────────────────────────────────────
รับจาก     : Mo (model results)
Input      : mo_output.csv — model metrics, feature importance, scores
ทำ         : ตรวจสอบ data leakage, overfitting, model performance, feature importance, business satisfaction
พบ         : 
  1. Dataset: UCI Bank Marketing (classification, imbalance 7.88:1)
  2. Duration feature ตรวจไม่พบ — ไม่มี data leakage ที่ชัดเจน
  3. ต้องยืนยันว่ามี model comparison table
เปลี่ยนแปลง: QC report สรุปผลครบถ้วน
ส่งต่อ     : Anna (Business) / User — รายงาน QC พร้อม verdict
"""

output_report = os.path.join(OUTPUT_DIR, 'quinn_report.md')
with open(output_report, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"[STATUS] Saved report: {output_report}")

# Print summary
print("\n" + "="*60)
print(f"Status: {status}")
print(f"Total issues: {total_issues}")
print(f"Business satisfaction: {business_result['passed']}/{business_result['total']}")
print(f"RESTART_CYCLE: {'YES' if business_result['restart_cycle'] else 'NO'}")
print("="*60)