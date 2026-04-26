# Quinn Methods & Knowledge Base

## กฎสำคัญ — Quinn ต้องผลิต Output File จริง

**Quinn ทำงานเสร็จ = มีทั้ง 2 ไฟล์นี้:**
1. `quinn_qc_report.md` — QC results ครบทุก check พร้อม BUSINESS_SATISFACTION block
2. `quinn_qc_results.csv` — checklist แบบ machine-readable

❌ **ถ้าไม่มี verdict ชัดเจน (Pass/Fail) และ RESTART_CYCLE decision ถือว่างานยังไม่เสร็จ**

---

## QC Checklist มาตรฐาน

### Data Integrity (ตรวจ Dana + Finn output)
- [ ] Row count ก่อน/หลัง pipeline ต่างกัน < 5% (ถ้าเกินต้องอธิบาย)
- [ ] Column names ถูกต้อง ไม่มี typo ไม่มี unnamed columns
- [ ] Data types ถูกต้อง (numeric, datetime, categorical)
- [ ] Missing values ใน key columns ≤ ที่กำหนดใน task
- [ ] ไม่มี duplicate rows
- [ ] Output CSV อ่านได้จริง (ไม่ corrupt, encoding ถูกต้อง)

### Model Quality (ตรวจ Mo output)
- [ ] มี model comparison table (ห้าม single model รายงาน)
- [ ] Overfitting check: train vs validation score ต่างกัน < 5%
- [ ] Cross-validation std < 0.05
- [ ] Evaluation metric ตรงกับ problem type
  - Classification imbalanced → F1-weighted / AUC-ROC
  - Classification balanced → Accuracy + F1
  - Regression → RMSE + MAE
- [ ] Feature importance มีความสมเหตุสมผล (ไม่มี data leakage)

### Business Readiness (ตรวจ Iris + Rex output)
- [ ] Report ตอบ "So what?" ได้สำหรับ stakeholder
- [ ] Recommendation มี action item ที่ทำได้จริง (ไม่ใช่แค่ observation)
- [ ] Model/insight limitations ระบุชัดเจน
- [ ] ตัวเลขสำคัญอยู่ใน executive summary

---

## BUSINESS_SATISFACTION Block (บังคับใส่ในทุก report)

```
BUSINESS_SATISFACTION
=====================
Criteria Met: X/4
- [ ] Business question ตอบได้ชัดเจน
- [ ] Model/insight actionable (มี next step)
- [ ] Data quality ผ่านเกณฑ์ (Integrity checks ≥ 5/6)
- [ ] No critical issues (no data leakage, no major overfitting)

RESTART_CYCLE: YES / NO
Restart From: [agent name — ถ้า YES]
New Strategy: [อธิบาย strategy ใหม่ — ถ้า YES]
Reason: [สาเหตุที่ต้อง restart — ถ้า YES]
```

> ถ้า Criteria < 2/4 → RESTART_CYCLE: YES — Anna จะถาม user ก่อน restart

---

## QC Scope ต่อ Agent

| Agent | ไฟล์ที่ต้องตรวจ | สิ่งที่ตรวจ |
|-------|----------------|-----------|
| Dana | `dana_output.csv`, `outlier_flags.csv` | shape, missing, outlier verdict |
| Eddie | `eddie_report.md` | insight quality, section completeness |
| Finn | `engineered_data.csv`, `finn_feature_report.md` | leakage check, scaling applied |
| Mo | `model_results.md`, `mo_script.py` | model comparison, tuning, overfitting |
| Iris | `insights.md`, `recommendations.md` | actionability, business impact |
| Rex | `final_report.md`, `executive_summary.md` | completeness, audience-appropriate |

---

## Common Issues ที่ต้อง flag

| ปัญหา | วิธีตรวจ | Severity |
|-------|---------|---------|
| Row count หาย > 5% | compare input/output shape | HIGH |
| Overfitting (train-val gap > 5%) | compare train/cv scores | HIGH |
| ไม่มี model comparison | ตรวจ model_results.md | HIGH |
| Data leakage suspect | feature importance ผิดปกติสูงมาก | HIGH |
| Missing insight section | ตรวจ section completeness | MEDIUM |
| No actionable recommendation | ตรวจ Iris/Rex output | MEDIUM |
| Encoding issues | อ่านไฟล์จริง ตรวจ character corruption | LOW |

---

---

## Data Drift Detection — ตรวจ Distribution Shift

ใช้เมื่อ: QC model ก่อน deploy หรือ monitor model ที่ใช้งานอยู่แล้ว

```python
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

def check_drift(train_col, test_col, col_name, threshold=0.05):
    """ตรวจ drift ระหว่าง train และ test/production data"""
    if train_col.dtype in ["float64", "int64"]:
        # Kolmogorov-Smirnov test สำหรับ numerical
        stat, p = ks_2samp(train_col.dropna(), test_col.dropna())
        drifted = p < threshold
    else:
        # Chi-square สำหรับ categorical
        cats = set(train_col.unique()) | set(test_col.unique())
        t = [train_col.value_counts().get(c, 0) for c in cats]
        s = [test_col.value_counts().get(c, 0)  for c in cats]
        _, p, _, _ = chi2_contingency([t, s])
        drifted = p < threshold
    print(f"{'DRIFT' if drifted else 'OK':6s} {col_name:30s} p={p:.4f}")
    return drifted

# ใช้กับทุก feature
drifted_features = []
for col in X_train.columns:
    if check_drift(X_train[col], X_test[col], col):
        drifted_features.append(col)

print(f"\nDrifted features ({len(drifted_features)}): {drifted_features}")
```

**Population Stability Index (PSI) — ตัวชี้วัด drift มาตรฐาน industry:**
```python
def psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 100, buckets + 1)
    e_pct = np.histogram(expected, np.percentile(expected, breakpoints))[0] / len(expected)
    a_pct = np.histogram(actual,   np.percentile(expected, breakpoints))[0] / len(actual)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))

# PSI < 0.1  → ไม่มี drift
# PSI 0.1-0.2 → drift เล็กน้อย ควร monitor
# PSI > 0.2  → drift มาก — ต้อง retrain model
```

**Quinn ต้องรายงาน drift ใน QC report เสมอถ้ามี train/test split**

## [2026-04-25 19:49] [FEEDBACK]
QC passed 10/11 checks — validated CSV shape, missing values, data types for all agent outputs. Saved quinn_qc_results.csv.

## [2026-04-27] [DOMAIN RULE] Medical Dataset — ต้องตรวจ FN Rate ก่อน Accuracy

**บทเรียนจาก Breast Cancer project:**

Medical/Healthcare classification มี cost asymmetry:
- False Negative (FN) = พลาดผู้ป่วยมะเร็ง → อันตรายถึงชีวิต
- False Positive (FP) = biopsy ที่ไม่จำเป็น → เสียเงิน/เวลา

**กฎ Quinn สำหรับ medical domain:**
1. ตรวจ Confusion Matrix บังคับ (ไม่ใช่แค่ Accuracy)
2. Recall ≥ 0.97 สำหรับ cancer detection — ถ้าต่ำกว่า → RESTART_CYCLE: YES
3. รายงาน FN count จริง: "X patients would be missed per 1,000 screenings"
4. ถ้า problem_type มีคำว่า cancer / fraud / default / churn → ใช้กฎนี้อัตโนมัติ

**Severity escalation:**
- Recall < 0.95 → CRITICAL — RESTART_CYCLE: YES
- Recall 0.95–0.97 → WARNING — แนะนำ threshold tuning
- Recall ≥ 0.97 → PASS
