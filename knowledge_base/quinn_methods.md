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

## [2026-04-25 19:49] [FEEDBACK]
QC passed 10/11 checks — validated CSV shape, missing values, data types for all agent outputs. Saved quinn_qc_results.csv.
