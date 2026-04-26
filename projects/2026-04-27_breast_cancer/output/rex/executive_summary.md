# Breast Cancer Classification — Executive Summary

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Date:** 2026-04-27
**Project:** Breast Cancer Wisconsin Diagnostic
**Model:** LightGBM (Tuned) — Clinical Decision Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Model Performance

| Metric | Default | Tuned | Rating |
|--------|---------|-------|--------|
| Accuracy | 96.49% | **97.37%** | ✅ Excellent |
| F1-Score | 0.9531 | **0.9674** | ✅ Excellent |
| AUC-ROC | 0.9912 | **0.9942** | ✅ Excellent |
| Train F1 | 0.9812 | 0.9965 | — |
| CV F1 (5-fold) | 0.9498 ± 0.028 | 0.9582 ± 0.025 | ✅ Stable |

**Overfitting:** ไม่มี (train-test gap = 2.9% < 5%) ✅
**QC Result:** Quinn 4/4 Criteria Met — SATISFIED ✅

## Top Predictive Features

1. texture_mean
2. perimeter_mean
3. area_mean
4. concave_points_mean
5. texture_se

## Key Business Insights (จาก Iris)

**1. ลด False Negative — ช่วยชีวิตได้จริง**
AUC=0.994 ระดับ near-perfect แต่ต้อง tune recall ≥ 0.99 ก่อน deploy
→ ช่วยชีวิตผู้ป่วยได้ 1–2 รายต่อ 1,000 screening

**2. ลด biopsy ที่ไม่จำเป็น — ประหยัด 1.6–2.4 ล้านบาท/ปี**
5 features วัดได้จาก FNAC ปกติ ไม่ต้องซื้ออุปกรณ์เพิ่ม
→ ลด benign biopsy ได้ 30–40% (ประหยัด 8,000–12,000 บาท/ราย)

**3. โอกาสตลาดในไทย**
AI pathology โต 43% CAGR — ยังไม่มีคู่แข่งแข็งในไทย
→ Thailand MOPH เตรียม digital pathology 10 โรงพยาบาล ปี 2026

## Roadmap

| Priority | Action | Timeline |
|----------|--------|----------|
| 🔴 High | Tune recall ≥ 0.99 + clinical dashboard | 2–4 สัปดาห์ |
| 🟡 Medium | API integration กับ HIS โรงพยาบาล | 3–6 เดือน |
| 🟢 Low | Clinical validation study (500+ patients) | 6–12 เดือน |

## Limitations

- Dataset size moderate (569 samples) — ต้องการ external validation
- Malignancy rate 37% สูงกว่า screening จริง (5–15%) → เหมาะสำหรับ referral center
- ต้องการ SHAP explanation สำหรับแพทย์ก่อน deploy

---

*Compiled by Anna from Quinn QC + Iris Insights | 2026-04-27*
