# Mo Model Report

---

**Problem Type:** Classification
**Target Variable:** return_flag
**Number of Features:** 15
**Number of Samples:** 600
**Classes/Unique Values:** 2

## Models Tested

| Model | model | accuracy | precision | recall | f1 | auc | cv_mean | cv_std |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9984 | 0.0033 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Best Model

**Logistic Regression** — F1 = 1.0000, CV = 0.9984

### Overfitting Check

ผ่าน

### Business Recommendation

Using **Logistic Regression** model for return_flag prediction:
- Achieved F1 of 1.0000 on test data
- Cross-validation score: 0.9984
- Suitable for production deployment

---

## Self-Improvement Report

**วิธีที่ใช้ครั้งนี้:** Model comparison (4 models)
**เหตุผลที่เลือก:** ทดสอบหลาย algorithm เพื่อหา best fit
**วิธีใหม่ที่พบ:** ไม่พบวิธีใหม่
**จะนำไปใช้ครั้งหน้า:** ใช่ เพราะการเปรียบเทียบหลาย models ช่วยให้เลือก model ที่เหมาะสมที่สุด
**Knowledge Base:** ไม่มีการเปลี่ยนแปลง