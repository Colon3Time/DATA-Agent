# Finn Methods & Knowledge Base

## กฎสำคัญ — Finn ต้องผลิต Output File จริง

**Finn ทำงานเสร็จ = มีทั้ง 3 ไฟล์นี้:**
1. `finn_script.py` — script feature engineering ที่รันได้จริง
2. `engineered_data.csv` — dataset ที่มี features ใหม่แล้ว **ต้องมีไฟล์นี้เสมอ**
3. `finn_feature_report.md` — สรุปการทำงาน

❌ **report อย่างเดียวไม่พอ** — ถ้าไม่มี `engineered_data.csv` ถือว่างานยังไม่เสร็จ

---

## Feature Selection — ทำก่อน Feature Engineering เสมอ

Feature Selection คือการเลือกเฉพาะ features ที่มีประโยชน์จริง ก่อนสร้าง features ใหม่

### 3 วิธีหลัก

| วิธี | แนวคิด | ตัวอย่าง | เหมาะกับ |
|------|--------|---------|---------|
| **Filter Methods** | คัดกรองด้วยสถิติก่อนสร้างโมเดล | Correlation, Chi-Square, Mutual Info | Dataset ใหญ่, เร็ว |
| **Wrapper Methods** | ทดลองสร้างโมเดลซ้ำๆ กับ subset ต่างๆ | RFE (Recursive Feature Elimination) | Dataset กลาง, แม่น |
| **Embedded Methods** | Feature selection รวมอยู่ใน training | Lasso (L1), Tree feature importance | ใช้ร่วมกับ model |

```python
# Filter: Correlation
corr = df.corr()['target'].abs().sort_values(ascending=False)
selected_features = corr[corr > 0.1].index.tolist()

# Embedded: Tree feature importance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
selected_features = importances[importances > 0.01].index.tolist()
```

### ประโยชน์ของ Feature Selection
- ลด Overfitting (ลบ noise features)
- เพิ่ม accuracy (โมเดลโฟกัสที่สิ่งสำคัญ)
- ลดเวลา training
- Interpretability ดีขึ้น

---

## Tidy Data Principles (ก่อน Engineering)

1. แต่ละ **variable** = 1 column
2. แต่ละ **observation** = 1 row
3. แต่ละ **value** = 1 cell (ห้ามรวมค่าหลายค่าในเซลล์เดียว)

---

## Feature Engineering Principles

| หลักการ | รายละเอียด |
|---------|-----------|
| Domain-first | สร้าง feature จาก business logic ก่อน เพราะ actionable กว่า statistical feature |
| Low cardinality encoding | ≤ 5 categories → One-Hot, > 5 categories → Target/Label Encoding |
| Scale-aware | Tree-based (RF, XGBoost) ไม่ต้อง scale — Linear/KNN ต้อง StandardScaler หรือ MinMaxScaler |
| Avoid leakage | ห้ามใช้ข้อมูลที่รู้ตอน predict ไม่ได้ (เช่น future dates) มาสร้าง feature |

## Datetime Feature Checklist

สำหรับทุก datetime column ให้สร้าง:
- `_hour`, `_dayofweek`, `_month`, `_quarter`
- `_is_weekend` (boolean)
- `_days_since_X` (ถ้ามี reference date)

## Interaction Feature Template

```python
# Ratio features
df['price_per_weight'] = df['price'] / (df['product_weight_g'] + 1)
df['review_payment_ratio'] = df['review_score'] / (df['payment_value'] + 1)

# Binning
df['payment_tier'] = pd.cut(df['payment_value'],
    bins=[0, 50, 150, 500, float('inf')],
    labels=['low', 'medium', 'high', 'premium'])
```

## Standard Packages

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
```


## เทคนิคที่พิสูจน์แล้วว่าได้ผล

- **Customer activity tiering**: แบ่ง customer tier ตาม order frequency → effective สำหรับ segmentation


## [2026-04-25 19:49] [FEEDBACK]
test3 retail: Feature Engineering succeeded - revenue_per_unit, discount_impact, regional_flag, month/day features. Input from eddie_output.csv via pipeline handoff.


## [2026-04-27 05:17] [DISCOVERY]
การใช้ target encoding อาจเหมาะสมสำหรับ cancer datasets ที่มี high cardinality แต่ dataset นี้เป็น numeric ทั้งหมด เลยไม่จำเป็น


## [2026-04-27 22:49] [DISCOVERY]
Penguins BMI (body mass / flipper length^2) — approximate body density
