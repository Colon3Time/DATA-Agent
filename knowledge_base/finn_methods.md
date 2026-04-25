# Finn Methods & Knowledge Base

## กฎสำคัญ — Finn ต้องผลิต Output File จริง

**Finn ทำงานเสร็จ = มีทั้ง 3 ไฟล์นี้:**
1. `finn_script.py` — script feature engineering ที่รันได้จริง
2. `engineered_data.csv` — dataset ที่มี features ใหม่แล้ว **ต้องมีไฟล์นี้เสมอ**
3. `finn_feature_report.md` — สรุปการทำงาน

❌ **report อย่างเดียวไม่พอ** — ถ้าไม่มี `engineered_data.csv` ถือว่างานยังไม่เสร็จ

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


## [2026-04-25 19:07] [DISCOVERY]
Task: ใช้ eddie_output.csv หรือ max_output.csv (ถ้ามีแล้ว) เป็น input ทำ feature engineering: สร้าง featur
Key finding: I'll start by checking the available input files in the projects/E-Commerce directory.


## [2026-04-25 19:07] [DISCOVERY]
Customer activity tiering based on order frequency is effective for segmentation


## [2026-04-25 19:49] [FEEDBACK]
test3 retail: Feature Engineering succeeded - revenue_per_unit, discount_impact, regional_flag, month/day features. Input from eddie_output.csv via pipeline handoff.
