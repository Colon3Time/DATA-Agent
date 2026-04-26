# Dana Methods — Knowledge Base

## Input Handling

```python
# ห้ามมี glob fallback หรือ path เดาเอง — ถ้าไม่มี --input ให้ exit ทันที
if not INPUT_PATH or not os.path.exists(INPUT_PATH):
    print(f'[ERROR] --input required: {INPUT_PATH}')
    import sys; sys.exit(1)
```

---

## ประเภทของ Missing Data (วิเคราะห์ก่อน impute ทุกครั้ง)

| ประเภท | ลักษณะ | ผลกระทบ | วิธี impute |
|--------|--------|---------|------------|
| **MCAR** (Missing Completely At Random) | หายไปแบบสุ่ม ไม่เกี่ยวกับตัวแปรอื่น | ไม่เกิด bias | Simple impute: Median/Mean |
| **MAR** (Missing At Random) | หายไปเกี่ยวกับตัวแปรอื่นที่มีอยู่ แต่ไม่เกี่ยวกับค่าที่หายไปเอง | จัดการได้ | KNNImputer / MICE (ใช้ตัวแปรอื่นช่วย) |
| **MNAR** (Missing Not At Random) | หายไปเพราะค่าของตัวเองนั้น (เช่น คนรวยไม่รายงานรายได้) | เกิด bias รุนแรง | Flag + รายงาน stakeholder ก่อน impute |

> วิธีตรวจ: ถ้า missing มีรูปแบบชัดเจน (เช่น กลุ่มเฉพาะ) → MAR/MNAR, ถ้ากระจายสุ่ม → MCAR

---

## Missing Values — Method Selection

| สถานการณ์ | วิธี |
|-----------|------|
| MCAR + missing < 5%, ไม่มี correlation | Median / Mode |
| MAR + missing ≥ 5%, มี correlation กับ column อื่น | KNNImputer (n_neighbors=5) |
| MAR + missing > 10%, ข้อมูลซับซ้อน | IterativeImputer (MICE, max_iter=10) |
| time series | forward fill → interpolate |
| MNAR หรือ missing > 60% ของ column | Flag + รายงาน — ห้าม impute โดยไม่แจ้ง |

**ค่า 0 ที่เป็น missing (domain impossible zeros):**
- Glucose=0, BloodPressure=0, BMI=0, SkinThickness=0, Insulin=0 → แปลงเป็น NaN ก่อน impute
- Pregnancies=0 → ค่าจริง ห้ามแปลง

**Post-imputation clip (บังคับ — ป้องกัน KNN ให้ค่าติดลบ):**

ทุก dataset ต้องกำหนด DOMAIN_MIN / DOMAIN_MAX ตาม domain ของข้อมูลนั้น ๆ (ตัวอย่างด้านล่างเป็น Diabetes dataset):

```python
# *** ตัวอย่าง: Diabetes dataset — ต้องเปลี่ยนให้ตรงกับ dataset จริงของ project ***
DOMAIN_MIN = {'Glucose':0,'BloodPressure':0,'SkinThickness':0,'Insulin':0,'BMI':0,'Pregnancies':0,'Age':0}
DOMAIN_MAX = {'Glucose':300,'BloodPressure':200,'SkinThickness':80,'Insulin':500,'BMI':70,'DiabetesPedigreeFunction':2.5}
for col, lo in DOMAIN_MIN.items():
    if col in df.columns: df[col] = df[col].clip(lower=lo)
for col, hi in DOMAIN_MAX.items():
    if col in df.columns: df[col] = df[col].clip(upper=hi)
```

---

## Outlier Detection — Likely Error vs Likely Real

ทุก outlier ต้องตัดสินใจก่อน clean:

| ประเภท | เกณฑ์ | การจัดการ |
|--------|-------|-----------|
| **Likely Error** | เกิน domain max/min, ค่าที่เป็นไปไม่ได้จริง | cap ด้วย median หรือ domain bound |
| **Likely Real** | IQR outlier แต่อยู่ใน domain ที่เป็นไปได้ | เก็บไว้ + บันทึกใน outlier_flags.csv |
| **Uncertain** | Isolation Forest detect แต่ไม่รู้แน่ชัด | flag + บันทึก ให้ agent อื่นตัดสินใจ |

**Domain bounds (Likely Error threshold) — กำหนดตาม domain ของ dataset จริง:**

ตัวอย่าง Diabetes/Medical dataset:
| Column | Max ที่เป็น Likely Error |
|--------|------------------------|
| Insulin | > 500 μU/mL |
| SkinThickness | > 80 mm |
| BloodPressure | > 200 หรือ < 20 mmHg |
| Glucose | > 300 mg/dL |
| BMI | > 70 kg/m² |
| DiabetesPedigreeFunction | > 2.5 |

> สำหรับ dataset อื่น → ต้องหา domain bounds ที่เหมาะสมก่อนเริ่มทำงาน (ค้นหาจาก data dictionary หรือ domain knowledge)

**Outlier detection code:**
```python
from sklearn.ensemble import IsolationForest
feat_cols = [c for c in df.select_dtypes(include='number').columns if c != 'Outcome']
outlier_records = []
for col in feat_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    for idx in df[(df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)].index:
        val = df.loc[idx, col]
        domain_hi = DOMAIN_MAX.get(col, float('inf'))
        if val > domain_hi or val < 0:
            verdict = 'Likely Error'
            df.loc[idx, col] = df[col].median()
        else:
            verdict = 'Likely Real'
        outlier_records.append({'row_index':idx,'column_name':col,'value':val,'verdict':verdict,'reason':f'{col}={val:.2f}'})

iso = IsolationForest(contamination=0.05, random_state=42)
iso_mask = iso.fit_predict(df[feat_cols]) == -1
for idx in df.index[iso_mask]:
    if not any(r['row_index']==idx for r in outlier_records):
        outlier_records.append({'row_index':idx,'column_name':'multivariate','value':None,'verdict':'Uncertain','reason':'Isolation Forest'})

df['is_outlier'] = 0
for r in outlier_records:
    if r['verdict'] != 'Likely Error': df.loc[r['row_index'], 'is_outlier'] = 1

import pandas as pd
flags_df = pd.DataFrame(outlier_records)
flags_df.to_csv(os.path.join(OUTPUT_DIR, 'outlier_flags.csv'), index=False)
```

---

## Data Quality Score — Formula

```python
n = len(df)
missing_after = df.isnull().sum().sum()
error_count = sum(1 for r in outlier_records if r['verdict'] == 'Likely Error')

completeness = (1 - missing_after / (n * len(df.columns))) * 100
validity = (1 - error_count / n) * 100          # นับเฉพาะ Error ที่ยังไม่แก้
overall = 0.5 * completeness + 0.5 * validity   # After ต้องสูงกว่า Before เสมอ
```

> ห้ามนับ Likely Real หรือ Uncertain เป็น invalid — พวกนั้นคือข้อมูลจริงที่เก็บไว้

---

## Output Files (บังคับทุก run)

| ไฟล์ | เนื้อหา |
|------|---------|
| `dana_output.csv` | data สะอาด (768 rows, 10 cols รวม is_outlier) |
| `dana_report.md` | missing, outlier summary, quality score Before→After |
| `outlier_flags.csv` | row_index, column_name, value, verdict, reason |

---

## Common Bugs — ห้ามทำ

```python
# ❌ glob fallback — ทำให้อ่านไฟล์ผิด
csvs = glob.glob(os.path.join(dir, '*.csv'))
INPUT_PATH = csvs[0]  # อาจได้ outlier_flags.csv แทน

# ❌ self-write script
with open(path, 'w') as f:
    f.write(script_content)  # NameError: script_content not defined

# ❌ Unicode ใน print — crash บน Windows cp874
print(f"Done ✓ → {path}")   # ใช้ "->" แทน

# ✅ ถูก
print(f"[STATUS] Done -> {path}")
```
