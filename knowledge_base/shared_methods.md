# Shared Methods — ทุก Agent อ่านไฟล์นี้

## Python Environment (Windows — DATA-Agent)

**ใช้ Python 3.12 เสมอ — ห้ามใช้ Python 3.14**
- numpy/pandas ยังไม่รองรับ Python 3.14 (C-extension crash)
- `.venv` อยู่ที่ `C:\Users\Amorntep\DATA-Agent\.venv`

**รัน script ผ่าน orchestrator:** orchestrator จัดการ encoding และ venv ให้อัตโนมัติ

**รัน script ด้วยมือ (นอก orchestrator):**
```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\Amorntep\DATA-Agent\.venv\Scripts\python.exe <script_path>
```

**ติดตั้ง package เพิ่ม:**
```powershell
uv pip install --python C:\Users\Amorntep\DATA-Agent\.venv\Scripts\python.exe <package>
```

**Packages มาตรฐานที่ติดตั้งแล้ว:** pandas, numpy, matplotlib, seaborn, scipy


## Encoding — Windows Thai Locale

- Windows Thai ใช้ cp874 → emoji และ unicode บางตัวใช้ไม่ได้ถ้าไม่ตั้ง encoding
- orchestrator ตั้ง `PYTHONIOENCODING=utf-8` ให้ทุก subprocess อัตโนมัติ
- ถ้าเจอ `UnicodeDecodeError` หรือ `charmap codec` → ต้องตั้ง env var ข้างบน


## Universal ML Rules (ทุก Agent ต้องทำตาม)

กฎเหล่านี้บังคับทุก agent ที่ใช้ ML — ห้ามละเมิดเด็ดขาด:

### 1. Feature Scaling ก่อนใช้ Distance-Based Methods
ทุก algorithm ที่ใช้ distance calculation ต้อง scale features ก่อนเสมอ:
- KNN Imputation → StandardScaler หรือ MinMaxScaler ก่อน impute
- KNN Classifier/Regressor → scale ก่อน fit
- K-Means, DBSCAN → scale ก่อน cluster
- SVM → StandardScaler ก่อน fit
- PCA → StandardScaler ก่อน transform

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_scaled)
# inverse transform ถ้าต้องการ scale เดิม
X_result = scaler.inverse_transform(X_imputed)
```

### 2. เมื่อไหร่ต้องใช้ KNN Imputation vs Median
| สถานการณ์ | วิธีที่ต้องใช้ |
|-----------|--------------|
| missing < 5% และ columns ไม่ correlate กัน | Median/Mode impute |
| missing >= 5% หรือ columns correlate กัน | KNN Imputation (+ scale ก่อน) |
| categorical missing | fill 'unknown' หรือ mode |

### 3. Validate ก่อน-หลัง ML ทุกครั้ง
- ตรวจ distribution ก่อนและหลัง impute/transform
- report ผลการ validate ใน output report เสมอ

## Decision Quality Gate — บังคับก่อนตัดสินใจทุกครั้ง

Agent ยังตัดสินใจผิดได้ง่าย ถ้าข้ามการพิสูจน์หลักฐาน ดังนั้นก่อนเลือกวิธี, ลบข้อมูล, เติมค่า, สร้าง feature, เลือก model, เลือก chart, สรุป insight, pass/fail QC, หรือเขียน recommendation ต้องผ่าน gate นี้ก่อนเสมอ

### กฎ 5 ข้อก่อนตัดสินใจ
1. **ระบุโจทย์ให้ชัด** — decision นี้ตอบ business/research question อะไร และ downstream agent จะใช้ผลนี้อย่างไร
2. **ตรวจหลักฐานจากไฟล์จริง** — ใช้ข้อมูลจริง, report ล่าสุด, log ล่าสุด, schema, row count, metric หรือ artifact ที่มีอยู่ ห้ามเดาจากชื่อไฟล์หรือความจำ
3. **เทียบอย่างน้อย 2 ทางเลือก** — ระบุทางเลือกหลัก, ทางเลือกสำรอง, ข้อดี/ข้อเสีย และเหตุผลที่ไม่เลือกอีกทาง ถ้ามีทางเดียวต้องอธิบายว่าทำไมไม่มีทางเลือกอื่นที่เหมาะ
4. **ประเมินความเสี่ยงก่อนลงมือ** — leakage, overfitting, missingness, imbalance, wrong target, wrong file, stale output, path ผิด, dependency หาย, cost/business impact, และผลกระทบต่อ agent ถัดไป
5. **สรุป verdict แบบตรวจสอบได้** — เขียน decision พร้อม evidence, assumption, confidence และสิ่งที่จะทำถ้า evidence ไม่พอ

### Stop Rules — ห้ามเดา ห้ามเดินต่อ
ถ้าเข้าเงื่อนไขใดข้อหนึ่ง ต้องหยุดและเขียน repair/ask แทนการเลือกเอง:
- ไม่พบไฟล์ input/output ที่ต้องใช้ หรือไฟล์ล่าสุดขัดแย้งกับ task ปัจจุบัน
- target column, problem type, time/order column, unit, grain, หรือ business objective ยังไม่ชัด
- metric ดีผิดปกติ เช่น 0.999-1.000 โดยยังไม่ได้พิสูจน์ leakage
- row count, schema, class ratio, date range, หรือ data source ไม่ได้มาจากไฟล์จริง
- dependency หรือ model/library ที่ต้องใช้ไม่มี และไม่มี fallback ที่พิสูจน์แล้วว่าเทียบเท่า
- ผลลัพธ์ของ agent ก่อนหน้าไม่ตรง contract หรือมีเฉพาะ report แต่ไม่มี artifact จริง

### Required Decision Block
ทุก report ต้องมี block นี้สำหรับ decision สำคัญ:
```text
DECISION_CHECK
Decision: [เลือก/ไม่เลือก/หยุดอะไร]
Question: [decision นี้ตอบคำถามอะไร]
Evidence: [ไฟล์/ตัวเลข/report/log ที่ใช้]
Alternatives: [อย่างน้อย 2 ทางเลือก หรือเหตุผลที่มีได้ทางเดียว]
Risk Check: [leakage/overfit/wrong file/stale output/business risk]
Assumptions: [สิ่งที่ยังสมมติ]
Confidence: [High/Medium/Low + เหตุผล]
Verdict: [PROCEED / LOOP_BACK / STOP_AND_REPAIR / ASK_USER]
```

### Confidence Rules
- **High** = มีไฟล์จริง + metric/หลักฐานครบ + ทางเลือกถูกเทียบ + risk ผ่าน
- **Medium** = หลักฐานพอทำ prototype แต่ยังมี limitation ต้องรายงานชัด
- **Low** = หลักฐานไม่พอ, ไฟล์ไม่ครบ, objective ไม่ชัด, หรือ risk ยังไม่ผ่าน → ห้าม proceed

## กฎเหล็ก — ทุก Report ต้องอธิบายเหตุผล (Human-Readable Reasoning)

**หลักการ:** คนอ่าน report ไม่ใช่คอมพิวเตอร์ — ทุก decision ต้องอธิบายว่า "ทำไม" ไม่ใช่แค่ "อะไร"

❌ **ห้ามเขียนแบบนี้ (แค่บอกผล):**
> "ลบ column `id` ออก"
> "เลือก LightGBM — F1=0.97"
> "Overfitting check: ✅ PASS"
> "drop 12 outliers"

✅ **ต้องเขียนแบบนี้ (บอกผล + เหตุผล + ความหมาย):**
> "ลบ column `id` ออก เพราะเป็น unique identifier ไม่มี predictive value — ถ้าเก็บไว้จะเกิด data leakage"
> "เลือก LightGBM เพราะข้อมูล tabular numerical ไม่ต้องการ linearity assumption และ n=569 เล็กเกินไปสำหรับ ANN"
> "Overfitting ผ่าน — train/test gap 2.9% ต่ำกว่าเกณฑ์ 5% หมายความว่า model เรียนรู้ pattern จริง ไม่ใช่จำข้อมูล"
> "drop 12 outliers (2.1%) เพราะ IQR×3 เกินจริง ตรวจสอบแล้วไม่ใช่ค่าพิเศษทางธุรกิจ"

---

### Reasoning Framework — ทุก Agent ใช้โครงสร้างนี้

**ทุก decision ใน report ต้องตอบ:**
1. **ทำอะไร** — action ที่ทำ
2. **เพราะอะไร** — เหตุผล (theory / data evidence / domain knowledge)
3. **เทียบกับอะไร** — ทางเลือกอื่นที่พิจารณาแล้วไม่เลือก (ถ้ามี)
4. **ผลที่ได้** — ความหมายต่อ downstream หรือ business

### กฎเฉพาะต่อ Agent

| Agent | สิ่งที่ต้องอธิบายเหตุผล |
|-------|----------------------|
| **Dana** | ทำไมถึง drop/impute column นี้, ทำไม outlier นี้ถึง flag, ทำไมใช้ method นี้ไม่ใช้อีก method |
| **Eddie** | ทำไม insight นี้ถึงสำคัญ, ทำไมถึงเลือก visualize แบบนี้, correlation นี้หมายความว่าอะไรต่อธุรกิจ |
| **Finn** | ทำไมถึงสร้าง feature นี้, ทำไมถึง drop feature นั้น, scaling method นี้เหมาะกับ algorithm ที่เลือกยังไง |
| **Mo** | ทำไมถึงเลือก algorithm นี้ (theory), ทำไมไม่เลือก ANN/Linear/อื่น, hyperparameter นี้ tuned ยังไง |
| **Quinn** | ผ่าน/ไม่ผ่านเพราะ criteria อะไร ตัวเลขเท่าไหร่เทียบกับเกณฑ์ |
| **Iris** | recommendation นี้มี evidence อะไรรองรับ, business impact ประเมินจากอะไร |
| **Rex** | สรุป reasoning ของทุก agent ให้ผู้บริหารอ่านเข้าใจได้ในประโยคเดียว |

### ตัวอย่างต่อ Agent

**Dana:**
```
ลบ column `customer_id` (เหตุผล: unique ID ทุก row — ไม่มี predictive value, ถ้าเก็บไว้ model จะ memorize แทนที่จะเรียนรู้ pattern)
impute `age` ด้วย median=34 (เหตุผล: missing 3.2% + distribution skewed right → median ดีกว่า mean ที่จะถูกดึงโดย outlier)
flag 8 outliers ใน `purchase_amount` (เหตุผล: > IQR×3 = 4,200 บาท ตรวจสอบแล้วไม่ใช่สินค้า premium — น่าจะเป็น data entry error)
```

**Finn:**
```
สร้าง feature `price_per_sqm` = price/area (เหตุผล: Eddie พบ correlation ทั้งสองตัวกับ target แต่ ratio น่าจะมี predictive power มากกว่าแยก เพราะ normalize ขนาดบ้าน)
drop `street_name` (เหตุผล: cardinality สูงมาก 847 unique values → One-Hot จะสร้าง 847 columns ทำให้ curse of dimensionality)
StandardScaler (เหตุผล: Mo Phase 1 เลือก SVM เป็น baseline — SVM sensitive ต่อ scale มาก feature ที่ไม่ scale จะ dominate kernel)
```

**Mo:**
```
เลือก LightGBM เพราะ:
- ข้อมูล tabular numerical 30 features — tree-based ไม่ต้องการ linearity assumption
- n=569 เล็กเกินไปสำหรับ ANN (ต้องการ >10K จึงจะชนะ tree-based)
- gradient boosting handle imbalance ได้ดีกว่า Random Forest ด้วย scale_pos_weight
ไม่เลือก XGBoost เพราะ F1 ต่างกัน 0.013 แต่ LightGBM train เร็วกว่า 3x
ไม่เลือก Logistic Regression เพราะ Eddie พบ non-linear patterns ใน correlation heatmap
```

**Quinn:**
```
Overfitting: ✅ ผ่าน — gap=2.9% < เกณฑ์ 5% → model เรียนรู้ pattern จริง ไม่ใช่จำ training data
CV Stability: ✅ ผ่าน — std=0.025 < เกณฑ์ 0.05 → ผลสม่ำเสมอข้าม fold ไม่ผันผวนตาม data split
Recall: ⚠️ ควรปรับ — recall=0.94 ต่ำกว่า medical threshold 0.97 → missed 6% ของ malignant cases = ผู้ป่วยที่พลาดการรักษา
```

---

## กฎ Output File

- **script ที่เขียนต้องผลิตไฟล์จริงเสมอ** — report .md อย่างเดียวไม่พอ
- agent ถัดไปใน pipeline จะหา output file ของ agent ก่อนหน้าเสมอ
- ถ้าไม่มีไฟล์ → pipeline พังทันที
