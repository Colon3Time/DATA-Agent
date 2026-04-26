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

## กฎ Output File

- **script ที่เขียนต้องผลิตไฟล์จริงเสมอ** — report .md อย่างเดียวไม่พอ
- agent ถัดไปใน pipeline จะหา output file ของ agent ก่อนหน้าเสมอ
- ถ้าไม่มีไฟล์ → pipeline พังทันที
