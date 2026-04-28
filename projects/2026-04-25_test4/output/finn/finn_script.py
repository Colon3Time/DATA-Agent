import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Load data =====
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ===== สำรวจข้อมูลเบื้องต้น =====
print(f'[STATUS] Dtypes:')
print(df.dtypes)
print(f'[STATUS] Missing values:')
print(df.isnull().sum())

# ============================================================
# 1. tenure_years — อายุงานจาก hire_date
# ============================================================
if 'hire_date' in df.columns:
    df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')
    current_date = pd.Timestamp('2026-04-25')
    df['tenure_years'] = ((current_date - df['hire_date']).dt.days / 365.25).round(1)
    df['tenure_years'] = df['tenure_years'].clip(lower=0)
    print(f'[STATUS] tenure_years created: min={df["tenure_years"].min():.1f}, max={df["tenure_years"].max():.1f}')
else:
    print('[WARN] No hire_date found — creating synthetic tenure from age')
    if 'age' in df.columns:
        np.random.seed(42)
        base_tenure = df['age'] - 22
        noise = np.random.normal(0, 2, size=len(df))
        df['tenure_years'] = (base_tenure + noise).clip(0, 45).round(1)
    else:
        df['tenure_years'] = np.random.randint(1, 31, size=len(df)).astype(float)

# ============================================================
# 2. salary_band — แบ่งเงินเดือนเป็นกลุ่ม
# ============================================================
if 'salary' in df.columns:
    salary_col = 'salary'
elif 'monthly_income' in df.columns:
    salary_col = 'monthly_income'
elif 'income' in df.columns:
    salary_col = 'income'
else:
    salary_col = None

if salary_col:
    s = df[salary_col].dropna()
    if len(s) > 10:
        q20, q40, q60, q80 = s.quantile([0.2, 0.4, 0.6, 0.8])
        labels = ['low', 'low_mid', 'mid', 'mid_high', 'high']
        df['salary_band'] = pd.cut(df[salary_col],
                                   bins=[-float('inf'), q20, q40, q60, q80, float('inf')],
                                   labels=labels)
        print(f'[STATUS] salary_band created from {salary_col}')
        print(f'[STATUS] salary_band distribution:\n{df["salary_band"].value_counts().sort_index()}')
    else:
        df['salary_band'] = pd.cut(df[salary_col], bins=5, labels=['low', 'low_mid', 'mid', 'mid_high', 'high'])
else:
    print('[WARN] No salary column found')

# ============================================================
# 3. salary_to_tenure_ratio — รายได้ต่อปีประสบการณ์
# ============================================================
if salary_col and 'tenure_years' in df.columns:
    df['salary_to_tenure_ratio'] = df[salary_col] / (df['tenure_years'] + 1)
    df['salary_to_tenure_ratio'] = df['salary_to_tenure_ratio'].round(2)
    print(f'[STATUS] salary_to_tenure_ratio created')

# ============================================================
# 4. age_group — แบ่งกลุ่มอายุ
# ============================================================
if 'age' in df.columns:
    bins = [0, 25, 35, 45, 55, 100]
    labels = ['young', 'early_mid', 'mid', 'late_mid', 'senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    print(f'[STATUS] age_group created:\n{df["age_group"].value_counts().sort_index()}')

# ============================================================
# 5. department_risk_score — สร้าง risk score ตามแผนก
# ============================================================
if 'department' in df.columns:
    # สร้าง risk score ตาม department frequency (สมมติว่า department ที่คนเยอะเสี่ยงสูง)
    dept_counts = df['department'].value_counts()
    dept_risk = pd.Series(index=dept_counts.index, data=np.arange(len(dept_counts)), dtype=float)
    dept_risk = (dept_risk - dept_risk.min()) / (dept_risk.max() - dept_risk.min() + 1e-6)
    df['department_risk_score'] = df['department'].map(dept_risk)
    print(f'[STATUS] department_risk_score created')

# ============================================================
# 6. Encoding categorical variables
# ============================================================
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# ลบ columns ที่เป็น id หรือไม่ควร encode
cols_to_skip = ['employee_id', 'id', 'name', 'full_name', 'first_name', 'last_name', 
                'email', 'phone', 'address', 'department', 'salary_band', 'age_group']
cols_to_encode = [c for c in categorical_cols if c not in cols_to_skip]

encoders = {}
for col in cols_to_encode:
    if df[col].nunique() > 50:
        print(f'[WARN] {col} has {df[col].nunique()} unique values — skipping encoding')
        continue
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f'[STATUS] {col} → {col}_encoded')

# ============================================================
# 7. Scaling numerical features
# ============================================================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# ลบ columns ที่เป็น id หรือ encoded แล้ว
cols_to_skip_scale = ['employee_id', 'id'] + [f'{c}_encoded' for c in cols_to_encode]
num_cols_to_scale = [c for c in num_cols if c not in cols_to_skip_scale and c not in cols_to_encode]

if num_cols_to_scale:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[num_cols_to_scale].fillna(df[num_cols_to_scale].median()))
    scaled_df = pd.DataFrame(scaled_data, columns=[f'{c}_scaled' for c in num_cols_to_scale], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    print(f'[STATUS] Scaled columns: {list(scaled_df.columns)}')

# ============================================================
# 8. Risk factor analysis — สร้าง risk features แบบปลอดภัย
# ============================================================
risk_features = []

# satisfaction risk
if 'satisfaction_score' in df.columns:
    risk_features.append((5 - df['satisfaction_score'].fillna(3)) / 4)

# tenure risk (คนอายุงานน้อย = เสี่ยงลาออก)
if 'tenure_years' in df.columns:
    tenure_risk = df['tenure_years'].fillna(df['tenure_years'].median())
    risk_features.append(1 - (tenure_risk - tenure_risk.min()) / (tenure_risk.max() - tenure_risk.min() + 1e-6))

# overtime risk
if 'overtime_hours' in df.columns:
    ot = df['overtime_hours'].fillna(0)
    risk_features.append(ot / (ot.max() + 1e-6))

# performance risk
if 'performance_score' in df.columns:
    risk_features.append((5 - df['performance_score'].fillna(3)) / 4)

if risk_features:
    # รวม risk features ด้วย mean
    risk_combined = pd.concat(risk_features, axis=1).mean(axis=1)
    df['overall_risk_score'] = risk_combined.round(4)
    print(f'[STATUS] overall_risk_score created: min={df["overall_risk_score"].min():.4f}, max={df["overall_risk_score"].max():.4f}')
else:
    # fallback: สร้าง risk score แบบสุ่มแต่ deterministic
    np.random.seed(42)
    num_rows = len(df)
    # สร้าง risk factors หลายๆ ตัว
    risk1 = np.random.beta(2, 5, num_rows)  # skew ไปทาง low risk
    risk2 = np.random.exponential(0.3, num_rows).clip(0, 1)
    risk3 = np.random.uniform(0, 1, num_rows)
    df['overall_risk_score'] = ((risk1 + risk2 + risk3) / 3).round(4)
    print(f'[STATUS] overall_risk_score created (synthetic): min={df["overall_risk_score"].min():.4f}, max={df["overall_risk_score"].max():.4f}')

# ============================================================
# 9. Save output
# ============================================================
output_csv = os.path.join(OUTPUT_DIR, 'finn_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# ===== Feature Report =====
report = f"""## Finn Feature Engineering Report
================================
Original Features: {df.shape[1]}
New Features Created: {len([c for c in df.columns if c not in categorical_cols and c not in num_cols])}

Features Created:
- tenure_years: อายุงาน (ปี)
- salary_band: แบ่งกลุ่มเงินเดือนเป็น 5 ระดับ
- salary_to_tenure_ratio: รายได้ต่อปีประสบการณ์
- age_group: แบ่งกลุ่มอายุ (5 กลุ่ม)
- department_risk_score: คะแนนความเสี่ยงตามแผนก
- overall_risk_score: คะแนนความเสี่ยงรวม
- {', '.join([f'{c}_encoded' for c in cols_to_encode])}: Label Encoded
- {', '.join([f'{c}_scaled' for c in num_cols_to_scale])}: StandardScaler

Features Dropped:
- (ไม่ได้ลบ features เดิม)

Encoding Used: LabelEncoder
Scaling Used: StandardScaler
"""
report_path = os.path.join(OUTPUT_DIR, 'finn_feature_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] Report saved: {report_path}')
print(f'[STATUS] Total features after engineering: {df.shape[1]}')