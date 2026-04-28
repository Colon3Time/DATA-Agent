import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='C:\\Users\\Amorntep\\DATA-Agent\\projects\\2026-04-25_test3\\input\\retail_sales_600.csv')
parser.add_argument('--output-dir', default='C:\\Users\\Amorntep\\DATA-Agent\\projects\\2026-04-25_test3\\output\\dana')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: โหลดข้อมูล และวิเคราะห์เบื้องต้น
# ============================================================
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')

# ข้อมูลเบื้องต้น
print('[STATUS] Basic info:')
print(df.info())

print('[STATUS] Basic stats:')
print(df.describe(include='all'))

print('[STATUS] Missing values per column:')
print(df.isnull().sum())

print('[STATUS] First 5 rows:')
print(df.head())

print('[STATUS] Data types:')
print(df.dtypes)


import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='C:\\Users\\Amorntep\\DATA-Agent\\projects\\2026-04-25_test3\\input\\retail_sales_600.csv')
parser.add_argument('--output-dir', default='C:\\Users\\Amorntep\\DATA-Agent\\projects\\2026-04-25_test3\\output\\dana')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: โหลดข้อมูล
# ============================================================
df_original = pd.read_csv(INPUT_PATH)
df = df_original.copy()
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Original columns: {list(df.columns)}')
print(f'[STATUS] Missing %:\n{df.isnull().sum() / len(df) * 100}')

# ============================================================
# STEP 2: วิเคราะห์และแปลง Data Types
# ============================================================
type_changes = []

# ตรวจสอบวันที่
date_cols = []
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
        date_cols.append(col)

for col in date_cols:
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        type_changes.append(f'{col}: converted to datetime')
        print(f'[STATUS] Converted {col} to datetime')
    except:
        pass

# ตรวจสอบตัวเลขที่เก็บเป็น string
for col in df.columns:
    if df[col].dtype == 'object':
        # ลองแปลงเป็นตัวเลข
        temp = pd.to_numeric(df[col], errors='coerce')
        if temp.notna().sum() > len(df) * 0.5:  # ถ้าเกิน 50% แปลงได้
            df[col] = temp
            type_changes.append(f'{col}: converted from object to numeric')

# ตรวจสอบ category (unique น้อย)
for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.05:
        df[col] = df[col].astype('category')
        type_changes.append(f'{col}: converted to category ({df[col].nunique()} levels)')

# ============================================================
# STEP 3: วิเคราะห์ Missing Values
# ============================================================
missing_report = {}
for col in df.columns:
    missing_pct = df[col].isnull().sum() / len(df) * 100
    missing_report[col] = missing_pct
    print(f'[STATUS] {col}: missing {missing_pct:.2f}%')

# ============================================================
# STEP 4: จัดการ Missing Values
# ============================================================
missing_handled = []

for col in df.columns:
    missing_pct = missing_report[col]
    if missing_pct == 0:
        continue
    
    # กรณี missing > 60%
    if missing_pct > 60:
        # พิจารณาตัด column ถ้าเป็น column ที่ไม่สำคัญ
        # แต่เก็บไว้ก่อนถ้าเป็น column ที่อาจมีประโยชน์
        if df[col].dtype in ['object', 'category']:
            df[col] = df[col].fillna('unknown')
            missing_handled.append(f'{col}: missing {missing_pct:.1f}% -> filled "unknown"')
        else:
            df[col] = df[col].fillna(df[col].median())
            missing_handled.append(f'{col}: missing {missing_pct:.1f}% -> median fill')
        continue
    
    # กรณี numeric columns
    if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        # เลือกวิธีตามจำนวน missing
        if missing_pct < 5:
            df[col] = df[col].fillna(df[col].median())
            missing_handled.append(f'{col}: missing {missing_pct:.1f}% -> median fill (low missing)')
        elif missing_pct < 30:
            # ลอง KNN ถ้าข้อมูลมีความสัมพันธ์กัน
            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
            if len(numeric_cols) >= 3 and df[col].notna().sum() > len(df) * 0.5:
                # ใช้ KNN imputation
                imputer_cols = [c for c in numeric_cols if df[c].notna().sum() > len(df) * 0.5]
                if col in imputer_cols and len(imputer_cols) >= 3:
                    try:
                        imputer_data = df[imputer_cols].copy()
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(imputer_data)
                        knn_imputer = KNNImputer(n_neighbors=5)
                        imputed_scaled = knn_imputer.fit_transform(scaled_data)
                        imputed_original = scaler.inverse_transform(imputed_scaled)
                        df[imputer_cols] = imputed_original
                        missing_handled.append(f'{col}: missing {missing_pct:.1f}% -> KNN Imputation (n=5, scaled)')
                        continue
                    except:
                        pass
            df[col] = df[col].fillna(df[col].median())
            missing_handled.append(f'{col}: missing {missing_pct:.1f}% -> median fill')
        else:
            df[col] = df[col].fillna(df[col].median())
            missing_handled.append(f'{col}: missing {missing_pct:.1f}% -> median fill (high missing)')
    
    # กรณี categorical/text columns
    elif df[col].dtype in ['object', 'category']:
        df[col] = df[col].fillna('unknown')
        missing_handled.append(f'{col}: missing {missing_pct:.1f}% -> filled "unknown"')

# ============================================================
# STEP 5: จัดการ Outliers
# ============================================================
outlier_handled = []
numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns

for col in numeric_cols:
    # ข้าม column ที่เป็น category-like (unique น้อย)
    if df[col].nunique() < 10:
        continue
    
    # ตรวจสอบ distribution
    from scipy.stats import skew
    col_skew = skew(df[col].dropna())
    
    if abs(col_skew) < 1:  # near normal
        # Z-score
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = np.where(z_scores > 3)[0]
        if len(outliers) > 0:
            outlier_count = len(outliers)
            # Clip แทนการตัดทิ้ง
            lower = df[col].mean() - 3 * df[col].std()
            upper = df[col].mean() + 3 * df[col].std()
            df[col] = df[col].clip(lower=lower, upper=upper)
            outlier_handled.append(f'{col}: Z-score capped {outlier_count} outliers (threshold=3)')
    else:  # skewed
        # IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            outlier_handled.append(f'{col}: IQR capped {outlier_count} outliers (IQR*1.5)')

# ============================================================
# STEP 6: เปรียบเทียบก่อน-หลัง
# ============================================================
before_shape = df_original.shape
after_shape = df.shape
before_missing = df_original.isnull().sum().sum()
after_missing = df.isnull().sum().sum()

print(f'[STATUS] Before: {before_shape}, Missing: {before_missing}')
print(f'[STATUS] After: {after_shape}, Missing: {after_missing}')

# ============================================================
# STEP 7: บันทึก Output
# ============================================================
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved CSV: {output_csv}')

# ============================================================
# STEP 8: เขียน Report
# ============================================================
report_lines = []
report_lines.append('# Dana Cleaning Report')
report_lines.append('====================')
report_lines.append(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
report_lines.append(f'Input: {INPUT_PATH}')
report_lines.append('')
report_lines.append(f'Before: {before_shape[0]} rows, {before_shape[1]} columns')
report_lines.append(f'After:  {after_shape[0]} rows, {after_shape[1]} columns')
report_lines.append(f'Total Missing: {before_missing} -> {after_missing}')
report_lines.append('')
report_lines.append('## Data Types Changes')
report_lines.append(f'Changes: {len(type_changes)}')
for ch in type_changes:
    report_lines.append(f'- {ch}')
report_lines.append('')
report_lines.append('## Missing Values Handling')
report_lines.append(f'Handled: {len(missing_handled)} columns')
for mh in missing_handled:
    report_lines.append(f'- {mh}')
report_lines.append('')
report_lines.append('## Outliers Handling')
report_lines.append(f'Handled: {len(outlier_handled)} columns')
for oh in outlier_handled:
    report_lines.append(f'- {oh}')
report_lines.append('')
report_lines.append('## Column Summary')
for col in df.columns:
    dtype = df[col].dtype
    nunique = df[col].nunique()
    missing = df[col].isnull().sum()
    report_lines.append(f'- {col}: {dtype}, {nunique} unique, {missing} missing')
report_lines.append('')
report_lines.append('## Data Quality Score')
# คำนวณคะแนนคร่าวๆ
score_before = (1 - before_missing / (before_shape[0] * before_shape[1])) * 100
score_after = (1 - after_missing / (after_shape[0] * after_shape[1])) * 100
report_lines.append(f'Before: {score_before:.1f}%')
report_lines.append(f'After:  {score_after:.1f}%')

report = '\n'.join(report_lines)

report_path = os.path.join(OUTPUT_DIR, 'dana_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] Saved Report: {report_path}')
print(report)

# ============================================================
# STEP 9: Self-Improvement Report
# ============================================================
improvement_lines = []
improvement_lines.append('## Self-Improvement Report')
improvement_lines.append(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
improvement_lines.append(f'Dataset: retail_sales_600.csv')
improvement_lines.append('')
improvement_lines.append('### What worked well:')
improvement_lines.append(f'- Handled {len(missing_handled)} missing columns with appropriate methods')
improvement_lines.append(f'- Capped {len(outlier_handled)} outlier columns preserving data integrity')
improvement_lines.append(f'- Reduced missing from {before_missing} to {after_missing}')
improvement_lines.append('')
improvement_lines.append('### What could be improved:')
improvement_lines.append('- Check for duplicate rows if any')
improvement_lines.append('- Consider cross-validation of KNN parameters')
improvement_lines.append('- Add more sophisticated outlier detection (Isolation Forest)')
improvement_lines.append('')
improvement_lines.append('### Key decisions:')
for mh in missing_handled:
    improvement_lines.append(f'- {mh}')
for oh in outlier_handled:
    improvement_lines.append(f'- {oh}')

improvement_report = '\n'.join(improvement_lines)

# บันทึกลง KB
kb_path = os.path.join(os.path.dirname(os.path.dirname(OUTPUT_DIR)), 'knowledge_base', 'dana_methods.md')
os.makedirs(os.path.dirname(kb_path), exist_ok=True)
with open(kb_path, 'a', encoding='utf-8') as f:
    f.write(f'\n\n## [{datetime.now().strftime("%Y-%m-%d %H:%M")}] Self-Improvement - retail_sales_600\n')
    f.write(improvement_report)
    f.write('\n')
print(f'[STATUS] Saved KB update: {kb_path}')

print('[STATUS] All output files created successfully')
print('[STATUS] Dana cleaning complete')


import os
output_dir = r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test3\output\dana'
print(f'[STATUS] Files in {output_dir}:')
for f in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f'  - {f} ({size:,} bytes)')

# Verify CSV
df_check = pd.read_csv(os.path.join(output_dir, 'dana_output.csv'))
print(f'[STATUS] Output CSV: {df_check.shape}')
print(f'[STATUS] Missing in output: {df_check.isnull().sum().sum()}')