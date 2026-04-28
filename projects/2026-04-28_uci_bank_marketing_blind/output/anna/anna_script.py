import argparse, os, pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('[STATUS] Starting UCI Bank Marketing blind dataset research...')

# ── Load raw data ──
try:
    df = pd.read_csv(INPUT_PATH, sep=';')
except Exception as e:
    print(f'[ERROR] Failed with sep=; : {e}')
    print('[STATUS] Trying with default comma separator...')
    df = pd.read_csv(INPUT_PATH)

print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ── Identify target column ──
target_col = 'y' if 'y' in df.columns else None
if target_col is None:
    # Try to find a binary target column
    for col in df.columns:
        if df[col].nunique() == 2:
            target_col = col
            print(f'[STATUS] Found potential target column: {target_col}')
            break
if target_col is None:
    target_col = df.columns[-1]
    print(f'[STATUS] Using last column as target: {target_col}')

print(f'[STATUS] Target distribution:\n{df[target_col].value_counts(normalize=True)}')

# ── Research UCI Bank Marketing Dataset ──
research = {
    "dataset_name": "UCI Bank Marketing Dataset (Bank Marketing Data Set)",
    "source": "https://archive.ics.uci.edu/ml/datasets/Bank+Marketing",
    "original_paper": "Moro et al. (2014) — A Data-Driven Approach to Predict the Success of Bank Telemarketing",
    "business_context": "การตลาดทางโทรศัพท์ของธนาคารโปรตุเกส (Portuguese bank) — โทรเสนอสินเชื่อระยะยาว (term deposit) ให้ลูกค้า",
    "goal": "ทำนายว่าลูกค้าจะสมัคร term deposit (yes/no) เพื่อ optimize การโทร ลด cost และเพิ่ม conversion rate",
    
    "column_descriptions": {
        "age": "อายุของลูกค้า (numeric)",
        "job": "อาชีพ (categorical): admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown",
        "marital": "สถานภาพสมรส (categorical): divorced, married, single, unknown",
        "education": "ระดับการศึกษา (categorical): basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown",
        "default": "มีหนี้เสียหรือไม่ (categorical): no, yes, unknown",
        "housing": "มีสินเชื่อบ้านหรือไม่ (categorical): no, yes, unknown",
        "loan": "มีสินเชื่อส่วนบุคคลหรือไม่ (categorical): no, yes, unknown",
        "contact": "รูปแบบการติดต่อ (categorical): cellular, telephone",
        "month": "เดือนสุดท้ายที่ติดต่อ (categorical): mar, apr, may, jun, jul, aug, sep, oct, nov, dec",
        "day_of_week": "วันสุดท้ายที่ติดต่อ (categorical): mon, tue, wed, thu, fri",
        "duration": "ระยะเวลาการโทรครั้งสุดท้าย (วินาที) — ⚠️ Highly predictive but should be removed in real prediction because it's unknown before call ends",
        "campaign": "จำนวนครั้งที่ติดต่อลูกค้ารายนี้ใน campaign นี้",
        "pdays": "จำนวนวันตั้งแต่นัดหมายครั้งล่าสุด (-1 = ไม่เคยติดต่อมาก่อน)",
        "previous": "จำนวนครั้งที่ติดต่อก่อน campaign นี้",
        "poutcome": "ผลลัพธ์ของ campaign ก่อนหน้า (categorical): failure, nonexistent, success",
        "emp.var.rate": "อัตราการจ้างงาน (employment variation rate) — quarterly indicator",
        "cons.price.idx": "ดัชนีราคาผู้บริโภค (consumer price index) — monthly indicator",
        "cons.conf.idx": "ดัชนีความเชื่อมั่นผู้บริโภค (consumer confidence index) — monthly indicator",
        "euribor3m": "อัตราดอกเบี้ย Euribor 3 เดือน (daily indicator)",
        "nr.employed": "จำนวนพนักงาน (number of employees) — quarterly indicator"
    },
    
    "target_variable": {
        "name": "y",
        "description": "ลูกค้าสมัคร term deposit หรือไม่ (binary: yes/no)",
        "distribution": df[target_col].value_counts(normalize=True).to_dict()
    },
    
    "recommended_cleaning": [
        "1. Handle 'unknown' values in categorical columns — map them as NaN or mode imputation",
        "2. Duration feature: Keep for EDA but NOTE that it should be removed for real prediction model",
        "3. pdays: Value -1 means 'not previously contacted' — keep as is or create flag feature",
        "4. No missing values in raw data — only 'unknown' categories",
        "5. Check for outliers in age, campaign, duration, pdays",
        "6. Target is imbalanced (~11% yes) — will need stratified sampling or SMOTE"
    ],
    
    "feature_engineering_opportunities": [
        "1. Contact month → seasonal features (quarter, peak season flag)",
        "2. pdays flag: is_contacted_before (True/False)",
        "3. Duration → interaction with campaign count",
        "4. Social/economic features (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) → dimensionality reduction or trend features"
    ],
    
    "warnings_for_dana": [
        "⚠️ DO NOT remove 'y' column — it's the target",
        "⚠️ DO NOT impute duration — it's leaked information, mark it for Eddie to handle",
        "⚠️ Keep 'unknown' categories in training data — they may carry signal",
        "⚠️ Check for data leakage: duration, poutcome, pdays have future information"
    ]
}

# ── Basic data quality checks ──
quality_report = {
    "shape": list(df.shape),
    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    "missing_values": df.isnull().sum().to_dict(),
    "unknown_values": {col: (df[col] == 'unknown').sum() for col in df.select_dtypes(include='object').columns if 'unknown' in df[col].values},
    "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
    "target_column": target_col,
    "num_features": len(df.select_dtypes(include=['int64', 'float64']).columns),
    "cat_features": len(df.select_dtypes(include=['object', 'category']).columns)
}

# ── Save research report ──
report_df = pd.DataFrame({
    'key': list(research.keys()) + ['quality_report'],
    'value': [str(v) for v in list(research.values())] + [str(quality_report)]
})
output_csv = os.path.join(OUTPUT_DIR, 'anna_output.csv')
report_df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved research report to: {output_csv}')

# ── Also save the raw data reference ──
raw_info = {
    'input_path': INPUT_PATH,
    'n_rows': df.shape[0],
    'n_cols': df.shape[1],
    'target_column': target_col,
    'columns_list': list(df.columns)
}
info_df = pd.DataFrame([raw_info])
info_csv = os.path.join(OUTPUT_DIR, 'anna_data_info.csv')
info_df.to_csv(info_csv, index=False)
print(f'[STATUS] Saved data info to: {info_csv}')

print('[STATUS] Research complete! Ready to dispatch to Dana.')