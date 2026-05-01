# ── STEP 1: Load data ──
import argparse, os, pandas as pd, numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
import warnings; warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()
INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not INPUT_PATH or not os.path.exists(INPUT_PATH):
    print(f'[ERROR] --input required and must exist: {INPUT_PATH}')
    import sys; sys.exit(1)

df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape} from {INPUT_PATH}')
df_original = df.copy()

# ── STEP 2: Zero-as-missing ──
ZERO_INVALID_COLS = [c for c in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] if c in df.columns]
for col in ZERO_INVALID_COLS:
    n = (df[col] == 0).sum()
    if n > 0:
        df[col] = df[col].replace(0, np.nan)
        print(f'[STATUS] {col}: {n} zeros -> NaN')

# ── STEP 3: KNN Imputation ──
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if df[num_cols].isnull().sum().sum() > 0:
    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols, index=df.index)
    print(f'[STATUS] KNN Imputation complete')

# ── STEP 3b: Post-imputation clip ──
DOMAIN_MIN = {'Glucose':0,'BloodPressure':0,'SkinThickness':0,'Insulin':0,'BMI':0,'Pregnancies':0,'Age':0}
DOMAIN_MAX = {'Glucose':300,'BloodPressure':200,'SkinThickness':80,'Insulin':500,'BMI':70,'DiabetesPedigreeFunction':2.5}
for col, lo in DOMAIN_MIN.items():
    if col in df.columns: df[col] = df[col].clip(lower=lo)
for col, hi in DOMAIN_MAX.items():
    if col in df.columns: df[col] = df[col].clip(upper=hi)
print('[STATUS] Post-imputation domain clip complete')

# ── STEP 4: Outlier Detection ──
feat_cols = [c for c in num_cols if c != 'Outcome']
outlier_records = []

for col in feat_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lo_b, hi_b = q1 - 1.5*iqr, q3 + 1.5*iqr
    domain_lo = DOMAIN_MIN.get(col, -np.inf)
    domain_hi = DOMAIN_MAX.get(col, np.inf)
    for idx in df[(df[col] < lo_b) | (df[col] > hi_b)].index:
        val = df.loc[idx, col]
        if val < domain_lo or val > domain_hi:
            verdict, action = 'Likely Error', 'capped'
            df.loc[idx, col] = df[col].median()
        else:
            verdict, action = 'Likely Real', 'flagged'
        outlier_records.append({'row_index':idx,'column_name':col,'value':val,'verdict':verdict,'reason':f'{col}={val:.2f} IQR outlier','action':action})

iso = IsolationForest(contamination=0.05, random_state=42)
iso_mask = iso.fit_predict(df[feat_cols]) == -1
for idx in df.index[iso_mask]:
    if not any(r['row_index']==idx for r in outlier_records):
        outlier_records.append({'row_index':idx,'column_name':'multivariate','value':None,'verdict':'Uncertain','reason':'Isolation Forest anomaly','action':'flagged'})

df['is_outlier'] = 0
for r in outlier_records:
    if r['verdict'] != 'Likely Error': df.loc[r['row_index'], 'is_outlier'] = 1

# ── STEP 5: Save outlier_flags.csv ──
flags_df = pd.DataFrame(outlier_records)
flags_path = os.path.join(OUTPUT_DIR, 'outlier_flags.csv')
flags_df.to_csv(flags_path, index=False)
print(f'[STATUS] Saved outlier_flags.csv: {len(flags_df)} rows')

# ── STEP 6: Data Quality Score ──
n = len(df)
missing_after = df.drop(columns=['is_outlier']).isnull().sum().sum()
likely_error_count = sum(1 for r in outlier_records if r['verdict']=='Likely Error')
completeness_before = (1 - df_original.isnull().sum().sum() / (len(df_original)*len(df_original.columns))) * 100
completeness_after  = (1 - missing_after / (n * (len(df.columns)-1))) * 100
validity_before = (1 - sum(1 for c in DOMAIN_MAX for i in df_original.index if c in df_original.columns and df_original.loc[i,c] > DOMAIN_MAX[c]) / max(n,1)) * 100
validity_after  = (1 - likely_error_count / max(n,1)) * 100
overall_before  = 0.5*completeness_before + 0.5*validity_before
overall_after   = 0.5*completeness_after  + 0.5*validity_after
print(f'[STATUS] Quality: {overall_before:.1f}% -> {overall_after:.1f}%')

# ── STEP 7: Save dana_output.csv ──
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved dana_output.csv: {df.shape}')

# ── STEP 8: Save dana_report.md ──
report_lines = []
report_lines.append('Dana Cleaning Report')
report_lines.append('====================')
report_lines.append(f'Before: {len(df_original)} rows, {len(df_original.columns)} columns')
report_lines.append(f'After:  {len(df)} rows, {len(df.columns)} columns')
report_lines.append('')
report_lines.append('Missing Values:')
has_missing = False
for col in df_original.columns:
    if col == 'is_outlier': continue
    before = df_original[col].isnull().sum()
    after = df[col].isnull().sum()
    if before > 0 or after > 0:
        has_missing = True
        report_lines.append(f'- {col}: Before {before}, After {after} rows missing -> KNN Imputation (n_neighbors=5)')
if not has_missing:
    report_lines.append('- No missing values detected')
report_lines.append('')
report_lines.append('Outlier Detection:')
report_lines.append('- Method: Isolation Forest (contamination=0.05) + IQR (1.5x)')
report_lines.append('- Likely Error (fixed):')
error_rows = [r for r in outlier_records if r['verdict']=='Likely Error']
if error_rows:
    for r in error_rows:
        report_lines.append(f'  - row {r["row_index"]}: {r["column_name"]}={r["value"]:.2f} -> capped with median (domain bound violation)')
else:
    report_lines.append('  - None')
report_lines.append('- Likely Real / Uncertain (kept + flagged):')
real_rows = [r for r in outlier_records if r['verdict']!='Likely Error']
if real_rows:
    for r in real_rows:
        report_lines.append(f'  - row {r["row_index"]}: {r["column_name"]}={r["value"]} -> is_outlier=1 ({r["verdict"]})')
else:
    report_lines.append('  - None')
report_lines.append(f'- outlier_flags.csv: {len(outlier_records)} rows total')
report_lines.append('')
report_lines.append('Data Quality Score:')
report_lines.append(f'- Completeness: Before {completeness_before:.1f}% -> After {completeness_after:.1f}%')
report_lines.append(f'- Validity: Before {validity_before:.1f}% -> After {validity_after:.1f}%')
report_lines.append(f'- Overall: Before {overall_before:.1f}% -> After {overall_after:.1f}%')
report_lines.append('')
report_lines.append('Column Stats (Before -> After):')
for col in num_cols:
    if col == 'Outcome': continue
    b_mean = df_original[col].mean()
    a_mean = df[col].mean()
    b_std = df_original[col].std()
    a_std = df[col].std()
    report_lines.append(f'- {col}: mean {b_mean:.2f} -> {a_mean:.2f}, std {b_std:.2f} -> {a_std:.2f}')
report_lines.append('')
report_lines.append('New Method Found: None')

report_path = os.path.join(OUTPUT_DIR, 'dana_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f'[STATUS] Saved dana_report.md')

# ── Agent Report ──
agent_report_lines = []
agent_report_lines.append('Agent Report - Dana')
agent_report_lines.append('====================')
agent_report_lines.append('รับจาก     : Scout (scout_output.csv)')
agent_report_lines.append('Input      : Data from UCI ML Diabetes dataset')
agent_report_lines.append('ทำ         :')
agent_report_lines.append('  - แปลง 0 เป็น NaN สำหรับ Glucose, BloodPressure, SkinThickness, Insulin, BMI')
agent_report_lines.append('  - KNN Imputation (n_neighbors=5)')
agent_report_lines.append('  - Domain clipping (min/max bounds)')
agent_report_lines.append('  - Outlier detection: IQR + Isolation Forest')
agent_report_lines.append(f'  - จําแนก outlier: Likely Error {len(error_rows)} rows, Likely Real/Uncertain {len(real_rows)} rows')
agent_report_lines.append(f'พบ         :')
agent_report_lines.append(f'  - Missing values after zero-to-NaN conversion: {df_original.isnull().sum().sum()} -> {missing_after}')
agent_report_lines.append(f'  - Data quality improved: {overall_before:.1f}% -> {overall_after:.1f}%')
agent_report_lines.append(f'  - Outlier patterns: IQR detect univariate extremes, Isolation Forest detect multivariate anomalies')
agent_report_lines.append(f'เปลี่ยนแปลง : Data cleaned, outliers flagged, quality score computed')
agent_report_lines.append(f'ส่งต่อ     : Finn - dana_output.csv (clean data with is_outlier column)')

agent_report_path = os.path.join(OUTPUT_DIR, 'dana_agent_report.md')
with open(agent_report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(agent_report_lines))
print(f'[STATUS] Saved dana_agent_report.md')
print('[STATUS] Dana cleaning complete')