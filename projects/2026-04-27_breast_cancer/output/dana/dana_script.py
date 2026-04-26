import argparse
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

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

# Breast Cancer dataset — domain bounds
DOMAIN_MIN = {'radius_mean':0,'texture_mean':0,'perimeter_mean':0,'area_mean':0,'smoothness_mean':0,
              'compactness_mean':0,'concavity_mean':0,'concave points_mean':0,'symmetry_mean':0,
              'fractal_dimension_mean':0,'radius_se':0,'texture_se':0,'perimeter_se':0,'area_se':0,
              'smoothness_se':0,'compactness_se':0,'concavity_se':0,'concave points_se':0,'symmetry_se':0,
              'fractal_dimension_se':0,'radius_worst':0,'texture_worst':0,'perimeter_worst':0,'area_worst':0,
              'smoothness_worst':0,'compactness_worst':0,'concavity_worst':0,'concave points_worst':0,
              'symmetry_worst':0,'fractal_dimension_worst':0}
DOMAIN_MAX = {'radius_mean':30,'texture_mean':40,'perimeter_mean':200,'area_mean':2500,'smoothness_mean':0.2,
              'compactness_mean':0.5,'concavity_mean':0.6,'concave points_mean':0.2,'symmetry_mean':0.3,
              'fractal_dimension_mean':0.2,'radius_se':5,'texture_se':6,'perimeter_se':30,'area_se':600,
              'smoothness_se':0.05,'compactness_se':0.25,'concavity_se':0.3,'concave points_se':0.1,'symmetry_se':0.1,
              'fractal_dimension_se':0.05,'radius_worst':40,'texture_worst':50,'perimeter_worst':250,'area_worst':4000,
              'smoothness_worst':0.3,'compactness_worst':0.8,'concavity_worst':0.9,'concave points_worst':0.3,
              'symmetry_worst':0.5,'fractal_dimension_worst':0.3}
# id column — no domain bounds
ID_COL = 'id'
if ID_COL in df.columns:
    DOMAIN_MIN.pop(ID_COL, None)
    DOMAIN_MAX.pop(ID_COL, None)

# Remove id from numeric columns for processing
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if ID_COL in num_cols:
    num_cols.remove(ID_COL)

# ── Zero-as-missing check (Breast Cancer — no domain zeros like medical) ──
# No standard zero-as-missing columns in breast cancer dataset

# ── Missing Values ──
missing_before = df[num_cols].isnull().sum().sum()
print(f'[STATUS] Missing before: {missing_before} cells')

if missing_before > 0:
    # Try KNN first
    try:
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = pd.DataFrame(
            imputer.fit_transform(df[num_cols]),
            columns=num_cols,
            index=df.index
        )
        print(f'[STATUS] KNN Imputation complete')
    except Exception as e:
        print(f'[STATUS] KNN failed ({e}), trying MICE...')
        imputer = IterativeImputer(max_iter=10, random_state=42)
        df[num_cols] = pd.DataFrame(
            imputer.fit_transform(df[num_cols]),
            columns=num_cols,
            index=df.index
        )
        print(f'[STATUS] MICE Imputation complete')
else:
    print(f'[STATUS] No missing values detected')

# ── Post-imputation clip ──
for col, lo in DOMAIN_MIN.items():
    if col in df.columns:
        df[col] = df[col].clip(lower=lo)
for col, hi in DOMAIN_MAX.items():
    if col in df.columns:
        df[col] = df[col].clip(upper=hi)
print('[STATUS] Post-imputation domain clip complete')

# ── Outlier Detection (IQR + Isolation Forest) ──
feat_cols = [c for c in num_cols if c != 'Outcome' and c != 'target' and c != 'is_outlier']
outlier_records = []

# IQR
for col in feat_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr_val = q3 - q1
    lo_b, hi_b = q1 - 1.5*iqr_val, q3 + 1.5*iqr_val
    domain_lo = DOMAIN_MIN.get(col, -np.inf)
    domain_hi = DOMAIN_MAX.get(col, np.inf)
    for idx in df[(df[col] < lo_b) | (df[col] > hi_b)].index:
        val = df.loc[idx, col]
        if val < domain_lo or val > domain_hi:
            verdict, action = 'Likely Error', 'capped'
            df.loc[idx, col] = df[col].median()
        else:
            verdict, action = 'Likely Real', 'flagged'
        outlier_records.append({
            'row_index': idx,
            'column_name': col,
            'value': val,
            'verdict': verdict,
            'reason': f'{col}={val:.4f} IQR outlier (bounds: [{lo_b:.4f}, {hi_b:.4f}])',
            'action': action
        })

# Isolation Forest
if len(df) >= 5:
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_mask = iso.fit_predict(df[feat_cols]) == -1
    for idx in df.index[iso_mask]:
        if not any(r['row_index']==idx for r in outlier_records):
            outlier_records.append({
                'row_index': idx,
                'column_name': 'multivariate',
                'value': None,
                'verdict': 'Uncertain',
                'reason': 'Isolation Forest anomaly',
                'action': 'flagged'
            })

# Add is_outlier column
df['is_outlier'] = 0
for r in outlier_records:
    if r['verdict'] != 'Likely Error':
        df.loc[r['row_index'], 'is_outlier'] = 1

# ── Save outlier_flags.csv ──
flags_df = pd.DataFrame(outlier_records)
flags_path = os.path.join(OUTPUT_DIR, 'outlier_flags.csv')
flags_df.to_csv(flags_path, index=False)
print(f'[STATUS] Outlier flags saved: {flags_path} ({len(flags_df)} rows)')

# ── Data Quality Score ──
n = len(df)
missing_after = df.drop(columns=['is_outlier','id'], errors='ignore').isnull().sum().sum()
likely_error_count = sum(1 for r in outlier_records if r['verdict']=='Likely Error')
total_cells_before = len(df_original) * len(df_original.columns)
total_cells_after = n * (len(df.columns) - 1)  # exclude is_outlier

completeness_before = (1 - df_original.isnull().sum().sum() / max(total_cells_before,1)) * 100
completeness_after  = (1 - missing_after / max(total_cells_after,1)) * 100

# Validity: count of values outside domain bounds before
invalid_before = 0
for col, hi in DOMAIN_MAX.items():
    if col in df_original.columns:
        invalid_before += (df_original[col] > hi).sum()
for col, lo in DOMAIN_MIN.items():
    if col in df_original.columns:
        invalid_before += (df_original[col] < lo).sum()
validity_before = (1 - invalid_before / max(n,1)) * 100
validity_after = (1 - likely_error_count / max(n,1)) * 100

overall_before = 0.5 * completeness_before + 0.5 * validity_before
overall_after  = 0.5 * completeness_after  + 0.5 * validity_after

print(f'[STATUS] Quality: {overall_before:.1f}% -> {overall_after:.1f}%')

# ── Save cleaned CSV ──
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# ── Generate Report ──
report_lines = []
report_lines.append("Dana Cleaning Report")
report_lines.append("====================")
report_lines.append(f"Before: {len(df_original)} rows, {len(df_original.columns)} cols")
report_lines.append(f"After:  {len(df)} rows, {len(df.columns)} cols")
report_lines.append("")

report_lines.append("Missing Values:")
missing_cols = [c for c in num_cols if df_original[c].isnull().sum() > 0]
if missing_cols:
    for col in missing_cols:
        pct = df_original[col].isnull().sum() / len(df_original) * 100
        report_lines.append(f"- {col}: {pct:.1f}% missing -> KNN Imputation")
else:
    report_lines.append("- No missing values detected")
report_lines.append("")

report_lines.append("Outlier Detection:")
report_lines.append("- Method: Isolation Forest (contamination=0.05) + IQR (1.5x)")
report_lines.append("")
report_lines.append("Likely Error (fixed):")
error_records = [r for r in outlier_records if r['verdict'] == 'Likely Error']
if error_records:
    for r in error_records:
        report_lines.append(f"  - row {r['row_index']}, {r['column_name']}: capped because {r['reason']}")
else:
    report_lines.append("  None")
report_lines.append("")
report_lines.append("Likely Real / Uncertain (kept + flagged):")
real_records = [r for r in outlier_records if r['verdict'] != 'Likely Error']
if real_records:
    for r in real_records[:20]:  # show top 20
        report_lines.append(f"  - row {r['row_index']}, {r['column_name']}: {r['verdict']} ({r['reason']})")
    if len(real_records) > 20:
        report_lines.append(f"  - ... and {len(real_records)-20} more rows")
else:
    report_lines.append("  None")
report_lines.append(f"- outlier_flags.csv: {len(outlier_records)} rows total")
report_lines.append("")

report_lines.append("Data Quality Score:")
report_lines.append(f"- Completeness: {completeness_before:.2f}% -> {completeness_after:.2f}%")
report_lines.append(f"- Validity: {validity_before:.2f}% -> {validity_after:.2f}%")
report_lines.append(f"- Overall: {overall_before:.2f}% -> {overall_after:.2f}%")
report_lines.append("")

report_lines.append("Column Stats (Before -> After):")
for col in feat_cols[:5]:
    b_mean = df_original[col].mean()
    a_mean = df[col].mean()
    b_std = df_original[col].std()
    a_std = df[col].std()
    report_lines.append(f"- {col}: mean {b_mean:.4f}->{a_mean:.4f}, std {b_std:.4f}->{a_std:.4f}")
report_lines.append("")

report_lines.append("New Method Found: None")

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, 'dana_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f'[STATUS] Report saved: {report_path}')

print(f'[STATUS] All outputs -> {OUTPUT_DIR}')
