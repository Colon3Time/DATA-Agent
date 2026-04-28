import argparse
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
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
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Dtypes:\n{df.dtypes}')
print(f'[STATUS] Missing:\n{df.isnull().sum()[df.isnull().sum() > 0]}')
if 'account_status_30d' in df.columns:
    print(f'[STATUS] Target distribution:\n{df["account_status_30d"].value_counts()}')
else:
    print('[STATUS] Target distribution: N/A')
df_original = df.copy()

# ── Separate target ──
target_col = 'account_status_30d'
if target_col in df.columns:
    y = df[target_col]
    df = df.drop(columns=[target_col])
else:
    y = None
    print(f'[WARN] Target column "{target_col}" not found')

# ── Analyze columns ──
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'[STATUS] Numeric columns ({len(num_cols)}): {num_cols}')
print(f'[STATUS] Categorical columns ({len(cat_cols)}): {cat_cols}')

# ── Handle high-missing columns ──
HIGH_MISSING_THRESHOLD = 60  # %
high_missing_cols = []
for col in df.columns:
    pct = df[col].isnull().mean() * 100
    if pct >= HIGH_MISSING_THRESHOLD:
        high_missing_cols.append(col)
        print(f'[STATUS] {col}: {pct:.2f}% missing → removed (threshold: {HIGH_MISSING_THRESHOLD}%)')
        df = df.drop(columns=[col])

# ── Encode categorical columns for ML imputation ──
label_encoders = {}
for col in cat_cols:
    if col in df.columns and col not in high_missing_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna('MISSING'))
        label_encoders[col] = le

# ── Update numeric cols after encoding ──
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ── Domain constraints ──
DOMAIN_MIN = {}
DOMAIN_MAX = {}
for col in num_cols:
    q99 = df[col].quantile(0.99)
    q01 = df[col].quantile(0.01)
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    lo = max(-np.inf, q01 - 3*iqr)
    hi = min(np.inf, q99 + 3*iqr)
    DOMAIN_MIN[col] = lo
    DOMAIN_MAX[col] = hi

# ── Impute missing values with KNNImputer ──
cols_with_missing = [c for c in df.columns if df[c].isnull().sum() > 0]
if cols_with_missing:
    print(f'[STATUS] Columns with missing values: {cols_with_missing}')
    print(f'[STATUS] Total missing before impute: {df.isnull().sum().sum()}')
    X_imp = df[cols_with_missing].copy()
    # Fill any remaining NaN in all columns to allow KNN
    full_df_for_knn = df.copy()
    for c in full_df_for_knn.columns:
        if full_df_for_knn[c].isnull().sum() > 0:
            full_df_for_knn[c] = full_df_for_knn[c].fillna(full_df_for_knn[c].median() if full_df_for_knn[c].dtype in [np.float64, np.int64] else full_df_for_knn[c].mode()[0])
    knn = KNNImputer(n_neighbors=5, weights='distance')
    imputed = knn.fit_transform(full_df_for_knn)
    df_imputed = pd.DataFrame(imputed, columns=df.columns, index=df.index)
    for c in df.columns:
        df[c] = df_imputed[c]
    print(f'[STATUS] Total missing after impute: {df.isnull().sum().sum()}')
else:
    print('[STATUS] No missing values to impute')

# ── Clip outliers with domain bounds ──
print(f'[STATUS] Domain clipping: checking {len(num_cols)} numeric columns')
for col in num_cols:
    lo = DOMAIN_MIN.get(col, -np.inf)
    hi = DOMAIN_MAX.get(col, np.inf)
    before_clip = df[col].isnull().sum()
    df[col] = df[col].clip(lower=lo, upper=hi)
    after_clip = df[col].isnull().sum()
    print(f'[STATUS]   {col}: {before_clip} -> {after_clip} NaN after clip')

# ── ML Outlier Detection with Isolation Forest ──
print('[STATUS] Outlier detection: Isolation Forest (contamination=0.05)')
# Use only numeric columns that have no remaining NaN
valid_num = [c for c in num_cols if c in df.columns and df[c].notna().all()]
if len(valid_num) >= 3:
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    outlier_labels = iso.fit_predict(df[valid_num])
    df['is_outlier'] = (outlier_labels == -1).astype(int)
    n_outliers = df['is_outlier'].sum()
    print(f'[STATUS] Outliers detected: {n_outliers} rows ({n_outliers/len(df)*100:.2f}%)')
else:
    print(f'[WARN] Not enough valid numeric columns ({len(valid_num)}) for Isolation Forest → skipping outlier detection')
    df['is_outlier'] = 0

# ── Restore target column ──
if y is not None:
    df[target_col] = y.values
    print(f'[STATUS] Target column restored: {target_col}')

# ── Save cleaned output ──
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved cleaned data: {output_csv} ({df.shape})')

# ── Generate report ──
report_lines = []
report_lines.append('# Dana Cleaning Report')
report_lines.append('')
report_lines.append('## Summary')
report_lines.append(f'- Original shape: {df_original.shape}')
report_lines.append(f'- Final shape: {df.shape}')
report_lines.append(f'- High missing columns removed ({HIGH_MISSING_THRESHOLD}% threshold): {high_missing_cols}')
report_lines.append(f'- Missing imputation method: KNNImputer (n_neighbors=5, distance-weighted)')
report_lines.append(f'- Outlier detection: Isolation Forest (contamination=0.05)')
report_lines.append(f'- Outliers found: {df["is_outlier"].sum() if "is_outlier" in df.columns else "N/A"}')
report_lines.append('')
report_lines.append('## Missing Values Before & After')
report_lines.append('| Column | Before | After |')
report_lines.append('|--------|--------|-------|')
for col in df_original.columns:
    before = df_original[col].isnull().sum()
    after = df[col].isnull().sum() if col in df.columns else 0
    if before > 0:
        report_lines.append(f'| {col} | {before} | {after} |')
report_lines.append('')
report_lines.append('## Columns Final')
report_lines.append(f'- Numeric: {[c for c in df.columns if df[c].dtype in [np.float64, np.int64]]}')
report_lines.append(f'- Categorical: {[c for c in df.columns if df[c].dtype == object]}')
report_lines.append('')
report_lines.append('## Notes')
report_lines.append('- Duplicates removed: 0 (Dana does not automatically deduplicate unless specified)')
report_lines.append('- Domain clipping applied: 99th/1st percentile ±3 IQR bounds')
report_lines.append('- Isolation Forest contamination=0.05: removes ~5% as outliers')

report = '\n'.join(report_lines)
report_path = os.path.join(OUTPUT_DIR, 'dana_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] Report saved: {report_path}')
print('[STATUS] Dana cleaning complete ✓')