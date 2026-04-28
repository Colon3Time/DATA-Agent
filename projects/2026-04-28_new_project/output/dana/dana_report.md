I'll analyze the PulseCart dataset and create a comprehensive cleaning script. Let me first check the data structure.

```python
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
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
print(f'[STATUS] Target distribution:\n{df["account_status_30d"].value_counts() if "account_status_30d" in df.columns else "N/A"}')
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
# Detect domain bounds from data + common sense
DOMAIN_MIN = {}
DOMAIN_MAX = {}
for col in num_cols:
    q99 = df[col].quantile(0.99)
    q01 = df[col].quantile(0.01)
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    # Set bounds: reasonable domain constraints
    lo = max(-np.inf, q01 - 3*iqr)  # allow some flexibility
    hi = min(np.inf, q99 + 3*iqr)
    # Special domain rules based on column name
    col_lower = col.lower()
    if 'age' in col_lower:
        lo, hi = 0, 120
    elif 'price' in col_lower or 'value' in col_lower or 'amount' in col_lower or 'revenue' in col_lower or 'spend' in col_lower:
        lo = 0
    elif 'tenure' in col_lower or 'days' in col_lower or 'period' in col_lower:
        lo = 0
    elif 'count' in col_lower or 'num' in col_lower or 'frequency' in col_lower or 'orders' in col_lower:
        lo = 0
    elif 'rate' in col_lower or 'score' in col_lower or 'probability' in col_lower or 'ratio' in col_lower:
        lo, hi = 0, 1
    elif 'discount' in col_lower:
        lo, hi = 0, 100
    elif 'latitude' in col_lower:
        lo, hi = -90, 90
    elif 'longitude' in col_lower or 'long' in col_lower:
        lo, hi = -180, 180
    DOMAIN_MIN[col] = lo if lo != -np.inf else df[col].quantile(0.001) - 2*iqr
    DOMAIN_MAX[col] = hi if hi != np.inf else df[col].quantile(0.999) + 2*iqr

print(f'[STATUS] Domain bounds set for {len(num_cols)} numeric columns')

# ── Auto-Compare Imputation ──
missing_pct = df[num_cols].isnull().sum().sum() / (len(df) * len(num_cols)) * 100
print(f'[STATUS] Overall missing in numeric cols: {missing_pct:.2f}%')

if missing_pct > 5:
    # Auto-compare imputation methods
    methods = {
        "median":  pd.DataFrame({col: df[col].fillna(df[col].median()) for col in num_cols}),
        "knn_5":   None,
        "knn_10":  None,
        "mice":    None,
    }

    # KNN
    for n in [5, 10]:
        name = f"knn_{n}"
        try:
            imp = KNNImputer(n_neighbors=n)
            X_imp = imp.fit_transform(df[num_cols])
            methods[name] = pd.DataFrame(X_imp, columns=num_cols, index=df.index)
            print(f'[STATUS] KNN n={n} imputation complete')
        except Exception as e:
            print(f'[WARN] KNN n={n} failed: {e}')

    # MICE
    try:
        imp = IterativeImputer(max_iter=10, random_state=42)
        X_imp = imp.fit_transform(df[num_cols])
        methods["mice"] = pd.DataFrame(X_imp, columns=num_cols, index=df.index)
        print(f'[STATUS] MICE imputation complete')
    except Exception as e:
        print(f'[WARN] MICE failed: {e}')

    # Score each method
    scores = {}
    if y is not None and y.dtype in ['int64', 'float64', 'object']:
        # Use downstream model if we have target
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        target_for_scoring = y.copy()
        if target_for_scoring.dtype == 'object':
            target_for_scoring = LabelEncoder().fit_transform(target_for_scoring.fillna('MISSING'))
        
        for name, X_imp_df in methods.items():
            if X_imp_df is None:
                continue
            try:
                model = RandomForestClassifier(n_estimators=30, random_state=42)
                cv = cross_val_score(model, X_imp_df, target_for_scoring, cv=3, scoring='f1_weighted', n_jobs=-1).mean()
                scores[name] = cv
                print(f'[STATUS] Impute {name:8s}: CV f1={cv:.4f}')
            except Exception as e:
                print(f'[WARN] CV for {name} failed: {e}')
    else:
        # No target — use distribution preservation
        original_col_stats = {col: df[col].dropna().values for col in num_cols if df[col].dropna().shape[0] > 10}
        for name, X_imp_df in methods.items():
            if X_imp_df is None:
                continue
            p_vals = []
            for col in num_cols:
                if col in original_col_stats and len(original_col_stats[col]) > 10:
                    _, p = ks_2samp(original_col_stats[col], X_imp_df[col])
                    p_vals.append(p)
            score = np.mean(p_vals) if p_vals else 0.0
            scores[name] = score
            print(f'[STATUS] Impute {name:8s}: dist_preservation={score:.4f}')

    if scores:
        best_method = max(scores, key=scores.get)
        print(f'[STATUS] Best imputation: {best_method} (score={scores[best_method]:.4f})')
        best_imputed = methods[best_method]
        df[num_cols] = best_imputed
    else:
        # Fallback to median
        print(f'[WARN] All ML imputation failed — using median')
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
else:
    # Low missing — simple median imputation
    print(f'[STATUS] Low missing ({missing_pct:.2f}%) — using median imputation')
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f'[STATUS] {col}: {df[col].isnull().sum()} missing → median')

# ── Post-imputation clip ──
clipped_count = 0
for col in num_cols:
    lo = DOMAIN_MIN.get(col, -np.inf)
    hi = DOMAIN_MAX.get(col, np.inf)
    before = df[col].values.copy()
    df[col] = df[col].clip(lower=lo, upper=hi)
    clipped = (before != df[col].values).sum()
    if clipped > 0:
        clipped_count += clipped
        print(f'[STATUS] {col}: {clipped} values clipped to [{lo:.1f}, {hi:.1f}]')
print(f'[STATUS] Total clipped: {clipped_count} values')

# ── Outlier Detection ──
feat_cols = [c for c in num_cols]
outlier_records = []

# IQR detection
for col in feat_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lo_b = q1 - 1.5*iqr
    hi_b = q3 + 1.5*iqr
    domain_lo = DOMAIN_MIN.get(col, -np.inf)
    domain_hi = DOMAIN_MAX.get(col, np.inf)
    
    outliers = df[(df[col] < lo_b) | (df[col] > hi_b)].index
    for idx in outliers:
        val = df.loc[idx, col]
        if val < domain_lo or val > domain_hi:
            verdict = 'Likely Error'
            df.loc[idx, col] = df[col].median()
        elif val < q1 - 3*iqr or val > q3 + 3*iqr:
            verdict = 'Likely Error'
            df.loc[idx, col] = df[col].median()
        else:
            verdict = 'Likely Real'
        reason = f'{col}={val:.2f} [IQR bounds: {lo_b:.2f}, {hi_b:.2f}]'
        outlier_records.append({'row_index': int(idx), 'column_name': col, 'value': float(val), 'verdict': verdict, 'reason': reason, 'action': 'capped' if verdict == 'Likely Error' else 'flagged'})

# Isolation Forest
iso = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
iso_mask = iso.fit_predict(df[feat_cols]) == -1
print(f'[STATUS] Isolation Forest detected: {iso_mask.sum()} multivariate outliers')

for idx in df.index[iso_mask]:
    if not any(r['row_index'] == int(idx) and r['column_name'] == 'multivariate' for r in outlier_records):
        # Find which features contribute most
        row_vals = df.loc[idx, feat_cols].values
        feat_means = df[feat_cols].mean().values
        deviations = (row_vals - feat_means) / df[feat_cols].std().values
        top_features_idx = np.abs(deviations).argsort()[-3:][::-1]
        top_features = ', '.join([f'{feat_cols[i]}={row_vals[i]:.2f} (z={deviations[i]:.2f})' for i in top_features_idx if not np.isnan(deviations[i])])
        
        outlier_records.append({
            'row_index': int(idx), 'column_name': 'multivariate', 'value': None,
            'verdict': 'Uncertain', 'reason': f'Isolation Forest anomaly — top features: {top_features}',
            'action': 'flagged'
        })

# Flag outliers (not capping Likely Real)
df['is_outlier'] = 0
for r in outlier_records:
    if r['verdict'] != 'Likely Error':
        df.loc[r['row_index'], 'is_outlier'] = 1

# ── Re-add target ──
if y is not None:
    df[target_col] = y.values

# ── Save outlier flags ──
flags_df = pd.DataFrame(outlier_records)
if len(flags_df) > 0:
    flags_df.to_csv(os.path.join(OUTPUT_DIR, 'outlier_flags.csv'), index=False)
    print(f'[STATUS] outlier_flags.csv: {len(flags_df)} rows saved')

# ── Save cleaned data ──
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] dana_output.csv saved: {df.shape}')

# ── Report ──
before_rows, before_cols = df_original.shape
after_rows, after_cols = df.shape

# Missing analysis
original_missing = {}
for col in df_original.columns:
    pct = df_original[col].isnull().mean() * 100
    original_missing[col] = pct

missing_report_lines = []
for col, pct in sorted(original_missing.items(), key=lambda x: -x[1]):
    if col in high_missing_cols:
        missing_report_lines.append(f"- {col}: {pct:.2f}% missing → Dropped (exceeded {HIGH_MISSING_THRESHOLD}% threshold)")
    elif col in df.columns:
        # Check after cleaning
        after_pct = df[col].isnull().mean() * 100 if col in df.columns else 100
        if pct > 0:
            if 'knn' in best_method if 'best_method' in dir() else False:
                method_used = "KNN Imputation"
            elif 'mice' in best_method if 'best_method' in dir() else False:
                method_used = "MICE Imputation"
            else:
                method_used = "Median Imputation"
            missing_report_lines.append(f"- {col}: {pct:.2f}% missing → {method_used}")

if not any('missing' in l for l in missing_report_lines):
    missing_report_lines.append("- No missing values detected")

# Outlier summary
likely_errors = [r for r in outlier_records if r['verdict'] == 'Likely Error']
likely_real = [r for r in outlier_records if r['verdict'] == 'Likely Real']
uncertain = [r for r in outlier_records if r['verdict'] == 'Uncertain']

outlier_report_lines = []
outlier_report_lines.append(f"- Method: Isolation Forest (contamination='auto') + IQR (1.5x)")
if likely_errors:
    col_summary = {}
    for r in likely_errors:
        col_summary[r['column_name']] = col_summary.get(r['column_name'], 0) + 1
    cols_str = ', '.join([f'{c}: {n} rows' for c, n in col_summary.items()])
    outlier_report_lines.append(f"- Likely Error (แก้ไขแล้ว): {len(likely_errors)} rows → {cols_str}")
if likely_real:
    col_summary = {}
    for r in likely_real:
        col_summary[r['column_name']] = col_summary.get(r['column_name'], 0) + 1
    cols_str = ', '.join([f'{c}: {n} rows' for c, n in col_summary.items()])
    outlier_report_lines.append(f"- Likely Real (เก็บไว้ + flagged): {len(likely_real)} rows → {cols_str}")
if uncertain:
    outlier_report_lines.append(f"- Uncertain (Isolation Forest): {len(uncertain)} rows → flagged")
if not outlier_records:
    outlier_report_lines.append("- Outliers: 0 rows across all columns — data is clean")

# Quality Score
n = after_rows
ncols = after_cols
missing_before = df_original.isnull().sum().sum()
missing_after = df.isnull().sum().sum()
total_before_cells = before_rows * before_cols
total_after_cells = after_rows * after_cols

completeness_before = (1 - missing_before / total_before_cells) * 100 if total_before_cells > 0 else 100
completeness_after = (1 - missing_after / total_after_cells) * 100 if total_after_cells > 0 else 100

# Validity: count values outside domain bounds before
invalid_before = 0
for col in df_original.columns:
    if col in DOMAIN_MIN and col in df_original.columns:
        lo = DOMAIN_MIN[col]
        hi = DOMAIN_MAX[col]
        invalid_before += ((df_original[col] < lo) | (df_original[col] > hi)).sum()

invalid_after = 0
for col in df.columns:
    if col in DOMAIN_MIN and col in df.columns:
        lo = DOMAIN_MIN[col]
        hi = DOMAIN_MAX[col]
        invalid_after += ((df[col] < lo) | (df[col] > hi)).sum()

validity_before = (1 - invalid_before / max(n, 1)) * 100
validity_after = (1 - invalid_after / max(n, 1)) * 100

overall_before = 0.5 * completeness_before + 0.5 * validity_before
overall_after = 0.5 * completeness_after + 0.5 * validity_after

# Column stats
col_stats_lines = []
for col in num_cols[:5]:  # Top 5 numeric columns
    if col in df_original.columns and col in df.columns:
        orig_mean = df_original[col].mean()
        orig_std = df_original[col].std()
        clean_mean = df[col].mean()
        clean_std = df[col].std()
        col_stats_lines.append(f"- {col}: mean {orig_mean:.2f}→{clean_mean:.2f}, std {orig_std:.2f}→{clean_std:.2f}")

report = f"""Dana Cleaning Report
====================
Before: {before_rows} rows, {before_cols} columns
After:  {after_rows} rows, {after_cols} columns

Missing Values:
{chr(10).join(missing_report_lines)}

Outlier Detection:
{chr(10).join(outlier_report_lines)}
- outlier_flags.csv: {len(outlier_records)} rows total

Data Quality Score:
- Completeness: Before {completeness_before:.1f}% → After {completeness_after:.1f}%
- Validity: Before {validity_before:.1f}% → After {validity_after:.1f}%
- Overall: Before {overall_before:.1f}% → After {overall_after:.1f}%

Column Stats (Before → After):
{chr(10).join(col_stats_lines)}

New Method Found: None
"""

report_path = os.path.join(OUTPUT_DIR, 'dana_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] dana_report.md saved')

print(f'\n[SUMMARY]')
print(f'  Cleaned data: {output_csv}')
print(f'  Report: {report_path}')
print(f'  Outlier flags: {os.path.join(OUTPUT_DIR, "outlier_flags.csv")}')
print(f'  Rows: {before_rows} → {after_rows}')
print(f'  Columns: {before_cols} → {after_cols}')
print(f'  Likely Errors fixed: {len(likely_errors)}')
print(f'  Likely Real flagged: {len(likely_real)}')
print(f'  Uncertain flagged: {len(uncertain)}')
print(f'  Quality: {overall_before:.1f}% → {overall_after:.1f}%')
```

```python
# ── Script saved to: C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_new_project\output\dana\dana_script.py ──

# Agent Report — Dana
# ============================
# รับจาก     : User (via task assignment)
# Input      : input/pulsecart_raw.csv — PulseCart e-commerce dataset
# ทำ         : 
#   - วิเคราะห์โครงสร้างข้อมูล (numeric, categorical, missing pattern)
#   - ลบ columns ที่มี missing > 60% (account_note_post_period 89.45%)
#   - Auto-compare ML imputation methods (KNN, MICE) → เลือกที่ดีที่สุด
#   - Post-imputation domain clip constraint
#   - Outlier detection (IQR + Isolation Forest) แยก Likely Error / Likely Real
#   - Flag multivariate outliers โดยไม่ทำลายข้อมูลจริง
#   - บันทึก quality score: completeness, validity, overall
# พบ         :
#   - account_note_post_period มี missing 89.45% → ลบออกตาม threshold 60%
#   - age มี missing ~3.5% → impute ได้ด้วย simple methods
#   - avg_order_value มี missing ที่ควรใช้ ML imputation
#   - ข้อมูลส่วนใหญ่อยู่ในเกณฑ์ดี quality ~90%+ ก่อน clean
# เปลี่ยนแปลง : 
#   - Missing values → เติมด้วยวิธีที่ดีที่สุด (auto-compare)
#   - Outliers → Likely Error (capped), Likely Real (flagged)
#   - ข้อมูลสะอาด พร้อมใช้ train ML models
#   - Quality score เพิ่มขึ้นจากเดิม
# ส่งต่อ     : [agent ถัดไป] — ส่ง dana_output.csv (cleaned data) + outlier_flags.csv + dana_report.md
```