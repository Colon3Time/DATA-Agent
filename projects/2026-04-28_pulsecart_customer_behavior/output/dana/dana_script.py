import argparse, os, pandas as pd, numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from pathlib import Path
import json
import warnings; warnings.filterwarnings('ignore')

# ── STEP 1: Parse arguments and load data ──
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\scout\scout_output.csv')
parser.add_argument('--output-dir', default=r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\dana')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape} from {INPUT_PATH}')
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Dtypes:\n{df.dtypes}')
print(f'[STATUS] First 5 rows:\n{df.head()}')

df_original = df.copy()

# ── STEP 2: Data Exploration ──
print(f'[STATUS] Missing count:\n{df.isnull().sum()}')
print(f'[STATUS] Missing %:\n{df.isnull().mean()*100}')
print(f'[STATUS] Numeric stats:\n{df.describe()}')

# ── STEP 3: Data type detection and conversion ──
# Check for date columns
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f'[STATUS] Converted {col} to datetime')

# ── STEP 4: Identify numeric and categorical columns ──
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f'[STATUS] Numeric columns ({len(num_cols)}): {num_cols}')
print(f'[STATUS] Categorical columns ({len(cat_cols)}): {cat_cols}')

# ── STEP 5: Analyze column contents for better decisions ──
for col in df.columns:
    if df[col].dtype == 'object':
        print(f'[STATUS] Unique values in {col}: {df[col].nunique(dropna=False)}')
        if df[col].nunique(dropna=False) <= 10:
            print(f'  Values: {df[col].value_counts(dropna=False).to_dict()}')

# ── STEP 6: Check for zeros that should be NaN ──
zero_cols = []
for c in num_cols:
    zero_count = (df[c] == 0).sum()
    if zero_count > 0:
        zero_cols.append(c)
        print(f'[STATUS] {c}: {zero_count} zeros ({zero_count/len(df)*100:.1f}%)')

# ── STEP 7: Domain-aware cleaning ──
# Detect domain based on column names
domain_bounds = {}

# Check for e-commerce / customer behavior dataset
if any(kw in col.lower() for col in df.columns for kw in ['price', 'revenue', 'amount', 'spend', 'total', 'cost']):
    print('[STATUS] Detected e-commerce/business dataset')
    for col in num_cols:
        lo = 0
        p99 = df[col].quantile(0.99)
        # Reasonable upper bound: 99th percentile * 3 or absolute domain max
        if any(kw in col.lower() for kw in ['quantity', 'qty', 'count', 'items']):
            hi = max(p99 * 3, 1000)
        elif any(kw in col.lower() for kw in ['price', 'revenue', 'amount', 'spend', 'total', 'cost']):
            hi = max(p99 * 3, 1000000)
        elif any(kw in col.lower() for kw in ['age', 'year']):
            hi = 120
        elif any(kw in col.lower() for kw in ['score', 'rating', 'rate', 'percent']):
            hi = max(p99 * 1.5, 100)
        else:
            hi = max(p99 * 3, 10000)
        domain_bounds[col] = (lo, hi)
        # Check column for unrealistic zeros
        if (df[col] == 0).sum() > 0.3 * len(df):
            # 30%+ zeros in non-count column might be missing
            if not any(kw in col.lower() for kw in ['flag', 'indicator', 'binary', 'id', 'count']):
                print(f'[STATUS] {col}: { (df[col] == 0).sum() } zeros ({ (df[col] == 0).sum()/len(df)*100:.0f}%)')
elif any(kw in col.lower() for col in df.columns for kw in ['glucose', 'blood', 'bmi', 'insulin', 'patient', 'diagnosis']):
    print('[STATUS] Detected medical dataset')
    DOMAIN_MIN = {'Glucose':0,'BloodPressure':0,'SkinThickness':0,'Insulin':0,'BMI':0,'Pregnancies':0,'Age':0}
    DOMAIN_MAX = {'Glucose':300,'BloodPressure':200,'SkinThickness':80,'Insulin':500,'BMI':70,'DiabetesPedigreeFunction':2.5}
    for col in num_cols:
        lo = DOMAIN_MIN.get(col, 0)
        hi = DOMAIN_MAX.get(col, df[col].quantile(0.99) * 3)
        domain_bounds[col] = (lo, hi)
else:
    print('[STATUS] Generic dataset — using percentile-based bounds')
    for col in num_cols:
        lo = max(0, df[col].quantile(0.01) - 3*df[col].std())
        p99 = df[col].quantile(0.99)
        hi = min(p99 * 3, df[col].quantile(0.999) * 1.5)
        domain_bounds[col] = (max(0, lo), max(hi, p99 * 2))

print(f'[STATUS] Domain bounds: {domain_bounds}')

# ── STEP 8: Handle zeros as missing (for specific columns where zero is invalid) ──
ZERO_INVALID_COLS = []
for col in num_cols:
    lo_bound = domain_bounds.get(col, (0, float('inf')))[0]
    # If lower bound is > 0 AND column genuinely shouldn't have zeros
    if lo_bound > 0:
        ZERO_INVALID_COLS.append(col)
    elif any(kw in col.lower() for kw in ['blood', 'glucose', 'bmi', 'insulin', 'skin']) and col in num_cols:
        ZERO_INVALID_COLS.append(col)
    # For e-commerce: price=0 is suspicious, quantity=0 might be valid
    elif any(kw in col.lower() for kw in ['price', 'revenue', 'amount', 'cost']) and col in num_cols:
        ZERO_INVALID_COLS.append(col)

print(f'[STATUS] Columns treated as zero=invalid: {ZERO_INVALID_COLS}')

for col in ZERO_INVALID_COLS:
    n = (df[col] == 0).sum()
    if n > 0:
        df[col] = df[col].replace(0, np.nan)
        print(f'[STATUS] {col}: {n} zeros -> NaN')

# ── STEP 9: Missing Imputation ──
missing_pct = df[num_cols].isnull().mean()
total_missing_cells = df[num_cols].isnull().sum().sum()
total_cells = len(df) * len(num_cols)
overall_missing_pct = total_missing_cells / total_cells * 100
print(f'[STATUS] Overall missing in numeric cols: {overall_missing_pct:.2f}%')

if total_missing_cells > 0:
    if overall_missing_pct <= 5:
        # Simple median imputation
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        print(f'[STATUS] Simple median imputation used (missing <= 5%)')
    else:
        # KNN Imputation for moderate missing
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols, index=df.index)
        print(f'[STATUS] KNNImputer (k=5) used (missing > 5%)')

    # ── Post-imputation clip ──
    for col, (lo, hi) in domain_bounds.items():
        if col in df.columns:
            before = df[col].min(), df[col].max()
            df[col] = df[col].clip(lower=lo, upper=hi)
            print(f'[STATUS] {col} clipped: [{before[0]:.2f}, {before[1]:.2f}] -> [{df[col].min():.2f}, {df[col].max():.2f}]')

# ── STEP 10: Handle categorical missing ──
for col in cat_cols:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        # If it has high cardinality, use "Unknown"
        if df[col].nunique() > 20:
            df[col] = df[col].fillna('Unknown')
            print(f'[STATUS] {col}: {n_miss} missing -> "Unknown" (high cardinality)')
        else:
            # Use mode for low cardinality
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            print(f'[STATUS] {col}: {n_miss} missing -> "{mode_val}" (mode)')

# ── STEP 11: Outlier Detection ──
# Only use numeric columns that are actual features (not IDs, strings, etc.)
feat_cols = [c for c in num_cols if not any(kw in c.lower() for kw in ['id', 'index', 'row', 'flag', 'outlier'])]
outlier_records = []

# Check if we have enough features
if len(feat_cols) >= 2:
    # IQR Method per column
    for col in feat_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo_b = q1 - 1.5*iqr
        hi_b = q3 + 1.5*iqr
        domain_lo, domain_hi = domain_bounds.get(col, (-np.inf, np.inf))
        
        outlier_mask = (df[col] < lo_b) | (df[col] > hi_b)
        for idx in df[outlier_mask].index:
            val = df.loc[idx, col]
            # Determine if Likely Error or Likely Real
            if val < domain_lo or val > domain_hi:
                verdict, action = 'Likely Error', 'capped'
                # Cap to domain bounds
                df.loc[idx, col] = df[col].median()
            elif val < 0:
                verdict, action = 'Likely Error', 'capped'
                df.loc[idx, col] = df[col].median()
            else:
                # Check if extremely extreme (beyond 3*IQR or 5*std)
                if val < q1 - 3*iqr or val > q3 + 3*iqr:
                    if val > domain_hi * 0.8:
                        verdict, action = 'Uncertain', 'flagged'
                    else:
                        verdict, action = 'Likely Real', 'flagged'
                else:
                    verdict, action = 'Likely Real', 'flagged'
            
            outlier_records.append({
                'row_index': int(idx),
                'column_name': col,
                'value': float(val) if not pd.isna(val) else None,
                'verdict': verdict,
                'reason': f'{col}={val:.2f} (IQR bounds: {lo_b:.2f}-{hi_b:.2f})',
                'action': action
            })
            print(f'[STATUS] Outlier: row={idx}, col={col}, val={val:.2f} -> {verdict}')

    # Isolation Forest (multivariate)
    if len(feat_cols) >= 3:
        try:
            iso = IsolationForest(contamination='auto', random_state=42)
            iso.fit(df[feat_cols])
            iso_pred = iso.predict(df[feat_cols])
            iso_scores = iso.score_samples(df[feat_cols])
            threshold = np.percentile(iso_scores, 5)
            
            for idx, (pred, score) in enumerate(zip(iso_pred, iso_scores)):
                if pred == -1 and score < threshold:
                    # Check if not already flagged
                    already = any(r['row_index'] == idx for r in outlier_records)
                    if not already:
                        outlier_records.append({
                            'row_index': int(idx),
                            'column_name': 'multivariate',
                            'value': float(score),
                            'verdict': 'Uncertain',
                            'reason': f'Isolation Forest anomaly score={score:.4f}',
                            'action': 'flagged'
                        })
                        print(f'[STATUS] Outlier: row={idx}, col=multivariate, score={score:.4f} -> Uncertain')
        except Exception as e:
            print(f'[WARN] Isolation Forest failed: {e}')

# ── STEP 12: Add outlier flag column ──
df['is_outlier'] = 0
for r in outlier_records:
    if r['verdict'] != 'Likely Error':
        df.loc[r['row_index'], 'is_outlier'] = 1

# ── STEP 13: Save outputs ──
# Main output CSV
output_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# Outlier flags CSV
if outlier_records:
    flags_df = pd.DataFrame(outlier_records)
    flags_csv = os.path.join(OUTPUT_DIR, 'outlier_flags.csv')
    flags_df.to_csv(flags_csv, index=False)
    print(f'[STATUS] Saved: {flags_csv} (flags: {len(outlier_records)} rows)')

# ── STEP 14: Calculate quality scores ──
n_total = len(df)
n_cols = len(df.columns) - 1  # exclude is_outlier column in count

# Completeness
missing_before = df_original.isnull().sum().sum()
missing_after = df.drop(columns=['is_outlier']).isnull().sum().sum()
completeness_before = (1 - missing_before / (len(df_original) * len(df_original.columns))) * 100
completeness_after = (1 - missing_after / (n_total * n_cols)) * 100

# Validity: ratio of non-Likely-Error rows
likely_error_count = sum(1 for r in outlier_records if r['verdict'] == 'Likely Error')
validity_before = 100  # Assume all valid before (no info)
validity_after = (1 - likely_error_count / max(n_total, 1)) * 100

# Overall
overall_before = (completeness_before + 100) / 2  # Assume 100% validity before
overall_after = 0.5 * completeness_after + 0.5 * validity_after

print(f'[STATUS] Quality: Completeness {completeness_before:.1f}% -> {completeness_after:.1f}%')
print(f'[STATUS] Quality: Validity 100.0% -> {validity_after:.1f}%')
print(f'[STATUS] Quality: Overall {overall_before:.1f}% -> {overall_after:.1f}%')

# ── STEP 15: Generate report ──
report_lines = []
report_lines.append('Dana Cleaning Report')
report_lines.append('===================')
report_lines.append('')
report_lines.append(f'Input: {INPUT_PATH}')
report_lines.append('')
report_lines.append(f'Before: {len(df_original)} rows, {len(df_original.columns)} columns')
report_lines.append(f'After:  {len(df)} rows, {len(df.columns)} columns')
report_lines.append('')

# Missing section
report_lines.append('Missing Values:')
miss_any = False
for col in df_original.columns:
    pct = df_original[col].isnull().mean() * 100
    if pct > 0:
        miss_any = True
        method = 'median' if overall_missing_pct <= 5 else 'KNNImputer(k=5)'
        report_lines.append(f'- {col}: {pct:.1f}% missing -> {method} imputation')
for col in df_original.columns:
    if df_original[col].isnull().mean() == 0:
        report_lines.append(f'- {col}: 0% missing -> no action needed')
if not miss_any:
    report_lines.append('- No missing values detected')
report_lines.append('')

# Zero as missing
if ZERO_INVALID_COLS:
    report_lines.append('Zero-as-Missing Conversion:')
    for col in ZERO_INVALID_COLS:
        n_zero = (df_original[col] == 0).sum()
        if n_zero > 0:
            report_lines.append(f'- {col}: {n_zero} zeros ({n_zero/len(df_original)*100:.1f}%) converted to NaN before imputation')
    report_lines.append('')

# Outlier section
report_lines.append('Outlier Detection:')
likely_errors = [r for r in outlier_records if r['verdict'] == 'Likely Error']
likely_reals = [r for r in outlier_records if r['verdict'] == 'Likely Real']
uncertains = [r for r in outlier_records if r['verdict'] == 'Uncertain']

if likely_errors:
    report_lines.append(f'- Method: IQR (1.5x) + Isolation Forest')
    report_lines.append(f'- Likely Error (corrected):')
    # Group by column
    by_col = {}
    for r in likely_errors:
        by_col.setdefault(r['column_name'], []).append(r)
    for col, items in by_col.items():
        report_lines.append(f'  - {col}: {len(items)} rows -> capped/median imputed')
else:
    report_lines.append('- Likely Error: None')

if likely_reals:
    report_lines.append(f'- Likely Real (flagged, preserved):')
    by_col = {}
    for r in likely_reals:
        by_col.setdefault(r['column_name'], []).append(r)
    for col, items in by_col.items():
        report_lines.append(f'  - {col}: {len(items)} rows -> is_outlier=1')
else:
    report_lines.append('- Likely Real: None')

if uncertains:
    report_lines.append(f'- Uncertain (flagged):')
    for r in uncertains:
        report_lines.append(f'  - row {r["row_index"]}: {r["column_name"]} ({r["value"]}) -> {r["reason"][:60]}')
else:
    report_lines.append('- Uncertain: None')

if not outlier_records:
    report_lines.append('- Outliers: 0 rows across all columns — data is clean')

report_lines.append(f'- outlier_flags.csv: {len(outlier_records)} rows (details in report)')
if outlier_records:
    report_lines.append(f'  Columns: row_index, column_name, value, verdict, reason')
report_lines.append('')

# Data Quality Score
report_lines.append('Data Quality Score:')
report_lines.append(f'- Completeness: Before {completeness_before:.1f}% -> After {completeness_after:.1f}%')
report_lines.append(f'- Validity: Before 100.0% -> After {validity_after:.1f}%')
report_lines.append(f'- Overall: Before {overall_before:.1f}% -> After {overall_after:.1f}%')
report_lines.append('')

# Column Stats
changed_cols = []
for col in num_cols[:min(10, len(num_cols))]:
    before_mean = df_original[col].mean()
    after_mean = df[col].mean()
    before_std = df_original[col].std()
    after_std = df[col].std()
    if abs(before_mean - after_mean) > 0.001 or abs(before_std - after_std) > 0.001:
        changed_cols.append((col, before_mean, after_mean, before_std, after_std))

if changed_cols:
    report_lines.append('Column Stats (Before -> After):')
    for col, bm, am, bs, as_ in changed_cols:
        report_lines.append(f'- {col}: mean {bm:.2f}->{am:.2f}, std {bs:.2f}->{as_:.2f}')
    report_lines.append('')

# New Method
report_lines.append('New Method Found:')
report_lines.append('None')
report_lines.append('')

report_lines.append('Zero-as-Missing Conversion:')
for col in ZERO_INVALID_COLS:
    n_zero = (df_original[col] == 0).sum()
    if n_zero > 0:
        report_lines.append(f'- {col}: {n_zero} zeros ({n_zero/len(df_original)*100:.1f}%) converted to NaN before imputation')
    else:
        report_lines.append(f'- {col}: 0 zeros found')

# ── Save report ──
report_text = '\n'.join(report_lines)
report_path = os.path.join(OUTPUT_DIR, 'dana_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f'[STATUS] Saved report: {report_path}')
print(report_text)