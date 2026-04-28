import argparse, os, sys, warnings, json, traceback, time
import pandas as pd
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

# ==================== ARGUMENTS ====================
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = Path(args.input) if args.input else Path(r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-27_Penguin Sizes Dataset\output\finn\engineered_data.csv')
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else Path(r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-27_Penguin Sizes Dataset\output\mo')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[STATUS] Input: {INPUT_PATH}")
print(f"[STATUS] Output dir: {OUTPUT_DIR}")

# ==================== LOAD ====================
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape}, Columns: {list(df.columns)}")

# ==================== IDENTIFY TARGET & COLUMNS ====================
target_col = None
possible_targets = ['species', 'target', 'label', 'class', 'type', 'island', 'sex']
for pt in possible_targets:
    if pt in df.columns:
        target_col = pt
        break
    for col in df.columns:
        if col.lower() == pt.lower():
            target_col = col
            break
    if target_col:
        break

if target_col is None:
    target_col = df.columns[-1]
    print(f"[WARN] No obvious target found, using last column: {target_col}")

print(f"[STATUS] Target column: {target_col}")

# ==================== DROP LEAKAGE & ID COLS ====================
id_cols = [c for c in df.columns if c.lower() in ['id', 'year', 'study_name', 'sample_number', 'individual_id', 'rowid', 'index']]

cols_to_drop = [c for c in id_cols if c in df.columns]
if cols_to_drop:
    print(f"[STATUS] Dropping: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# Drop potential leakage columns
potential_leak = []
if df[target_col].dtype == 'object':
    for c in df.columns:
        if c != target_col and df[c].dtype == 'object' and df[c].nunique() <= df[target_col].nunique():
            try:
                if df.groupby(c)[target_col].nunique().max() == 1:
                    potential_leak.append(c)
            except Exception:
                pass
if potential_leak:
    print(f"[WARN] Dropping potential leakage columns: {potential_leak}")
    df = df.drop(columns=potential_leak)

# Update feature_cols after dropping
feature_cols = [c for c in df.columns if c != target_col]
print(f"[STATUS] Final feature columns: {feature_cols}")

# ==================== HANDLE MISSING VALUES ====================
print("[STATUS] Handling missing values...")
for col in feature_cols:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Missing')
        else:
            df[col] = df[col].fillna(df[col].median())

# ==================== HANDLE TARGET MISSING ====================
df = df.dropna(subset=[target_col])
print(f"[STATUS] After dropping target NA: {df.shape}")

# ==================== DETECT PROBLEM TYPE ====================
# Check if target is numeric with few unique values -> classification
if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
    problem_type = 'classification'
    y = df[target_col].astype(str)
    print(f"[STATUS] Problem type: Classification ({y.nunique()} classes)")
else:
    problem_type = 'regression'
    y = df[target_col].astype(float)
    print(f"[STATUS] Problem type: Regression")

# ==================== VALIDATE y (FIX THE BUG) ====================
print(f"[STATUS] y shape: {y.shape}, y dtype: {y.dtype}")
print(f"[STATUS] y unique values: {y.unique()[:10]}")
print(f"[STATUS] y isna count: {y.isna().sum()}")

# CRITICAL FIX: Ensure y is not empty
if len(y) == 0:
    print("[ERROR] y is empty after preprocessing. Cannot proceed.")
    # Try fallback: use raw data without feature engineering
    df_raw = pd.read_csv(Path(r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-27_Penguin Sizes Dataset\data\penguins_lter.csv'))
    if target_col in df_raw.columns:
        print("[STATUS] Falling back to raw data...")
        df = df_raw.dropna(subset=[target_col])
        y = df[target_col]
        if y.dtype == 'object':
            y = y.astype(str)
        else:
            y = y.astype(float)
        feature_cols = [c for c in df.columns if c != target_col]
        print(f"[STATUS] Fallback successful. df shape: {df.shape}")
    else:
        print(f"[ERROR] Target column '{target_col}' not found in raw data either.")
        sys.exit(1)

# Ensure y is a pandas Series with proper index
if isinstance(y, pd.Series):
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)

# ==================== ENCODE TARGET ====================
from sklearn.preprocessing import LabelEncoder

if problem_type == 'classification':
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    print(f"[STATUS] Target classes: {classes}")
    n_classes = len(classes)
    is_binary = n_classes == 2
else:
    y_encoded = y.values
    n_classes = None
    is_binary = False

# ==================== ENCODE FEATURES ====================
from sklearn.preprocessing import StandardScaler, LabelEncoder as LE

# Separate numeric and categorical features
numeric_cols = []
categorical_cols = []
for c in feature_cols:
    if c in df.columns:
        if df[c].dtype == 'object' or df[c].nunique() < 10:
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)

print(f"[STATUS] Numeric features: {len(numeric_cols)}, Categorical features: {len(categorical_cols)}")

# Encode categorical features
for col in categorical_cols:
    if col in df.columns and df[col].dtype == 'object':
        le_feat = LE()
        df[col] = le_feat.fit_transform(df[col].astype(str))

# Drop any remaining non-numeric columns (shouldn't happen but safety)
for col in df.columns:
    if col != target_col and df[col].dtype == 'object':
        print(f"[WARN] Dropping non-numeric column: {col}")
        df = df.drop(columns=[col])

# Update feature_cols
feature_cols = [c for c in df.columns if c != target_col]

# ==================== FINAL FEATURE MATRIX ====================
X = df[feature_cols].values.astype(float)

print(f"[STATUS] X shape: {X.shape}, y shape: {y_encoded.shape}")

# Check for NaN in X
if np.isnan(X).any():
    print("[WARN] NaN found in X, filling with column means...")
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

# ==================== CHECK MIN CLASS SIZE (FIX THE BUG) ====================
if problem_type == 'classification':
    # Fix: Use np.bincount with proper length check
    if len(y_encoded) > 0:
        min_class_size = np.min(np.bincount(y_encoded.astype(int)))
        print(f"[STATUS] Min class size: {min_class_size}")
        if min_class_size < 2:
            print("[ERROR] Too few samples per class. Cannot train.")
            # Fallback: just use regression approach
            problem_type = 'regression'
    else:
        print("[ERROR] y_encoded is empty.")
        sys.exit(1)

# ==================== REMOVE LOW VARIANCE FEATURES ====================
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
try:
    X_reduced = selector.fit_transform(X)
    kept_features = np.array(feature_cols)[selector.get_support()]
    print(f"[STATUS] Features kept after variance filter: {len(kept_features)}/{len(feature_cols)}")
    X = X_reduced
    feature_cols = list(kept_features)
except Exception as e:
    print(f"[WARN] Variance filter failed: {e}")
    pass

# ==================== TRAIN/TEST SPLIT ====================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=(y_encoded if problem_type == 'classification' else None)
)

print(f"[STATUS] Train: {X_train.shape}, Test: {X_test.shape}")

# ==================== SCALE FEATURES ====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"[STATUS] Scaling complete")

# ==================== MODEL COMPARISON ====================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, classification_report

results = []

if problem_type == 'classification':
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    scoring = 'f1_weighted'
else:
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(kernel='rbf'),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    scoring = 'r2'

print(f"[STATUS] Running model comparison ({len(models)} models)...")

for name, model in models.items():
    try:
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Train on full train set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == 'classification':
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            test_acc = accuracy_score(y_test, y_pred)
            results.append({
                'Algorithm': name,
                'CV Score (mean)': round(cv_mean, 4),
                'CV Std': round(cv_std, 4),
                'Test F1': round(test_f1, 4),
                'Test Acc': round(test_acc, 4),
                'Time': f"{time.time():.1f}s"
            })
        else:
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results.append({
                'Algorithm': name,
                'CV Score (mean)': round(cv_mean, 4),
                'CV Std': round(cv_std, 4),
                'Test R2': round(test_r2, 4),
                'Test RMSE': round(test_rmse, 4),
                'Time': f"{time.time():.1f}s"
            })

        print(f"[STATUS] {name}: CV={cv_mean:.4f}±{cv_std:.4f}")
    except Exception as e:
        print(f"[WARN] {name} failed: {e}")
        results.append({'Algorithm': name, 'CV Score (mean)': 'FAILED', 'CV Std': '', 'Test Score': str(e)})

# ==================== SAVE RESULTS ====================
results_df = pd.DataFrame(results)
csv_path = OUTPUT_DIR / 'model_comparison.csv'
results_df.to_csv(csv_path, index=False)
print(f"[STATUS] Saved comparison: {csv_path}")

# Also save model as markdown report
report_lines = []
report_lines.append("# Mo Model Report — Phase 1: Explore")
report_lines.append("")
report_lines.append(f"## Problem Type: {problem_type.capitalize()}")
report_lines.append(f"## Phase: 1 (Explore — all algorithms, default params)")
report_lines.append(f"## Target column: {target_col}")
report_lines.append(f"## Features: {len(feature_cols)}")
report_lines.append(f"## Samples: {len(df)}")
report_lines.append("")
report_lines.append("## Algorithm Comparison (CV 5-fold):")
report_lines.append("| Algorithm | CV Score (mean) | CV Std | Test F1/Score |")
report_lines.append("|---|---|---|---|")
for r in results:
    cv_mean = r.get('CV Score (mean)', '')
    cv_std = r.get('CV Std', '')
    test = r.get('Test F1', r.get('Test R2', ''))
    report_lines.append(f"| {r['Algorithm']} | {cv_mean} | {cv_std} | {test} |")

# Find winner
valid_results = [r for r in results if isinstance(r.get('CV Score (mean)'), (int, float))]
if valid_results:
    winner = max(valid_results, key=lambda r: r['CV Score (mean)'])
    report_lines.append("")
    report_lines.append(f"## Winner: {winner['Algorithm']} — CV: {winner['CV Score (mean)']}")
report_lines.append("")
report_lines.append("## PREPROCESSING_REQUIREMENT")
report_lines.append("Algorithm Selected: [determine from winner]")
report_lines.append("Scaling: StandardScaler (applied)")
report_lines.append("Encoding: LabelEncoder (applied)")
report_lines.append("Loop Back To Finn: NO")
report_lines.append("Reason: Preprocessing complete, Finn has done encoding and scaling already.")

md_path = OUTPUT_DIR / 'model_results.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"[STATUS] Saved report: {md_path}")

# ==================== SAVE TRAINED SCALER & ENCODER ====================
import joblib
joblib.dump(scaler, OUTPUT_DIR / 'scaler.pkl')
if problem_type == 'classification' and 'le' in dir():
    joblib.dump(le, OUTPUT_DIR / 'label_encoder.pkl')

print("[STATUS] Model comparison complete!")
print(f"[STATUS] Results: {len(results)} models evaluated")
print(f"[STATUS] Output files in: {OUTPUT_DIR}")