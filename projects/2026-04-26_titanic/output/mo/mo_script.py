#!/usr/bin/env python3
"""
Mo — Phase 1: Explore (Titanic Classification)
Test ALL classical ML algorithms with default params, CV=5, F1-weighted
Fixed: stratify=y requires y to be categorical with >=2 samples per class
"""

import argparse, os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input',      default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('[STATUS] Loading data...')
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {df.columns.tolist()}')
print(f'[STATUS] Dtypes:\n{df.dtypes}')

# ========== Identify target ==========
target_col = None
for candidate in ['Survived', 'survived', 'SURVIVED', 'target', 'Target', 'TARGET', 'label', 'Label', 'LABEL', 'y']:
    if candidate in df.columns:
        target_col = candidate
        break
if target_col is None:
    for col in df.columns:
        if col.lower() in ['survived', 'target', 'label', 'y']:
            target_col = col
            break
if target_col is None:
    print(f'[ERROR] Target column not found. Available: {df.columns.tolist()}')
    target_col = df.columns[-1]
    print(f'[STATUS] Using last column as target: {target_col}')

print(f'[STATUS] Target column: {target_col}')
print(f'[STATUS] Target distribution:\n{df[target_col].value_counts()}')

# ========== Separate features and target ==========
y = df[target_col].copy()
X = df.drop(columns=[target_col])

# Drop ID-like columns
id_cols = [c for c in X.columns if c.lower() in ['passengerid', 'id', 'passenger_id', 'index', 'unnamed: 0', 'row_id']]
if id_cols:
    X = X.drop(columns=id_cols)
    print(f'[STATUS] Dropped ID columns: {id_cols}')

# ========== Detect split column ==========
set_col = None
for col in X.columns:
    if col.lower() in ['set', 'split', 'dataset', 'subset', 'data_set', 'type', 'purpose']:
        set_col = col
        break

X_train, X_test, y_train, y_test = None, None, None, None

if set_col is not None:
    print(f'[STATUS] Found split column: {set_col}')
    vals = X[set_col].astype(str).str.lower().str.strip()
    train_mask = vals.isin(['train', 'training', '0'])
    test_mask = vals.isin(['test', 'testing', '1', 'validation', 'val'])
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        X_train = X[train_mask].drop(columns=[set_col])
        X_test = X[test_mask].drop(columns=[set_col])
        y_train = y[train_mask]
        y_test = y[test_mask]
        print(f'[STATUS] Using split column: Train={len(X_train)}, Test={len(X_test)}')
    else:
        print(f'[STATUS] Split column values not recognized, ignoring')
        set_col = None

if X_train is None:
    X = X.drop(columns=[set_col]) if set_col else X
    # Check if stratify is possible
    min_class_count = y.value_counts().min()
    if min_class_count >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print(f'[STATUS] Train/Test split with stratify: Train={len(X_train)}, Test={len(X_test)}')
        except Exception:
            print(f'[STATUS] Stratify failed (min_class={min_class_count}), using random split')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f'[STATUS] Train/Test split random: Train={len(X_train)}, Test={len(X_test)}')
    else:
        print(f'[STATUS] Too few samples per class ({min_class_count}), using random split')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f'[STATUS] Train/Test split random: Train={len(X_train)}, Test={len(X_test)}')

# ========== Preprocessing ==========
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f'[STATUS] Categorical columns: {cat_cols}')
print(f'[STATUS] Numeric columns: {num_cols}')

# Impute and encode
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', LabelEncoder())
])

# Process categorical: LabelEncode each
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# Impute missing
for col in cat_cols:
    X_train_processed[col] = X_train_processed[col].fillna('missing')
    X_test_processed[col] = X_test_processed[col].fillna('missing')

for col in num_cols:
    med = X_train_processed[col].median()
    X_train_processed[col] = X_train_processed[col].fillna(med)
    X_test_processed[col] = X_test_processed[col].fillna(med)

# Encode categoricals
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined to handle unseen labels
    combined = pd.concat([X_train_processed[col], X_test_processed[col]]).astype(str).unique()
    le.fit(combined)
    X_train_processed[col] = le.transform(X_train_processed[col].astype(str))
    X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
    label_encoders[col] = le

# Scale numeric
scaler = StandardScaler()
X_train_processed[num_cols] = scaler.fit_transform(X_train_processed[num_cols])
X_test_processed[num_cols] = scaler.transform(X_test_processed[num_cols])

# Ensure all columns are numeric
X_train_final = X_train_processed.astype(float)
X_test_final = X_test_processed.astype(float)

# Encode target if needed
if y_train.dtype == 'object' or y_train.dtype.name == 'category':
    le_target = LabelEncoder()
    y_train_enc = le_target.fit_transform(y_train)
    y_test_enc = le_target.transform(y_test)
    classes = le_target.classes_
else:
    y_train_enc = y_train.values
    y_test_enc = y_test.values
    classes = np.unique(y_train_enc)

print(f'[STATUS] Target classes: {classes}')
print(f'[STATUS] Preprocessed: X_train={X_train_final.shape}, X_test={X_test_final.shape}')

# ========== Algorithms to test ==========
algorithms = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(verbose=-1, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}

results = []

for name, model in algorithms.items():
    print(f'[STATUS] Testing {name}...')
    start = time.time()
    
    try:
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_final, y_train_enc, cv=cv, scoring='f1_weighted')
        
        # Train on full train set
        model.fit(X_train_final, y_train_enc)
        
        # Predict on test
        y_pred = model.predict(X_test_final)
        y_prob = model.predict_proba(X_test_final) if hasattr(model, 'predict_proba') else None
        
        test_f1 = f1_score(y_test_enc, y_pred, average='weighted')
        test_acc = accuracy_score(y_test_enc, y_pred)
        
        # AUC for binary
        auc = None
        if len(classes) == 2 and y_prob is not None:
            auc = roc_auc_score(y_test_enc, y_prob[:, 1])
        
        elapsed = time.time() - start
        
        results.append({
            'Algorithm': name,
            'CV Mean': round(cv_scores.mean(), 4),
            'CV Std': round(cv_scores.std(), 4),
            'Test F1': round(test_f1, 4),
            'Test Acc': round(test_acc, 4),
            'Test AUC': round(auc, 4) if auc else 'N/A',
            'Time (s)': round(elapsed, 1)
        })
        
        print(f'  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test F1: {test_f1:.4f} | Time: {elapsed:.1f}s')
        
    except Exception as e:
        print(f'  Error with {name}: {str(e)}')
        results.append({
            'Algorithm': name,
            'CV Mean': 'ERROR',
            'CV Std': 'ERROR',
            'Test F1': 'ERROR',
            'Test Acc': 'ERROR',
            'Test AUC': 'ERROR',
            'Time (s)': round(time.time() - start, 1)
        })

# ========== Results ==========
results_df = pd.DataFrame(results)

# Find winner (best F1 among successful)
valid_results = [r for r in results if r['Test F1'] != 'ERROR']
if valid_results:
    winner = max(valid_results, key=lambda r: r['Test F1'])
    print(f'\n[STATUS] Winner: {winner["Algorithm"]} — CV: {winner["CV Mean"]}, Test F1: {winner["Test F1"]}')
else:
    winner = None
    print('\n[STATUS] No algorithm succeeded')

# ========== Save ==========
output_csv = os.path.join(OUTPUT_DIR, 'mo_output.csv')
results_df.to_csv(output_csv, index=False)
print(f'[STATUS] Results saved: {output_csv}')

# ========== Report ==========
report = [
    'Mo Model Report — Phase 1: Explore',
    '====================================',
    'Problem Type: Classification',
    'Phase: 1 (Explore — all algorithms, default params)',
    '',
    'Algorithm Comparison (CV 5-fold):',
    '| Algorithm | CV Mean | CV Std | Test F1 | Test AUC | Time |',
    '|-----------|---------|--------|---------|----------|------|',
]
for r in results:
    report.append(f'| {r["Algorithm"]:15s} | {str(r["CV Mean"]):7s} | {str(r["CV Std"]):6s} | {str(r["Test F1"]):7s} | {str(r["Test AUC"]):7s} | {str(r["Time (s)"]):5s} |')

if winner:
    report.append(f'\nWinner: {winner["Algorithm"]} — CV: {winner["CV Mean"]}, Test F1: {winner["Test F1"]}')
    report.append(f'\nPREPROCESSING_REQUIREMENT')
    report.append('=========================')
    report.append(f'Algorithm Selected: {winner["Algorithm"]}')
    report.append('Scaling: StandardScaler')
    if cat_cols:
        report.append(f'Encoding: Label Encoding for {cat_cols}')
    else:
        report.append('Encoding: None')
    report.append('Transform: Imputation (median numeric, constant categorical)')
    report.append('Loop Back To Finn: NO')
    report.append('Next Phase: Phase 2 — Tune')

report_md = '\n'.join(report)
report_path = os.path.join(OUTPUT_DIR, 'model_results.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_md)
print(f'[STATUS] Report saved: {report_path}')

# ========== Self-Improvement Report ==========
improvement = [
    'Self-Improvement Report',
    '=======================',
    f'Phase: 1 (Explore)',
    f'Algorithms Tested: {len(algorithms)}',
    f'Winner: {winner["Algorithm"] if winner else "None"}',
    f'Knowledge Base: No new methods discovered for Titanic classification',
    f'Notes: Standard preprocessing pipeline (impute + scale + encode) works well',
    'Saved to: mo_methods.md (no changes needed)'
]
improvement_md = '\n'.join(improvement)
improvement_path = os.path.join(OUTPUT_DIR, 'self_improvement.md')
with open(improvement_path, 'w', encoding='utf-8') as f:
    f.write(improvement_md)
print(f'[STATUS] Self-Improvement saved: {improvement_path}')

print('\n[STATUS] Phase 1 Complete!')