import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import time

warnings.filterwarnings('ignore')

# ============================================================
# ARGUMENT PARSING & PATH HANDLING
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# AUTO-FIX: If input is a directory, find CSV inside
# ============================================================
if os.path.isdir(INPUT_PATH):
    csv_files = glob.glob(os.path.join(INPUT_PATH, '*.csv'))
    if csv_files:
        INPUT_PATH = csv_files[0]
        print(f'[STATUS] Input is directory, using: {INPUT_PATH}')
    else:
        print(f'[ERROR] No CSV found in directory: {INPUT_PATH}')
        sys.exit(1)

# ============================================================
# LOAD DATA
# ============================================================
print(f'[STATUS] Loading data from: {INPUT_PATH}')
if not os.path.exists(INPUT_PATH):
    print(f'[ERROR] Input file not found: {INPUT_PATH}')
    sys.exit(1)

try:
    df = pd.read_csv(INPUT_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_PATH, encoding='latin1')
except Exception as e:
    print(f'[ERROR] Could not read CSV: {e}')
    sys.exit(1)

print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ============================================================
# PREPARE FEATURES & TARGET
# ============================================================
TARGET = 'Outcome'

if TARGET not in df.columns:
    alt_targets = [c for c in df.columns if 'outcome' in c.lower() or 'target' in c.lower()]
    if alt_targets:
        TARGET = alt_targets[0]
        print(f'[STATUS] Using target column: {TARGET}')
    else:
        last_col = df.columns[-1]
        if df[last_col].nunique() <= 10:
            TARGET = last_col
            print(f'[STATUS] Using last column as target: {TARGET}')
        else:
            print(f'[ERROR] Target column not found. Available columns: {list(df.columns)}')
            sys.exit(1)

exclude_cols = {TARGET, 'Unnamed: 0', 'id', 'ID', 'Id', 'index', 'Index'}
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f'[STATUS] Features: {len(feature_cols)}, Target: {TARGET}')

X = df[feature_cols].copy()
y = df[TARGET].copy()

# Handle missing values
if X.isnull().sum().sum() > 0:
    print(f'[STATUS] Handling missing values in features...')
    for col in X.columns:
        if X[col].dtype in ['object', 'category']:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')
        else:
            X[col] = X[col].fillna(X[col].median())

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Handle target encoding if needed
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

# Convert to numeric
X = X.astype(float)
y = pd.Series(y).astype(int)

print(f'[STATUS] X shape: {X.shape}, y distribution: {pd.Series(y).value_counts().to_dict()}')

# ============================================================
# TRAIN/TEST SPLIT — with safety check for stratify
# ============================================================
test_size = 0.2
random_state = 42
min_class_count = pd.Series(y).value_counts().min()

if min_class_count < 2:
    print(f'[WARNING] Minimum class count = {min_class_count}, cannot stratify. Using random split.')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
elif min_class_count < int(1 / test_size) + 1:
    print(f'[WARNING] Small class ({min_class_count}) may cause stratify issues. Using random split.')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
else:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except Exception as e:
        print(f'[WARNING] Stratify failed: {e}. Using random split.')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

print(f'[STATUS] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')

# ============================================================
# FEATURE SCALING (for algorithms that need it)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MODEL COMPARISON
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print(f'\n{"="*80}')
print(f'Model Comparison (5-fold Cross-Validation + Test Set)')
print(f'{"="*80}')

for name, model in models.items():
    start_time = time.time()
    
    # Scale data for SVM, KNN, Logistic Regression
    if name in ['SVM', 'KNN', 'Logistic Regression']:
        X_cv = X_train_scaled
        X_t = X_test_scaled
    else:
        X_cv = X_train.values
        X_t = X_test.values
    
    try:
        # Cross-validation
        cv_scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train on full training set
        model.fit(X_cv, y_train)
        
        # Predictions
        y_pred = model.predict(X_t)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC for binary classification
        auc_val = 0.0
        if len(np.unique(y)) == 2:
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_t)[:, 1]
                    auc_val = roc_auc_score(y_test, y_proba)
                else:
                    y_scores = model.decision_function(X_t) if hasattr(model, 'decision_function') else y_pred
                    if len(np.unique(y_scores)) > 1:
                        auc_val = roc_auc_score(y_test, y_scores)
            except:
                auc_val = 0.0
        
        elapsed = time.time() - start_time
        
        results.append({
            'Algorithm': name,
            'CV_Accuracy_Mean': round(cv_mean, 4),
            'CV_Accuracy_Std': round(cv_std, 4),
            'Test_Accuracy': round(acc, 4),
            'Test_F1': round(f1, 4),
            'Test_Precision': round(prec, 4),
            'Test_Recall': round(rec, 4),
            'Test_AUC': round(auc_val, 4),
            'Time': f'{elapsed:.2f}s'
        })
        
        print(f'{name:20s} | CV: {cv_mean:.4f} ± {cv_std:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc_val:.4f} | Time: {elapsed:.2f}s')
        
    except Exception as e:
        print(f'{name:20s} | ERROR: {e}')
        results.append({
            'Algorithm': name,
            'CV_Accuracy_Mean': 0,
            'CV_Accuracy_Std': 0,
            'Test_Accuracy': 0,
            'Test_F1': 0,
            'Test_Precision': 0,
            'Test_Recall': 0,
            'Test_AUC': 0,
            'Time': 'ERROR'
        })

# ============================================================
# RESULTS TABLE
# ============================================================
print(f'\n{"="*80}')
print('SUMMARY RESULTS')
print(f'{"="*80}')
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_Accuracy', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

# ============================================================
# SAVE OUTPUTS
# ============================================================
# Save CSV results
csv_path = os.path.join(OUTPUT_DIR, 'model_comparison_results.csv')
results_df.to_csv(csv_path, index=False)
print(f'\n[STATUS] Results saved to: {csv_path}')

# Save detailed report
report_path = os.path.join(OUTPUT_DIR, 'mo_model_results.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f'# Mo Model Report\n')
    f.write(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    f.write(f'## Data Summary\n')
    f.write(f'- Input file: {Path(INPUT_PATH).name}\n')
    f.write(f'- Rows: {X.shape[0]}, Features: {X.shape[1]}\n')
    f.write(f'- Target: {TARGET}\n')
    f.write(f'- Target distribution: {pd.Series(y).value_counts().to_dict()}\n\n')
    f.write(f'## Model Comparison\n\n')
    f.write(results_df.to_markdown(index=False) if hasattr(pd, 'to_markdown') else results_df.to_string())
    f.write(f'\n\n## Best Model\n')
    best = results_df.iloc[0]
    f.write(f'- **{best["Algorithm"]}**\n')
    f.write(f'- Test Accuracy: {best["Test_Accuracy"]:.4f}\n')
    f.write(f'- Test F1: {best["Test_F1"]:.4f}\n')
    f.write(f'- Test AUC: {best["Test_AUC"]:.4f}\n')
    f.write(f'- CV Score: {best["CV_Accuracy_Mean"]:.4f} ± {best["CV_Accuracy_Std"]:.4f}\n')
print(f'[STATUS] Report saved to: {report_path}')

print(f'\n{"="*80}')
print(f'[COMPLETE] All done!')
print(f'{"="*80}')
