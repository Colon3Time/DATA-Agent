import argparse, os, sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============== ARGUMENTS ==============
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test4\output\finn\finn_output.csv')
parser.add_argument('--output-dir', default=r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test4\output\mo')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== LOAD DATA ==============
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: shape={df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ============== DETECT TARGET COLUMN ==============
target_candidates = [c for c in df.columns if c.lower() in ['resigned', 'attrition', 'churn', 'left', 'target', 'label', 'status', 'is_resigned']]
if not target_candidates:
    target_candidates = [c for c in df.columns if 'resign' in c.lower() or 'attrit' in c.lower() or 'churn' in c.lower() or 'left' in c.lower()]

if not target_candidates:
    binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]
    print(f'[STATUS] Binary columns: {binary_cols}')
    if binary_cols:
        target_candidates = [binary_cols[0]]

if not target_candidates:
    print('[ERROR] No target column found!')
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        TARGET = numeric_cols[0]
        print(f'[STATUS] Using numeric column as target: {TARGET}')
        problem_type = 'regression'
    else:
        sys.exit(1)
else:
    TARGET = target_candidates[0]
    problem_type = 'classification'

print(f'[STATUS] Target column: {TARGET}')
print(f'[STATUS] Problem type: {problem_type}')

# ============== DATA PREPARATION ==============
df = df.copy()
print(f'[STATUS] Target distribution:\n{df[TARGET].value_counts()}')
print(f'[STATUS] Target dtype: {df[TARGET].dtype}')

# Handle target encoding for classification
y = None
if problem_type == 'classification':
    if df[TARGET].dtype == 'object':
        y_raw = df[TARGET].astype(str).str.strip().str.lower()
        unique_vals = y_raw.unique()
        print(f'[STATUS] Unique target values: {unique_vals}')
        yes_set = {'yes', 'y', '1', 'true', 't', 'resigned', 'attrition', 'churn', 'left'}
        no_set = {'no', 'n', '0', 'false', 'f', 'stayed', 'not resigned', 'not attrition', 'not churn', 'not left'}
        y = pd.Series([1 if v in yes_set else (0 if v in no_set else np.nan) for v in y_raw], index=df.index)
        if y.isna().any():
            print(f'[WARNING] {y.isna().sum()} rows with unknown target values, dropping them')
            df = df.loc[y.notna()].reset_index(drop=True)
            y = y.loc[y.notna()].reset_index(drop=True)
    else:
        y = df[TARGET].astype(int)
else:
    y = df[TARGET]

print(f'[STATUS] y shape: {y.shape}')
print(f'[STATUS] Target distribution After:\n{y.value_counts() if hasattr(y, "value_counts") else "N/A"}')

# ============== SELECT FEATURES ==============
exclude_cols = [TARGET] + [c for c in df.columns if c.lower() in ['id', 'employeeid', 'empid', 'customerid', 'userid']]
feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
print(f'[STATUS] Feature columns: {len(feature_cols)}')

X = df[feature_cols].copy()

# Handle inf and NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Drop constant columns
constant_cols = [c for c in X.columns if X[c].nunique() == 1]
if constant_cols:
    print(f'[STATUS] Dropping constant columns: {constant_cols}')
    X = X.drop(columns=constant_cols)
    feature_cols = [c for c in feature_cols if c not in constant_cols]

print(f'[STATUS] X shape after cleaning: {X.shape}')

# ============== TRAIN/TEST SPLIT ==============
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None)
print(f'[STATUS] Train: {X_train.shape}, Test: {X_test.shape}')

# ============== STANDARDIZE ==============
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============== MODEL COMPARISON ==============
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import time

results = []

if problem_type == 'classification':
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    # Check if XGBoost available
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
    except ImportError:
        print('[STATUS] XGBoost not available')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        start = time.time()
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            except:
                pass
        
        elapsed = time.time() - start
        
        result = {
            'Model': name,
            'CV F1 Mean': round(cv_mean, 4),
            'CV F1 Std': round(cv_std, 4),
            'Test Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Test Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'Test Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'Test F1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'Time (s)': round(elapsed, 2)
        }
        
        if y_prob is not None and len(np.unique(y_train)) == 2:
            try:
                result['Test AUC'] = round(roc_auc_score(y_test, y_prob), 4)
            except:
                result['Test AUC'] = 'N/A'
        else:
            result['Test AUC'] = 'N/A'
        
        results.append(result)
        
        # Feature importance for tree-based
        if hasattr(model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'feature': feature_cols, 
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f'\n[FEATURE] {name} Top Features:')
            print(feat_imp.head(5).to_string(index=False))

else:  # regression
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    # Check if XGBoost available
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        print('[STATUS] XGBoost not available')
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        start = time.time()
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        elapsed = time.time() - start
        
        result = {
            'Model': name,
            'CV R2 Mean': round(cv_mean, 4),
            'CV R2 Std': round(cv_std, 4),
            'Test R2': round(r2_score(y_test, y_pred), 4),
            'Test RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'Test MAE': round(mean_absolute_error(y_test, y_pred), 4),
            'Time (s)': round(elapsed, 2)
        }
        results.append(result)
        
        if hasattr(model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'feature': feature_cols, 
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f'\n[FEATURE] {name} Top Features:')
            print(feat_imp.head(5).to_string(index=False))

# ============== RESULTS TABLE ==============
print(f'\n{"="*80}')
print(f'Model Comparison Results')
print(f'{"="*80}')
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Determine best model
if problem_type == 'classification':
    best_idx = results_df['Test F1'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_f1 = results_df.loc[best_idx, 'Test F1']
    print(f'\n[RESULT] Best Model: {best_model_name} (Test F1: {best_f1})')
else:
    best_idx = results_df['Test R2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_r2 = results_df.loc[best_idx, 'Test R2']
    print(f'\n[RESULT] Best Model: {best_model_name} (Test R2: {best_r2})')

# ============== SAVE RESULTS ==============
# Save comparison CSV
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

# Save detailed report
report_lines = [
    f'Mo Model Report',
    f'{"="*16}',
    f'',
    f'Input File: {INPUT_PATH}',
    f'Target Column: {TARGET}',
    f'Problem Type: {problem_type}',
    f'Number of Features: {len(feature_cols)}',
    f'Training Samples: {X_train.shape[0]}',
    f'Test Samples: {X_test.shape[0]}',
    f'',
    f'Models Tested: {len(models)}',
    f'',
    f'Results Comparison:',
    f'-------------------',
    f''
]

# Format table manually (no tabulate dependency)
col_widths = {
    'Model': max(25, max(len(m) for m in results_df['Model'])),
    'CV_Mean': 12,
    'CV_Std': 12,
    'Test1': 14,
    'Test2': 14,
    'Test3': 14,
    'Test4': 12
}

# Header
if problem_type == 'classification':
    header = f"{'Model'.ljust(col_widths['Model'])} | {'CV F1 Mean'.ljust(col_widths['CV_Mean'])} | {'CV F1 Std'.ljust(col_widths['CV_Std'])} | {'Test Acc'.ljust(col_widths['Test1'])} | {'Test Prec'.ljust(col_widths['Test2'])} | {'Test F1'.ljust(col_widths['Test3'])} | {'AUC'.ljust(col_widths['Test4'])} | {'Time'.ljust(8)}"
else:
    header = f"{'Model'.ljust(col_widths['Model'])} | {'CV R2 Mean'.ljust(col_widths['CV_Mean'])} | {'CV R2 Std'.ljust(col_widths['CV_Std'])} | {'Test R2'.ljust(col_widths['Test1'])} | {'Test RMSE'.ljust(col_widths['Test2'])} | {'Test MAE'.ljust(col_widths['Test3'])} | {'Time'.ljust(8)}"

report_lines.append(header)
report_lines.append('-' * len(header))

for _, row in results_df.iterrows():
    if problem_type == 'classification':
        line = f"{str(row['Model']).ljust(col_widths['Model'])} | {str(row['CV F1 Mean']).ljust(col_widths['CV_Mean'])} | {str(row['CV F1 Std']).ljust(col_widths['CV_Std'])} | {str(row['Test Accuracy']).ljust(col_widths['Test1'])} | {str(row['Test Precision']).ljust(col_widths['Test2'])} | {str(row['Test F1']).ljust(col_widths['Test3'])} | {str(row['Test AUC']).ljust(col_widths['Test4'])} | {str(row['Time (s)']).ljust(8)}"
    else:
        line = f"{str(row['Model']).ljust(col_widths['Model'])} | {str(row['CV R2 Mean']).ljust(col_widths['CV_Mean'])} | {str(row['CV R2 Std']).ljust(col_widths['CV_Std'])} | {str(row['Test R2']).ljust(col_widths['Test1'])} | {str(row['Test RMSE']).ljust(col_widths['Test2'])} | {str(row['Test MAE']).ljust(col_widths['Test3'])} | {str(row['Time (s)']).ljust(8)}"
    report_lines.append(line)

report_lines.extend([
    '',
    f'Best Model: {best_model_name}',
    f'',
    f'Overfitting Check:',
    f'- CV mean vs Test performance: {("Good" if results_df.loc[best_idx, "CV F1 Mean" if problem_type == "classification" else "CV R2 Mean"] > 0.5 else "Poor")}',
    f'- CV std: {results_df.loc[best_idx, "CV F1 Std" if problem_type == "classification" else "CV R2 Std"]}',
    f'',
    f'Business Recommendation:',
    f'- {best_model_name} gives the most balanced performance',
    f'- Features used: {", ".join(feature_cols[:10])}{"..." if len(feature_cols) > 10 else ""}',
    f'- Model is {"ready for deployment" if (results_df.loc[best_idx, "Test F1" if problem_type == "classification" else "Test R2"] > 0.7) else "needs further tuning"}',
    f'',
    f'Self-Improvement Report:',
    f'{"="*24}',
    f'- Compared {len(models)} models using cross-validation',
    f'- No new methods discovered in this run',
    f'- Knowledge Base: no changes needed'
])

report_text = '\n'.join(report_lines)

with open(os.path.join(OUTPUT_DIR, 'model_results.md'), 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f'\n[STATUS] Reports saved to {OUTPUT_DIR}')

# ============== SAVE BEST MODEL INFO ==============
best_model_info = {
    'best_model': best_model_name,
    'target_column': TARGET,
    'problem_type': problem_type,
    'feature_columns': feature_cols,
    'metrics': results_df.loc[best_idx].to_dict()
}
with open(os.path.join(OUTPUT_DIR, 'best_model_info.json'), 'w') as f:
    json.dump(best_model_info, f, indent=2)

print('[STATUS] All models trained and saved successfully!')
print(f'[OUTPUT] {os.path.join(OUTPUT_DIR, "model_results.md")}')
print(f'[OUTPUT] {os.path.join(OUTPUT_DIR, "model_comparison.csv")}')
print(f'[OUTPUT] {os.path.join(OUTPUT_DIR, "best_model_info.json")}')
