import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import traceback
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# Parse arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Find the correct input
# ============================================================
if not INPUT_PATH or not os.path.exists(INPUT_PATH):
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
    except NameError:
        project_root = Path(os.getcwd()).resolve().parent.parent.parent.parent
    dana_path = project_root / 'output' / 'dana' / 'dana_output.csv'
    if dana_path.exists():
        INPUT_PATH = str(dana_path)
    else:
        parent = Path(args.input).parent if args.input else Path.cwd()
        csvs = sorted(parent.glob('**/*.csv'))
        if csvs:
            for c in csvs:
                name = c.name.lower()
                if 'feature_importance' not in name and 'model_results' not in name:
                    INPUT_PATH = str(c)
                    break
            else:
                INPUT_PATH = str(csvs[0])
        else:
            csv_path = Path(args.input)
            if csv_path.exists() and csv_path.suffix == '.csv':
                INPUT_PATH = str(csv_path)

print(f'[STATUS] INPUT PATH: {INPUT_PATH}')
print(f'[STATUS] OUTPUT DIR: {OUTPUT_DIR}')

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded dataset: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ============================================================
# Detect target column
# ============================================================
target_col = 'resigned'
if target_col not in df.columns:
    print(f'[ERROR] Target column "{target_col}" not found in: {list(df.columns)}')
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
            target_col = col
            print(f'[INFO] Auto-detected target column: {target_col}')
            break
    else:
        print('[FATAL] No binary target column found')
        last_col = df.columns[-1]
        if df[last_col].dtype in ['int64', 'float64']:
            target_col = last_col
            print(f'[INFO] Using last column as target: {target_col}')
        else:
            print(f'[FATAL] Last column "{last_col}" also not numeric - exiting')
            sys.exit(1)

print(f'[STATUS] Target column: {target_col}')

# ============================================================
# Basic EDA
# ============================================================
print('\n[STATUS] === BASIC EDA ===')
print(f'Shape: {df.shape}')
print(f'Target distribution:\n{df[target_col].value_counts().to_dict()}')
print(f'Missing values:\n{df.isnull().sum().to_dict()}')
print(f'Dtypes:\n{df.dtypes.to_dict()}')

# ============================================================
# Prepare data for modeling
# ============================================================
# Drop rows with missing target
df_model = df.dropna(subset=[target_col]).copy()
print(f'[STATUS] After dropping missing target: {df_model.shape}')

# Separate features and target
X = df_model.drop(columns=[target_col])
y = df_model[target_col].astype(int)

# Identify column types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'[STATUS] Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}')

# Drop columns that are identifiers or too high cardinality
drop_cols = []
for col in X.columns:
    if col.lower() in ['id', 'customer_id', 'user_id', 'member_id']:
        drop_cols.append(col)
    elif X[col].dtype == 'object' and X[col].nunique() > 50:
        drop_cols.append(col)
        print(f'[INFO] Dropping high-cardinality column: {col} ({X[col].nunique()} unique)')

if drop_cols:
    X = X.drop(columns=drop_cols)
    print(f'[STATUS] Dropped columns: {drop_cols}')

# Re-identify column types after dropping
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Check for constant columns
for col in X.columns:
    if X[col].nunique() <= 1:
        if col in numeric_cols:
            numeric_cols.remove(col)
        elif col in categorical_cols:
            categorical_cols.remove(col)
        X = X.drop(columns=[col])
        print(f'[INFO] Dropping constant column: {col}')

print(f'[STATUS] After cleaning: Numeric={len(numeric_cols)}, Categorical={len(categorical_cols)}')

# ============================================================
# Handle missing values
# ============================================================
for col in numeric_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna('missing')

# ============================================================
# Train/test split
# ============================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'[STATUS] Train: {X_train.shape}, Test: {X_test.shape}')

# ============================================================
# Build preprocessing pipelines
# ============================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# ============================================================
# Model comparison
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print('\n[STATUS] === MODEL COMPARISON ===')

for name, model in models.items():
    try:
        start_time = time.time()
        
        # Build pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # CV score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train on full train set
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else np.zeros(len(y_test))
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.0
        
        elapsed = time.time() - start_time
        
        results.append({
            'Model': name,
            'CV Score (mean)': round(cv_mean, 4),
            'CV Std': round(cv_std, 4),
            'Test Accuracy': round(acc, 4),
            'Test Precision': round(prec, 4),
            'Test Recall': round(rec, 4),
            'Test F1': round(f1, 4),
            'Test AUC': round(auc, 4),
            'Time (s)': round(elapsed, 2)
        })
        
        print(f'  {name:25s} | CV: {cv_mean:.4f} +/- {cv_std:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed:.2f}s')
        
    except Exception as e:
        print(f'  {name:25s} | ERROR: {str(e)[:50]}')
        results.append({
            'Model': name,
            'CV Score (mean)': 0.0,
            'CV Std': 0.0,
            'Test Accuracy': 0.0,
            'Test Precision': 0.0,
            'Test Recall': 0.0,
            'Test F1': 0.0,
            'Test AUC': 0.0,
            'Time (s)': 0.0
        })

# ============================================================
# Try XGBoost and LightGBM if available
# ============================================================
try:
    import xgboost as xgb
    start_time = time.time()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Handle class imbalance with scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42,
        use_label_encoder=False, eval_metric='logloss'
    )
    xgb_model.fit(X_train_processed, y_train)
    
    cv_scores = cross_val_score(xgb_model, X_train_processed, y_train, cv=cv, scoring='f1_weighted')
    y_pred = xgb_model.predict(X_test_processed)
    y_prob = xgb_model.predict_proba(X_test_processed)[:, 1]
    
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    elapsed = time.time() - start_time
    
    results.append({
        'Model': 'XGBoost',
        'CV Score (mean)': round(cv_scores.mean(), 4),
        'CV Std': round(cv_scores.std(), 4),
        'Test Accuracy': round(acc, 4),
        'Test Precision': round(prec, 4),
        'Test Recall': round(rec, 4),
        'Test F1': round(f1, 4),
        'Test AUC': round(auc, 4),
        'Time (s)': round(elapsed, 2)
    })
    print(f'  {"XGBoost":25s} | CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed:.2f}s')
except ImportError:
    print('[STATUS] XGBoost not available, skipping')
except Exception as e:
    print(f'[WARNING] XGBoost error: {str(e)[:80]}')

try:
    import lightgbm as lgb
    start_time = time.time()
    
    # Calculate class weight
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        class_weight='balanced', random_state=42, verbose=-1
    )
    lgb_model.fit(X_train_processed, y_train)
    
    cv_scores = cross_val_score(lgb_model, X_train_processed, y_train, cv=cv, scoring='f1_weighted')
    y_pred = lgb_model.predict(X_test_processed)
    y_prob = lgb_model.predict_proba(X_test_processed)[:, 1]
    
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    elapsed = time.time() - start_time
    
    results.append({
        'Model': 'LightGBM',
        'CV Score (mean)': round(cv_scores.mean(), 4),
        'CV Std': round(cv_scores.std(), 4),
        'Test Accuracy': round(acc, 4),
        'Test Precision': round(prec, 4),
        'Test Recall': round(rec, 4),
        'Test F1': round(f1, 4),
        'Test AUC': round(auc, 4),
        'Time (s)': round(elapsed, 2)
    })
    print(f'  {"LightGBM":25s} | CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed:.2f}s')
except ImportError:
    print('[STATUS] LightGBM not available, skipping')
except Exception as e:
    print(f'[WARNING] LightGBM error: {str(e)[:80]}')

# ============================================================
# Results DataFrame
# ============================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('CV Score (mean)', ascending=False).reset_index(drop=True)

print('\n[STATUS] === FINAL RESULTS (sorted by CV Score) ===')
print(results_df.to_string(index=False))

# Best model
best_idx = results_df['CV Score (mean)'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_cv = results_df.loc[best_idx, 'CV Score (mean)']
best_f1 = results_df.loc[best_idx, 'Test F1']

print(f'\n[STATUS] Best Model: {best_model_name}')
print(f'[STATUS] CV Score: {best_cv:.4f}')
print(f'[STATUS] Test F1: {best_f1:.4f}')

# ============================================================
# Retrain best model for feature importance
# ============================================================
print('\n[STATUS] === FEATURE IMPORTANCE ANALYSIS ===')

best_pipeline = None
feature_importance_df = None

# Try to get feature importance from best model
try:
    # Retrain the best model on full training data
    if best_model_name in models:
        best_model = models[best_model_name]
        best_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', best_model)
        ])
        best_pipeline.fit(X_train, y_train)
        
        # Get feature names after preprocessing
        cat_feature_names = []
        if categorical_cols:
            ohe = preprocessor.named_transformers_['cat']
            cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
        else:
            cat_feature_names = []
        
        all_feature_names = numeric_cols + cat_feature_names
        
        # Try to get feature importance
        if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = best_pipeline.named_steps['classifier'].feature_importances_
            if len(importances) == len(all_feature_names):
                feature_importance_df = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                print(feature_importance_df.head(20).to_string(index=False))
        
        elif hasattr(best_pipeline.named_steps['classifier'], 'coef_'):
            coef = best_pipeline.named_steps['classifier'].coef_[0]
            if len(coef) == len(all_feature_names):
                feature_importance_df = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': np.abs(coef)
                }).sort_values('importance', ascending=False)
                print(feature_importance_df.head(20).to_string(index=False))
    
    elif 'XGB' in best_model_name and 'xgb_model' in dir():
        # Use xgb_model directly
        if hasattr(xgb_model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'feature': [f'f{i}' for i in range(len(xgb_model.feature_importances_))],
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance_df.head(20).to_string(index=False))
    
    elif 'Light' in best_model_name and 'lgb_model' in dir():
        if hasattr(lgb_model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'feature': [f'f{i}' for i in range(len(lgb_model.feature_importances_))],
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance_df.head(20).to_string(index=False))
            
except Exception as e:
    print(f'[WARNING] Feature importance extraction failed: {str(e)[:80]}')

# ============================================================
# Confusion matrix for best model
# ============================================================
print('\n[STATUS] === CONFUSION MATRIX (Best Model) ===')
try:
    if best_pipeline is not None:
        y_pred_best = best_pipeline.predict(X_test)
    elif 'XGB' in best_model_name and 'xgb_model' in dir():
        y_pred_best = xgb_model.predict(X_test_processed)
    elif 'Light' in best_model_name and 'lgb_model' in dir():
        y_pred_best = lgb_model.predict(X_test_processed)
    else:
        y_pred_best = best_pipeline.predict(X_test) if best_pipeline else np.zeros(len(y_test))
    
    cm = confusion_matrix(y_test, y_pred_best)
    print(f'Confusion Matrix:')
    print(f'  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}')
    print(f'  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}')
    
    print(f'\nClassification Report:')
    print(classification_report(y_test, y_pred_best, zero_division=0))
    
except Exception as e:
    print(f'[WARNING] Confusion matrix failed: {str(e)[:80]}')

# ============================================================
# Save outputs
# ============================================================
print('\n[STATUS] === SAVING OUTPUTS ===')

# 1. Save model results CSV
results_csv = os.path.join(OUTPUT_DIR, 'mo_model_results.csv')
results_df.to_csv(results_csv, index=False)
print(f'[STATUS] Saved model results: {results_csv}')

# 2. Save feature importance CSV
if feature_importance_df is not None:
    fi_csv = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    feature_importance_df.to_csv(fi_csv, index=False)
    print(f'[STATUS] Saved feature importance: {fi_csv}')

# 3. Save predictions CSV (for DANA compatibility)
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred_best if 'y_pred_best' in dir() else [],
    'probability': y_prob if 'y_prob' in dir() else []
})
pred_csv = os.path.join(OUTPUT_DIR, 'dana_output.csv')
predictions_df.to_csv(pred_csv, index=False)
print(f'[STATUS] Saved predictions: {pred_csv}')

# 4. Generate Markdown report
md_lines = []
md_lines.append('# Mo Model Report — Phase 1: Explore')
md_lines.append(f'')
md_lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
md_lines.append(f'')
md_lines.append(f'**Input Dataset:** {INPUT_PATH}')
md_lines.append(f'**Dataset Shape:** {df.shape}')
md_lines.append(f'**Target Column:** {target_col}')
md_lines.append(f'')
md_lines.append(f'## Basic EDA')
md_lines.append(f'')
md_lines.append(f'- **Shape:** {df.shape}')
md_lines.append(f'- **Target distribution:** {df[target_col].value_counts().to_dict()}')
md_lines.append(f'- **Missing values:** {df.isnull().sum().to_dict()}')
md_lines.append(f'- **Numeric columns:** {len(numeric_cols)}')
md_lines.append(f'- **Categorical columns:** {len(categorical_cols)}')
md_lines.append(f'')
md_lines.append(f'## Algorithm Comparison (CV 5-fold)')
md_lines.append(f'')
md_lines.append(f'| Model | CV Score (mean) | CV Std | Test F1 | Test AUC | Time (s) |')
md_lines.append(f'|-------|-----------------|--------|---------|----------|----------|')

for _, row in results_df.iterrows():
    md_lines.append(f'| {row["Model"]} | {row["CV Score (mean)"]:.4f} | {row["CV Std"]:.4f} | {row["Test F1"]:.4f} | {row["Test AUC"]:.4f} | {row["Time (s)"]:.2f} |')

md_lines.append(f'')
md_lines.append(f'## Best Model: {best_model_name}')
md_lines.append(f'')
md_lines.append(f'- **CV Score:** {best_cv:.4f}')
md_lines.append(f'- **Test F1:** {best_f1:.4f}')
md_lines.append(f'- **Test AUC:** {results_df.loc[best_idx, "Test AUC"]:.4f}')
md_lines.append(f'- **Test Accuracy:** {results_df.loc[best_idx, "Test Accuracy"]:.4f}')
md_lines.append(f'- **Test Precision:** {results_df.loc[best_idx, "Test Precision"]:.4f}')
md_lines.append(f'- **Test Recall:** {results_df.loc[best_idx, "Test Recall"]:.4f}')
md_lines.append(f'')
md_lines.append(f'## Confusion Matrix')
md_lines.append(f'')
try:
    md_lines.append(f'| | Predicted 0 | Predicted 1 |')
    md_lines.append(f'|-------|-------------|-------------|')
    md_lines.append(f'| Actual 0 | {cm[0,0]} | {cm[0,1]} |')
    md_lines.append(f'| Actual 1 | {cm[1,0]} | {cm[1,1]} |')
except:
    md_lines.append('(Could not compute)')
md_lines.append(f'')

if feature_importance_df is not None:
    md_lines.append(f'## Feature Importance (Top 10)')
    md_lines.append(f'')
    md_lines.append(f'| Feature | Importance |')
    md_lines.append(f'|---------|-----------|')
    for _, row in feature_importance_df.head(10).iterrows():
        md_lines.append(f'| {row["feature"]} | {row["importance"]:.4f} |')
    md_lines.append(f'')

md_lines.append(f'## Business Recommendation')
md_lines.append(f'')
md_lines.append(f'- **Recommended Model:** {best_model_name}')
md_lines.append(f'- This model was selected based on highest CV score and test performance.')
if best_f1 > 0.7:
    md_lines.append(f'- The model shows good performance (F1 > 0.7) and can be used for prediction.')
else:
    md_lines.append(f'- The model performance is moderate. Consider feature engineering or data collection to improve.')
md_lines.append(f'')
md_lines.append(f'---')
md_lines.append(f'*Report generated by Mo — Model Builder & Evaluator*')

# Write markdown report
md_content = '\n'.join(md_lines)
report_md = os.path.join(OUTPUT_DIR, 'model_results.md')
with open(report_md, 'w', encoding='utf-8') as f:
    f.write(md_content)
print(f'[STATUS] Saved markdown report: {report_md}')

print('\n[STATUS] === DONE ===')
