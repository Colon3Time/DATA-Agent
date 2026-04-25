import argparse, os, pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir or 'projects/E-Commerce/output/mo'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
if not INPUT_PATH:
    # Auto-detect path
    base = 'projects/E-Commerce'
    candidates = [
        'projects/E-Commerce/output/finn/finn_output.csv',
        'projects/E-Commerce/output/finn_output.csv',
        'projects/E-Commerce/output/finn/finn_processed.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            INPUT_PATH = p
            break
    if not INPUT_PATH:
        # Try glob
        csvs = sorted(Path('projects/E-Commerce').rglob('finn_output.csv'))
        if csvs:
            INPUT_PATH = str(csvs[0])

print(f"[STATUS] Loading data from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Data shape: {df.shape}")
print(f"[STATUS] Columns: {list(df.columns)}")
print(f"[STATUS] Dtypes:\n{df.dtypes}")
print(f"[STATUS] Head:\n{df.head()}")

# === Auto-detect target column ===
target_col = None
for col in df.columns:
    col_lower = col.lower()
    if 'churn' in col_lower or 'target' in col_lower or 'label' in col_lower or 'y' == col_lower:
        target_col = col
        break

if target_col is None:
    # Try last column (common pattern for binary target)
    target_col = df.columns[-1]
    
print(f"[STATUS] Detected target column: {target_col}")

# === Inspect target ===
print(f"\n[STATUS] Target value counts:\n{df[target_col].value_counts()}")
print(f"[STATUS] Target unique values: {df[target_col].nunique()}")
print(f"[STATUS] Target dtype: {df[target_col].dtype}")

# === Data cleaning & preprocessing ===
# Drop columns that are identifiers or all NaN
cols_to_drop = [c for c in df.columns if c.lower() in ['id', 'customer_id', 'user_id', 'index', 'unnamed: 0']]
# Also drop any column that is a datetime string
for c in df.columns:
    if df[c].dtype == 'object':
        try:
            if pd.to_datetime(df[c], errors='coerce').notna().sum() > 0.8 * len(df):
                cols_to_drop.append(c)
        except:
            pass

cols_to_drop = list(set(cols_to_drop))
if cols_to_drop:
    print(f"[STATUS] Dropping identifier columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop, errors='ignore')

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Handle missing values in features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('missing')
        # Label encode categorical
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    elif pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(0)

# Ensure all columns are numeric
for col in X.columns:
    if X[col].dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

print(f"[STATUS] Feature shape after preprocessing: {X.shape}")
print(f"[STATUS] Target shape: {y.shape}")

# === Determine problem type ===
n_unique_targets = y.nunique()
if n_unique_targets == 2:
    problem_type = 'Binary Classification'
    is_binary = True
else:
    problem_type = 'Multi-class Classification' if n_unique_targets <= 20 else 'Regression'
    is_binary = False

# Check if it's actually regression (float target with many values)
if y.dtype in [np.float64, np.float32] and n_unique_targets > 20:
    problem_type = 'Regression'
    is_binary = False

print(f"[STATUS] Problem type: {problem_type} (n_unique={n_unique_targets})")

# === Train/Test Split ===
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import time

# Stratified split for classification, regular for regression
if problem_type in ['Binary Classification', 'Multi-class Classification']:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define models to compare ===
if problem_type == 'Binary Classification':
    # Check for imbalance
    class_dist = y.value_counts(normalize=True)
    print(f"[STATUS] Class distribution:\n{class_dist}")
    imbalance_ratio = class_dist.min() / class_dist.max()
    is_imbalanced = imbalance_ratio < 0.3
    print(f"[STATUS] Imbalance ratio: {imbalance_ratio:.4f} {'(IMBALANCED)' if is_imbalanced else '(balanced)'}")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced' if is_imbalanced else None),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced' if is_imbalanced else None),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced' if is_imbalanced else None),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=class_dist.max()/class_dist.min() if is_imbalanced else 1, verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced' if is_imbalanced else None, verbose=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced' if is_imbalanced else None),
    }
    primary_metric = 'f1_weighted'
    secondary_metric = 'roc_auc'
    
elif problem_type == 'Multi-class Classification':
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
    }
    primary_metric = 'f1_weighted'
    secondary_metric = 'accuracy'
    
else:  # Regression
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf'),
    }
    primary_metric = 'neg_root_mean_squared_error'
    secondary_metric = 'r2'

# === Train and evaluate all models ===
results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if problem_type != 'Regression' else KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model_name = name
    print(f"\n[STATUS] Training: {model_name}")
    
    start_time = time.time()
    
    try:
        # Cross-validation
        if problem_type == 'Regression':
            cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            cv_scores_rmse = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
            cv_scores = {'r2': cv_scores_r2, 'neg_rmse': cv_scores_rmse}
        elif is_imbalanced:
            cv_scores_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
            cv_scores_auc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
            cv_scores = {'f1_weighted': cv_scores_f1, 'roc_auc': cv_scores_auc}
        else:
            cv_scores_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
            cv_scores_acc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            cv_scores = {'f1_weighted': cv_scores_f1, 'accuracy': cv_scores_acc}
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)
        
        # Metrics
        if problem_type == 'Regression':
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            metrics = {
                'Test RMSE': round(test_rmse, 4),
                'Test MAE': round(test_mae, 4),
                'Test R2': round(test_r2, 4),
                'CV R2': f"{cv_scores['r2'].mean():.4f} ± {cv_scores['r2'].std():.4f}",
                'CV RMSE': f"{abs(cv_scores['neg_rmse'].mean()):.4f} ± {cv_scores['neg_rmse'].std():.4f}",
            }
        else:
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            if is_binary:
                test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                try:
                    test_auc = roc_auc_score(y_test, y_proba[:, 1])
                except:
                    test_auc = 0.5
                metrics = {
                    'Test Accuracy': round(test_acc, 4),
                    'Test F1 (weighted)': round(test_f1, 4),
                    'Test Precision': round(test_precision, 4),
                    'Test Recall': round(test_recall, 4),
                    'Test AUC-ROC': round(test_auc, 4),
                }
                # CV metrics
                metrics['CV F1'] = f"{cv_scores['f1_weighted'].mean():.4f} ± {cv_scores['f1_weighted'].std():.4f}"
                metrics['CV AUC'] = f"{cv_scores['roc_auc'].mean():.4f} ± {cv_scores['roc_auc'].std():.4f}"
            else:
                test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics = {
                    'Test Accuracy': round(test_acc, 4),
                    'Test F1 (weighted)': round(test_f1, 4),
                    'Test Precision': round(test_precision, 4),
                    'Test Recall': round(test_recall, 4),
                }
        
        elapsed = round(time.time() - start_time, 2)
        metrics['Train Time (s)'] = elapsed
        
        results.append({
            'Model': model_name,
            'Type': problem_type,
            **metrics
        })
        
        print(f"[STATUS] {model_name} done in {elapsed}s")
        
    except Exception as e:
        print(f"[WARNING] {model_name} failed: {str(e)}")
        results.append({
            'Model': model_name,
            'Type': problem_type,
            'Error': str(e)
        })

# === Create results DataFrame and save ===
results_df = pd.DataFrame(results)
print(f"\n[STATUS] Results summary:\n{results_df.to_string()}")

# Save detailed results
results_csv = os.path.join(OUTPUT_DIR, 'mo_results_detailed.csv')
results_df.to_csv(results_csv, index=False)
print(f"[STATUS] Saved detailed results: {results_csv}")

# === Find best model ===
if problem_type == 'Regression':
    # Higher R2 is better
    if 'Test R2' in results_df.columns:
        valid_results = results_df.dropna(subset=['Test R2'])
        if len(valid_results) > 0:
            best_row = valid_results.loc[valid_results['Test R2'].idxmax()]
            best_model_name = best_row['Model']
        else:
            best_model_name = results_df.iloc[0]['Model']
    else:
        best_model_name = results_df.iloc[0]['Model']
else:
    if 'Test F1 (weighted)' in results_df.columns:
        valid_results = results_df.dropna(subset=['Test F1 (weighted)'])
        if len(valid_results) > 0:
            best_row = valid_results.loc[valid_results['Test F1 (weighted)'].idxmax()]
            best_model_name = best_row['Model']
        else:
            best_model_name = results_df.iloc[0]['Model']
    else:
        best_model_name = results_df.iloc[0]['Model']

print(f"[STATUS] Best model: {best_model_name}")

# === Train best model for feature importance ===
best_model_class = None
for name, model in models.items():
    if name == best_model_name:
        best_model_class = model
        break

if best_model_class is None:
    best_model_class = list(models.values())[0]

best_model_class.fit(X_train_scaled, y_train)
y_pred_best = best_model_class.predict(X_test_scaled)

# === Feature Importance Analysis ===
feature_importance = None
if hasattr(best_model_class, 'feature_importances_'):
    importance_values = best_model_class.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    print(f"\n[STATUS] Top 10 Feature Importances:\n{feature_importance.head(10).to_string()}")
elif hasattr(best_model_class, 'coef_'):
    coef_values = np.abs(best_model_class.coef_).flatten() if len(best_model_class.coef_.shape) > 1 else np.abs(best_model_class.coef_)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': coef_values
    }).sort_values('importance', ascending=False)
    print(f"\n[STATUS] Top 10 Coefficients:\n{feature_importance.head(10).to_string()}")

# Save feature importance
if feature_importance is not None:
    fi_csv = os.path.join(OUTPUT_DIR, 'mo_feature_importance.csv')
    feature_importance.to_csv(fi_csv, index=False)
    print(f"[STATUS] Saved feature importance: {fi_csv}")

# === Overfitting Check ===
overfitting_notes = []
if 'CV F1' in results_df.columns and 'Test F1 (weighted)' in results_df.columns:
    best_result = results_df[results_df['Model'] == best_model_name].iloc[0]
    cv_f1 = float(str(best_result['CV F1']).split(' ± ')[0])
    test_f1 = float(best_result['Test F1 (weighted)'])
    gap = cv_f1 - test_f1
    if abs(gap) > 0.1:
        overfitting_notes.append(f"⚠️ Potential overfitting: CV F1={cv_f1:.4f} vs Test F1={test_f1:.4f} (gap={abs(gap):.4f})")
    else:
        overfitting_notes.append(f"✅ Good generalization: CV F1={cv_f1:.4f} vs Test F1={test_f1:.4f} (gap={abs(gap):.4f})")

if 'CV R2' in results_df.columns and 'Test R2' in results_df.columns:
    best_result = results_df[results_df['Model'] == best_model_name].iloc[0]
    cv_r2 = float(str(best_result['CV R2']).split(' ± ')[0])
    test_r2 = float(best_result['Test R2'])
    gap = cv_r2 - test_r2
    if abs(gap) > 0.1:
        overfitting_notes.append(f"⚠️ Potential overfitting: CV R²={cv_r2:.4f} vs Test R²={test_r2:.4f} (gap={abs(gap):.4f})")
    else:
        overfitting_notes.append(f"✅ Good generalization: CV R²={cv_r2:.4f} vs Test R²={test_r2:.4f} (gap={abs(gap):.4f})")

# === Confusion Matrix for classification ===
conf_matrix_str = ""
if problem_type != 'Regression':
    cm = confusion_matrix(y_test, y_pred_best)
    conf_matrix_str = f"\nConfusion Matrix:\n{cm}"

# === Generate Business Recommendation ===
business_rec = f"The best model is **{best_model_name}** which achieved "
if problem_type == 'Regression':
    best_result_row = results_df[results_df['Model'] == best_model_name].iloc[0]
    r2_val = best_result_row.get('Test R2', 'N/A')
    rmse_val = best_result_row.get('Test RMSE', 'N/A')
    business_rec += f"R² = {r2_val} and RMSE = {rmse_val}."
else:
    best_result_row = results_df[results_df['Model'] == best_model_name].iloc[0]
    f1_val = best_result_row.get('Test F1 (weighted)', 'N/A')
    acc_val = best_result_row.get('Test Accuracy', 'N/A')
    business_rec += f"F1-score = {f1_val} and Accuracy = {acc_val}."

# Feature importance top 5
top_features_str = ""
if feature_importance is not None:
    top5 = feature_importance.head(5)['feature'].tolist()
    top_features_str = ', '.join(top5)

# === Determine sample size and completeness ===
total_samples = len(df)
missing_pct = df[target_col].isna().mean() * 100 if df[target_col].isna().any() else 0

# === ===== WRITE REPORT ===== ===
report_content = f"""
# Mo Model Report

## Overview
- **Problem Type**: {problem_type}
- **Target Column**: `{target_col}`
- **Total Samples**: {total_samples}
- **Features Used**: {X.shape[1]}
- **Missing Rate in Target**: {missing_pct:.1f}%
- **Class Distribution**: {df[target_col].value_counts().to_dict() if problem_type != 'Regression' else f'Range: [{y.min():.4f}, {y.max():.4f}]'}

---

## Models Tested

| Model | Type | Time (s) |
|-------|------|----------|
"""

for _, row in results_df.iterrows():
    model_name = row.get('Model', '?')
    model_type = row.get('Type', problem_type)
    train_time = row.get('Train Time (s)', 'N/A')
    report_content += f"| {model_name} | {model_type} | {train_time} |\n"

report_content += f"""

## Results Comparison

### Metrics Table
\n