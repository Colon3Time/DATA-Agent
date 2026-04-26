import argparse, os, pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
parser.add_argument('--save-script', default='')
parser.add_argument('--save-report', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input if args.input else r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test3\output\finn\finn_output.csv'
OUTPUT_DIR = args.output_dir if args.output_dir else r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test3\output\mo'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCRIPT_PATH = args.save_script if args.save_script else os.path.join(OUTPUT_DIR, 'mo_script.py')
REPORT_PATH = args.save_report if args.save_report else os.path.join(OUTPUT_DIR, 'mo_report.md')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'mo_output.csv')

# Install missing packages if needed
try:
    import xgboost
except ImportError:
    os.system('pip install xgboost -q')
    import xgboost

try:
    import lightgbm
except ImportError:
    os.system('pip install lightgbm -q')
    import lightgbm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
                            classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')

# === [1] Explore Data ===
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Nulls:\n{df.isnull().sum()}')
print(f'[STATUS] Target distributions:')

# Identify target column
potential_targets = ['return_flag', 'is_return', 'returned', 'target', 'label']
target_col = None
for t in potential_targets:
    if t in df.columns:
        target_col = t
        break

if target_col is None:
    # Try to find binary/classification column
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].nunique() <= 10 and col != 'customer_id':
            target_col = col
            break
    if target_col is None:
        # Use last column as target
        target_col = df.columns[-1]

print(f'[STATUS] Using target: {target_col}')
print(df[target_col].value_counts().to_string())

# === [2] Prepare Features ===
df_model = df.copy()

# Drop non-feature columns
cols_to_drop = [target_col]
for skip in ['customer_id', 'user_id', 'transaction_id', 'order_id', 'Unnamed: 0']:
    if skip in df_model.columns and skip != target_col:
        cols_to_drop.append(skip)

X = df_model.drop(columns=cols_to_drop)
y = df_model[target_col]

# Encode categorical features
cat_cols = X.select_dtypes(include=['object', 'category']).columns
for c in cat_cols:
    X[c] = LabelEncoder().fit_transform(X[c].astype(str))

# Handle missing values
X = X.fillna(X.median(numeric_only=True))
for c in X.select_dtypes(include=['object']).columns:
    X[c] = X[c].fillna('Unknown')
    X[c] = LabelEncoder().fit_transform(X[c])

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Convert target to numeric if needed
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

print(f'[STATUS] Features: {X.shape[1]}, Samples: {len(X)}, Target classes: {len(np.unique(y))}')

# === [3] Determine Problem Type ===
n_classes = len(np.unique(y))
if n_classes <= 10:
    problem_type = 'classification'
    is_binary = (n_classes == 2)
else:
    problem_type = 'regression'
    is_binary = False

print(f'[STATUS] Problem type: {problem_type}, Classes: {n_classes}')

# === [4] Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None)

# === [5] Train Multiple Models ===
results = []
models = {}
predictions = {}

if problem_type == 'classification':
    model_configs = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced' if is_binary else None)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced' if is_binary else None)),
        ('XGBoost', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False)),
        ('LightGBM', lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced' if is_binary else None, verbose=-1))
    ]
    
    for name, model in model_configs:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # CV score
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            result = {
                'model': name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # AUC for binary classification
            if is_binary:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    result['auc'] = roc_auc_score(y_test, y_proba)
                except:
                    result['auc'] = 0.0
                
                try:
                    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
                except:
                    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            else:
                try:
                    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
                except:
                    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            
            result['cv_mean'] = cv_scores.mean()
            result['cv_std'] = cv_scores.std()
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                result['has_importance'] = True
                feat_imp = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                result['top_features'] = feat_imp.head(5)['feature'].tolist()
            elif hasattr(model, 'coef_') and len(model.coef_.shape) == 1:
                result['has_importance'] = True
                feat_imp = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(model.coef_)
                }).sort_values('importance', ascending=False)
                result['top_features'] = feat_imp.head(5)['feature'].tolist()
            else:
                result['has_importance'] = False
                result['top_features'] = []
            
            results.append(result)
            models[name] = model
            predictions[name] = y_pred
            
            print(f'[STATUS] {name}: F1={result["f1"]:.4f}, CV={result["cv_mean"]:.4f}±{result["cv_std"]:.4f}')
            
        except Exception as e:
            print(f'[STATUS] {name} failed: {str(e)}')
            results.append({'model': name, 'error': str(e)})

else:  # Regression
    model_configs = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge(alpha=1.0, random_state=42)),
        ('Lasso', Lasso(alpha=0.01, random_state=42)),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('XGBoost', XGBRegressor(n_estimators=100, random_state=42)),
        ('LightGBM', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    for name, model in model_configs:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
            
            result = {
                'model': name,
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
                'cv_rmse_std': np.sqrt(-cv_scores).std()
            }
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                result['has_importance'] = True
                feat_imp = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                result['top_features'] = feat_imp.head(5)['feature'].tolist()
            elif hasattr(model, 'coef_'):
                result['has_importance'] = True
                feat_imp = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(model.coef_)
                }).sort_values('importance', ascending=False)
                result['top_features'] = feat_imp.head(5)['feature'].tolist()
            else:
                result['has_importance'] = False
                result['top_features'] = []
            
            results.append(result)
            models[name] = model
            predictions[name] = y_pred
            
            print(f'[STATUS] {name}: RMSE={result["rmse"]:.4f}, R2={result["r2"]:.4f}, CV_RMSE={result["cv_rmse_mean"]:.4f}±{result["cv_rmse_std"]:.4f}')
            
        except Exception as e:
            print(f'[STATUS] {name} failed: {str(e)}')
            results.append({'model': name, 'error': str(e)})

# === [6] Find Best Model ===
valid_results = [r for r in results if 'error' not in r]

if valid_results:
    if problem_type == 'classification':
        best_model = max(valid_results, key=lambda r: r['f1'])
        best_name = best_model['model']
        best_f1 = best_model['f1']
        best_cv = best_model['cv_mean']
        best_metric = 'F1'
    else:
        best_model = min(valid_results, key=lambda r: r['rmse'])
        best_name = best_model['model']
        best_f1 = best_model['rmse']
        best_cv = best_model['cv_rmse_mean']
        best_metric = 'RMSE'
    
    print(f'\n[STATUS] Best model: {best_name} ({best_metric}={best_f1:.4f}, CV={best_cv:.4f})')
    
    # === [7] Overfitting Check ===
    best_obj = models[best_name]
    
    if problem_type == 'classification':
        train_pred = best_obj.predict(X_train)
        train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
        test_f1 = best_model['f1']
        gap = abs(train_f1 - test_f1)
        print(f'[STATUS] Overfitting check: Train F1={train_f1:.4f}, Test F1={test_f1:.4f}, Gap={gap:.4f}')
        overfit_status = 'ผ่าน' if gap < 0.15 else '⚠️ อาจ overfit'
    else:
        train_pred = best_obj.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = best_model['rmse']
        gap = abs(train_rmse - test_rmse)
        print(f'[STATUS] Overfitting check: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}, Gap={gap:.4f}')
        overfit_status = 'ผ่าน' if gap < 0.15 * train_rmse else '⚠️ อาจ overfit'
    
    # === [8] Save Results CSV ===
    results_df = pd.DataFrame(valid_results)
    # Keep only serializable columns
    output_cols = [c for c in results_df.columns if c not in ['has_importance', 'top_features']]
    results_df[output_cols].to_csv(OUTPUT_CSV, index=False)
    print(f'[STATUS] Saved results CSV: {OUTPUT_CSV}')

# === [9] Generate Report ===
report_lines = []
report_lines.append('# Mo Model Report\n')
report_lines.append('---\n')
report_lines.append(f'**Problem Type:** {problem_type.capitalize()}')
report_lines.append(f'**Target Variable:** {target_col}')
report_lines.append(f'**Number of Features:** {X.shape[1]}')
report_lines.append(f'**Number of Samples:** {len(X)}')
report_lines.append(f'**Classes/Unique Values:** {n_classes}\n')
report_lines.append('## Models Tested\n')

if valid_results:
    report_lines.append('| Model | ' + ' | '.join([k for k in valid_results[0].keys() if k not in ['has_importance', 'top_features', 'error']]) + ' |')
    report_lines.append('|' + '---|' * len([k for k in valid_results[0].keys() if k not in ['has_importance', 'top_features', 'error']]))
    
    for r in valid_results:
        vals = []
        for k in r.keys():
            if k in ['has_importance', 'top_features', 'error']:
                continue
            v = r[k]
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        report_lines.append('| ' + ' | '.join(vals) + ' |')
    
    report_lines.append(f'\n## Best Model\n')
    report_lines.append(f'**{best_name}** — {best_metric} = {best_f1:.4f}, CV = {best_cv:.4f}\n')
    
    if best_model.get('has_importance', False) and best_model.get('top_features'):
        report_lines.append('### Top 5 Features\n')
        for i, feat in enumerate(best_model['top_features'], 1):
            report_lines.append(f'{i}. **{feat}**')
        report_lines.append('')
    
    report_lines.append('### Overfitting Check\n')
    report_lines.append(f'{overfit_status}\n')
    
    report_lines.append('### Business Recommendation\n')
    report_lines.append(f'Using **{best_name}** model for {target_col} prediction:')
    report_lines.append(f'- Achieved {best_metric} of {best_f1:.4f} on test data')
    report_lines.append(f'- Cross-validation score: {best_cv:.4f}')
    report_lines.append('- Suitable for production deployment' if 'ผ่าน' in overfit_status else '- May need further tuning to reduce overfitting')
    report_lines.append('')
    
    # Cross-validation check report
    report_lines.append('---\n')
    report_lines.append('## Self-Improvement Report\n')
    report_lines.append(f'**วิธีที่ใช้ครั้งนี้:** Model comparison ({len(valid_results)} models)')
    report_lines.append(f'**เหตุผลที่เลือก:** ทดสอบหลาย algorithm เพื่อหา best fit')
    report_lines.append(f'**วิธีใหม่ที่พบ:** ไม่พบวิธีใหม่')
    report_lines.append(f'**จะนำไปใช้ครั้งหน้า:** ใช่ เพราะการเปรียบเทียบหลาย models ช่วยให้เลือก model ที่เหมาะสมที่สุด')
    report_lines.append(f'**Knowledge Base:** ไม่มีการเปลี่ยนแปลง')
else:
    report_lines.append('⚠️ No models were successfully trained.\n')

report_content = '\n'.join(report_lines)

# Save report
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f'[STATUS] Saved report: {REPORT_PATH}')

print(f'\n[STATUS] ✅ Done! Output files:')
print(f'  - Report: {REPORT_PATH}')
print(f'  - Results CSV: {OUTPUT_CSV}')
print(f'  - Script: {SCRIPT_PATH}')
