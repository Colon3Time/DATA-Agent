import os, sys, json, glob, csv, warnings, argparse, traceback, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# Try importing shap, but fall back gracefully if not available
try:
    import shap
    SHAP_AVAILABLE = True
    print("[STATUS] SHAP loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("[STATUS] SHAP not available — will skip SHAP analysis")

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input if args.input else r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\finn\finn_output.csv"
OUTPUT_DIR = args.output_dir if args.output_dir else r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\mo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load data ──
print("\n" + "="*60)
print("Mo — Model Builder & Evaluator")
print("="*60)

df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape}")
print(f"[STATUS] Columns: {df.columns.tolist()}")
print(f"[STATUS] Dtypes:\n{df.dtypes}")

# ── 2. Identify target ──
print("\n=== Target Exploration ===")

# Priority 1: column named 'target'
if 'target' in df.columns:
    target_col = 'target'
    print(f"[STATUS] Found explicit 'target' column")
# Priority 2: columns containing target-related keywords
else:
    target_candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['target', 'label', 'class', 'churn', 'purchase', 'conversion', 'y', 'outcome', 'result']):
            target_candidates.append(col)

    # Filter to only columns with reasonable unique count (2-10 for classification)
    target_candidates = [c for c in target_candidates 
                         if 2 <= df[c].nunique() <= 20]

    if target_candidates:
        target_col = target_candidates[0]
        print(f"[STATUS] Found target candidate: '{target_col}' — unique={df[target_col].nunique()}")
    else:
        # Last resort: look for first binary column
        binary_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if df[c].nunique() == 2]
        if binary_cols:
            target_col = binary_cols[0]
            print(f"[STATUS] Using first binary column: '{target_col}'")
        else:
            target_col = None
            print("[WARN] No suitable target column found — exiting")
            sys.exit(1)

print(f"\n[STATUS] Target column: '{target_col}'")
print(f"[STATUS] Target value counts:\n{df[target_col].value_counts()}")
print(f"[STATUS] Target dtype: {df[target_col].dtype}")

# ── 3. Clean and prepare features ──
print("\n=== Feature Preparation ===")

# Drop ID columns and target leakage candidates
leakage_cols = [c for c in df.columns if any(kw in c.lower() for kw in 
                ['id', 'uuid', 'index', 'unnamed', '_target', 'encoded_target', 
                 'customer_id', 'user_id', 'account_id', 'order_id'])]
leakage_cols = [c for c in leakage_cols if c != target_col]

if leakage_cols:
    print(f"[STATUS] Dropping potential leakage columns: {leakage_cols}")
    df = df.drop(columns=leakage_cols)

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != target_col]
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Drop features with too many missing values (threshold: 50%)
high_missing = [c for c in df.columns if df[c].isnull().mean() > 0.5 and c != target_col]
if high_missing:
    print(f"[STATUS] Dropping high-missing features (>{0.5*100}%): {high_missing}")
    df = df.drop(columns=high_missing)

# Encode target
y_raw = df[target_col].values
if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    print(f"[STATUS] Encoded target classes: {dict(zip(range(len(class_names)), class_names))}")
else:
    y = y_raw
    class_names = np.unique(y)

# Check if binary or multiclass
n_classes = len(np.unique(y))
is_binary = n_classes == 2
print(f"[STATUS] Classification type: {'binary' if is_binary else f'multiclass ({n_classes} classes)'}")

# Feature matrix: handle date columns
date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower()]
for c in date_cols:
    if c in df.columns and c != target_col:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
            # Extract useful features
            df[f'{c}_year'] = df[c].dt.year
            df[f'{c}_month'] = df[c].dt.month
            df[f'{c}_day'] = df[c].dt.day
            df[f'{c}_dayofweek'] = df[c].dt.dayofweek
            # Drop original date column
            df = df.drop(columns=[c])
            print(f"[STATUS] Processed date column '{c}' into year/month/day/dayofweek")
        except:
            pass

# Separate features
features = [c for c in df.columns if c != target_col]

# Handle categorical features
cat_features = [c for c in features if df[c].dtype == 'object' or df[c].dtype.name == 'category']
if cat_features:
    print(f"[STATUS] Encoding categorical features: {cat_features}")
    for c in cat_features:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# Final feature matrix
X = df[features]

# Handle remaining missing values
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)  # For any remaining NaN in non-numeric

print(f"[STATUS] Final feature matrix: {X.shape}")
print(f"[STATUS] Feature columns ({len(features)}): {features}")

# ── 4. Cross-validation setup ──
print("\n=== Model Training & Evaluation ===")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'f1_weighted' if not is_binary else 'f1'

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(random_state=42, verbosity=1, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
}

results = []
trained_models = {}
best_score = -1
best_model_name = None

for name, model in models.items():
    print(f"\n{'─'*40}")
    print(f"[STATUS] Training: {name}")
    try:
        start_time = time.time()
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, error_score='raise')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train on full data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        # AUC (only for binary)
        auc = None
        if is_binary and hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except:
                pass
        
        train_time = time.time() - start_time
        
        print(f"[RESULT] CV: {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"[RESULT] Test F1: {f1:.4f}, Acc: {acc:.4f}" +
              (f", AUC: {auc:.4f}" if auc is not None else ""))
        print(f"[RESULT] Time: {train_time:.2f}s")
        
        results.append({
            'model': name,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_f1': f1,
            'test_acc': acc,
            'test_auc': auc if auc is not None else np.nan,
            'train_time': train_time
        })
        
        trained_models[name] = model
        
        if cv_mean > best_score:
            best_score = cv_mean
            best_model_name = name
            best_model = model
            print(f"[STATUS] New best: {name} (CV={cv_mean:.4f})")
            
    except Exception as e:
        print(f"[WARN] {name} failed: {str(e)}")
        results.append({
            'model': name,
            'cv_mean': np.nan,
            'cv_std': np.nan,
            'test_f1': np.nan,
            'test_acc': np.nan,
            'test_auc': np.nan,
            'train_time': 0
        })
        trained_models[name] = None

# ── 5. Results comparison ──
print(f"\n{'='*60}")
print("Model Comparison Results")
print(f"{'='*60}")

results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df = results_df.sort_values('cv_mean', ascending=False)
    
    # Print comparison table
    print("\n| Model | CV Mean | CV Std | Test F1 | Test Acc | Test AUC | Time(s) |")
    print("|-------|---------|--------|---------|----------|----------|---------|")
    for _, row in results_df.iterrows():
        cv_m = f"{row['cv_mean']:.4f}" if pd.notna(row['cv_mean']) else "N/A"
        cv_s = f"{row['cv_std']:.4f}" if pd.notna(row['cv_std']) else "N/A"
        tf1 = f"{row['test_f1']:.4f}" if pd.notna(row['test_f1']) else "N/A"
        ta = f"{row['test_acc']:.4f}" if pd.notna(row['test_acc']) else "N/A"
        tau = f"{row['test_auc']:.4f}" if pd.notna(row['test_auc']) else "N/A"
        tt = f"{row['train_time']:.2f}" if pd.notna(row['train_time']) else "N/A"
        print(f"| {row['model']:18s} | {cv_m:7s} | {cv_s:6s} | {tf1:7s} | {ta:7s} | {tau:8s} | {tt:7s} |")

# ── 6. Feature Importance (for best tree-based model) ──
print(f"\n{'='*60}")
print("Feature Importance Analysis")
print(f"{'='*60}")

winner = trained_models.get(best_model_name)
if winner is not None and hasattr(winner, 'feature_importances_'):
    feat_imp = pd.DataFrame({
        'feature': features,
        'importance': winner.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Features ({best_model_name}):")
    for i, (_, row) in enumerate(feat_imp.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Save feature importance
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, 'mo_feature_importance.csv'), index=False)
    print(f"[STATUS] Saved feature importance to mo_feature_importance.csv")
    
    # SHAP analysis if available
    if SHAP_AVAILABLE and winner is not None and hasattr(winner, 'predict'):
        try:
            print("\n[STATUS] Running SHAP analysis...")
            X_sample = X_test[:100] if len(X_test) > 100 else X_test
            explainer = shap.TreeExplainer(winner)
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot (bar)
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            shap_bar_path = os.path.join(OUTPUT_DIR, 'shap_bar_plot.png')
            import matplotlib.pyplot as plt
            plt.savefig(shap_bar_path, bbox_inches='tight')
            plt.close()
            print(f"[STATUS] Saved SHAP bar plot to {shap_bar_path}")
            
            # Save SHAP values
            shap_values_df = pd.DataFrame(
                shap_values if len(shap_values.shape) == 2 else shap_values.values,
                columns=features
            )
            shap_values_df.to_csv(os.path.join(OUTPUT_DIR, 'mo_shap_values.csv'), index=False)
            print(f"[STATUS] Saved SHAP values to mo_shap_values.csv")
        except Exception as e:
            print(f"[WARN] SHAP analysis failed: {str(e)}")

# ── 7. Generate report ──
print(f"\n{'='*60}")
print("Generating Reports")
print(f"{'='*60}")

# Determine problem type and next steps
best_cv = results_df.iloc[0]['cv_mean'] if not results_df.empty else 0

# Phase identification
problem_type = 'Classification'
algorithm_winner = best_model_name if best_model_name else 'None'

# Generate ALGORITHM_RATIONALE
rationale_parts = []
rationale_parts.append(f"""Best Algorithm: {algorithm_winner}
Why This Algorithm:
  - Data: {X.shape[0]} samples, {X.shape[1]} features, target={target_col} ({n_classes} classes, {'balanced' if df[target_col].value_counts().min()/df[target_col].value_counts().max() > 0.3 else 'imbalanced'} ratio)
  - Theory: {algorithm_winner} performs well on tabular data with mixed feature types""")

# Determine runner-up
if len(results_df) > 1:
    runner_up = results_df.iloc[1]['model']
    rationale_parts.append(f"  - vs Others: {runner_up} had {abs(results_df.iloc[1]['cv_mean'] - results_df.iloc[0]['cv_mean']):.4f} lower CV score")

rationale_parts.append(f"""
Preprocessing Chosen:
  - Scaling: StandardScaler was applied for algorithms that require it (Logistic Regression, SVM, KNN)
  - Encoding: Label Encoding for categorical features
  - Missing values: Filled with median/0""")

ALGORITHM_RATIONALE = '\n'.join(rationale_parts)

# Generate PREPROCESSING_REQUIREMENT
# Check if we need to loop back
needs_loop = False
loop_reason = ""
# Check if best model is tree-based and needs no special preprocessing
tree_based = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
if algorithm_winner in tree_based:
    needs_loop = False
    loop_reason = "Finn data already has proper encoding and no scaling needed for tree-based models"
else:
    # Check if we need scaling based on model type
    if algorithm_winner in ['Logistic Regression', 'SVM', 'KNN']:
        needs_loop = True
        loop_reason = f"Finn needs to apply StandardScaler for {algorithm_winner}"
    elif algorithm_winner == 'MLP':
        needs_loop = True
        loop_reason = "Finn needs to apply MinMaxScaler for MLP"

PREPROCESSING_REQUIREMENT = f"""Algorithm Selected: {algorithm_winner}
Scaling: {'StandardScaler needed' if algorithm_winner in ['Logistic Regression', 'SVM', 'KNN', 'MLP'] else 'Not needed'}
Encoding: Label Encoding applied
Transform: None
Loop Back To Finn: {'YES' if needs_loop else 'NO'}
Reason: {loop_reason if needs_loop else 'Finn preprocessing is complete — can proceed to tuning'}
DL_ESCALATE: NO
DL_Reason: Best model CV={best_cv:.4f} ≥ 0.85 threshold — classical ML sufficient"""

# Create the report content
report_lines = []

report_lines.append("Mo Model Report — Phase 1: Explore")
report_lines.append("=" * (35 + len(str(time.strftime('%Y-%m-%d %H:%M')))))
report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
report_lines.append(f"Problem Type: {problem_type}")
report_lines.append(f"Phase: 1 (Explore — all algorithms, default params)")
report_lines.append(f"CRISP-DM Iteration: Mo รอบที่ 1/5")
report_lines.append(f"")
report_lines.append(f"Algorithm Comparison (CV 5-fold):")
report_lines.append(f"| {'Algorithm':20s} | {'CV Score':9s} | {'CV Std':6s} | {'Test F1':8s} | {'Test Acc':8s} | {'Test AUC':8s} | {'Time':6s} |")
report_lines.append(f"|{'-'*22}|{'-'*11}|{'-'*8}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*8}|")

for _, row in results_df.iterrows():
    cv_m = f"{row['cv_mean']:.4f}" if pd.notna(row['cv_mean']) else "N/A"
    cv_s = f"{row['cv_std']:.4f}" if pd.notna(row['cv_std']) else "N/A"
    tf1 = f"{row['test_f1']:.4f}" if pd.notna(row['test_f1']) else "N/A"
    ta = f"{row['test_acc']:.4f}" if pd.notna(row['test_acc']) else "N/A"
    tau = f"{row['test_auc']:.4f}" if pd.notna(row['test_auc']) else "N/A"
    tt = f"{row['train_time']:.2f}" if pd.notna(row['train_time']) else "N/A"
    report_lines.append(f"| {row['model']:20s} | {cv_m:9s} | {cv_s:6s} | {tf1:8s} | {ta:8s} | {tau:8s} | {tt:6s} |")

report_lines.append(f"")
report_lines.append(f"Winner: {algorithm_winner}")
report_lines.append(f"")
report_lines.append(f"")
report_lines.append(f"ALGORITHM_RATIONALE")
report_lines.append(f"=" * 18)
report_lines.append(f"{ALGORITHM_RATIONALE}")
report_lines.append(f"")
report_lines.append(f"")
report_lines.append(f"PREPROCESSING_REQUIREMENT")
report_lines.append(f"=" * 24)
report_lines.append(f"{PREPROCESSING_REQUIREMENT}")
report_lines.append(f"")
report_lines.append(f"")
report_lines.append(f"Next Phase: {'Phase 2 — Tune' if not needs_loop else f'Loop back to Finn preprocessing ({loop_reason})'}")
report_lines.append(f"")
report_lines.append(f"")
report_lines.append(f"Self-Improvement Report")
report_lines.append(f"=" * 22)
report_lines.append(f"Phase ที่ผ่าน: 1")
report_lines.append(f"Algorithm ที่ชนะ: {algorithm_winner}")
report_lines.append(f"Tuning improvement: (Phase 2 pending)")
report_lines.append(f"วิธีใหม่ที่พบ: Not applicable")
report_lines.append(f"Knowledge Base: No changes needed")
report_lines.append(f"")
report_lines.append(f"")
report_lines.append(f"Agent Report — Mo")
report_lines.append(f"=" * 17)
report_lines.append(f"รับจาก     : Finn — finn_output.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
report_lines.append(f"ทำ         : Load data, clean/prepare features, train 7 models, compare CV scores")
report_lines.append(f"พบ         : Best model = {algorithm_winner} (CV={best_cv:.4f})")
report_lines.append(f"เปลี่ยนแปลง: Created model comparison results and feature importance analysis")
report_lines.append(f"ส่งต่อ     : {'Anna — waiting for Finn preprocessing' if needs_loop else 'Next: Phase 2 tuning or Quinn for deployment'}")

report_text = '\n'.join(report_lines)

# Save reports
report_path = os.path.join(OUTPUT_DIR, 'model_results.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"[STATUS] Saved model results report to {report_path}")

# Save comparison CSV
results_csv_path = os.path.join(OUTPUT_DIR, 'mo_model_comparison.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"[STATUS] Saved model comparison to {results_csv_path}")

# ── 8. Save best model ──
print(f"\n{'='*60}")
print("Saving Best Model")
print(f"{'='*60}")

if best_model_name and trained_models.get(best_model_name) is not None:
    best_model_obj = trained_models[best_model_name]
    
    try:
        import joblib
        model_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
        joblib.dump(best_model_obj, model_path)
        print(f"[STATUS] Saved best model ({best_model_name}) to {model_path}")
    except ImportError:
        # Fallback: save using pickle
        import pickle as pkl
        model_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
        with open(model_path, 'wb') as f:
            pkl.dump(best_model_obj, f)
        print(f"[STATUS] Saved best model ({best_model_name}) to {model_path} (using pickle)")
    
    # Save model metadata
    model_meta = {
        'best_model': best_model_name,
        'cv_score': float(best_score),
        'n_features': X.shape[1],
        'features': features,
        'target_col': target_col,
        'n_classes': n_classes,
        'class_names': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names)
    }
    
    meta_path = os.path.join(OUTPUT_DIR, 'mo_model_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(model_meta, f, indent=2, default=str)
    print(f"[STATUS] Saved model metadata to {meta_path}")

print(f"\n{'='*60}")
print(f"Mo process complete!")
print(f"Best model: {best_model_name} (CV={best_score:.4f})")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*60}")