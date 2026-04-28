# Mo — Model Builder & Evaluator

I'll start by reading the Knowledge Base, then analyze the input data and build models.

## Step 1: Read Knowledge Base

Let me first check what's available in the knowledge base.

```python
import os, sys, json, glob, csv, warnings, argparse, traceback, time
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

INPUT_PATH = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\finn\finn_output.csv"
OUTPUT_DIR = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\mo"

# Check KB files
kb_dir = r"C:\Users\Amorntep\DATA-Agent\knowledge_base"
if os.path.exists(kb_dir):
    kb_files = [f for f in os.listdir(kb_dir) if f.endswith('.md')]
    print(f"[STATUS] KB files found: {kb_files}")
    for kb in kb_files:
        with open(os.path.join(kb_dir, kb), 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"\n=== {kb} (first 500 chars) ===")
        print(content[:500])
else:
    print(f"[STATUS] No KB directory at {kb_dir}")

# Check for Finn output to understand the data
print(f"\n[STATUS] Loading Finn output...")
if os.path.exists(INPUT_PATH):
    df = pd.read_csv(INPUT_PATH)
    print(f"[STATUS] Shape: {df.shape}")
    print(f"[STATUS] Columns: {df.columns.tolist()}")
    print(f"[STATUS] Types:\n{df.dtypes}")
    print(f"[STATUS] Head:\n{df.head()}")
    print(f"[STATUS] Describe:\n{df.describe()}")
else:
    print(f"[ERROR] Input file not found: {INPUT_PATH}")
    sys.exit(1)
```

## Step 2: Explore Data — Find Target Column

Let me understand the data structure and identify the target column.

```python
import os, sys, json, warnings, traceback, time
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

INPUT_PATH = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\finn\finn_output.csv"
OUTPUT_DIR = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\mo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape}")

# Explore target candidates
print("\n=== Data Overview ===")
print(f"Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  {i:3d}. {col:40s} dtype={df[col].dtype}  unique={df[col].nunique()}  nulls={df[col].isna().sum()}")

print("\n=== Last 30 columns (likely target/derived) ===")
for col in df.columns[-30:]:
    print(f"  {col:40s} dtype={df[col].dtype}  unique={df[col].nunique()}  sample={df[col].iloc[:3].tolist()}")

# Look for target-like columns
target_candidates = []
for col in df.columns:
    col_lower = col.lower()
    if any(kw in col_lower for kw in ['target', 'label', 'class', 'churn', 'purchase', 'conversion', 'score', 'category', 'segment', 'cluster']):
        target_candidates.append(col)

print(f"\n=== Target Candidates ===")
for col in target_candidates:
    print(f"  {col}: unique={df[col].nunique()}, nulls={df[col].isna().sum()}, value_counts={df[col].value_counts().head().to_dict()}")

# Check for binary columns (0/1 or Yes/No)
binary_cols = []
for col in df.columns:
    if df[col].nunique() <= 2:
        binary_cols.append(col)
        if df[col].dtype == 'object':
            print(f"  {col}: values={df[col].unique().tolist()}")
        else:
            print(f"  {col}: values={df[col].unique().tolist()}")

# Look for the most likely target — check if any column has 'target' in name
target_col = None
for col in df.columns:
    if 'target' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
        target_col = col
        print(f"\n[INFO] Found target column: {target_col}")
        break

if target_col is None:
    # If no explicit target, look for business-relevant binary columns
    print("\n[INFO] No explicit target found. Looking for business-relevant columns...")
    # Check for common business targets
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['churn', 'purchase', 'conversion', 'will_buy', 'buy_flag', 'is_churn', 'responded']):
            target_col = col
            print(f"[INFO] Found business target: {target_col}")
            break

if target_col is None:
    # Last resort: pick a binary column
    for col in df.columns:
        if df[col].nunique() == 2 and 'id' not in col.lower():
            target_col = col
            print(f"[INFO] Using binary column as target: {target_col}")
            break

if target_col is None:
    print("[ERROR] No suitable target column found")
    print("All columns:", df.columns.tolist())
    sys.exit(1)

print(f"\n=== Final Target: {target_col} ===")
print(df[target_col].value_counts())
print(f"dtype: {df[target_col].dtype}")
```

Let me also check the Finn processing history to understand what features were created.

```python
# Check Finn output for processing history
finn_dir = os.path.dirname(INPUT_PATH)
finn_files = os.listdir(finn_dir)
print(f"[STATUS] Finn output directory files: {finn_files}")

# Check if there's a Finn report
finn_report = os.path.join(finn_dir, 'finn_report.md')
if os.path.exists(finn_report):
    with open(finn_report, 'r', encoding='utf-8') as f:
        print(finn_report[:2000])

# Load full data inspection
df = pd.read_csv(INPUT_PATH)
print(f"\n=== Full Column Analysis ===")
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        stats = f"min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}, std={df[col].std():.3f}"
    else:
        stats = f"unique={df[col].nunique()}, top={df[col].mode().iloc[0] if df[col].nunique() > 0 else 'NA'}"
    print(f"  {col:40s} {str(df[col].dtype):10s} {stats}")
```

## Step 3: Phase 1 — Explore All Algorithms

Now I'll build and compare all models.

```python
import os, sys, json, warnings, traceback, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings('ignore')

INPUT_PATH = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\finn\finn_output.csv"
OUTPUT_DIR = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\mo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Phase 1: Explore — Test all algorithms with default params ──
print("=" * 60)
print("PHASE 1: EXPLORE — Testing all algorithms with default params")
print("=" * 60)

# Load data
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape}")

# ── Find target and identify features ──
# Check Finn report first
finn_report_path = os.path.join(os.path.dirname(INPUT_PATH), 'finn_report.md')
target_col = None
if os.path.exists(finn_report_path):
    with open(finn_report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Look for target column info in Finn report
    import re
    target_match = re.search(r'target(?:\s*column)?[:\s]+([\w_]+)', content, re.IGNORECASE)
    if target_match:
        target_col = target_match.group(1)
        print(f"[INFO] Target from Finn report: {target_col}")

if target_col is None or target_col not in df.columns:
    # Auto-detect target
    for col in df.columns:
        if 'target' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
            target_col = col
            break
    if target_col is None:
        for col in df.columns:
            if 'churn' in col.lower() or 'purchase' in col.lower() or 'conversion' in col.lower():
                target_col = col
                break
    if target_col is None:
        for col in df.columns:
            if df[col].nunique() == 2 and 'id' not in col.lower():
                target_col = col
                break

if target_col is None:
    print("[ERROR] No target column found")
    sys.exit(1)

print(f"[INFO] Target column: {target_col}")
print(f"[INFO] Target distribution:\n{df[target_col].value_counts()}")

# ── Validate input ──
# Drop ID/leakage columns
cols_to_drop = []
for col in df.columns:
    col_lower = col.lower()
    if any(kw in col_lower for kw in ['customer_id', 'order_id', 'user_id', 'account_id', 'transaction_id', 
                                       'unnamed', 'index', 'id.']):
        if col != target_col:
            cols_to_drop.append(col)

# Drop target-like leakage columns (columns that contain target name)
target_lower = target_col.lower()
leak_cols = [c for c in df.columns if c != target_col and target_lower in c.lower()]
cols_to_drop.extend(leak_cols)

if cols_to_drop:
    print(f"[INFO] Dropping {len(cols_to_drop)} columns: {cols_to_drop[:10]}...")
    df = df.drop(columns=cols_to_drop)

# Handle missing values
df = df.dropna()  # Simple approach for now

# ── Prepare X and y ──
y = df[target_col]
X = df.drop(columns=[target_col])

# Handle categorical features
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"[INFO] Categorical columns ({len(cat_cols)}): {cat_cols[:5]}...")

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Detect problem type
unique_targets = y.nunique()
problem_type = 'binary' if unique_targets == 2 else 'multiclass' if unique_targets < 20 else 'regression'
print(f"[INFO] Problem type: {problem_type} (unique targets: {unique_targets})")

# Encode target if needed
target_le = LabelEncoder()
if y.dtype == 'object':
    y_encoded = target_le.fit_transform(y.astype(str))
    y_binary = y_encoded
else:
    y_encoded = y.values
    y_binary = y.values

# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary if problem_type == 'binary' else None
)
print(f"[STATUS] Train: {X_train.shape}, Test: {X_test.shape}")

# ── Scale for linear models ──
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Algorithm Pool ──
algorithms = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

# Add SVM only if n < 5000
if len(df) < 5000:
    algorithms['SVM'] = SVC(kernel='rbf', random_state=42, probability=True)

# ── Cross-validation settings ──
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Results storage ──
results = {}
best_model = None
best_score = -1
best_name = None

print(f"\n{'='*60}")
print("Model Evaluation (5-fold CV + Test Set)")
print(f"{'='*60}")

for name, model in algorithms.items():
    start_time = time.time()
    
    try:
        # Scale features for distance-based models
        if name in ['Logistic Regression', 'SVM', 'KNN']:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train.values, X_test.values
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1_weighted')
        
        # Train on full training set
        model.fit(X_tr, y_train)
        
        # Test predictions
        y_pred = model.predict(X_te)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # AUC if binary
        auc = None
        if problem_type == 'binary':
            try:
                y_proba = model.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                pass
        
        elapsed = time.time() - start_time
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'test_f1': f1,
            'test_precision': precision,
            'test_recall': recall,
            'test_auc': auc,
            'time': elapsed,
            'model': model
        }
        
        print(f"\n{'─'*50}")
        print(f"{name}:")
        print(f"  CV Score (F1): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test F1:       {f1:.4f}")
        print(f"  Test Precision:{precision:.4f}")
        print(f"  Test Recall:   {recall:.4f}")
        if auc is not None:
            print(f"  Test AUC:      {auc:.4f}")
        print(f"  Time:          {elapsed:.2f}s")
        
        # Track best
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
            
    except Exception as e:
        print(f"\n[ERROR] {name}: {e}")
        results[name] = {
            'cv_mean': -1,
            'cv_std': 0,
            'test_accuracy': -1,
            'test_f1': -1,
            'test_precision': -1,
            'test_recall': -1,
            'test_auc': None,
            'time': 0,
            'model': None
        }

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_name} (Test F1: {best_score:.4f})")
print(f"{'='*60}")

# ── Feature Importance for tree-based models ──
feature_importance = None
if best_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nTop 10 Features ({best_name}):")
    print(feature_importance.head(10).to_string(index=False))

# ── Confusion Matrix ──
if best_name:
    y_pred_best = best_model.predict(X_test_scaled if best_name in ['Logistic Regression', 'SVM', 'KNN'] else X_test.values)
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix ({best_name}):")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_best))

# ── Check if Deep Learning escalation needed ──
dl_escalate = best_score < 0.85
print(f"\n[INFO] DL escalation needed: {dl_escalate} (best F1: {best_score:.4f})")

# ── Preprocessing Requirement ──
scaling_needed = 'StandardScaler (for LR/SVM/KNN)'
encoding_needed = 'Label Encoding (already done)'
special_transform = 'None'

if best_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    scaling_needed = 'None (tree-based)'
    special_transform = 'None (tree-based handles non-linear)'
elif best_name in ['Logistic Regression', 'SVM', 'KNN']:
    scaling_needed = 'StandardScaler (done)'

loop_back = 'NO'  # Finn already provided scaled data

# ── Save output CSV with predictions ──
# Predict on full dataset
full_X_scaled = scaler.transform(X) if best_name in ['Logistic Regression', 'SVM', 'KNN'] else X.values
full_pred = best_model.predict(full_X_scaled)

df_output = df.copy()
df_output['mo_prediction'] = full_pred
if problem_type == 'binary':
    try:
        full_proba = best_model.predict_proba(full_X_scaled)[:, 1]
        df_output['mo_prediction_probability'] = full_proba
    except:
        pass

output_csv = os.path.join(OUTPUT_DIR, 'mo_output.csv')
df_output.to_csv(output_csv, index=False)
print(f"\n[STATUS] Saved mo_output.csv: {output_csv}")

# Display results summary table
print(f"\n{'='*60}")
print("RESULTS SUMMARY TABLE")
print(f"{'='*60}")
results_summary = []
for name, r in results.items():
    results_summary.append({
        'Algorithm': name,
        'CV Mean': f"{r['cv_mean']:.4f}",
        'CV Std': f"{r['cv_std']:.4f}",
        'Test F1': f"{r['test_f1']:.4f}",
        'Test AUC': f"{r['test_auc']:.4f}" if r['test_auc'] else 'N/A',
        'Time(s)': f"{r['time']:.2f}"
    })
results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))
```

## Step 4: Phase 2 — Hyperparameter Tuning

Now I'll tune the best model.

```python
import os, sys, json, warnings, traceback, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

INPUT_PATH = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\finn\finn_output.csv"
OUTPUT_DIR = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_pulsecart_customer_behavior\output\mo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Phase 2: Hyperparameter Tuning ──
print("=" * 60)
print("PHASE 2: HYPERPARAMETER TUNING")
print("=" * 60)

# Load data
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape}")

# Target detection (same as Phase 1)
target_col = None
finn_report_path = os.path.join(os.path.dirname(INPUT_PATH), 'finn_report.md')
if os.path.exists(finn_report_path):
    import re
    with open(finn_report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    target_match = re.search(r'target(?:\s*column)?[:\s]+([\w_]+)', content, re.IGNORECASE)
    if target_match:
        target_col = target_match.group(1)

if target_col is None or target_col not in df.columns:
    for col in df.columns:
        if 'target' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
            target_col = col; break
if target_col is None:
    for col in ['churn', 'purchase', 'conversion', 'will_buy', 'buy_flag', 'is_churn', 'responded']:
        if col in df.columns: target_col = col; break
if target_col is None:
    for col in df.columns:
        if df[col].nunique() == 2 and 'id' not in col.lower(): target_col = col; break

print(f"[INFO] Target: {target_col}")

# Drop ID/leakage columns
cols_to_drop = [c for c in df.columns if any(kw in c.lower() for kw in ['customer_id', 'order_id', 'user_id', 'account_id', 'unnamed', 'index', 'id.']) and c != target_col]
target_lower = target_col.lower()
leak_cols = [c for c in df.columns if c != target_col and target_lower in c.lower()]
df = df.drop(columns=cols_to_drop + leak_cols)
df = df.dropna()

# Prepare data
y = df[target_col]
X = df.drop(columns=[target_col])

# Encode categorical
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Encode target
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y.astype(str))
else:
    y = y.values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Identify best algorithm from full comparison (re-run Phase 1 logic) ──
algorithms_test = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1), X_train_s, X_test_s),
    'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_train.values, X_test.values),
    'XGBoost': (XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0), X_train.values, X_test.values),
    'LightGBM': (LGBMClassifier(n_estimators=100, random_state=42, verbose=-1), X_train.values, X_test.values),
    'KNN': (KNeighborsClassifier(), X_train_s, X_test_s),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = {}
for name, (model, X_tr, X_te) in algorithms_test.items():
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1_weighted')
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    f1 = f1_score(y_test, y_pred, average='weighted')
    scores[name] = {'cv_mean': cv_scores.mean(), 'test_f1': f1}

best_name = max(scores, key=lambda k: scores[k]['test_f1'])
best_f1_default = scores[best_name]['test_f1']
print(f"\n[INFO] Best algorithm (Phase 1): {best_name} (Test F1: {best_f1_default:.4f})")

# ── Hyperparameter Tuning ──
print(f"\n{'─'*50}")
print(f"Tuning {best_name}...")

n_iter = 50
param_dist = {}
estimator = None

if best_name == 'XGBoost':
    estimator = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
    }
    X_tr, X_te = X_train.values, X_test.values
elif best_name == 'LightGBM':
    estimator = LGBMClassifier(random_state=42, verbose=-1)
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
    }
    X_tr, X_te = X_train.values, X_test.values
elif best_name == 'Random Forest':
    estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    X_tr, X_te = X_train.values, X_test.values
elif best_name == 'Logistic Regression':
    estimator = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    param_dist = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
    }
    X_tr, X_te = X_train_s, X_test_s
elif best_name == 'SVM':
    estimator = SVC(random_state=42, probability=True)
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto'],
    }
    X_tr, X_te = X_train_s, X_test_s
elif best_name == 'KNN':
    estimator = KNeighborsClassifier()
    param_dist = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
    }
    X_tr, X_te = X_train_s, X_test_s

if estimator is not None and param_dist:
    search = RandomizedSearchCV(
        estimator, param_dist,
        n_iter=n_iter, cv=5, scoring='f1_weighted',
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_tr, y_train)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = search.best_score_
    
    # Test with tuned model
    y_pred_tuned = best_model.predict(X_te)
    f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    
    print(f"\nBest CV Score: {best_cv_score:.4f}")
    print(f"Best Params: {best_params}")
    print(f"Tuned Test F1: {f1_tuned:.4f} (vs default {best_f1_default:.4f})")
    print(f"Improvement: {(f1_tuned - best_f1_default) / best_f1_default * 100:+.2f}%")
    
    # Overfitting check
    overfit_gap = best_cv_score - f1_tuned
    print(f"Overfitting gap: {overfit_gap:.4f} {'✓ Pass' if overfit_gap < 0.05 else '⚠ Check'}")
    
    # AUC if binary
    if len(np.unique(y_test)) == 2:
        y_proba = best_model.predict_proba(X_te)[:, 1]
        auc_tuned = roc_auc_score(y_test, y_proba)
        print(f"Tuned Test AUC: {auc_tuned:.4f}")
else:
    print("[ERROR] Could not set up tuning")
    best_model = algorithms_test[best_name][0]
    best_model.fit(X_tr, y_train)
    y_pred_tuned = best_model.predict(X_te)
    f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')
    best_params = {}
    best_cv_score = 0

# ── Feature Importance (if tree-based) ──
if best_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nTop 10 Features ({best_name}):")
    for i, row in feat_imp.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")

# ── SHAP Analysis ──
print(f"\n{'─'*50}")
print("SHAP Values Analysis (first 100 test rows)")
try:
    if best_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_te[:100])
        shap.summary_plot(shap_values, X_te[:100], feature_names=X.columns.tolist())
        print("[STATUS] SHAP analysis complete")
except Exception as e:
    print(f"[INFO] SHAP skipped: {e}")

# ── Decision: Use tuned or default? ──
improvement_pct = (f1_tuned - best_f1_default) / max(best_f1_default, 0.001) * 100
use_tuned = improvement_pct >= 1.0

if use_tuned:
    print(f"\n✓ DECISION: Using TUNED model (improvement: {improvement_pct:+.2f}%)")
    final_model = best_model
    final_f1 = f1_tuned
else:
    print(f"\n○ DECISION: Using DEFAULT model (improvement: {improvement_pct:+.2f}% < 1%)")
    # Re-train default
    final_model = algorithms_test[best_name][0]
    final_model.fit(X_tr if best_name in ['Random Forest', 'XGBoost', 'LightGBM'] else X_tr, y_train)
    y_pred_final = final_model.predict(X_te)
    final_f1 = f1_score(y_test, y_pred_final, average='weighted')

# ── Final output ──
print(f"\n{'='*60}")
print(f"FINAL MODEL: {best_name} {'(Tuned)' if use_tuned else '(Default)'}")
print(f"Final Test F1: {final_f1:.4f}")
print(f"{'='*60}")

# ── Predict on full dataset ──
full_X = X.values if best_name in ['Random Forest', 'XGBoost', 'LightGBM'] else scaler.transform(X)
full_pred = final_model.predict(full_X)

df_output = df.copy()
df_output['mo_prediction'] = full_pred
if len(np.unique(y)) == 2:
    try:
        full_proba = final_model.predict_proba(full_X)[:, 1]
        df_output['mo_prediction_probability'] = full_proba
    except:
        pass

# Save output
output_csv = os.path.join(OUTPUT_DIR, 'mo_output.csv')
df_output.to_csv(output_csv, index=False)
print(f"[STATUS] Saved: {output_csv}")

# Save model params report
print(f"\n{'─'*50}")
print(f"Best Params: {best_params}")
print(f"CV Score: {best_cv_score:.4f}")
print(f"Test F1: {final_f1:.4f}")
if len(np.unique(y_test)) == 2:
    print(f"Test AUC: {auc_tuned:.4f}")
print(f"Overfitting Gap: {overfit_gap:.4f}")
```

## Step 5: Generate Full Report

Now I'll create the comprehensive report.

```python
import os, json
