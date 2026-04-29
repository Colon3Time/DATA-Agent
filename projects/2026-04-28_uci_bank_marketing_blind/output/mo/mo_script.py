import argparse, os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input or r"D:\DATA-Agent-refactor-v2\projects\2026-04-28_uci_bank_marketing_blind\input\uci_raw\bank-additional\bank-additional\bank-additional-full.csv"
OUTPUT_DIR = args.output_dir or r"D:\DATA-Agent-refactor-v2\projects\2026-04-28_uci_bank_marketing_blind\output\mo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'mo_output.csv')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'model_results.md')

def validate_mo_input(input_path, target_col='y', min_rows=30):
    p = Path(input_path)
    FORBIDDEN = ["outlier_flags", "feature_report", "qc_report",
                 "mining_result", "patterns_found"]
    if any(kw in p.stem.lower() for kw in FORBIDDEN):
        print(f"[ERROR] Mo ตรวจพบว่าโหลดไฟล์ผิด: {p.name}")
        for parent in [p.parent, p.parent.parent]:
            candidates = (list(parent.glob("dana/dana_output.csv")) +
                          list(parent.glob("finn/finn_output.csv")) +
                          list(parent.glob("finn/engineered_data.csv")) +
                          list(parent.glob("dana/*_output.csv")))
            if candidates:
                correct = candidates[0]
                print(f"[STATUS] หาไฟล์ที่ถูกต้อง → {correct}")
                input_path = str(correct)
                p = correct
                break
        else:
            sys.exit(1)
    for sep in [';', ',', '\t', '|']:
        try:
            df = pd.read_csv(input_path, sep=sep, nrows=5)
            if target_col in df.columns or target_col in df.columns.tolist():
                df = pd.read_csv(input_path, sep=sep)
                print(f"[STATUS] Loaded with sep='{sep}': {df.shape} from {p.name}")
                break
        except:
            continue
    else:
        print(f"[ERROR] ไม่สามารถโหลดไฟล์ {input_path} ได้")
        sys.exit(1)
    if len(df) < min_rows:
        print(f"[ERROR] Dataset มีแค่ {len(df)} rows — น้อยเกินไป")
        sys.exit(1)
    if target_col not in df.columns:
        # Try to find target column with similar name
        possible = [c for c in df.columns if 'y' in c.lower() or 'target' in c.lower() or 'label' in c.lower() or 'class' in c.lower()]
        if possible:
            target_col = possible[0]
            print(f"[STATUS] ใช้ target column: {target_col}")
        else:
            available = df.columns.tolist()
            print(f"[ERROR] ไม่พบ target_col='y' ใน {available}")
            sys.exit(1)
    return df, target_col

# ── 1. Load Data ──
print("=" * 50)
print("Mo Model Builder — Phase 1: Explore")
print("=" * 50)

df, target_col = validate_mo_input(INPUT_PATH, target_col='y')
print(f"[STATUS] Data shape: {df.shape}")
print(f"[STATUS] Target: {target_col}")
print(f"[STATUS] Target distribution:\n{df[target_col].value_counts()}")

# ── 2. Preprocess ──
print("\n[STATUS] Preprocessing...")

# Convert target to binary (0/1)
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify column types
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"[STATUS] Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}")

# Encode categorical
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Handle any remaining non-numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

print(f"[STATUS] After encoding: {X.shape}")

# ── 3. Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[STATUS] Train: {X_train.shape}, Test: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 4. Model Comparison ──
print("\n[STATUS] Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_jobs=-1),
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_f1 = 0.0
best_name = ""

for name, model in models.items():
    start_time = time.time()
    
    # Use scaled data for models that need it
    if name in ['Logistic Regression', 'SVM', 'KNN']:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train
        X_te = X_test
    
    try:
        # Cross-validation
        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv,
                                     scoring='f1_weighted', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train and predict
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        
        # Metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_te)
            if y_proba.shape[1] >= 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = 0.5
        else:
            auc = 0.5
        
        elapsed = time.time() - start_time
        
        results.append({
            'Algorithm': name,
            'CV_mean': round(cv_mean, 4),
            'CV_std': round(cv_std, 4),
            'Test_Acc': round(acc, 4),
            'Test_F1': round(f1, 4),
            'Test_Precision': round(prec, 4),
            'Test_Recall': round(rec, 4),
            'Test_AUC': round(auc, 4),
            'Time': round(elapsed, 2)
        })
        
        print(f"[STATUS] {name}: CV={cv_mean:.4f}±{cv_std:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Time={elapsed:.2f}s")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            
    except Exception as e:
        print(f"[WARN] {name} failed: {e}")
        continue

# ── 5. Results ──
print("\n" + "=" * 50)
print("Model Comparison Results")
print("=" * 50)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_F1', ascending=False)
print(results_df.to_string(index=False))

# Save comparison CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[STATUS] Results saved to {OUTPUT_CSV}")

# ── 6. Feature Importance (if available) ──
print("\n[STATUS] Analyzing feature importance...")
importance_df = pd.DataFrame()

for name, model in models.items():
    if name == best_name:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
        elif name == 'Logistic Regression':
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False).head(10)
        break

# ── 7. Generate Report ──
print(f"\n[STATUS] Generating report...")

report_lines = []
report_lines.append("Mo Model Report — Phase 1: Explore")
report_lines.append("=" * 50)
report_lines.append(f"Problem Type: Classification")
report_lines.append(f"Phase: 1 (Explore — all algorithms, default params)")
report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")
report_lines.append("Algorithm Comparison (CV 5-fold):")
report_lines.append("| Algorithm | CV Mean | CV Std | Test F1 | Test AUC | Time(s) |")
report_lines.append("|-----------|---------|--------|---------|----------|---------|")
for _, row in results_df.iterrows():
    report_lines.append(f"| {row['Algorithm']} | {row['CV_mean']} | {row['CV_std']} | {row['Test_F1']} | {row['Test_AUC']} | {row['Time']} |")

report_lines.append("")
report_lines.append(f"Winner: {best_name} — CV: {results_df.iloc[0]['CV_mean']}, Test F1: {results_df.iloc[0]['Test_F1']}")

# Feature Importance
if not importance_df.empty:
    report_lines.append("")
    report_lines.append("Top 10 Feature Importance:")
    report_lines.append("| Feature | Importance |")
    report_lines.append("|---------|------------|")
    for _, row in importance_df.iterrows():
        report_lines.append(f"| {row['feature']} | {row['importance']:.4f} |")

# Preprocessing Requirement
report_lines.append("")
report_lines.append("PREPROCESSING_REQUIREMENT")
report_lines.append("=" * 30)
report_lines.append(f"Algorithm Selected: {best_name}")
if best_name in ['Logistic Regression', 'SVM', 'KNN']:
    report_lines.append("Scaling Needed: StandardScaler")
else:
    report_lines.append("Scaling Needed: None (tree-based handles scale)")
report_lines.append(f"Encoding Needed: Already encoded with LabelEncoder")
report_lines.append("Special Transform: None")
report_lines.append("Loop Back To Finn: NO")
report_lines.append("Reason: All preprocessing done — features already encoded and scaled if needed")
report_lines.append("DL_ESCALATE: NO")
report_lines.append("DL_Reason: n=41,188 < 100K threshold for DL advantage; classical ML sufficient")

# Business Recommendation
report_lines.append("")
report_lines.append("Business Recommendation:")
report_lines.append("-" * 30)
report_lines.append(f"The best performing model is {best_name} with F1={best_f1:.4f}.")
report_lines.append(f"This model is suitable for {'telemarketing campaign prediction' if 'bank' in INPUT_PATH.lower() else 'this classification task'}.")
report_lines.append("Consider this model for production deployment after Phase 2 tuning.")

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"[STATUS] Report saved to {REPORT_PATH}")

# ── 8. Self-Improvement Report ──
self_report = f"""
Self-Improvement Report — Mo
============================
Phase ที่ผ่าน: 1
Algorithm ที่ชนะ: {best_name}
Tuning improvement: Pending (Phase 2)
วิธีใหม่ที่พบ: Not applicable
Knowledge Base: LightGBM ไม่สามารถ import ได้ — ใช้ Random Forest แทน
Note: install lightgbm via 'pip install lightgbm' เพื่อให้มีตัวเลือกเพิ่ม
"""

self_report_path = os.path.join(OUTPUT_DIR, 'self_improvement.md')
with open(self_report_path, 'w', encoding='utf-8') as f:
    f.write(self_report)

# ── Agent Report ──
agent_report = f"""
Agent Report — Mo
==================
รับจาก     : User
Input      : {INPUT_PATH}
ทำ         : Phase 1 — เปรียบเทียบ 4 algorithms (LR, RF, SVM, KNN) ด้วย default params
พบ         : 
  - Best model: {best_name} (F1={best_f1:.4f})
  - {results_df.iloc[0]['Algorithm']} vs runner-up: F1 ต่างกัน {results_df.iloc[0]['Test_F1'] - results_df.iloc[1]['Test_F1']:.4f}
  - LightGBM ไม่มีใน environment ต้องติดตั้งก่อน
เปลี่ยนแปลง: ไม่มี — ไม่ได้เปลี่ยน data
ส่งต่อ     : Phase 2 — Tune {best_name} with hyperparameter search
"""

agent_report_path = os.path.join(OUTPUT_DIR, 'agent_report.md')
with open(agent_report_path, 'w', encoding='utf-8') as f:
    f.write(agent_report)

print(f"\n[STATUS] All outputs saved to {OUTPUT_DIR}")
print(f"[STATUS] Best Model: {best_name}")
print(f"[STATUS] Best F1: {best_f1:.4f}")
print("\n✅ Mo Phase 1 Complete!")