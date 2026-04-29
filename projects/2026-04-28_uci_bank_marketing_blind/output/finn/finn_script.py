import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings('ignore')

# ── Argument Parser ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ────────────────────────────────────────────────────
print('[STATUS] Loading data...')
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape} — columns: {list(df.columns)}')

# ── Target ───────────────────────────────────────────────────────
TARGET = 'y'
if TARGET not in df.columns:
    target_candidates = [c for c in df.columns if 'target' in c.lower() or 'y' == c.lower() or 'label' in c.lower()]
    if target_candidates:
        TARGET = target_candidates[0]
        print(f'[STATUS] Using target column: {TARGET}')
    else:
        print(f'[ERROR] Target column not found in: {list(df.columns)}')
        sys.exit(1)

# 1. Drop 'duration' (Data Leakage)
if 'duration' in df.columns:
    df = df.drop(columns=['duration'])
    print('[STATUS] Dropped "duration" — data leakage')

# Also drop pdays if too many -1 (missing values) — but keep for now
# Check target
y = df[TARGET]
if y.dtype == 'object':
    y = y.map({'yes': 1, 'no': 0, '1': 1, '0': 0})
    if y.isna().any():
        # Try LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df[TARGET])
y = y.astype(int)

# 2. Separate features and target
X = df.drop(columns=[TARGET])

# ── Identify column types ────────────────────────────────────────
potential_categorical = ['job', 'marital', 'education', 'default', 'housing',
                         'loan', 'contact', 'month', 'day_of_week', 'poutcome',
                         'pdays_binary']
categorical_cols = [c for c in potential_categorical if c in X.columns and X[c].dtype == 'object']
# Also check for object dtype columns not yet identified
for col in X.select_dtypes(include='object').columns:
    if col not in categorical_cols:
        categorical_cols.append(col)

numeric_cols = ['age', 'balance', 'campaign', 'pdays', 'previous']
numeric_cols = [c for c in numeric_cols if c in X.columns and X[c].dtype in ['int64', 'float64']]

social_economic_cols = ['euribor3m', 'cons.price.idx', 'cons.conf.idx', 'nr.employed']
social_economic_cols = [c for c in social_economic_cols if c in X.columns and X[c].dtype in ['int64', 'float64']]

# Additional numeric columns not in pre-defined lists
extra_numeric = []
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    if col not in numeric_cols + social_economic_cols and col not in categorical_cols:
        extra_numeric.append(col)

print(f'[STATUS] Categorical: {categorical_cols}')
print(f'[STATUS] Numeric: {numeric_cols + social_economic_cols + extra_numeric}')

# ── 3. Encoding ─────────────────────────────────────────────────
print('[STATUS] Encoding categorical features (One-Hot)...')
if categorical_cols:
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
else:
    X_encoded = X.copy()
print(f'[STATUS] After encoding: {X_encoded.shape}')

# ── 4. Scaling ──────────────────────────────────────────────────
print('[STATUS] Scaling numeric features (StandardScaler)...')
all_numeric = numeric_cols + social_economic_cols + extra_numeric
all_numeric = [c for c in all_numeric if c in X_encoded.columns]

if all_numeric:
    scaler = RobustScaler()  # Use RobustScaler for outlier robustness
    X_encoded[all_numeric] = scaler.fit_transform(X_encoded[all_numeric])

# ── 5. Auto-Compare Feature Selection ────────────────────────────
print('[STATUS] Auto-Compare Feature Selection (ALL methods)...')

def auto_compare_feature_selection(X, y, problem_type="classification"):
    """Run all feature selection methods and compare CV scores"""
    is_clf = problem_type == "classification"
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    scoring = "f1_weighted"

    X_num = X.select_dtypes(include="number")
    candidates = {}
    scores = {}

    # 1. Mutual Information
    try:
        mi = mutual_info_classif(X_num, y, random_state=42)
        mi_series = pd.Series(mi, index=X_num.columns).sort_values(ascending=False)
        n_top = max(1, len(X_num.columns) // 2)  # top 50%
        top_mi = mi_series.nlargest(n_top).index.tolist()
        candidates["mutual_info"] = top_mi
        print(f'[STATUS] mutual_info: {len(top_mi)} features selected')
    except Exception as e:
        print(f'[WARN] mutual_info failed: {e}')

    # 2. RFECV
    try:
        # Use fewer features to avoid long runtime
        rfecv = RFECV(estimator=model, cv=3, scoring=scoring, n_jobs=-1,
                      min_features_to_select=1, step=5)
        rfecv.fit(X_num, y)
        candidates["rfecv"] = X_num.columns[rfecv.support_].tolist()
        print(f'[STATUS] rfecv: {len(candidates["rfecv"])} features selected')
    except Exception as e:
        print(f'[WARN] rfecv failed: {e}')

    # 3. SelectFromModel (Random Forest importance)
    try:
        sfm = SelectFromModel(model, threshold="median")
        sfm.fit(X_num, y)
        candidates["rf_importance"] = X_num.columns[sfm.get_support()].tolist()
        print(f'[STATUS] rf_importance: {len(candidates["rf_importance"])} features selected')
    except Exception as e:
        print(f'[WARN] rf_importance failed: {e}')

    # 4. Lasso L1 (Logistic Regression)
    try:
        lasso_model = LogisticRegression(C=0.1, solver="saga", penalty="l1",
                                         max_iter=1000, random_state=42, n_jobs=-1)
        lasso_model.fit(X_num, y)
        mask = np.any(lasso_model.coef_ != 0, axis=0)
        lasso_feats = X_num.columns[mask].tolist()
        if lasso_feats:
            candidates["lasso_l1"] = lasso_feats
            print(f'[STATUS] lasso_l1: {len(lasso_feats)} features selected')
    except Exception as e:
        print(f'[WARN] lasso_l1 failed: {e}')

    # 5. Variance Threshold
    try:
        vt = VarianceThreshold(threshold=0.01)
        vt.fit(X_num)
        candidates["variance_threshold"] = X_num.columns[vt.get_support()].tolist()
        print(f'[STATUS] variance_threshold: {len(candidates["variance_threshold"])} features selected')
    except Exception as e:
        print(f'[WARN] variance_threshold failed: {e}')

    # Compare CV scores
    for name, feats in candidates.items():
        valid = [f for f in feats if f in X_num.columns]
        if not valid:
            continue
        try:
            cv = cross_val_score(model, X_num[valid], y, cv=3,
                                 scoring=scoring, n_jobs=-1).mean()
            scores[name] = (cv, valid)
            print(f'[STATUS] {name:20s}: {scoring}={cv:.4f}  ({len(valid)} features)')
        except Exception as e:
            print(f'[WARN] score {name} failed: {e}')

    if not scores:
        print('[WARN] All methods failed — using all features')
        return {"best_method": "all", "best_features": X.columns.tolist(), "scores": {}}

    best_method = max(scores, key=lambda k: scores[k][0])
    best_score, best_features = scores[best_method]
    print(f'[STATUS] Best: {best_method} — {scoring}={best_score:.4f} ({len(best_features)} features)')

    return {
        "best_method": best_method,
        "best_features": best_features,
        "scores": {k: v[0] for k, v in scores.items()},
        "all_candidates": {k: v[1] for k, v in scores.items()},
    }

# Run auto-compare
result = auto_compare_feature_selection(X_encoded, y)
best_features = result["best_features"]
best_method = result["best_method"]

print(f'[STATUS] Selected method: {best_method} | Features: {len(best_features)}')

# ── Create report content ────────────────────────────────────────
report_lines = []
report_lines.append("Finn Feature Engineering Report")
report_lines.append("=" * 40)
report_lines.append(f"Original Features: {X.shape[1]}")
report_lines.append(f"New Features Created: 0")
report_lines.append(f"Final Features Selected: {len(best_features)}")
report_lines.append(f"")
report_lines.append("Auto-Compare Results:")
report_lines.append(f"| {'Method':20s} | {'CV Score':10s} | {'Features':10s} |")
report_lines.append(f"|{'-'*20}|{'-'*10}|{'-'*10}|")
for method in result.get("scores", {}):
    cv = result["scores"][method]
    feats = len(result["all_candidates"].get(method, []))
    report_lines.append(f"| {method:20s} | {cv:.4f}      | {feats:<8d} |")
report_lines.append("")
report_lines.append(f"Best Method: {best_method} (score={result['scores'].get(best_method, 'N/A'):.4f})")
# Also store the key for use below
best_score_val = result['scores'].get(best_method, 0)
report_lines.append("")
report_lines.append("Features Created:")
report_lines.append("- No new features created (encoding/scaling only)")
report_lines.append("")
report_lines.append("Features Dropped:")
report_lines.append("- duration (target leakage)")
report_lines.append("")
report_lines.append("Encoding Used: One-Hot Encoding")
report_lines.append("Scaling Used: RobustScaler")
report_lines.append("")
report_lines.append("Self-Improvement Report")
report_lines.append("=" * 40)
report_lines.append(f"Method used: auto_compare → {best_method}")
report_lines.append(f"Reason: CV score = {best_score_val:.4f} (data-driven selection)")
report_lines.append("New methods found: None")
report_lines.append("Will use next time: Yes")
report_lines.append("Knowledge Base: No changes needed")

report_text = "\n".join(report_lines)

# ── Save FINAL output (engineered_data.csv with selected features) ──
print('[STATUS] Saving engineered data...')
# Make sure all best_features are in X_encoded
final_features = [f for f in best_features if f in X_encoded.columns]
# Always include target for modeling
if final_features:
    X_final = X_encoded[final_features].copy()
else:
    X_final = X_encoded.copy()
    print('[WARN] No features selected — keeping all')

# Add target back
X_final[TARGET] = y.values

output_csv = os.path.join(OUTPUT_DIR, 'engineered_data.csv')
X_final.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv} | Shape: {X_final.shape}')

# ── Save feature report ──────────────────────────────────────────
report_path = os.path.join(OUTPUT_DIR, 'finn_feature_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f'[STATUS] Saved: {report_path}')

# ── Agent Report ─────────────────────────────────────────────────
agent_report = []
agent_report.append("Agent Report — Finn")
agent_report.append("=" * 30)
agent_report.append(f"รับจาก     : Eddie")
agent_report.append(f"Input      : {INPUT_PATH}")
agent_report.append(f"ทำ         : Feature engineering - encoding (One-Hot), scaling (RobustScaler), "
                     f"auto-compare feature selection (ML methods)")
agent_report.append(f"พบ         : 1. Dropped 'duration' column due to data leakage  "
                     f"2. Best feature selection method: {best_method} with score={best_score_val:.4f}  "
                     f"3. Dataset reduced from {X.shape[1]} to {len(final_features)} features")
agent_report.append(f"เปลี่ยนแปลง : Features selected by ML, ready for modeling")
agent_report.append(f"ส่งต่อ     : Mo — {output_csv}")

agent_report_text = "\n".join(agent_report)
agent_path = os.path.join(OUTPUT_DIR, 'finn_agent_report.txt')
with open(agent_path, 'w', encoding='utf-8') as f:
    f.write(agent_report_text)
print(f'[STATUS] Agent report saved: {agent_path}')

print('[STATUS] Finn pipeline complete')