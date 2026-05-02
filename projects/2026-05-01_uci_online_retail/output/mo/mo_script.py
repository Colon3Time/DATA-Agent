import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

INPUT_PATH = Path(args.input)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "mo_report.md"
OUTPUT_CSV = OUTPUT_DIR / "mo_output.csv"
REPAIR_PATH = OUTPUT_DIR / "REPAIR.md"

df = pd.read_csv(INPUT_PATH)
finn_dir = INPUT_PATH.parent
manifest_path = finn_dir / "finn_feature_manifest.json"
if not manifest_path.exists():
    raise FileNotFoundError(f"Missing Finn feature manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

ID_COLS = set(manifest.get("id_cols", []))
DATETIME_COLS = set(manifest.get("datetime_cols", []))
TARGETS = manifest.get("targets", {})


def write_repair_and_raise(message):
    REPAIR_PATH.write_text(f"# Mo Repair Required\n\n{message}\n", encoding="utf-8")
    raise RuntimeError(message)


def feature_frame(data, target_col, spec):
    excluded = set(spec.get("exclude_features", [])) | ID_COLS | DATETIME_COLS | set(TARGETS.keys()) | {target_col}
    features = [c for c in data.columns if c not in excluded]
    X = data[features].copy()
    for c in X.columns:
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            if c in {"grain"}:
                X = X.drop(columns=[c])
            else:
                X[c] = pd.factorize(X[c].astype(str))[0]
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, list(X.columns), sorted(excluded)


CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}
REGRESSORS = {
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
}


def classification_metrics(model, X_train, X_test, y_train, y_test, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    out = {
        "Test_F1": f1_score(y_test, pred, zero_division=0),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
    }
    if proba is not None and len(np.unique(y_test)) == 2:
        out["ROC_AUC"] = roc_auc_score(y_test, proba)
        out["PR_AUC"] = average_precision_score(y_test, proba)
    return out


def regression_metrics(model, X_train, X_test, y_train, y_test, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "R2": r2_score(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": mean_absolute_error(y_test, pred),
    }


rows = []
report = [
    "# Mo Model Report",
    "",
    f"Input: `{INPUT_PATH}`",
    f"Manifest: `{manifest_path}`",
    f"Rows: {len(df):,}",
    "",
    "## Random Split Validation",
]

best_by_target = {}
for target_col, spec in TARGETS.items():
    if target_col not in df.columns:
        continue
    task = spec.get("task", "classification")
    X, features, excluded = feature_frame(df, target_col, spec)
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    if X.empty or y.nunique() < 2:
        continue

    if task == "classification":
        y = (y == sorted(y.unique())[-1]).astype(int)
        stratify = y if y.nunique() == 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        models = CLASSIFIERS
        score_key = "Test_F1"
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y.astype(float), test_size=0.2, random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        models = REGRESSORS
        score_key = "R2"

    report += ["", f"### Target `{target_col}`", f"- Task: {task}", f"- Features used: {len(features)}", f"- Excluded: {', '.join(excluded)}", ""]
    report.append("| Model | CV | Test Metric | Secondary |")
    report.append("|---|---:|---:|---|")
    best = None
    for name, model in models.items():
        scale = name in {"LogisticRegression", "Ridge"}
        X_cv = StandardScaler().fit_transform(X_train) if scale else X_train
        scoring = "f1" if task == "classification" else "r2"
        cv_score = float(cross_val_score(model, X_cv, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean())
        if task == "classification":
            metrics = classification_metrics(model, X_train, X_test, y_train, y_test, scale=scale)
            if metrics.get("Test_F1") == 1.0 and cv_score > 0.999:
                write_repair_and_raise(f"Leakage signature detected for {target_col}/{name}: Test_F1=1.0 and CV_F1={cv_score:.4f}")
            secondary = f"ROC-AUC={metrics.get('ROC_AUC', np.nan):.4f}, PR-AUC={metrics.get('PR_AUC', np.nan):.4f}"
            test_metric = metrics["Test_F1"]
        else:
            metrics = regression_metrics(model, X_train, X_test, y_train, y_test, scale=scale)
            if metrics.get("R2") == 1.0:
                write_repair_and_raise(f"Leakage signature detected for {target_col}/{name}: R2=1.0")
            secondary = f"RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}"
            test_metric = metrics["R2"]
        rows.append({"target": target_col, "model": name, "task": task, "cv": cv_score, **metrics})
        report.append(f"| {name} | {cv_score:.4f} | {test_metric:.4f} | {secondary} |")
        if best is None or test_metric > best[0]:
            best = (test_metric, name, metrics, features)
    best_by_target[target_col] = best


oot_results = []
oot_spec = manifest.get("oot_split")
if oot_spec:
    oot_path = finn_dir / oot_spec.get("table", "")
    if oot_path.exists():
        oot_df = pd.read_csv(oot_path)
        label = f"is_churned_{oot_spec.get('label_window_days', 90)}d"
        if label in oot_df.columns:
            spec = {"task": "classification", "exclude_features": ["recency_days", "clv_proxy", "monetary", "avg_order_value", "is_high_value", "is_churned_180d"]}
            X, features, excluded = feature_frame(oot_df, label, spec)
            y = pd.to_numeric(oot_df[label], errors="coerce")
            valid = y.notna()
            X = X.loc[valid]
            y = y.loc[valid].astype(int)
            if y.nunique() == 2 and len(X) >= 50:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
                metrics = classification_metrics(model, X_train, X_test, y_train, y_test)
                oot_results.append({"target": label, **metrics})

report += ["", "## OOT Validation"]
if oot_results:
    random_churn = next((r for r in rows if r["target"] == "is_churned_180d" and "ROC_AUC" in r), None)
    for res in oot_results:
        report.append(f"- `{res['target']}` ROC-AUC={res.get('ROC_AUC', np.nan):.4f}, PR-AUC={res.get('PR_AUC', np.nan):.4f}, F1={res.get('Test_F1', np.nan):.4f}")
        if random_churn and abs(float(random_churn.get("ROC_AUC", 0)) - float(res.get("ROC_AUC", 0))) > 0.10:
            report.append("- WARNING: OOT ROC-AUC differs from random split by more than 0.10.")
else:
    report.append("- OOT split unavailable or insufficient class variation.")

pd.DataFrame(rows).to_csv(OUTPUT_DIR / "model_results.csv", index=False)
out = df.copy()
for target_col, best in best_by_target.items():
    if not best:
        continue
    out[f"{target_col}_best_model"] = best[1]
out.to_csv(OUTPUT_CSV, index=False)
REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

print(f"[STATUS] Saved {OUTPUT_CSV}")
print(f"[STATUS] Saved {REPORT_PATH}")
