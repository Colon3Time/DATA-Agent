import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

input_path = Path(args.input)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)
base_drop_cols = {
    "Customer ID",
    "customer_id",
    "first_purchase",
    "last_purchase",
    "grain",
    "is_churned_180d",
    "is_high_value",
    "clv_proxy",
}
feature_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in base_drop_cols and c.lower() not in {x.lower() for x in base_drop_cols}
]

rows = []

if "is_churned_180d" in df.columns and df["is_churned_180d"].nunique() > 1:
    y = df["is_churned_180d"].astype(int)
    churn_leakage = {"recency_days"}
    churn_features = [c for c in feature_cols if c not in churn_leakage]
    X = df[churn_features].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "logistic_churn": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "rf_churn": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample", min_samples_leaf=5),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "task": "churn",
            "model": name,
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob),
            "f1": f1_score(y_test, pred),
            "n_test": len(y_test),
        })

if "is_high_value" in df.columns and df["is_high_value"].nunique() > 1:
    y = df["is_high_value"].astype(int)
    value_leakage = {"monetary", "avg_order_value", "clv_proxy"}
    value_features = [c for c in feature_cols if c not in value_leakage]
    X = df[value_features].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample", min_samples_leaf=5)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    rows.append({
        "task": "high_value",
        "model": "rf_high_value",
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "f1": f1_score(y_test, pred),
        "n_test": len(y_test),
    })

if "clv_proxy" in df.columns:
    y = df["clv_proxy"]
    reg_features = [c for c in feature_cols if c not in {"monetary", "avg_order_value"}]
    X_reg = df[reg_features].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "task": "clv_proxy",
        "model": "rf_regressor",
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
        "n_test": len(y_test),
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(output_dir / "model_comparison.csv", index=False)

report = f"""# Mo Baseline Modeling Report

Input: {input_path}
Rows: {len(df):,}
Base feature count: {len(feature_cols)}

## Results

{comparison.to_markdown(index=False)}

## Validation Caveat

These are baseline models on Finn's customer-level analytical table. Target-derived direct leakage was removed per task (`recency_days` for churn; `monetary`/`avg_order_value` for high-value and CLV proxy). The churn and CLV targets are still derived from full historical data for pipeline validation. Production-grade modeling must rebuild labels with a cutoff date and compute features only before that cutoff.

MODEL_GOVERNANCE
================
grain: customer
leakage_status: acceptable for baseline, not final deployment
required_next_step: time-cutoff validation before business claim
"""
(output_dir / "model_results.md").write_text(report, encoding="utf-8")
(output_dir / "mo_report.md").write_text(report, encoding="utf-8")
(output_dir.parent / "agent_report_mo.md").write_text("Agent Report - Mo\nOutput: model_results.md, model_comparison.csv\n", encoding="utf-8")
print("[STATUS] Mo complete")
