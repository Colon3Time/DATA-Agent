import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_uci_bank_marketing_blind")
DEFAULT_INPUT = Path(r"C:\Users\Amorntep\DAta-agent\projects\2026-04-28_uci_bank_marketing_blind\input\uci_raw\bank-additional\bank-additional\bank-additional-full.csv")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "mo"
TARGET_COL = "y"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--n-iter", type=int, default=30)
    return parser.parse_known_args()[0]


def load_training_data(input_path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series, str, Path]:
    path = Path(input_path)
    
    # ตรวจสอบไฟล์ input โดยตรง
    if not path.exists() or path.suffix.lower() != ".csv":
        # ถ้าไม่มีหรือไม่ใช่ csv ลองหาใน parent folder
        candidates = list(PROJECT_ROOT.glob("**/*bank-additional-full.csv"))
        if candidates:
            path = candidates[0]
        else:
            # fallback ไปยัง DEFAULT_INPUT
            path = DEFAULT_INPUT
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    df = pd.read_csv(path, sep=None, engine="python")  # auto-detect separator
    
    # แสดง column names สำหรับ debug
    print(f"[STATUS] Columns in dataset: {df.columns.tolist()}")
    
    # ค้นหา target column
    if target_col in df.columns:
        pass
    else:
        # ลองหาชื่อที่ใกล้เคียง
        candidates = [c for c in df.columns if c.lower() in {"y", "target", "label", "response", "class"}]
        if candidates:
            target_col = candidates[0]
            print(f"[STATUS] Using target column: {target_col}")
        else:
            raise ValueError(f"target column '{target_col}' not found in {path}. Available columns: {df.columns.tolist()}")
    
    # แยก features และ target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    
    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
    
    # Encode target ถ้าเป็น string
    if y.dtype == "object" or y.dtype.name == "category":
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y.astype(str)), name=target_col)
        print(f"[STATUS] Target encoded. Classes: {encoder.classes_.tolist()}, Value counts: {y.value_counts().to_dict()}")
    
    # แปลงทุก column เป็น numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    print(f"[STATUS] Loaded data: {df.shape}, X shape: {X.shape}, y shape: {y.shape}")
    print(f"[STATUS] Target distribution: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y, target_col, path


def metrics_row(label: str, model, X_train, X_test, y_train, y_test, cv_scores, elapsed: float, params: dict):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_f1 = f1_score(y_train, train_pred, average="weighted")
    test_f1 = f1_score(y_test, test_pred, average="weighted")
    test_acc = accuracy_score(y_test, test_pred)
    
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)
    else:
        test_auc = np.nan
    
    return {
        "phase": "Phase 2 Tune",
        "algorithm": label,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "train_test_gap": float(train_f1 - test_f1),
        "test_acc": float(test_acc),
        "test_auc": float(test_auc) if not np.isnan(test_auc) else None,
        "test_precision": float(precision_score(y_test, test_pred, average="weighted")),
        "test_recall": float(recall_score(y_test, test_pred, average="weighted")),
        "train_time_sec": float(elapsed),
        "params": json.dumps(params)
    }


def main():
    args = parse_args()
    
    # ตรวจสอบและแก้ไข output dir
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[STATUS] Input: {args.input}")
    print(f"[STATUS] Output dir: {output_dir}")
    print(f"[STATUS] Target column: {args.target}")
    
    # โหลดข้อมูล
    X, y, target_col, input_path = load_training_data(args.input, args.target)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[STATUS] Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Baseline Random Forest
    print("[STATUS] Training Random Forest baseline...")
    rf_default = RandomForestClassifier(random_state=42, n_jobs=-1)
    start = time.time()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_default, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
    
    rf_default.fit(X_train, y_train)
    elapsed = time.time() - start
    
    baseline_row = metrics_row(
        "RandomForest_default", rf_default,
        X_train, X_test, y_train, y_test,
        cv_scores, elapsed, {"n_estimators": 100, "max_depth": None}
    )
    print(f"[STATUS] Baseline CV F1: {baseline_row['cv_mean']:.4f} ± {baseline_row['cv_std']:.4f}")
    
    # Hyperparameter Tuning with RandomizedSearchCV
    print("[STATUS] Starting randomized search...")
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"]
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        rf_base, param_dist,
        n_iter=min(args.n_iter, 50),
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    start = time.time()
    random_search.fit(X_train, y_train)
    elapsed = time.time() - start
    
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_
    
    print(f"[STATUS] Best CV score: {best_cv_score:.4f}")
    print(f"[STATUS] Best params: {best_params}")
    
    tuned_row = metrics_row(
        "RandomForest_tuned", random_search.best_estimator_,
        X_train, X_test, y_train, y_test,
        [random_search.cv_results_["mean_test_score"][random_search.best_index_]],
        elapsed, best_params
    )
    
    # Overfitting check
    overfit_gap = tuned_row["train_test_gap"]
    if overfit_gap > 0.1:
        overfit_warning = f"WARNING: Large gap ({overfit_gap:.4f}) - possible overfitting"
    else:
        overfit_warning = f"OK: Gap ({overfit_gap:.4f}) - acceptable"
    
    print(f"[STATUS] Train-test gap: {overfit_gap:.4f} - {overfit_warning}")
    
    # บันทึกผล
    results = [baseline_row, tuned_row]
    results_df = pd.DataFrame(results)
    
    # Save CSV
    output_csv = output_dir / "mo_model_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"[STATUS] Saved results to {output_csv}")
    
    # Save detailed report
    improvement = ((tuned_row["test_f1"] - baseline_row["test_f1"]) / baseline_row["test_f1"]) * 100 if baseline_row["test_f1"] > 0 else 0
    report = f"""Mo Model Report — Phase 2: Hyperparameter Tuning
==================================================
Input file: {input_path}
Phase: 2 (Tune - RandomizedSearchCV on RandomForest)
Search iterations: {args.n_iter}

Baseline Results (RandomForest default):
- CV F1: {baseline_row['cv_mean']:.4f} ± {baseline_row['cv_std']:.4f}
- Test F1: {baseline_row['test_f1']:.4f}
- Test AUC: {baseline_row['test_auc']}
- Train F1: {baseline_row['train_f1']:.4f}

Tuned Results (RandomForest best):
- Best CV F1: {best_cv_score:.4f}
- Test F1: {tuned_row['test_f1']:.4f}
- Test AUC: {tuned_row['test_auc']}
- Train F1: {tuned_row['train_f1']:.4f}
- Improvement over baseline: {improvement:.2f}%

Best Hyperparameters:
{json.dumps(best_params, indent=2)}

Overfitting Check:
- Train F1: {tuned_row['train_f1']:.4f}
- Test F1: {tuned_row['test_f1']:.4f}
- Gap: {overfit_gap:.4f}
- Status: {overfit_warning}

ALGORITHM_RATIONALE
===================
Best Algorithm: RandomForest (Tuned)
Why This Algorithm:
  - ข้อมูล: tabular with mixed numerical/categorical features, banking data
  - Theory: ensemble of decision trees - handles non-linearity, feature interactions
  - vs others: robust to outliers, provides feature importance

NEXT_STEP: DONE
"""
    
    report_path = output_dir / "model_results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[STATUS] Saved report to {report_path}")
    
    # Agent Report
    agent_report = f"""
Agent Report — Mo
===================
รับจาก     : User (script execution)
Input      : {input_path}
ทำ         : Load data, train RandomForest baseline + hyperparameter tuning
พบ         : Data loaded successfully, baseline CV F1={baseline_row['cv_mean']:.4f}, tuned CV F1={best_cv_score:.4f}
เปลี่ยนแปลง: Model performance improved by {improvement:.2f}%
ส่งต่อ     : Quinn - model_results.csv และ model_results.md
"""
    print(agent_report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)