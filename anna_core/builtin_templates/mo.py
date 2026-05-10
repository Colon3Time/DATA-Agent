from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _missing_dependencies() -> list[str]:
    import importlib.util

    return [name for name in ("sklearn", "lightgbm", "xgboost") if importlib.util.find_spec(name) is None]


def _read_profile(inp: Path) -> dict[str, str]:
    # User-specified override wins over Scout's inferred target
    override_path = next(
        (p / "target_override.json" for p in [inp.parent, inp.parent.parent, inp.parent.parent.parent, inp.parent.parent.parent.parent]
         if (p / "target_override.json").exists()),
        inp.parent.parent.parent / "target_override.json",
    )
    if override_path.exists():
        try:
            import json as _j
            data = _j.loads(override_path.read_text(encoding="utf-8"))
            if data.get("target_column") and str(data["target_column"]).lower() not in {"unknown", ""}:
                return {
                    "target_column": str(data["target_column"]),
                    "problem_type": str(data.get("problem_type", "regression")),
                }
        except Exception:
            pass
    profile = inp.parent.parent / "scout" / "dataset_profile.md"
    out = {"target_column": "unknown", "problem_type": "classification"}
    if not profile.exists():
        return out
    text = profile.read_text(encoding="utf-8", errors="ignore")
    for key in ("target_column", "problem_type"):
        for line in text.splitlines():
            if line.lower().startswith(key):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    out[key] = parts[1].strip()
                    break
    return out


def _prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise SystemExit(f"Mo target '{target}' is missing from Finn output")
    y = df[target]
    drop_cols = [target, "split"]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise SystemExit("Mo has no numeric engineered features")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y


def _split(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 50:
        raise SystemExit("Not enough rows for Mo model fitting")
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = max(1, int(n * 0.8))
    return idx[:cut], idx[cut:]


def _split_from_labels(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    # honour explicit split column if Finn created one
    if "split" in df.columns:
        labels = df["split"].astype(str).str.lower().reset_index(drop=True)
        train_idx = labels.index[labels == "train"].to_numpy()
        test_idx = labels.index[labels == "test"].to_numpy()
        if len(train_idx) > 0 and len(test_idx) > 0 and len(train_idx) + len(test_idx) <= n:
            return train_idx, test_idx
    # stratify by freq__Metric when available (long-format datasets with heterogeneous metrics)
    if "freq__Metric" in df.columns:
        try:
            from sklearn.model_selection import train_test_split
            idx = np.arange(n)
            strat = df["freq__Metric"].astype(str).to_numpy()
            train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=strat)
            return train_idx, test_idx
        except Exception:
            pass
    return None


def _classification_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    labels = sorted({*map(str, y_true.tolist()), *map(str, pred.tolist())})
    total = len(y_true)
    accuracy = float(np.mean(np.asarray(y_true.astype(str)) == np.asarray(pred.astype(str)))) if total else 0.0
    f1s = []
    supports = []
    yt = y_true.astype(str).to_numpy()
    yp = np.asarray(pred).astype(str)
    for label in labels:
        tp = int(((yt == label) & (yp == label)).sum())
        fp = int(((yt != label) & (yp == label)).sum())
        fn = int(((yt == label) & (yp != label)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        support = int((yt == label).sum())
        f1s.append(f1)
        supports.append(support)
    weighted_f1 = float(np.average(f1s, weights=supports)) if sum(supports) else 0.0
    return {"accuracy": accuracy, "f1_weighted": weighted_f1}


def _fit_nearest_centroid(X: pd.DataFrame, y: pd.Series) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    y_str = y.astype(str)
    for label in sorted(y_str.unique()):
        centroids[label] = X.loc[y_str == label].mean(axis=0).to_numpy(dtype=float)
    return centroids


def _predict_nearest_centroid(model: dict[str, np.ndarray], X: pd.DataFrame) -> np.ndarray:
    labels = list(model)
    matrix = np.vstack([model[label] for label in labels])
    x = X.to_numpy(dtype=float)
    dists = ((x[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
    return np.asarray([labels[i] for i in dists.argmin(axis=1)])


def _evaluate_classification(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    train_idx, test_idx = _split(len(X))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.astype(str))
    y_test_enc = le.transform(y_test.astype(str))

    majority = y_train.astype(str).mode().iloc[0]
    majority_pred_str = np.full(len(y_test), majority, dtype=object)

    candidates: list[tuple[str, object]] = [
        ("majority_baseline", None),
        ("logistic_regression", Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=42))])),
        ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    try:
        from xgboost import XGBClassifier
        candidates.append(("xgboost", XGBClassifier(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")))
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        candidates.append(("lightgbm", LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)))
    except ImportError:
        pass

    rows = []
    fitted: dict[str, object] = {}
    for name, model in candidates:
        try:
            if model is None:
                pred_str = majority_pred_str
            else:
                model.fit(X_train, y_train_enc)
                pred_enc = model.predict(X_test)
                pred_str = le.inverse_transform(pred_enc.astype(int))
                fitted[name] = model
            rows.append({"model": name, **_classification_metrics(y_test, pred_str)})
        except Exception as exc:
            print(f"[WARN] Mo skipped {name}: {exc}")

    comparison = pd.DataFrame(rows).sort_values(["f1_weighted", "accuracy"], ascending=False).reset_index(drop=True)
    winner = comparison.iloc[0]["model"]

    if winner in fitted:
        full_enc = fitted[winner].predict(X)
        full_pred = le.inverse_transform(full_enc.astype(int))
    else:
        full_pred = np.full(len(X), majority, dtype=object)

    return comparison, full_pred


def _regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - pred)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 0.0 if denom == 0 else float(1.0 - np.sum((y_true - pred) ** 2) / denom)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _fit_ridge_numpy(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame, alpha: float) -> np.ndarray:
    x_train = X_train.to_numpy(dtype=float)
    x_pred = X_pred.to_numpy(dtype=float)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_pred = (x_pred - mean) / std
    design = np.column_stack([np.ones(len(x_train)), x_train])
    pred_design = np.column_stack([np.ones(len(x_pred)), x_pred])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
    return pred_design @ coef


def _evaluate_regression(X: pd.DataFrame, y: pd.Series, split_idx: tuple[np.ndarray, np.ndarray] | None = None, y_raw: pd.Series | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    train_idx, test_idx = split_idx if split_idx is not None else _split(len(X))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = pd.to_numeric(y.iloc[train_idx], errors="coerce").fillna(0).to_numpy(dtype=float)
    y_test_arr = pd.to_numeric(y.iloc[test_idx], errors="coerce").fillna(0).to_numpy(dtype=float)
    y_raw_test = None
    if y_raw is not None:
        y_raw_test = pd.to_numeric(y_raw.iloc[test_idx], errors="coerce").fillna(0).to_numpy(dtype=float)

    candidates: list[tuple[str, object]] = [
        ("mean_baseline", None),
        ("numpy_ridge_alpha_1", ("numpy_ridge", 1.0)),
        ("numpy_ridge_alpha_10", ("numpy_ridge", 10.0)),
    ]
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        candidates.extend([
            ("ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
            ("lasso", Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=2000))])),
            ("elasticnet", Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.1, max_iter=2000))])),
            ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ])
    except ImportError as exc:
        print(f"[WARN] Mo sklearn models unavailable; using numpy fallback: {exc}")

    # optionally add XGBoost / LightGBM if installed
    try:
        from xgboost import XGBRegressor
        candidates.append(("xgboost", XGBRegressor(n_estimators=100, random_state=42, verbosity=0)))
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        candidates.append(("lightgbm", LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)))
    except ImportError:
        pass

    rows = []
    fitted: dict[str, object] = {}
    mean_val = float(y_train.mean()) if len(y_train) else 0.0

    for name, model in candidates:
        try:
            if model is None:
                pred = np.full(len(y_test_arr), mean_val)
            elif isinstance(model, tuple) and model[0] == "numpy_ridge":
                pred = _fit_ridge_numpy(X_train, y_train, X_test, alpha=float(model[1]))
                fitted[name] = model
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                fitted[name] = model
            log_metrics = _regression_metrics(y_test_arr, pred)
            row = {
                "model": name,
                "rmse": log_metrics["rmse"],
                "mae": log_metrics["mae"],
                "r2": log_metrics["r2"],
                "rmse_log_scale": log_metrics["rmse"],
                "mae_log_scale": log_metrics["mae"],
                "r2_log_scale": log_metrics["r2"],
            }
            if y_raw_test is not None:
                pred_raw = np.expm1(np.clip(pred, 0, 700))
                raw_metrics = _regression_metrics(y_raw_test, pred_raw)
                row.update({
                    "rmse_inverse_value": raw_metrics["rmse"],
                    "mae_inverse_value": raw_metrics["mae"],
                    "r2_inverse_value": raw_metrics["r2"],
                    "rmse_original_scale": raw_metrics["rmse"],
                    "mae_original_scale": raw_metrics["mae"],
                    "r2_original_scale": raw_metrics["r2"],
                })
            rows.append(row)
        except Exception as exc:
            print(f"[WARN] Mo skipped {name}: {exc}")

    comparison = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    winner = comparison.iloc[0]["model"]

    if winner in fitted:
        if isinstance(fitted[winner], tuple) and fitted[winner][0] == "numpy_ridge":
            full_pred = _fit_ridge_numpy(X_train, y_train, X, alpha=float(fitted[winner][1]))
        else:
            full_pred = fitted[winner].predict(X)
    else:
        full_pred = np.full(len(X), mean_val)

    return comparison, full_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    profile = _read_profile(inp)
    target = profile.get("target_column", "unknown")
    problem_type = profile.get("problem_type", "classification").lower()
    if target.lower() == "unknown":
        raise SystemExit("Mo requires Scout target_column; got unknown")
    X, y = _prepare_xy(df, target)

    if problem_type == "classification":
        comparison, pred = _evaluate_classification(X, y)
        metric_lines = [
            f"accuracy: {comparison.iloc[0]['accuracy']:.6f}",
            f"f1_weighted: {comparison.iloc[0]['f1_weighted']:.6f}",
            "PR-AUC/Average Precision: not computed in builtin fallback; requires probabilistic classifier",
            "positive-class metrics: reported through weighted F1 across all classes",
            "threshold/cost-benefit: not applicable to deterministic multiclass fallback",
            "calibration: not available for nearest-centroid fallback; Brier requires probabilities",
        ]
    elif problem_type == "regression":
        y_numeric = pd.to_numeric(y, errors="coerce").fillna(0)
        # Hard GAID override: target column Value is always trained on log1p(Value).
        y_min = float(y_numeric.min())
        use_log = target == "Value"
        log_clip_count = int((y_numeric < 0).sum()) if use_log else 0
        y_raw = y_numeric.copy()
        if use_log:
            y_fit = pd.Series(np.log1p(y_numeric.clip(lower=0)), index=y_numeric.index)
        else:
            y_fit = y_numeric
        split_idx = _split_from_labels(df, len(X))
        split_source = "Finn prepared split labels" if split_idx is not None else "deterministic holdout split 80/20"
        if split_idx is None:
            split_idx = _split(len(X))
        comparison, pred = _evaluate_regression(
            X, y_fit, split_idx=split_idx,
            y_raw=y_raw if use_log else None,
        )
        output_pred = np.expm1(np.clip(pred, 0, 700)) if use_log else pred
        metric_lines = [
            f"rmse_log_scale: {comparison.iloc[0]['rmse_log_scale']:.6f}",
            f"mae_log_scale: {comparison.iloc[0]['mae_log_scale']:.6f}",
            f"r2_log_scale: {comparison.iloc[0]['r2_log_scale']:.6f}",
        ]
        if use_log:
            metric_lines.extend([
                "target_trained: log1p(Value)",
                "target_transform: log1p(Value)",
                "inverse_transform: np.expm1",
            ])
            if log_clip_count:
                metric_lines.append(f"target_transform_note: clipped {log_clip_count} negative Value rows to 0 before log1p")
            metric_lines.append("target_transform_note: hard override for target Value")
            if "rmse_inverse_value" in comparison.columns:
                metric_lines.extend([
                    f"rmse_original_scale: {comparison.iloc[0]['rmse_inverse_value']:.6f}",
                    f"mae_original_scale: {comparison.iloc[0]['mae_inverse_value']:.6f}",
                    f"r2_original_scale: {comparison.iloc[0]['r2_inverse_value']:.6f}",
                ])
        metric_lines.append(f"train_rows: {len(split_idx[0])}")
        metric_lines.append(f"test_rows: {len(split_idx[1])}")
        metric_lines.append(f"validation: {split_source}")
    else:
        raise SystemExit(f"Mo builtin supports classification/regression, got {problem_type}")

    missing_dependencies = _missing_dependencies()
    if missing_dependencies:
        metric_lines.append(f"missing_dependencies: {', '.join(missing_dependencies)}")

    comparison.to_csv(out / "model_comparison.csv", index=False)
    comparison.to_csv(out / "model_results.csv", index=False)
    if problem_type == "regression" and target == "Value":
        out_df = pd.DataFrame({"prediction_log_scale": pred, "prediction": output_pred, "actual": y.to_numpy()})
    else:
        out_df = pd.DataFrame({"prediction": pred, "actual": y.to_numpy()})
    out_df.to_csv(out / "mo_output.csv", index=False)
    report_lines = [
        "MO_REPORT",
        "=========",
        "",
        "PRODUCTION_READINESS",
        "====================",
        f"problem_type: {problem_type}",
        f"target_column: {target}",
        f"winner model: {comparison.iloc[0]['model']}",
        *metric_lines,
        "business impact: use predictions as an analytical signal only until domain validation is complete",
        "risk: builtin model is a minimum viable benchmark, not production approval",
        "confidence: Low" if problem_type == "classification" and comparison.iloc[0].get("f1_weighted", 0) < 0.85 else "confidence: Medium",
    ]
    (out / "mo_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (out / "mo_summary.json").write_text(
        json.dumps({"target_column": target, "problem_type": problem_type, "winner": comparison.iloc[0].to_dict()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[STATUS] CSV saved: {out / 'mo_output.csv'}")
    print(f"[STATUS] Model comparison saved: {out / 'model_comparison.csv'}")
    print(f"[STATUS] Report saved: {out / 'mo_report.md'}")


if __name__ == "__main__":
    main()
