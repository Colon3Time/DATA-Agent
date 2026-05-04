from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_profile(inp: Path) -> dict[str, str]:
    profile = inp.parent.parent / "scout" / "dataset_profile.md"
    if not profile.exists():
        return {"target_column": "monetary", "problem_type": "regression"}
    text = profile.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    for key in ("target_column", "problem_type"):
        for line in text.splitlines():
            if line.lower().startswith(key):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    out[key] = parts[1].strip()
                    break
    out.setdefault("target_column", "monetary")
    out.setdefault("problem_type", "regression")
    return out


def _choose_target(df: pd.DataFrame, profile: dict[str, str]) -> str:
    target = profile.get("target_column", "monetary")
    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        return target
    for candidate in ("monetary", "frequency", "recency_days", "avg_revenue"):
        if candidate in df.columns and pd.api.types.is_numeric_dtype(df[candidate]):
            return candidate
    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric:
        return numeric[0]
    raise SystemExit("No numeric target available for Mo fallback")


def _build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    work = df.copy()
    work = work.drop(columns=[c for c in ("Customer ID", "Customer_ID") if c in work.columns], errors="ignore")
    work = work.select_dtypes(include=[np.number]).copy()
    if target not in work.columns:
        raise SystemExit(f"Target {target} not present in numeric feature table")
    y = work[target].astype(float)
    X = work.drop(columns=[target], errors="ignore")
    X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def _train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if len(X) < 50:
        raise SystemExit("Not enough rows for Mo fallback model fitting")
    rng = np.random.default_rng(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = max(1, int(len(idx) * (1 - test_size)))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom == 0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _fit_linear_regression(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    ym = y.to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    return coef


def _predict_linear_regression(coef: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    return Xm @ coef


def _evaluate_models(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    X_train, X_test, y_train, y_test = _train_test_split(X, y)
    y_train_arr = y_train.to_numpy(dtype=float)
    y_test_arr = y_test.to_numpy(dtype=float)

    mean_pred = np.full(len(y_test_arr), y_train_arr.mean() if len(y_train_arr) else 0.0)
    linear_coef = _fit_linear_regression(X_train, y_train)
    linear_pred = _predict_linear_regression(linear_coef, X_test)

    rows = [
        {
            "model": "dummy_mean",
            "rmse": _rmse(y_test_arr, mean_pred),
            "mae": _mae(y_test_arr, mean_pred),
            "r2": _r2(y_test_arr, mean_pred),
        },
        {
            "model": "linear_regression",
            "rmse": _rmse(y_test_arr, linear_pred),
            "mae": _mae(y_test_arr, linear_pred),
            "r2": _r2(y_test_arr, linear_pred),
        },
    ]
    comp = pd.DataFrame(rows).sort_values(["rmse", "mae"], ascending=True).reset_index(drop=True)
    preds = {
        "dummy_mean": mean_pred,
        "linear_regression": linear_pred,
        "y_test": y_test_arr,
        "linear_coef": linear_coef,
        "x_columns": np.array(X.columns, dtype=object),
    }
    return comp, preds


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
    target = _choose_target(df, profile)
    X, y = _build_features(df, target)
    comparison, preds = _evaluate_models(X, y)
    winner = comparison.iloc[0].to_dict()

    comp_csv = out / "model_comparison.csv"
    comparison.to_csv(comp_csv, index=False)

    results_csv = out / "model_results.csv"
    comparison.to_csv(results_csv, index=False)

    report = out / "mo_report.md"
    report.write_text(
        "\n".join(
            [
                "MO_REPORT",
                "=========",
                "",
                "PRODUCTION_READINESS",
                "====================",
                f"target_column: {target}",
                f"winner model: {winner['model']}",
                f"rmse: {winner['rmse']:.6f}",
                f"mae: {winner['mae']:.6f}",
                f"r2: {winner['r2']:.6f}",
                "validation: holdout split 80/20",
                "calibration: not applicable for regression fallback",
                "threshold strategy: not applicable for regression fallback",
                "business impact: use predicted target as a prioritization signal",
                "risk: this is a fallback model, not a production benchmark",
                "confidence: Medium",
            ]
        ),
        encoding="utf-8",
    )

    if winner["model"] == "linear_regression":
        pred = _predict_linear_regression(preds["linear_coef"], X)
    else:
        pred = np.full(len(X), float(y.mean()) if len(y) else 0.0)
    output = pd.DataFrame({"prediction": pred, "actual": y.to_numpy(dtype=float)})
    output_csv = out / "mo_output.csv"
    output.to_csv(output_csv, index=False)

    print(f"[STATUS] CSV saved: {output_csv}")
    print(f"[STATUS] Model comparison saved: {comp_csv}")
    print(f"[STATUS] Report saved: {report}")


if __name__ == "__main__":
    main()
