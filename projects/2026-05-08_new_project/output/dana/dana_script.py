from __future__ import annotations
import argparse, csv, json
import re
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

def _load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise SystemExit(f"Unsupported input: {path}")

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _normalize_col_name(name: str) -> str:
    return str(name).strip().replace(" ", "_")

def _detect_target(df: pd.DataFrame) -> str | None:
    for kw in ["target", "label", "outcome", "review_score", "churn", "class", "status"]:
        for col in df.columns:
            if col.lower() == kw or col.lower().startswith(kw):
                return col
    return None

def _looks_like_date_col(col: str, series: pd.Series) -> bool:
    lc = col.lower()
    date_kw = ("date", "time", "timestamp", "datetime", "period", "month", "year")
    if any(k in lc for k in date_kw):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(20)
        if sample.empty:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.6
    return False

def _looks_like_id_col(col: str, series: pd.Series) -> bool:
    lc = col.lower()
    id_kw = ("_id", "id", "key", "code", "no", "num", "uuid", "guid")
    if lc in {"id", "row_id"} or lc.endswith("_id"):
        return True
    if any(k in lc for k in id_kw) and not any(k in lc for k in ("date", "price", "grid")):
        if series.nunique(dropna=True) >= max(10, int(len(series) * 0.5)):
            return True
    return False

def _classify_role(df: pd.DataFrame, col: str, target: str | None) -> str:
    if target and col == target:
        return "label"
    series = df[col]
    if _looks_like_date_col(col, series):
        return "date"
    if _looks_like_id_col(col, series):
        return "id"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"

def _build_role_map(df: pd.DataFrame, target: str | None) -> dict[str, str]:
    return {col: _classify_role(df, col, target) for col in df.columns}

def _role_summary(role_map: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"id": [], "date": [], "label": [], "numeric": [], "categorical": []}
    for col, role in role_map.items():
        out.setdefault(role, []).append(col)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="")
    p.add_argument("--output-dir", default="")
    args = p.parse_args()
    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _load_input(inp)
    raw_shape = df.shape

    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # User override takes precedence over keyword detection
    _override_path = inp.parent.parent.parent / "target_override.json"
    target = None
    if _override_path.exists():
        try:
            import json as _j
            _ov = _j.loads(_override_path.read_text(encoding="utf-8"))
            _t = str(_ov.get("target_column", "")).strip()
            if _t and _t.lower() not in {"unknown", ""} and _t in df.columns:
                target = _t
        except Exception:
            pass
    if target is None:
        target = _detect_target(df)
    role_map = _build_role_map(df, target)
    role_summary = _role_summary(role_map)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or c.lower() in {"quantity", "price", "unitprice", "freight_value", "payment_value"}]
    for col in numeric_cols:
        df[col] = _safe_numeric(df[col])

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    elif "Invoice_Date" in df.columns:
        df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"], errors="coerce")

    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].astype("string").str.strip()
        df.loc[df["CustomerID"].isin(["", "0", "0.0", "nan", "None"]), "CustomerID"] = pd.NA
    if "Customer_ID" in df.columns:
        df["Customer_ID"] = df["Customer_ID"].astype("string").str.strip()
        df.loc[df["Customer_ID"].isin(["", "0", "0.0", "nan", "None"]), "Customer_ID"] = pd.NA

    if "Description" in df.columns and "StockCode" in df.columns:
        desc_map = (
            df.dropna(subset=["Description"])
              .groupby("StockCode")["Description"]
              .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
              .to_dict()
        )
        df["Description"] = df["Description"].fillna(df["StockCode"].map(desc_map))

    if "Invoice" in df.columns:
        df["Invoice"] = df["Invoice"].astype("string").str.strip()
    elif "InvoiceNo" in df.columns:
        df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()

    if "Quantity" in df.columns:
        df["is_return"] = (df["Quantity"] < 0).fillna(False).astype(int)
    else:
        df["is_return"] = 0

    if "Invoice" in df.columns:
        cancel_col = df["Invoice"].astype("string").str.upper().str.startswith("C")
        df["is_cancellation"] = cancel_col.fillna(False).astype(int)
    elif "InvoiceNo" in df.columns:
        cancel_col = df["InvoiceNo"].astype("string").str.upper().str.startswith("C")
        df["is_cancellation"] = cancel_col.fillna(False).astype(int)
    else:
        df["is_cancellation"] = 0

    num_for_outlier = [c for c in ["Quantity", "UnitPrice", "Price", "FreightValue", "payment_value"] if c in df.columns]
    if num_for_outlier:
        z = pd.DataFrame(index=df.index)
        for col in num_for_outlier:
            s = df[col]
            if s.notna().sum() < 8:
                continue
            med = s.median()
            mad = (s - med).abs().median()
            if mad and not np.isclose(mad, 0):
                z[col] = ((s - med).abs() / (1.4826 * mad)).fillna(0)
        if not z.empty:
            outlier_mask = z.max(axis=1) > 6
            df["is_outlier"] = outlier_mask.fillna(False).astype(int)
        else:
            df["is_outlier"] = 0
    else:
        df["is_outlier"] = 0

    key_cols = [c for c in ["CustomerID", "Customer_ID", "Invoice", "InvoiceNo", "StockCode", "Description"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    cleaned_shape = df.shape
    out_csv = out / "dana_output.csv"
    df.to_csv(out_csv, index=False)
    flags = out / "outlier_flags.csv"
    flag_rows = []
    for idx, row in df[df["is_outlier"] == 1].head(10000).iterrows():
        flag_rows.append([int(idx), "composite", "", "Likely Real", "robust z-score > 6", "flagged"])
    with flags.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "column_name", "value", "verdict", "reason", "action"])
        w.writerows(flag_rows)

    role_path = out / "column_roles.json"
    role_payload = {
        "target_column": target,
        "roles": role_map,
        "role_summary": role_summary,
        "notes": {
            "id": "Columns used as identifiers or keys; downstream should not treat them as numeric features.",
            "date": "Datetime or date-like columns; keep for time-aware analysis and splitting.",
            "label": "Supervised target column if detected.",
            "numeric": "Safe candidate features for numeric analysis after cleaning.",
            "categorical": "Non-numeric feature columns.",
        },
    }
    role_path.write_text(json.dumps(role_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    missing_before = int(pd.isna(_load_input(inp)).sum().sum()) if inp.suffix.lower() == ".csv" else 0
    missing_after = int(df.isna().sum().sum())
    rpt = out / "dana_report.md"
    rpt.write_text(
        "DANA_REPORT\n===========\nDATA_QUALITY_AUDIT\n=================\n"
        f"raw_shape: {raw_shape[0]:,} x {raw_shape[1]}\n"
        f"clean_shape: {cleaned_shape[0]:,} x {cleaned_shape[1]}\n"
        f"target_column: {target or 'unknown'}\n"
        f"missing_before: {missing_before:,}\n"
        f"missing_after: {missing_after:,}\n"
        f"rows_removed_duplicates: {raw_shape[0] - cleaned_shape[0]:,}\n"
        f"is_outlier_count: {int(df['is_outlier'].sum()):,}\n"
        f"role_counts: id={len(role_summary.get('id', []))}, date={len(role_summary.get('date', []))}, label={len(role_summary.get('label', []))}, numeric={len(role_summary.get('numeric', []))}, categorical={len(role_summary.get('categorical', []))}\n"
        f"label_column: {target or 'unknown'}\n"
        f"id_columns: {', '.join(role_summary.get('id', [])[:12]) or 'none'}\n"
        f"date_columns: {', '.join(role_summary.get('date', [])[:12]) or 'none'}\n"
        "missing_handling: preserve raw semantics; normalize blanks to NA; limited imputation only where safe\n"
        "outlier_strategy: robust z-score on numeric columns + conservative flagging\n"
        "train_only_safeguards: none applied at cleaning stage\n"
        "bias_impact: duplicate rows removed only; no target-aware transforms applied\n"
        "feature_usage_guard: id-like columns should be excluded from numeric model features; label should be excluded from feature calculations when supervision is present\n"
        "downstream_warnings: cancellations/returns are preserved and flagged; confirm target leakage before modeling\n"
        "DATASET_RISK_REGISTER\n"
        "Source credibility: Medium\n"
        "License/usage: unknown from input file\n"
        "Business fit: High for retail behavior analysis\n"
        f"Target suitability: {target or 'ambiguous'}\n"
        "Recency/deployment fit: depends on source workbook\n"
        "Leakage risks: target and date-derived fields must be reviewed downstream\n"
        "Bias/coverage risks: anonymous customers / cancellation rows / missing descriptions\n"
        "Data dictionary: partial from workbook headers\n"
        "Verdict: Use with caveats\n",
        encoding="utf-8"
    )
    out_csv = out / "dana_output.csv"
    print(f"[STATUS] Cleaned: {raw_shape} -> {cleaned_shape}")
    print(f"[STATUS] CSV saved: {out_csv}")
    print(f"[STATUS] Role map saved: {role_path}")
    print(f"[STATUS] Report saved: {rpt}")
    print(f"[STATUS] Flags saved: {flags}")

if __name__ == "__main__":
    main()