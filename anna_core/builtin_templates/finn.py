from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_role_blob(inp: Path) -> dict[str, object]:
    roles_path = inp.parent.parent / "dana" / "column_roles.json"
    if not roles_path.exists():
        return {}
    try:
        return json.loads(roles_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _read_profile(inp: Path) -> dict[str, str]:
    profile = inp.parent.parent / "scout" / "dataset_profile.md"
    if not profile.exists():
        return {"target_column": "unknown", "problem_type": "classification"}
    text = profile.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    for key in ("target_column", "problem_type"):
        for line in text.splitlines():
            if line.lower().startswith(key):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    out[key] = parts[1].strip()
                    break
    out.setdefault("target_column", "unknown")
    out.setdefault("problem_type", "classification")
    return out


def _pick_customer_col(df: pd.DataFrame) -> str | None:
    for name in ("Customer_ID", "Customer ID", "customer_id", "CustomerID", "firm_id", "survey_id", "entity_id", "respondent_id"):
        if name in df.columns:
            return name
    return None


def _pick_invoice_col(df: pd.DataFrame) -> str | None:
    for name in ("Invoice", "InvoiceNo", "invoice", "order_id", "OrderID", "survey_id"):
        if name in df.columns:
            return name
    return None


def _pick_value_col(df: pd.DataFrame, exclude: set[str]) -> str | None:
    candidates = [
        "revenue",
        "annual_sales_million_thb",
        "annual_sales",
        "sales",
        "turnover",
        "value",
        "amount",
        "profit",
        "employment",
        "employees",
        "firm_size",
        "capacity_utilization_pct",
    ]
    for name in candidates:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]) and name not in exclude:
            return name
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    return numeric[0] if numeric else None


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    for name in ("InvoiceDate", "Invoice_Date", "date", "Date", "order_date"):
        if name in df.columns:
            df[name] = pd.to_datetime(df[name], errors="coerce")
            return df
    return df


def _build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    cust_col = _pick_customer_col(df)
    if cust_col is None:
        cust_col = "row_id"
        df = df.copy()
        df[cust_col] = [f"row_{i}" for i in range(len(df))]
    inv_col = _pick_invoice_col(df)

    work = df.copy()
    work[cust_col] = work[cust_col].astype("string").str.strip()
    work = work[work[cust_col].notna()].copy()
    work = _ensure_datetime(work)

    if "revenue" not in work.columns:
        qty = pd.to_numeric(work["Quantity"], errors="coerce").fillna(0) if "Quantity" in work.columns else None
        price = pd.to_numeric(work["Price"], errors="coerce").fillna(0) if "Price" in work.columns else None
        if qty is not None and price is not None:
            work["revenue"] = qty * price
        else:
            value_col = _pick_value_col(work, exclude={cust_col})
            if value_col is not None:
                work["revenue"] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
            else:
                work["revenue"] = 0.0
    else:
        work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce").fillna(0)

    time_col = None
    for candidate in ("InvoiceDate", "Invoice_Date", "date", "Date", "order_date"):
        if candidate in work.columns:
            time_col = candidate
            break
    if time_col:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        anchor = work[time_col].max()
        recency_source = work.groupby(cust_col)[time_col].max()
        recency_days = (anchor - recency_source).dt.days.fillna(0)
    elif "survey_year" in work.columns:
        year = pd.to_numeric(work["survey_year"], errors="coerce")
        anchor = year.max()
        recency_source = work.groupby(cust_col)["survey_year"].max()
        recency_days = ((anchor - recency_source).fillna(0) * 365).astype(float)
    else:
        recency_days = pd.Series(0, index=work.groupby(cust_col).size().index)

    if inv_col:
        frequency = work.groupby(cust_col)[inv_col].nunique(dropna=True)
    else:
        frequency = work.groupby(cust_col).size()

    monetary = work.groupby(cust_col)["revenue"].sum()
    total_rows = work.groupby(cust_col).size()
    avg_revenue = work.groupby(cust_col)["revenue"].mean()

    rfm = pd.DataFrame(
        {
            "Customer ID": monetary.index.astype(str),
            "Customer_ID": monetary.index.astype(str),
            "recency_days": recency_days.reindex(monetary.index).fillna(recency_days.median() if len(recency_days) else 0).astype(float),
            "frequency": frequency.reindex(monetary.index).fillna(0).astype(float),
            "monetary": monetary.reindex(monetary.index).fillna(0).astype(float),
            "row_count": total_rows.reindex(monetary.index).fillna(0).astype(float),
            "avg_revenue": avg_revenue.reindex(monetary.index).fillna(0).astype(float),
        }
    )
    if "Country" in work.columns:
        country = work.groupby(cust_col)["Country"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0] if len(s) else "")
        rfm["country"] = country.reindex(monetary.index).fillna("")
    if "is_return" in work.columns:
        rfm["return_rate"] = work.groupby(cust_col)["is_return"].mean().reindex(monetary.index).fillna(0).astype(float)
    if "is_outlier" in work.columns:
        rfm["outlier_rate"] = work.groupby(cust_col)["is_outlier"].mean().reindex(monetary.index).fillna(0).astype(float)

    rfm = rfm.sort_values(["monetary", "frequency", "recency_days"], ascending=[False, False, True]).reset_index(drop=True)
    return rfm


def _write_manifest(out_dir: Path, profile: dict[str, str], rfm: pd.DataFrame) -> None:
    manifest = {
        "targets": {
            profile.get("target_column", "unknown"): {
                "exclude_features": ["Customer ID", "Customer_ID"],
                "feature_columns": [c for c in rfm.columns if c not in {"Customer ID", "Customer_ID"}],
                "problem_type": profile.get("problem_type", "classification"),
            }
        },
        "notes": "Builtin Finn manifest for Mo handoff.",
    }
    (out_dir / "finn_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_report(out_dir: Path, inp_name: str, profile: dict[str, str], rfm: pd.DataFrame) -> None:
    report = out_dir / "finn_report.md"
    report.write_text(
        "\n".join(
            [
                "FINN_REPORT",
                "===========",
                "",
                "FEATURE_GOVERNANCE",
                "=================",
                f"feature_lineage: derived from {inp_name}",
                "prediction_time_availability: customer-level RFM features only",
                "leakage_controls: drop raw ids/date/label-like fields before modeling",
                "train_only_transforms: aggregation only; no target-aware transforms",
                "temporal/OOT support columns: recency_days is computed relative to latest observed date",
                "actionability: hand off customer-level features to Mo",
                "warnings: confirm target ownership from Scout and use column_roles.json when available",
                f"target column : {profile.get('target_column', 'unknown')}",
                f"selected features : {', '.join([c for c in rfm.columns if c not in {'Customer ID', 'Customer_ID'}][:10])}",
                f"engineered_columns: {len(rfm.columns)}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    role_blob = _read_role_blob(inp)
    profile = _read_profile(inp)
    rfm = _build_rfm(df)

    out_csv = out / "finn_output.csv"
    rfm.to_csv(out_csv, index=False)
    (out / "engineered_data.csv").write_text(rfm.to_csv(index=False), encoding="utf-8")
    _write_manifest(out, profile, rfm)
    _write_report(out, inp.name, profile, rfm)

    # Lightweight summary for debugging and downstream traceability.
    summary = {
        "rows": len(rfm),
        "columns": list(rfm.columns),
        "target_column": profile.get("target_column", "unknown"),
        "problem_type": profile.get("problem_type", "classification"),
        "role_keys": sorted(role_blob.get("roles", {}).keys())[:20] if isinstance(role_blob.get("roles", {}), dict) else [],
    }
    (out / "finn_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[STATUS] CSV saved: {out_csv}")
    print(f"[STATUS] Manifest saved: {out / 'finn_feature_manifest.json'}")
    print(f"[STATUS] Report saved: {out / 'finn_report.md'}")


if __name__ == "__main__":
    main()
