from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_profile(inp: Path) -> dict[str, str]:
    # User-specified override wins over Scout's inferred target
    override_path = inp.parent.parent.parent / "target_override.json"
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


def _retail_rfm_possible(df: pd.DataFrame) -> bool:
    cols = {str(c).lower().replace(" ", "_") for c in df.columns}
    return bool(
        {"quantity", "price"} <= cols
        and any(c in cols for c in {"customer_id", "customerid", "customer_id"})
        and any(c in cols for c in {"invoicedate", "invoice_date", "date", "order_date"})
    )


def _safe_name(name: object) -> str:
    return str(name).strip().replace(" ", "_")


def _build_generic_features(df: pd.DataFrame, profile: dict[str, str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    target = profile.get("target_column", "unknown")
    if target not in df.columns and profile.get("problem_type", "").lower() in {"classification", "regression"}:
        raise SystemExit(f"Scout target '{target}' is missing from Finn input")

    work = df.copy()
    id_like = []
    for col in work.columns:
        lc = str(col).lower()
        nunique = work[col].nunique(dropna=True)
        unique_ratio = nunique / max(len(work), 1)
        if col != target and (lc.endswith("_id") or lc in {"id", "uuid"} or unique_ratio > 0.98):
            id_like.append(col)

    feature_source = work.drop(columns=id_like, errors="ignore")
    y = feature_source[target] if target in feature_source.columns else None
    X = feature_source.drop(columns=[target], errors="ignore")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    pieces: list[pd.DataFrame] = []
    if numeric_cols:
        num = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan)
        num = num.fillna(num.median(numeric_only=True)).fillna(0)
        pieces.append(num.add_prefix("num__"))
    for col in cat_cols:
        series = X[col].astype("string").fillna("__missing__")
        if series.nunique(dropna=True) <= 30:
            dummies = pd.get_dummies(series, prefix=f"cat__{_safe_name(col)}", dummy_na=False, dtype="int8")
            pieces.append(dummies)
        else:
            freq = series.map(series.value_counts(normalize=True)).astype(float)
            pieces.append(pd.DataFrame({f"freq__{_safe_name(col)}": freq}))

    features = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=work.index)
    if y is not None:
        features[target] = y.to_numpy()
    return features.reset_index(drop=True), id_like, list(features.columns)


def _write_manifest(out_dir: Path, profile: dict[str, str], feature_columns: list[str], excluded: list[str]) -> None:
    target = profile.get("target_column", "unknown")
    manifest = {
        "targets": {
            target: {
                "exclude_features": sorted(set([target, *excluded])),
                "feature_columns": [c for c in feature_columns if c != target],
                "problem_type": profile.get("problem_type", "classification"),
            }
        },
        "notes": "Builtin Finn generic feature table. RFM is used only for explicit retail transaction schemas.",
    }
    (out_dir / "finn_feature_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_report(out_dir: Path, inp_name: str, profile: dict[str, str], feature_columns: list[str], excluded: list[str]) -> None:
    target = profile.get("target_column", "unknown")
    selected = [c for c in feature_columns if c != target][:30]
    report = "\n".join(
        [
            "FINN_REPORT",
            "===========",
            "",
            "FEATURE_GOVERNANCE",
            "==================",
            f"feature_lineage: derived from {inp_name}",
            "feature_mode: generic supervised feature table",
            "prediction_time_availability: row-level columns only; target is retained only for Mo training",
            "leakage_controls: high-cardinality id-like columns excluded; target excluded from feature list",
            "train_only_transforms: encoding/imputation recipe is deterministic and does not use target statistics",
            f"target column : {target}",
            f"problem_type : {profile.get('problem_type', 'classification')}",
            f"excluded columns : {', '.join(excluded) if excluded else 'none'}",
            f"selected features : {', '.join(selected) if selected else 'none'}",
            f"engineered_columns: {len(feature_columns)}",
            "handoff_status: ready for Mo",
        ]
    )
    (out_dir / "finn_report.md").write_text(report, encoding="utf-8")


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
    if _retail_rfm_possible(df) and profile.get("target_column", "").lower() in {"unknown", "", "monetary"}:
        raise SystemExit("Retail segmentation/RFM fallback requires explicit Scout target/problem ownership")

    features, excluded, feature_columns = _build_generic_features(df, profile)
    out_csv = out / "finn_output.csv"
    features.to_csv(out_csv, index=False)
    features.to_csv(out / "engineered_data.csv", index=False)
    _write_manifest(out, profile, feature_columns, excluded)
    _write_report(out, inp.name, profile, feature_columns, excluded)
    (out / "finn_summary.json").write_text(
        json.dumps(
            {
                "rows": len(features),
                "columns": list(features.columns),
                "target_column": profile.get("target_column", "unknown"),
                "problem_type": profile.get("problem_type", "classification"),
                "feature_mode": "generic_supervised",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[STATUS] CSV saved: {out_csv}")
    print(f"[STATUS] Report saved: {out / 'finn_report.md'}")


if __name__ == "__main__":
    main()