from __future__ import annotations

from pathlib import Path
from typing import Any


HANDOFF_SCHEMAS: dict[tuple[str, str], dict[str, Any]] = {
    ("scout", "dana"): {
        "required": [],
        "min_rows": 100,
    },
    ("dana", "eddie"): {
        "required": ["Invoice", "StockCode", "Quantity", "Price", "Customer ID", "InvoiceDate", "Country", "is_outlier"],
        "min_rows": 100,
    },
    ("eddie", "finn"): {
        "required": ["Customer ID", "StockCode", "Quantity", "Price", "InvoiceDate", "revenue", "is_return"],
        "alternatives": {"Invoice": ["Invoice", "InvoiceNo"]},
        "min_rows": 100,
    },
    ("finn", "mo"): {
        "required": ["Customer ID", "recency_days", "frequency", "monetary"],
        "manifest_required": "finn_feature_manifest.json",
        "manifest_keys": ["targets"],
        "min_rows": 50,
    },
    ("finn", "iris"): {
        "required": ["Customer ID", "recency_days", "frequency", "monetary"],
        "min_rows": 50,
    },
    ("eddie", "max"): {
        "required": ["Description", "StockCode", "Quantity", "InvoiceDate"],
        "dtypes": {"Quantity": "numeric"},
        "min_rows": 100,
    },
}


def validate_handoff(
    producer: str,
    consumer: str,
    csv_path: str | Path,
    project_dir: str | Path | None = None,
) -> tuple[bool, list[str]]:
    import json
    import pandas as pd

    key = (producer.lower(), consumer.lower())
    schema = HANDOFF_SCHEMAS.get(key)
    if not schema:
        return True, []

    path = Path(csv_path)
    errors: list[str] = []
    try:
        df = pd.read_csv(path, nrows=max(int(schema.get("min_rows", 5)), 5))
    except Exception as exc:
        return False, [f"Cannot read CSV: {exc}"]

    required = schema.get("required", [])
    missing = [col for col in required if col not in df.columns]
    for canonical, alternatives in schema.get("alternatives", {}).items():
        if canonical not in df.columns and not any(alt in df.columns for alt in alternatives):
            missing.append(f"{canonical} (or one of {alternatives})")
    if missing:
        errors.append(f"Missing columns: {missing}. Expected: {required}")

    min_rows = int(schema.get("min_rows", 0) or 0)
    if min_rows and len(df) < min_rows:
        errors.append(f"Expected at least {min_rows} rows, found fewer in quick scan/read.")

    for col, dtype in schema.get("dtypes", {}).items():
        if col in df.columns and dtype == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() < 0.95:
                errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}")

    manifest_name = schema.get("manifest_required")
    if manifest_name:
        manifest_path = path.parent / manifest_name
        if project_dir and not manifest_path.exists():
            manifest_path = Path(project_dir) / "output" / producer.lower() / manifest_name
        if not manifest_path.exists():
            errors.append(f"Missing manifest: {manifest_name}")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for key_name in schema.get("manifest_keys", []):
                    if key_name not in manifest:
                        errors.append(f"Manifest missing key: {key_name}")
                targets = manifest.get("targets", {})
                if isinstance(targets, dict):
                    for target, spec in targets.items():
                        if not isinstance(spec, dict) or "exclude_features" not in spec:
                            errors.append(f"Manifest target '{target}' missing exclude_features")
            except Exception as exc:
                errors.append(f"Cannot read manifest {manifest_name}: {exc}")

    return len(errors) == 0, errors


def infer_producer_from_path(input_path: str | Path, project_dir: Path | None) -> str:
    if not project_dir:
        return ""
    try:
        path = Path(input_path).resolve()
        output_root = (project_dir / "output").resolve()
        rel = path.relative_to(output_root)
    except Exception:
        return ""
    return rel.parts[0].lower() if rel.parts else ""
