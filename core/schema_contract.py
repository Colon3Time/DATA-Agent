from __future__ import annotations

from pathlib import Path
from typing import Any


HANDOFF_SCHEMAS: dict[tuple[str, str], dict[str, Any]] = {
    ("scout", "dana"): {
        "required": [],
        "min_rows": 100,
    },
    ("dana", "eddie"): {
        # Eddie is an exploratory bridge stage, so the contract should not
        # hardcode a specific downstream domain schema. The guard only needs
        # to ensure Dana produced a substantive cleaned table.
        "required": [],
        "min_rows": 100,
    },
    ("eddie", "finn"): {
        # Finn can build customer-level features from generic row-level tables.
        "required": [],
        "min_rows": 100,
    },
    ("eddie", "iris_eda"): {
        "required": [],
        "min_rows": 20,
    },
    ("finn", "mo"): {
        # Finn output is project-specific. For retail datasets it may be RFM;
        # for generic supervised datasets it should be a row-level feature table
        # with the Scout-owned target still present. Validate that shape via the
        # manifest/target checks below instead of hardcoding customer columns.
        "required": [],
        "alternatives": {},
        "manifest_required": "finn_feature_manifest.json",
        "manifest_keys": ["targets"],
        "min_rows": 50,
    },
    ("finn", "iris"): {
        "required": [],
        "alternatives": {},
        "min_rows": 50,
    },
    ("eddie", "max"): {
        "required": ["Description", "StockCode", "Quantity", "InvoiceDate"],
        "dtypes": {"Quantity": "numeric"},
        "min_rows": 100,
    },
}


def _is_generic_survey_table(df) -> bool:
    columns = {str(col).strip().lower().replace(" ", "_") for col in df.columns}
    survey_markers = {
        "country",
        "country_name",
        "country_code",
        "dataset",
        "metric",
        "value",
        "source",
        "source_year",
        "source_type",
        "source_category",
        "source_file",
        "iso3",
        "survey_id",
        "firm_id",
        "entity_id",
        "respondent_id",
        "survey_year",
        "province_code",
        "province_name",
        "industry_isic_like",
        "years_operating",
        "province",
        "province_name",
        "region",
        "sector",
        "business_size",
        "year",
        "number_of_msme",
    }
    return len(columns & survey_markers) >= 4


def _is_generic_clustering_project(project_dir: str | Path | None) -> bool:
    if not project_dir:
        return False
    project_path = Path(project_dir)
    profile_path = project_path / "output" / "scout" / "dataset_profile.md"
    if not profile_path.exists():
        return False
    text = profile_path.read_text(encoding="utf-8", errors="ignore").lower()
    problem_type = ""
    for line in text.splitlines():
        if line.startswith("problem_type"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                problem_type = parts[1].strip().lower()
            break
    return problem_type in {"clustering", "unknown", "dataset_discovery"}


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
    alternatives_map = schema.get("alternatives", {})
    generic_survey_mode = _is_generic_survey_table(df) or _is_generic_clustering_project(project_dir)
    if generic_survey_mode and key in {("dana", "eddie"), ("eddie", "finn")}:
        required = []
        alternatives_map = {}
    if key in {("finn", "mo"), ("finn", "iris")} and project_dir:
        import json as _json
        target = ""
        problem_type = ""
        # target_override.json wins over Scout's dataset_profile.md
        override_path = Path(project_dir) / "target_override.json"
        if override_path.exists():
            try:
                _ov = _json.loads(override_path.read_text(encoding="utf-8"))
                _t = str(_ov.get("target_column", "")).strip()
                if _t and _t.lower() not in {"unknown", ""}:
                    target = _t
                problem_type = str(_ov.get("problem_type", "")).strip().lower()
            except Exception:
                pass
        if not target:
            profile_path = Path(project_dir) / "output" / "scout" / "dataset_profile.md"
            if profile_path.exists():
                text = profile_path.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    if line.lower().startswith("target_column"):
                        target = line.split(":", 1)[1].strip() if ":" in line else ""
                    elif line.lower().startswith("problem_type"):
                        problem_type = line.split(":", 1)[1].strip().lower() if ":" in line else ""
        if problem_type in {"classification", "regression"} and target and target.lower() != "unknown":
            required = list(dict.fromkeys([*required, target]))
    missing = []
    for col in required:
        if col in df.columns:
            continue
        alternatives = alternatives_map.get(col, [])
        if alternatives and any(alt in df.columns for alt in alternatives):
            continue
        missing.append(col)
    for canonical, alternatives in alternatives_map.items():
        if canonical in df.columns:
            continue
        if any(alt in df.columns for alt in alternatives):
            continue
        if canonical not in missing:
            missing.append(f"{canonical} (or one of {alternatives})")
    if missing:
        errors.append(f"Missing columns: {missing}. Expected: {required}")

    min_rows = int(schema.get("min_rows", 0) or 0)
    if min_rows:
        hard_floor = max(5, min_rows // 4)
        if len(df) < hard_floor:
            errors.append(
                f"Expected at least {min_rows} rows, found fewer in quick scan/read."
            )

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
