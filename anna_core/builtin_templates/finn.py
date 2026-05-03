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


def _role_columns(role_blob: dict[str, object], role_name: str) -> list[str]:
    roles = role_blob.get("roles", {})
    if not isinstance(roles, dict):
        return []
    return [str(c) for c, r in roles.items() if str(r).lower() == role_name.lower()]


def _drop_columns_by_role(df: pd.DataFrame, role_blob: dict[str, object], role_name: str) -> pd.DataFrame:
    role_cols = {c.strip().lower() for c in _role_columns(role_blob, role_name)}
    if not role_cols:
        return df
    keep = [c for c in df.columns if c.strip().lower() not in role_cols]
    return df.loc[:, keep].copy()


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
    role_drop = ["id", "date", "label"]
    if role_blob:
        role_drop.extend(_role_columns(role_blob, "id"))
        role_drop.extend(_role_columns(role_blob, "date"))
        role_drop.extend(_role_columns(role_blob, "label"))
    drop_like = [
        c for c in df.columns
        if c.lower().endswith("_id")
        or "target" in c.lower()
        or "label" in c.lower()
        or "outcome" in c.lower()
        or "leak" in c.lower()
    ]
    drop_cols = sorted(set(drop_like + role_drop))
    base = df.drop(columns=drop_cols, errors="ignore").copy()
    hard_drop = {c.lower() for c in role_drop}
    base = base.loc[:, [c for c in base.columns if c.strip().lower() not in hard_drop]].copy()
    if role_blob:
        base = _drop_columns_by_role(base, role_blob, "id")
        base = _drop_columns_by_role(base, role_blob, "date")
        base = _drop_columns_by_role(base, role_blob, "label")
    num = base.select_dtypes(include=[np.number]).columns.tolist()
    for c in num:
        base[f"{c}_sq"] = base[c] ** 2
        base[f"{c}_log1p"] = np.log1p(base[c].clip(lower=0)) if (base[c] >= 0).all() else base[c]
    base["row_count"] = len(base)

    out_csv = out / "finn_output.csv"
    base.to_csv(out_csv, index=False)
    (out / "engineered_data.csv").write_text(base.head(25).to_csv(index=False), encoding="utf-8")

    rpt = out / "finn_report.md"
    rpt.write_text(
        f"""FINN_REPORT
===========
FEATURE_GOVERNANCE
=================
feature_lineage: derived from {inp.name}
prediction_time_availability: cleaned, row-level features only
leakage_controls: dropped Dana roles id/date/label and target-like columns
train_only_transforms: none beyond algebraic transforms
temporal/OOT support columns: date-like roles removed from features and should be preserved only in analysis branches
actionability: pass-through engineered features for Mo
warnings: verify target column from Eddie before model training; if column_roles.json exists, trust it over name-only heuristics
engineered_columns: {len(base.columns)}
""",
        encoding="utf-8",
    )

    print(f"[STATUS] CSV saved: {out_csv}")
    print(f"[STATUS] Report saved: {rpt}")


if __name__ == "__main__":
    main()
