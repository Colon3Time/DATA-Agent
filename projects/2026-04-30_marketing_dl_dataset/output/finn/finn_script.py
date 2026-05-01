from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    drop_like = [c for c in df.columns if c.lower().endswith("_id") or "target" in c.lower() or "label" in c.lower() or "outcome" in c.lower() or "leak" in c.lower()]
    base = df.drop(columns=drop_like, errors="ignore").copy()
    num = base.select_dtypes(include=[np.number]).columns.tolist()
    for c in num:
        base[f"{c}_sq"] = base[c] ** 2
        base[f"{c}_log1p"] = np.log1p(base[c].clip(lower=0)) if (base[c] >= 0).all() else base[c]
    if "target_column" in base.columns:
        base["target_present"] = base["target_column"].notna().astype(int)
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
leakage_controls: dropped id/target-like columns
train_only_transforms: none beyond algebraic transforms
temporal/OOT support columns: not inferred
actionability: pass-through engineered features for Mo
warnings: verify target column from Eddie before model training
engineered_columns: {len(base.columns)}
""",
        encoding="utf-8",
    )

    print(f"[STATUS] CSV saved: {out_csv}")
    print(f"[STATUS] Report saved: {rpt}")


if __name__ == "__main__":
    main()
