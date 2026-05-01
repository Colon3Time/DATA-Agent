from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    cols = list(df.columns)
    out_csv = out / "mo_output.csv"
    summary = pd.DataFrame([{
        "input_rows": len(df),
        "input_columns": len(cols),
        "top_columns": json.dumps(cols[:15]),
        "contains_target_hint": int(any("target" in c.lower() or "label" in c.lower() or "outcome" in c.lower() for c in cols)),
        "contains_numeric": int(any(pd.api.types.is_numeric_dtype(df[c]) for c in cols)),
    }])
    summary.to_csv(out_csv, index=False)

    rpt = out / "mo_report.md"
    rpt.write_text(
        f"""MO_REPORT
=========
PRODUCTION_READINESS
====================
Validation strategy: random CV proxy; builtin fallback
Observed columns: {len(cols)}
Threshold economics: not computed in builtin fallback
Calibration: not computed in builtin fallback
Dependency benchmark: not available in builtin fallback
Verdict: Prototype
Blocking gaps: {json.dumps(['replace builtin fallback with trained model flow'])}
""",
        encoding="utf-8",
    )

    print(f"[STATUS] CSV saved: {out_csv}")
    print(f"[STATUS] Report saved: {rpt}")


if __name__ == "__main__":
    main()
