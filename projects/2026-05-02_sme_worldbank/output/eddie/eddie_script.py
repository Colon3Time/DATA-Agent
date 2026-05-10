from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_profile(inp: Path) -> dict[str, str]:
    profile = inp.parent.parent / "scout" / "dataset_profile.md"
    if not profile.exists():
        return {"target_column": "unknown", "problem_type": "classification"}
    text = profile.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    for key in ("target_column", "problem_type", "business_question"):
        for line in text.splitlines():
            if line.lower().startswith(f"{key.lower()}"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    out[key] = parts[1].strip()
                    break
    out.setdefault("target_column", "unknown")
    out.setdefault("problem_type", "classification")
    return out


def _normalize_customer_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Customer_ID" in df.columns and "Customer ID" not in df.columns:
        df["Customer ID"] = df["Customer_ID"]
    elif "Customer ID" in df.columns and "Customer_ID" not in df.columns:
        df["Customer_ID"] = df["Customer ID"]
    return df


def _ensure_revenue(df: pd.DataFrame) -> pd.DataFrame:
    if "revenue" not in df.columns:
        if "Quantity" in df.columns and "Price" in df.columns:
            qty = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
            price = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
            df["revenue"] = qty * price
        else:
            df["revenue"] = 0.0
    return df


def _write_report(out_dir: Path, inp_name: str, profile: dict[str, str], df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    top_cols = []
    if not numeric.empty:
        top_cols = list(numeric.mean().sort_values(ascending=False).head(5).index)
    report = out_dir / "eddie_report.md"
    report.write_text(
        "\n".join(
            [
                "EDDIE_REPORT",
                "============",
                "",
                "PIPELINE_SPEC",
                "=============",
                f"problem_type: {profile.get('problem_type', 'classification')}",
                f"target_column: {profile.get('target_column', 'unknown')}",
                "recommended_model: baseline + tree model comparison",
                "preprocessing: preserve row-level transaction columns, normalize customer id aliases, carry outlier flags forward",
                f"key_features: {', '.join(top_cols) if top_cols else 'none'}",
                "",
                "BUSINESS_EDA_FRAME",
                "==================",
                f"business question: what behavior patterns in {inp_name} should be preserved for modeling and business follow-up?",
                "decision owner: analytics / modeling team",
                f"target kpi: {profile.get('target_column', 'unknown')}",
                f"strongest evidence: quick scan of the cleaned table with {len(df.columns)} columns",
                "causality status: correlational only",
                "temporal/leakage risk: date and customer identifiers must be handled carefully downstream",
                "imbalance/skew risk: verify class balance or heavy-tailed drivers before modeling",
                "validation strategy: use Finn + Mo validation after this bridge report",
                "",
                "BUSINESS_EDA_BRIEF",
                "==================",
                f"Insight: cleaned transaction data still preserves the main transaction structure from {inp_name}.",
                f"Evidence: {len(df):,} rows and {len(df.columns)} columns after cleaning.",
                "Business hypothesis: revenue/retention signals can be derived from transaction and customer history.",
                "Follow-up question: which fields should be excluded or encoded before Finn builds features?",
                "Next handoff: iris_eda / Finn",
                "Risk / caveat: exploratory framing only; do not replace Scout target ownership.",
                "Confidence: Medium",
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
    df = _normalize_customer_column(df)
    df = _ensure_revenue(df)

    out_csv = out / "eddie_output.csv"
    df.to_csv(out_csv, index=False)
    _write_report(out, inp.name, _read_profile(inp), df)

    # Compact summary files help downstream agents stay deterministic.
    summary = out / "eddie_summary.csv"
    pd.DataFrame(
        {
            "column": list(df.columns),
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_pct": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    ).to_csv(summary, index=False)

    print(f"[STATUS] CSV saved: {out_csv}")
    print(f"[STATUS] Summary saved: {summary}")
    print(f"[STATUS] Report saved: {out / 'eddie_report.md'}")


if __name__ == "__main__":
    main()