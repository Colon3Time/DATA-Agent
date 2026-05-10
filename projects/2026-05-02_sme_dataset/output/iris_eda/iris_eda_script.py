from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _infer_business_hypothesis(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return "Dataset is mostly non-numeric, so business hypothesis should focus on segmentation and data quality."
    miss_pct = numeric.isna().mean().sort_values(ascending=False)
    top_missing = miss_pct.index[0]
    return f"Columns with the most missingness start with {top_missing}; this may be the best follow-up area for Finn/Mo."


def _column_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = max(len(df), 1)
    for col in df.columns:
        ser = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(ser.dtype),
                "missing_pct": round(float(ser.isna().mean() * 100), 2),
                "n_unique": int(ser.nunique(dropna=True)),
                "sample_value": "" if ser.dropna().empty else str(ser.dropna().iloc[0])[:80],
                "role_hint": _role_hint(col),
                "coverage": round(float((1.0 - ser.isna().mean()) * 100), 2),
                "share": round(float(ser.notna().sum() / total * 100), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "n_unique"], ascending=[False, False])


def _role_hint(name: str) -> str:
    n = name.lower()
    if any(key in n for key in ("target", "label", "outcome", "churn", "default", "review_score")):
        return "target-candidate"
    if n.endswith("_id") or n in {"id", "customer id", "customer_id"}:
        return "identifier"
    if any(key in n for key in ("date", "time", "month", "period", "year")):
        return "time"
    if any(key in n for key in ("amount", "price", "revenue", "score", "count", "qty", "quantity")):
        return "numeric-driver"
    return "feature"


def _top_findings(df: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    if len(df.columns) == 0:
        return ["No columns found."]
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        miss = numeric.isna().mean().sort_values(ascending=False)
        if not miss.empty:
            findings.append(f"Highest numeric missingness: {miss.index[0]} ({miss.iloc[0]*100:.1f}%).")
        corr = numeric.corr().abs()
        if corr.shape[0] >= 2:
            tri = corr.where(~np.eye(corr.shape[0], dtype=bool))
            if tri.max().max() > 0.7:
                pair = tri.stack().idxmax()
                findings.append(f"Strong numeric relationship detected between {pair[0]} and {pair[1]}.")
    if not findings:
        findings.append("No strong numeric signal surfaced from the quick bridge scan.")
    return findings


def _write_report(report_path: Path, df: pd.DataFrame, summary_csv: Path) -> None:
    insights = _top_findings(df)
    business_hypothesis = _infer_business_hypothesis(df)
    report_path.write_text(
        "\n".join(
            [
                "Iris EDA Bridge Report",
                "=======================",
                "",
                "BUSINESS_EDA_BRIEF",
                "==================",
                f"Insight: {insights[0]}",
                f"Evidence: quick bridge scan from {summary_csv.name} and column profiling.",
                "Business hypothesis: the strongest operational leverage is likely in the most incomplete or most skewed drivers.",
                "Follow-up question: which of the flagged columns should Finn keep, derive, or exclude before modeling?",
                "Next handoff: Finn",
                "Risk / caveat: this is exploratory and should not overwrite Scout target ownership.",
                "Confidence: Medium",
                "",
                "Observations",
                "------------",
                f"Rows: {len(df):,}",
                f"Columns: {len(df.columns):,}",
                business_hypothesis,
                "",
                "Top findings:",
                *[f"- {item}" for item in insights],
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
    summary = _column_table(df)
    summary_csv = out / "iris_eda_output.csv"
    summary.to_csv(summary_csv, index=False)
    report_path = out / "iris_eda_report.md"
    _write_report(report_path, df, summary_csv)

    print(f"[STATUS] Loaded: {df.shape}")
    print(f"[STATUS] Summary CSV saved: {summary_csv}")
    print(f"[STATUS] Report saved: {report_path}")


if __name__ == "__main__":
    main()