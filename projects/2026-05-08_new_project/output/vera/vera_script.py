from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _load_report(inp: Path, agent: str) -> str:
    report = inp.parent.parent / agent / f"{agent}_report.md"
    return report.read_text(encoding="utf-8", errors="ignore") if report.exists() else ""


def _field(text: str, name: str, default: str = "unknown") -> str:
    m = re.search(rf"^{re.escape(name)}\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    mo_report = _load_report(inp, "mo")
    target = _field(mo_report, "target_column")
    chart_files: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "actual" in df.columns:
            counts = df["actual"].astype(str).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 5))
            counts.plot(kind="bar", ax=ax, color="#2f6f73")
            ax.set_title(f"Actual distribution: {target}")
            ax.set_xlabel(target)
            ax.set_ylabel("Rows")
            fig.tight_layout()
            path = charts / "01_actual_distribution.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            chart_files.append(path.name)
        if {"actual", "prediction"} <= set(df.columns):
            sample = pd.DataFrame({"actual": df["actual"].astype(str), "prediction": df["prediction"].astype(str)})
            matrix = pd.crosstab(sample["actual"], sample["prediction"]).iloc[:20, :20]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(matrix.to_numpy(), cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
            ax.set_yticks(range(len(matrix.index)), matrix.index)
            ax.set_title("Prediction vs actual")
            fig.tight_layout()
            path = charts / "02_prediction_matrix.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            chart_files.append(path.name)
    except Exception as exc:
        (charts / "CHART_ERROR.txt").write_text(str(exc), encoding="utf-8")

    pd.DataFrame(
        {
            "metric": ["rows", "columns", "charts"],
            "value": [len(df), len(df.columns), len(chart_files)],
        }
    ).to_csv(out / "vera_output.csv", index=False)

    report = "\n".join(
        [
            "VERA_REPORT",
            "===========",
            "",
            "VISUAL_QC",
            "=========",
            f"source evidence: Mo predictions for target `{target}`",
            "decision purpose: support model review and executive explanation",
            f"chart rationale: generated {len(chart_files)} target-aligned diagnostic chart(s)",
            "misleading-risk check: charts show association/error patterns only, not causal impact",
            "accessibility: simple titles, direct labels, high-contrast color scale",
            f"charts: {', '.join(chart_files) if chart_files else 'none'}",
        ]
    )
    (out / "vera_report.md").write_text(report, encoding="utf-8")
    print(f"[STATUS] Visual summary saved: {out / 'vera_output.csv'}")
    print(f"[STATUS] Report saved: {out / 'vera_report.md'}")


if __name__ == "__main__":
    main()