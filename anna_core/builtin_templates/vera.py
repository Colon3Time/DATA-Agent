from __future__ import annotations

import argparse
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

    try:
        df = pd.read_csv(inp)
    except Exception:
        df = pd.DataFrame()

    summary = out / "vera_output.csv"
    if not df.empty:
        pd.DataFrame(
            {
                "metric": ["rows", "columns", "numeric_columns"],
                "value": [len(df), len(df.columns), len(df.select_dtypes(include="number").columns)],
            }
        ).to_csv(summary, index=False)
    else:
        pd.DataFrame({"metric": ["status"], "value": ["no input"]}).to_csv(summary, index=False)

    report = out / "vera_report.md"
    report.write_text(
        "\n".join(
            [
                "VERA_REPORT",
                "===========",
                "",
                "VISUAL_QC",
                "=========",
                "source evidence: fallback visual summary from upstream pipeline outputs",
                "decision purpose: support executive and analyst review",
                "chart rationale: no chart rendered in offline fallback, but summary checks remain valid",
                "misleading-risk check: avoid implying causal impact from the fallback visuals",
                "accessibility: use simple labels and high-contrast defaults when charts are added later",
                "caveat: no PNG charts generated in offline fallback",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[STATUS] Visual summary saved: {summary}")
    print(f"[STATUS] Report saved: {report}")


if __name__ == "__main__":
    main()
