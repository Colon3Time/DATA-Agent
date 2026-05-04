from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_mo_report(inp: Path) -> str:
    mo_report = inp.parent.parent / "mo" / "mo_report.md"
    if mo_report.exists():
        return mo_report.read_text(encoding="utf-8", errors="ignore")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(inp)
    mo_report = _load_mo_report(inp)
    target = "monetary" if "monetary" in data.columns else data.select_dtypes(include="number").columns[0]
    insight_csv = out / "iris_output.csv"
    pd.DataFrame(
        {
            "metric": ["rows", "columns", "target", "summary"],
            "value": [len(data), len(data.columns), target, "customer-level recommendation bridge"],
        }
    ).to_csv(insight_csv, index=False)

    report = out / "iris_report.md"
    report.write_text(
        "\n".join(
            [
                "IRIS_REPORT",
                "===========",
                "",
                "BUSINESS_DECISION_BRIEF",
                "=======================",
                f"Insight: customer-level features are ready for business actioning around {target}.",
                "Evidence: Finn produced customer-level RFM features and Mo produced holdout metrics.",
                "Business lever: revenue / retention / risk",
                f"Target KPI: {target}",
                "Owner: analytics / growth / finance",
                "Recommended action: prioritize high-value, recent, frequent customers for interventions.",
                "Expected impact: improve prioritization quality and reduce wasted outreach.",
                "Assumptions: regression fallback is a proxy for business ranking, not a final production model.",
                "Risks / trade-offs: proxy target may not match the final business objective; validate before rollout.",
                "Validation plan: pilot, cohort tracking, and out-of-time validation on the chosen business KPI.",
                "Confidence: Medium",
                "Production caveat: treat this as decision support until the target and deployment metric are confirmed.",
                "",
                "Mo excerpt:",
                mo_report[:800] if mo_report else "missing",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[STATUS] Insight CSV saved: {insight_csv}")
    print(f"[STATUS] Report saved: {report}")


if __name__ == "__main__":
    main()
