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

    data = pd.read_csv(inp)
    mo_report = _load_report(inp, "mo")
    target = _field(mo_report, "target_column", "unknown")
    problem_type = _field(mo_report, "problem_type", "unknown")
    winner = _field(mo_report, "winner model", "unknown")

    pd.DataFrame(
        {
            "metric": ["rows", "columns", "target_column", "problem_type", "winner_model"],
            "value": [len(data), len(data.columns), target, problem_type, winner],
        }
    ).to_csv(out / "iris_output.csv", index=False)

    report = "\n".join(
        [
            "IRIS_REPORT",
            "===========",
            "",
            "BUSINESS_DECISION_BRIEF",
            "=======================",
            f"Insight: model evidence is available for target `{target}`.",
            f"Evidence: Mo benchmark selected `{winner}` for a `{problem_type}` task.",
            "Business lever: data governance, analytical prioritization, and risk review",
            f"Target KPI: {target}",
            "Owner: analytics / data product team",
            "Recommended action: review class balance, top error modes, and source coverage before operational rollout.",
            "Expected impact: improves confidence in downstream decisions by tying actions to the declared Scout target.",
            "Assumptions: builtin benchmark is a minimum viable analytical check, not final model selection.",
            "Risks / trade-offs: source metadata may encode collection-year patterns rather than deployable business behavior.",
            "Validation plan: out-of-time validation if a real time axis exists, plus holdout review and domain sign-off.",
            "Confidence: Low" if "confidence: low" in mo_report.lower() else "Confidence: Medium",
            "Production caveat: do not approve production until Quinn passes target-aligned QC.",
            "",
            "Mo excerpt:",
            mo_report[:800] if mo_report else "missing",
        ]
    )
    (out / "iris_report.md").write_text(report, encoding="utf-8")
    print(f"[STATUS] Insight CSV saved: {out / 'iris_output.csv'}")
    print(f"[STATUS] Report saved: {out / 'iris_report.md'}")


if __name__ == "__main__":
    main()
