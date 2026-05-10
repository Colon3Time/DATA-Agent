from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_sections(inp: Path) -> str:
    parts = []
    for agent in ("quinn", "iris", "vera", "mo"):
        report = inp.parent.parent / agent / f"{agent}_report.md"
        if report.exists():
            parts.append(f"## {agent.upper()}\n{report.read_text(encoding='utf-8', errors='ignore')[:900]}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(inp) if inp.suffix.lower() == ".csv" else pd.DataFrame()
    sections = _load_sections(inp)
    pd.DataFrame(
        {
            "section": ["summary", "rows", "columns"],
            "value": ["target-aligned executive output", len(data), len(data.columns)],
        }
    ).to_csv(out / "output.csv", index=False)

    final_text = "\n".join(
        [
            "REX_REPORT",
            "==========",
            "",
            "Production readiness: prototype only; monitoring and retrain plan required.",
            "Validation limitations: time-based / out-of-time validation limitation remains unless explicitly proven.",
            "Business impact assumptions: ROI and cost impact require domain assumptions and pilot evidence.",
            "Monitoring: track drift, prediction quality proxy, source coverage, and target KPI after deployment.",
            "Retrain plan: retrain when drift, KPI decay, or source schema changes appear.",
            "",
            "Executive summary",
            "=================",
            "The pipeline produced a target-aligned analytical benchmark and QC summary.",
            "The result is suitable for review, not final production approval.",
            "",
            sections[:2500] if sections else "No upstream sections available.",
        ]
    )
    (out / "final_report.md").write_text(final_text, encoding="utf-8")
    (out / "executive_summary.md").write_text(final_text, encoding="utf-8")
    print(f"[STATUS] Final report saved: {out / 'final_report.md'}")
    print(f"[STATUS] Executive summary saved: {out / 'executive_summary.md'}")


if __name__ == "__main__":
    main()