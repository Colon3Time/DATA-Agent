from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_sections(inp: Path) -> str:
    parts = []
    for agent in ("quinn", "iris", "vera", "mo"):
        report = inp.parent.parent / agent / f"{agent}_report.md"
        if report.exists():
            parts.append(f"## {agent.upper()}\n{report.read_text(encoding='utf-8', errors='ignore')[:800]}")
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

    out_csv = out / "output.csv"
    pd.DataFrame(
        {
            "section": ["summary", "rows", "columns"],
            "value": ["final executive output", len(data), len(data.columns)],
        }
    ).to_csv(out_csv, index=False)

    final_report = out / "final_report.md"
    final_report.write_text(
        "\n".join(
            [
                "REX_REPORT",
                "==========",
                "",
                "Production readiness: prototype only; monitoring and retrain plan required.",
                "Validation limitations: out-of-time validation is still recommended before rollout.",
                "Business impact assumptions: treat fallback metrics as directional and confirm on the real KPI.",
                "Monitoring: track drift, accuracy proxy, and business KPI after deployment.",
                "Retrain plan: retrain when drift or KPI decay appears.",
                "",
                "Executive summary",
                "=================",
                "The pipeline completed with a fallback offline path.",
                "The generated outputs are usable for review but not final production approval.",
                "",
                sections[:2000] if sections else "No upstream sections available.",
            ]
        ),
        encoding="utf-8",
    )

    executive = out / "executive_summary.md"
    executive.write_text(final_report.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    print(f"[STATUS] Final report saved: {final_report}")
    print(f"[STATUS] Executive summary saved: {executive}")


if __name__ == "__main__":
    main()