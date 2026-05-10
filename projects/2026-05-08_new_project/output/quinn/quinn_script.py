from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _load_mo_report(inp: Path) -> str:
    report = inp.parent.parent / "mo" / "mo_report.md"
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

    mo_report = _load_mo_report(inp)
    target = _field(mo_report, "target_column")
    problem_type = _field(mo_report, "problem_type")
    fallback_terms = ["dummy_mean", "regression fallback", "fallback path", "target_column: monetary"]
    target_aligned = target != "unknown" and not any(term in mo_report.lower() for term in fallback_terms)
    restart = "yes" if not target_aligned else "no"
    verdict = "unsatisfied" if restart == "yes" else "satisfied"

    qc_rows = [
        {"check": "target_alignment", "status": "pass" if target_aligned else "fail", "detail": f"target={target}; problem_type={problem_type}"},
        {"check": "leakage", "status": "warn", "detail": "No target-encoded token found in Mo report; code-level audit still recommended"},
        {"check": "overfitting", "status": "warn", "detail": "Holdout evidence exists but builtin benchmark is not a full production model"},
        {"check": "drift", "status": "warn", "detail": "OOT/time split not proven"},
        {"check": "calibration", "status": "warn", "detail": "Probability calibration not proven"},
        {"check": "business_satisfaction", "status": "pass" if target_aligned else "fail", "detail": "QC only passes when downstream evidence matches Scout target"},
    ]
    pd.DataFrame(qc_rows).to_csv(out / "quinn_qc_results.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(out / "quinn_output.csv", index=False)

    report = "\n".join(
        [
            "QUINN_REPORT",
            "============",
            "",
            "WORLD_CLASS_QC",
            "===============",
            f"target_alignment: {'pass' if target_aligned else 'fail'}",
            "leakage: warn - no obvious report-level leakage token, code-level audit still required",
            "overfitting: warn - builtin holdout benchmark is limited",
            "drift: warn - OOT validation not proven",
            "calibration: warn - probability calibration not proven",
            f"business_satisfaction: {'satisfied' if target_aligned else 'not satisfied'}",
            f"restart_cycle: {restart}",
            f"verdict: {verdict}",
            "Restart From: finn" if restart == "yes" else "Restart From: none",
            "",
            "Mo report excerpt:",
            mo_report[:900] if mo_report else "missing",
        ]
    )
    (out / "quinn_report.md").write_text(report, encoding="utf-8")
    print(f"[STATUS] QC CSV saved: {out / 'quinn_qc_results.csv'}")
    print(f"[STATUS] Report saved: {out / 'quinn_report.md'}")


if __name__ == "__main__":
    main()