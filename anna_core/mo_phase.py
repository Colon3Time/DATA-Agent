from __future__ import annotations

import re
from pathlib import Path


def detect_mo_phase(task: str) -> int | None:
    """Return the explicit CRISP-DM Mo phase requested by a dispatch task."""
    text = task.lower()
    if re.search(r"\bphase\s*3\b|phase\s*three|validate|validation|final validation", text):
        return 3
    if re.search(r"\bphase\s*2\b|phase\s*two|tune|tuning|randomizedsearchcv|hyperparameter", text):
        return 2
    if re.search(r"\bphase\s*1\b|phase\s*one|explore|default params|all algorithms", text):
        return 1
    return None


def mo_script_matches_phase(script_path: Path, phase: int | None) -> bool:
    """Protect Mo from reusing a script generated for another CRISP-DM phase."""
    if phase is None or not script_path.exists():
        return True
    try:
        text = script_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False

    if phase == 2:
        return "randomizedsearchcv" in text and (
            "phase 2" in text or "hyperparameter" in text or "tuning" in text
        )
    if phase == 3:
        has_phase3 = "phase 3" in text or "final validation" in text or "validate" in text
        compares_default = "default" in text and ("tuned" in text or "best_params" in text)
        return has_phase3 and compares_default
    if phase == 1:
        explores = ("phase 1" in text or "explore" in text) and "randomizedsearchcv" not in text
        return explores
    return True


def mo_report_matches_phase(report_path: Path, phase: int | None) -> bool:
    if phase is None:
        return True
    try:
        text = report_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    if phase == 2:
        return "phase: 2" in text or "phase 2" in text or "hyperparameter tuning" in text
    if phase == 3:
        return "phase: 3" in text or "phase 3" in text or "final validation" in text
    if phase == 1:
        return "phase: 1" in text or "phase 1" in text
    return True


def sync_mo_canonical_report(output_dir: Path, phase: int | None) -> Path | None:
    """Keep mo_report.md aligned with the latest real Mo phase artifact."""
    candidates_by_phase = {
        3: ["validation_results.md", "model_validation.md", "model_results.md", "agent_report.md"],
        2: ["model_results.md", "tuning_results.md", "agent_report.md"],
        1: ["model_results.md", "mo_report.md", "agent_report.md"],
        None: ["model_results.md", "agent_report.md", "mo_report.md"],
    }
    for name in candidates_by_phase.get(phase, candidates_by_phase[None]):
        candidate = output_dir / name
        if candidate.exists() and candidate.name != "mo_report.md":
            if phase is None or mo_report_matches_phase(candidate, phase):
                canonical = output_dir / "mo_report.md"
                canonical.write_text(candidate.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                return canonical
    return None
