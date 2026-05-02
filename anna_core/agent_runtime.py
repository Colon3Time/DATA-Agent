from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .mo_phase import detect_mo_phase


def output_dir_for(project_dir: Path | None, agent_name: str) -> Path | None:
    return (project_dir / "output" / agent_name) if project_dir else None


def delete_old_scripts(output_dir: Path | None) -> None:
    if output_dir and output_dir.exists():
        for old_py in output_dir.glob("*.py"):
            old_py.unlink()


def latest_input_file(project_dir: Path | None) -> str:
    if not project_dir:
        return ""
    input_dir = project_dir / "input"
    if not input_dir.exists():
        return ""
    extract_input_archives(input_dir)
    sqlites = sorted(input_dir.glob("**/*.sqlite"), key=lambda x: x.stat().st_mtime, reverse=True)
    if sqlites:
        return str(sqlites[0])
    csvs = sorted(
        input_dir.glob("**/*.csv"),
        key=lambda x: (x.stat().st_size, x.stat().st_mtime),
        reverse=True,
    )
    return str(csvs[0]) if csvs else ""


def latest_output_csv(project_dir: Path | None) -> str:
    if not project_dir:
        return ""
    output_root = project_dir / "output"
    if not output_root.exists():
        return ""
    csvs = sorted(output_root.glob("**/*_output.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(csvs[0]) if csvs else ""


def latest_rex_meeting_report(project_dir: Path | None) -> str:
    """Return the newest Rex executive/meeting report, if one exists."""
    if not project_dir:
        return ""
    rex_dir = project_dir / "output" / "rex"
    if not rex_dir.exists():
        return ""
    candidates: list[Path] = []
    for name in ("meeting_presentation.md", "executive_summary.md", "final_report.md"):
        p = rex_dir / name
        if p.exists():
            candidates.append(p)
    candidates.extend(rex_dir.glob("meeting_presentation*.md"))
    unique = list(dict.fromkeys(candidates))
    if not unique:
        return ""
    return str(sorted(unique, key=lambda x: x.stat().st_mtime, reverse=True)[0])


def should_regenerate_vera_script_for_meeting_report(
    script_path: Path | None,
    task: str,
    project_dir: Path | None,
) -> bool:
    """Vera visuals must be regenerated when a newer meeting report is available."""
    if not script_path or not project_dir:
        return False
    meeting_report = latest_rex_meeting_report(project_dir)
    if not meeting_report:
        return False
    task_lc = task.lower()
    explicit_alignment = any(
        key in task_lc
        for key in (
            "meeting",
            "presentation",
            "executive",
            "slide",
            "กราฟ",
            "ประชุม",
            "นำเสนอ",
        )
    )
    report_is_newer = Path(meeting_report).stat().st_mtime >= script_path.stat().st_mtime
    return explicit_alignment or report_is_newer


def scout_input_csv(project_dir: Path | None) -> str:
    if not project_dir:
        return ""
    input_dir = project_dir / "input"
    if not input_dir.exists():
        return ""
    extract_input_archives(input_dir)
    csvs = sorted(
        input_dir.glob("**/*.csv"),
        key=lambda x: (x.stat().st_size, x.stat().st_mtime),
        reverse=True,
    )
    return str(csvs[0]) if csvs else ""


def extract_input_archives(input_dir: Path) -> None:
    """Unpack downloaded zip datasets in input/ so downstream agents see real CSV files."""
    for archive in input_dir.glob("**/*.zip"):
        marker = archive.with_suffix(archive.suffix + ".extracted")
        if marker.exists():
            continue
        target = archive.parent / archive.stem
        target.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
            marker.write_text("ok\n", encoding="utf-8")
        except Exception:
            continue


def extract_python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def build_agent_path_message(
    agent_name: str,
    task: str,
    input_path: str,
    output_dir: Path | None,
    project_dir: Path | None,
) -> str:
    path_lines: list[str] = []
    if input_path:
        path_lines.append(f"Input file path : {input_path}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path_lines.append(f"Save CSV to     : {output_dir / f'{agent_name}_output.csv'}")
        path_lines.append(f"Save script to  : {output_dir / f'{agent_name}_script.py'}")
        path_lines.append(f"Save report to  : {output_dir / f'{agent_name}_report.md'}")
    if agent_name == "mo" and project_dir:
        phase = detect_mo_phase(task)
        if phase == 2:
            path_lines.append(f"Phase 1 report  : {project_dir / 'output' / 'mo' / 'mo_report.md'}")
            path_lines.append("Required phase  : Phase 2 Tune; run RandomizedSearchCV, do not reuse Phase 1 report")
        elif phase == 3:
            path_lines.append(f"Phase 2 report  : {project_dir / 'output' / 'mo' / 'model_results.md'}")
            path_lines.append(f"Phase 2 CSV     : {project_dir / 'output' / 'mo' / 'model_comparison.csv'}")
            path_lines.append("Required phase  : Phase 3 Validate; compare tuned model against default")
    if agent_name == "vera" and project_dir:
        meeting_report = latest_rex_meeting_report(project_dir)
        if meeting_report:
            path_lines.append(f"Meeting report  : {meeting_report}")
            path_lines.append(
                "Visual brief    : Build the chart plan from the meeting/executive report storyline first. "
                "Every important chart must map to a meeting section, claim, KPI, or recommendation; "
                "do not create decorative or column-driven charts that do not support the meeting report."
            )
    if agent_name == "scout" and project_dir:
        path_lines.append(f"Save dataset to : {project_dir / 'input'}/ ← ไฟล์ข้อมูลจริงต้องอยู่ที่นี่เท่านั้น")
    return "\n".join(path_lines) + f"\n\nTask: {task}" if path_lines else task
