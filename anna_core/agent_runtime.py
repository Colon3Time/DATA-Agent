from __future__ import annotations

import re
from pathlib import Path


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


def scout_input_csv(project_dir: Path | None) -> str:
    if not project_dir:
        return ""
    input_dir = project_dir / "input"
    if not input_dir.exists():
        return ""
    csvs = sorted(
        input_dir.glob("**/*.csv"),
        key=lambda x: (x.stat().st_size, x.stat().st_mtime),
        reverse=True,
    )
    return str(csvs[0]) if csvs else ""


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
    if agent_name == "scout" and project_dir:
        path_lines.append(f"Save dataset to : {project_dir / 'input'}/ ← ไฟล์ข้อมูลจริงต้องอยู่ที่นี่เท่านั้น")
    return "\n".join(path_lines) + f"\n\nTask: {task}" if path_lines else task
