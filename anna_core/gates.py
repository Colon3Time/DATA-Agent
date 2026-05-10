from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


def archive_agent_outputs_before_rerun(agent_dir: Path, agent_name: str) -> str:
    if not agent_dir.exists():
        return ""
    preserve_dirs = {"_archive", "__pycache__", "_tmp", ".gitkeep"}
    items = [
        p
        for p in agent_dir.iterdir()
        if p.name not in preserve_dirs
        and p.suffix.lower() != ".py"
        and ".run_id" not in p.name
    ]
    if not items:
        return ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = agent_dir / "_archive" / ts
    archive.mkdir(parents=True, exist_ok=True)
    for p in items:
        shutil.move(str(p), str(archive / p.name))
    try:
        rel = archive.relative_to(agent_dir.parent)
    except ValueError:
        rel = archive
    print(f"[ARCHIVE] {agent_name} outputs → {rel}")
    return str(archive)


def quinn_hard_gate(project_dir: Path | None) -> tuple[bool, str, str]:
    if not project_dir:
        return False, "", ""
    q = project_dir / "output" / "quinn" / "quinn_report.md"
    if not q.exists():
        q = project_dir / "output" / "quinn" / "quinn_qc_report.md"
    if not q.exists():
        return False, "", ""
    txt = q.read_text(encoding="utf-8", errors="ignore")
    txt_lower = txt.lower()
    if "restart_cycle: yes" not in txt_lower and "restart_cycle:yes" not in txt_lower:
        return False, "", ""
    m = re.search(r"restart\s+from[:\s]+(\w+)", txt, re.IGNORECASE)
    target = m.group(1).lower() if m else "finn"
    reason = f"Quinn ordered RESTART_CYCLE: YES (restart required; Restart From: {target.upper()})"
    return True, reason, target


def agent_output_newer_than(project_dir: Path, agent: str, reference: Path) -> bool:
    agent_dir = project_dir / "output" / agent
    if not agent_dir.exists() or not reference.exists():
        return False
    ref_time = reference.stat().st_mtime
    for p in agent_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".csv", ".md", ".json"} and p.stat().st_mtime > ref_time:
            return True
    return False


def check_pipeline_spec(output_dir: Path) -> bool:
    if not output_dir or not output_dir.exists():
        return False
    reports = sorted(output_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not reports:
        return False
    text = reports[0].read_text(encoding="utf-8", errors="ignore")
    required = ["PIPELINE_SPEC", "problem_type", "recommended_model", "target_column"]
    return all(k.lower() in text.lower() for k in required)


def resolve_input_path(prev_agent: str, raw_path: str, project_dir: Path | None) -> str:
    if prev_agent != "scout" or not raw_path or not project_dir:
        return raw_path
    if not raw_path.endswith(".md"):
        return raw_path
    input_dir = project_dir / "input"
    if not input_dir.exists():
        return raw_path
    csvs = sorted(input_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(csvs[0]) if csvs else raw_path
