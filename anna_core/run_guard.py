from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path


RUN_STATE_DIR = "_run_state"
CURRENT_RUN_ID_FILE = "current_run_id.txt"
CURRENT_RUN_META_FILE = "current_run_meta.json"
UPSTREAM_SOURCES = {
    "dana": ("scout",),
    "eddie": ("dana",),
    "iris_eda": ("eddie",),
    "finn": ("eddie",),
    "mo": ("finn",),
    "iris": ("finn",),
    "quinn": ("mo",),
    "vera": ("mo",),
    "rex": ("mo",),
}


def project_run_state_dir(project_dir: Path) -> Path:
    return project_dir / "logs" / RUN_STATE_DIR


def current_run_id(project_dir: Path | None) -> str:
    if not project_dir:
        return ""
    run_id_file = project_run_state_dir(project_dir) / CURRENT_RUN_ID_FILE
    if not run_id_file.exists():
        return ""
    return run_id_file.read_text(encoding="utf-8", errors="ignore").strip()


def begin_project_run(project_dir: Path | None, reason: str = "") -> str:
    if not project_dir:
        return ""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    state_dir = project_run_state_dir(project_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / CURRENT_RUN_ID_FILE).write_text(run_id, encoding="utf-8")
    meta = {
        "run_id": run_id,
        "reason": reason,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (state_dir / CURRENT_RUN_META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id


def output_run_marker(output_path: Path) -> Path:
    if is_run_marker(output_path):
        return output_path
    return output_path.with_name(output_path.name + ".run_id")


def is_run_marker(path: Path) -> bool:
    return path.name.endswith(".run_id")


def mark_output_current(output_path: Path, project_dir: Path | None, run_id: str | None = None) -> None:
    if not project_dir:
        return
    if not output_path.exists() or not output_path.is_file() or is_run_marker(output_path):
        return
    run_id = run_id or current_run_id(project_dir)
    if not run_id:
        return
    marker = output_run_marker(output_path)
    marker.write_text(run_id, encoding="utf-8")


def output_run_id(output_path: Path) -> str:
    marker = output_run_marker(output_path)
    if not marker.exists():
        return ""
    return marker.read_text(encoding="utf-8", errors="ignore").strip()


def output_is_current(output_path: Path, project_dir: Path | None) -> bool:
    if not project_dir:
        return True
    run_id = current_run_id(project_dir)
    if not run_id:
        return True
    if not output_path.exists():
        return False
    return output_run_id(output_path) == run_id


def archive_project_outputs(project_dir: Path | None) -> Path | None:
    if not project_dir:
        return None
    output_root = project_dir / "output"
    if not output_root.exists():
        return None
    archive_root = output_root / "_archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    moved = False
    for agent_dir in output_root.iterdir():
        if not agent_dir.is_dir() or agent_dir.name == "_archive":
            continue
        files = [p for p in agent_dir.iterdir() if p.name != ".gitkeep"]
        if not files:
            continue
        dest_dir = archive_root / agent_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in files:
            shutil.move(str(item), str(dest_dir / item.name))
            moved = True
    if moved:
        return archive_root
    return None


def promote_upstream_outputs(project_dir: Path | None, consumer_agent: str, run_id: str | None = None) -> None:
    if not project_dir:
        return
    run_id = run_id or current_run_id(project_dir)
    if not run_id:
        return
    for source_agent in UPSTREAM_SOURCES.get(consumer_agent.lower(), ()):
        source_dir = project_dir / "output" / source_agent
        if not source_dir.exists():
            continue
        for path in source_dir.rglob("*"):
            if path.is_file() and not is_run_marker(path):
                mark_output_current(path, project_dir, run_id)


def promote_pipeline_outputs(project_dir: Path | None, output_paths: list[str] | list[Path], run_id: str | None = None) -> None:
    if not project_dir:
        return
    run_id = run_id or current_run_id(project_dir)
    if not run_id:
        return
    for raw in output_paths:
        try:
            path = Path(raw)
        except Exception:
            continue
        if path.exists() and path.is_file() and not is_run_marker(path):
            mark_output_current(path, project_dir, run_id)
