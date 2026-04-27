from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable


ActiveProjectFn = Callable[[], Path | None]


class RawLogger:
    """Append compact runtime logs to global and active-project logs."""

    def __init__(self, logs_dir: Path, active_project: ActiveProjectFn) -> None:
        self.logs_dir = logs_dir
        self.active_project = active_project

    def log(self, role: str, content: str, task: str = "", output: str = "") -> None:
        ts = datetime.now().strftime("%H:%M")
        date = datetime.now().strftime("%Y-%m-%d")

        if role.lower() == "user":
            line = f"[{ts}] User: {content[:300]}\n"
        elif role.lower() in ("anna", "anna summary"):
            line = f"[{ts}] Anna | Action: {content[:200]}\n"
        elif role.lower() == "system":
            task_part = f" | {task}" if task else ""
            line = f"[{ts}] [SYS{task_part}] {content[:200]}\n"
        else:
            parts = [f"[{ts}] {role.upper()}"]
            if task:
                parts.append(f"Task: {task[:100]}")
            parts.append(f"Action: {content[:200]}")
            if output:
                parts.append(f"→ {output}")
            line = " | ".join(parts) + "\n"

        self.logs_dir.mkdir(exist_ok=True)
        with open(self.logs_dir / f"{date}_raw.md", "a", encoding="utf-8") as fp:
            fp.write(line)

        active_project = self.active_project()
        if active_project:
            proj_log_dir = active_project / "logs"
            proj_log_dir.mkdir(exist_ok=True)
            with open(proj_log_dir / f"{date}_raw.md", "a", encoding="utf-8") as fp:
                fp.write(line)


class SessionMemoryStore:
    """Persist short session summaries for Anna to load next time."""

    def __init__(self, knowledge_dir: Path, max_entries: int = 50) -> None:
        self.knowledge_dir = knowledge_dir
        self.max_entries = max_entries

    @property
    def memory_file(self) -> Path:
        return self.knowledge_dir / "anna_session_memory.md"

    def tail(self, max_chars: int = 2000) -> str:
        if not self.memory_file.exists():
            return ""
        return self.memory_file.read_text(encoding="utf-8")[-max_chars:]

    def save(self, project_name: str, agents_done: list[str], summary_text: str) -> None:
        self.knowledge_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        short = summary_text.strip()[:400].replace("\n", " ")
        entry = f"\n## [{ts}] {project_name}\nAgents: {', '.join(agents_done)}\n{short}\n"
        if self.memory_file.exists():
            existing = self.memory_file.read_text(encoding="utf-8")
            entries = [e for e in existing.split("\n## [") if e.strip()]
            if len(entries) >= self.max_entries:
                entries = entries[-(self.max_entries - 1):]
            self.memory_file.write_text("\n## [".join([""] + entries) + entry, encoding="utf-8")
        else:
            self.memory_file.write_text(entry, encoding="utf-8")

