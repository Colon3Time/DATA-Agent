from __future__ import annotations

import re
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

    def _entries(self) -> list[str]:
        if not self.memory_file.exists():
            return []
        text = self.memory_file.read_text(encoding="utf-8")
        return [entry.strip() for entry in text.split("\n## [") if entry.strip()]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def load_relevant(self, query: str, top_n: int = 4) -> str:
        """Return the most relevant past sessions for the current query."""
        entries = self._entries()
        if not entries:
            return ""
        if not query.strip():
            return "\n\n".join(entries[-top_n:])
        if len(entries) <= top_n:
            return "\n\n".join(entries)

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return "\n\n".join(entries[-top_n:])

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            matrix = TfidfVectorizer(min_df=1).fit_transform([query] + entries)
            scores = cosine_similarity(matrix[0:1], matrix[1:])[0]
            top_idx = scores.argsort()[::-1][:top_n]
            return "\n\n".join(entries[i] for i in sorted(top_idx))
        except Exception:
            scored = []
            for entry in entries:
                entry_tokens = self._tokenize(entry)
                overlap = len(query_tokens & entry_tokens)
                bonus = 1 if any(term in entry.lower() for term in ("fail", "error", "rework", "leakage", "review")) else 0
                scored.append((overlap + bonus, entry))
            top = sorted(scored, key=lambda item: item[0], reverse=True)[:top_n]
            return "\n\n".join(entry for _score, entry in top if _score > 0) or "\n\n".join(entries[-top_n:])

    def build_context(self, query: str, *, tail_chars: int = 1200, top_n: int = 4) -> str:
        """Combine recent memory with query-relevant memory without overfilling context."""
        tail = self.tail(tail_chars).strip()
        relevant = self.load_relevant(query, top_n=top_n).strip()
        if not tail:
            return relevant
        if not relevant or relevant == tail:
            return tail
        if relevant in tail:
            return tail
        return f"{relevant}\n\n--- Recent Session Memory ---\n{tail}"

    def save(self, project_name: str, agents_done: list[str], summary_text: str) -> None:
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
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
