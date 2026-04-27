from __future__ import annotations

import re
from pathlib import Path


class ProjectDetector:
    """Detect an active project from Anna text without guessing newest project."""

    def __init__(self, projects_dir: Path) -> None:
        self.projects_dir = projects_dir

    def detect(self, text: str) -> Path | None:
        if not self.projects_dir.exists():
            return None
        candidates = sorted(
            (p for p in self.projects_dir.iterdir() if p.is_dir()),
            key=lambda x: len(x.name),
            reverse=True,
        )
        for p in candidates:
            if p.name in text:
                return p
        match = re.search(r"projects[/\\]([\w\-][^\n\"'\\/:*?<>|]*)", text)
        if match:
            p = self.projects_dir / match.group(1).strip()
            if p.exists():
                return p
        return None


class AgentSpecLoader:
    """Load and rank agent markdown specs for Anna planning context."""

    def __init__(
        self,
        agents_dir: Path,
        valid_agents: set[str],
        chars_per_agent: int = 5000,
        max_total_chars: int = 80_000,
    ) -> None:
        self.agents_dir = agents_dir
        self.valid_agents = valid_agents
        self.chars_per_agent = chars_per_agent
        self.max_total_chars = max_total_chars

    def load(self, user_input: str = "") -> str:
        all_names = sorted(self.valid_agents)
        ordered = self._rank_agents(all_names, user_input) if user_input else all_names

        parts: list[str] = []
        total = 0
        for name in ordered:
            f = self.agents_dir / f"{name}.md"
            if not f.exists():
                continue
            content = f.read_text(encoding="utf-8")[: self.chars_per_agent]
            chunk = f"\n\n=== AGENT SPEC: {name.upper()} ===\n{content}"
            if total + len(chunk) > self.max_total_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "".join(parts)

    def _rank_agents(self, all_names: list[str], user_input: str) -> list[str]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            descs = []
            for name in all_names:
                f = self.agents_dir / f"{name}.md"
                descs.append(f.read_text(encoding="utf-8")[:300] if f.exists() else name)
            mat = TfidfVectorizer(min_df=1).fit_transform([user_input] + descs)
            scores = cos_sim(mat[0:1], mat[1:])[0]
            return [all_names[i] for i in scores.argsort()[::-1]]
        except Exception:
            return all_names

