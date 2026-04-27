from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable


LogFn = Callable[[str, str, str, str], None]


class KnowledgeBase:
    """File-backed knowledge base with lightweight relevance search."""

    def __init__(self, knowledge_dir: Path, log: LogFn | None = None) -> None:
        self.knowledge_dir = knowledge_dir
        self.log = log

    def _log(self, role: str, content: str, task: str = "", output: str = "") -> None:
        if self.log:
            self.log(role, content, task, output)

    def load(self, agent_name: str) -> str:
        files = sorted(self.knowledge_dir.glob(f"{agent_name}_*.md"))
        if not files:
            return ""
        parts: list[str] = []
        for f in files:
            try:
                parts.append(f.read_text(encoding="utf-8"))
            except Exception:
                pass
        return "\n\n".join(parts)

    def load_relevant(self, agent_name: str, task: str, top_n: int = 6) -> str:
        kb = self.load(agent_name)
        if not kb:
            return ""
        sections = [s.strip() for s in re.split(r"\n(?=##)", kb.strip()) if s.strip()]
        if len(sections) <= top_n:
            return kb
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            docs = [task] + sections
            mat = TfidfVectorizer(min_df=1).fit_transform(docs)
            scores = cos_sim(mat[0:1], mat[1:])[0]
            top_idx = scores.argsort()[::-1][:top_n]
            return "\n\n".join(sections[i] for i in sorted(top_idx))
        except ImportError:
            words = set(task.lower().split())
            scored = [(len(words & set(s.lower().split())), s) for s in sections]
            return "\n\n".join(s for _, s in sorted(scored, reverse=True)[:top_n])

    def save(self, agent_name: str, content: str, entry_type: str = "discovery") -> None:
        self.knowledge_dir.mkdir(exist_ok=True)
        f = self.knowledge_dir / f"{agent_name}_methods.md"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        tag_map = {"feedback": "FEEDBACK", "proven": "PROVEN", "deprecated": "DEPRECATED"}
        tag = tag_map.get(entry_type, "DISCOVERY")
        with open(f, "a", encoding="utf-8") as fp:
            fp.write(f"\n\n## [{ts}] [{tag}]\n{content.strip()}\n")
        self._log("system", f"KB [{tag}] {agent_name}: {content[:80]}", task="kb_update")

    def consolidate(self, agent_name: str) -> None:
        kb = self.load(agent_name)
        if not kb:
            return
        sections = [s.strip() for s in re.split(r"\n(?=##)", kb.strip()) if s.strip()]
        if len(sections) < 10:
            return

        proven = [s for s in sections if "[PROVEN]" in s]
        deprecated = [s for s in sections if "[DEPRECATED]" in s]
        normal = [s for s in sections if "[PROVEN]" not in s and "[DEPRECATED]" not in s]
        removed_deprecated = len(deprecated)

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            if len(normal) >= 2:
                mat = TfidfVectorizer(min_df=1).fit_transform(normal)
                sim = cos_sim(mat)
                remove: set[int] = set()
                for i in range(len(normal)):
                    if i in remove:
                        continue
                    for j in range(i + 1, len(normal)):
                        if j not in remove and sim[i, j] > 0.85:
                            remove.add(i)
                normal = [normal[i] for i in range(len(normal)) if i not in remove]
        except ImportError:
            pass

        kept = proven + normal
        f = self.knowledge_dir / f"{agent_name}_methods.md"
        f.write_text("\n\n".join(kept), encoding="utf-8")
        removed_total = len(sections) - len(kept)
        if removed_total:
            self._log(
                "system",
                f"KB consolidate {agent_name}: ลบ {removed_total} entries "
                f"(deprecated={removed_deprecated}, duplicate={removed_total-removed_deprecated})",
                task="kb_consolidate",
            )

