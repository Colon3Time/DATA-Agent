from __future__ import annotations

from pathlib import Path


class PipelineStore:
    """Path-based handoff store for agent outputs."""

    def __init__(self, pipeline_dir: Path) -> None:
        self.pipeline_dir = pipeline_dir

    def write(self, agent_name: str, file_path: str) -> None:
        self.pipeline_dir.mkdir(exist_ok=True)
        (self.pipeline_dir / f"{agent_name}_path.txt").write_text(str(file_path), encoding="utf-8")

    def read(self, agent_name: str) -> str:
        f = self.pipeline_dir / f"{agent_name}_path.txt"
        return f.read_text(encoding="utf-8").strip() if f.exists() else ""

    def clear(self) -> None:
        if self.pipeline_dir.exists():
            for f in self.pipeline_dir.glob("*_path.txt"):
                f.unlink()

    def completed_agents(self) -> list[str]:
        if not self.pipeline_dir.exists():
            return []
        done: list[str] = []
        for pf in self.pipeline_dir.glob("*_path.txt"):
            agent = pf.stem.replace("_path", "")
            val = pf.read_text(encoding="utf-8").strip()
            if Path(val).exists():
                done.append(agent)
        return done

    def path_files(self) -> list[Path]:
        return sorted(self.pipeline_dir.glob("*_path.txt")) if self.pipeline_dir.exists() else []

