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

    # Stem substrings that indicate a file is a main data output (not supplementary).
    _MAIN_DATA_STEMS = ("_output", "_data", "engineered", "cleaned", "processed", "features", "train")
    # Stem substrings that identify known supplementary/diagnostic files — excluded from 2b fallback.
    _SUPPLEMENTARY_STEMS = (
        "correlat", "mi_score", "feature_score", "clustered", "summary",
        "metric", "importance", "flag", "shap", "outlier", "report",
    )
    _CSV_HANDOFF_AGENTS = {"scout", "dana", "eddie", "max", "finn"}

    def rebuild_from_project(self, project_dir: Path) -> None:
        """Repopulate store from a project's existing output files (for resume).

        CSV priority:
          1. <agent>_output.csv          — canonical name
          2a. any *.csv whose stem contains a main-data keyword (engineered, cleaned, _data, …)
          2b. any remaining *.csv        — last resort, avoids picking supplementary files first
          3. <agent>_report.md
          4. any *.md
        """
        self.clear()
        output_root = project_dir / "output"
        if not output_root.exists():
            return
        for agent_dir in output_root.iterdir():
            if not agent_dir.is_dir():
                continue
            agent = agent_dir.name
            # Priority 1: canonical <agent>_output.csv
            named_csv = sorted(agent_dir.glob(f"{agent}_output.csv"), key=lambda x: x.stat().st_mtime)
            if named_csv:
                self.write(agent, str(named_csv[-1]))
                continue
            # Priority 2a: CSVs whose stem suggests main data AND don't match the denylist
            # (prevents e.g. "training_metrics.csv" winning via the "train" keyword)
            all_csv = sorted(agent_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime)
            main_csv = [f for f in all_csv
                        if any(pat in f.stem.lower() for pat in self._MAIN_DATA_STEMS)
                        and not any(pat in f.stem.lower() for pat in self._SUPPLEMENTARY_STEMS)]
            if main_csv:
                self.write(agent, str(main_csv[-1]))
                continue
            # Priority 2b: non-supplementary CSVs (denylist filters known diagnostic files)
            allowed_csv = [f for f in all_csv
                           if not any(pat in f.stem.lower() for pat in self._SUPPLEMENTARY_STEMS)]
            if allowed_csv:
                self.write(agent, str(allowed_csv[-1]))
                continue
            # Priority 2c: last resort — any CSV (even supplementary) if nothing else exists
            if all_csv:
                self.write(agent, str(all_csv[-1]))
                continue
            if agent in self._CSV_HANDOFF_AGENTS:
                continue
            # Priority 3: <agent>_report.md
            named_md = sorted(agent_dir.glob(f"{agent}_report.md"), key=lambda x: x.stat().st_mtime)
            if named_md:
                self.write(agent, str(named_md[-1]))
                continue
            # Priority 4: any *.md
            any_md = sorted(agent_dir.glob("*.md"), key=lambda x: x.stat().st_mtime)
            if any_md:
                self.write(agent, str(any_md[-1]))

    def path_files(self) -> list[Path]:
        return sorted(self.pipeline_dir.glob("*_path.txt")) if self.pipeline_dir.exists() else []
