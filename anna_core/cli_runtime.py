from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .pipeline_store import PipelineStore
from .state import OrchestratorState


@dataclass(frozen=True)
class CliPalette:
    reset: str
    bold: str
    dim: str
    cyan: str
    green: str
    yellow: str
    red: str
    blue: str
    magenta: str
    white: str


class CliRenderer:
    def __init__(
        self,
        *,
        state: OrchestratorState,
        pipeline: PipelineStore,
        projects_dir: Path,
        deepseek_model: str,
        claude_model: str,
        claude_limit: int,
        mode: str,
        palette: CliPalette,
    ) -> None:
        self.state = state
        self.pipeline = pipeline
        self.projects_dir = projects_dir
        self.deepseek_model = deepseek_model
        self.claude_model = claude_model
        self.claude_limit = claude_limit
        self.mode = mode
        self.p = palette

    def _infer_project_from_pipeline(self) -> Path | None:
        projects_root = self.projects_dir.resolve()
        for pf in reversed(self.pipeline.path_files()):
            value = pf.read_text(encoding="utf-8").strip()
            if not value:
                continue
            output_path = Path(value)
            if not output_path.is_absolute():
                output_path = (Path.cwd() / output_path).resolve()
            else:
                output_path = output_path.resolve()
            try:
                rel = output_path.relative_to(projects_root)
            except ValueError:
                continue
            if rel.parts:
                project = self.projects_dir / rel.parts[0]
                if project.exists():
                    return project
        return None

    def print_header(self) -> None:
        p = self.p
        ds_ok = bool(os.environ.get("DEEPSEEK_API_KEY"))
        cl_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        lines = [
            "DataScienceOS - Anna (CEO)",
            f"DeepSeek: {self.deepseek_model} | Claude: {self.claude_model}",
            f"PATH-BASED pipeline v2 | mode: {self.mode} | Claude limit: {self.claude_limit}",
            "Type /help for commands",
        ]
        width = max(len(line) for line in lines) + 4
        print("+" + "-" * width + "+")
        for line in lines:
            print("|  " + line.ljust(width - 2) + "|")
        print("+" + "-" * width + "+")
        print()
        ds_str = "OK" if ds_ok else "MISSING KEY"
        cl_str = "OK" if cl_ok else "MISSING KEY"
        print(
            f"  {p.blue}{p.bold}DeepSeek:{p.reset} {ds_str}    "
            f"{p.magenta}{p.bold}Claude:{p.reset} {cl_str}  "
            f"{p.dim}(limit: {self.claude_limit} calls/session){p.reset}"
        )
        print()

    def print_status(self) -> None:
        p = self.p
        print("\n+-- STATUS --------------------------------------------------+")
        if self.state.active_project is None:
            self.state.active_project = self._infer_project_from_pipeline()
        project = self.state.active_project.name if self.state.active_project else "none"
        print(f"| Project : {p.bold}{project}{p.reset}")
        print(f"| Claude  : {self.state.claude_calls}/{self.claude_limit} calls")
        if self.state.agent_iter_count:
            iters = ", ".join(f"{a}x{n}" for a, n in self.state.agent_iter_count.items() if n > 1)
            if iters:
                print(f"| Iterations: {iters}")
        path_files = self.pipeline.path_files()
        if path_files:
            print("| Pipeline outputs:")
            for pf in path_files:
                agent = pf.stem.replace("_path", "")
                value = pf.read_text(encoding="utf-8").strip()
                exists_mark = "OK" if Path(value).exists() else "MISSING"
                print(f"|   {agent:<10} {exists_mark:<7} {value[-70:]}")
        else:
            print("| No pipeline output yet")
        print("+------------------------------------------------------------+")

    def print_claude_usage(self) -> None:
        p = self.p
        used = self.state.claude_calls
        limit = self.claude_limit
        pct = int(used / limit * 100) if limit > 0 else 100
        bar_len = 20
        filled = int(bar_len * used / limit) if limit > 0 else bar_len
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"\n  {p.magenta}{p.bold}Claude usage:{p.reset}  [{bar}]  {used}/{limit} ({pct}%)")
        if used >= limit:
            print("  Limit reached. DeepSeek will be used instead.")
        else:
            print(f"  Remaining calls: {limit - used}. Reset with /end or end session.")
        print()

    def print_help(self) -> None:
        print(
            "\n"
            "+-- COMMANDS -----------------------------------------------+\n"
            "| Plain text                 -> ask Anna / run pipeline\n"
            "| /project <name>            -> set active project  (aliases: /p, /proj)\n"
            "| /resume [name] [task]      -> continue pipeline   (alias: /r)\n"
            "| /run-all [task]            -> run full agent sequence from Scout to Rex\n"
            "| /status                    -> show pipeline state (aliases: /s, /st)\n"
            "| /kb <agent>                -> show agent knowledge base\n"
            "| /claude                    -> show Claude usage\n"
            "| /end                       -> reset session\n"
            "| /exit                      -> quit\n"
            "| @agent <task>              -> run one agent with DeepSeek\n"
            "| @agent! <task>             -> run one agent with Claude discover\n"
            "| !! <task>                  -> Anna discover mode\n"
            "+------------------------------------------------------------+\n"
        )

    def print_kb(self, name: str, content: str) -> None:
        if content:
            print(f"\n+-- KB: {name} " + "-" * max(0, 48 - len(name)) + "+")
            print(content)
            print("+------------------------------------------------------------+")
        else:
            print(f"  [{name}] no Knowledge Base yet")

    def resolve_project(self, name: str) -> tuple[Path | None, str]:
        project = self.projects_dir / name
        if project.exists():
            return project, ""
        matches = [
            d for d in self.projects_dir.iterdir()
            if d.is_dir() and name.lower() in d.name.lower()
        ] if self.projects_dir.exists() else []
        if len(matches) == 1:
            return matches[0], ""
        if len(matches) > 1:
            return None, f"Multiple projects matched: {', '.join(m.name for m in matches)}"
        return None, f"Project not found: {name}"
