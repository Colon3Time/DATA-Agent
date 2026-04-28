from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    agents_dir: Path
    logs_dir: Path
    knowledge_dir: Path
    pipeline_dir: Path
    projects_dir: Path
    deepseek_url: str
    deepseek_model: str
    claude_model: str
    mode: str
    claude_limit: int
    step_mode: bool
    no_color: bool
    terminal_title: bool


def load_config(base_dir: Path, argv: list[str] | None = None) -> AppConfig:
    args = list(sys.argv if argv is None else argv)
    mode = "light"
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args):
            mode = args[idx + 1]

    claude_limit = 10
    if "--claude-limit" in args:
        idx = args.index("--claude-limit")
        if idx + 1 < len(args):
            try:
                claude_limit = int(args[idx + 1])
            except ValueError:
                pass

    return AppConfig(
        base_dir=base_dir,
        agents_dir=base_dir / "agents",
        logs_dir=base_dir / "logs",
        knowledge_dir=base_dir / "knowledge_base",
        pipeline_dir=base_dir / "pipeline",
        projects_dir=base_dir / "projects",
        deepseek_url="https://api.deepseek.com/chat/completions",
        deepseek_model="deepseek-chat",
        claude_model="claude-sonnet-4-6",
        mode=mode,
        claude_limit=claude_limit,
        step_mode="--auto" not in args,
        no_color="--no-color" in args,
        terminal_title="--no-title" not in args,
    )
