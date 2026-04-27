from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OrchestratorState:
    """Mutable runtime state for one orchestrator process."""

    anna_history: list[dict[str, str]] = field(default_factory=list)
    active_project: Path | None = None
    claude_calls: int = 0
    agent_iter_count: dict[str, int] = field(default_factory=dict)
    current_proc: subprocess.Popen[Any] | None = None
    stop_requested: threading.Event = field(default_factory=threading.Event)

    def reset_session(self) -> None:
        self.anna_history.clear()
        self.active_project = None
        self.claude_calls = 0
        self.agent_iter_count.clear()

    def reset_pipeline(self) -> None:
        self.agent_iter_count.clear()
        self.active_project = None

