from __future__ import annotations

import os
import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path

from .state import OrchestratorState


@dataclass(frozen=True)
class ScriptRunResult:
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class CommandRunResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def combined_output(self) -> str:
        return self.stdout + self.stderr


_DANGEROUS_SHELL_PATTERNS = [
    re.compile(r"\brm\s+.*(-r|-rf|-fr)\b", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b.*\b-Recurse\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[sq]\b", re.IGNORECASE),
    re.compile(r"\brmdir\s+/s\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\b.*\b-[xfd]+", re.IGNORECASE),
]


def is_shell_command_allowed(cmd: str) -> tuple[bool, str]:
    for pattern in _DANGEROUS_SHELL_PATTERNS:
        if pattern.search(cmd):
            return False, f"blocked by command policy: {pattern.pattern}"
    return True, ""


def run_shell_command(cmd: str, cwd: Path, timeout_seconds: int = 60) -> CommandRunResult:
    """Run a shell command through a single controlled helper."""
    allowed, reason = is_shell_command_allowed(cmd)
    if not allowed:
        return CommandRunResult(stdout="", stderr=reason, returncode=126)
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_seconds,
        cwd=str(cwd),
    )
    return CommandRunResult(
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        returncode=result.returncode,
    )


def run_inline_python(code: str, cwd: Path, timeout_seconds: int = 30) -> CommandRunResult:
    """Run inline Python in a subprocess instead of using exec."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_seconds,
        cwd=str(cwd),
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    return CommandRunResult(
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        returncode=result.returncode,
    )


def run_python_script(
    script_path: Path,
    input_path: str,
    output_dir: Path,
    state: OrchestratorState,
    timeout_seconds: int = 300,
) -> ScriptRunResult:
    """Run an agent script and keep process control in shared state."""
    state.stop_requested.clear()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(script_path),
            "--input",
            input_path,
            "--output-dir",
            str(output_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    state.current_proc = proc
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        proc.kill()
        proc.communicate()
        state.current_proc = None
        raise KeyboardInterrupt
    finally:
        state.current_proc = None

    return ScriptRunResult(
        stdout=stdout or "",
        stderr=stderr or "",
        returncode=proc.returncode,
    )
