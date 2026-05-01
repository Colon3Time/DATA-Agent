from __future__ import annotations

import os
import subprocess
import sys
import re
import threading
import time
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


def _format_duration(seconds: int) -> str:
    minutes, secs = divmod(max(0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_python_script(
    script_path: Path,
    input_path: str,
    output_dir: Path,
    state: OrchestratorState,
    timeout_seconds: int = 300,
) -> ScriptRunResult:
    """Run an agent script and keep process control in shared state."""
    state.stop_requested.clear()
    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        input_path,
        "--output-dir",
        str(output_dir),
    ]
    try:
        project_dir = output_dir.parent.parent
        profile = project_dir / "output" / "scout" / "dataset_profile.md"
        if profile.exists():
            text = profile.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"target_column\s*:\s*(\S+)", text)
            if m and m.group(1).lower() != "unknown":
                cmd.extend(["--target", m.group(1)])
    except Exception:
        pass
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    with state._proc_lock:
        state._active_procs.add(proc)
    result: dict[str, str] = {"stdout": "", "stderr": ""}

    def _communicate() -> None:
        stdout, stderr = proc.communicate()
        result["stdout"] = stdout or ""
        result["stderr"] = stderr or ""

    worker = threading.Thread(target=_communicate, daemon=True)
    worker.start()
    try:
        started = time.monotonic()
        last_printed = -1
        agent_label = output_dir.name.upper()
        while worker.is_alive():
            worker.join(timeout=1)
            elapsed = int(time.monotonic() - started)
            remaining = timeout_seconds - elapsed
            if elapsed >= timeout_seconds:
                raise subprocess.TimeoutExpired(cmd, timeout_seconds)
            if elapsed == 0 or elapsed - last_printed >= 5:
                last_printed = elapsed
                print(
                    f"\r  {agent_label} running... "
                    f"elapsed {_format_duration(elapsed)} | "
                    f"left {_format_duration(remaining)}",
                    end="",
                    flush=True,
                )
        print(
            f"\r  {agent_label} finished in {_format_duration(int(time.monotonic() - started))}"
            + " " * 30
        )
    except KeyboardInterrupt:
        proc.kill()
        worker.join(timeout=5)
        with state._proc_lock:
            state._active_procs.discard(proc)
        raise KeyboardInterrupt
    except subprocess.TimeoutExpired:
        proc.kill()
        worker.join(timeout=5)
        with state._proc_lock:
            state._active_procs.discard(proc)
        timeout_msg = f"Script timed out after {timeout_seconds} seconds"
        stderr = ((result["stderr"] or "") + ("\n" if result["stderr"] else "") + timeout_msg)
        print(f"\r  {output_dir.name.upper()} timed out after {_format_duration(timeout_seconds)}" + " " * 20)
        return ScriptRunResult(
            stdout=result["stdout"] or "",
            stderr=stderr,
            returncode=124,
        )
    finally:
        with state._proc_lock:
            state._active_procs.discard(proc)

    return ScriptRunResult(
        stdout=result["stdout"] or "",
        stderr=result["stderr"] or "",
        returncode=proc.returncode,
    )
