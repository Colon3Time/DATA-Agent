from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

from .state import OrchestratorState


@dataclass(frozen=True)
class TerminalPalette:
    reset: str
    bold: str
    dim: str
    yellow: str
    red: str
    blue: str
    magenta: str


class LLMClient:
    """DeepSeek and Codex CLI client with shared usage tracking."""

    def __init__(
        self,
        *,
        state: OrchestratorState,
        deepseek_url: str,
        deepseek_model: str,
        codex_model: str | None = None,
        codex_limit: int | None = None,
        claude_model: str | None = None,
        claude_limit: int | None = None,
        palette: TerminalPalette,
    ) -> None:
        self.state = state
        self.deepseek_url = deepseek_url
        self.deepseek_model = deepseek_model
        self.codex_model = codex_model or claude_model or "gpt-5.5"
        self.codex_limit = codex_limit if codex_limit is not None else (claude_limit if claude_limit is not None else 10)
        self.claude_model = claude_model or self.codex_model
        self.claude_limit = claude_limit if claude_limit is not None else self.codex_limit
        self.palette = palette

    def call_deepseek(
        self,
        system_prompt: str,
        user_message: str,
        label: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """DeepSeek API, streaming, OpenAI-compatible."""
        p = self.palette
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print(f"{p.red}  [ERROR] DEEPSEEK_API_KEY not found in .env{p.reset}")
            return "[ERROR] DEEPSEEK_API_KEY not found in .env"
        if label:
            bar = "-" * max(0, 46 - len(label))
            print(f"\n{p.blue}[{p.bold}{label}{p.reset}{p.blue}] {bar}{p.reset}")
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        try:
            response = requests.post(
                self.deepseek_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": self.deepseek_model, "messages": messages, "stream": True},
                stream=True,
                timeout=180,
            )
            if response.status_code != 200:
                body = response.text[:500]
                print(f"{p.red}  [ERROR] DeepSeek HTTP {response.status_code}: {body}{p.reset}")
                return f"[ERROR] DeepSeek HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            print(f"{p.red}  [ERROR] DeepSeek connection failed{p.reset}")
            return "[ERROR] DeepSeek connection failed"
        except requests.exceptions.Timeout:
            print(f"{p.red}  [ERROR] DeepSeek timeout{p.reset}")
            return "[ERROR] DeepSeek timeout"

        full: list[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
            if text == "[DONE]":
                break
            try:
                token = json.loads(text)["choices"][0]["delta"].get("content", "")
                print(token, end="", flush=True)
                full.append(token)
            except (json.JSONDecodeError, KeyError):
                pass
        print()
        return "".join(full)

    def _codex_launcher(self) -> str | None:
        candidates = [
            shutil.which("codex.cmd"),
            shutil.which("codex"),
        ]
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(str(Path(appdata) / "npm" / "codex.cmd"))
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return None

    def call_codex(self, system_prompt: str, user_message: str, label: str = "") -> str:
        """Try Codex CLI first, then fall back to DeepSeek when unavailable."""
        p = self.palette
        if self.state.codex_calls >= self.codex_limit:
            print(
                f"\n{p.yellow}  [WARN] Codex limit reached {self.state.codex_calls}/{self.codex_limit} calls "
                f"-> use DeepSeek instead{p.reset}"
            )
            return self.call_deepseek(system_prompt, user_message, label=f"{label} (via DeepSeek[limit])")

        codex_bin = self._codex_launcher()
        if not codex_bin:
            print(f"\n{p.yellow}  [WARN] codex CLI not found -> use DeepSeek instead{p.reset}")
            return self.call_deepseek(system_prompt, user_message, label=f"{label} (via DeepSeek)")

        self.state.codex_calls += 1
        remaining = self.codex_limit - self.state.codex_calls
        print(f"\n{p.magenta}{'-' * 55}{p.reset}")
        print(
            f"{p.magenta}  [CODEX CLI] {p.bold}{label}{p.reset}  "
            f"{p.dim}[{self.state.codex_calls}/{self.codex_limit} left {remaining}]{p.reset}"
        )
        print(f"{p.magenta}{'-' * 55}{p.reset}")

        prompt = (
            f"{system_prompt}\n\n"
            f"User request:\n{user_message}\n\n"
            "Return only the answer content. If you need to propose file edits, "
            "describe them clearly in the response."
        )
        output_file: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as tmp:
                output_file = tmp.name
            cmd = [
                codex_bin,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--skip-git-repo-check",
                "--cd",
                str(Path.cwd()),
                "--sandbox",
                "read-only",
                "--model",
                self.codex_model,
                "--output-last-message",
                output_file,
                "-",
            ]
            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=180,
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(stderr or f"codex exec failed with exit code {result.returncode}")
            if output_file and Path(output_file).exists():
                output_text = Path(output_file).read_text(encoding="utf-8").strip()
            else:
                output_text = (result.stdout or "").strip()
            if output_text:
                print(output_text)
            return output_text
        except Exception as e:
            print(f"\n{p.yellow}  [WARN] CODEX CLI error ({e}) -> fallback DeepSeek{p.reset}")
            return self.call_deepseek(system_prompt, user_message, label=f"{label} (via DeepSeek)")
        finally:
            if output_file and Path(output_file).exists():
                try:
                    Path(output_file).unlink()
                except OSError:
                    pass

    def call_claude(self, system_prompt: str, user_message: str, label: str = "") -> str:
        return self.call_codex(system_prompt, user_message, label=label)
