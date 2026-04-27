from __future__ import annotations

import json
import os
from dataclasses import dataclass

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
    """DeepSeek/Claude client with Claude usage tracked in shared state."""

    def __init__(
        self,
        *,
        state: OrchestratorState,
        deepseek_url: str,
        deepseek_model: str,
        claude_model: str,
        claude_limit: int,
        palette: TerminalPalette,
    ) -> None:
        self.state = state
        self.deepseek_url = deepseek_url
        self.deepseek_model = deepseek_model
        self.claude_model = claude_model
        self.claude_limit = claude_limit
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
            print(f"{p.red}  ✗ DEEPSEEK_API_KEY not found in .env{p.reset}")
            return "[ERROR] DEEPSEEK_API_KEY not found in .env"
        if label:
            bar = "─" * max(0, 46 - len(label))
            print(f"\n{p.blue}┌─ {p.bold}{label}{p.reset}{p.blue} {bar}┐{p.reset}")
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
                print(f"{p.red}  ✗ DeepSeek HTTP {response.status_code}: {body}{p.reset}")
                return f"[ERROR] DeepSeek HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            print(f"{p.red}  ✗ DeepSeek connection failed{p.reset}")
            return "[ERROR] DeepSeek connection failed"
        except requests.exceptions.Timeout:
            print(f"{p.red}  ✗ DeepSeek timeout{p.reset}")
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

    def call_claude(self, system_prompt: str, user_message: str, label: str = "") -> str:
        """Try Claude first, then fall back to DeepSeek when unavailable."""
        p = self.palette
        if self.state.claude_calls >= self.claude_limit:
            print(
                f"\n{p.yellow}  ⚠ Claude limit ถึง {self.state.claude_calls}/{self.claude_limit} calls แล้ว "
                f"→ ใช้ DeepSeek แทน{p.reset}"
            )
            return self.call_deepseek(system_prompt, user_message, label=f"{label} (via DeepSeek[limit])")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic as _ant

                self.state.claude_calls += 1
                remaining = self.claude_limit - self.state.claude_calls
                print(f"\n{p.magenta}{'━'*55}{p.reset}")
                print(
                    f"{p.magenta}  ✦ CLAUDE  {p.bold}{label}{p.reset}  "
                    f"{p.dim}[{self.state.claude_calls}/{self.claude_limit} — เหลือ {remaining}]{p.reset}"
                )
                print(f"{p.magenta}{'━'*55}{p.reset}")
                client = _ant.Anthropic(api_key=api_key)
                with client.messages.stream(
                    model=self.claude_model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                ) as stream:
                    full: list[str] = []
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                        full.append(text)
                print()
                return "".join(full)
            except Exception as e:
                self.state.claude_calls -= 1
                msg = str(e)
                if "credit" in msg.lower():
                    print(f"\n{p.red}  ✗ CLAUDE credit หมด{p.reset} {p.yellow}→ fallback DeepSeek{p.reset}")
                else:
                    print(f"\n{p.yellow}  ⚠ CLAUDE error ({e}) → fallback DeepSeek{p.reset}")
        else:
            print(f"\n{p.yellow}  ⚠ ไม่พบ ANTHROPIC_API_KEY → ใช้ DeepSeek แทน{p.reset}")

        return self.call_deepseek(system_prompt, user_message, label=f"{label} (via DeepSeek)")

