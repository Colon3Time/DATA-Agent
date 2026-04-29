from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable


RejectFn = Callable[[str], None]


class DispatchParser:
    """Parse Anna dispatch and ask-user tags."""

    def __init__(self, valid_agents: Iterable[str], on_reject: RejectFn | None = None) -> None:
        self.valid_agents = {agent.lower() for agent in valid_agents}
        self.on_reject = on_reject
        self.dispatch_re = re.compile(r"<DISPATCH>(.*?)</DISPATCH>", re.DOTALL)
        self.ask_user_re = re.compile(r"<ASK_USER>(.*?)</ASK_USER>", re.DOTALL)
        self.ask_codex_re = re.compile(r"<ASK_CODEX>(.*?)</ASK_CODEX>", re.DOTALL)

    def parse_dispatches(self, text: str) -> list[dict]:
        results: list[dict] = []
        for match in self.dispatch_re.finditer(text):
            raw = match.group(1).strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            parsed = self._parse_json_payload(raw)
            if parsed is None:
                continue
            agent = parsed.get("agent", "").lower().strip()
            task = parsed.get("task", "").strip()
            if agent in self.valid_agents and task and task not in ("...", ""):
                results.append(parsed)
            elif agent and self.on_reject:
                self.on_reject(agent)
        return results

    def parse_ask_user(self, text: str) -> str | None:
        match = self.ask_user_re.search(text)
        return match.group(1).strip() if match else None

    def parse_ask_codex(self, text: str) -> str | None:
        match = self.ask_codex_re.search(text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _parse_json_payload(raw: str) -> dict | None:
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        try:
            data = json.loads("{" + raw + "}")
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        inline = re.search(r"\{.*?\}", raw, re.DOTALL)
        if inline:
            try:
                data = json.loads(inline.group())
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        return None

