from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable


RejectFn = Callable[[str], None]


class DispatchParser:
    """Parse Anna dispatch and ask-user tags."""

    REQUIRED_FIELDS = {"agent", "task"}
    OPTIONAL_FIELDS = {"discover", "parallel_group"}
    ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
    VAGUE_TASKS = {"...", "analyze this", "ทำต่อ", "วิเคราะห์", "analyze", "continue"}
    MIN_TASK_LENGTH = 12

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
            if "\n" in raw or "\r" in raw:
                if self.on_reject:
                    self.on_reject("<malformed>")
                continue
            parsed = self._parse_json_payload(raw)
            if parsed is None:
                if self.on_reject:
                    self.on_reject("<malformed>")
                continue
            agent_value = parsed.get("agent", "")
            task_value = parsed.get("task", "")
            agent = agent_value.lower().strip() if isinstance(agent_value, str) else ""
            task = task_value.strip() if isinstance(task_value, str) else ""
            if self._payload_matches_contract(parsed, agent, task):
                parsed["agent"] = agent
                parsed["task"] = task
                results.append(parsed)
            elif agent and self.on_reject:
                self.on_reject(agent)
            elif self.on_reject:
                self.on_reject("<malformed>")
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
            return None

    def _payload_matches_contract(self, parsed: dict, agent: str, task: str) -> bool:
        if set(parsed) - self.ALLOWED_FIELDS:
            return False
        if not self.REQUIRED_FIELDS.issubset(parsed):
            return False
        if agent not in self.valid_agents:
            return False
        if len(task) < self.MIN_TASK_LENGTH or task.lower() in self.VAGUE_TASKS or "\n" in task or "\r" in task:
            return False
        if "discover" in parsed and not isinstance(parsed["discover"], bool):
            return False
        if "parallel_group" in parsed:
            parallel_group = parsed["parallel_group"]
            if not isinstance(parallel_group, str) or not parallel_group.strip() or "\n" in parallel_group or "\r" in parallel_group:
                return False
            parsed["parallel_group"] = parallel_group.strip()
        return True

