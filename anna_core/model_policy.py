from __future__ import annotations

from dataclasses import dataclass

from .router import TaskRoute


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    reason: str


class ModelPolicy:
    """Map task routes to concrete provider choices."""

    def __init__(self, *, codex_enabled: bool = False) -> None:
        self.codex_enabled = codex_enabled

    def _provider(self, provider: str) -> str:
        if provider == "codex" and not self.codex_enabled:
            return "deepseek"
        return provider

    def choose(self, stage: str, route: TaskRoute | None = None, *, fallback: str = "deepseek") -> ModelChoice:
        route = route or TaskRoute(
            intent="chat",
            provider=fallback,
            review_provider="deepseek",
            risk="low",
            code_heavy=False,
            needs_retrieval=False,
            reason="fallback route",
        )

        if stage in {"agent_execute"}:
            return ModelChoice("deepseek", "agent execution stays on the fast executor model")

        if stage in {"agent_discover"}:
            provider = self._provider(route.provider)
            return ModelChoice(provider, f"discovery follows route provider={provider}")

        if stage in {"anna_repair", "anna_review"}:
            provider = self._provider(route.provider)
            return ModelChoice(provider, f"repair/review follows route risk={route.risk}")

        if stage in {"anna_summary"}:
            provider = self._provider(route.provider)
            return ModelChoice(provider, f"summary follows route risk={route.risk}")

        if stage in {"anna_primary"}:
            return ModelChoice(self._provider(route.provider), route.reason)

        return ModelChoice(self._provider(fallback), f"fallback provider={fallback}")
