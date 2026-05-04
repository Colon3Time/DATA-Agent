from __future__ import annotations

from dataclasses import dataclass

from .router import TaskRoute


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    reason: str


class ModelPolicy:
    """Map task routes to concrete provider choices."""

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
            return ModelChoice("codex", "discovery benefits from stronger planning and code synthesis")

        if stage in {"anna_repair", "anna_review"}:
            provider = "codex" if route.risk in {"medium", "high", "critical"} or route.code_heavy else "deepseek"
            return ModelChoice(provider, f"repair/review follows route risk={route.risk}")

        if stage in {"anna_summary"}:
            provider = "codex" if route.code_heavy or route.risk in {"high", "critical"} or route.needs_retrieval else "deepseek"
            return ModelChoice(provider, f"summary follows route risk={route.risk}")

        if stage in {"anna_primary"}:
            provider = route.provider
            if route.needs_retrieval and provider == "deepseek" and route.risk in {"medium", "high", "critical"}:
                provider = "codex"
            return ModelChoice(provider, route.reason)

        return ModelChoice(fallback, f"fallback provider={fallback}")

