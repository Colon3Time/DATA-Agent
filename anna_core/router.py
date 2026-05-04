from __future__ import annotations

from dataclasses import dataclass, field


_HIGH_RISK_TERMS = {
    "medical",
    "medicine",
    "health",
    "legal",
    "law",
    "finance",
    "financial",
    "security",
    "privacy",
    "breach",
    "risk",
    "compliance",
}

_CODE_TERMS = {
    "code",
    "script",
    "python",
    "file",
    "path",
    "edit",
    "write",
    "fix",
    "bug",
    "error",
    "refactor",
    "test",
    "pipeline",
    "dispatch",
    "model",
    "train",
    "evaluate",
    "validation",
}

_RETRIEVAL_TERMS = {
    "memory",
    "remember",
    "previous",
    "prior",
    "session",
    "last time",
    "latest",
    "recent",
    "today",
    "current",
    "now",
    "again",
    "kb",
    "knowledge base",
}


@dataclass(frozen=True)
class TaskRoute:
    intent: str
    provider: str
    review_provider: str
    risk: str
    code_heavy: bool
    needs_retrieval: bool
    reason: str
    signals: tuple[str, ...] = field(default_factory=tuple)


class TaskRouter:
    """Conservative task router that chooses provider and review posture."""

    def route(self, text: str, *, intent: str = "pipeline", context: str = "") -> TaskRoute:
        text_l = text.lower()
        context_l = context.lower()
        combined = f"{text_l}\n{context_l}".strip()

        code_hits = [term for term in _CODE_TERMS if term in combined]
        risk_hits = [term for term in _HIGH_RISK_TERMS if term in combined]
        retrieval_hits = [term for term in _RETRIEVAL_TERMS if term in combined]

        code_heavy = len(code_hits) >= 2 or any(term in text_l for term in ("@", "<dispatch>", "<write_file>", "<run_shell>", "<run_python>"))
        needs_retrieval = bool(retrieval_hits) or any(term in text_l for term in ("latest", "recent", "current", "today", "remember"))

        if intent == "chat" and not code_heavy and not risk_hits:
            provider = "deepseek"
            risk = "low"
        elif risk_hits:
            provider = "codex"
            risk = "high" if len(risk_hits) < 3 else "critical"
        elif code_heavy or intent == "pipeline":
            provider = "codex" if code_heavy else "deepseek"
            risk = "medium" if code_heavy else "low"
        else:
            provider = "deepseek"
            risk = "low"

        review_provider = "codex" if risk in {"medium", "high", "critical"} or code_heavy else "deepseek"
        signals = tuple(sorted(set(code_hits + risk_hits + retrieval_hits)))
        reason_bits = [
            f"intent={intent}",
            f"risk={risk}",
            f"provider={provider}",
        ]
        if code_hits:
            reason_bits.append(f"code={','.join(sorted(set(code_hits))[:5])}")
        if retrieval_hits:
            reason_bits.append("retrieval=yes")
        if risk_hits:
            reason_bits.append(f"risk_terms={','.join(sorted(set(risk_hits))[:5])}")
        return TaskRoute(
            intent=intent,
            provider=provider,
            review_provider=review_provider,
            risk=risk,
            code_heavy=code_heavy,
            needs_retrieval=needs_retrieval,
            reason="; ".join(reason_bits),
            signals=signals,
        )

