from __future__ import annotations

from dataclasses import dataclass, field

from .router import TaskRoute


@dataclass(frozen=True)
class ReviewFinding:
    severity: str
    message: str
    suggestion: str = ""


@dataclass(frozen=True)
class ReviewResult:
    passed: bool
    severity: str
    findings: tuple[ReviewFinding, ...] = field(default_factory=tuple)
    repair_provider: str = "deepseek"
    repair_hint: str = ""

    @property
    def needs_repair(self) -> bool:
        return not self.passed


class ResponseReviewer:
    """Deterministic critic that flags weak plans before they reach execution."""

    def review_anna(
        self,
        *,
        user_input: str,
        anna_response: str,
        intent: str,
        dispatches: list[dict],
        plan_issues: list[str] | None = None,
        action_results: str = "",
        route: TaskRoute | None = None,
    ) -> ReviewResult:
        route = route or TaskRoute(
            intent=intent,
            provider="deepseek",
            review_provider="deepseek",
            risk="low",
            code_heavy=False,
            needs_retrieval=False,
            reason="fallback route",
        )
        findings: list[ReviewFinding] = []
        plan_issues = plan_issues or []
        text = anna_response.lower()
        input_l = user_input.lower()

        if plan_issues:
            findings.append(
                ReviewFinding(
                    "high",
                    f"dispatch plan failed validation: {plan_issues[0]}",
                    "repair the dispatch JSON and keep CREATE_DIR before DISPATCH when the project is new",
                )
            )

        if intent == "pipeline" and "<dispatch>" in text and not dispatches:
            findings.append(
                ReviewFinding(
                    "high",
                    "dispatch tag exists but no valid dispatch parsed",
                    "repair the response before execution",
                )
            )

        if intent == "pipeline" and not dispatches and "<ask_user>" not in text and not action_results:
            findings.append(
                ReviewFinding(
                    "medium",
                    "pipeline request has no dispatch or user clarification",
                    "either dispatch a valid plan or ask the user for missing context",
                )
            )

        if any(term in input_l for term in ("latest", "current", "recent", "today")) and not any(
            tag in text for tag in ("<research>", "<ask_deepseek>", "<read_file>", "<run_shell>")
        ):
            findings.append(
                ReviewFinding(
                    "low",
                    "freshness-sensitive request did not explicitly retrieve new evidence",
                    "load relevant files or ask the retrieval model before finalizing",
                )
            )

        if action_results and "ERROR" in action_results:
            findings.append(
                ReviewFinding(
                    "medium",
                    "action execution produced an error",
                    "repair the action output before returning it as final",
                )
            )

        if any(token in text for token in ("todo", "fixme", "placeholder")):
            findings.append(
                ReviewFinding(
                    "high",
                    "response still contains stub markers",
                    "rewrite the response without placeholder language",
                )
            )

        severity_order = {"low": 0, "medium": 1, "high": 2}
        severity = max((finding.severity for finding in findings), key=lambda s: severity_order.get(s, 0), default="low")
        passed = not any(finding.severity in {"medium", "high"} for finding in findings)
        repair_provider = "codex" if route.risk in {"medium", "high", "critical"} or route.code_heavy or severity in {"medium", "high"} else "deepseek"
        repair_hint = findings[0].suggestion if findings else ""
        return ReviewResult(
            passed=passed,
            severity=severity,
            findings=tuple(findings),
            repair_provider=repair_provider,
            repair_hint=repair_hint,
        )
