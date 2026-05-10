from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from .model_policy import ModelPolicy
from .reviewer import ResponseReviewer
from .router import TaskRouter


@dataclass(frozen=True)
class EvalCase:
    name: str
    user_input: str
    intent: str
    expected_provider: str
    expected_review_provider: str
    expected_risk: str
    response: str = ""


DEFAULT_EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        name="simple_chat",
        user_input="อธิบายแนวคิด data leakage แบบสั้น ๆ",
        intent="chat",
        expected_provider="deepseek",
        expected_review_provider="deepseek",
        expected_risk="low",
        response="อธิบายได้ค่ะ",
    ),
    EvalCase(
        name="pipeline_code",
        user_input="ช่วยแก้ script python เพื่อ clean csv และ train model",
        intent="pipeline",
        expected_provider="deepseek",
        expected_review_provider="deepseek",
        expected_risk="medium",
        response="<DISPATCH>{\"agent\":\"dana\",\"task\":\"clean data\"}</DISPATCH>",
    ),
    EvalCase(
        name="explicit_codex_request",
        user_input="ช่วยใช้ codex ตรวจ script นี้",
        intent="pipeline",
        expected_provider="deepseek",
        expected_review_provider="codex",
        expected_risk="low",
        response="ช่วยตรวจ code ให้หน่อย",
    ),
    EvalCase(
        name="freshness_request",
        user_input="ข้อมูลล่าสุดของ project นี้คืออะไร",
        intent="pipeline",
        expected_provider="deepseek",
        expected_review_provider="deepseek",
        expected_risk="low",
        response="ขอดูไฟล์ล่าสุดก่อนค่ะ",
    ),
    EvalCase(
        name="high_risk_reasoning",
        user_input="ช่วยประเมินความเสี่ยงทางกฎหมายของ pipeline นี้",
        intent="chat",
        expected_provider="deepseek",
        expected_review_provider="deepseek",
        expected_risk="high",
        response="ต้องตรวจสอบเพิ่ม",
    ),
)


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    route_provider: str
    review_provider: str
    risk: str
    review_passed: bool
    notes: str


def run_system_eval(cases: tuple[EvalCase, ...] = DEFAULT_EVAL_CASES) -> list[EvalResult]:
    router = TaskRouter()
    policy = ModelPolicy()
    reviewer = ResponseReviewer()
    results: list[EvalResult] = []

    for case in cases:
        route = router.route(case.user_input, intent=case.intent, context=case.response)
        choice = policy.choose("anna_primary", route)
        review = reviewer.review_anna(
            user_input=case.user_input,
            anna_response=case.response,
            intent=case.intent,
            dispatches=[],
            plan_issues=[],
            action_results="",
            route=route,
        )
        passed = (
            choice.provider == case.expected_provider
            and route.review_provider == case.expected_review_provider
            and route.risk == case.expected_risk
        )
        notes = route.reason
        if review.findings:
            notes += f" | review={review.severity}:{review.findings[0].message}"
        results.append(
            EvalResult(
                name=case.name,
                passed=passed,
                route_provider=choice.provider,
                review_provider=route.review_provider,
                risk=route.risk,
                review_passed=review.passed,
                notes=notes,
            )
        )
    return results


def build_eval_report(results: list[EvalResult]) -> str:
    passed = sum(1 for result in results if result.passed)
    lines = [
        "# Anna system evaluation",
        "",
        f"score: {passed}/{len(results)}",
        "",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"- {result.name}: {status} | provider={result.route_provider} | review={result.review_provider} | risk={result.risk} | {result.notes}"
        )
    return "\n".join(lines) + "\n"


def write_eval_report(path: str | Path, results: list[EvalResult]) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_eval_report(results), encoding="utf-8")
    return report_path


def run_and_write_default_eval(path: str | Path) -> Path:
    return write_eval_report(path, run_system_eval())
