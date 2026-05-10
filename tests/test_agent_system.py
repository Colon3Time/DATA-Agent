import shutil
import unittest
from pathlib import Path

from anna_core.evaluation import build_eval_report, run_system_eval
from anna_core.logging import SessionMemoryStore
from anna_core.model_policy import ModelPolicy
from anna_core.reviewer import ResponseReviewer
from anna_core.router import TaskRouter


class TaskRouterTests(unittest.TestCase):
    def test_routes_code_heavy_pipeline_to_deepseek_first(self):
        router = TaskRouter()
        route = router.route("ช่วยแก้ script python เพื่อ clean csv และ train model", intent="pipeline")
        self.assertEqual(route.provider, "deepseek")
        self.assertEqual(route.review_provider, "deepseek")
        self.assertEqual(route.risk, "medium")

    def test_routes_explicit_codex_request_to_codex(self):
        router = TaskRouter()
        route = router.route("ช่วยใช้ codex ตรวจ script นี้", intent="pipeline")
        self.assertEqual(route.provider, "codex")
        self.assertEqual(route.review_provider, "codex")

    def test_routes_simple_chat_to_deepseek(self):
        router = TaskRouter()
        route = router.route("อธิบาย data leakage แบบสั้น ๆ", intent="chat")
        self.assertEqual(route.provider, "deepseek")
        self.assertEqual(route.risk, "low")


class ModelPolicyTests(unittest.TestCase):
    def test_summary_uses_route_provider(self):
        router = TaskRouter()
        policy = ModelPolicy()
        route = router.route("train model from csv and compare metrics", intent="pipeline")
        choice = policy.choose("anna_summary", route)
        self.assertEqual(choice.provider, "deepseek")

    def test_execute_stays_on_deepseek(self):
        policy = ModelPolicy()
        choice = policy.choose("agent_execute")
        self.assertEqual(choice.provider, "deepseek")


class ReviewerTests(unittest.TestCase):
    def test_detects_missing_dispatch_for_pipeline_request(self):
        reviewer = ResponseReviewer()
        route = TaskRouter().route("ช่วย train model จากข้อมูลนี้", intent="pipeline")
        review = reviewer.review_anna(
            user_input="ช่วย train model จากข้อมูลนี้",
            anna_response="ฉันจะทำให้ค่ะ แต่ยังไม่ส่ง dispatch",
            intent="pipeline",
            dispatches=[],
            plan_issues=[],
            action_results="",
            route=route,
        )
        self.assertFalse(review.passed)
        self.assertEqual(review.severity, "medium")


class SessionMemoryTests(unittest.TestCase):
    def test_build_context_uses_relevant_entries(self):
        knowledge_dir = Path.cwd() / "tmp" / "agent_system_test_memory"
        shutil.rmtree(knowledge_dir, ignore_errors=True)
        self.addCleanup(lambda: shutil.rmtree(knowledge_dir, ignore_errors=True))
        store = SessionMemoryStore(knowledge_dir)
        store.save("project-a", ["scout"], "Scout discovered a fresh ecommerce dataset and selected review_score.")
        store.save("project-b", ["mo"], "Mo found leakage in the training loop and needed a reviewer pass.")

        context = store.build_context("leakage reviewer pass")
        self.assertIn("leakage", context.lower())
        self.assertIn("Mo found leakage", context)


class EvaluationTests(unittest.TestCase):
    def test_default_eval_runs(self):
        results = run_system_eval()
        self.assertTrue(results)
        report = build_eval_report(results)
        self.assertIn("Anna system evaluation", report)
