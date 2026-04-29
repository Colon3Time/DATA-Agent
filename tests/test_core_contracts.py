import tempfile
import shutil
import unittest
from pathlib import Path

from anna_core.anna_contract import validate_dispatch_plan
from anna_core.config import load_config
from anna_core.dispatcher import DispatchParser
from anna_core.mo_phase import detect_mo_phase, mo_script_matches_phase, sync_mo_canonical_report
from anna_core.pipeline_store import PipelineStore
from anna_core.runner import is_shell_command_allowed


VALID_AGENTS = {"scout", "dana", "eddie", "max", "finn", "mo", "iris", "vera", "quinn", "rex"}


class DispatchParserTests(unittest.TestCase):
    def test_parses_valid_dispatch(self):
        parser = DispatchParser(VALID_AGENTS)
        text = '<DISPATCH>{"agent":"finn","task":"prepare features from dana output"}</DISPATCH>'
        self.assertEqual(
            parser.parse_dispatches(text),
            [{"agent": "finn", "task": "prepare features from dana output"}],
        )

    def test_rejects_unknown_agent(self):
        rejected = []
        parser = DispatchParser(VALID_AGENTS, on_reject=rejected.append)
        text = '<DISPATCH>{"agent":"unknown","task":"do work"}</DISPATCH>'
        self.assertEqual(parser.parse_dispatches(text), [])
        self.assertEqual(rejected, ["unknown"])

    def test_extracts_ask_user(self):
        parser = DispatchParser(VALID_AGENTS)
        self.assertEqual(parser.parse_ask_user("<ASK_USER>confirm?</ASK_USER>"), "confirm?")


class AnnaPlanValidationTests(unittest.TestCase):
    def test_requires_active_project(self):
        issues = validate_dispatch_plan(
            [{"agent": "dana", "task": "clean the selected dataset and save dana_output.csv"}],
            active_project=None,
            read_pipeline=lambda _agent: "",
        )
        self.assertIn("No active project was selected or created before DISPATCH.", issues)

    def test_blocks_mo_before_finn(self):
        issues = validate_dispatch_plan(
            [{"agent": "mo", "task": "train model using cleaned dataset and compare baseline algorithms"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertTrue(any("Mo is dispatched before Finn" in issue for issue in issues))

    def test_allows_mo_when_finn_exists(self):
        issues = validate_dispatch_plan(
            [{"agent": "mo", "task": "train model using finn_output.csv and compare algorithms"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda agent: "projects/demo/output/finn/finn_output.csv" if agent == "finn" else "",
        )
        self.assertFalse(issues)

    def test_blocks_eddie_before_dana(self):
        issues = validate_dispatch_plan(
            [{"agent": "eddie", "task": "run EDA from the current project dataset"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertTrue(any("Eddie is dispatched before Dana" in issue for issue in issues))

    def test_blocks_finn_before_dana_and_eddie(self):
        issues = validate_dispatch_plan(
            [{"agent": "finn", "task": "engineer final features from the current project dataset"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertTrue(any("Finn is dispatched before Dana" in issue for issue in issues))
        self.assertTrue(any("Finn is dispatched before Eddie" in issue for issue in issues))

    def test_allows_finn_when_existing_finn_output_already_exists(self):
        issues = validate_dispatch_plan(
            [{"agent": "finn", "task": "verify the existing engineered dataset"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda agent: "projects/demo/output/finn/engineered_data.csv" if agent == "finn" else "",
        )
        self.assertFalse(issues)

    def test_allows_ordered_dana_eddie_finn_mo_plan(self):
        issues = validate_dispatch_plan(
            [
                {"agent": "dana", "task": "clean scout_output.csv and save dana_output.csv"},
                {"agent": "eddie", "task": "run EDA on dana_output.csv and save eddie_output.csv"},
                {"agent": "finn", "task": "engineer final features and save finn_output.csv"},
                {"agent": "mo", "task": "train model using finn_output.csv and compare algorithms"},
            ],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertFalse(issues)


class PipelineStoreTests(unittest.TestCase):
    def test_rebuild_ignores_report_only_data_handoff_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "project"
            max_dir = project / "output" / "max"
            scout_dir = project / "output" / "scout"
            max_dir.mkdir(parents=True)
            scout_dir.mkdir(parents=True)
            (max_dir / "max_report.md").write_text("report only", encoding="utf-8")
            (scout_dir / "scout_output.csv").write_text("a,b\n1,2\n", encoding="utf-8")

            store = PipelineStore(base / "pipeline")
            store.rebuild_from_project(project)

            self.assertEqual(store.read("max"), "")
            self.assertTrue(store.read("scout").endswith("scout_output.csv"))


class MoPhaseGuardTests(unittest.TestCase):
    def test_detects_mo_phase_from_task(self):
        self.assertEqual(detect_mo_phase("Dispatch Mo Phase 2 Tune with RandomizedSearchCV"), 2)
        self.assertEqual(detect_mo_phase("Dispatch Mo Phase 3 Validate tuned model against default"), 3)

    def test_blocks_phase2_script_for_phase3_dispatch(self):
        tmp = Path("tmp") / "test_mo_phase_script"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            script = tmp / "mo_script.py"
            script.write_text("from sklearn.model_selection import RandomizedSearchCV\n# Phase 2 tuning\n", encoding="utf-8")
            self.assertTrue(mo_script_matches_phase(script, 2))
            self.assertFalse(mo_script_matches_phase(script, 3))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_syncs_mo_report_to_phase2_results(self):
        out = Path("tmp") / "test_mo_phase_report"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        try:
            (out / "mo_report.md").write_text("Mo Model Report - Phase 1: Explore", encoding="utf-8")
            (out / "model_results.md").write_text("Mo Model Report - Phase 2: Hyperparameter Tuning", encoding="utf-8")
            sync_mo_canonical_report(out, 2)
            self.assertIn("Phase 2", (out / "mo_report.md").read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(out, ignore_errors=True)


class RunnerPolicyTests(unittest.TestCase):
    def test_blocks_dangerous_shell(self):
        allowed, reason = is_shell_command_allowed("git reset --hard")
        self.assertFalse(allowed)
        self.assertIn("blocked", reason)

    def test_allows_readonly_shell(self):
        allowed, reason = is_shell_command_allowed("git status --short")
        self.assertTrue(allowed)
        self.assertEqual(reason, "")


class ConfigTests(unittest.TestCase):
    def test_load_config_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(
                Path(tmp),
                ["orchestrator_v3.py", "--mode", "full", "--claude-limit", "3", "--auto", "--no-color", "--no-title"],
            )
            self.assertEqual(cfg.mode, "full")
            self.assertEqual(cfg.claude_limit, 3)
            self.assertFalse(cfg.step_mode)
            self.assertTrue(cfg.no_color)
            self.assertFalse(cfg.terminal_title)


if __name__ == "__main__":
    unittest.main()
