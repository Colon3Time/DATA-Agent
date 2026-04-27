import tempfile
import unittest
from pathlib import Path

from anna_core.anna_contract import validate_dispatch_plan
from anna_core.config import load_config
from anna_core.dispatcher import DispatchParser
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
            cfg = load_config(Path(tmp), ["orchestrator_v3.py", "--mode", "full", "--claude-limit", "3", "--auto"])
            self.assertEqual(cfg.mode, "full")
            self.assertEqual(cfg.claude_limit, 3)
            self.assertFalse(cfg.step_mode)


if __name__ == "__main__":
    unittest.main()

