import tempfile
import shutil
import contextlib
import io
import unittest
from pathlib import Path

from anna_core.anna_contract import validate_dispatch_plan
from anna_core.config import load_config
from anna_core.dispatcher import DispatchParser
from anna_core.agent_runtime import build_agent_path_message, should_regenerate_vera_script_for_meeting_report
from anna_core.mo_phase import detect_mo_phase, mo_script_matches_phase, sync_mo_canonical_report
from anna_core.pipeline_store import PipelineStore
from anna_core.runner import is_shell_command_allowed
from orchestrator_v3 import _print_gate_fail_recovery, _write_repair_note, resolve_agent_input, validate_agent_output


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

    def test_rejects_malformed_json(self):
        rejected = []
        parser = DispatchParser(VALID_AGENTS, on_reject=rejected.append)
        text = '<DISPATCH>{"agent":"scout","task":"find dataset",}</DISPATCH>'
        self.assertEqual(parser.parse_dispatches(text), [])
        self.assertEqual(rejected, ["<malformed>"])

    def test_rejects_payload_without_outer_json_object(self):
        rejected = []
        parser = DispatchParser(VALID_AGENTS, on_reject=rejected.append)
        text = '<DISPATCH>"agent":"scout","task":"find dataset"</DISPATCH>'
        self.assertEqual(parser.parse_dispatches(text), [])
        self.assertEqual(rejected, ["<malformed>"])

    def test_normalizes_valid_agent_name(self):
        parser = DispatchParser(VALID_AGENTS)
        text = '<DISPATCH>{"agent":"Scout","task":"find dataset and write dataset_profile.md"}</DISPATCH>'
        self.assertEqual(
            parser.parse_dispatches(text),
            [{"agent": "scout", "task": "find dataset and write dataset_profile.md"}],
        )

    def test_rejects_task_with_escaped_newline(self):
        parser = DispatchParser(VALID_AGENTS)
        text = '<DISPATCH>{"agent":"scout","task":"find marketing dataset\\nand save scout_output.csv"}</DISPATCH>'
        self.assertEqual(parser.parse_dispatches(text), [])

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

    def test_blocks_create_dir_after_dispatch(self):
        issues = validate_dispatch_plan(
            [{"agent": "scout", "task": "find marketing dataset and save scout_output.csv"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
            source_text='<DISPATCH>{"agent":"scout","task":"find marketing dataset and save scout_output.csv"}</DISPATCH>\n<CREATE_DIR path="projects/demo/input"/>',
        )
        self.assertTrue(any("CREATE_DIR appears after DISPATCH" in issue for issue in issues))

    def test_blocks_multiline_task_value(self):
        issues = validate_dispatch_plan(
            [{"agent": "scout", "task": "find marketing dataset\nsave scout_output.csv"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertTrue(any("single-line JSON string" in issue for issue in issues))

    def test_blocks_invalid_agent_in_validator(self):
        issues = validate_dispatch_plan(
            [{"agent": "anna", "task": "find marketing dataset and save scout_output.csv"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertTrue(any("invalid agent" in issue for issue in issues))

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


class ScoutGateTests(unittest.TestCase):
    def _write_scout_project(self, tmp: str, profile_text: str) -> tuple[Path, Path]:
        project = Path(tmp) / "project"
        scout_dir = project / "output" / "scout"
        scout_dir.mkdir(parents=True)
        csv_path = scout_dir / "scout_output.csv"
        csv_path.write_text("a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(25)) + "\n", encoding="utf-8")
        (scout_dir / "scout_report.md").write_text("DATASET_RISK_REGISTER\nrisk: reviewed\n", encoding="utf-8")
        (scout_dir / "dataset_profile.md").write_text(profile_text, encoding="utf-8")
        return project, csv_path

    def test_scout_gate_allows_clustering_without_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, csv_path = self._write_scout_project(
                tmp,
                "DATASET_PROFILE\nrows: 25\ncols: 2\ntarget_column: unknown\nproblem_type : clustering\n",
            )

            ok, reason = validate_agent_output("scout", str(csv_path), project)

            self.assertTrue(ok, reason)

    def test_scout_gate_blocks_supervised_without_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, csv_path = self._write_scout_project(
                tmp,
                "DATASET_PROFILE\nrows: 25\ncols: 2\ntarget_column: unknown\nproblem_type : classification\n",
            )

            ok, reason = validate_agent_output("scout", str(csv_path), project)

            self.assertFalse(ok)
            self.assertIn("target_column=unknown", reason)

    def test_missing_scout_profile_is_generated_from_shortlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            scout_dir = project / "output" / "scout"
            scout_dir.mkdir(parents=True)
            csv_path = scout_dir / "scout_sme_datasets.csv"
            csv_path.write_text(
                "name,source,url,license,combined_score\n"
                "World Bank Enterprise Surveys,World Bank,https://www.enterprisesurveys.org/en/data,CC BY,0.932\n"
                "SME Loan Default,Kaggle,https://www.kaggle.com/datasets?search=sme+loan,Kaggle Community,0.81\n",
                encoding="utf-8",
            )
            (scout_dir / "scout_report.md").write_text(
                "DATASET_RISK_REGISTER\nrisk: reviewed\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("scout", str(csv_path), project)

            self.assertFalse(ok)
            self.assertIn("มีแค่", reason)
            profile = scout_dir / "dataset_profile.md"
            self.assertTrue(profile.exists())
            text = profile.read_text(encoding="utf-8")
            self.assertIn("problem_type : dataset_discovery", text)
            self.assertIn("selected_candidate: World Bank Enterprise Surveys", text)

    def test_gate_fail_writes_repair_note_for_manual_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, csv_path = self._write_scout_project(
                tmp,
                "DATASET_PROFILE\nrows: 25\ncols: 2\ntarget_column: unknown\nproblem_type : classification\n",
            )
            ok, reason = validate_agent_output("scout", str(csv_path), project)
            self.assertFalse(ok)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("scout", reason, project)

            note = project / "logs" / "latest_repair.md"
            agent_note = project / "output" / "scout" / "REPAIR.md"
            self.assertTrue(note.exists())
            self.assertTrue(agent_note.exists())
            text = note.read_text(encoding="utf-8")
            self.assertIn("target_column=unknown", text)
            self.assertIn("/resume project", text)
            self.assertIn("dispatch Scout", text)


class ErrorRecoveryTests(unittest.TestCase):
    def _csv(self, path: Path, rows: int = 25) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(rows)) + "\n", encoding="utf-8")

    def _base_project(self, tmp: str) -> Path:
        project = Path(tmp) / "project"
        scout_dir = project / "output" / "scout"
        scout_dir.mkdir(parents=True)
        self._csv(scout_dir / "scout_output.csv")
        (scout_dir / "dataset_profile.md").write_text(
            "DATASET_PROFILE\nrows: 25\ncols: 2\ntarget_column: unknown\nproblem_type : clustering\n",
            encoding="utf-8",
        )
        (scout_dir / "scout_report.md").write_text("DATASET_RISK_REGISTER\nrisk: reviewed\n", encoding="utf-8")
        return project

    def test_missing_scout_risk_register_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            (project / "output" / "scout" / "scout_report.md").write_text("# report without required block\n", encoding="utf-8")
            scout_csv = project / "output" / "scout" / "scout_output.csv"

            ok, reason = validate_agent_output("scout", str(scout_csv), project)
            self.assertFalse(ok)
            self.assertIn("DATASET_RISK_REGISTER", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("scout", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("missing DATASET_RISK_REGISTER", text)
            self.assertIn("scout_report.md", text)

    def test_dana_gate_failure_records_upstream_and_resume_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            dana_dir = project / "output" / "dana"
            self._csv(dana_dir / "dana_output.csv")
            (dana_dir / "dana_report.md").write_text("# missing audit block\n", encoding="utf-8")

            ok, reason = validate_agent_output("dana", str(dana_dir / "dana_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("DATA_QUALITY_AUDIT", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("dana", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("upstream_expected: output\\scout\\scout_output.csv", text)
            self.assertIn("upstream_exists: True", text)
            self.assertIn("/resume project", text)

    def test_script_failure_repair_note_keeps_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            script = project / "output" / "dana" / "dana_script.py"
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("raise ValueError('bad data')\n", encoding="utf-8")

            note = _write_repair_note(
                "dana",
                "script",
                "script failed after auto-fix: dana_script.py",
                project,
                "แก้ dana_script.py แล้ว rerun @dana ด้วย input เดิม",
                output_path=str(project / "output" / "dana"),
                stderr="Traceback\nValueError: bad data",
            )

            self.assertIsNotNone(note)
            text = note.read_text(encoding="utf-8")
            self.assertIn("kind: script", text)
            self.assertIn("Traceback", text)
            self.assertIn("dana_script.py", text)
            self.assertIn("@dana", text)

    def test_gate_fail_print_falls_back_when_terminal_rejects_unicode(self):
        class RejectBoxDrawing(io.StringIO):
            def write(self, s):
                if any(ch in s for ch in "╔═╗║╠╝"):
                    raise UnicodeEncodeError("cp874", s, 0, 1, "character maps to <undefined>")
                return super().write(s)

        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            stream = RejectBoxDrawing()

            with contextlib.redirect_stdout(stream):
                _print_gate_fail_recovery("scout", "forced failure", project)

            self.assertIn("*** GATE FAIL - SCOUT ***", stream.getvalue())
            self.assertTrue((project / "logs" / "latest_repair.md").exists())

    def test_missing_previous_agent_output_fails_with_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)

            with self.assertRaisesRegex(RuntimeError, "EDDIE input missing"):
                with contextlib.redirect_stdout(io.StringIO()):
                    resolve_agent_input("eddie", "dana", project)

            note = project / "logs" / "latest_repair.md"
            self.assertTrue(note.exists())
            text = note.read_text(encoding="utf-8")
            self.assertIn("kind: missing-output", text)
            self.assertIn("dana_output.csv", text)
            self.assertIn("rerun DANA", text)
            self.assertIn("@eddie", text)

    def test_existing_previous_agent_output_resolves_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            dana_csv = project / "output" / "dana" / "dana_output.csv"
            self._csv(dana_csv)

            path = resolve_agent_input("eddie", "dana", project)

            self.assertEqual(Path(path), dana_csv)

    def test_too_small_output_csv_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            dana_dir = project / "output" / "dana"
            self._csv(dana_dir / "dana_output.csv", rows=3)
            (dana_dir / "dana_report.md").write_text("DATA_QUALITY_AUDIT\nok\n", encoding="utf-8")

            ok, reason = validate_agent_output("dana", str(dana_dir / "dana_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("มีแค่ 3 rows", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("dana", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("มีแค่ 3 rows", text)
            self.assertIn("dana_report.md", text)
            self.assertIn("rerun DANA", text)

    def test_dana_row_loss_incomplete_output_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            self._csv(project / "output" / "scout" / "scout_output.csv", rows=100)
            dana_dir = project / "output" / "dana"
            self._csv(dana_dir / "dana_output.csv", rows=50)
            (dana_dir / "dana_report.md").write_text("DATA_QUALITY_AUDIT\nok\n", encoding="utf-8")

            ok, reason = validate_agent_output("dana", str(dana_dir / "dana_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("rows หาย", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("dana", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("rows หาย", text)
            self.assertIn("upstream_expected: output\\scout\\scout_output.csv", text)

    def test_finn_output_missing_target_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            (project / "output" / "scout" / "dataset_profile.md").write_text(
                "DATASET_PROFILE\nrows: 25\ncols: 3\ntarget_column: label\nproblem_type : classification\n",
                encoding="utf-8",
            )
            eddie_dir = project / "output" / "eddie"
            eddie_dir.mkdir(parents=True, exist_ok=True)
            (eddie_dir / "eddie_output.csv").write_text("a,label\n1,0\n2,1\n", encoding="utf-8")
            finn_dir = project / "output" / "finn"
            self._csv(finn_dir / "finn_output.csv")
            (finn_dir / "finn_report.md").write_text("FEATURE_GOVERNANCE\nok\n", encoding="utf-8")

            ok, reason = validate_agent_output("finn", str(finn_dir / "finn_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("target 'label' หาย", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("finn", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("target 'label' หาย", text)
            self.assertIn("finn_report.md", text)
            self.assertIn("upstream_expected: output\\eddie\\eddie_output.csv", text)

    def test_mo_missing_finn_output_fails_with_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)

            with self.assertRaisesRegex(RuntimeError, "MO input missing"):
                with contextlib.redirect_stdout(io.StringIO()):
                    resolve_agent_input("mo", "finn", project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("kind: missing-output", text)
            self.assertIn("engineered_data.csv", text)
            self.assertIn("finn_output.csv", text)
            self.assertIn("rerun FINN", text)

    def test_mo_perfect_metric_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            mo_dir = project / "output" / "mo"
            mo_dir.mkdir(parents=True, exist_ok=True)
            (mo_dir / "mo_output.csv").write_text("model,accuracy\nxgb,1.0\n", encoding="utf-8")
            (mo_dir / "model_comparison.csv").write_text("model,accuracy,f1\nxgb,1.0,0.999\n", encoding="utf-8")
            (mo_dir / "model_results.md").write_text(
                "Model results\naccuracy: 1.0\nf1: 0.999\nleakage check pending\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("mo", str(mo_dir / "mo_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("perfect/near-perfect metric", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("mo", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("perfect/near-perfect metric", text)
            self.assertIn("model_results.md", text)
            self.assertIn("upstream_expected: output\\finn\\engineered_data.csv", text)

    def test_mo_report_with_na_metrics_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            mo_dir = project / "output" / "mo"
            mo_dir.mkdir(parents=True, exist_ok=True)
            (mo_dir / "mo_output.csv").write_text("model,accuracy\nbaseline,0.7\n", encoding="utf-8")
            (mo_dir / "model_results.md").write_text(
                "winner: logistic regression\ntest f1: N/A\ncv score: N/A\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("mo", str(mo_dir / "mo_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("N/A metrics", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("mo", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("N/A metrics", text)
            self.assertIn("model_results.md", text)

    def test_mo_output_without_report_or_metrics_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            mo_dir = project / "output" / "mo"
            mo_dir.mkdir(parents=True, exist_ok=True)
            (mo_dir / "mo_output.csv").write_text("row_id,prediction\n1,0\n2,1\n", encoding="utf-8")

            ok, reason = validate_agent_output("mo", str(mo_dir / "mo_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("missing Mo model report", reason)
            self.assertIn("missing readable model metrics", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("mo", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("missing Mo model report", text)
            self.assertIn("missing readable model metrics", text)

    def test_max_missing_pattern_report_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            max_dir = project / "output" / "max"
            self._csv(max_dir / "max_output.csv")

            ok, reason = validate_agent_output("max", str(max_dir / "max_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("missing pattern report", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("max", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("missing pattern report", text)
            self.assertIn("max_report.md", text)

    def test_max_report_missing_pattern_validity_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            max_dir = project / "output" / "max"
            self._csv(max_dir / "max_output.csv")
            (max_dir / "max_report.md").write_text("Association mining summary without required block\n", encoding="utf-8")

            ok, reason = validate_agent_output("max", str(max_dir / "max_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("PATTERN_VALIDITY", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("max", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("PATTERN_VALIDITY", text)
            self.assertIn("max_report.md", text)

    def test_iris_missing_report_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            iris_dir = project / "output" / "iris"
            self._csv(iris_dir / "iris_output.csv")

            ok, reason = validate_agent_output("iris", str(iris_dir / "iris_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("missing iris_report.md", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("iris", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("missing iris_report.md", text)
            self.assertIn("iris_report.md", text)

    def test_iris_report_missing_business_rigor_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            iris_dir = project / "output" / "iris"
            self._csv(iris_dir / "iris_output.csv")
            (iris_dir / "iris_report.md").write_text("Business notes only, no validation plan\n", encoding="utf-8")

            ok, reason = validate_agent_output("iris", str(iris_dir / "iris_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("missing business rigor", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("iris", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("missing business rigor", text)
            self.assertIn("iris_report.md", text)

    def test_quinn_missing_report_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            quinn_dir = project / "output" / "quinn"
            self._csv(quinn_dir / "quinn_output.csv")

            ok, reason = validate_agent_output("quinn", str(quinn_dir / "quinn_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("missing quinn_report.md", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("quinn", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("missing quinn_report.md", text)
            self.assertIn("quinn_report.md", text)

    def test_quinn_unsatisfied_verdict_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            quinn_dir = project / "output" / "quinn"
            self._csv(quinn_dir / "quinn_output.csv")
            (quinn_dir / "quinn_report.md").write_text(
                "verdict: unsatisfied\nleakage overfitting drift calibration business_satisfaction\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("quinn", str(quinn_dir / "quinn_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("requires restart", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("quinn", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("requires restart", text)
            self.assertIn("quinn_report.md", text)

    def test_vera_missing_report_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            vera_dir = project / "output" / "vera"
            self._csv(vera_dir / "vera_output.csv")

            ok, reason = validate_agent_output("vera", str(vera_dir / "vera_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("missing vera_report.md", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("vera", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("missing vera_report.md", text)
            self.assertIn("vera_report.md", text)

    def test_vera_charts_dir_without_png_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            vera_dir = project / "output" / "vera"
            self._csv(vera_dir / "vera_output.csv")
            (vera_dir / "vera_report.md").write_text("VISUAL_QC\nchart review complete\n", encoding="utf-8")
            (vera_dir / "charts").mkdir(parents=True)

            ok, reason = validate_agent_output("vera", str(vera_dir / "vera_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("no PNG charts", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("vera", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("no PNG charts", text)
            self.assertIn("vera_report.md", text)

    def test_rex_missing_final_report_gets_repair_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            rex_dir = project / "output" / "rex"
            self._csv(rex_dir / "rex_output.csv")

            ok, reason = validate_agent_output("rex", str(rex_dir / "rex_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("missing final executive report", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("rex", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("missing final executive report", text)
            self.assertIn("final_report.md", text)

    def test_rex_blocks_when_quinn_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._base_project(tmp)
            quinn_dir = project / "output" / "quinn"
            quinn_dir.mkdir(parents=True, exist_ok=True)
            (quinn_dir / "quinn_report.md").write_text("restart_cycle: yes\n", encoding="utf-8")
            rex_dir = project / "output" / "rex"
            self._csv(rex_dir / "rex_output.csv")
            (rex_dir / "final_report.md").write_text(
                "assumptions cost roi production readiness monitoring retrain time-based validation limitation\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("rex", str(rex_dir / "rex_output.csv"), project)
            self.assertFalse(ok)
            self.assertIn("Quinn failed", reason)

            with contextlib.redirect_stdout(io.StringIO()):
                _print_gate_fail_recovery("rex", reason, project)

            text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
            self.assertIn("Quinn failed", text)
            self.assertIn("final_report.md", text)


class VeraMeetingReportTests(unittest.TestCase):
    def test_vera_prompt_includes_meeting_report_visual_brief(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            rex_dir = project / "output" / "rex"
            vera_dir = project / "output" / "vera"
            rex_dir.mkdir(parents=True)
            report = rex_dir / "meeting_presentation.md"
            report.write_text("# Meeting\nROI and risk storyline", encoding="utf-8")

            msg = build_agent_path_message(
                "vera",
                "สร้างกราฟให้ตรงกับ meeting report",
                str(project / "output" / "finn" / "finn_output.csv"),
                vera_dir,
                project,
            )

            self.assertIn(str(report), msg)
            self.assertIn("Build the chart plan from the meeting/executive report storyline first", msg)
            self.assertIn("do not create decorative or column-driven charts", msg)

    def test_vera_regenerates_stale_script_when_meeting_report_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            rex_dir = project / "output" / "rex"
            vera_dir = project / "output" / "vera"
            rex_dir.mkdir(parents=True)
            vera_dir.mkdir(parents=True)
            script = vera_dir / "vera_script.py"
            script.write_text("print('old chart script')\n", encoding="utf-8")
            (rex_dir / "meeting_presentation.md").write_text("# New meeting report\n", encoding="utf-8")

            self.assertTrue(
                should_regenerate_vera_script_for_meeting_report(
                    script,
                    "ปรับกราฟให้เข้ากับ meeting report",
                    project,
                )
            )


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
