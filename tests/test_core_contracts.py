import tempfile
import shutil
import contextlib
import io
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

from anna_core.anna_contract import validate_dispatch_plan
from anna_core.action_executor import ActionExecutor, ActionPalette
from anna_core.actions import WorkspacePaths
from anna_core.config import load_config
from anna_core.dispatcher import DispatchParser
from anna_core.agent_runtime import build_agent_path_message, extract_python_blocks, should_regenerate_vera_script_for_meeting_report
from anna_core.mo_phase import detect_mo_phase, mo_script_matches_phase, sync_mo_canonical_report
from anna_core.pipeline_store import PipelineStore
from anna_core.run_guard import begin_project_run, mark_output_current, output_is_current, promote_pipeline_outputs
from anna_core.runner import builtin_agent_script, is_shell_command_allowed, scout_output_is_placeholder
from core.schema_contract import validate_handoff
import orchestrator_v3 as orch
from orchestrator_v3 import (
    STATE,
    _print_gate_fail_recovery,
    ensure_vera_chart_artifact,
    _write_repair_note,
    handle_cli_command,
    normalize_scout_output_for_handoff,
    resolve_agent_input,
    validate_agent_output,
)


VALID_AGENTS = {"scout", "dana", "eddie", "iris_eda", "max", "finn", "mo", "iris", "vera", "quinn", "rex"}


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

    def test_rejects_multiline_dispatch_payload(self):
        parser = DispatchParser(VALID_AGENTS)
        text = '<DISPATCH>{\n"agent":"scout","task":"find marketing dataset and save profile"\n}</DISPATCH>'
        self.assertEqual(parser.parse_dispatches(text), [])

    def test_rejects_short_placeholder_task(self):
        parser = DispatchParser(VALID_AGENTS)
        text = '<DISPATCH>{"agent":"dana","task":"clean data"}</DISPATCH>'
        self.assertEqual(parser.parse_dispatches(text), [])

    def test_rejects_unsupported_dispatch_field(self):
        parser = DispatchParser(VALID_AGENTS)
        text = '<DISPATCH>{"agent":"scout","task":"find dataset and write profile","priority":"high"}</DISPATCH>'
        self.assertEqual(parser.parse_dispatches(text), [])

    def test_rejects_non_boolean_discover(self):
        parser = DispatchParser(VALID_AGENTS)
        text = '<DISPATCH>{"agent":"scout","task":"find dataset and write profile","discover":"true"}</DISPATCH>'
        self.assertEqual(parser.parse_dispatches(text), [])

    def test_rejects_vague_thai_task(self):
        parser = DispatchParser(VALID_AGENTS)
        text = '<DISPATCH>{"agent":"dana","task":"ทำต่อ"}</DISPATCH>'
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

    def test_blocks_unsupported_dispatch_field_in_validator(self):
        issues = validate_dispatch_plan(
            [{"agent": "scout", "task": "find marketing dataset and save scout_output.csv", "priority": "high"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertTrue(any("unsupported field" in issue for issue in issues))

    def test_blocks_invalid_optional_field_types_in_validator(self):
        issues = validate_dispatch_plan(
            [{"agent": "scout", "task": "find marketing dataset and save scout_output.csv", "discover": "true"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertTrue(any("discover" in issue and "boolean" in issue for issue in issues))

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

    def test_allows_eddie_with_explicit_input_path_without_dana(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "scout_output.csv"
            csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

            issues = validate_dispatch_plan(
                [{"agent": "eddie", "task": f"run EDA using {csv_path} and save eddie_output.csv"}],
                active_project=Path("projects/demo"),
                read_pipeline=lambda _agent: "",
            )

            self.assertFalse(any("Eddie is dispatched before Dana" in issue for issue in issues), issues)

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

    def test_guided_mode_blocks_multi_agent_plan(self):
        issues = validate_dispatch_plan(
            [
                {"agent": "dana", "task": "clean scout_output.csv and save dana_output.csv"},
                {"agent": "eddie", "task": "run EDA on dana_output.csv and save eddie_output.csv"},
            ],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
            mode="guided",
        )
        self.assertTrue(any("Guided mode allows only one DISPATCH" in issue for issue in issues))

    def test_guided_mode_allows_single_next_agent(self):
        issues = validate_dispatch_plan(
            [{"agent": "dana", "task": "clean scout_output.csv and save dana_output.csv"}],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
            mode="guided",
        )
        self.assertFalse(issues)

    def test_allows_iris_eda_between_eddie_and_finn(self):
        issues = validate_dispatch_plan(
            [
                {"agent": "dana", "task": "clean scout_output.csv and save dana_output.csv"},
                {"agent": "eddie", "task": "run EDA on dana_output.csv and save eddie_output.csv"},
                {"agent": "iris_eda", "task": "write a business insight bridge from eddie_output.csv"},
                {"agent": "finn", "task": "engineer final features and save finn_output.csv"},
            ],
            active_project=Path("projects/demo"),
            read_pipeline=lambda _agent: "",
        )
        self.assertFalse(issues)

    def test_dana_to_eddie_accepts_gaid_style_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            output_dir = project / "output" / "dana"
            output_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "Year": [1998 + (i % 3) for i in range(120)],
                    "Country": ["Algeria"] * 120,
                    "ISO3": ["DZA"] * 120,
                    "Metric": ["Example"] * 120,
                    "Value": [float(i % 7) for i in range(120)],
                    "Dataset": ["GAID"] * 120,
                    "Source": ["Source"] * 120,
                    "Source_Category": ["Category"] * 120,
                    "Source_File": ["file.xlsx"] * 120,
                    "Source_Type": ["xlsx"] * 120,
                    "Source_Year": [2021] * 120,
                    "is_outlier": [0] * 119 + [1],
                }
            )
            csv_path = output_dir / "dana_output.csv"
            df.to_csv(csv_path, index=False)

            ok, errors = validate_handoff("dana", "eddie", csv_path, project)

            self.assertTrue(ok, errors)

    def test_run_agent_falls_back_to_builtin_when_llm_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            dana_dir = project / "output" / "dana"
            dana_dir.mkdir(parents=True, exist_ok=True)
            eddie_dir = project / "output" / "eddie"
            eddie_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(
                {
                    "Year": [1998 + (i % 3) for i in range(120)],
                    "Country": ["Algeria"] * 120,
                    "ISO3": ["DZA"] * 120,
                    "Metric": ["Example"] * 120,
                    "Value": [float(i % 7) for i in range(120)],
                    "Dataset": ["GAID"] * 120,
                    "Source": ["Source"] * 120,
                    "Source_Category": ["Category"] * 120,
                    "Source_File": ["file.xlsx"] * 120,
                    "Source_Type": ["xlsx"] * 120,
                    "Source_Year": [2021] * 120,
                    "is_outlier": [0] * 119 + [1],
                }
            )
            (dana_dir / "dana_output.csv").write_text(df.to_csv(index=False), encoding="utf-8")

            previous_project = orch.STATE.active_project
            previous_counts = dict(orch.STATE.agent_iter_count)
            try:
                orch.STATE.active_project = project
                begin_project_run(project, reason="unit-test")
                mark_output_current(dana_dir / "dana_output.csv", project)

                with mock.patch.object(orch, "call_agent_llm", return_value="[ERROR] DEEPSEEK_API_KEY not found in .env"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        out_path = orch.run_agent(
                            "eddie",
                            "EDA for GAID Master — Input: projects/test/output/dana/dana_output.csv",
                            prev_agent="dana",
                            project_dir=project,
                        )

                self.assertTrue((eddie_dir / "eddie_output.csv").exists())
                self.assertEqual(Path(out_path).name, "eddie_output.csv")
                columns = pd.read_csv(eddie_dir / "eddie_output.csv").columns
                self.assertIn("Value", columns)
                self.assertNotIn("revenue", columns)
            finally:
                orch.STATE.active_project = previous_project
                orch.STATE.agent_iter_count.clear()
                orch.STATE.agent_iter_count.update(previous_counts)


class ActionExecutorRoutingTests(unittest.TestCase):
    def _executor(self, tmp: Path, calls: list[tuple[str, str]]) -> ActionExecutor:
        return ActionExecutor(
            base_dir=tmp,
            projects_dir=tmp / "projects",
            workspace_paths=WorkspacePaths(tmp),
            log=lambda *_args: None,
            save_kb=lambda *_args: None,
            ask_deepseek=lambda _system, prompt, _label: calls.append(("deepseek", prompt)) or "deepseek-answer",
            ask_claude=lambda _system, prompt, _label: calls.append(("claude", prompt)) or "claude-answer",
            set_active_project=lambda _project: None,
            palette=ActionPalette("", "", "", "", "", "", "", ""),
        )

    def test_ask_codex_is_not_executed_as_claude_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calls: list[tuple[str, str]] = []
            executor = self._executor(Path(tmpdir), calls)

            with contextlib.redirect_stdout(io.StringIO()):
                result = executor.execute("<ASK_CODEX>แก้ dispatch validation</ASK_CODEX>")

            self.assertEqual(calls, [])
            self.assertEqual(result, "")

    def test_ask_claude_still_routes_to_claude(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calls: list[tuple[str, str]] = []
            executor = self._executor(Path(tmpdir), calls)

            with contextlib.redirect_stdout(io.StringIO()):
                result = executor.execute("<ASK_CLAUDE>ช่วยคิดวิธีแก้</ASK_CLAUDE>")

            self.assertEqual(calls, [("claude", "ช่วยคิดวิธีแก้")])
            self.assertIn("[ASK_CLAUDE]", result)


class AgentRuntimeTests(unittest.TestCase):
    def test_extract_python_blocks_accepts_common_fence_variants(self):
        text = "``` Python\r\nprint('a')\r\n```\n```py\nprint('b')\n```"
        blocks = extract_python_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertIn("print('a')", blocks[0])
        self.assertIn("print('b')", blocks[1])

    def test_generated_script_fallback_keeps_existing_report_when_no_code_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output" / "scout"
            output_dir.mkdir(parents=True)
            report = output_dir / "scout_report.md"
            report.write_text("report only\n", encoding="utf-8")

            out = orch.run_generated_script_agent(
                "scout",
                "report only task",
                "",
                output_dir,
                report,
                [],
                None,
            )

            self.assertTrue(report.exists())
            self.assertEqual(Path(out), report)

    def test_llm_response_with_script_timeout_argument_is_available(self):
        response = "```python\nimport requests\nrequests.get('https://example.com', timeout=10)\n```"
        self.assertFalse(orch.llm_response_is_unavailable(response))

    def test_llm_error_response_is_unavailable(self):
        self.assertTrue(orch.llm_response_is_unavailable("[ERROR] DeepSeek timeout"))

    def test_scout_generated_script_handoff_prefers_scout_output_over_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            input_dir = project / "input"
            output_dir = project / "output" / "scout"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            (input_dir / "raw.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            report = output_dir / "scout_report.md"
            report.write_text("DATASET_RISK_REGISTER\n", encoding="utf-8")
            code = """
import argparse, os
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()
os.makedirs(args.output_dir, exist_ok=True)
rows = []
for i in range(120):
    rows.append({
        'Country': 'Thailand',
        'Year': 2000 + i,
        'Indicator': 'GDP',
        'Value': i,
    })
df = pd.DataFrame(rows)
output_csv = os.path.join(args.output_dir, 'scout_output.csv')
df.to_csv(output_csv, index=False)
profile = 'DATASET_PROFILE\\n===============\\nrows         : 120\\ncols         : 4\\ntarget_column: Value\\nproblem_type : regression\\nDATASET_RISK_REGISTER\\n'
with open(os.path.join(args.output_dir, 'dataset_profile.md'), 'w', encoding='utf-8') as fh:
    fh.write(profile)
with open(os.path.join(args.output_dir, 'scout_report.md'), 'w', encoding='utf-8') as fh:
    fh.write('DATASET_RISK_REGISTER\\n')
print(f'[STATUS] Saved: {output_csv}')
""" + ("\n# filler for script validator" * 60)

            out = orch.run_generated_script_agent(
                "scout",
                "create scout output",
                "",
                output_dir,
                report,
                [code],
                project,
            )

            self.assertEqual(Path(out), output_dir / "scout_output.csv")

    def test_scout_output_normalization_converts_wide_economic_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            output_dir = project / "output" / "scout"
            output_dir.mkdir(parents=True)
            (output_dir / "scout_report.md").write_text("Scout report without risk\n", encoding="utf-8")
            wide = pd.DataFrame({
                "Country": ["Thailand"] * 15,
                "Year": list(range(2010, 2025)),
                "GDP": list(range(15)),
                "Inflation": list(range(15)),
            })
            csv_path = output_dir / "scout_output.csv"
            wide.to_csv(csv_path, index=False)

            out = normalize_scout_output_for_handoff(project, output_dir)

            normalized = pd.read_csv(out)
            self.assertEqual(set(normalized.columns), {"Country", "Year", "Indicator", "Value"})
            self.assertEqual(len(normalized), 30)
            self.assertTrue((output_dir / "dataset_profile.md").exists())
            report = (output_dir / "scout_report.md").read_text(encoding="utf-8")
            self.assertIn("DATASET_RISK_REGISTER", report)

    def test_scout_existing_script_path_normalizes_before_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            input_dir = project / "input"
            output_dir = project / "output" / "scout"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            source = input_dir / "wide.csv"
            pd.DataFrame({
                "Country": ["Thailand"] * 15,
                "Year": list(range(2010, 2025)),
                "GDP": list(range(15)),
                "Inflation": list(range(15)),
            }).to_csv(source, index=False)
            script = output_dir / "scout_script.py"
            code = """
import argparse, os
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()
os.makedirs(args.output_dir, exist_ok=True)
df = pd.read_csv(args.input)
df.to_csv(os.path.join(args.output_dir, 'scout_output.csv'), index=False)
open(os.path.join(args.output_dir, 'scout_report.md'), 'w', encoding='utf-8').write('DATASET_RISK_REGISTER\\n')
""" + ("\n# filler for script validator" * 60)
            script.write_text(code, encoding="utf-8")

            out = orch.run_existing_script_agent("scout", "run existing scout", script, str(source), output_dir, project_dir=project)

            normalized = pd.read_csv(out)
            self.assertEqual(set(normalized.columns), {"Country", "Year", "Indicator", "Value"})
            self.assertEqual(len(normalized), 30)

    def test_run_script_does_not_overwrite_scout_output_with_input_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            input_dir = project / "input"
            output_dir = project / "output" / "scout"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            source = input_dir / "wide.csv"
            pd.DataFrame({
                "Country": ["Thailand"] * 15,
                "Year": list(range(2010, 2025)),
                "GDP": list(range(15)),
                "Inflation": list(range(15)),
            }).to_csv(source, index=False)
            script = output_dir / "scout_script.py"
            code = """
import argparse, os
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()
os.makedirs(args.output_dir, exist_ok=True)
df = pd.read_csv(args.input)
records = []
for _, row in df.iterrows():
    for indicator in ['GDP', 'Inflation']:
        records.append({'Country': row['Country'], 'Year': row['Year'], 'Indicator': indicator, 'Value': row[indicator]})
pd.DataFrame(records).to_csv(os.path.join(args.output_dir, 'scout_output.csv'), index=False)
open(os.path.join(args.output_dir, 'scout_report.md'), 'w', encoding='utf-8').write('DATASET_RISK_REGISTER\\n')
""" + ("\n# filler for script validator" * 60)
            script.write_text(code, encoding="utf-8")

            out, rc, stderr = orch.run_script(script, str(source), output_dir)

            self.assertEqual(rc, 0, stderr)
            normalized = pd.read_csv(out)
            self.assertEqual(set(normalized.columns), {"Country", "Year", "Indicator", "Value"})
            self.assertEqual(len(normalized), 30)

    def test_scout_placeholder_heuristic_allows_small_real_long_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "scout_output.csv"
            pd.DataFrame({
                "Country": ["Thailand"] * 30,
                "Year": list(range(1990, 2020)),
                "Indicator": ["GDP"] * 30,
                "Value": list(range(30)),
            }).to_csv(csv_path, index=False)

            self.assertFalse(scout_output_is_placeholder(csv_path))

    def test_scout_report_only_handoff_fails_without_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            scout_dir = project / "output" / "scout"
            scout_dir.mkdir(parents=True)
            report = scout_dir / "scout_report.md"
            report.write_text("Scout report\n\nDATASET_RISK_REGISTER\n" + ("details\n" * 10), encoding="utf-8")

            ok, msg = validate_agent_output("scout", str(report), project)

            self.assertFalse(ok)
            self.assertIn("report-only", msg)

    def test_vera_fallback_chart_creates_png_from_numeric_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            input_csv = project / "output" / "finn" / "finn_output.csv"
            vera_dir = project / "output" / "vera"
            input_csv.parent.mkdir(parents=True)
            vera_dir.mkdir(parents=True)
            (vera_dir / "vera_report.md").write_text("VISUAL_QC\ncharts: none\n", encoding="utf-8")
            pd.DataFrame({"feature": [1.0, 2.5, 3.0], "label": [0, 1, 1]}).to_csv(input_csv, index=False)

            chart = ensure_vera_chart_artifact(vera_dir, str(input_csv))

            self.assertTrue(Path(chart).exists())
            self.assertEqual(Path(chart).suffix.lower(), ".png")

    def test_generated_script_syntax_error_reaches_autofix_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output" / "scout"
            output_dir.mkdir(parents=True)
            script = output_dir / "bad_script.py"
            script.write_text("def broken(:\n", encoding="utf-8")

            out, rc, stderr = orch.run_script(script, "", output_dir)

            self.assertEqual(rc, 127)
            self.assertTrue(script.exists())
            self.assertIn("unusable script", stderr)

    def test_repair_note_uses_plan_as_fallback_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            note = _write_repair_note("scout", "gate", "bad scout output", project, "rerun SCOUT with real dataset")
            text = note.read_text(encoding="utf-8")
            self.assertIn("- task: rerun SCOUT with real dataset", text)


class LLMClientTests(unittest.TestCase):
    def test_deepseek_stream_ignores_stdout_oserror(self):
        from anna_core.llm import LLMClient, TerminalPalette
        from anna_core.state import OrchestratorState

        class FakeResponse:
            status_code = 200

            def iter_lines(self):
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}'
                yield b"data: [DONE]"

        def bad_print(*_args, **_kwargs):
            raise OSError(22, "Invalid argument")

        client = LLMClient(
            state=OrchestratorState(),
            deepseek_url="https://example.invalid",
            deepseek_model="deepseek-chat",
            palette=TerminalPalette("", "", "", "", "", "", ""),
        )

        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "x"}):
            with mock.patch("requests.post", return_value=FakeResponse()):
                with mock.patch("builtins.print", side_effect=bad_print):
                    self.assertEqual(client.call_deepseek("system", "user"), "hello")


class PipelineStoreTests(unittest.TestCase):
    def test_completed_agents_ignore_directory_handoff_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "project"
            finn_dir = project / "output" / "finn"
            finn_dir.mkdir(parents=True)

            store = PipelineStore(base / "pipeline")
            store.project_dir = project
            store.write("finn", str(finn_dir))

            self.assertNotIn("finn", store.completed_agents())

    def test_orchestrator_pipeline_read_ignores_directory_handoff_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "project"
            finn_dir = project / "output" / "finn"
            finn_dir.mkdir(parents=True)

            original_pipeline = orch.PIPELINE
            original_project = orch.STATE.active_project
            try:
                orch.PIPELINE = PipelineStore(base / "pipeline")
                orch.PIPELINE.project_dir = project
                orch.STATE.active_project = project
                orch.PIPELINE.write("finn", str(finn_dir))

                self.assertEqual(orch.pipeline_read("finn"), "")
            finally:
                orch.PIPELINE = original_pipeline
                orch.STATE.active_project = original_project

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

    def test_completed_agents_ignore_stale_outputs_after_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "project"
            dana_dir = project / "output" / "dana"
            dana_dir.mkdir(parents=True)
            dana_csv = dana_dir / "dana_output.csv"
            dana_csv.write_text("a,b\n1,2\n", encoding="utf-8")

            store = PipelineStore(base / "pipeline")
            store.rebuild_from_project(project)
            self.assertIn("dana", store.completed_agents())

            begin_project_run(project, reason="fresh-run")
            self.assertNotIn("dana", store.completed_agents())


class RunGuardTests(unittest.TestCase):
    def test_resolve_agent_input_prefers_explicit_task_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir(parents=True)
            explicit = Path(tmp) / "scout_output.csv"
            explicit.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

            resolved = resolve_agent_input(
                "eddie",
                "",
                project,
                task=f"run EDA using {explicit} and write eddie_output.csv",
            )

            self.assertEqual(Path(resolved), explicit)

    def test_resolve_agent_input_rejects_stale_upstream_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            dana_dir = project / "output" / "dana"
            dana_dir.mkdir(parents=True)
            dana_csv = dana_dir / "dana_output.csv"
            dana_csv.write_text("a,b\n1,2\n", encoding="utf-8")

            begin_project_run(project, reason="first")
            mark_output_current(dana_csv, project)
            begin_project_run(project, reason="second")

            with self.assertRaisesRegex(RuntimeError, "STALE output detected"):
                resolve_agent_input("eddie", "dana", project, task="run EDA on dana_output.csv")

    def test_promote_pipeline_outputs_marks_resumed_files_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            finn_dir = project / "output" / "finn"
            finn_dir.mkdir(parents=True)
            finn_csv = finn_dir / "finn_output.csv"
            finn_csv.write_text("a,b\n1,2\n", encoding="utf-8")

            begin_project_run(project, reason="first")
            mark_output_current(finn_csv, project)
            self.assertTrue(output_is_current(finn_csv, project))

            begin_project_run(project, reason="resume")
            self.assertFalse(output_is_current(finn_csv, project))

            promote_pipeline_outputs(project, [finn_csv])

            self.assertTrue(output_is_current(finn_csv, project))


class IrisEDABridgeTests(unittest.TestCase):
    def test_resolve_agent_input_uses_eddie_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            eddie_dir = project / "output" / "eddie"
            eddie_dir.mkdir(parents=True)
            eddie_csv = eddie_dir / "eddie_output.csv"
            eddie_csv.write_text(
                "a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(25)) + "\n",
                encoding="utf-8",
            )

            begin_project_run(project, reason="first")
            mark_output_current(eddie_csv, project)

            with contextlib.redirect_stdout(io.StringIO()):
                resolved = resolve_agent_input("iris_eda", "eddie", project, task="bridge analysis from Eddie")

            self.assertEqual(Path(resolved), eddie_csv)

    def test_iris_eda_gate_requires_bridge_brief(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            out_dir = project / "output" / "iris_eda"
            out_dir.mkdir(parents=True)
            csv_path = out_dir / "iris_eda_output.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            report = out_dir / "iris_eda_report.md"
            report.write_text(
                "BUSINESS_EDA_BRIEF\n=================\n"
                "Insight: retention is weaker in one cohort\n"
                "Evidence: eddie exploration and summary stats\n"
                "Follow-up: send to finn\n"
                "Risk: correlation only\n"
                "Confidence: Medium\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("iris_eda", str(csv_path), project)

            self.assertTrue(ok, reason)


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


class RepairCommandTests(unittest.TestCase):
    def test_repair_note_records_task_for_auto_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir(parents=True, exist_ok=True)

            note = _write_repair_note(
                "dana",
                "gate",
                "missing audit block",
                project,
                "rerun SCOUT แล้วค่อย rerun DANA",
                task="clean scout output and write DATA_QUALITY_AUDIT",
            )

            self.assertIsNotNone(note)
            text = note.read_text(encoding="utf-8")
            self.assertIn("- task: clean scout output and write DATA_QUALITY_AUDIT", text)

    def test_repair_auto_reruns_latest_note(self):
        import orchestrator_v3 as orch

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir(parents=True, exist_ok=True)
            _write_repair_note(
                "dana",
                "gate",
                "missing audit block",
                project,
                "rerun SCOUT แล้วค่อย rerun DANA",
                task="clean scout output and write DATA_QUALITY_AUDIT",
            )

            original_project = STATE.active_project
            original_run_agent = orch.run_agent
            original_validate = orch.validate_agent_output
            calls: list[tuple[str, str, Path | None]] = []
            try:
                STATE.active_project = project

                def fake_run_agent(agent_name, task, prev_agent="", project_dir=None, discover=False):
                    calls.append((agent_name, task, project_dir))
                    out_dir = project / "output" / agent_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out = out_dir / f"{agent_name}_output.csv"
                    out.write_text("a,b\n1,2\n", encoding="utf-8")
                    return str(out)

                orch.run_agent = fake_run_agent
                orch.validate_agent_output = lambda agent, out, proj: (True, "")

                with contextlib.redirect_stdout(io.StringIO()):
                    result = handle_cli_command("/repair auto")

                self.assertEqual(result, "handled")
                self.assertEqual([call[0] for call in calls], ["scout", "dana"])
                self.assertIn("clean scout output", calls[1][1])
                self.assertEqual(calls[1][2], project)
            finally:
                orch.run_agent = original_run_agent
                orch.validate_agent_output = original_validate
                STATE.active_project = original_project

    def test_repair_auto_reruns_missing_upstream_before_downstream(self):
        import orchestrator_v3 as orch

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir(parents=True, exist_ok=True)
            note = _write_repair_note(
                "dana",
                "missing-output",
                f"DANA input missing: expected upstream output {project / 'output' / 'scout' / 'scout_output.csv'}",
                project,
                "rerun SCOUT before rerun DANA",
                task="clean Scout output and write dana_output.csv",
            )
            with note.open("a", encoding="utf-8") as fh:
                fh.write("\n- upstream_expected: output\\scout\\scout_output.csv\n")

            original_project = STATE.active_project
            original_run_agent = orch.run_agent
            original_validate = orch.validate_agent_output
            calls: list[tuple[str, str, Path | None]] = []
            try:
                STATE.active_project = project

                def fake_run_agent(agent_name, task, prev_agent="", project_dir=None, discover=False):
                    calls.append((agent_name, task, project_dir))
                    out_dir = project / "output" / agent_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out = out_dir / f"{agent_name}_output.csv"
                    out.write_text("a,b\n1,2\n", encoding="utf-8")
                    return str(out)

                orch.run_agent = fake_run_agent
                orch.validate_agent_output = lambda agent, out, proj: (True, "")

                with contextlib.redirect_stdout(io.StringIO()):
                    result = handle_cli_command("/repair auto")

                self.assertEqual(result, "handled")
                self.assertEqual([call[0] for call in calls], ["scout", "dana"])
                self.assertIn("Repair upstream output for DANA", calls[0][1])
                self.assertIn("clean Scout output", calls[1][1])
            finally:
                orch.run_agent = original_run_agent
                orch.validate_agent_output = original_validate
                STATE.active_project = original_project

    def test_repair_auto_stops_when_upstream_does_not_create_required_file(self):
        import orchestrator_v3 as orch

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir(parents=True, exist_ok=True)
            note = _write_repair_note(
                "dana",
                "missing-output",
                f"DANA input missing: expected upstream output {project / 'output' / 'scout' / 'scout_output.csv'}",
                project,
                "rerun SCOUT before rerun DANA",
                task="clean Scout output and write dana_output.csv",
            )
            with note.open("a", encoding="utf-8") as fh:
                fh.write("\n- upstream_expected: output\\scout\\scout_output.csv\n")

            original_project = STATE.active_project
            original_run_agent = orch.run_agent
            original_validate = orch.validate_agent_output
            calls: list[str] = []
            try:
                STATE.active_project = project

                def fake_run_agent(agent_name, task, prev_agent="", project_dir=None, discover=False):
                    calls.append(agent_name)
                    out_dir = project / "output" / agent_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    return str(out_dir)

                orch.run_agent = fake_run_agent
                orch.validate_agent_output = lambda agent, out, proj: (True, "")

                with contextlib.redirect_stdout(io.StringIO()):
                    result = handle_cli_command("/repair auto")

                self.assertEqual(result, "handled")
                self.assertEqual(calls, ["scout"])
                text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
                self.assertIn("- agent: scout", text)
                self.assertIn("- task: Repair upstream output for DANA", text)
                self.assertIn("did not create required file", text)
            finally:
                orch.run_agent = original_run_agent
                orch.validate_agent_output = original_validate
                STATE.active_project = original_project

    def test_repair_auto_checks_required_file_for_primary_agent(self):
        import orchestrator_v3 as orch

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir(parents=True, exist_ok=True)
            note = _write_repair_note(
                "scout",
                "missing-output",
                "Scout did not create scout_output.csv",
                project,
                "rerun SCOUT and ensure required output exists",
                task="write scout_output.csv",
            )
            with note.open("a", encoding="utf-8") as fh:
                fh.write("\n- upstream_expected: output\\scout\\scout_output.csv\n")

            original_project = STATE.active_project
            original_run_agent = orch.run_agent
            original_validate = orch.validate_agent_output
            try:
                STATE.active_project = project

                def fake_run_agent(agent_name, task, prev_agent="", project_dir=None, discover=False):
                    out_dir = project / "output" / agent_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    return str(out_dir)

                orch.run_agent = fake_run_agent
                orch.validate_agent_output = lambda agent, out, proj: (True, "")

                with contextlib.redirect_stdout(io.StringIO()):
                    result = handle_cli_command("/repair auto")

                self.assertEqual(result, "handled")
                text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
                self.assertIn("- agent: scout", text)
                self.assertIn("- task: write scout_output.csv", text)
                self.assertIn("repair did not create required file", text)
                task_line = next(line for line in text.splitlines() if line.startswith("- task:"))
                self.assertNotIn("\n", task_line)
            finally:
                orch.run_agent = original_run_agent
                orch.validate_agent_output = original_validate
                STATE.active_project = original_project

    def test_repair_auto_reads_required_file_from_problem_text(self):
        import orchestrator_v3 as orch

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir(parents=True, exist_ok=True)
            required = project / "output" / "scout" / "scout_output.csv"
            _write_repair_note(
                "scout",
                "missing-output",
                f"repair did not create required file: {required}",
                project,
                "rerun SCOUT and ensure required output exists",
                task="write scout_output.csv",
            )

            original_project = STATE.active_project
            original_run_agent = orch.run_agent
            original_validate = orch.validate_agent_output
            try:
                STATE.active_project = project

                def fake_run_agent(agent_name, task, prev_agent="", project_dir=None, discover=False):
                    out_dir = project / "output" / agent_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    return str(out_dir)

                orch.run_agent = fake_run_agent
                orch.validate_agent_output = lambda agent, out, proj: (True, "")

                with contextlib.redirect_stdout(io.StringIO()):
                    result = handle_cli_command("/repair auto")

                self.assertEqual(result, "handled")
                text = (project / "logs" / "latest_repair.md").read_text(encoding="utf-8")
                self.assertIn("repair did not create required file", text)
                self.assertFalse(required.exists())
            finally:
                orch.run_agent = original_run_agent
                orch.validate_agent_output = original_validate
                STATE.active_project = original_project


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


class BuiltinTemplateContractTests(unittest.TestCase):
    def test_finn_builtin_keeps_generic_target_and_avoids_rfm_storyline(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            eddie_dir = project / "output" / "eddie"
            scout_dir = project / "output" / "scout"
            finn_dir = project / "output" / "finn"
            eddie_dir.mkdir(parents=True)
            scout_dir.mkdir(parents=True)
            finn_dir.mkdir(parents=True)
            scout_dir.joinpath("dataset_profile.md").write_text(
                "target_column: Source_Year\nproblem_type : classification\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "Year": [2020, 2021, 2022, 2023] * 30,
                    "Country": ["A", "B", "C", "D"] * 30,
                    "Metric": ["m1", "m2"] * 60,
                    "Value": [float(i) for i in range(120)],
                    "Source_Year": [2021, 2024, 2024, 2026] * 30,
                }
            ).to_csv(eddie_dir / "eddie_output.csv", index=False)

            script = finn_dir / "finn_script.py"
            script.write_text(builtin_agent_script("finn"), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(eddie_dir / "eddie_output.csv"),
                    "--output-dir",
                    str(finn_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            out = pd.read_csv(finn_dir / "finn_output.csv", nrows=5)
            self.assertIn("Source_Year", out.columns)
            self.assertNotIn("monetary", out.columns)
            self.assertNotIn("recency_days", out.columns)
            report = (finn_dir / "finn_report.md").read_text(encoding="utf-8")
            self.assertIn("feature_mode: generic supervised feature table", report)

    def test_mo_gate_rejects_target_mismatch_fallback_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            scout_dir = project / "output" / "scout"
            mo_dir = project / "output" / "mo"
            scout_dir.mkdir(parents=True)
            mo_dir.mkdir(parents=True)
            scout_dir.joinpath("dataset_profile.md").write_text(
                "target_column: Source_Year\nproblem_type : classification\n",
                encoding="utf-8",
            )
            (mo_dir / "mo_output.csv").write_text("prediction,actual\n1,1\n", encoding="utf-8")
            (mo_dir / "model_comparison.csv").write_text("model,rmse,mae,r2\ndummy_mean,0,0,0\n", encoding="utf-8")
            (mo_dir / "mo_report.md").write_text(
                "MO_REPORT\nPRODUCTION_READINESS\ntarget_column: monetary\nwinner model: dummy_mean\nrmse: 0\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("mo", str(mo_dir / "mo_output.csv"), project)

            self.assertFalse(ok)
            self.assertIn("target_column mismatch", reason)
            self.assertIn("fallback/dummy", reason)

    def test_run_marker_does_not_create_nested_marker_names(self):
        from anna_core.run_guard import output_run_marker

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            out_dir = project / "output" / "dana"
            out_dir.mkdir(parents=True)
            csv_path = out_dir / "dana_output.csv"
            csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
            begin_project_run(project, reason="unit-test")
            mark_output_current(csv_path, project)
            marker = output_run_marker(csv_path)
            mark_output_current(marker, project)

            self.assertTrue(marker.exists())
            self.assertFalse((out_dir / "dana_output.csv.run_id.run_id").exists())


class QuinnRestartDownstreamTests(unittest.TestCase):
    def test_quinn_restart_does_not_block_failure_aware_downstream_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            q_dir = project / "output" / "quinn"
            q_dir.mkdir(parents=True)
            q_dir.joinpath("quinn_report.md").write_text(
                "WORLD_CLASS_QC\nRESTART_CYCLE: YES\nRestart From: Finn\n",
                encoding="utf-8",
            )

            self.assertTrue(orch.enforce_quinn_gate("rex", project))

    def test_rex_can_pass_with_failure_aware_report_after_quinn_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            q_dir = project / "output" / "quinn"
            r_dir = project / "output" / "rex"
            q_dir.mkdir(parents=True)
            r_dir.mkdir(parents=True)
            q_dir.joinpath("quinn_report.md").write_text(
                "WORLD_CLASS_QC\nRESTART_CYCLE: YES\nVerdict: Unsatisfied\n",
                encoding="utf-8",
            )
            r_dir.joinpath("final_report.md").write_text(
                "Executive issue report\n"
                "restart_cycle acknowledged; pipeline blocked; not production-ready\n"
                "Business impact assumptions: cost and ROI are not claimed.\n"
                "Production readiness: prototype only, monitoring and retrain required.\n"
                "Validation limitations: time-based / OOT validation limitation remains.\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("rex", str(r_dir / "final_report.md"), project)

            self.assertTrue(ok, reason)

    def test_rex_blocks_success_claim_after_quinn_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            q_dir = project / "output" / "quinn"
            r_dir = project / "output" / "rex"
            q_dir.mkdir(parents=True)
            r_dir.mkdir(parents=True)
            q_dir.joinpath("quinn_report.md").write_text(
                "WORLD_CLASS_QC\nRESTART_CYCLE: YES\nVerdict: Unsatisfied\n",
                encoding="utf-8",
            )
            r_dir.joinpath("final_report.md").write_text(
                "Cycle complete and production-ready.\n"
                "Business impact assumptions: cost.\n"
                "Production readiness: monitoring.\n"
                "Validation limitations: time-based validation limitation.\n",
                encoding="utf-8",
            )

            ok, reason = validate_agent_output("rex", str(r_dir / "final_report.md"), project)

            self.assertFalse(ok)
            self.assertIn("failure-aware", reason)


class ConfigTests(unittest.TestCase):
    def test_default_mode_is_guided(self):
        cfg = load_config(
            Path("tmp"),
            ["orchestrator_v3.py", "--no-color", "--no-title"],
        )
        self.assertEqual(cfg.mode, "guided")
        self.assertFalse(cfg.codex_enabled)

    def test_load_config_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(
                Path(tmp),
                ["orchestrator_v3.py", "--mode", "full", "--claude-limit", "3", "--auto", "--enable-codex", "--no-color", "--no-title"],
            )
            self.assertEqual(cfg.mode, "full")
            self.assertEqual(cfg.claude_limit, 3)
            self.assertTrue(cfg.codex_enabled)
            self.assertFalse(cfg.step_mode)
            self.assertTrue(cfg.no_color)
            self.assertFalse(cfg.terminal_title)


if __name__ == "__main__":
    unittest.main()
