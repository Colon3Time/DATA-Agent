from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .run_guard import begin_project_run, promote_upstream_outputs


@dataclass(frozen=True)
class CommandPalette:
    reset: str
    bold: str
    dim: str
    cyan: str
    green: str
    yellow: str
    red: str
    blue: str
    magenta: str
    white: str


@dataclass
class AppCommands:
    state: object
    cli: object
    pipeline: object
    projects_dir: Path
    palette: CommandPalette
    set_tab_title: Callable[[str], None]
    notify_tab: Callable[[bool, str], None]
    run_pipeline: Callable[[str], None]
    run_agent: Callable[..., str]
    validate_agent_output: Callable[[str, str, Path | None], tuple[bool, str]]
    print_gate_recovery: Callable[[str, str, Path | None], None]
    write_repair_note: Callable[..., Path | None]
    delete_old_scripts: Callable[[Path], None]
    output_dir_for: Callable[[Path | None, str], Path]
    quinn_hard_gate: Callable[[Path], tuple[bool, str, str]]
    agent_output_newer_than: Callable[[Path, str, Path], bool]
    load_kb: Callable[[str], str]
    load_relevant_kb: Callable[[str, str, int], str]
    save_kb: Callable[[str, str, str], None]
    call_deepseek: Callable[[str, str, str], str]
    call_claude: Callable[[str, str, str], str]
    codex_enabled: bool
    anna_system: str
    get_last_pipeline_project: Callable[[], Path | None]
    set_last_pipeline_project: Callable[[Path | None], None]
    pipeline_clear: Callable[[], None]

    def print_help(self) -> None:
        p = self.palette
        print(
            f"\n{p.cyan}Slash commands:{p.reset}\n"
            f"  {p.bold}/project <name>{p.reset}       เลือกโปรเจกต์  (alias: /p, /proj)\n"
            f"  {p.bold}/resume [name] [task]{p.reset} ทำต่อจากโปรเจกต์/active project  (alias: /r)\n"
            f"  {p.bold}/repair [auto]{p.reset}        เปิด note ล่าสุด หรือสั่ง auto repair จาก note\n"
            f"  {p.bold}/status{p.reset}               ดูสถานะ pipeline  (alias: /s)\n"
            f"  {p.bold}/kb <agent>{p.reset}           ดู knowledge base\n"
            f"  {p.bold}/claude{p.reset}               ดู Codex usage\n"
            f"  {p.bold}/eval{p.reset}                 รัน routing / reviewer smoke test\n"
            f"  {p.bold}/end{p.reset}                  reset session\n"
            f"  {p.bold}/exit{p.reset}                 ออกจากระบบ\n"
        )
        self.cli.print_help()

    def print_latest_repair_note(self) -> None:
        p = self.palette
        project = self.state.active_project
        if not project:
            print(f"{p.red}  ยังไม่ได้เลือก project{p.reset}")
            print("  ใช้: /project <name>")
            return
        note = project / "logs" / "latest_repair.md"
        if not note.exists():
            print(f"{p.yellow}  ยังไม่มี repair note สำหรับ {project.name}{p.reset}")
            return
        text = note.read_text(encoding="utf-8", errors="ignore")
        kind = re.search(r"^- kind:\s*(.+)$", text, re.MULTILINE)
        agent = re.search(r"^- agent:\s*(.+)$", text, re.MULTILINE)
        problem = re.search(r"^- problem:\s*(.+)$", text, re.MULTILINE)
        plan = re.search(r"^- plan:\s*(.+)$", text, re.MULTILINE)
        task = re.search(r"^- task:\s*(.+)$", text, re.MULTILINE)

        print(f"\n{p.cyan}Repair note:{p.reset} {note}")
        if any((kind, agent, problem, plan)):
            print(f"{p.yellow}  สรุป:{p.reset}")
            if kind:
                print(f"  kind   : {kind.group(1).strip()}")
            if agent:
                print(f"  agent  : {agent.group(1).strip()}")
            if problem:
                print(f"  problem: {problem.group(1).strip()}")
            if task:
                print(f"  task   : {task.group(1).strip()}")
            if plan:
                print(f"  plan   : {plan.group(1).strip()}")
        print(f"\n{text[-4000:]}")

    def _parse_repair_note_fields(self, text: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for line in text.splitlines():
            match = re.match(r"^- ([^:]+):\s*(.*)$", line)
            if match:
                fields[match.group(1).strip().lower()] = match.group(2).strip()
        return fields

    def _load_latest_repair_note(self, project_dir: Path | None) -> tuple[Path, str, dict[str, str]] | None:
        if not project_dir:
            return None
        note = project_dir / "logs" / "latest_repair.md"
        if not note.exists():
            return None
        text = note.read_text(encoding="utf-8", errors="ignore")
        return note, text, self._parse_repair_note_fields(text)

    def _infer_repair_upstream_agent(self, fields: dict[str, str], downstream_agent: str) -> str:
        kind = fields.get("kind", "").strip().lower()
        haystacks = [fields.get("upstream_expected", "")]
        if kind in {"missing-output", "stale-output", "schema-mismatch"}:
            haystacks.extend([
                fields.get("problem", ""),
                fields.get("input", ""),
            ])
        for text in haystacks:
            match = re.search(r"output[\\/]+([a-z_]+)[\\/]+[a-z_]+_output\.csv", text, re.IGNORECASE)
            if match:
                upstream = match.group(1).lower()
                if upstream != downstream_agent:
                    return upstream

        if kind not in {"missing-output", "stale-output", "schema-mismatch"}:
            return ""

        plan = fields.get("plan", "")
        match = re.search(r"rerun\s+@?([a-z_]+)", plan, re.IGNORECASE)
        if match:
            upstream = match.group(1).lower()
            if upstream != downstream_agent:
                    return upstream
        return ""

    def _repair_expected_upstream_path(self, fields: dict[str, str], project: Path) -> Path | None:
        expected = fields.get("upstream_expected", "").strip()
        if not expected:
            problem = fields.get("problem", "")
            match = re.search(r"(?:required file|expected upstream output):\s*(.+?)(?:\.\s|$)", problem, re.IGNORECASE)
            if match:
                expected = match.group(1).strip()
        if not expected:
            return None
        expected_path = Path(expected)
        if expected_path.is_absolute():
            return expected_path
        return project / expected_path

    def _repair_note_task(self, task: str) -> str:
        return re.sub(r"\s+", " ", task).strip()

    def _run_repair_agent_once(self, agent: str, task: str, project: Path) -> tuple[bool, str]:
        p = self.palette
        try:
            out = self.run_agent(agent, task, project_dir=project)
            ok, msg = self.validate_agent_output(agent, out, project)
        except RuntimeError as exc:
            msg = str(exc)
            if agent in ("scout", "dana", "eddie", "finn", "mo"):
                self.print_gate_recovery(agent, msg, project)
            else:
                print(f"{p.yellow}  {agent.upper()} repair stopped: {msg}{p.reset}")
            return False, msg

        if ok:
            return True, ""
        if agent in ("scout", "dana", "eddie", "finn", "mo"):
            self.print_gate_recovery(agent, msg, project)
        else:
            print(f"{p.yellow}  ⚠ {agent.upper()} ยังไม่ผ่าน validation: {msg}{p.reset}")
        return False, msg

    def _run_latest_repair_note(self) -> None:
        p = self.palette
        project = self.state.active_project
        if not project:
            print(f"{p.red}  ยังไม่ได้เลือก project{p.reset}")
            print("  ใช้: /project <name>")
            return

        loaded = self._load_latest_repair_note(project)
        if not loaded:
            print(f"{p.yellow}  ยังไม่มี repair note สำหรับ {project.name}{p.reset}")
            return

        note, _text, fields = loaded
        agent = fields.get("agent", "").strip().lower()
        task = fields.get("task", "").strip()
        kind = fields.get("kind", "").strip().lower()
        plan = fields.get("plan", "").strip()

        if not agent:
            print(f"{p.red}  repair note ไม่มี agent ที่จะ rerun ได้{p.reset}")
            print(f"  ใช้ /repair เพื่อดูรายละเอียด: {note}")
            return

        if not task or task == "(unknown)":
            print(f"{p.red}  repair note ยังไม่มี task สำหรับ auto repair ของ {agent.upper()}{p.reset}")
            print(f"  ใช้ /repair เพื่อดู note ล่าสุด: {note}")
            return

        print(f"\n{p.cyan}Auto repair:{p.reset} {p.bold}{agent.upper()}{p.reset}")
        if kind:
            print(f"  kind : {kind}")
        if plan:
            print(f"  plan : {plan}")
        print(f"  note : {note}")

        rerun_task = task
        if plan and plan not in rerun_task:
            rerun_task = f"{task}\n\nLatest repair note:\n{plan}"

        run_id = begin_project_run(project, reason=f"repair:{agent}")
        upstream_agent = self._infer_repair_upstream_agent(fields, agent)
        if upstream_agent:
            upstream_task = (
                f"Repair upstream output for {agent.upper()}. {plan or fields.get('problem', '')}\n\n"
                f"Downstream task waiting for this output:\n{task}"
            ).strip()
            print(f"  upstream: rerun {upstream_agent.upper()} ก่อน {agent.upper()}")
            upstream_ok, _ = self._run_repair_agent_once(upstream_agent, upstream_task, project)
            if not upstream_ok:
                print(f"{p.red}  auto repair stopped at upstream {upstream_agent.upper()}{p.reset}")
                return
            expected_upstream = self._repair_expected_upstream_path(fields, project)
            if expected_upstream and not expected_upstream.exists():
                msg = f"upstream repair did not create required file: {expected_upstream}"
                print(f"{p.red}  {msg}{p.reset}")
                self.write_repair_note(
                    upstream_agent,
                    "missing-output",
                    msg,
                    project,
                    f"rerun {upstream_agent.upper()} and ensure required output exists",
                    task=self._repair_note_task(upstream_task),
                )
                return

        promote_upstream_outputs(project, agent, run_id)
        ok, _msg = self._run_repair_agent_once(agent, rerun_task, project)
        expected_output = self._repair_expected_upstream_path(fields, project)
        if ok and expected_output and not expected_output.exists():
            msg = f"repair did not create required file: {expected_output}"
            print(f"{p.red}  {msg}{p.reset}")
            self.write_repair_note(
                agent,
                "missing-output",
                msg,
                project,
                f"rerun {agent.upper()} and ensure required output exists",
                task=self._repair_note_task(rerun_task),
            )
            return
        if ok:
            print(f"{p.green}  ✓ auto repair completed for {agent.upper()}{p.reset}")
            return

    def anna_discover(self, user_input: str) -> None:
        anna_kb = self.load_relevant_kb("anna", user_input, top_n=6) if user_input else self.load_kb("anna")
        system = self.anna_system + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
        if not self.codex_enabled:
            result = self.call_deepseek(system, user_input, label="ANNA discover")
        else:
            result = self.call_claude(system, user_input, label="ANNA discover")
        self.save_kb("anna", f"Task: {user_input}\nDiscovery:\n{result}")

    def read_cli_input(self) -> str | None:
        p = self.palette
        try:
            proj = f" {p.dim}[{self.state.active_project.name}]{p.reset}" if self.state.active_project else ""
            proj_title = f" [{self.state.active_project.name}]" if self.state.active_project else ""
            self.set_tab_title(f"🟢 Anna{proj_title} — พร้อม")
            return input(f"{p.bold}{p.white}คุณ{p.reset}{proj}{p.bold}{p.white}:{p.reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{p.yellow}  ลาก่อนค่ะ{p.reset}")
            return None

    def _split_resume_args(self, raw: str) -> tuple[str, str]:
        raw = raw.strip()
        if not raw:
            return "", ""
        if self.projects_dir.exists():
            project_names = sorted(
                [p.name for p in self.projects_dir.iterdir() if p.is_dir()],
                key=len,
                reverse=True,
            )
            raw_lower = raw.lower()
            for project_name in project_names:
                if raw_lower == project_name.lower():
                    return project_name, ""
                prefix = project_name.lower() + " "
                if raw_lower.startswith(prefix):
                    return project_name, raw[len(project_name):].strip()
        parts = raw.split(" ", 1)
        return parts[0], parts[1].strip() if len(parts) > 1 else ""

    def resume_project(self, name: str = "", extra_instruction: str = "") -> None:
        p = self.palette
        if not name and self.state.active_project:
            project = self.state.active_project
            message = ""
        else:
            project, message = self.cli.resolve_project(name)
        if not project:
            color = p.yellow if message.startswith("พบหลาย") else p.red
            print(f"{color}  {message}{p.reset}")
            return
        self.state.active_project = project
        self.pipeline.rebuild_from_project(project)
        self.set_last_pipeline_project(project)
        done = self.pipeline.completed_agents()
        print(f"\n{p.yellow}  Resume:{p.reset} {p.bold}{project.name}{p.reset}")
        print(f"  เสร็จแล้ว: {p.green}{', '.join(done) or 'ไม่มี'}{p.reset}")
        resume_msg = (
            f"Resume project {project.name}. "
            f"Agents ที่เสร็จแล้ว: {', '.join(done) or 'ไม่มี'}. "
            f"วิเคราะห์ว่าต้องทำอะไรต่อใน CRISP-DM pipeline แล้ว dispatch ต่อทันที"
        )
        if extra_instruction:
            resume_msg += f"\n\nคำสั่งเพิ่มเติมจาก user:\n{extra_instruction}"
        try:
            self.run_pipeline(resume_msg)
        except KeyboardInterrupt:
            print(f"\n{p.yellow}  หยุด resume pipeline{p.reset}")

    def dispatch_project(self, d: dict) -> Path | None:
        raw = str(d.get("project", "") or "").strip()
        if not raw:
            return self.state.active_project
        project, message = self.cli.resolve_project(raw)
        if project:
            return project
        p = self.palette
        print(f"{p.yellow}  ⚠ dispatch project ignored: {message}{p.reset}")
        return self.state.active_project

    def run_all_pipeline_command(self, extra_instruction: str = "") -> None:
        p = self.palette
        force_past_quinn = "--force-past-quinn" in extra_instruction
        extra_instruction = extra_instruction.replace("--force-past-quinn", "").strip()
        project = self.state.active_project
        if not project:
            print(f"{p.red}  ยังไม่ได้เลือก project{p.reset}")
            print("  ใช้: /project <name>")
            return
        blocked, reason, target = self.quinn_hard_gate(project)
        if blocked and not force_past_quinn:
            q_report = project / "output" / "quinn" / "quinn_report.md"
            if not q_report.exists():
                q_report = project / "output" / "quinn" / "quinn_qc_report.md"
            if not self.agent_output_newer_than(project, target, q_report):
                print(f"\n{p.red}[PIPELINE BLOCKED]{p.reset} {reason}")
                print(f"{p.yellow}Dispatch {target.upper()} first with fix.{p.reset}")
                self.write_repair_note("run-all", "quinn-restart-required", reason, project, f"rerun @{target} with fix, then retry")
                return
        input_dir = project / "input"
        has_input_data = input_dir.exists() and any(input_dir.glob("*"))
        resume_from_scout = False
        if not has_input_data:
            scout_csv = project / "output" / "scout" / "scout_output.csv"
            if scout_csv.exists():
                ok, msg = self.validate_agent_output("scout", str(scout_csv), project)
                if not ok:
                    self.print_gate_recovery("scout", msg, project)
                    print(f"{p.red}  RUN-ALL stopped at SCOUT{p.reset}")
                    return
                resume_from_scout = True
                print(f"{p.yellow}  input/ ว่าง แต่ Scout output ผ่าน gate แล้ว — เริ่มต่อจาก DANA{p.reset}")
            else:
                print(f"{p.red}  Project นี้ไม่มี input data: {input_dir}{p.reset}")
                return

        self.set_last_pipeline_project(project)
        self.state.reset_pipeline()
        if resume_from_scout:
            self.pipeline.rebuild_from_project(project)
        else:
            self.pipeline.clear()

        print(f"\n{p.cyan}  RUN-ALL:{p.reset} {p.bold}{project.name}{p.reset}")
        mode = "resume from valid Scout output" if resume_from_scout else "deterministic sequence, no Anna planning prompt"
        print(f"{p.dim}  mode: {mode}{p.reset}")

        blind_rule = (
            "ห้ามอ่าน answer_key ระหว่างทำงาน. "
            "ให้ตัดสินใจเองตามหน้าที่ agent และบันทึก output/report ของตัวเองให้ครบ. "
            "ถ้า input มีหลายไฟล์/หลายชั้น folder ให้เลือกไฟล์ข้อมูลหลักที่เหมาะสมเอง. "
            "ถ้า CSV ไม่ใช่ comma delimiter ให้ detect delimiter เอง เช่น sep=None, engine='python'."
        )
        if extra_instruction:
            blind_rule += " " + extra_instruction

        sequence: list[tuple[str, str]] = [
            ("scout", "เริ่ม pipeline จากข้อมูลใน input/ ของ project นี้ ตรวจไฟล์ทั้งหมด เลือก dataset หลัก สร้าง scout_output.csv และ dataset_profile.md. ถ้ามีไฟล์ workbook (.xlsx/.xls) ให้ใช้ไฟล์นั้นเป็น source หลักและ export row-level dataset จริงออกมา. ต้องมี DATASET_RISK_REGISTER ระบุ source credibility, license, business fit, target suitability, recency, leakage risk, bias/coverage risk และ verdict. ใช้ scout_shortlist.md เฉพาะกรณียังไม่ได้ dataset จริงเท่านั้น และห้ามปล่อย placeholder/manifest 5 แถวใน scout_output.csv. " + blind_rule),
            ("dana", "ทำ data cleaning จาก Scout output สร้าง dana_output.csv และ dana_report.md. ห้ามใช้ target ใน outlier detection และห้ามลบ target. ต้องมี DATA_QUALITY_AUDIT ระบุ before/after quality, removals, imputation, outlier strategy, train-only safeguards, bias impact และ downstream warnings. " + blind_rule),
            ("eddie", "ทำ EDA จาก Dana output หา pattern/relationships และเขียน PIPELINE_SPEC ให้ครบ สร้าง eddie_output.csv และ eddie_report.md. ถ้ามี output/dana/column_roles.json ให้ใช้เป็น context เพื่อแยก id/date/label ออกจาก feature analysis. ต้องมี BUSINESS_EDA_FRAME ระบุ business question, owner, KPI, effect size, causality status, temporal/leakage risk, imbalance/skew risk และ validation strategy. " + blind_rule),
            ("iris_eda", "สรุป early business insight จาก Eddie output สำหรับ bridge ไป Finn/Mo เท่านั้น สร้าง iris_eda_output.csv และ iris_eda_report.md. ต้องมี BUSINESS_EDA_BRIEF ระบุ insight, evidence, business hypothesis, risk, follow-up question, next handoff และ confidence. ห้ามเขียน final recommendation แบบ Iris รอบท้าย และห้ามเปลี่ยน target หรือ schema ownership. " + blind_rule),
            ("finn", "ทำ feature engineering/feature selection จาก Eddie output สร้าง finn_output.csv และ finn_report.md. ใช้ target จาก Scout เท่านั้น เก็บ target เป็น label ห้ามเลือก target เป็น feature. ต้องอ่าน output/dana/column_roles.json ถ้ามี และต้องตัด/กัน id, date, label ออกจาก feature set ใน supervised tasks จริง ๆ ไม่ใช่แค่ตรวจชื่อคอลัมน์. ต้องมี FEATURE_GOVERNANCE ระบุ lineage, prediction-time availability, leakage controls, train-only transforms, temporal/OOT support, actionability และ warnings. " + blind_rule),
            ("mo", "train และ compare models จาก Finn output สร้าง mo_output.csv, model report และ metrics. ถ้า F1/AUC/Accuracy ใกล้ 1.0 ให้ถือว่าอาจ leakage และรายงาน fail. ต้องมี PR-AUC/positive-class metrics/threshold economics/calibration/OOT readiness เมื่อเป็น classification. " + blind_rule),
            ("quinn", "ตรวจ QC/model/data/business satisfaction จากผลก่อนหน้า สร้าง quinn_output.csv และ quinn_report.md. ต้องตรวจ target consistency, leakage columns, perfect metrics, report/CSV contradiction และ WORLD_CLASS_QC. " + blind_rule),
            ("iris", "สรุป business insights/action recommendations จากผล pipeline สร้าง iris_output.csv และ iris_report.md. ต้องมี BUSINESS_DECISION_BRIEF ระบุ business lever, KPI, owner, assumptions, risks, validation plan, confidence และห้ามแนะนำ action จากหลักฐานอ่อนโดยไม่ติด caveat. " + blind_rule),
            ("vera", "สร้าง visualization/report ที่เหมาะสมจากผล pipeline สร้าง vera_output.csv และ vera_report.md. ต้องมี VISUAL_QC สำหรับ chart สำคัญ ระบุ source evidence, decision purpose, chart rationale, misleading-risk check, accessibility และ caveat. " + blind_rule),
            ("rex", "รวม final executive report จากทุก agent สร้าง rex_output.csv และ final report. ห้ามสรุปว่า success ถ้า Quinn fail หรือ Mo metrics/report ขัดกัน. " + blind_rule),
        ]

        if resume_from_scout:
            sequence = sequence[1:]
            prev_agent = "scout"
            completed: list[str] = ["scout"]
        else:
            prev_agent = ""
            completed = []
        sidecar_agents = {"iris_eda"}
        run_id = begin_project_run(project, reason="run-all")
        if resume_from_scout:
            promote_upstream_outputs(project, "dana", run_id)
        for agent, task in sequence:
            try:
                self.delete_old_scripts(self.output_dir_for(project, agent))
                out = self.run_agent(agent, task, prev_agent=prev_agent, project_dir=project, force_past_quinn=force_past_quinn)
                if not out:
                    print(f"{p.red}  RUN-ALL stopped at {agent.upper()}{p.reset}")
                    return
                ok, msg = self.validate_agent_output(agent, out, project)
                if not ok:
                    if agent in ("scout", "dana", "eddie", "finn", "mo"):
                        self.print_gate_recovery(agent, msg, project)
                        print(f"{p.red}  RUN-ALL stopped at {agent.upper()}{p.reset}")
                        return
                    print(f"{p.yellow}  ⚠ {agent.upper()} output warning: {msg}{p.reset}")
                completed.append(agent)
                if agent not in sidecar_agents:
                    prev_agent = agent
            except KeyboardInterrupt:
                print(f"\n{p.yellow}  RUN-ALL stopped by user{p.reset}")
                return
            except Exception as e:
                try:
                    print(f"{p.red}  RUN-ALL failed at {agent.upper()}: {e}{p.reset}")
                except OSError:
                    pass
                return

        print(f"\n{p.green}  RUN-ALL complete:{p.reset} {', '.join(completed)}")

    @staticmethod
    def looks_like_run_all(user_input: str) -> bool:
        lower = user_input.lower()
        return (
            ("run pipeline" in lower or "pipeline ทั้งระบบ" in lower or "ทั้งระบบ" in lower)
            and ("agent" in lower or "pipeline" in lower)
        ) or ("เริ่มpipeline" in lower) or ("เริ่ม pipeline" in lower and "ใหม่" in lower)

    def run_direct_agent_command(self, user_input: str) -> None:
        p = self.palette
        parts = user_input[1:].split(" ", 1)
        agent_part = parts[0].lower()
        task = parts[1] if len(parts) > 1 else ""
        if not task:
            print(f"{p.red}  ใช้งาน:{p.reset} @{agent_part} <task>")
            return
        discover = agent_part.endswith("!")
        agent_name = agent_part.rstrip("!")
        force_past_quinn = "--force-past-quinn" in task
        task = task.replace("--force-past-quinn", "").strip()
        self.set_tab_title(f"⏳ {agent_name.upper()} — กำลังรัน...")
        run_id = begin_project_run(self.state.active_project, reason=f"direct:{agent_name}")
        promote_upstream_outputs(self.state.active_project, agent_name, run_id)
        self.run_agent(agent_name, task, project_dir=self.state.active_project, discover=discover, force_past_quinn=force_past_quinn)
        self.notify_tab(success=True, label=f"{agent_name.upper()} เสร็จ")

    def handle_cli_command(self, user_input: str) -> str:
        p = self.palette
        if user_input.startswith("/"):
            parts = user_input[1:].split(" ", 1)
            slash_cmd = parts[0].lower()
            slash_arg = parts[1].strip() if len(parts) > 1 else ""
            alias = {
                "p": "project",
                "proj": "project",
                "project": "project",
                "r": "resume",
                "resume": "resume",
                "run": "run-all",
                "run-all": "run-all",
                "all": "run-all",
                "repair": "repair",
                "fix": "repair",
                "s": "status",
                "st": "status",
                "status": "status",
                "kb": "kb",
                "help": "help",
                "h": "help",
                "?": "help",
                "claude": "claude",
                "eval": "eval",
                "benchmark": "eval",
                "end": "end session",
                "end-session": "end session",
                "exit": "exit",
                "quit": "exit",
            }.get(slash_cmd)
            if alias is None:
                print(f"{p.yellow}  ไม่รู้จักคำสั่ง /{slash_cmd} — ใช้ /help เพื่อดูคำสั่ง{p.reset}")
                return "handled"
            user_input = f"{alias} {slash_arg}".strip()

        lower = user_input.lower()
        if lower in ("exit", "quit"):
            print(f"{p.yellow}  ลาก่อนค่ะ{p.reset}")
            return "exit"
        if lower == "end session":
            self.state.reset_session()
            print(f"{p.yellow}  ANNA:{p.reset} เริ่ม session ใหม่แล้วค่ะ  {p.dim}(Codex calls reset → 0){p.reset}")
            return "handled"
        if lower == "help":
            self.print_help()
            return "handled"
        if lower == "run-all" or lower.startswith("run-all "):
            extra = user_input[7:].strip() if lower.startswith("run-all ") else ""
            self.run_all_pipeline_command(extra)
            return "handled"
        if lower == "repair" or lower.startswith("repair "):
            repair_arg = user_input[6:].strip() if lower.startswith("repair ") else ""
            if repair_arg.lower() in ("auto", "run", "repair", "rerun", "fix"):
                self._run_latest_repair_note()
            else:
                self.print_latest_repair_note()
            return "handled"
        if lower == "project":
            self.cli.print_status()
            print("  ใช้: /project <name>")
            return "handled"
        if lower.startswith("project "):
            name = user_input[8:].strip()
            project, _message = self.cli.resolve_project(name)
            self.state.active_project = project
            if project:
                self.pipeline.rebuild_from_project(project)
                self.set_last_pipeline_project(project)
            else:
                self.pipeline.clear()
                self.set_last_pipeline_project(None)
            status = f"{self.palette.green}{self.state.active_project}{self.palette.reset}" if self.state.active_project else f"{self.palette.red}ไม่พบ project นี้{self.palette.reset}"
            print(f"{self.palette.yellow}  ANNA:{self.palette.reset} Active project → {status}")
            return "handled"
        if lower.startswith("kb "):
            name = user_input[3:].strip()
            self.cli.print_kb(name, self.load_kb(name))
            return "handled"
        if lower in ("status", "stauts", "stats", "stat", "สถานะ"):
            self.cli.print_status()
            return "handled"
        if lower == "resume":
            self.resume_project()
            return "handled"
        if lower.startswith("resume "):
            project_name, extra_instruction = self._split_resume_args(user_input[7:].strip())
            self.resume_project(project_name, extra_instruction)
            return "handled"
        if lower in ("claude", "claude status", "codex", "codex status"):
            self.cli.print_claude_usage()
            return "handled"
        if lower == "eval":
            from .evaluation import build_eval_report, run_system_eval, run_and_write_default_eval

            results = run_system_eval()
            report = build_eval_report(results)
            eval_path = Path.cwd() / "tmp" / "anna_system_eval.md"
            try:
                run_and_write_default_eval(eval_path)
            except Exception:
                eval_path.parent.mkdir(parents=True, exist_ok=True)
                eval_path.write_text(report, encoding="utf-8")
            passed = sum(1 for result in results if result.passed)
            print(f"\n{self.palette.cyan}Anna system eval:{self.palette.reset} {self.palette.bold}{passed}/{len(results)}{self.palette.reset}")
            print(report)
            print(f"{self.palette.dim}  report: {eval_path}{self.palette.reset}")
            return "handled"
        if user_input.startswith("!!"):
            self.anna_discover(user_input[2:].strip())
            return "handled"
        if user_input.startswith("@"):
            self.run_direct_agent_command(user_input)
            return "handled"
        if self.looks_like_run_all(user_input):
            self.run_all_pipeline_command(user_input)
            return "handled"
        return "pipeline"

    def main_loop(self, read_cli_input: Callable[[], str | None]) -> None:
        from .runner import run_python_script  # imported lazily only to keep module narrow
        _ = run_python_script  # keep lint quiet if unused in future
        while True:
            user_input = read_cli_input()
            if user_input is None:
                break
            if not user_input:
                continue
            try:
                command_result = self.handle_cli_command(user_input)
            except OSError:
                if user_input.strip().lower() in ("exit", "quit", "/exit", "/quit"):
                    break
                continue
            if command_result == "exit":
                break
            if command_result == "handled":
                continue
            try:
                self.run_pipeline(user_input)
                self.notify_tab(success=True, label="เสร็จสิ้น — พร้อมรับคำสั่ง")
            except KeyboardInterrupt:
                with self.state._proc_lock:
                    _kb_procs = list(self.state._active_procs)
                for _kp in _kb_procs:
                    try:
                        _kp.kill()
                    except OSError:
                        pass
                print(f"\n{self.palette.yellow}  หยุด pipeline แล้ว — พร้อมรับคำสั่งใหม่{self.palette.reset}")
                self.state.stop_requested.clear()
                self.notify_tab(success=False, label="หยุดกลางคัน")
