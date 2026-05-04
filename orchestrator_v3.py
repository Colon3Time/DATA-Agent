"""
DataScienceOS Orchestrator — Path-Based Pipeline v2
Pipeline ส่ง FILE PATH เท่านั้น ไม่ส่ง content
Agent ที่มี script → รัน script จริง (subprocess)
Agent ที่ไม่มี script → LLM สร้าง report + Python code
"""

import re
import os
import sys
import shutil
import threading
import concurrent.futures
import time
from datetime import datetime
from pathlib import Path
from anna_core.env import load_app_env

from anna_core.agent_runtime import (
    build_agent_path_message,
    delete_old_scripts,
    extract_python_blocks,
    latest_input_file,
    latest_output_csv,
    output_dir_for,
    scout_input_csv,
    should_regenerate_vera_script_for_meeting_report,
)
from anna_core.actions import WorkspacePaths
from anna_core.action_executor import ActionExecutor, ActionPalette
from anna_core.anna_contract import ANNA_OUTPUT_CONTRACT, validate_dispatch_plan
from anna_core.cli_runtime import CliPalette, CliRenderer
from anna_core.app_commands import AppCommands, CommandPalette
from anna_core.config import load_config
from anna_core.dispatcher import DispatchParser
from anna_core.intent import IntentClassifier
from anna_core.evaluation import build_eval_report, run_system_eval, run_and_write_default_eval
from anna_core.gates import (
    agent_output_newer_than as _agent_output_newer_than,
    archive_agent_outputs_before_rerun,
    check_pipeline_spec,
    quinn_hard_gate,
    resolve_input_path,
)
from anna_core.kb import KnowledgeBase
from anna_core.logging import RawLogger, SessionMemoryStore
from anna_core.llm import LLMClient, TerminalPalette
from anna_core.model_policy import ModelPolicy
from anna_core.reviewer import ResponseReviewer
from anna_core.router import TaskRouter
from anna_core.pipeline_store import PipelineStore
from anna_core.pipeline_runtime import (
    build_anna_system_prompt,
    build_summary_prompt,
    collect_report_sections,
    group_dispatches,
    list_projects,
)
from anna_core.mo_phase import detect_mo_phase, mo_script_matches_phase, sync_mo_canonical_report
from anna_core.project import AgentSpecLoader, ProjectDetector
from anna_core.run_guard import begin_project_run, mark_output_current, output_is_current, promote_upstream_outputs
from anna_core.runner import run_python_script
from core.schema_contract import infer_producer_from_path, validate_handoff
from anna_core.runner import agent_script_is_usable, builtin_agent_script
from anna_core.runner import agent_script_is_placeholder
from anna_core.state import OrchestratorState

if sys.platform == "win32":
    import msvcrt

BASE_DIR = Path(__file__).parent
load_app_env(BASE_DIR / ".env")
CONFIG = load_config(BASE_DIR)


def _configure_text_streams() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_configure_text_streams()

# ── Colors ────────────────────────────────────────────────────────────────────
def _supports_ansi() -> bool:
    if CONFIG.no_color or not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    return bool(
        os.environ.get("WT_SESSION")
        or os.environ.get("ANSICON")
        or os.environ.get("TERM_PROGRAM")
        or os.environ.get("ConEmuANSI") == "ON"
    )


COLOR_ENABLED = _supports_ansi()

if COLOR_ENABLED:
    RST = "\033[0m"
    BLD = "\033[1m"
    DIM = "\033[2m"
    CY  = "\033[96m"   # Bright Cyan   — UI / borders
    GR  = "\033[92m"   # Bright Green  — success / ✓
    YL  = "\033[93m"   # Bright Yellow — Anna / separators
    RD  = "\033[91m"   # Bright Red    — errors / ✗
    BL  = "\033[94m"   # Bright Blue   — DeepSeek
    MG  = "\033[95m"   # Bright Magenta — Claude
    WH  = "\033[97m"   # Bright White
else:
    RST = BLD = DIM = CY = GR = YL = RD = BL = MG = WH = ""

TERMINAL_TITLE_ENABLED = CONFIG.terminal_title and COLOR_ENABLED

# ── Config ────────────────────────────────────────────────────────────────────
DEEPSEEK_URL = CONFIG.deepseek_url
DEEPSEEK_MODEL = CONFIG.deepseek_model
CLAUDE_MODEL = CONFIG.claude_model
AGENTS_DIR = CONFIG.agents_dir
LOGS_DIR = CONFIG.logs_dir
KNOWLEDGE_DIR = CONFIG.knowledge_dir
PIPELINE_DIR = CONFIG.pipeline_dir
PROJECTS_DIR = CONFIG.projects_dir
MODE = CONFIG.mode
CLAUDE_LIMIT = CONFIG.claude_limit
STEP_MODE = CONFIG.step_mode

ANNA_PERSONA_GUARD = """

---
## Anna Persona Guard (non-negotiable)
- Anna is female.
- When Anna replies in Thai, Anna must use a feminine voice.
- Use "ดิฉัน" or "Anna" for self-reference when needed.
- End polite Thai sentences with "ค่ะ" or "คะ" as appropriate.
- Never use "ครับ", "คับ", or "ฮะ" in Anna's own voice.
- Session memory may contain older replies with "ครับ"; treat those as stale style examples and do not imitate them.
"""

ANNA_SYSTEM = (BASE_DIR / "CLAUDE.md").read_text(encoding="utf-8") + ANNA_PERSONA_GUARD

VALID_AGENTS = {"scout", "dana", "eddie", "iris_eda", "max", "finn", "mo", "iris", "vera", "quinn", "rex"}
SIDE_CAR_AGENTS = {"iris_eda"}

# ── CRISP-DM Phase Mapping ──────────────────────────────────────────────────
CRISP_DM_PHASES = {
    "data_understanding": ["scout", "eddie"],
    "analysis_bridge": ["iris_eda"],
    "data_preparation":   ["dana", "max", "finn"],
    "modeling":           ["mo"],
    "evaluation":         ["quinn", "iris"],
    "deployment":         ["vera", "rex"],
}
AGENT_TO_PHASE = {a: p for p, agents in CRISP_DM_PHASES.items() for a in agents}
MAX_AGENT_ITER = 5  # สูงสุดที่ agent เดียวกันรันซ้ำได้ใน 1 pipeline (CRISP-DM: explore→preprocess→tune→validate)

# ── Terminal Tab Notification ─────────────────────────────────────────────────

def notify_tab(success: bool = True, label: str = ""):
    """แจ้งเตือน Windows Terminal tab เมื่อ pipeline เสร็จ
    - Bell (\\a) → Windows Terminal แสดง dot notification บน tab ที่ไม่ได้ focus
    - OSC 0     → เปลี่ยน tab title ให้แสดงสถานะ
    """
    if not TERMINAL_TITLE_ENABLED:
        return
    print('\a', end='', flush=True)
    icon  = "✅" if success else "⚠️"
    title = f"{icon} Anna — {label}" if label else (f"{icon} Anna — พร้อม" if success else f"{icon} Anna — หยุด")
    print(f'\033]0;{title}\007', end='', flush=True)


def set_tab_title(title: str):
    """เปลี่ยนชื่อ tab terminal ระหว่าง pipeline รัน"""
    if not TERMINAL_TITLE_ENABLED:
        return
    print(f'\033]0;{title}\007', end='', flush=True)


def set_pipeline_stage(agent_name: str, stage: str, detail: str = "") -> None:
    """Show a compact live stage indicator in terminal title and logs."""
    label = f"{agent_name.upper()} :: {stage}"
    if detail:
        label += f" — {detail[:48]}"
    print(f"{DIM}  [stage] {label}{RST}")
    set_tab_title(f"⏳ {label}")


# ── Intent Classifier ─────────────────────────────────────────────────────────

_PIPELINE_KW = {
    # agent names
    "scout", "dana", "eddie", "max", "finn", "mo", "iris", "vera", "quinn", "rex",
    # action words (Thai + English)
    "ให้", "dispatch", "รัน", "run", "ทำ", "วิเคราะห์", "cleaning", "eda",
    "pipeline", "dataset", "data", "โมเดล", "model", "train", "predict",
    "insight", "visualization", "report", "clean", "ข้อมูล", "สร้าง", "project",
    "csv", "excel", "xlsx", "json", "ml", "deep", "learning", "sklearn",
    "world-class", "world class", "production-ready", "production readiness",
    "business rigor", "executive-ready", "cost-benefit", "expected value",
    "pr-auc", "oot", "out-of-time", "calibration",
}
_CHAT_KW = {
    "สวัสดี", "hello", "hi", "ขอบคุณ", "thanks", "ok", "โอเค", "เข้าใจ",
    "ดี", "เยี่ยม", "ตกลง", "อธิบาย", "explain", "คือ", "หมายถึง",
}

def classify_intent(text: str) -> str:
    return INTENT_CLASSIFIER.classify(text)


STATE = OrchestratorState()
LLM = LLMClient(
    state=STATE,
    deepseek_url=DEEPSEEK_URL,
    deepseek_model=DEEPSEEK_MODEL,
    claude_model=CLAUDE_MODEL,
    claude_limit=CLAUDE_LIMIT,
    palette=TerminalPalette(
        reset=RST,
        bold=BLD,
        dim=DIM,
        yellow=YL,
        red=RD,
        blue=BL,
        magenta=MG,
    ),
)
KB = KnowledgeBase(KNOWLEDGE_DIR, log=lambda role, content, task="", output="": log_raw(role, content, task, output))
DISPATCHER = DispatchParser(
    VALID_AGENTS,
    on_reject=lambda agent: print(f"{RD}  ✗ dispatch ถูกปฏิเสธ — agent='{agent}' ไม่ถูกต้อง{RST}"),
)
PIPELINE = PipelineStore(PIPELINE_DIR)
_STATE_LOCK = threading.Lock()   # ป้องกัน race condition ใน parallel agent execution
RAW_LOGGER = RawLogger(LOGS_DIR, active_project=lambda: STATE.active_project)
SESSION_MEMORY = SessionMemoryStore(KNOWLEDGE_DIR)
WORKSPACE_PATHS = WorkspacePaths(BASE_DIR)
PROJECT_DETECTOR = ProjectDetector(PROJECTS_DIR)
AGENT_SPEC_LOADER = AgentSpecLoader(AGENTS_DIR, VALID_AGENTS)
INTENT_CLASSIFIER = IntentClassifier(_PIPELINE_KW, _CHAT_KW)
TASK_ROUTER = TaskRouter()
MODEL_POLICY = ModelPolicy()
RESPONSE_REVIEWER = ResponseReviewer()
ACTION_EXECUTOR = ActionExecutor(
    base_dir=BASE_DIR,
    projects_dir=PROJECTS_DIR,
    workspace_paths=WORKSPACE_PATHS,
    log=lambda role, content, task="", output="": log_raw(role, content, task, output),
    save_kb=lambda agent, content, entry_type="discovery": save_kb(agent, content, entry_type),
    ask_deepseek=lambda system, user, label="": call_deepseek(system, user, label=label),
    ask_claude=lambda system, user, label="": call_claude(system, user, label=label),
    set_active_project=lambda project: setattr(STATE, "active_project", project),
    palette=ActionPalette(
        reset=RST,
        bold=BLD,
        dim=DIM,
        cyan=CY,
        green=GR,
        red=RD,
        blue=BL,
        magenta=MG,
    ),
)
CLI = CliRenderer(
    state=STATE,
    pipeline=PIPELINE,
    projects_dir=PROJECTS_DIR,
    deepseek_model=DEEPSEEK_MODEL,
    claude_model=CLAUDE_MODEL,
    claude_limit=CLAUDE_LIMIT,
    mode=MODE,
    palette=CliPalette(
        reset=RST,
        bold=BLD,
        dim=DIM,
        cyan=CY,
        green=GR,
        yellow=YL,
        red=RD,
        blue=BL,
        magenta=MG,
        white=WH,
    ),
)


def _esc_monitor():
    """Background thread: กด ESC เพื่อหยุด script ที่รันอยู่ โดยไม่ปิดโปรแกรม"""
    while True:
        if sys.platform == "win32" and msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':
                # ดูด extra bytes ที่ terminal บางตัวส่งตาม ESC ออกให้หมด
                time.sleep(0.02)
                while msvcrt.kbhit():
                    msvcrt.getch()
                with STATE._proc_lock:
                    _esc_procs = list(STATE._active_procs)
                if _esc_procs:
                    print(f"\n{YL}  [ESC] หยุด script ({len(_esc_procs)}) — กลับไปรอคำสั่ง...{RST}")
                    for _ep in _esc_procs:
                        try:
                            _ep.kill()
                        except OSError:
                            pass
                    STATE.stop_requested.set()
        time.sleep(0.05)


# ── LLM Callers ───────────────────────────────────────────────────────────────

def call_deepseek(system_prompt: str, user_message: str, label: str = "", history: list|None = None) -> str:
    return LLM.call_deepseek(system_prompt, user_message, label=label, history=history)


def call_claude(system_prompt: str, user_message: str, label: str = "") -> str:
    return LLM.call_claude(system_prompt, user_message, label=label)


def call_model(provider: str, system_prompt: str, user_message: str, label: str = "", history: list | None = None) -> str:
    provider = (provider or "deepseek").lower()
    if provider == "codex":
        return call_claude(system_prompt, user_message, label=label)
    return call_deepseek(system_prompt, user_message, label=label, history=history)


# ── Knowledge Base ────────────────────────────────────────────────────────────

def load_kb(agent_name: str) -> str:
    return KB.load(agent_name)

def load_relevant_kb(agent_name: str, task: str, top_n: int = 6) -> str:
    return KB.load_relevant(agent_name, task, top_n=top_n)

def save_kb(agent_name: str, content: str, entry_type: str = "discovery"):
    KB.save(agent_name, content, entry_type=entry_type)

def consolidate_kb(agent_name: str):
    KB.consolidate(agent_name)


# ── Step Confirmation ────────────────────────────────────────────────────────

def confirm_next_step(done_agent: str, next_agent: str, next_task: str,
                      step_num: int, total: int) -> str:
    """
    ถามผู้ใช้ก่อนรัน agent ถัดไป
    คืนค่า: 'y' = ไปต่อ | 'n' = หยุด pipeline | 's' = ข้าม agent นี้
    """
    print(f"\n{YL}{'─'*55}{RST}")
    print(f"{YL}  ✓ {BLD}{done_agent.upper()}{RST}{YL} เสร็จแล้ว  ({step_num}/{total}){RST}")
    print(f"{CY}  → ถัดไป: {BLD}{next_agent.upper()}{RST}")
    print(f"{DIM}    {next_task[:80]}{'...' if len(next_task) > 80 else ''}{RST}")
    print(f"{YL}{'─'*55}{RST}")
    try:
        ans = input(
            f"  {BLD}ไปต่อไหม? [y=ใช่ / n=หยุด / s=ข้ามstep นี้]:{RST} "
        ).strip().lower()
    except EOFError:
        ans = "y"
    if ans not in ("y", "n", "s"):
        ans = "y"
    log_raw("anna", f"step confirm: {done_agent}→{next_agent} user={ans}", task="step-mode")
    return ans


# ── Pipeline (PATH-BASED) ─────────────────────────────────────────────────────

def pipeline_write(agent_name: str, file_path: str):
    PIPELINE.write(agent_name, file_path)

def pipeline_read(agent_name: str) -> str:
    path = PIPELINE.read(agent_name)
    if not path:
        return ""
    project_dir = PIPELINE.project_dir or STATE.active_project
    if project_dir and infer_producer_from_path(path, project_dir):
        if not output_is_current(Path(path), project_dir):
            return ""
    return path

_last_pipeline_project: Path | None = None


def pipeline_clear():
    """Clear PIPELINE only when the active project changes — preserves paths for resume."""
    global _last_pipeline_project
    if STATE.active_project != _last_pipeline_project:
        PIPELINE.clear()
        _last_pipeline_project = STATE.active_project


# ── Script Runner ─────────────────────────────────────────────────────────────

def _strip_python_fences(text: str) -> str:
    """Remove markdown code fences accidentally written into .py files."""
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*```(?:python|py)?\s*\r?\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\r?\n```\s*$", "\n", cleaned)
    return cleaned


def sanitize_python_script_file(script_path: Path) -> None:
    """Make any agent-generated .py file executable even if LLM included fences."""
    if script_path.suffix != ".py" or not script_path.exists():
        return
    try:
        original = script_path.read_text(encoding="utf-8")
        cleaned = _strip_python_fences(original)
        if cleaned != original:
            script_path.write_text(cleaned, encoding="utf-8")
            log_raw("system", f"sanitized markdown fences from {script_path.name}", task="script-sanitize")
    except Exception:
        pass


def find_agent_script(agent_name: str, project_dir: Path|None) -> Path|None:
    if not project_dir:
        return None
    agent_dir = project_dir / "output" / agent_name
    if agent_dir.exists():
        scripts = sorted(agent_dir.glob("*.py"), key=lambda x: x.stat().st_mtime)
        for script in reversed(scripts):
            ok, _reason = agent_script_is_usable(script)
            if ok:
                if agent_name == "finn":
                    try:
                        text = script.read_text(encoding="utf-8", errors="ignore").lower()
                        if "finn_feature_manifest" not in text or "recency_days" not in text or "monetary" not in text:
                            continue
                    except Exception:
                        continue
                return script
    return None


def scout_download_task_requires_fresh_script(task: str, project_dir: Path | None) -> bool:
    """Scout shortlist scripts must not be reused for confirmed download tasks."""
    if not project_dir:
        return False
    task_lc = task.lower()
    download_markers = (
        "download",
        "load dataset",
        "save dataset",
        "confirmed",
        "confirm",
        "walmart",
        "kaggle",
        "โหลด",
        "ดาวน์โหลด",
        "ยืนยัน",
        "บันทึก",
    )
    if not any(marker in task_lc for marker in download_markers):
        return False
    return not bool(scout_input_csv(project_dir))


def scout_script_handles_confirmed_download(script: Path) -> bool:
    try:
        text = script.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    required_markers = (
        "walmart-recruiting-store-sales-forecasting",
        "kaggle",
        "input_dir",
    )
    return all(marker in text for marker in required_markers)


def normalize_scout_loaded_dataset(project_dir: Path | None, output_dir: Path) -> str:
    """Make a real input dataset usable by the normal scout->dana handoff."""
    if not project_dir:
        return ""
    src_text = scout_input_csv(project_dir)
    if not src_text:
        return ""
    src = Path(src_text)
    if not src.exists():
        return ""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "scout_output.csv"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    _write_scout_profile_fallback(project_dir, dest)
    return str(dest)

def validate_or_discard_script(script: Path) -> bool:
    ok, reason = agent_script_is_usable(script)
    if not ok:
        try:
            script.unlink(missing_ok=True)
        except Exception:
            pass
        log_raw("system", f"discarded unusable script: {script.name} ({reason})", task="script-validate")
    return ok


def write_builtin_script(agent_name: str, output_dir: Path) -> Path | None:
    code = builtin_agent_script(agent_name)
    if not code:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{agent_name}_script.py"
    path.write_text(code, encoding="utf-8")
    return path

def run_script(script_path: Path, input_path: str, output_dir: Path) -> tuple[str, int, str]:
    """Returns (output_path, returncode, stderr)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitize_python_script_file(script_path)
    ok, reason = agent_script_is_usable(script_path)
    if not ok:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass
        return str(output_dir), 127, f"unusable script: {reason}"
    set_pipeline_stage(output_dir.name, "script", script_path.name)
    print(f"\n{CY}  ▶ SCRIPT{RST}  {BLD}{script_path.name}{RST}  {DIM}← {input_path or 'no input'}{RST}")
    print(f"{DIM}  กด ESC เพื่อหยุด script นี้{RST}")

    result = run_python_script(script_path, input_path, output_dir, STATE, timeout_seconds=1800)

    if STATE.stop_requested.is_set():
        return str(output_dir), -999, "หยุดโดยผู้ใช้ (ESC)"

    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0:
        set_pipeline_stage(output_dir.name, "script-error", f"exit {result.returncode}")
        print(f"{RD}  ╔══ SCRIPT ERROR ══╗{RST}")
        print(f"{RD}{result.stderr[:500]}{RST}")
        print(f"{RD}  ╚{'═'*18}╝{RST}")

    csvs = sorted(output_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    canonical = output_dir / f"{output_dir.name}_output.csv"
    if canonical.exists():
        out_path = str(canonical)
    else:
        supplementary = (
            "correlat", "mi_score", "feature_score", "summary", "metric",
            "importance", "flag", "shap", "outlier", "report", "comparison",
        )
        main_csvs = [
            f for f in csvs
            if not any(pat in f.stem.lower() for pat in supplementary)
        ]
        out_path = str(main_csvs[0] if main_csvs else csvs[0]) if csvs else str(output_dir)
    if result.returncode == 0 and not csvs:
        if output_dir.name == "scout":
            project_dir = output_dir.parent.parent
            scout_csv = normalize_scout_loaded_dataset(project_dir, output_dir)
            if scout_csv:
                return scout_csv, 0, result.stderr
        fake_err = "Script ran successfully but produced no CSV in OUTPUT_DIR. Must add: df.to_csv(os.path.join(OUTPUT_DIR, 'output.csv'), index=False)"
        return out_path, -1, fake_err
    if result.returncode == 0 and output_dir.name == "scout":
        project_dir = output_dir.parent.parent
        scout_csv = normalize_scout_loaded_dataset(project_dir, output_dir)
        if scout_csv:
            return scout_csv, 0, result.stderr
    set_pipeline_stage(output_dir.name, "script-done", Path(out_path).name if out_path else "no-output")
    return out_path, result.returncode, result.stderr


# ── Agent Runner ──────────────────────────────────────────────────────────────

def get_system_prompt(agent_name: str, task: str = "") -> str:
    f = AGENTS_DIR / f"{agent_name}.md"
    base = f.read_text(encoding="utf-8") if f.exists() else f"You are {agent_name}, a data science specialist."
    base += (
        "\n\n---\n## Pre-Work Protocol (ทำก่อนทุกครั้ง — บังคับ)\n"
        "1. อ่าน Knowledge Base ด้านล่างทั้งหมดก่อนเริ่มทำงาน\n"
        "2. [FEEDBACK] = ข้อแนะนำจากการแก้งาน — ใช้เป็น guideline แต่ตัดสินใจตามบริบทจริง ไม่ใช่กฎตายตัว\n"
        "3. [DISCOVERY] = วิธีที่พิสูจน์แล้วว่าได้ผลดี — ใช้ก่อนเสมอถ้าเหมาะสม\n"
        "4. อ่าน Input file path ที่ระบุใน task message แล้วโหลดข้อมูลจาก path นั้นทันที\n"
        "5. บันทึก Self-Improvement Report ทุกครั้งหลังทำงานเสร็จ\n"
        "6. เมื่อทำงานเสร็จ ต้องเขียน Agent Report ก่อนส่งผลต่อเสมอ:\n"
        "```\n"
        "Agent Report — [ชื่อ Agent]\n"
        "============================\n"
        "รับจาก     : [agent ก่อนหน้า หรือ User]\n"
        "Input      : [อธิบายสั้นๆ ว่าได้รับอะไรมา]\n"
        "ทำ         : [ทำอะไรบ้าง]\n"
        "พบ         : [สิ่งสำคัญที่พบ 2-3 ข้อ]\n"
        "เปลี่ยนแปลง: [data หรือ insight เปลี่ยนยังไง]\n"
        "ส่งต่อ     : [agent ถัดไป] — [ส่งอะไรไป]\n"
        "```\n"
        "\n\n---\n## ⚠ กฎสำคัญที่สุด — ห้ามละเมิด\n"
        "1. **ห้ามใช้ `<thinking>` tags, XML tool calls, หรือ `<assistant_tool_use>` syntax ใดๆ**\n"
        "2. **ต้องตอบด้วย Python code block เสมอ** (```python ... ```) — ไม่ใช่แค่ plan หรือ text\n"
        "3. **ห้ามแกล้งทำเป็นว่าอ่านไฟล์ได้** — คุณอ่านไม่ได้ ต้องเขียน script ที่รันจริง\n"
        "4. **Script ต้องโหลดข้อมูลจริงจาก INPUT_PATH และ save CSV จริงใน OUTPUT_DIR**\n\n"
        "---\n## กฎการเขียน Python Script (บังคับ)\n"
        "ทุก script ต้องรับ argument ผ่าน argparse เท่านั้น ห้าม hardcode path\n\n"
        "```python\n"
        "import argparse, os, pandas as pd\n"
        "from pathlib import Path\n\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--input',      default='')\n"
        "parser.add_argument('--output-dir', default='')\n"
        "args, _ = parser.parse_known_args()\n\n"
        "INPUT_PATH = args.input\n"
        "OUTPUT_DIR = args.output_dir\n"
        "os.makedirs(OUTPUT_DIR, exist_ok=True)\n\n"
        "# ถ้า input เป็น .md ให้หา CSV จาก parent folder แทน\n"
        "if INPUT_PATH.endswith('.md'):\n"
        "    parent = Path(INPUT_PATH).parent.parent\n"
        "    csvs = sorted(parent.glob('**/dana_output.csv')) + sorted(parent.glob('**/*_output.csv'))\n"
        "    if csvs: INPUT_PATH = str(csvs[0])\n\n"
        "df = pd.read_csv(INPUT_PATH)\n"
        "print(f'[STATUS] Loaded: {df.shape}')\n"
        "# ... ทำงานตามที่ได้รับมอบหมาย ...\n"
        "output_csv = os.path.join(OUTPUT_DIR, '{agent_name}_output.csv')\n"
        "df.to_csv(output_csv, index=False)\n"
        "print(f'[STATUS] Saved: {output_csv}')\n"
        "```\n\n"
        "- ห้ามใช้ r'C:\\\\...' หรือ path ตายตัวใดๆ\n"
        "- ถ้า input เป็น folder ให้ glob หา .csv ข้างใน\n"
        "- output ทุกไฟล์ต้อง save ใน OUTPUT_DIR\n"
        "- ต้องมี print('[STATUS] ...') เพื่อแสดงความคืบหน้า\n"
    )
    if agent_name in {"mo", "iris", "quinn", "rex"}:
        base += (
            "\n\n---\n## World-Class Analytics Standard (mandatory)\n"
            "For any predictive analytics project, the work must be executive-ready and production-aware, not only metric-focused.\n"
            "- Do not rely on F1/ROC-AUC alone when the target is imbalanced. Report PR-AUC / Average Precision and positive-class precision, recall, and F1.\n"
            "- Include threshold strategy. For business campaigns, include an Expected Value / Cost-Benefit matrix with false-positive cost, false-negative/opportunity cost, and selected decision threshold.\n"
            "- Use validation that matches deployment risk. If data has time, month, date, period, macroeconomic, campaign, or sequential fields, add time-based split or out-of-time validation. If unavailable, state why and mark as production limitation.\n"
            "- Check calibration for probability models, preferably Brier score or calibration curve summary, before recommending automated decisions.\n"
            "- Benchmark strong tabular models when dependencies exist: XGBoost, LightGBM, or CatBoost. If unavailable, report the dependency constraint and a concrete provisioning step before production.\n"
            "- Separate prototype readiness from production readiness. A report may be executive-ready while still requiring OOT validation, dependency provisioning, monitoring, and retraining before production.\n"
            "- Translate technical metrics into business impact with stated assumptions. Never present ROI or cost savings without the assumptions used.\n"
        )
    if agent_name == "scout":
        base += (
            "\n\n---\n## Scout Data Sourcing Rigor Guard (mandatory)\n"
            "- Must include DATASET_RISK_REGISTER: source credibility, license, business fit, target suitability, recency/deployment fit, leakage risks, bias/coverage risks, data dictionary, and verdict.\n"
            "- Do not present a dataset as production-grade unless source, license, target definition, recency, documentation, and limitations are documented.\n"
        )
    if agent_name == "dana":
        base += (
            "\n\n---\n## Dana Data Quality Governance Guard (mandatory)\n"
            "- Must include DATA_QUALITY_AUDIT: raw/cleaned shape, completeness/validity change, removals, imputation, outlier strategy, train-only safeguards, bias/coverage impact, and downstream warnings.\n"
            "- Cleaning must preserve raw meaning, document before/after effects, and avoid fitting target-aware transforms on future/test data.\n"
        )
    if agent_name == "eddie":
        base += (
            "\n\n---\n## Eddie Business EDA Guard (mandatory)\n"
            "- Must include BUSINESS_EDA_FRAME: business question, decision owner, target KPI, strongest evidence, causality status, temporal/leakage risk, imbalance/skew risk, validation strategy, and next-agent warnings.\n"
            "- Do not overstate correlations as causality; include effect size and business meaning for top findings.\n"
        )
    if agent_name == "finn":
        base += (
            "\n\n---\n## Finn Feature Governance Guard (mandatory)\n"
            "- Must include FEATURE_GOVERNANCE: feature lineage, prediction-time availability, leakage controls, train-only transforms, temporal/OOT support columns, actionability, and warnings for Mo/Iris.\n"
            "- Drop or flag post-outcome, target-derived, ID-memorization, and future aggregate features.\n"
        )
    if agent_name == "max":
        base += (
            "\n\n---\n## Max Pattern Validity Guard (mandatory)\n"
            "- Must include PATTERN_VALIDITY for important patterns: evidence metric, sample size, stability check, spurious risk, business actionability, confidence, and whether to send to Iris.\n"
            "- Exploratory or unstable patterns must be labeled as such and cannot become recommendations without validation.\n"
        )
    if agent_name == "vera":
        base += (
            "\n\n---\n## Vera Visualization Integrity Guard (mandatory)\n"
            "- Must include VISUAL_QC for each important chart: source evidence, decision purpose, audience, chart rationale, misleading-risk check, accessibility check, and caveat status.\n"
            "- Do not visualize unsupported claims; show uncertainty or caveats where evidence is incomplete.\n"
        )
    if agent_name == "iris":
        base += (
            "\n\n---\n## Iris Business Rigor Guard (mandatory)\n"
            "- Every recommendation must map to a business lever: revenue, cost, risk, retention, conversion, productivity, compliance, or customer experience.\n"
            "- Every recommendation must name an owner/team, target KPI, expected direction of impact, assumptions, implementation effort, and key risk.\n"
            "- Separate proven insight from hypothesis. If evidence is correlational, label it correlational and propose validation such as A/B test, pilot, cohort analysis, or causal test.\n"
            "- Include unit economics when relevant: CAC, LTV, margin, cost per action, conversion value, false-positive cost, opportunity cost, payback, or ROI assumptions.\n"
            "- Do not recommend action from weak evidence. If statistical significance, effect size, sample size, or model validation is missing, mark confidence as Low/Medium and request more analysis.\n"
            "- Required output block: BUSINESS_DECISION_BRIEF with Evidence, Business lever, KPI, Owner, Action, Expected impact, Assumptions, Risks, Validation plan, Confidence.\n"
        )
    if agent_name == "mo":
        mo_phase = detect_mo_phase(task)
        if mo_phase == 2:
            base += (
                "\n\n---\n## Mo Phase 2 Dispatch Guard (บังคับ)\n"
                "- งานนี้คือ Phase 2 Tune เท่านั้น ห้ามทำ Phase 1 Explore ซ้ำ\n"
                "- ต้องอ่านผล Phase 1 เดิมเพื่อหา best algorithm แล้ว tune เฉพาะ algorithm นั้น\n"
                "- ต้องใช้ RandomizedSearchCV จริง และบันทึก best_params_, best_score_, CV setup, train/test metrics\n"
                "- ต้องเปรียบเทียบ tuned model กับ default model ของ algorithm เดียวกัน\n"
                "- ต้องรายงาน PR-AUC, positive-class precision/recall/F1, calibration check, threshold strategy และ expected-value framework ถ้าเป็น classification\n"
                "- ถ้าข้อมูลมีลำดับเวลา/month/date/period/macro variables ต้องเพิ่ม time-based หรือ out-of-time validation หรือเขียน limitation ชัดเจน\n"
                "- ต้องเขียน output/mo/mo_report.md ใหม่เป็น Phase 2 report ห้ามคง report Phase 1 เดิมไว้\n"
                "- ถ้า best algorithm จาก Phase 1 คือ Random Forest ให้ tune RandomForestClassifier เท่านั้น\n"
            )
        elif mo_phase == 3:
            base += (
                "\n\n---\n## Mo Phase 3 Dispatch Guard (บังคับ)\n"
                "- งานนี้คือ Phase 3 Validate เท่านั้น ห้ามทำ Phase 1 Explore หรือ Phase 2 Tune ซ้ำ\n"
                "- ต้อง validate tuned model จาก Phase 2 และเปรียบเทียบกับ default model\n"
                "- ต้องรายงาน final validation metrics, overfitting gap, leakage check, และเลือก final model\n"
                "- ต้องมี production-readiness verdict: OOT/time split status, PR-AUC status, calibration status, cost-benefit threshold, dependency benchmark status\n"
                "- ต้องเขียน output/mo/mo_report.md ใหม่เป็น Phase 3 report ห้ามคง report Phase 1/2 เดิมไว้\n"
            )
    if agent_name == "iris_eda":
        base += (
            "\n\n---\n## Iris EDA Bridge Guard (mandatory)\n"
            "- งานนี้คือ early business insight หลัง Eddie EDA เท่านั้น ห้ามทำ final decision brief หรือ modeling recommendation แทน Iris รอบสุดท้าย\n"
            "- ต้องอ่าน Eddie output/report แล้วสรุปเป็น BUSINESS_EDA_BRIEF ที่บอก insight, evidence, business hypothesis, key risk, follow-up question, next handoff, และ confidence\n"
            "- ต้องชี้ชัดว่าอะไรคือ evidence ที่ยืนยันได้จริง และอะไรคือ hypothesis ที่ยังต้องส่งต่อให้ Finn/Mo ตรวจต่อ\n"
            "- ห้ามเปลี่ยน target_column, problem_type, หรือ schema ownership ของ Scout\n"
            "- ถ้าเจอ conflict ระหว่าง Eddie report กับ dataset_profile ให้บันทึก conflict และขอ repair แทนการเดาเอง\n"
        )
    kb = load_relevant_kb(agent_name, task) if task else load_kb(agent_name)
    if kb:
        base += f"\n\n---\n## Knowledge Base — {agent_name}\n{kb}"
    return base


def extract_key_blocks(text: str) -> str:
    """ดึง structured blocks สำคัญจาก report — Anna ต้องเห็น blocks เหล่านี้เพื่อ dispatch ถูกต้อง"""
    KEY_BLOCKS = [
        "PIPELINE_SPEC",
        "INSIGHT_QUALITY",
        "BUSINESS_SATISFACTION",
        "DATASET_PROFILE",
        "PREPROCESSING_REQUIREMENT",
        "DATASET_RISK_REGISTER",
        "DATA_QUALITY_AUDIT",
        "BUSINESS_EDA_FRAME",
        "BUSINESS_EDA_BRIEF",
        "FEATURE_GOVERNANCE",
        "PRODUCTION_READINESS",
        "PATTERN_VALIDITY",
        "BUSINESS_DECISION_BRIEF",
        "VISUAL_QC",
        "WORLD_CLASS_QC",
        "DL_ESCALATE",
        "RESTART_CYCLE",
        "Loop Back To Finn",
        "NEED_CLAUDE",
    ]
    found = []
    for block in KEY_BLOCKS:
        # หา block จาก heading ไปจนถึง heading ถัดไปหรือจบไฟล์
        m = re.search(
            rf'({re.escape(block)}.*?)(?=\n(?:#{1,3} |\Z))',
            text, re.DOTALL | re.IGNORECASE,
        )
        if m:
            found.append(m.group(1).strip()[:1500])
    return "\n\n".join(found)


def read_report_summary(output_dir: Path, agent_name: str, max_chars: int = 1200) -> str:
    """อ่าน report: header 1200 chars + structured blocks สำคัญทั้งหมด"""
    if not output_dir or not output_dir.exists():
        return ""
    reports = sorted(output_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not reports:
        return ""
    try:
        full = reports[0].read_text(encoding="utf-8")
        header = full[:max_chars]
        blocks = extract_key_blocks(full)
        if blocks and blocks not in header:
            return header + "\n\n--- Key Blocks ---\n" + blocks
        return header
    except Exception:
        return ""


def auto_extract_kb_learning(agent_name: str, result: str):
    """Extract 'วิธีใหม่ที่พบ' from Self-Improvement Report and save as DISCOVERY."""
    m = re.search(
        r'วิธีใหม่ที่พบ\s*[:：]\s*(.+?)(?=\nจะนำไปใช้|\nKnowledge|\Z)',
        result, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return
    new_method = m.group(1).strip()
    if any(skip in new_method for skip in ["ไม่พบ", "ไม่มี", "–", "-", "N/A", "none"]):
        return
    save_kb(agent_name, new_method[:400], entry_type="discovery")


def _read_project_target(project_dir: Path | None) -> str:
    """Return the Scout-declared target column for this project, if known."""
    if not project_dir:
        return ""
    profile_path = project_dir / "output" / "scout" / "dataset_profile.md"
    if not profile_path.exists():
        return ""
    m = re.search(
        r"target_column\s*:\s*([^\s,|]+)",
        profile_path.read_text(encoding="utf-8", errors="ignore"),
        re.IGNORECASE,
    )
    if not m:
        return ""
    target = m.group(1).strip()
    return "" if target.lower() == "unknown" else target


def _read_profile_field(profile_text: str, field_name: str) -> str:
    m = re.search(rf"^{re.escape(field_name)}\s*:\s*(.+?)\s*$",
                  profile_text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _problem_type_allows_missing_target(problem_type: str) -> bool:
    return problem_type.strip().lower() in {
        "clustering",
        "unsupervised",
        "segmentation",
        "association_rules",
        "anomaly_detection",
        "dataset_discovery",
    }


def _write_scout_profile_fallback(project_dir: Path | None, output_path: Path | None = None) -> bool:
    """Create dataset_profile.md from existing Scout artifacts when dispatch skipped it."""
    if not project_dir:
        return False
    scout_dir = project_dir / "output" / "scout"
    profile_path = scout_dir / "dataset_profile.md"
    if profile_path.exists() or not scout_dir.exists():
        return False

    source_path = output_path if output_path and output_path.exists() else None
    if not source_path:
        for name in ("scout_output.csv", "scout_sme_datasets.csv", "dataset_candidates.json"):
            candidate = scout_dir / name
            if candidate.exists():
                source_path = candidate
                break
    if not source_path:
        return False

    rows = "unknown"
    cols = "unknown"
    top_candidate = "unknown"
    selected_source = "unknown"
    selected_url = "unknown"
    license_text = "unknown"
    score = "unknown"
    notes = "Generated by orchestrator fallback because Scout did not write dataset_profile.md."

    try:
        if source_path.suffix.lower() == ".csv":
            import csv as _csv
            with open(source_path, newline="", encoding="utf-8-sig", errors="ignore") as fh:
                reader = _csv.DictReader(fh)
                fieldnames = reader.fieldnames or []
                first = next(reader, None)
                count = 1 if first else 0
                for _ in reader:
                    count += 1
            rows = str(count)
            cols = str(len(fieldnames))
            if first:
                top_candidate = first.get("name") or first.get("dataset") or "unknown"
                selected_source = first.get("source") or "unknown"
                selected_url = first.get("url") or "unknown"
                license_text = first.get("license") or "unknown"
                score = first.get("combined_score") or first.get("quality_score") or "unknown"
        elif source_path.name == "dataset_candidates.json":
            import json as _json
            data = _json.loads(source_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, list):
                rows = str(len(data))
                cols = str(len(data[0])) if data and isinstance(data[0], dict) else "unknown"
                if data and isinstance(data[0], dict):
                    first = data[0]
                    top_candidate = str(first.get("name") or "unknown")
                    selected_source = str(first.get("source") or "unknown")
                    selected_url = str(first.get("url") or "unknown")
                    license_text = str(first.get("license") or "unknown")
                    score = str(first.get("combined_score") or first.get("quality_score") or "unknown")
    except Exception as e:
        notes = f"Generated by orchestrator fallback; source parse warning: {e}"

    text = (
        "DATASET_PROFILE\n"
        "===============\n"
        f"rows         : {rows}\n"
        f"cols         : {cols}\n"
        "dtypes       : discovery_metadata\n"
        "missing      : unknown\n"
        "target_column: unknown\n"
        "problem_type : dataset_discovery\n"
        "recommended_scaling: not_applicable\n"
        f"source_file  : {source_path.name}\n"
        f"selected_candidate: {top_candidate}\n"
        f"selected_source: {selected_source}\n"
        f"selected_url : {selected_url}\n"
        f"license      : {license_text}\n"
        f"combined_score: {score}\n"
        f"notes        : {notes}\n"
    )
    profile_path.write_text(text, encoding="utf-8")
    return True


def _project_profile_field(project_dir: Path | None, field_name: str) -> str:
    if not project_dir:
        return ""
    profile_path = project_dir / "output" / "scout" / "dataset_profile.md"
    if not profile_path.exists():
        return ""
    return _read_profile_field(
        profile_path.read_text(encoding="utf-8", errors="ignore"),
        field_name,
    )


def _forbidden_model_column(col: str, target: str = "") -> bool:
    """Columns that must not appear as model features in a predictive handoff."""
    lc = col.strip().lower()
    target_lc = target.strip().lower()
    if target_lc and lc == target_lc:
        return False
    id_like_exact = {
        "customer_id", "user_id", "account_id", "client_id", "member_id",
        "transaction_id", "order_id", "record_id", "row_id",
    }
    return (
        lc in id_like_exact
        or lc.endswith("_id")
        or "target_encoded" in lc
        or lc.endswith("_target")
        or "post_period" in lc
        or "postperiod" in lc
        or "account_note" in lc
        or lc.endswith("_note")
        or "reason" in lc
        or lc in {"duration"}  # UCI Bank Marketing: post-call duration leakage for pre-call deployment.
    )


def _read_csv_shape(path: Path) -> tuple[int, int]:
    import pandas as _pd

    df_head = _pd.read_csv(str(path), nrows=1)
    with open(str(path), encoding="utf-8") as fh:
        rows = sum(1 for _ in fh) - 1
    return rows, df_head.shape[1]


def _read_column_roles(project_dir: Path | None) -> dict[str, object]:
    if not project_dir:
        return {}
    roles_path = project_dir / "output" / "dana" / "column_roles.json"
    if not roles_path.exists():
        return {}
    try:
        return json.loads(roles_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _role_columns(role_blob: dict[str, object], role_name: str) -> list[str]:
    roles = role_blob.get("roles", {})
    if not isinstance(roles, dict):
        return []
    return [str(c) for c, r in roles.items() if str(r).lower() == role_name.lower()]


def _has_supervised_target(role_blob: dict[str, object]) -> bool:
    target = role_blob.get("target_column")
    return bool(target and str(target).strip().lower() not in {"", "unknown", "none", "null"})


def _feature_guard_note(role_blob: dict[str, object]) -> str:
    if not role_blob:
        return "No Dana role metadata found."
    supervised = _has_supervised_target(role_blob)
    ids = _role_columns(role_blob, "id")
    dates = _role_columns(role_blob, "date")
    labels = _role_columns(role_blob, "label")
    return f"Dana roles: supervised={supervised}; id={ids[:12]}; date={dates[:12]}; label={labels[:12]}"


def _binary_target_signature(csv_path: Path, target: str) -> tuple[bool, float, set[str]]:
    """Return (has_target, positive_rate, unique_values_as_strings)."""
    import pandas as _pd

    if not target:
        return False, 0.0, set()
    df = _pd.read_csv(str(csv_path), usecols=lambda c: c == target)
    if target not in df.columns:
        return False, 0.0, set()
    s = df[target].dropna()
    values = {str(v).strip().lower() for v in s.unique()}
    num = _pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return True, float(num.mean()), values
    positives = {"1", "yes", "true", "y", ">50k", ">50k.", "subscribed", "churn"}
    return True, float(s.astype(str).str.strip().str.lower().isin(positives).mean()), values


def _score_like_columns(df) -> list[str]:
    return [
        c for c in df.select_dtypes(include="number").columns
        if any(k in c.lower() for k in ("f1", "auc", "accuracy", "precision", "recall"))
    ]


def validate_agent_output(agent_name: str, output_path: str,
                          project_dir: Path | None = None) -> tuple[bool, str]:
    """ตรวจสอบ output ของ agent — รวม gate ตาม CLAUDE.md spec"""
    if not output_path:
        return False, "ไม่มี output path"
    p = Path(output_path)
    if not p.exists():
        return False, f"ไฟล์ไม่มีอยู่: {p.name}"

    if agent_name == "scout":
        fallback_project = project_dir or (p.parent.parent.parent if len(p.parents) >= 3 else None)
        _write_scout_profile_fallback(fallback_project, p)

    if p.suffix == ".csv":
        try:
            import pandas as _pd
            df = _pd.read_csv(str(p), nrows=5)
            if df.shape[1] == 0:
                return False, "CSV ว่าง (0 columns)"
            with open(str(p), encoding="utf-8") as fh:
                full_rows = sum(1 for _ in fh) - 1
            if agent_name in ("scout", "dana", "eddie", "finn", "max") and full_rows < 20:
                if agent_name == "scout":
                    return False, (
                        f"{p.name} มีแค่ {full_rows} rows — Scout ต้องส่ง dataset จริงหรือ shortlist.md "
                        f"แต่ห้ามใช้ CSV placeholder/manifest แทน dataset output"
                    )
                return False, (f"{p.name} มีแค่ {full_rows} rows — "
                               f"อาจโหลดไฟล์ผิด (outlier_flags? ควรเป็น *_output.csv)")
            base_msg = f"{p.name} ({full_rows} rows, {df.shape[1]} cols)"
        except Exception as e:
            return False, f"อ่าน CSV ไม่ได้: {e}"
    elif p.suffix == ".md":
        size = p.stat().st_size
        if size < 50:
            return False, f"report เล็กเกินไป ({size} bytes)"
        base_msg = f"{p.name} ({size:,} bytes)"
    else:
        return True, p.name

    # ── Agent-specific gates (CLAUDE.md spec) ─────────────────────────────────
    proj = project_dir or (p.parent.parent.parent if p.suffix == ".csv" else None)

    if agent_name == "scout" and proj:
        profile_path = proj / "output" / "scout" / "dataset_profile.md"
        scout_report = proj / "output" / "scout" / "scout_report.md"
        problem_type = ""
        if scout_report.exists():
            txt = scout_report.read_text(encoding="utf-8", errors="ignore").lower()
            if "dataset_risk_register" not in txt:
                return False, "Scout gate FAIL: missing DATASET_RISK_REGISTER"
        if profile_path.exists():
            import re as _re
            profile_text = profile_path.read_text(encoding="utf-8", errors="ignore")
            problem_type = _read_profile_field(profile_text, "problem_type")
            target_column = _read_profile_field(profile_text, "target_column")
            # Supervised workflows need a target. Clustering/unsupervised profiles
            # legitimately have no target column.
            if (
                target_column.lower() == "unknown"
                and not _problem_type_allows_missing_target(problem_type)
            ):
                return False, "Scout gate FAIL: target_column=unknown — Scout ต้อง dispatch ใหม่"
            # rows < 1,000 gate ใช้เฉพาะ multi-table/join datasets
            # single-table datasets เช่น breast cancer (569 rows) ไม่ควรถูก block
            rows_match = _re.search(r"rows\s*:\s*([\d,]+)", profile_text)
            if rows_match:
                rows_val = int(rows_match.group(1).replace(",", ""))
                _join_signals = ("join", "merged", "tables:", "source_tables", "multi-table")
                _is_join = any(sig in profile_text.lower() for sig in _join_signals)
                if _is_join and rows_val < 1000:
                    return False, f"Scout gate FAIL: rows={rows_val} < 1,000 หลัง JOIN — อาจ join ผิด"
            # Olist dataset — target ต้องเป็น review_score
            _olist_signals = ("olist_orders", "order_reviews", "review_score", "olist_order_reviews")
            if any(sig in profile_text.lower() for sig in _olist_signals):
                m_target = _re.search(r"target_column:\s*(\S+)", profile_text)
                if m_target and m_target.group(1).lower() != "review_score":
                    return False, (f"Scout gate FAIL (Olist): target_column='{m_target.group(1)}' "
                                   f"ต้องเป็น 'review_score' สำหรับ Olist dataset")
        if p.suffix == ".csv":
            try:
                from anna_core.runner import scout_output_is_placeholder

                if not _problem_type_allows_missing_target(problem_type) and scout_output_is_placeholder(p):
                    return False, (
                        f"Scout gate FAIL: {p.name} ดูเหมือน placeholder/manifest ไม่ใช่ dataset จริง "
                        f"ให้ scout download หรือ assemble dataset จริงก่อน"
                    )
            except Exception as e:
                return False, f"Scout gate FAIL: ตรวจ placeholder ของ scout ไม่ได้: {e}"

    elif agent_name == "dana" and proj:
        scout_csv = proj / "output" / "scout" / "scout_output.csv"
        dana_report = proj / "output" / "dana" / "dana_report.md"
        if dana_report.exists():
            txt = dana_report.read_text(encoding="utf-8", errors="ignore").lower()
            if "data_quality_audit" not in txt:
                return False, "Dana gate FAIL: missing DATA_QUALITY_AUDIT"
        if scout_csv.exists() and p.suffix == ".csv":
            try:
                import pandas as _pd
                with open(str(scout_csv), encoding="utf-8") as fh:
                    scout_rows = sum(1 for _ in fh) - 1
                with open(str(p), encoding="utf-8") as fh:
                    dana_rows = sum(1 for _ in fh) - 1
                if scout_rows > 0:
                    loss_pct = (scout_rows - dana_rows) / scout_rows
                    if loss_pct > 0.20:
                        return False, (f"Dana gate FAIL: rows หาย {loss_pct:.0%} "
                                       f"({scout_rows:,} → {dana_rows:,}) เกิน 20%")
                # ตรวจ target column ไม่หาย
                target = _read_project_target(proj)
                _df_check = _pd.read_csv(str(p), nrows=5)
                if target:
                    if target not in _df_check.columns:
                        return False, f"Dana gate FAIL: target '{target}' หายออกจาก output"
                    flags_path = proj / "output" / "dana" / "outlier_flags.csv"
                    if flags_path.exists():
                        flags_head = _pd.read_csv(str(flags_path), nrows=1000)
                        if "column_name" in flags_head.columns:
                            flagged_cols = {
                                str(v).strip().lower()
                                for v in flags_head["column_name"].dropna().unique()
                            }
                            if target.lower() in flagged_cols:
                                return False, (
                                    f"Dana gate FAIL: target '{target}' ถูกใช้ใน outlier detection"
                                )
                    if "is_outlier" in _df_check.columns:
                        target_sig = _binary_target_signature(p, target)
                        if target_sig[0] and set(_df_check[target].dropna().astype(str).str.strip().str.lower().unique()) <= {"0", "1"}:
                            if "is_outlier" == target.lower():
                                return False, "Dana gate FAIL: is_outlier กลายเป็น target"
                key_cols = [c for c in _df_check.columns if c.lower() in {"customer_id", "client_id", "account_id", "user_id"}]
                for key in key_cols:
                    sample = _pd.read_csv(str(p), usecols=[key])
                    trimmed = sample[key].astype(str).str.strip()
                    dupes = trimmed.duplicated().sum()
                    unique_ratio = trimmed.nunique(dropna=True) / max(len(trimmed), 1)
                    if dupes and unique_ratio >= 0.95:
                        return False, (
                            f"Dana gate FAIL: key column '{key}' ยังมี duplicates หลัง trim ({dupes} rows)"
                        )
            except Exception as e:
                return False, f"Dana gate exception: {e}"

    elif agent_name == "eddie" and proj:
        report = proj / "output" / "eddie" / "eddie_report.md"
        if report.exists():
            txt = report.read_text(encoding="utf-8", errors="ignore").lower()
            if "business_eda_frame" not in txt:
                return False, "Eddie gate FAIL: missing BUSINESS_EDA_FRAME"
            for required_kw in ("pipeline_spec", "problem_type", "target_column"):
                if required_kw not in txt:
                    return False, f"Eddie gate FAIL: PIPELINE_SPEC ไม่มี '{required_kw}'"
            if "problem_type : unknown" in txt or "problem_type: unknown" in txt:
                return False, "Eddie gate FAIL: problem_type=unknown — Eddie ต้อง dispatch ใหม่"
            scout_profile = proj / "output" / "scout" / "dataset_profile.md"
            if scout_profile.exists():
                import re as _re
                scout_txt = scout_profile.read_text(encoding="utf-8", errors="ignore")
                m_scout = _re.search(r"target_column\s*:\s*(\S+)", scout_txt, _re.IGNORECASE)
                m_eddie = _re.search(r"target_column\s*:\s*(\S+)", txt, _re.IGNORECASE)
                if m_scout and m_eddie:
                    scout_target = m_scout.group(1).strip().lower()
                    eddie_target = m_eddie.group(1).strip().lower()
                    if scout_target != "unknown" and eddie_target != scout_target:
                        return False, (
                            f"Eddie gate FAIL: target_column mismatch "
                            f"(Scout='{scout_target}', Eddie='{eddie_target}')"
                        )

    elif agent_name == "finn" and proj and p.suffix == ".csv":
        try:
            import pandas as _pd
            target = _read_project_target(proj)
            role_blob = _read_column_roles(proj)
            df_head = _pd.read_csv(str(p), nrows=5)
            cols_lower = {c: c.lower() for c in df_head.columns}
            customer_key_cols = {"customer_id", "customer id"}
            rfm_like = {"recency_days", "frequency", "monetary"}
            aggregated_customer_table = rfm_like.issubset(set(cols_lower.values()))
            leak_cols = [
                c for c, lc in cols_lower.items()
                if _forbidden_model_column(c, target)
                and not (aggregated_customer_table and lc in customer_key_cols)
            ]
            if leak_cols:
                return False, f"Finn gate FAIL: leakage/id-like features in output: {leak_cols[:8]}"
            if _has_supervised_target(role_blob):
                forbidden_roles = set(_role_columns(role_blob, "id")) | set(_role_columns(role_blob, "date")) | set(_role_columns(role_blob, "label"))
                forbidden_lower = {c.lower() for c in forbidden_roles}
                supervised_leaks = [c for c in df_head.columns if c.lower() in forbidden_lower and c.lower() != str(target).lower()]
                if supervised_leaks:
                    return False, (
                        f"Finn gate FAIL: supervised task must exclude Dana roles id/date/label: "
                        f"{supervised_leaks[:8]}"
                    )
            if target:
                if target not in df_head.columns:
                    return False, f"Finn gate FAIL: target '{target}' หายออกจาก output"
                prev = proj / "output" / "eddie" / "eddie_output.csv"
                if prev.exists():
                    has_prev, prev_rate, prev_values = _binary_target_signature(prev, target)
                    has_now, now_rate, now_values = _binary_target_signature(p, target)
                    if has_prev and has_now:
                        if len(prev_values) <= 3 and len(now_values) <= 3 and abs(prev_rate - now_rate) > 0.02:
                            return False, (
                                f"Finn gate FAIL: target distribution changed "
                                f"({prev_rate:.4f} → {now_rate:.4f})"
                            )
                report = proj / "output" / "finn" / "finn_report.md"
                if report.exists():
                    txt = report.read_text(encoding="utf-8", errors="ignore")
                    if "feature_governance" not in txt.lower():
                        return False, "Finn gate FAIL: missing FEATURE_GOVERNANCE"
                    m = re.search(r"target column\s*:\s*([^\s`]+)", txt, re.IGNORECASE)
                    if m and m.group(1).strip().lower() != target.lower():
                        return False, (
                            f"Finn gate FAIL: target_column mismatch "
                            f"(Scout='{target}', Finn='{m.group(1).strip()}')"
                        )
                    selected = re.search(
                        r"selected features.*?(?:##\s*\d+\.|\Z)",
                        txt,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if selected and re.search(rf"[-*]\s*`?{re.escape(target)}`?\b", selected.group(0), re.IGNORECASE):
                        return False, f"Finn gate FAIL: target '{target}' ถูกเลือกเป็น feature"
        except Exception as e:
            return False, f"Finn gate exception: {e}"

    elif agent_name == "max" and proj:
        report_candidates = [
            proj / "output" / "max" / "max_report.md",
            proj / "output" / "max" / "mining_results.md",
            proj / "output" / "max" / "patterns_found.md",
        ]
        existing = [r for r in report_candidates if r.exists()]
        if not existing:
            return False, "Max gate FAIL: missing pattern report"
        if existing:
            txt = "\n".join(r.read_text(encoding="utf-8", errors="ignore").lower() for r in existing)
            if "pattern_validity" not in txt:
                return False, "Max gate FAIL: missing PATTERN_VALIDITY"

    elif agent_name == "mo" and proj:
        mo_dir = proj / "output" / "mo"
        report_candidates = [
            mo_dir / "model_results.md",
            mo_dir / "mo_report.md",
            mo_dir / "agent_report.md",
        ]
        existing_reports = [r for r in report_candidates if r.exists()]
        report = existing_reports[0] if existing_reports else report_candidates[0]
        comparison = proj / "output" / "mo" / "model_comparison.csv"
        suspect_msgs: list[str] = []
        try:
            if not existing_reports:
                suspect_msgs.append("missing Mo model report")
            mo_csvs = [comparison] if comparison.exists() else []
            for extra_csv in mo_dir.glob("*.csv"):
                if extra_csv not in mo_csvs:
                    mo_csvs.append(extra_csv)
            score_evidence_found = False
            for csv_path in mo_csvs:
                import pandas as _pd
                cmp_df = _pd.read_csv(str(csv_path))
                score_cols = _score_like_columns(cmp_df)
                if score_cols:
                    score_evidence_found = True
                    if (cmp_df[score_cols] >= 0.999).any().any():
                        suspect_msgs.append(f"perfect/near-perfect metric detected in {csv_path.name}")
            if existing_reports:
                txt = report.read_text(encoding="utf-8", errors="ignore").lower()
                if re.search(r"\b(winner|best model|algorithm selected)\s*:\s*none\b", txt):
                    suspect_msgs.append("model report selected None")
                if "n/a" in txt and ("test f1" in txt or "test auc" in txt or "cv score" in txt):
                    suspect_msgs.append("model report has N/A metrics")
                for token in ("target_encoded", "customer_id_target", "post_period", "account_note"):
                    if token in txt:
                        suspect_msgs.append(f"leakage feature mentioned: {token}")
                        break
                if any(term in txt for term in ("classification", "classifier", "binary", "multiclass")):
                    required_groups = {
                        "PR-AUC/Average Precision": ("pr-auc", "average precision"),
                        "positive-class metrics": ("positive-class", "positive class", "minority class"),
                        "threshold/cost-benefit": ("threshold", "expected value", "cost-benefit", "cost benefit"),
                        "calibration": ("calibration", "brier"),
                    }
                    missing = [
                        name for name, terms in required_groups.items()
                        if not any(term in txt for term in terms)
                    ]
                    if missing:
                        suspect_msgs.append("missing world-class classification evidence: " + ", ".join(missing))
                    temporal_terms = ("date", "month", "period", "macro", "campaign")
                    if any(term in txt for term in temporal_terms) and not any(
                        term in txt for term in ("time-based", "out-of-time", "oot", "temporal split")
                    ):
                        suspect_msgs.append("temporal/macro risk mentioned without time-based or OOT validation")
                metric_terms = ("f1", "auc", "accuracy", "precision", "recall", "rmse", "mae", "r2", "pr-auc")
                if any(term in txt for term in metric_terms):
                    score_evidence_found = True
            if not score_evidence_found:
                suspect_msgs.append("missing readable model metrics")
            if suspect_msgs:
                return False, "Mo gate FAIL: likely leakage — " + "; ".join(suspect_msgs)
        except Exception as e:
            return False, f"Mo gate exception: {e}"

    elif agent_name == "iris_eda" and proj:
        report_candidates = [
            proj / "output" / "iris_eda" / "iris_eda_report.md",
            proj / "output" / "iris_eda" / "analysis_bridge.md",
            proj / "output" / "iris_eda" / "business_insight.md",
        ]
        existing_reports = [r for r in report_candidates if r.exists()]
        if not existing_reports:
            return False, "Iris EDA gate FAIL: missing iris_eda_report.md"
        txt = existing_reports[0].read_text(encoding="utf-8", errors="ignore").lower()
        required_groups = {
            "business eda brief": ("business_eda_brief", "business eda brief"),
            "evidence": ("evidence", "finding", "pattern", "metric"),
            "follow-up": ("follow-up", "follow up", "next step", "handoff", "finn", "mo"),
            "risk/caveat": ("risk", "limitation", "caveat", "uncertainty", "conflict"),
            "confidence": ("confidence", "low", "medium", "high"),
        }
        missing = [
            name for name, terms in required_groups.items()
            if not any(term in txt for term in terms)
        ]
        if missing:
            return False, "Iris EDA gate FAIL: missing bridge rigor — " + ", ".join(missing)

    elif agent_name == "iris" and proj:
        report = proj / "output" / "iris" / "iris_report.md"
        if not report.exists():
            return False, "Iris gate FAIL: missing iris_report.md"
        txt = report.read_text(encoding="utf-8", errors="ignore").lower()
        required_groups = {
            "business decision brief": ("business_decision_brief", "business decision brief"),
            "business lever/KPI": ("business lever", "revenue", "cost", "risk", "retention", "conversion", "kpi"),
            "owner/action": ("owner", "team", "action"),
            "assumptions/risks": ("assumption", "assumptions", "risk", "risks"),
            "validation plan": ("validation plan", "a/b test", "pilot", "cohort", "causal"),
            "confidence": ("confidence", "low", "medium", "high"),
        }
        missing = [
            name for name, terms in required_groups.items()
            if not any(term in txt for term in terms)
        ]
        if missing:
            return False, "Iris gate FAIL: missing business rigor — " + ", ".join(missing)

    elif agent_name == "quinn" and proj:
        report = proj / "output" / "quinn" / "quinn_report.md"
        if not report.exists():
            return False, "Quinn gate FAIL: missing quinn_report.md"
        txt = report.read_text(encoding="utf-8", errors="ignore").lower()
        if "restart_cycle: yes" in txt or "verdict: unsatisfied" in txt or "status: fail" in txt:
            return False, "Quinn HARD GATE FAIL: QC verdict requires restart_cycle / restart required. Pipeline blocked until fixed."
        required = ("leakage", "overfitting", "drift", "calibration", "business_satisfaction")
        missing = [term for term in required if term not in txt]
        if missing:
            return False, "Quinn gate FAIL: missing QC evidence — " + ", ".join(missing)

    elif agent_name == "vera" and proj:
        report = proj / "output" / "vera" / "vera_report.md"
        if not report.exists():
            return False, "Vera gate FAIL: missing vera_report.md"
        txt = report.read_text(encoding="utf-8", errors="ignore").lower()
        if "visual_qc" not in txt:
            return False, "Vera gate FAIL: missing VISUAL_QC"
        charts_dir = proj / "output" / "vera" / "charts"
        if charts_dir.exists() and not list(charts_dir.glob("*.png")):
            return False, "Vera gate FAIL: charts/ exists but no PNG charts were produced"

    elif agent_name == "rex" and proj:
        quinn_report = proj / "output" / "quinn" / "quinn_report.md"
        if quinn_report.exists():
            txt = quinn_report.read_text(encoding="utf-8", errors="ignore").lower()
            if "restart_cycle: yes" in txt or "verdict: unsatisfied" in txt:
                return False, "Rex gate FAIL: Quinn failed, final success report is blocked"
        rex_reports = [
            proj / "output" / "rex" / "final_report.md",
            proj / "output" / "rex" / "executive_summary.md",
            proj / "output" / "rex" / "meeting_presentation.md",
        ]
        rex_dir = proj / "output" / "rex"
        existing_reports = [p for p in rex_reports if p.exists()]
        if rex_dir.exists():
            existing_reports.extend(sorted(rex_dir.glob("meeting_presentation*.md")))
        existing_reports = list(dict.fromkeys(existing_reports))
        if not existing_reports:
            return False, "Rex gate FAIL: missing final executive report"
        if existing_reports:
            txt = "\n".join(p.read_text(encoding="utf-8", errors="ignore").lower() for p in existing_reports)
            required_groups = {
                "business impact assumptions": ("assumption", "assumptions", "cost", "roi"),
                "production readiness": ("production readiness", "prototype", "monitoring", "retrain"),
                "validation limitations": ("time-based", "out-of-time", "oot", "validation limitation"),
            }
            missing = [
                name for name, terms in required_groups.items()
                if not any(term in txt for term in terms)
            ]
            if missing:
                return False, "Rex gate FAIL: missing executive rigor — " + ", ".join(missing)

    if proj:
        try:
            mark_output_current(p, proj)
        except Exception:
            pass
    return True, base_msg


# Maps each gated agent to the (prev_agent, expected_csv) it should rerun from.
_GATE_RERUN_FROM: dict[str, tuple[str, str]] = {
    "scout": ("",       ""),
    "dana":  ("scout",  "scout_output.csv"),
    "eddie": ("dana",   "dana_output.csv"),
    "iris_eda": ("eddie", "eddie_output.csv"),
    "finn":  ("eddie",  "eddie_output.csv"),
    "mo":    ("finn",   "engineered_data.csv"),
}


def _gate_recovery_hint(agent: str, msg: str, project_dir: Path | None) -> str:
    prev, csv_name = _GATE_RERUN_FROM.get(agent, ("", ""))
    if prev and csv_name and project_dir:
        src = project_dir / "output" / prev / csv_name
        if src.exists():
            return f"rerun {agent.upper()} จาก {src.name} ({src.parent.name}/)"
        return f"rerun {prev.upper()} ก่อน แล้วค่อย rerun {agent.upper()}"
    elif agent == "scout":
        problem_type = _project_profile_field(project_dir, "problem_type")
        if _problem_type_allows_missing_target(problem_type):
            return (
                "Scout เป็นงาน unsupervised/clustering — ไม่ต้องระบุ target_column. "
                "ถ้ายังเห็น gate fail นี้ ให้ restart orchestrator แล้ว /resume project"
            )
        if "DATASET_RISK_REGISTER" in msg:
            return "แก้ scout_report.md ให้มี DATASET_RISK_REGISTER แล้ว rerun Scout หรือ /resume project"
        if "target_column=unknown" in msg:
            return (
                "dispatch Scout ใหม่พร้อมกำหนดโจทย์ supervised ให้ชัด "
                "หรือเปลี่ยน problem_type เป็น clustering ถ้าเป็น RFM/segmentation"
            )
        if "placeholder" in msg.lower() or "manifest" in msg.lower() or "dataset จริง" in msg:
            return (
                "dispatch Scout ใหม่ — ให้ download/assemble dataset จริง, "
                "เขียน scout_output.csv เป็นข้อมูลจริง, dataset_profile.md ต้องมี target_column ชัดเจน และมี DATASET_RISK_REGISTER"
            )
        else:
            return "dispatch Scout ใหม่ — ระบุ target_column ให้ชัดเจน"
    else:
        return f"ตรวจสอบ output ของ {agent} แล้ว dispatch ใหม่"


def _safe_rel(path: Path, base: Path | None) -> str:
    try:
        return str(path.relative_to(base)) if base else str(path)
    except ValueError:
        return str(path)


def _expected_handoff_candidates(agent_name: str, project_dir: Path | None) -> list[Path]:
    if not project_dir:
        return []
    candidates: dict[str, list[Path]] = {
        "dana": [project_dir / "output" / "scout" / "scout_output.csv"],
        "eddie": [project_dir / "output" / "dana" / "dana_output.csv"],
        "iris_eda": [project_dir / "output" / "eddie" / "eddie_output.csv"],
        "finn": [project_dir / "output" / "eddie" / "eddie_output.csv"],
        "mo": [
            project_dir / "output" / "finn" / "engineered_data.csv",
            project_dir / "output" / "finn" / "finn_output.csv",
        ],
        "iris": [
            project_dir / "output" / "finn" / "engineered_data.csv",
            project_dir / "output" / "finn" / "finn_output.csv",
        ],
    }
    return candidates.get(agent_name, [])


def enforce_quinn_gate(agent_name: str, project_dir: Path | None, force_past_quinn: bool = False) -> bool:
    if agent_name not in {"iris", "vera", "rex"}:
        return True
    blocked, reason, target = quinn_hard_gate(project_dir)
    if not blocked:
        return True
    if force_past_quinn:
        print(f"{YL}  ⚠ --force-past-quinn override: {reason}{RST}")
        log_raw("system", f"--force-past-quinn override for {agent_name}: {reason}", task="quinn-gate")
        return True
    print(f"\n{RD}[PIPELINE BLOCKED]{RST} {reason}")
    print(f"{YL}Dispatch {target.upper()} first with fix.{RST}")
    _write_repair_note(agent_name, "quinn-restart-required", reason, project_dir, f"rerun @{target} with fix, then retry")
    return False


def _missing_handoff_message(agent_name: str, project_dir: Path | None) -> str:
    expected = _expected_handoff_candidates(agent_name, project_dir)
    expected_text = ", ".join(str(p) for p in expected) if expected else "(none)"
    prev, _csv = _GATE_RERUN_FROM.get(agent_name, ("", ""))
    rerun = f"rerun {prev.upper()} ก่อน แล้วค่อย rerun {agent_name.upper()}" if prev else f"rerun {agent_name.upper()}"
    return f"{agent_name.upper()} input missing: expected upstream output {expected_text}. Plan: {rerun}"


def _agent_report_candidates(agent: str, agent_dir: Path) -> list[Path]:
    names_by_agent = {
        "mo": ["model_results.md", "mo_report.md", "agent_report.md"],
        "max": ["max_report.md", "mining_results.md", "patterns_found.md"],
        "rex": ["meeting_presentation.md", "executive_summary.md", "final_report.md", "rex_report.md"],
    }
    names = names_by_agent.get(agent, [f"{agent}_report.md", "agent_report.md"])
    candidates = [agent_dir / name for name in names]
    candidates.extend(sorted(agent_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True))
    return list(dict.fromkeys(candidates))


def _write_repair_note(
    agent: str,
    kind: str,
    problem: str,
    project_dir: Path | None,
    rerun_hint: str,
    task: str = "",
    output_path: str = "",
    stderr: str = "",
) -> Path | None:
    if not project_dir:
        return None
    try:
        agent_dir = project_dir / "output" / agent
        logs_dir = project_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        agent_dir.mkdir(parents=True, exist_ok=True)

        script = find_agent_script(agent, project_dir)
        profile = project_dir / "output" / "scout" / "dataset_profile.md"
        report_candidates = _agent_report_candidates(agent, agent_dir)
        report = next((p for p in report_candidates if p.exists()), report_candidates[0])
        upstream_agent, upstream_csv = _GATE_RERUN_FROM.get(agent, ("", ""))
        upstream_path = project_dir / "output" / upstream_agent / upstream_csv if upstream_agent else Path()
        try:
            input_path = resolve_agent_input(agent, upstream_agent, project_dir, write_repair=False) if agent != "scout" else (scout_input_csv(project_dir) or "")
        except Exception:
            input_path = ""

        lines = [
            "# Latest Pipeline Repair",
            "",
            f"- kind: {kind}",
            f"- agent: {agent}",
            f"- task: {task or '(unknown)'}",
            f"- project: {project_dir}",
            f"- problem: {problem}",
            f"- plan: {rerun_hint}",
            f"- output: {output_path or '(none)'}",
            f"- input: {input_path or '(none)'}",
            f"- script: {script if script else '(none yet)'}",
            f"- report: {report}",
            f"- report_exists: {report.exists()}",
            f"- report_candidates: {', '.join(str(p) for p in report_candidates[:5])}",
            f"- profile: {profile if profile.exists() else '(none)'}",
        ]
        if upstream_agent:
            lines.append(f"- upstream_expected: {_safe_rel(upstream_path, project_dir)}")
            lines.append(f"- upstream_exists: {upstream_path.exists()}")
        if stderr:
            lines += ["", "## Error", "```text", stderr.strip()[:3000], "```"]
        lines += [
            "",
            "## Manual Recovery",
            f"1. Open/edit the script above or the profile/report listed above.",
            f"2. Re-run this agent with: @{agent} {rerun_hint}",
            "3. Or continue the project with: /resume " + project_dir.name,
            "4. If Codex changed orchestrator/source files while the app was open, restart orchestrator first.",
        ]
        text = "\n".join(lines) + "\n"
        note = logs_dir / "latest_repair.md"
        note.write_text(text, encoding="utf-8")
        (agent_dir / "REPAIR.md").write_text(text, encoding="utf-8")
        try:
            log_raw("system", f"repair note written: {note}", task="repair")
        except Exception:
            pass
        return note
    except Exception as e:
        try:
            log_raw("system", f"repair note write failed: {e}", task="repair")
        except Exception:
            pass
        return None


def _parse_repair_note_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^- ([^:]+):\s*(.*)$", line)
        if match:
            fields[match.group(1).strip().lower()] = match.group(2).strip()
    return fields


def _load_latest_repair_note(project_dir: Path | None) -> tuple[Path, str, dict[str, str]] | None:
    if not project_dir:
        return None
    note = project_dir / "logs" / "latest_repair.md"
    if not note.exists():
        return None
    text = note.read_text(encoding="utf-8", errors="ignore")
    return note, text, _parse_repair_note_fields(text)


def _print_gate_fail_recovery(agent: str, msg: str, project_dir: Path | None, task: str = "") -> None:
    """Print a clear, actionable recovery message when a pipeline gate fails."""
    rerun_hint = _gate_recovery_hint(agent, msg, project_dir)
    repair_note = _write_repair_note(agent, "gate", msg, project_dir, rerun_hint, task=task)

    try:
        bar = "═" * 53
        print(f"\n{RD}╔{bar}╗{RST}")
        print(f"{RD}║{RST}  {BLD}GATE FAIL — {agent.upper()}{RST}")
        print(f"{RD}╠{bar}╣{RST}")
        print(f"{RD}║{RST}  ปัญหา : {msg}")
        print(f"{RD}║{RST}  แผน   : {rerun_hint}")
        if repair_note:
            print(f"{RD}║{RST}  คู่มือ : {repair_note}")
        print(f"{RD}╚{bar}╝{RST}")
    except UnicodeEncodeError:
        print(f"\n*** GATE FAIL - {agent.upper()} ***")
        print(f"Problem: {msg}")
        print(f"Plan   : {rerun_hint}")
        if repair_note:
            print(f"Repair : {repair_note}")
    log_raw("system", f"gate FAIL recovery hint: {agent} — {msg} → {rerun_hint}", task="gate")


def anna_autofix_script(agent_name: str, task: str, script: Path,
                        input_path: str, output_dir: Path, stderr: str) -> tuple[str, bool]:
    """Anna ใช้ Claude วิเคราะห์และแก้ script เมื่อ DeepSeek retry ทั้งหมดล้มเหลว"""
    print(f"\n{YL}{'─'*55}{RST}")
    print(f"{YL}  ⟳ ANNA AUTO-FIX{RST}  {BLD}{agent_name.upper()}{RST}  (Claude กำลังวิเคราะห์...)")
    log_raw("anna", f"anna-autofix start: {agent_name}", task="anna-autofix")

    anna_kb = load_relevant_kb("anna", task, top_n=8) if task else load_kb("anna")
    anna_system = ANNA_SYSTEM + (f"\n\n---\n## Anna KB\n{anna_kb}" if anna_kb else "")

    try:
        rel_path = script.relative_to(BASE_DIR).as_posix()
    except ValueError:
        rel_path = script.name

    script_content = script.read_text(encoding="utf-8")
    fix_prompt = (
        f"Script ของ {agent_name} fail ทุก retry แล้ว ดิฉันต้องแก้ไขด้วยตัวเองในฐานะ Anna CEO\n\n"
        f"Script:\n```python\n{script_content[:3000]}\n```\n\n"
        f"Error:\n```\n{stderr[:1000]}\n```\n\n"
        f"Input path: {input_path}\nOutput dir: {output_dir}\nTask: {task}\n\n"
        f"วิเคราะห์ error และแก้ script ให้รันได้โดยใช้:\n"
        f'<WRITE_FILE path="{rel_path}">...fixed python code ทั้งไฟล์...</WRITE_FILE>\n\n'
        f"ตอบเป็น WRITE_FILE เท่านั้น ไม่ต้องอธิบาย"
    )

    anna_resp = call_claude(anna_system, fix_prompt, label=f"ANNA autofix {agent_name}")
    action_results = execute_anna_actions(anna_resp)

    if action_results and "[WRITE_FILE:" in action_results:
        print(f"{GR}  ✓ Anna เขียน script ใหม่ — รันทดสอบ...{RST}")
        output_path, returncode, _ = run_script(script, input_path, output_dir)
        if returncode == 0:
            print(f"{GR}  ✓ ANNA AUTO-FIX สำเร็จ!{RST}")
            log_raw("anna", f"anna-autofix SUCCESS: {agent_name}", task="anna-autofix")
            return output_path, True
        print(f"{RD}  ✗ Script ยังไม่สำเร็จหลัง Anna fix{RST}")
    else:
        print(f"{RD}  ✗ Anna ไม่ได้ส่ง WRITE_FILE กลับมา{RST}")

    log_raw("anna", f"anna-autofix FAILED: {agent_name}", task="anna-autofix")
    return str(output_dir), False


def resolve_agent_input(
    agent_name: str,
    prev_agent: str,
    project_dir: Path | None,
    task: str = "",
    write_repair: bool = True,
) -> str:
    expected_candidates = _expected_handoff_candidates(agent_name, project_dir)
    if expected_candidates:
        for candidate in expected_candidates:
            if candidate.exists():
                if agent_name == "dana":
                    print(f"{CY}  ⟳ Dana input → scout_output.csv (forced){RST}")
                    log_raw("system", "Dana input forced to scout_output.csv", task="dana")
                elif agent_name == "mo" and candidate.name == "engineered_data.csv":
                    print(f"{CY}  ⟳ Mo input → finn/engineered_data.csv (forced){RST}")
                    log_raw("system", "Mo input forced to finn/engineered_data.csv", task="mo")
                elif agent_name == "iris" and candidate.name in ("engineered_data.csv", "finn_output.csv"):
                    print(f"{CY}  ⟳ Iris input → finn/{candidate.name} (forced){RST}")
                    log_raw("system", f"Iris input forced to finn/{candidate.name}", task="iris")
                producer = "finn" if agent_name in {"mo", "iris"} else candidate.parent.name
                if project_dir and not output_is_current(candidate, project_dir):
                    stale_msg = (
                        f"STALE output detected for {candidate.name}; current run no longer accepts prior generation "
                        f"for {producer.upper()} → {agent_name.upper()}"
                    )
                    if write_repair:
                        _write_repair_note(agent_name, "stale-output", stale_msg, project_dir, f"rerun @{producer} or repair upstream output")
                    raise RuntimeError(stale_msg)
                if task.strip():
                    ok, schema_errors = validate_handoff(producer, agent_name, candidate, project_dir)
                    if not ok:
                        schema_msg = f"Schema contract FAIL: {producer.upper()} → {agent_name.upper()}\n" + "\n".join(schema_errors)
                        if write_repair:
                            _write_repair_note(agent_name, "schema-mismatch", schema_msg, project_dir, f"rerun @{producer} or check upstream output")
                        raise RuntimeError(schema_msg)
                    print(f"{GR}  ✓ Schema OK: {producer.upper()} → {agent_name.upper()}{RST}")
                    log_raw("system", "Schema validation passed", task=agent_name)
                return str(candidate)

        msg = _missing_handoff_message(agent_name, project_dir)
        if not write_repair:
            return ""
        if write_repair:
            prev, _csv = _GATE_RERUN_FROM.get(agent_name, ("", ""))
            hint = f"rerun {prev.upper()} ก่อน แล้วค่อย rerun {agent_name.upper()}" if prev else f"rerun {agent_name.upper()}"
            note = _write_repair_note(agent_name, "missing-output", msg, project_dir, hint, task=task)
            if note:
                print(f"{RD}  ✗ {msg}{RST}")
                print(f"{YL}  Repair note: {note}{RST}")
        raise RuntimeError(msg)

    # Dana ต้องใช้ scout_output.csv เสมอ — ห้าม fallback ไป input/ หรือ CSV อื่น
    if agent_name == "dana" and prev_agent == "scout" and project_dir:
        scout_csv = project_dir / "output" / "scout" / "scout_output.csv"
        if scout_csv.exists():
            if not output_is_current(scout_csv, project_dir):
                msg = "STALE scout output detected; rerun SCOUT before DANA"
                if write_repair:
                    _write_repair_note(agent_name, "stale-output", msg, project_dir, "rerun SCOUT แล้วค่อย rerun DANA", task=task)
                raise RuntimeError(msg)
            print(f"{CY}  ⟳ Dana input → scout_output.csv (forced){RST}")
            log_raw("system", f"Dana input forced to scout_output.csv", task="dana")
            return str(scout_csv)

    if agent_name == "iris_eda" and project_dir:
        eddie_csv = project_dir / "output" / "eddie" / "eddie_output.csv"
        if eddie_csv.exists():
            if not output_is_current(eddie_csv, project_dir):
                msg = "STALE Eddie output detected; rerun EDDIE before IRIS_EDA"
                if write_repair:
                    _write_repair_note(agent_name, "stale-output", msg, project_dir, "rerun EDDIE แล้วค่อย rerun IRIS_EDA", task=task)
                raise RuntimeError(msg)
            print(f"{CY}  ⟳ Iris EDA input → eddie_output.csv (forced){RST}")
            log_raw("system", "Iris EDA input forced to eddie_output.csv", task="iris_eda")
            if task.strip():
                ok, schema_errors = validate_handoff("eddie", agent_name, eddie_csv, project_dir)
                if not ok:
                    schema_msg = f"Schema contract FAIL: EDDIE → IRIS_EDA\n" + "\n".join(schema_errors)
                    if write_repair:
                        _write_repair_note(agent_name, "schema-mismatch", schema_msg, project_dir, "rerun EDDIE or check upstream output")
                    raise RuntimeError(schema_msg)
                print(f"{GR}  ✓ Schema OK: EDDIE → IRIS_EDA{RST}")
                log_raw("system", "Schema validation passed", task=agent_name)
            return str(eddie_csv)

    if agent_name == "mo" and project_dir:
        engineered_csv = project_dir / "output" / "finn" / "engineered_data.csv"
        if engineered_csv.exists():
            if not output_is_current(engineered_csv, project_dir):
                msg = "STALE Finn engineered_data output detected; rerun FINN before MO"
                if write_repair:
                    _write_repair_note(agent_name, "stale-output", msg, project_dir, "rerun FINN แล้วค่อย rerun MO", task=task)
                raise RuntimeError(msg)
            print(f"{CY}  ⟳ Mo input → finn/engineered_data.csv (forced){RST}")
            log_raw("system", "Mo input forced to finn/engineered_data.csv", task="mo")
            return str(engineered_csv)

    raw_input_path = pipeline_read(prev_agent) if prev_agent else ""
    input_path = resolve_input_path(prev_agent, raw_input_path, project_dir)
    if input_path != raw_input_path and input_path:
        print(f"{CY}  ⟳ input resolved:{RST} {DIM}{input_path}{RST}")
        log_raw("system", f"resolve input: {raw_input_path} → {input_path}", task=f"{agent_name}")

    if not input_path and project_dir:
        input_path = latest_input_file(project_dir)
        if input_path:
            label = "input sqlite fallback" if input_path.endswith(".sqlite") else "input fallback"
            print(f"{CY}  ⟳ {label}:{RST} {DIM}{input_path}{RST}")
            log_raw("system", f"{label} → {input_path}", task=agent_name)

    if input_path and input_path.endswith(".md") and project_dir:
        # ลองหา CSV จาก prev_agent โดยตรงก่อน — ป้องกันหยิบ CSV ผิด agent
        specific_csv: str = ""
        if prev_agent:
            candidate = project_dir / "output" / prev_agent / f"{prev_agent}_output.csv"
            if candidate.exists():
                specific_csv = str(candidate)
        fallback = specific_csv or latest_output_csv(project_dir)
        if fallback:
            old_input = input_path
            input_path = fallback
            src = f"{prev_agent}_output.csv" if specific_csv else "latest_output_csv"
            print(f"{YL}  ⟳ input .md → CSV fallback [{src}]:{RST} {DIM}{input_path}{RST}")
            log_raw("system", f"input .md fallback [{src}]: {old_input} → {input_path}", task=agent_name)

    if agent_name in ("vera", "rex") and input_path and input_path.endswith(".csv") and project_dir:
        try:
            import pandas as _pd

            _df = _pd.read_csv(input_path, nrows=5)
            if _df.shape[0] < 20 or _df.shape[1] < 5:
                for _ag in ["finn", "mo", "dana"]:
                    _candidate = project_dir / "output" / _ag / f"{_ag}_output.csv"
                    if _candidate.exists():
                        print(f"{YL}  ⟳ {agent_name} input too small → fallback to {_ag}_output.csv{RST}")
                        log_raw("system", f"{agent_name} input fallback: {input_path} → {_candidate}", task=agent_name)
                        input_path = str(_candidate)
                        break
        except Exception:
            pass
    if input_path and project_dir and infer_producer_from_path(input_path, project_dir):
        if not output_is_current(Path(input_path), project_dir):
            msg = f"STALE output detected: {Path(input_path).name} belongs to a prior run"
            if write_repair:
                _write_repair_note(agent_name, "stale-output", msg, project_dir, f"rerun @{prev_agent} or repair upstream output", task=task)
            raise RuntimeError(msg)
    if input_path and project_dir:
        producer = infer_producer_from_path(input_path, project_dir) or prev_agent
        if task.strip():
            ok, schema_errors = validate_handoff(producer, agent_name, input_path, project_dir)
            if not ok:
                schema_msg = f"Schema contract FAIL: {producer.upper()} → {agent_name.upper()}\n" + "\n".join(schema_errors)
                if write_repair:
                    _write_repair_note(agent_name, "schema-mismatch", schema_msg, project_dir, f"rerun @{producer} or check upstream output")
                raise RuntimeError(schema_msg)
            if producer:
                print(f"{GR}  ✓ Schema OK: {producer.upper()} → {agent_name.upper()}{RST}")
                log_raw("system", "Schema validation passed", task=agent_name)
    return input_path


def run_script_with_deepseek_autofix(
    agent_name: str,
    task: str,
    script: Path,
    input_path: str,
    output_dir: Path,
    max_retries: int = 15,
) -> tuple[str, int, str]:
    output_path = str(output_dir)
    returncode = -1
    stderr = ""
    for attempt in range(max_retries):
        output_path, returncode, stderr = run_script(script, input_path, output_dir)
        if returncode == 0:
            break
        if attempt < max_retries - 1:
            print(f"\n{YL}  ⟳ Script error (รอบ {attempt+1}/{max_retries}) → DeepSeek กำลังแก้ไข...{RST}")
            script_content = script.read_text(encoding="utf-8")
            fix_prompt = (
                f"Script นี้ error:\n\n```python\n{script_content[:3000]}\n```\n\n"
                f"Error:\n```\n{stderr[:800]}\n```\n\n"
                f"Input path: {input_path}\nOutput dir: {output_dir}\n\n"
                f"แก้ script ให้รันได้ ตอบเป็น python code block เดียวเท่านั้น"
            )
            fixed = call_deepseek(
                get_system_prompt(agent_name, task=task),
                fix_prompt,
                label=f"{agent_name.upper()} auto-fix #{attempt+1}",
            )
            blocks = extract_python_blocks(fixed)
            if blocks:
                script.write_text("\n\n".join(blocks), encoding="utf-8")
                print(f"{GR}  ✓ Script แก้แล้ว — รันใหม่...{RST}")
                log_raw("system", f"auto-fix {agent_name} script attempt {attempt+1}", task="auto-fix")
            else:
                print(f"{RD}  ✗ DeepSeek ไม่ส่ง code กลับมา — หยุด retry{RST}")
                break
        else:
            print(f"{RD}  ✗ Script ยังไม่สำเร็จหลัง {max_retries} รอบ → Anna auto-fix (Claude){RST}")
    return output_path, returncode, stderr


def handle_failed_script(
    agent_name: str,
    task: str,
    script: Path,
    input_path: str,
    output_dir: Path,
    stderr: str,
) -> tuple[str, bool]:
    output_path, success = anna_autofix_script(agent_name, task, script, input_path, output_dir, stderr)
    if success:
        return output_path, True

    project_dir = output_dir.parent.parent if output_dir else None
    rerun_hint = f"แก้ {script.name} แล้ว rerun @{agent_name} ด้วย input เดิม"
    repair_note = _write_repair_note(
        agent_name,
        "script",
        f"script failed after auto-fix: {script.name}",
        project_dir,
        rerun_hint,
        task=task,
        output_path=output_path,
        stderr=stderr,
    )

    print(f"\n{RD}{'─'*55}{RST}")
    print(f"{RD}  ✗ Auto-fix ทั้งหมด + Anna ล้มเหลว{RST}")
    print(f"{YL}  Anna ต้องการความช่วยเหลือจากคุณ{RST}")
    print(f"{YL}  Agent: {BLD}{agent_name.upper()}{RST}  |  Script: {script.name}{RST}")
    print(f"{YL}  Error: {DIM}{stderr[:200]}{RST}")
    if repair_note:
        print(f"{YL}  Repair note: {repair_note}{RST}")
    try:
        choice = input(
            f"  {BLD}ข้าม agent นี้ต่อ (skip) หรือหยุด pipeline (stop)? [skip/stop]:{RST} "
        ).strip().lower()
    except EOFError:
        choice = "skip"
    log_raw("anna", f"ask user: {agent_name} auto-fix failed — user chose {choice}", task="auto-fix")
    if choice == "stop":
        raise RuntimeError(f"{agent_name} script failed after all retries")
    return output_path, False


def complete_agent_handoff(agent_name: str, output_path: str, task: str, message: str) -> str:
    pipeline_write(agent_name, output_path)
    log_raw(agent_name, message, task=task, output=output_path)
    log_raw("system", f"pipeline handoff: {agent_name} → {output_path}", task="pipeline")
    if STATE.active_project and agent_name == "finn":
        companion = Path(output_path).parent / "engineered_data.csv"
        if companion.exists():
            try:
                mark_output_current(companion, STATE.active_project)
            except Exception:
                pass
    print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} done{RST}  {DIM}→ {output_path}{RST}")
    return output_path


def run_existing_script_agent(
    agent_name: str,
    task: str,
    script: Path,
    input_path: str,
    output_dir: Path,
) -> str:
    archive_path = archive_agent_outputs_before_rerun(Path(output_dir), agent_name)
    if archive_path:
        log_raw("system", f"Archived old outputs → {archive_path}", task=agent_name)
    if not validate_or_discard_script(script):
        fallback = write_builtin_script(agent_name, output_dir)
        if fallback:
            script = fallback
        else:
            return complete_agent_handoff(
                agent_name,
                str(output_dir),
                task,
                f"discarded unusable script {script.name}; no builtin fallback",
            )
    output_path, returncode, stderr = run_script_with_deepseek_autofix(
        agent_name, task, script, input_path, output_dir, max_retries=15
    )
    if returncode != 0:
        output_path, _success = handle_failed_script(agent_name, task, script, input_path, output_dir, stderr)

    if agent_name == "mo":
        sync_mo_canonical_report(output_dir, detect_mo_phase(task))
    report_summary = read_report_summary(output_dir, agent_name)
    action_msg = f"รัน script {script.name} สำเร็จ"
    if report_summary:
        action_msg += f"\n{report_summary}"
    return complete_agent_handoff(agent_name, output_path, task, action_msg)


def call_agent_llm(
    agent_name: str,
    task: str,
    system: str,
    message: str,
    discover: bool,
) -> str:
    if discover:
        result = call_claude(system, task, label=f"{agent_name.upper()} discover")
        first_para = result.strip().split("\n\n")[0][:400]
        save_kb(agent_name, f"Task: {task[:100]}\nKey finding: {first_para}", entry_type="discovery")
        return result
    return call_deepseek(system, message, label=f"{agent_name.upper()} execute")


def ensure_python_blocks(
    agent_name: str,
    task: str,
    system: str,
    input_path: str,
    output_dir: Path,
    result: str,
) -> list[str]:
    code_blocks = extract_python_blocks(result)
    if code_blocks:
        return code_blocks
    for retry in range(3):
        print(f"{YL}  ⟳ ไม่พบ Python code block — retry #{retry+1} (บังคับ code){RST}")
        force_msg = (
            f"คำตอบของคุณต้องมี Python code block เท่านั้น ห้ามตอบเป็น text\n"
            f"เขียน script ที่รันได้ทันที โดย:\n"
            f"- อ่านข้อมูลจาก INPUT_PATH (args.input)\n"
            f"- ประมวลผลตาม task\n"
            f"- save CSV ไปที่ OUTPUT_DIR (args.output_dir)\n"
            f"ตอบเป็น ```python ... ``` เท่านั้น ห้ามอธิบาย\n\n"
            f"Task: {task}\nInput: {input_path}\nOutput dir: {output_dir}"
        )
        forced = call_deepseek(system, force_msg, label=f"{agent_name.upper()} force-code #{retry+1}")
        code_blocks = extract_python_blocks(forced)
        if code_blocks:
            return code_blocks
    return []


def run_generated_script_agent(
    agent_name: str,
    task: str,
    input_path: str,
    output_dir: Path,
    report_path: Path,
    code_blocks: list[str],
    project_dir: Path | None,
) -> str:
    if not code_blocks:
        fallback = write_builtin_script(agent_name, output_dir)
        if fallback:
            py_path = fallback
            print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} builtin script saved — กำลังรัน...{RST}  {DIM}→ {py_path}{RST}")
            output_path, returncode, stderr = run_script_with_deepseek_autofix(
                agent_name, task, py_path, input_path, output_dir, max_retries=15
            )
            if returncode != 0:
                output_path, ok = handle_failed_script(agent_name, task, py_path, input_path, output_dir, stderr)
                if not ok:
                    output_path = str(report_path)
            return complete_agent_handoff(agent_name, output_path, task, f"builtin fallback script → {output_path}")
        return complete_agent_handoff(agent_name, str(report_path), task, f"no executable script blocks; used report {report_path.name}")
    py_path = output_dir / f"{agent_name}_script.py"
    py_path.write_text(_strip_python_fences("\n\n".join(code_blocks)), encoding="utf-8")
    if not validate_or_discard_script(py_path):
        return complete_agent_handoff(agent_name, str(report_path), task, f"discarded unusable generated script; used report {report_path.name}")
    print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} script saved — กำลังรัน...{RST}  {DIM}→ {py_path}{RST}")

    output_path, returncode, stderr = run_script_with_deepseek_autofix(
        agent_name, task, py_path, input_path, output_dir, max_retries=15
    )
    if returncode != 0:
        output_path, ok = handle_failed_script(agent_name, task, py_path, input_path, output_dir, stderr)
        if not ok:
            print(f"{RD}  ✗ Auto-fix ทั้งหมดล้มเหลว — ใช้ report แทน{RST}")
            output_path = str(report_path)

    if agent_name == "mo":
        sync_mo_canonical_report(output_dir, detect_mo_phase(task))
    if agent_name == "scout" and project_dir:
        output_path = latest_input_file(project_dir) or scout_input_csv(project_dir) or output_path

    return complete_agent_handoff(agent_name, output_path, task, f"รัน script (DeepSeek+run) → {output_path}")


def handle_report_only_agent(
    agent_name: str,
    task: str,
    report_path: Path,
    project_dir: Path | None,
) -> str:
    if agent_name == "scout" and project_dir:
        scout_csv = scout_input_csv(project_dir)
        if scout_csv:
            pipeline_write(agent_name, scout_csv)
            log_raw(agent_name, "พบ dataset ใน input/ — pipeline → CSV", task=task, output=scout_csv)
            log_raw("system", f"pipeline handoff: scout → {scout_csv}", task="pipeline")
            print(f"{GR}  ✓ {BLD}SCOUT{RST}{GR} dataset ready in input/{RST}  {DIM}→ {scout_csv}{RST}")
            return scout_csv

    pipeline_write(agent_name, str(report_path))
    log_raw(agent_name, "สร้าง report (DeepSeek)", task=task, output=str(report_path))
    log_raw("system", f"pipeline handoff: {agent_name} → {report_path}", task="pipeline")
    print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} report saved{RST}  {DIM}→ {report_path}{RST}")
    return str(report_path)


def run_agent(agent_name: str, task: str, prev_agent: str = "",
              project_dir: Path|None = None, discover: bool = False,
              force_past_quinn: bool = False) -> str:
    if not enforce_quinn_gate(agent_name, project_dir, force_past_quinn=force_past_quinn):
        return ""
    # CRISP-DM iteration guard — ป้องกัน infinite loop (lock ป้องกัน race ใน parallel)
    with _STATE_LOCK:
        STATE.agent_iter_count[agent_name] = STATE.agent_iter_count.get(agent_name, 0) + 1
        iter_count = STATE.agent_iter_count[agent_name]
    if iter_count > MAX_AGENT_ITER:
        print(f"\n{RD}  ✗ {BLD}{agent_name.upper()}{RST}{RD} ถึง max iterations ({MAX_AGENT_ITER}) — ข้าม CRISP-DM loop{RST}")
        log_raw("system", f"CRISP-DM loop guard: {agent_name} ถึง max {MAX_AGENT_ITER} iterations", task="loop-guard")
        return pipeline_read(agent_name) or ""

    iter_label = f" [{iter_count}/{MAX_AGENT_ITER}]" if iter_count > 1 else ""
    bar = "─" * max(0, 48 - len(agent_name) - len(iter_label))
    print(f"\n{CY}┌─ {BLD}{agent_name.upper()}{RST}{CY}{YL}{iter_label}{RST}{CY} {bar}┐{RST}")
    set_pipeline_stage(agent_name, "start", task)

    input_path = resolve_agent_input(agent_name, prev_agent, project_dir, task=task)
    output_dir = output_dir_for(project_dir, agent_name)

    # ── Priority 1: script จริง (+ auto-fix via DeepSeek ถ้า error) ──────────
    script = find_agent_script(agent_name, project_dir)
    if output_dir and not discover and builtin_agent_script(agent_name):
        script = None
    if script and output_dir and not discover:
        archive_path = archive_agent_outputs_before_rerun(Path(output_dir), agent_name)
        if archive_path:
            log_raw("system", f"Archived old outputs → {archive_path}", task=agent_name)
        if (
            agent_name == "scout"
            and scout_download_task_requires_fresh_script(task, project_dir)
            and not scout_script_handles_confirmed_download(script)
        ):
            print(f"{YL}  ⟳ Scout confirmed dataset load requested but input/ is empty — regenerating script{RST}")
            log_raw("system", f"Scout stale script skipped for confirmed load: {script}", task="scout-task-guard")
            script = None
        elif agent_name == "vera" and should_regenerate_vera_script_for_meeting_report(script, task, project_dir):
            print(f"{YL}  ⟳ Vera meeting report alignment requested/newer — regenerating script{RST}")
            log_raw("system", f"Vera script skipped for meeting report alignment: {script}", task="vera-meeting-guard")
            script = None
        if not script:
            pass
        elif agent_name != "mo" or mo_script_matches_phase(script, detect_mo_phase(task)):
            set_pipeline_stage(agent_name, "script-reuse", script.name)
            return run_existing_script_agent(agent_name, task, script, input_path, output_dir)
        else:
            mo_phase = detect_mo_phase(task)
            print(
                f"{YL}  ⟳ Mo Phase {mo_phase} requested but {script.name} is for another phase — regenerating script{RST}"
            )
            log_raw("system", f"Mo phase guard skipped stale script: {script}", task="mo-phase-guard")
    # ── ลบ script เก่าเฉพาะตอนจะสร้างใหม่ (Priority 1 ไม่ผ่าน) ──────────────
    delete_old_scripts(output_dir)

    if output_dir and not discover and not script:
        builtin_script = write_builtin_script(agent_name, output_dir)
        if builtin_script:
            set_pipeline_stage(agent_name, "script-reuse", builtin_script.name)
            return run_existing_script_agent(agent_name, task, builtin_script, input_path, output_dir)

    # ── Priority 2: LLM ───────────────────────────────────────
    system = get_system_prompt(agent_name, task=task)

    message = build_agent_path_message(agent_name, task, input_path, output_dir, project_dir)

    set_pipeline_stage(agent_name, "llm", "building response")
    result = call_agent_llm(agent_name, task, system, message, discover)

    # Auto-extract Self-Improvement discovery ไปเก็บ KB
    auto_extract_kb_learning(agent_name, result)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_agent_outputs_before_rerun(output_dir, agent_name)
        if archive_path:
            log_raw("system", f"Archived old outputs → {archive_path}", task=agent_name)

        report_path = output_dir / f"{agent_name}_report.md"
        report_path.write_text(result, encoding="utf-8")

        set_pipeline_stage(agent_name, "extract-code", "scan response")
        code_blocks = ensure_python_blocks(agent_name, task, system, input_path, output_dir, result)

        if code_blocks:
            set_pipeline_stage(agent_name, "run-code", f"{len(code_blocks)} block(s)")
            return run_generated_script_agent(
                agent_name, task, input_path, output_dir, report_path, code_blocks, project_dir
            )

        if agent_name == "iris_eda":
            set_pipeline_stage(agent_name, "report-only", report_path.name)
            return handle_report_only_agent(agent_name, task, report_path, project_dir)

        set_pipeline_stage(agent_name, "report-only", report_path.name)
        return handle_report_only_agent(agent_name, task, report_path, project_dir)

    log_raw(agent_name, result[:200], task=task)
    return result


def parse_dispatches(text: str) -> list[dict]:
    return DISPATCHER.parse_dispatches(text)

def parse_ask_user(text: str) -> str|None:
    return DISPATCHER.parse_ask_user(text)

def parse_ask_codex(text: str) -> str|None:
    return DISPATCHER.parse_ask_codex(text)


# ── Project Detection ─────────────────────────────────────────────────────────

def detect_project(text: str) -> Path|None:
    return PROJECT_DETECTOR.detect(text)


# ── Anna Full-Power Action Executor ──────────────────────────────────────────

def execute_anna_actions(response: str) -> str:
    return ACTION_EXECUTOR.execute(response)


def print_text_box(title: str, body: str, border_color: str = YL) -> None:
    print(f"\n{border_color}┌─ {title} ─────────────────────────────────────────────┐{RST}")
    print(f"{border_color}│{RST}  {body}")
    print(f"{border_color}└────────────────────────────────────────────────────┘{RST}")


def response_has_anna_actions(response: str) -> bool:
    action_tags = (
        "READ_FILE",
        "RUN_SHELL",
        "WRITE_FILE",
        "APPEND_FILE",
        "EDIT_FILE",
        "CREATE_DIR",
        "DELETE_FILE",
        "UPDATE_KB",
        "ASK_DEEPSEEK",
        "ASK_CLAUDE",
        "ASK_CODEX",
        "RESEARCH",
        "RUN_PYTHON",
    )
    return any(f"<{tag}" in response for tag in action_tags)


def summarize_action_results(
    anna_system: str,
    action_results: str,
    label: str = "ANNA",
    provider: str = "deepseek",
) -> str:
    followup = (
        "ผลลัพธ์จากการดำเนินการ:\n\n"
        f"{action_results}\n\n"
        "สรุปผลให้ผู้ใช้เป็นภาษาไทยเท่านั้น\n"
        "ห้ามส่ง action tags เพิ่ม เช่น <RUN_SHELL>, <RUN_PYTHON>, <READ_FILE>, <DISPATCH>\n"
        "ถ้าผลลัพธ์ไม่พอ ให้บอกว่าข้อมูลไม่พอและถามผู้ใช้เป็นภาษาไทยธรรมดา"
    )
    return call_model(provider, anna_system, followup, label=label, history=STATE.anna_history)


def execute_anna_actions_until_stable(
    anna_response: str,
    anna_system: str,
    user_input: str,
    max_rounds: int = 3,
    summary_provider: str = "deepseek",
) -> tuple[str, str]:
    all_results: list[str] = []
    current = anna_response
    for round_no in range(max_rounds):
        action_results = execute_anna_actions(current)
        if not action_results:
            return current, "\n\n".join(all_results)
        all_results.append(action_results)
        print(f"\n{CY}  ⟳ ส่งผลลัพธ์กลับให้ Anna...{RST}")
        STATE.anna_history.append({"role": "user", "content": user_input if round_no == 0 else "action follow-up"})
        STATE.anna_history.append({"role": "assistant", "content": current})
        current = summarize_action_results(anna_system, action_results, label="ANNA", provider=summary_provider)
        STATE.anna_history.append({"role": "user", "content": "action results summary request"})
        STATE.anna_history.append({"role": "assistant", "content": current})
        if not response_has_anna_actions(current):
            return current, "\n\n".join(all_results)
    return (
        "Anna หยุดการรัน action เพิ่มแล้วค่ะ เพราะระบบเจอ action ต่อเนื่องหลายรอบเกินไป "
        "กรุณาสั่งงานใหม่ให้ชัดเจนว่าต้องการให้ตรวจไฟล์หรือรันคำสั่งอะไร",
        "\n\n".join(all_results),
    )


# ── Anna Auto-Fix ─────────────────────────────────────────────────────────────

def _anna_autofix_response(
    original: str,
    user_input: str,
    anna_system: str,
    action_errors: str = "",
    provider: str = "codex",
) -> str:
    """Anna auto-fix: dispatch malformed หรือ action errors → Claude วิเคราะห์ใหม่
    (เหมือน agent script auto-fix แต่ใช้กับ planning ของ Anna เอง)
    """
    print(f"\n{YL}{'─'*55}{RST}")
    print(f"{YL}  ⟳ ANNA auto-fix{RST}  (Claude กำลังวิเคราะห์และแก้ไข...)")
    log_raw("anna", "anna-autofix start: planning error", task="anna-autofix")

    issues = []
    if "<DISPATCH>" in original and not parse_dispatches(original):
        issues.append("- dispatch tags มีอยู่แต่ JSON malformed หรือ agent name ไม่อยู่ใน valid list")
    if action_errors:
        issues.append(f"- action execution มี error:\n{action_errors[:400]}")

    fix_prompt = (
        f"Anna ตอบกลับมาแต่มีปัญหาต่อไปนี้:\n"
        + "\n".join(issues) + "\n\n"
        f"Response เดิม:\n```\n{original[:2000]}\n```\n\n"
        f"Task เดิมของ user: {user_input}\n\n"
        f"วิเคราะห์ปัญหาและตอบใหม่ให้ถูกต้อง "
        f"ถ้าต้อง dispatch ให้เรียก CREATE_DIR ก่อน DISPATCH เสมอสำหรับ project ใหม่, "
        f"ให้ task เป็น JSON string บรรทัดเดียวโดยไม่มี newline หรือ \\n, "
        f"และให้แน่ใจว่า JSON valid และ agent name ถูกต้อง "
        f"(valid agents: scout, dana, eddie, iris_eda, max, finn, mo, iris, vera, quinn, rex)"
    )

    fixed = call_model(provider, anna_system, fix_prompt, label="ANNA auto-fix")
    log_raw("anna", f"anna-autofix done: dispatches={len(parse_dispatches(fixed))}", task="anna-autofix")
    return fixed


def _anna_repair_dispatch_plan(
    anna_response: str,
    user_input: str,
    anna_system: str,
    issues: list[str],
    provider: str = "codex",
) -> str:
    print(f"\n{YL}{'─'*55}{RST}")
    print(f"{YL}  ⟳ ANNA plan validation{RST}  (repairing dispatch plan...)")
    log_raw("anna", f"dispatch validation issues: {'; '.join(issues)}", task="anna-plan-guard")
    repair_prompt = (
        "Anna's dispatch plan failed validation.\n\n"
        "Issues:\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\n\nOriginal user request:\n"
        + user_input[:1500]
        + "\n\nAnna response:\n```\n"
        + anna_response[:3000]
        + "\n```\n\n"
        "Rewrite the response so it satisfies the Anna Output Contract v3. "
        "If agent work is needed, include project CREATE_DIR or existing project reference before DISPATCH. "
        "For a new project, CREATE_DIR tags must appear before the first DISPATCH. "
        "Use valid one-line DISPATCH JSON only; task must be a single-line string with no newline characters or escaped \\n sequences. "
        "Use only valid agents: scout, dana, eddie, iris_eda, max, finn, mo, iris, vera, quinn, rex. "
        "If required context is missing, ASK_USER instead of guessing."
    )
    fixed = call_model(provider, anna_system, repair_prompt, label="ANNA plan repair")
    log_raw("anna", f"dispatch validation repaired: dispatches={len(parse_dispatches(fixed))}", task="anna-plan-guard")
    return fixed


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def load_agent_specs(user_input: str = "") -> str:
    return AGENT_SPEC_LOADER.load(user_input)


def run_pipeline(user_input: str):
    STATE.reset_pipeline()
    # PIPELINE cleared lazily in pipeline_clear() — only when project changes

    set_tab_title("⏳ Anna — กำลังรัน...")

    anna_kb = load_relevant_kb("anna", user_input, top_n=8) if user_input else load_kb("anna")
    projects_list = list_projects(PROJECTS_DIR)

    # Intent classifier — โหลด agent specs เฉพาะเมื่อต้องการ pipeline จริงๆ
    intent      = classify_intent(user_input)
    agent_specs = load_agent_specs(user_input) if intent == "pipeline" else ""
    if intent == "chat":
        print(f"{DIM}  [intent: chat — skip agent specs]{RST}")

    # Session memory — combine recent memory with query-relevant memory
    session_mem = SESSION_MEMORY.build_context(user_input, tail_chars=1600, top_n=4)

    route = TASK_ROUTER.route(user_input, intent=intent, context=f"{anna_kb}\n{session_mem}")
    anna_choice = MODEL_POLICY.choose("anna_primary", route)
    summary_choice = MODEL_POLICY.choose("anna_summary", route)
    repair_choice = MODEL_POLICY.choose("anna_repair", route)

    anna_system = build_anna_system_prompt(
        ANNA_SYSTEM + ANNA_OUTPUT_CONTRACT,
        anna_kb=anna_kb,
        session_mem=session_mem,
        projects_list=projects_list,
        agent_specs=agent_specs,
    )

    # Auto-consolidate KB ทุก 10 project
    project_count = len(list(PROJECTS_DIR.iterdir())) if PROJECTS_DIR.exists() else 0
    if project_count > 0 and project_count % 10 == 0:
        print(f"{DIM}  ⟳ KB consolidation (project #{project_count})...{RST}")
        for ag in VALID_AGENTS | {"anna"}:
            consolidate_kb(ag)

    print(f"\n{YL}{'═'*55}{RST}")
    anna_response = call_model(anna_choice.provider, anna_system, user_input, label="ANNA", history=STATE.anna_history)
    log_raw("User", user_input)
    log_raw("Anna", anna_response, task=f"รับคำสั่งจาก User และวางแผน dispatch [{anna_choice.provider}]")

    # เก็บ dispatch จาก response แรกก่อน — ป้องกันหายหลัง action execution
    first_dispatches = parse_dispatches(anna_response)

    # Execute full-power actions แล้ว feed ผลกลับให้ Anna
    anna_response, action_results = execute_anna_actions_until_stable(
        anna_response,
        anna_system,
        user_input,
        summary_provider=summary_choice.provider,
    )
    if not action_results:
        STATE.anna_history.append({"role": "user",      "content": user_input})
        STATE.anna_history.append({"role": "assistant", "content": anna_response})

    ask = parse_ask_user(anna_response)
    if ask:
        print_text_box("ANNA", ask, YL)
        ans = input(f"  {BLD}คุณ (y/n):{RST} ").strip().lower()
        if ans != "y":
            print(f"\n{YL}  ANNA:{RST} เข้าใจแล้วค่ะ หยุดการทำงาน")
            return

    codex = parse_ask_codex(anna_response)
    if codex:
        print_text_box("CODEX", codex, MG)

    dispatches = parse_dispatches(anna_response)

    # fallback: ถ้า response ที่ 2 ไม่มี dispatch → ใช้จาก response แรก
    if not dispatches and first_dispatches:
        print(f"\n{CY}  ⟳ dispatch จาก response แรก ({len(first_dispatches)} รายการ){RST}")
        dispatches = first_dispatches

    review = RESPONSE_REVIEWER.review_anna(
        user_input=user_input,
        anna_response=anna_response,
        intent=intent,
        dispatches=dispatches,
        plan_issues=[],
        action_results=action_results,
        route=route,
    )

    # ── Anna auto-fix (เหมือน agent script auto-fix) ──────────────────────────
    # trigger เมื่อ: (1) มี <DISPATCH> แต่ parse ไม่ได้  (2) action มี ERROR และยังไม่มี dispatch
    # ไม่ trigger ถ้ามี valid dispatches อยู่แล้ว (เช่น action เล็กๆ fail แต่ dispatch ดีอยู่)
    dispatch_attempted = "<DISPATCH>" in anna_response
    action_had_errors  = bool(action_results) and "ERROR" in action_results
    if (dispatch_attempted and not dispatches) or (action_had_errors and not dispatches):
        repair_provider = review.repair_provider if (review.findings or action_had_errors or dispatch_attempted) else repair_choice.provider
        fixed = _anna_autofix_response(
            anna_response, user_input, anna_system,
            action_errors=action_results if action_had_errors else "",
            provider=repair_provider,
        )
        if action_had_errors:
            new_acts = execute_anna_actions(fixed)
            # สร้าง summary เฉพาะกรณีที่ fixed ยังไม่มี dispatch (ไม่งั้นจะทับ dispatch ดี)
            if new_acts and not parse_dispatches(fixed):
                fp = f"ผลลัพธ์จากการดำเนินการ:\n\n{new_acts}\n\nโปรดสรุปและตอบผู้ใช้เป็นภาษาไทย"
                fixed = call_model(summary_choice.provider, anna_system, fp, label="ANNA auto-fix summary")
        anna_response = fixed
        STATE.anna_history[-1] = {"role": "assistant", "content": anna_response}
        dispatches = parse_dispatches(anna_response)

    if not dispatches:
        return

    # detect project จาก response เฉพาะกรณีที่ยังไม่มี STATE.active_project เท่านั้น
    # ไม่ override project ที่ user ตั้งไว้ผ่าน "project <name>" command
    if STATE.active_project is None:
        STATE.active_project = detect_project(anna_response)
    pipeline_clear()   # clear stale paths NOW — before validate reads pipeline
    if STATE.active_project is not None:
        PIPELINE.rebuild_from_project(STATE.active_project)

    plan_issues = validate_dispatch_plan(
        dispatches,
        active_project=STATE.active_project,
        read_pipeline=pipeline_read,
        source_text=anna_response,
    )
    if plan_issues:
        repair_provider = review.repair_provider if (review.findings or plan_issues) else repair_choice.provider
        fixed = _anna_repair_dispatch_plan(
            anna_response,
            user_input,
            anna_system,
            plan_issues,
            provider=repair_provider,
        )
        repaired_actions = execute_anna_actions(fixed)
        if repaired_actions:
            print(f"\n{CY}  ⟳ ส่งผลลัพธ์ plan repair กลับให้ Anna...{RST}")
            followup = f"ผลลัพธ์จากการดำเนินการ:\n\n{repaired_actions}\n\nโปรดตอบด้วย dispatch plan ที่ถูกต้องตาม Anna Output Contract v3"
            fixed = call_model(summary_choice.provider, anna_system, followup, label="ANNA plan repair summary", history=STATE.anna_history)
        anna_response = fixed
        STATE.anna_history[-1] = {"role": "assistant", "content": anna_response}
        if STATE.active_project is None:
            STATE.active_project = detect_project(anna_response)
            pipeline_clear()   # project resolved after repair — clear again if it changed
        if STATE.active_project is not None:
            PIPELINE.rebuild_from_project(STATE.active_project)
        dispatches = parse_dispatches(anna_response)
        plan_issues = validate_dispatch_plan(
            dispatches,
            active_project=STATE.active_project,
            read_pipeline=pipeline_read,
            source_text=anna_response,
        )
        if plan_issues:
            print(f"{RD}  ✗ Anna plan ยังไม่ผ่าน validation:{RST}")
            for issue in plan_issues:
                print(f"{RD}    - {issue}{RST}")
            ask = parse_ask_user(anna_response)
            if ask:
                print_text_box("ANNA", ask, YL)
            codex = parse_ask_codex(anna_response)
            if codex:
                print_text_box("CODEX", codex, MG)
            return
    if STATE.active_project is not None:
        run_id = begin_project_run(STATE.active_project, reason="pipeline")
        if dispatches:
            promote_upstream_outputs(STATE.active_project, dispatches[0].get("agent", ""), run_id)
    proj_name = STATE.active_project.name if STATE.active_project else "unknown"
    print(f"\n{CY}  ⟳ Pipeline:{RST} {BLD}{len(dispatches)} agent(s){RST}  {DIM}│ project: {proj_name}{RST}")

    prev_agent = ""
    completed  = []
    _stop_pipeline = False

    dispatch_groups = group_dispatches(dispatches)

    for grp_i, grp in enumerate(dispatch_groups):
        if not grp:
            continue

        # Step confirmation (ถามทีละกลุ่ม)
        if STEP_MODE and completed:
            label = grp[0].get("agent", "")
            if len(grp) > 1:
                label = "+".join(d.get("agent","") for d in grp)
            ans = confirm_next_step(prev_agent, label, grp[0].get("task",""),
                                    grp_i, len(dispatch_groups))
            if ans == "n":
                print(f"\n{YL}  หยุด pipeline ตามที่คุณสั่ง{RST}")
                _stop_pipeline = True
                break
            elif ans == "s":
                print(f"\n{YL}  ข้าม {BLD}{label.upper()}{RST}")
                continue

        if len(grp) > 1 and grp[0].get("parallel_group"):
            # ── Parallel execution ────────────────────────────────
            print(f"\n{CY}  ⟳ Parallel:{RST} {BLD}{'+'.join(d['agent'] for d in grp)}{RST}")
            _parallel_outputs: dict[str, str] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(grp)) as _pool:
                _futs = {
                    _pool.submit(run_agent, d["agent"], d["task"],
                                 prev_agent, COMMANDS.dispatch_project(d), d.get("discover", False)): d["agent"]
                    for d in grp
                }
                _fut_projects = {
                    fut: COMMANDS.dispatch_project(d)
                    for fut, d in zip(_futs.keys(), grp)
                }
                _fut_tasks = {
                    fut: d.get("task", "")
                    for fut, d in zip(_futs.keys(), grp)
                }
                for _f in concurrent.futures.as_completed(_futs):
                    _ag = _futs[_f]
                    _proj = _fut_projects.get(_f, STATE.active_project)
                    _task = _fut_tasks.get(_f, "")
                    try:
                        _out = _f.result()
                        _parallel_outputs[_ag] = _out
                        ok, msg = validate_agent_output(_ag, _out, _proj)
                        if not ok:
                            if _ag in ("scout", "dana", "eddie", "finn", "mo"):
                                _print_gate_fail_recovery(_ag, msg, _proj, task=_task)
                                _stop_pipeline = True
                            else:
                                print(f"{YL}  ⚠ {BLD}{_ag.upper()}{RST}{YL} output warning: {msg}{RST}")
                    except Exception as _e:
                        print(f"{RD}  ✗ parallel {_ag} error: {_e}{RST}")
                    completed.append(_ag)
            # prev_agent = agent ที่เขียน pipeline output จริง (ไม่ใช่แค่ตัวท้ายในลิสต์)
            _data_agents = [d["agent"] for d in grp
                            if pipeline_read(d["agent"]) and d["agent"] not in ("vera", "rex", "quinn", "iris_eda")]
            if _data_agents:
                prev_agent = _data_agents[-1]
            if _stop_pipeline:
                break
        else:
            # ── Sequential (เดิม) ─────────────────────────────────
            d       = grp[0]
            agent   = d.get("agent", "").lower()
            task    = d.get("task", "")
            discover = d.get("discover", False)
            if not agent or not task:
                continue
            current_project = COMMANDS.dispatch_project(d)
            if current_project and current_project != STATE.active_project:
                STATE.active_project = current_project
                global _last_pipeline_project
                _last_pipeline_project = current_project
            out = run_agent(agent, task, prev_agent=prev_agent,
                            project_dir=current_project, discover=discover)
            ok, msg = validate_agent_output(agent, out, current_project)
            if not ok:
                if agent in ("scout", "dana", "eddie", "finn", "mo"):
                    _print_gate_fail_recovery(agent, msg, current_project, task=task)
                    _stop_pipeline = True
                    break
                else:
                    print(f"{YL}  ⚠ {BLD}{agent.upper()}{RST}{YL} output warning: {msg}{RST}")

            # ── PIPELINE_SPEC guard: Eddie ต้องเขียน PIPELINE_SPEC ครบ ──────
            if agent == "eddie" and current_project:
                eddie_out_dir = current_project / "output" / "eddie"
                for _retry in range(2):
                    if check_pipeline_spec(eddie_out_dir):
                        break
                    print(f"\n{YL}  ⚠ Eddie report ขาด PIPELINE_SPEC — retry {_retry+1}/2{RST}")
                    log_raw("system", f"PIPELINE_SPEC missing — Eddie retry {_retry+1}", task="pipeline-guard")
                    out = run_agent(
                        "eddie",
                        task + " — บังคับเขียน PIPELINE_SPEC block ให้ครบ: problem_type, target_column, recommended_model, preprocessing, key_features",
                        prev_agent=prev_agent,
                        project_dir=current_project,
                    )
                else:
                    if not check_pipeline_spec(eddie_out_dir):
                        print(f"{RD}  ✗ Eddie ไม่เขียน PIPELINE_SPEC หลัง 2 retry — Anna จะต้องเดาค่าเอง{RST}")
                        log_raw("system", "PIPELINE_SPEC missing after 2 retries", task="pipeline-guard")

            completed.append(agent)
            if agent not in SIDE_CAR_AGENTS:
                prev_agent = agent

    if completed:
        print(f"\n{YL}{'═'*55}{RST}")
        last_path = pipeline_read(completed[-1])

        reports_block = collect_report_sections(
            completed,
            read_pipeline=pipeline_read,
            extract_key_blocks=extract_key_blocks,
            read_report_summary=read_report_summary,
        )
        # ตรวจสอบ CRISP-DM phase ที่เสร็จแล้ว
        completed_phases = list(dict.fromkeys(
            AGENT_TO_PHASE.get(a, "unknown") for a in completed
        ))
        iter_status = ", ".join(f"{a}×{n}" for a, n in STATE.agent_iter_count.items() if n > 1)

        summary_msg = build_summary_prompt(
            completed=completed,
            completed_phases=completed_phases,
            iter_status=iter_status,
            last_path=last_path,
            reports_block=reports_block,
        )
        summary = call_model(summary_choice.provider, anna_system, summary_msg, label="ANNA summary", history=STATE.anna_history)
        STATE.anna_history.append({"role": "user",      "content": summary_msg})
        STATE.anna_history.append({"role": "assistant", "content": summary})

        # บันทึก session memory
        save_session_memory(
            project_name=STATE.active_project.name if STATE.active_project else "unknown",
            agents_done=completed,
            summary_text=summary,
        )

        # Auto-continue: CRISP-DM loop — รันต่อเนื่อง (max 10 รอบ)
        if _stop_pipeline:
            return
        for _cont in range(10):
            cont_dispatches = parse_dispatches(summary)
            if not cont_dispatches:
                break
            print(f"\n{CY}  ⟳ Anna แนะนำ agent ถัดไป:{RST} {BLD}{len(cont_dispatches)} agent(s){RST}")
            _stop_cont = False
            for d in cont_dispatches:
                agent    = d.get("agent", "").lower()
                task     = d.get("task", "")
                discover = d.get("discover", False)
                if not agent or not task:
                    continue

                # Step confirmation ก่อนทุก agent ใน auto-continue
                if STEP_MODE:
                    ans = confirm_next_step(
                        prev_agent or "anna", agent, task,
                        len(completed) + 1, len(completed) + len(cont_dispatches),
                    )
                    if ans == "n":
                        print(f"\n{YL}  หยุด pipeline ตามที่คุณสั่ง{RST}")
                        _stop_cont = True
                        break
                    elif ans == "s":
                        print(f"\n{YL}  ข้าม {BLD}{agent.upper()}{RST}")
                        continue

                run_agent(agent, task, prev_agent=prev_agent,
                          project_dir=STATE.active_project, discover=discover)
                completed.append(agent)
                if agent not in SIDE_CAR_AGENTS:
                    prev_agent = agent
                print(f"\n{GR}  ✓ {BLD}{agent}{RST}{GR} เสร็จแล้ว  (total: {len(completed)}){RST}")

            if _stop_cont:
                return

            # อัปเดต summary หลังรัน batch ใหม่
            last_path = pipeline_read(completed[-1])
            cont_msg = (
                f"ทีมเพิ่มเติมที่เสร็จแล้ว: {', '.join(d.get('agent','') for d in cont_dispatches)}\n"
                f"Output ล่าสุด: {last_path}\n"
                f"รวม agent ทั้งหมดที่เสร็จ: {', '.join(completed)}\n"
                f"สรุปผลและระบุว่า pipeline เสร็จสมบูรณ์หรือต้องดำเนินการต่อ"
            )
            summary = call_model(summary_choice.provider, anna_system, cont_msg, label="ANNA summary", history=STATE.anna_history)
            STATE.anna_history.append({"role": "user",      "content": cont_msg})
            STATE.anna_history.append({"role": "assistant", "content": summary})


# ── Session Memory ────────────────────────────────────────────────────────────

def save_session_memory(project_name: str, agents_done: list, summary_text: str):
    SESSION_MEMORY.save(project_name, agents_done, summary_text)


# ── Logging ───────────────────────────────────────────────────────────────────

def log_raw(role: str, content: str, task: str = "", output: str = ""):
    RAW_LOGGER.log(role, content, task=task, output=output)


# ── CLI ───────────────────────────────────────────────────────────────────────

COMMANDS = AppCommands(
    state=STATE,
    cli=CLI,
    pipeline=PIPELINE,
    projects_dir=PROJECTS_DIR,
    palette=CommandPalette(
        reset=RST,
        bold=BLD,
        dim=DIM,
        cyan=CY,
        green=GR,
        yellow=YL,
        red=RD,
        blue=BL,
        magenta=MG,
        white=WH,
    ),
    set_tab_title=set_tab_title,
    notify_tab=notify_tab,
    run_pipeline=run_pipeline,
    run_agent=run_agent,
    validate_agent_output=validate_agent_output,
    print_gate_recovery=_print_gate_fail_recovery,
    write_repair_note=lambda agent, kind, problem, project_dir, rerun_hint: _write_repair_note(
        agent, kind, problem, project_dir, rerun_hint
    ),
    delete_old_scripts=delete_old_scripts,
    output_dir_for=output_dir_for,
    quinn_hard_gate=quinn_hard_gate,
    agent_output_newer_than=_agent_output_newer_than,
    load_kb=load_kb,
    load_relevant_kb=load_relevant_kb,
    save_kb=save_kb,
    call_claude=call_claude,
    anna_system=ANNA_SYSTEM,
    get_last_pipeline_project=lambda: _last_pipeline_project,
    set_last_pipeline_project=lambda project: globals().__setitem__("_last_pipeline_project", project),
    pipeline_clear=pipeline_clear,
)


def print_help():
    COMMANDS.print_help()


def print_latest_repair_note() -> None:
    COMMANDS.print_latest_repair_note()


def _run_latest_repair_note() -> None:
    COMMANDS._run_latest_repair_note()


def anna_discover(user_input: str):
    COMMANDS.anna_discover(user_input)


def read_cli_input() -> str | None:
    return COMMANDS.read_cli_input()


def _split_resume_args(raw: str) -> tuple[str, str]:
    return COMMANDS._split_resume_args(raw)


def resume_project(name: str = "", extra_instruction: str = "") -> None:
    COMMANDS.resume_project(name, extra_instruction)


def dispatch_project(d: dict) -> Path | None:
    return COMMANDS.dispatch_project(d)


def run_all_pipeline_command(extra_instruction: str = "") -> None:
    COMMANDS.run_all_pipeline_command(extra_instruction)


def _looks_like_run_all(user_input: str) -> bool:
    return AppCommands.looks_like_run_all(user_input)


def run_direct_agent_command(user_input: str) -> None:
    COMMANDS.run_direct_agent_command(user_input)


def _sync_command_hooks() -> None:
    COMMANDS.run_pipeline = run_pipeline
    COMMANDS.run_agent = run_agent
    COMMANDS.validate_agent_output = validate_agent_output
    COMMANDS.print_gate_recovery = _print_gate_fail_recovery
    COMMANDS.write_repair_note = _write_repair_note
    COMMANDS.delete_old_scripts = delete_old_scripts
    COMMANDS.output_dir_for = output_dir_for
    COMMANDS.quinn_hard_gate = quinn_hard_gate
    COMMANDS.agent_output_newer_than = _agent_output_newer_than
    COMMANDS.load_kb = load_kb
    COMMANDS.load_relevant_kb = load_relevant_kb
    COMMANDS.save_kb = save_kb
    COMMANDS.call_claude = call_claude
    COMMANDS.get_last_pipeline_project = lambda: _last_pipeline_project
    COMMANDS.set_last_pipeline_project = lambda project: globals().__setitem__("_last_pipeline_project", project)
    COMMANDS.pipeline_clear = pipeline_clear


def handle_cli_command(user_input: str) -> str:
    _sync_command_hooks()
    return COMMANDS.handle_cli_command(user_input)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    CLI.print_header()

    _mon = threading.Thread(target=_esc_monitor, daemon=True)
    _mon.start()

    _sync_command_hooks()
    COMMANDS.main_loop(COMMANDS.read_cli_input)


if __name__ == "__main__":
    main()
