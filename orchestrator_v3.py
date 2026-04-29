"""
DataScienceOS Orchestrator — Path-Based Pipeline v2
Pipeline ส่ง FILE PATH เท่านั้น ไม่ส่ง content
Agent ที่มี script → รัน script จริง (subprocess)
Agent ที่ไม่มี script → LLM สร้าง report + Python code
"""

import re
import os
import sys
import threading
import concurrent.futures
import time
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
)
from anna_core.actions import WorkspacePaths
from anna_core.action_executor import ActionExecutor, ActionPalette
from anna_core.anna_contract import ANNA_OUTPUT_CONTRACT, validate_dispatch_plan
from anna_core.cli_runtime import CliPalette, CliRenderer
from anna_core.config import load_config
from anna_core.dispatcher import DispatchParser
from anna_core.intent import IntentClassifier
from anna_core.kb import KnowledgeBase
from anna_core.logging import RawLogger, SessionMemoryStore
from anna_core.llm import LLMClient, TerminalPalette
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
from anna_core.runner import run_python_script
from anna_core.state import OrchestratorState

if sys.platform == "win32":
    import msvcrt

BASE_DIR = Path(__file__).parent
load_app_env(BASE_DIR / ".env")
CONFIG = load_config(BASE_DIR)

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

VALID_AGENTS = {"scout", "dana", "eddie", "max", "finn", "mo", "iris", "vera", "quinn", "rex"}

# ── CRISP-DM Phase Mapping ──────────────────────────────────────────────────
CRISP_DM_PHASES = {
    "data_understanding": ["scout", "eddie"],
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


# ── Intent Classifier ─────────────────────────────────────────────────────────

_PIPELINE_KW = {
    # agent names
    "scout", "dana", "eddie", "max", "finn", "mo", "iris", "vera", "quinn", "rex",
    # action words (Thai + English)
    "ให้", "dispatch", "รัน", "run", "ทำ", "วิเคราะห์", "cleaning", "eda",
    "pipeline", "dataset", "data", "โมเดล", "model", "train", "predict",
    "insight", "visualization", "report", "clean", "ข้อมูล", "สร้าง", "project",
    "csv", "excel", "xlsx", "json", "ml", "deep", "learning", "sklearn",
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
    return PIPELINE.read(agent_name)

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
        if scripts:
            return scripts[-1]
    return None

def run_script(script_path: Path, input_path: str, output_dir: Path) -> tuple[str, int, str]:
    """Returns (output_path, returncode, stderr)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitize_python_script_file(script_path)
    print(f"\n{CY}  ▶ SCRIPT{RST}  {BLD}{script_path.name}{RST}  {DIM}← {input_path or 'no input'}{RST}")
    print(f"{DIM}  กด ESC เพื่อหยุด script นี้{RST}")

    result = run_python_script(script_path, input_path, output_dir, STATE, timeout_seconds=300)

    if STATE.stop_requested.is_set():
        return str(output_dir), -999, "หยุดโดยผู้ใช้ (ESC)"

    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0:
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
        fake_err = "Script ran successfully but produced no CSV in OUTPUT_DIR. Must add: df.to_csv(os.path.join(OUTPUT_DIR, 'output.csv'), index=False)"
        return out_path, -1, fake_err
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
    if agent_name == "mo":
        mo_phase = detect_mo_phase(task)
        if mo_phase == 2:
            base += (
                "\n\n---\n## Mo Phase 2 Dispatch Guard (บังคับ)\n"
                "- งานนี้คือ Phase 2 Tune เท่านั้น ห้ามทำ Phase 1 Explore ซ้ำ\n"
                "- ต้องอ่านผล Phase 1 เดิมเพื่อหา best algorithm แล้ว tune เฉพาะ algorithm นั้น\n"
                "- ต้องใช้ RandomizedSearchCV จริง และบันทึก best_params_, best_score_, CV setup, train/test metrics\n"
                "- ต้องเปรียบเทียบ tuned model กับ default model ของ algorithm เดียวกัน\n"
                "- ต้องเขียน output/mo/mo_report.md ใหม่เป็น Phase 2 report ห้ามคง report Phase 1 เดิมไว้\n"
                "- ถ้า best algorithm จาก Phase 1 คือ Random Forest ให้ tune RandomForestClassifier เท่านั้น\n"
            )
        elif mo_phase == 3:
            base += (
                "\n\n---\n## Mo Phase 3 Dispatch Guard (บังคับ)\n"
                "- งานนี้คือ Phase 3 Validate เท่านั้น ห้ามทำ Phase 1 Explore หรือ Phase 2 Tune ซ้ำ\n"
                "- ต้อง validate tuned model จาก Phase 2 และเปรียบเทียบกับ default model\n"
                "- ต้องรายงาน final validation metrics, overfitting gap, leakage check, และเลือก final model\n"
                "- ต้องเขียน output/mo/mo_report.md ใหม่เป็น Phase 3 report ห้ามคง report Phase 1/2 เดิมไว้\n"
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
    rows = sum(1 for _ in open(str(path), encoding="utf-8")) - 1
    return rows, df_head.shape[1]


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

    if p.suffix == ".csv":
        try:
            import pandas as _pd
            df = _pd.read_csv(str(p), nrows=5)
            if df.shape[1] == 0:
                return False, "CSV ว่าง (0 columns)"
            full_rows = sum(1 for _ in open(str(p), encoding="utf-8")) - 1
            if agent_name in ("scout", "dana", "eddie", "finn", "max") and full_rows < 20:
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
        if profile_path.exists():
            import re as _re
            profile_text = profile_path.read_text(encoding="utf-8", errors="ignore")
            # target_column ห้ามเป็น unknown
            if "target_column: unknown" in profile_text:
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

    elif agent_name == "dana" and proj:
        scout_csv = proj / "output" / "scout" / "scout_output.csv"
        if scout_csv.exists() and p.suffix == ".csv":
            try:
                import pandas as _pd
                scout_rows = sum(1 for _ in open(str(scout_csv), encoding="utf-8")) - 1
                dana_rows  = sum(1 for _ in open(str(p), encoding="utf-8")) - 1
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
                    dupes = sample[key].astype(str).str.strip().duplicated().sum()
                    if dupes:
                        return False, (
                            f"Dana gate FAIL: key column '{key}' ยังมี duplicates หลัง trim ({dupes} rows)"
                        )
            except Exception as e:
                return False, f"Dana gate exception: {e}"

    elif agent_name == "eddie" and proj:
        report = proj / "output" / "eddie" / "eddie_report.md"
        if report.exists():
            txt = report.read_text(encoding="utf-8", errors="ignore").lower()
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
            df_head = _pd.read_csv(str(p), nrows=5)
            cols_lower = {c: c.lower() for c in df_head.columns}
            leak_cols = [
                c for c, lc in cols_lower.items()
                if _forbidden_model_column(c, target)
            ]
            if leak_cols:
                return False, f"Finn gate FAIL: leakage/id-like features in output: {leak_cols[:8]}"
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

    elif agent_name == "mo" and proj:
        report = proj / "output" / "mo" / "model_results.md"
        comparison = proj / "output" / "mo" / "model_comparison.csv"
        suspect_msgs: list[str] = []
        try:
            mo_csvs = [comparison] if comparison.exists() else []
            for extra_csv in (proj / "output" / "mo").glob("*.csv"):
                if extra_csv not in mo_csvs:
                    mo_csvs.append(extra_csv)
            for csv_path in mo_csvs:
                import pandas as _pd
                cmp_df = _pd.read_csv(str(csv_path))
                score_cols = _score_like_columns(cmp_df)
                if score_cols and (cmp_df[score_cols] >= 0.999).any().any():
                    suspect_msgs.append(f"perfect/near-perfect metric detected in {csv_path.name}")
            if report.exists():
                txt = report.read_text(encoding="utf-8", errors="ignore").lower()
                if re.search(r"\b(winner|best model|algorithm selected)\s*:\s*none\b", txt):
                    suspect_msgs.append("model report selected None")
                if "n/a" in txt and ("test f1" in txt or "test auc" in txt or "cv score" in txt):
                    suspect_msgs.append("model report has N/A metrics")
                for token in ("target_encoded", "customer_id_target", "post_period", "account_note"):
                    if token in txt:
                        suspect_msgs.append(f"leakage feature mentioned: {token}")
                        break
            if suspect_msgs:
                return False, "Mo gate FAIL: likely leakage — " + "; ".join(suspect_msgs)
        except Exception as e:
            return False, f"Mo gate exception: {e}"

    elif agent_name == "quinn" and proj:
        report = proj / "output" / "quinn" / "quinn_report.md"
        if report.exists():
            txt = report.read_text(encoding="utf-8", errors="ignore").lower()
            if "restart_cycle: yes" in txt or "verdict: unsatisfied" in txt or "status: fail" in txt:
                return False, "Quinn gate FAIL: QC verdict requires restart"

    elif agent_name == "rex" and proj:
        quinn_report = proj / "output" / "quinn" / "quinn_report.md"
        if quinn_report.exists():
            txt = quinn_report.read_text(encoding="utf-8", errors="ignore").lower()
            if "restart_cycle: yes" in txt or "verdict: unsatisfied" in txt:
                return False, "Rex gate FAIL: Quinn failed, final success report is blocked"

    return True, base_msg


# Maps each gated agent to the (prev_agent, expected_csv) it should rerun from.
_GATE_RERUN_FROM: dict[str, tuple[str, str]] = {
    "scout": ("",       ""),
    "dana":  ("scout",  "scout_output.csv"),
    "eddie": ("dana",   "dana_output.csv"),
    "finn":  ("eddie",  "eddie_output.csv"),
    "mo":    ("finn",   "engineered_data.csv"),
}


def _print_gate_fail_recovery(agent: str, msg: str, project_dir: Path | None) -> None:
    """Print a clear, actionable recovery message when a pipeline gate fails."""
    prev, csv_name = _GATE_RERUN_FROM.get(agent, ("", ""))
    if prev and csv_name and project_dir:
        src = project_dir / "output" / prev / csv_name
        if src.exists():
            rerun_hint = f"rerun {agent.upper()} จาก {src.name} ({src.parent.name}/)"
        else:
            rerun_hint = f"rerun {prev.upper()} ก่อน แล้วค่อย rerun {agent.upper()}"
    elif agent == "scout":
        rerun_hint = "dispatch scout ใหม่ — ระบุ target_column ให้ชัดเจน"
    else:
        rerun_hint = f"ตรวจสอบ output ของ {agent} แล้ว dispatch ใหม่"

    bar = "═" * 53
    print(f"\n{RD}╔{bar}╗{RST}")
    print(f"{RD}║{RST}  {BLD}GATE FAIL — {agent.upper()}{RST}")
    print(f"{RD}╠{bar}╣{RST}")
    print(f"{RD}║{RST}  ปัญหา : {msg}")
    print(f"{RD}║{RST}  แผน   : {rerun_hint}")
    print(f"{RD}╚{bar}╝{RST}")
    log_raw("system", f"gate FAIL recovery hint: {agent} — {msg} → {rerun_hint}", task="gate")


def check_pipeline_spec(output_dir: Path) -> bool:
    """ตรวจว่า eddie_report.md มี PIPELINE_SPEC block ครบหรือไม่"""
    if not output_dir or not output_dir.exists():
        return False
    reports = sorted(output_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not reports:
        return False
    text = reports[0].read_text(encoding="utf-8", errors="ignore")
    required = ["PIPELINE_SPEC", "problem_type", "recommended_model", "target_column"]
    return all(k.lower() in text.lower() for k in required)


def resolve_input_path(prev_agent: str, raw_path: str, project_dir: Path | None) -> str:
    """If Scout's pipeline points to a .md report, find actual CSV in project input/ instead."""
    if prev_agent != "scout" or not raw_path or not project_dir:
        return raw_path
    if not raw_path.endswith(".md"):
        return raw_path
    input_dir = project_dir / "input"
    if not input_dir.exists():
        return raw_path
    csvs = sorted(input_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(csvs[0]) if csvs else raw_path


def anna_autofix_script(agent_name: str, task: str, script: Path,
                        input_path: str, output_dir: Path, stderr: str) -> tuple[str, bool]:
    """Anna ใช้ Claude วิเคราะห์และแก้ script เมื่อ DeepSeek retry ทั้งหมดล้มเหลว"""
    print(f"\n{YL}{'─'*55}{RST}")
    print(f"{YL}  ⟳ ANNA AUTO-FIX{RST}  {BLD}{agent_name.upper()}{RST}  (Claude กำลังวิเคราะห์...)")
    log_raw("anna", f"anna-autofix start: {agent_name}", task="anna-autofix")

    anna_kb = load_kb("anna")
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


def resolve_agent_input(agent_name: str, prev_agent: str, project_dir: Path | None) -> str:
    # Dana ต้องใช้ scout_output.csv เสมอ — ห้าม fallback ไป input/ หรือ CSV อื่น
    if agent_name == "dana" and prev_agent == "scout" and project_dir:
        scout_csv = project_dir / "output" / "scout" / "scout_output.csv"
        if scout_csv.exists():
            print(f"{CY}  ⟳ Dana input → scout_output.csv (forced){RST}")
            log_raw("system", f"Dana input forced to scout_output.csv", task="dana")
            return str(scout_csv)

    if agent_name == "mo" and project_dir:
        engineered_csv = project_dir / "output" / "finn" / "engineered_data.csv"
        if engineered_csv.exists():
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

    print(f"\n{RD}{'─'*55}{RST}")
    print(f"{RD}  ✗ Auto-fix ทั้งหมด + Anna ล้มเหลว{RST}")
    print(f"{YL}  Anna ต้องการความช่วยเหลือจากคุณ{RST}")
    print(f"{YL}  Agent: {BLD}{agent_name.upper()}{RST}  |  Script: {script.name}{RST}")
    print(f"{YL}  Error: {DIM}{stderr[:200]}{RST}")
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
    print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} done{RST}  {DIM}→ {output_path}{RST}")
    return output_path


def run_existing_script_agent(
    agent_name: str,
    task: str,
    script: Path,
    input_path: str,
    output_dir: Path,
) -> str:
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
    py_path = output_dir / f"{agent_name}_script.py"
    py_path.write_text(_strip_python_fences("\n\n".join(code_blocks)), encoding="utf-8")
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
        output_path = scout_input_csv(project_dir) or output_path

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
              project_dir: Path|None = None, discover: bool = False) -> str:
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

    input_path = resolve_agent_input(agent_name, prev_agent, project_dir)
    output_dir = output_dir_for(project_dir, agent_name)

    # ── Priority 1: script จริง (+ auto-fix via DeepSeek ถ้า error) ──────────
    script = find_agent_script(agent_name, project_dir)
    if script and output_dir and not discover:
        mo_phase = detect_mo_phase(task) if agent_name == "mo" else None
        if agent_name != "mo" or mo_script_matches_phase(script, mo_phase):
            return run_existing_script_agent(agent_name, task, script, input_path, output_dir)
        print(
            f"{YL}  ⟳ Mo Phase {mo_phase} requested but {script.name} is for another phase — regenerating script{RST}"
        )
        log_raw("system", f"Mo phase guard skipped stale script: {script}", task="mo-phase-guard")

    # ── ลบ script เก่าเฉพาะตอนจะสร้างใหม่ (Priority 1 ไม่ผ่าน) ──────────────
    delete_old_scripts(output_dir)

    # ── Priority 2: LLM ───────────────────────────────────────
    system = get_system_prompt(agent_name, task=task)

    message = build_agent_path_message(agent_name, task, input_path, output_dir, project_dir)

    result = call_agent_llm(agent_name, task, system, message, discover)

    # Auto-extract Self-Improvement discovery ไปเก็บ KB
    auto_extract_kb_learning(agent_name, result)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"{agent_name}_report.md"
        report_path.write_text(result, encoding="utf-8")

        code_blocks = ensure_python_blocks(agent_name, task, system, input_path, output_dir, result)

        if code_blocks:
            return run_generated_script_agent(
                agent_name, task, input_path, output_dir, report_path, code_blocks, project_dir
            )

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


def summarize_action_results(anna_system: str, action_results: str, label: str = "ANNA") -> str:
    followup = (
        "ผลลัพธ์จากการดำเนินการ:\n\n"
        f"{action_results}\n\n"
        "สรุปผลให้ผู้ใช้เป็นภาษาไทยเท่านั้น\n"
        "ห้ามส่ง action tags เพิ่ม เช่น <RUN_SHELL>, <RUN_PYTHON>, <READ_FILE>, <DISPATCH>\n"
        "ถ้าผลลัพธ์ไม่พอ ให้บอกว่าข้อมูลไม่พอและถามผู้ใช้เป็นภาษาไทยธรรมดา"
    )
    return call_deepseek(anna_system, followup, label=label, history=STATE.anna_history)


def execute_anna_actions_until_stable(
    anna_response: str,
    anna_system: str,
    user_input: str,
    max_rounds: int = 3,
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
        current = summarize_action_results(anna_system, action_results, label="ANNA")
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

def _anna_autofix_response(original: str, user_input: str, anna_system: str,
                            action_errors: str = "") -> str:
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
        f"ถ้าต้อง dispatch ให้แน่ใจว่า JSON valid และ agent name ถูกต้อง "
        f"(valid agents: scout, dana, eddie, max, finn, mo, iris, vera, quinn, rex)"
    )

    fixed = call_claude(anna_system, fix_prompt, label="ANNA auto-fix")
    log_raw("anna", f"anna-autofix done: dispatches={len(parse_dispatches(fixed))}", task="anna-autofix")
    return fixed


def _anna_repair_dispatch_plan(
    anna_response: str,
    user_input: str,
    anna_system: str,
    issues: list[str],
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
        "Use valid DISPATCH JSON only. If required context is missing, ASK_USER instead of guessing."
    )
    fixed = call_claude(anna_system, repair_prompt, label="ANNA plan repair")
    log_raw("anna", f"dispatch validation repaired: dispatches={len(parse_dispatches(fixed))}", task="anna-plan-guard")
    return fixed


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def load_agent_specs(user_input: str = "") -> str:
    return AGENT_SPEC_LOADER.load(user_input)


def run_pipeline(user_input: str):
    STATE.reset_pipeline()
    # PIPELINE cleared lazily in pipeline_clear() — only when project changes

    set_tab_title("⏳ Anna — กำลังรัน...")

    anna_kb = load_kb("anna")
    projects_list = list_projects(PROJECTS_DIR)

    # Intent classifier — โหลด agent specs เฉพาะเมื่อต้องการ pipeline จริงๆ
    intent      = classify_intent(user_input)
    agent_specs = load_agent_specs(user_input) if intent == "pipeline" else ""
    if intent == "chat":
        print(f"{DIM}  [intent: chat — skip agent specs]{RST}")

    # Session memory — โหลด recent session summaries ให้ Anna จำได้ข้าม session
    session_mem = SESSION_MEMORY.tail(2000)

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
    anna_response = call_deepseek(anna_system, user_input, label="ANNA", history=STATE.anna_history)
    log_raw("User", user_input)
    log_raw("Anna", anna_response, task="รับคำสั่งจาก User และวางแผน dispatch")

    # เก็บ dispatch จาก response แรกก่อน — ป้องกันหายหลัง action execution
    first_dispatches = parse_dispatches(anna_response)

    # Execute full-power actions แล้ว feed ผลกลับให้ Anna
    anna_response, action_results = execute_anna_actions_until_stable(
        anna_response,
        anna_system,
        user_input,
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

    # ── Anna auto-fix (เหมือน agent script auto-fix) ──────────────────────────
    # trigger เมื่อ: (1) มี <DISPATCH> แต่ parse ไม่ได้  (2) action มี ERROR และยังไม่มี dispatch
    # ไม่ trigger ถ้ามี valid dispatches อยู่แล้ว (เช่น action เล็กๆ fail แต่ dispatch ดีอยู่)
    dispatch_attempted = "<DISPATCH>" in anna_response
    action_had_errors  = bool(action_results) and "ERROR" in action_results
    if (dispatch_attempted and not dispatches) or (action_had_errors and not dispatches):
        fixed = _anna_autofix_response(
            anna_response, user_input, anna_system,
            action_errors=action_results if action_had_errors else "",
        )
        if action_had_errors:
            new_acts = execute_anna_actions(fixed)
            # สร้าง summary เฉพาะกรณีที่ fixed ยังไม่มี dispatch (ไม่งั้นจะทับ dispatch ดี)
            if new_acts and not parse_dispatches(fixed):
                fp = f"ผลลัพธ์จากการดำเนินการ:\n\n{new_acts}\n\nโปรดสรุปและตอบผู้ใช้เป็นภาษาไทย"
                fixed = call_deepseek(anna_system, fp, label="ANNA auto-fix summary")
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
    )
    if plan_issues:
        fixed = _anna_repair_dispatch_plan(anna_response, user_input, anna_system, plan_issues)
        repaired_actions = execute_anna_actions(fixed)
        if repaired_actions:
            print(f"\n{CY}  ⟳ ส่งผลลัพธ์ plan repair กลับให้ Anna...{RST}")
            followup = f"ผลลัพธ์จากการดำเนินการ:\n\n{repaired_actions}\n\nโปรดตอบด้วย dispatch plan ที่ถูกต้องตาม Anna Output Contract v3"
            fixed = call_deepseek(anna_system, followup, label="ANNA plan repair summary", history=STATE.anna_history)
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
                                 prev_agent, dispatch_project(d), d.get("discover", False)): d["agent"]
                    for d in grp
                }
                _fut_projects = {
                    fut: dispatch_project(d)
                    for fut, d in zip(_futs.keys(), grp)
                }
                for _f in concurrent.futures.as_completed(_futs):
                    _ag = _futs[_f]
                    _proj = _fut_projects.get(_f, STATE.active_project)
                    try:
                        _out = _f.result()
                        _parallel_outputs[_ag] = _out
                        ok, msg = validate_agent_output(_ag, _out, _proj)
                        if not ok:
                            if _ag in ("scout", "dana", "eddie", "finn", "mo"):
                                _print_gate_fail_recovery(_ag, msg, _proj)
                                _stop_pipeline = True
                            else:
                                print(f"{YL}  ⚠ {BLD}{_ag.upper()}{RST}{YL} output warning: {msg}{RST}")
                    except Exception as _e:
                        print(f"{RD}  ✗ parallel {_ag} error: {_e}{RST}")
                    completed.append(_ag)
            # prev_agent = agent ที่เขียน pipeline output จริง (ไม่ใช่แค่ตัวท้ายในลิสต์)
            _data_agents = [d["agent"] for d in grp
                            if pipeline_read(d["agent"]) and d["agent"] not in ("vera", "rex", "quinn")]
            prev_agent = _data_agents[-1] if _data_agents else grp[-1]["agent"]
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
            current_project = dispatch_project(d)
            if current_project and current_project != STATE.active_project:
                STATE.active_project = current_project
                global _last_pipeline_project
                _last_pipeline_project = current_project
            out = run_agent(agent, task, prev_agent=prev_agent,
                            project_dir=current_project, discover=discover)
            ok, msg = validate_agent_output(agent, out, current_project)
            if not ok:
                if agent in ("scout", "dana", "eddie", "finn", "mo"):
                    _print_gate_fail_recovery(agent, msg, current_project)
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
        summary = call_deepseek(anna_system, summary_msg, label="ANNA summary", history=STATE.anna_history)
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
            summary = call_deepseek(anna_system, cont_msg, label="ANNA summary", history=STATE.anna_history)
            STATE.anna_history.append({"role": "user",      "content": cont_msg})
            STATE.anna_history.append({"role": "assistant", "content": summary})


# ── Session Memory ────────────────────────────────────────────────────────────

def save_session_memory(project_name: str, agents_done: list, summary_text: str):
    SESSION_MEMORY.save(project_name, agents_done, summary_text)


# ── Logging ───────────────────────────────────────────────────────────────────

def log_raw(role: str, content: str, task: str = "", output: str = ""):
    RAW_LOGGER.log(role, content, task=task, output=output)


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_help():
    print(
        f"\n{CY}Slash commands:{RST}\n"
        f"  {BLD}/project <name>{RST}       เลือกโปรเจกต์  (alias: /p, /proj)\n"
        f"  {BLD}/resume [name] [task]{RST} ทำต่อจากโปรเจกต์/active project  (alias: /r)\n"
        f"  {BLD}/status{RST}               ดูสถานะ pipeline  (alias: /s)\n"
        f"  {BLD}/kb <agent>{RST}           ดู knowledge base\n"
        f"  {BLD}/claude{RST}               ดู Codex usage\n"
        f"  {BLD}/end{RST}                  reset session\n"
        f"  {BLD}/exit{RST}                 ออกจากระบบ\n"
    )
    CLI.print_help()


def anna_discover(user_input: str):
    anna_kb = load_kb("anna")
    system  = ANNA_SYSTEM + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
    result  = call_claude(system, user_input, label="ANNA discover")
    save_kb("anna", f"Task: {user_input}\nDiscovery:\n{result}")


def read_cli_input() -> str | None:
    try:
        proj = f" {DIM}[{STATE.active_project.name}]{RST}" if STATE.active_project else ""
        proj_title = f" [{STATE.active_project.name}]" if STATE.active_project else ""
        set_tab_title(f"🟢 Anna{proj_title} — พร้อม")
        return input(f"{BLD}{WH}คุณ{RST}{proj}{BLD}{WH}:{RST} ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{YL}  ลาก่อนค่ะ{RST}")
        return None


def _split_resume_args(raw: str) -> tuple[str, str]:
    """Split `resume <project> <extra instruction>` without requiring quotes."""
    raw = raw.strip()
    if not raw:
        return "", ""
    if PROJECTS_DIR.exists():
        project_names = sorted(
            [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()],
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


def resume_project(name: str = "", extra_instruction: str = "") -> None:
    if not name and STATE.active_project:
        project = STATE.active_project
        message = ""
    else:
        project, message = CLI.resolve_project(name)
    if not project:
        color = YL if message.startswith("พบหลาย") else RD
        print(f"{color}  {message}{RST}")
        return
    STATE.active_project = project
    PIPELINE.rebuild_from_project(project)
    global _last_pipeline_project
    _last_pipeline_project = project   # prevent pipeline_clear() from wiping the rebuild
    done = PIPELINE.completed_agents()
    print(f"\n{YL}  Resume:{RST} {BLD}{project.name}{RST}")
    print(f"  เสร็จแล้ว: {GR}{', '.join(done) or 'ไม่มี'}{RST}")
    resume_msg = (
        f"Resume project {project.name}. "
        f"Agents ที่เสร็จแล้ว: {', '.join(done) or 'ไม่มี'}. "
        f"วิเคราะห์ว่าต้องทำอะไรต่อใน CRISP-DM pipeline แล้ว dispatch ต่อทันที"
    )
    if extra_instruction:
        resume_msg += f"\n\nคำสั่งเพิ่มเติมจาก user:\n{extra_instruction}"
    try:
        run_pipeline(resume_msg)
    except KeyboardInterrupt:
        print(f"\n{YL}  หยุด resume pipeline{RST}")


def dispatch_project(d: dict) -> Path | None:
    """Resolve optional project field from a DISPATCH object."""
    raw = str(d.get("project", "") or "").strip()
    if not raw:
        return STATE.active_project
    project, message = CLI.resolve_project(raw)
    if project:
        return project
    print(f"{YL}  ⚠ dispatch project ignored: {message}{RST}")
    return STATE.active_project


def run_all_pipeline_command(extra_instruction: str = "") -> None:
    project = STATE.active_project
    if not project:
        print(f"{RD}  ยังไม่ได้เลือก project{RST}")
        print("  ใช้: /project <name>")
        return
    input_dir = project / "input"
    if not input_dir.exists() or not any(input_dir.glob("*")):
        print(f"{RD}  Project นี้ไม่มี input data: {input_dir}{RST}")
        return

    PIPELINE.clear()
    global _last_pipeline_project
    _last_pipeline_project = project
    STATE.reset_pipeline()

    print(f"\n{CY}  RUN-ALL:{RST} {BLD}{project.name}{RST}")
    print(f"{DIM}  mode: deterministic sequence, no Anna planning prompt{RST}")

    blind_rule = (
        "ห้ามอ่าน answer_key ระหว่างทำงาน. "
        "ให้ตัดสินใจเองตามหน้าที่ agent และบันทึก output/report ของตัวเองให้ครบ. "
        "ถ้า input มีหลายไฟล์/หลายชั้น folder ให้เลือกไฟล์ข้อมูลหลักที่เหมาะสมเอง. "
        "ถ้า CSV ไม่ใช่ comma delimiter ให้ detect delimiter เอง เช่น sep=None, engine='python'."
    )
    if extra_instruction:
        blind_rule += " " + extra_instruction

    sequence: list[tuple[str, str]] = [
        ("scout", "เริ่ม pipeline จากข้อมูลใน input/ ของ project นี้ ตรวจไฟล์ทั้งหมด เลือก dataset หลัก สร้าง scout_output.csv และ dataset_profile.md. " + blind_rule),
        ("dana", "ทำ data cleaning จาก Scout output สร้าง dana_output.csv และ dana_report.md. ห้ามใช้ target ใน outlier detection และห้ามลบ target. " + blind_rule),
        ("eddie", "ทำ EDA จาก Dana output หา pattern/relationships และเขียน PIPELINE_SPEC ให้ครบ สร้าง eddie_output.csv และ eddie_report.md. " + blind_rule),
        ("finn", "ทำ feature engineering/feature selection จาก Eddie output สร้าง finn_output.csv และ finn_report.md. ใช้ target จาก Scout เท่านั้น เก็บ target เป็น label ห้ามเลือก target เป็น feature. " + blind_rule),
        ("mo", "train และ compare models จาก Finn output สร้าง mo_output.csv, model report และ metrics. ถ้า F1/AUC/Accuracy ใกล้ 1.0 ให้ถือว่าอาจ leakage และรายงาน fail. " + blind_rule),
        ("quinn", "ตรวจ QC/model/data/business satisfaction จากผลก่อนหน้า สร้าง quinn_output.csv และ quinn_report.md. ต้องตรวจ target consistency, leakage columns, perfect metrics, และ report/CSV contradiction. " + blind_rule),
        ("iris", "สรุป business insights/action recommendations จากผล pipeline สร้าง iris_output.csv และ iris_report.md. " + blind_rule),
        ("vera", "สร้าง visualization/report ที่เหมาะสมจากผล pipeline สร้าง vera_output.csv และ vera_report.md. " + blind_rule),
        ("rex", "รวม final executive report จากทุก agent สร้าง rex_output.csv และ final report. ห้ามสรุปว่า success ถ้า Quinn fail หรือ Mo metrics/report ขัดกัน. " + blind_rule),
    ]

    prev_agent = ""
    completed: list[str] = []
    for agent, task in sequence:
        try:
            # RUN-ALL is a fresh pipeline run. Do not reuse stale generated scripts
            # from older experiments because they can lock in wrong target choices.
            delete_old_scripts(output_dir_for(project, agent))
            out = run_agent(agent, task, prev_agent=prev_agent, project_dir=project)
            ok, msg = validate_agent_output(agent, out, project)
            if not ok:
                if agent in ("scout", "dana", "eddie", "finn", "mo"):
                    _print_gate_fail_recovery(agent, msg, project)
                    print(f"{RD}  RUN-ALL stopped at {agent.upper()}{RST}")
                    return
                print(f"{YL}  ⚠ {agent.upper()} output warning: {msg}{RST}")
            completed.append(agent)
            prev_agent = agent
        except KeyboardInterrupt:
            print(f"\n{YL}  RUN-ALL stopped by user{RST}")
            return
        except Exception as e:
            print(f"{RD}  RUN-ALL failed at {agent.upper()}: {e}{RST}")
            return

    print(f"\n{GR}  RUN-ALL complete:{RST} {', '.join(completed)}")


def _looks_like_run_all(user_input: str) -> bool:
    lower = user_input.lower()
    return (
        ("run pipeline" in lower or "pipeline ทั้งระบบ" in lower or "ทั้งระบบ" in lower)
        and ("agent" in lower or "pipeline" in lower)
    ) or ("เริ่มpipeline" in lower) or ("เริ่ม pipeline" in lower and "ใหม่" in lower)


def run_direct_agent_command(user_input: str) -> None:
    parts = user_input[1:].split(" ", 1)
    agent_part = parts[0].lower()
    task = parts[1] if len(parts) > 1 else ""
    if not task:
        print(f"{RD}  ใช้งาน:{RST} @{agent_part} <task>")
        return
    discover = agent_part.endswith("!")
    agent_name = agent_part.rstrip("!")
    set_tab_title(f"⏳ {agent_name.upper()} — กำลังรัน...")
    run_agent(agent_name, task, project_dir=STATE.active_project, discover=discover)
    notify_tab(success=True, label=f"{agent_name.upper()} เสร็จ")


def handle_cli_command(user_input: str) -> str:
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
            "s": "status",
            "st": "status",
            "status": "status",
            "kb": "kb",
            "help": "help",
            "h": "help",
            "?": "help",
            "claude": "claude",
            "end": "end session",
            "end-session": "end session",
            "exit": "exit",
            "quit": "exit",
        }.get(slash_cmd)
        if alias is None:
            print(f"{YL}  ไม่รู้จักคำสั่ง /{slash_cmd} — ใช้ /help เพื่อดูคำสั่ง{RST}")
            return "handled"
        user_input = f"{alias} {slash_arg}".strip()

    lower = user_input.lower()
    if lower in ("exit", "quit"):
        print(f"{YL}  ลาก่อนค่ะ{RST}")
        return "exit"
    if lower == "end session":
        STATE.reset_session()
        print(f"{YL}  ANNA:{RST} เริ่ม session ใหม่แล้วค่ะ  {DIM}(Codex calls reset → 0/{CLAUDE_LIMIT}){RST}")
        return "handled"
    if lower == "help":
        print_help()
        return "handled"
    if lower == "run-all" or lower.startswith("run-all "):
        extra = user_input[7:].strip() if lower.startswith("run-all ") else ""
        run_all_pipeline_command(extra)
        return "handled"
    if lower == "project":
        CLI.print_status()
        print(f"  ใช้: /project <name>")
        return "handled"
    if lower.startswith("project "):
        name = user_input[8:].strip()
        project, _message = CLI.resolve_project(name)
        STATE.active_project = project
        global _last_pipeline_project
        if project:
            PIPELINE.rebuild_from_project(project)
            _last_pipeline_project = project
        else:
            PIPELINE.clear()
            _last_pipeline_project = None
        status = f"{GR}{STATE.active_project}{RST}" if STATE.active_project else f"{RD}ไม่พบ project นี้{RST}"
        print(f"{YL}  ANNA:{RST} Active project → {status}")
        return "handled"
    if lower.startswith("kb "):
        name = user_input[3:].strip()
        CLI.print_kb(name, load_kb(name))
        return "handled"
    if lower in ("status", "stauts", "stats", "stat", "สถานะ"):
        CLI.print_status()
        return "handled"
    if lower == "resume":
        resume_project()
        return "handled"
    if lower.startswith("resume "):
        project_name, extra_instruction = _split_resume_args(user_input[7:].strip())
        resume_project(project_name, extra_instruction)
        return "handled"
    if lower in ("claude", "claude status", "codex", "codex status"):
        CLI.print_claude_usage()
        return "handled"
    if user_input.startswith("!!"):
        anna_discover(user_input[2:].strip())
        return "handled"
    if user_input.startswith("@"):
        run_direct_agent_command(user_input)
        return "handled"
    if _looks_like_run_all(user_input):
        run_all_pipeline_command(user_input)
        return "handled"
    return "pipeline"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    CLI.print_header()

    # ── ESC monitor (daemon thread) ───────────────────────────
    _mon = threading.Thread(target=_esc_monitor, daemon=True)
    _mon.start()

    # ── Main loop ─────────────────────────────────────────────
    while True:
        user_input = read_cli_input()
        if user_input is None:
            break

        if not user_input:
            continue
        command_result = handle_cli_command(user_input)
        if command_result == "exit":
            break
        if command_result == "handled":
            continue

        try:
            run_pipeline(user_input)
            notify_tab(success=True, label="เสร็จสิ้น — พร้อมรับคำสั่ง")
        except KeyboardInterrupt:
            with STATE._proc_lock:
                _kb_procs = list(STATE._active_procs)
            for _kp in _kb_procs:
                try:
                    _kp.kill()
                except OSError:
                    pass
            print(f"\n{YL}  หยุด pipeline แล้ว — พร้อมรับคำสั่งใหม่{RST}")
            STATE.stop_requested.clear()
            notify_tab(success=False, label="หยุดกลางคัน")


if __name__ == "__main__":
    main()
