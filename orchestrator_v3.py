"""
DataScienceOS Orchestrator — Path-Based Pipeline v2
Pipeline ส่ง FILE PATH เท่านั้น ไม่ส่ง content
Agent ที่มี script → รัน script จริง (subprocess)
Agent ที่ไม่มี script → LLM สร้าง report + Python code
"""

import re
import sys
import threading
import concurrent.futures
import time
from pathlib import Path
from dotenv import load_dotenv

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
from anna_core.project import AgentSpecLoader, ProjectDetector
from anna_core.runner import run_python_script
from anna_core.state import OrchestratorState

if sys.platform == "win32":
    import msvcrt

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
CONFIG = load_config(BASE_DIR)

# ── Colors ────────────────────────────────────────────────────────────────────
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

ANNA_SYSTEM = (BASE_DIR / "CLAUDE.md").read_text(encoding="utf-8")

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
    print('\a', end='', flush=True)
    icon  = "✅" if success else "⚠️"
    title = f"{icon} Anna — {label}" if label else (f"{icon} Anna — พร้อม" if success else f"{icon} Anna — หยุด")
    print(f'\033]0;{title}\007', end='', flush=True)


def set_tab_title(title: str):
    """เปลี่ยนชื่อ tab terminal ระหว่าง pipeline รัน"""
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
                if STATE.current_proc is not None:
                    print(f"\n{YL}  [ESC] หยุด script — กลับไปรอคำสั่ง...{RST}")
                    STATE.current_proc.kill()
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

def pipeline_clear():
    PIPELINE.clear()


# ── Script Runner ─────────────────────────────────────────────────────────────

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
    out_path = str(csvs[0]) if csvs else str(output_dir)
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


def validate_agent_output(agent_name: str, output_path: str) -> tuple[bool, str]:
    """ตรวจสอบว่า output ของ agent มีอยู่จริงและไม่ว่าง — non-blocking, warn only"""
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
            # ตรวจ row count — ถ้า < 20 rows อาจโหลดไฟล์ผิด (เช่น outlier_flags.csv)
            full_rows = sum(1 for _ in open(str(p), encoding="utf-8")) - 1
            if full_rows < 20:
                return False, (f"{p.name} มีแค่ {full_rows} rows — "
                               f"อาจโหลดไฟล์ผิด (outlier_flags? ควรเป็น *_output.csv)")
            return True, f"{p.name} ({full_rows} rows, {df.shape[1]} cols)"
        except Exception as e:
            return False, f"อ่าน CSV ไม่ได้: {e}"
    if p.suffix == ".md":
        size = p.stat().st_size
        if size < 50:
            return False, f"report เล็กเกินไป ({size} bytes)"
        return True, f"{p.name} ({size:,} bytes)"
    return True, p.name


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
        fallback = latest_output_csv(project_dir)
        if fallback:
            old_input = input_path
            input_path = fallback
            print(f"{YL}  ⟳ input .md → CSV fallback:{RST} {DIM}{input_path}{RST}")
            log_raw("system", f"input .md fallback: {old_input} → {input_path}", task=agent_name)

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
    py_path.write_text("\n\n".join(code_blocks), encoding="utf-8")
    print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} script saved — กำลังรัน...{RST}  {DIM}→ {py_path}{RST}")

    output_path, returncode, stderr = run_script_with_deepseek_autofix(
        agent_name, task, py_path, input_path, output_dir, max_retries=15
    )
    if returncode != 0:
        output_path, ok = handle_failed_script(agent_name, task, py_path, input_path, output_dir, stderr)
        if not ok:
            print(f"{RD}  ✗ Auto-fix ทั้งหมดล้มเหลว — ใช้ report แทน{RST}")
            output_path = str(report_path)

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
    # CRISP-DM iteration guard — ป้องกัน infinite loop
    STATE.agent_iter_count[agent_name] = STATE.agent_iter_count.get(agent_name, 0) + 1
    if STATE.agent_iter_count[agent_name] > MAX_AGENT_ITER:
        print(f"\n{RD}  ✗ {BLD}{agent_name.upper()}{RST}{RD} ถึง max iterations ({MAX_AGENT_ITER}) — ข้าม CRISP-DM loop{RST}")
        log_raw("system", f"CRISP-DM loop guard: {agent_name} ถึง max {MAX_AGENT_ITER} iterations", task="loop-guard")
        return pipeline_read(agent_name) or ""

    iter_label = f" [{STATE.agent_iter_count[agent_name]}/{MAX_AGENT_ITER}]" if STATE.agent_iter_count[agent_name] > 1 else ""
    bar = "─" * max(0, 48 - len(agent_name) - len(iter_label))
    print(f"\n{CY}┌─ {BLD}{agent_name.upper()}{RST}{CY}{YL}{iter_label}{RST}{CY} {bar}┐{RST}")

    input_path = resolve_agent_input(agent_name, prev_agent, project_dir)
    output_dir = output_dir_for(project_dir, agent_name)

    # ── ลบ script เก่าออกก่อนทุกครั้ง — บังคับให้ LLM สร้างใหม่เสมอ ──────────
    delete_old_scripts(output_dir)

    # ── Priority 1: script จริง (+ auto-fix via DeepSeek ถ้า error) ──────────
    script = find_agent_script(agent_name, project_dir)
    if script and output_dir and not discover:
        return run_existing_script_agent(agent_name, task, script, input_path, output_dir)

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


# ── Project Detection ─────────────────────────────────────────────────────────

def detect_project(text: str) -> Path|None:
    return PROJECT_DETECTOR.detect(text)


# ── Anna Full-Power Action Executor ──────────────────────────────────────────

def execute_anna_actions(response: str) -> str:
    return ACTION_EXECUTOR.execute(response)


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
    action_results = execute_anna_actions(anna_response)
    if action_results:
        print(f"\n{CY}  ⟳ ส่งผลลัพธ์กลับให้ Anna...{RST}")
        followup = f"ผลลัพธ์จากการดำเนินการ:\n\n{action_results}\n\nโปรดสรุปและตอบผู้ใช้เป็นภาษาไทย"
        STATE.anna_history.append({"role": "user",      "content": user_input})
        STATE.anna_history.append({"role": "assistant", "content": anna_response})
        anna_response = call_deepseek(anna_system, followup, label="ANNA", history=STATE.anna_history)
        STATE.anna_history.append({"role": "user",      "content": followup})
        STATE.anna_history.append({"role": "assistant", "content": anna_response})
    else:
        STATE.anna_history.append({"role": "user",      "content": user_input})
        STATE.anna_history.append({"role": "assistant", "content": anna_response})

    ask = parse_ask_user(anna_response)
    if ask:
        print(f"\n{YL}┌─ ANNA ─────────────────────────────────────────────┐{RST}")
        print(f"{YL}│{RST}  {ask}")
        print(f"{YL}└────────────────────────────────────────────────────┘{RST}")
        ans = input(f"  {BLD}คุณ (y/n):{RST} ").strip().lower()
        if ans != "y":
            print(f"\n{YL}  ANNA:{RST} เข้าใจแล้วค่ะ หยุดการทำงาน")
            return

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
                print(f"\n{YL}┌─ ANNA ─────────────────────────────────────────────┐{RST}")
                print(f"{YL}│{RST}  {ask}")
                print(f"{YL}└────────────────────────────────────────────────────┘{RST}")
            return

    pipeline_clear()
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(grp)) as _pool:
                _futs = {
                    _pool.submit(run_agent, d["agent"], d["task"],
                                 prev_agent, STATE.active_project, d.get("discover", False)): d["agent"]
                    for d in grp
                }
                for _f in concurrent.futures.as_completed(_futs):
                    _ag = _futs[_f]
                    try:
                        _out = _f.result()
                        ok, msg = validate_agent_output(_ag, _out)
                        if not ok:
                            print(f"{YL}  ⚠ {BLD}{_ag.upper()}{RST}{YL} output warning: {msg}{RST}")
                    except Exception as _e:
                        print(f"{RD}  ✗ parallel {_ag} error: {_e}{RST}")
                    completed.append(_ag)
            prev_agent = grp[-1]["agent"]
        else:
            # ── Sequential (เดิม) ─────────────────────────────────
            d       = grp[0]
            agent   = d.get("agent", "").lower()
            task    = d.get("task", "")
            discover = d.get("discover", False)
            if not agent or not task:
                continue
            out = run_agent(agent, task, prev_agent=prev_agent,
                            project_dir=STATE.active_project, discover=discover)
            ok, msg = validate_agent_output(agent, out)
            if not ok:
                print(f"{YL}  ⚠ {BLD}{agent.upper()}{RST}{YL} output warning: {msg}{RST}")

            # ── PIPELINE_SPEC guard: Eddie ต้องเขียน PIPELINE_SPEC ครบ ──────
            if agent == "eddie" and STATE.active_project:
                eddie_out_dir = STATE.active_project / "output" / "eddie"
                for _retry in range(2):
                    if check_pipeline_spec(eddie_out_dir):
                        break
                    print(f"\n{YL}  ⚠ Eddie report ขาด PIPELINE_SPEC — retry {_retry+1}/2{RST}")
                    log_raw("system", f"PIPELINE_SPEC missing — Eddie retry {_retry+1}", task="pipeline-guard")
                    out = run_agent(
                        "eddie",
                        task + " — บังคับเขียน PIPELINE_SPEC block ให้ครบ: problem_type, target_column, recommended_model, preprocessing, key_features",
                        prev_agent=prev_agent,
                        project_dir=STATE.active_project,
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


def resume_project(name: str) -> None:
    project, message = CLI.resolve_project(name)
    if not project:
        color = YL if message.startswith("พบหลาย") else RD
        print(f"{color}  {message}{RST}")
        return
    STATE.active_project = project
    done = PIPELINE.completed_agents()
    print(f"\n{YL}  Resume:{RST} {BLD}{project.name}{RST}")
    print(f"  เสร็จแล้ว: {GR}{', '.join(done) or 'ไม่มี'}{RST}")
    resume_msg = (
        f"Resume project {project.name}. "
        f"Agents ที่เสร็จแล้ว: {', '.join(done) or 'ไม่มี'}. "
        f"วิเคราะห์ว่าต้องทำอะไรต่อใน CRISP-DM pipeline แล้ว dispatch ต่อทันที"
    )
    try:
        run_pipeline(resume_msg)
    except KeyboardInterrupt:
        print(f"\n{YL}  หยุด resume pipeline{RST}")


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
    lower = user_input.lower()
    if lower in ("exit", "quit"):
        print(f"{YL}  ลาก่อนค่ะ{RST}")
        return "exit"
    if lower == "end session":
        STATE.reset_session()
        print(f"{YL}  ANNA:{RST} เริ่ม session ใหม่แล้วค่ะ  {DIM}(Claude calls reset → 0/{CLAUDE_LIMIT}){RST}")
        return "handled"
    if lower == "help":
        print_help()
        return "handled"
    if lower.startswith("project "):
        name = user_input[8:].strip()
        project, _message = CLI.resolve_project(name)
        STATE.active_project = project
        status = f"{GR}{STATE.active_project}{RST}" if STATE.active_project else f"{RD}ไม่พบ project นี้{RST}"
        print(f"{YL}  ANNA:{RST} Active project → {status}")
        return "handled"
    if lower.startswith("kb "):
        name = user_input[3:].strip()
        CLI.print_kb(name, load_kb(name))
        return "handled"
    if lower == "status":
        CLI.print_status()
        return "handled"
    if lower.startswith("resume "):
        resume_project(user_input[7:].strip())
        return "handled"
    if lower in ("claude", "claude status"):
        CLI.print_claude_usage()
        return "handled"
    if user_input.startswith("!!"):
        anna_discover(user_input[2:].strip())
        return "handled"
    if user_input.startswith("@"):
        run_direct_agent_command(user_input)
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
            if STATE.current_proc:
                STATE.current_proc.kill()
            print(f"\n{YL}  หยุด pipeline แล้ว — พร้อมรับคำสั่งใหม่{RST}")
            STATE.stop_requested.clear()
            notify_tab(success=False, label="หยุดกลางคัน")


if __name__ == "__main__":
    main()

