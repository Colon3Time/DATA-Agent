"""
DataScienceOS Orchestrator — Path-Based Pipeline v2
Pipeline ส่ง FILE PATH เท่านั้น ไม่ส่ง content
Agent ที่มี script → รัน script จริง (subprocess)
Agent ที่ไม่มี script → LLM สร้าง report + Python code
"""

import os
import re
import json
import subprocess
import requests
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

if sys.platform == "win32":
    import msvcrt

load_dotenv(Path(__file__).parent / ".env")

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
DEEPSEEK_URL   = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
CLAUDE_MODEL   = "claude-sonnet-4-6"

BASE_DIR      = Path(__file__).parent
AGENTS_DIR    = BASE_DIR / "agents"
LOGS_DIR      = BASE_DIR / "logs"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
PIPELINE_DIR  = BASE_DIR / "pipeline"
PROJECTS_DIR  = BASE_DIR / "projects"

MODE = "light"
if "--mode" in sys.argv:
    idx = sys.argv.index("--mode")
    if idx + 1 < len(sys.argv):
        MODE = sys.argv[idx + 1]

CLAUDE_LIMIT = 10
if "--claude-limit" in sys.argv:
    idx = sys.argv.index("--claude-limit")
    if idx + 1 < len(sys.argv):
        try:
            CLAUDE_LIMIT = int(sys.argv[idx + 1])
        except ValueError:
            pass

# ── Step Mode — ถามยืนยันก่อนรัน agent ถัดไปทุกครั้ง ──────────────────────────
# ค่าเริ่มต้น: เปิด (True) — ใช้ --auto เพื่อปิดและรันต่อเนื่องอัตโนมัติ
STEP_MODE: bool = "--auto" not in sys.argv

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

anna_history:     list      = []
active_project:   Path|None = None
claude_calls:     int       = 0
agent_iter_count: dict      = {}  # ติดตาม iteration ของแต่ละ agent ในรอบ pipeline

_current_proc:    subprocess.Popen | None = None
_stop_requested:  threading.Event         = threading.Event()


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
                if _current_proc is not None:
                    print(f"\n{YL}  [ESC] หยุด script — กลับไปรอคำสั่ง...{RST}")
                    _current_proc.kill()
                    _stop_requested.set()
        time.sleep(0.05)


# ── LLM Callers ───────────────────────────────────────────────────────────────

def call_deepseek(system_prompt: str, user_message: str, label: str = "", history: list|None = None) -> str:
    """DeepSeek API — streaming, OpenAI-compatible"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print(f"{RD}  ✗ DEEPSEEK_API_KEY not found in .env{RST}")
        return "[ERROR] DEEPSEEK_API_KEY not found in .env"
    if label:
        bar = "─" * max(0, 46 - len(label))
        print(f"\n{BL}┌─ {BLD}{label}{RST}{BL} {bar}┐{RST}")
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    try:
        response = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": DEEPSEEK_MODEL, "messages": messages, "stream": True},
            stream=True, timeout=180,
        )
        if response.status_code != 200:
            body = response.text[:500]
            print(f"{RD}  ✗ DeepSeek HTTP {response.status_code}: {body}{RST}")
            return f"[ERROR] DeepSeek HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        print(f"{RD}  ✗ DeepSeek connection failed{RST}")
        return "[ERROR] DeepSeek connection failed"
    except requests.exceptions.Timeout:
        print(f"{RD}  ✗ DeepSeek timeout{RST}")
        return "[ERROR] DeepSeek timeout"

    full = []
    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if text.startswith("data: "):
            text = text[6:]
        if text == "[DONE]":
            break
        try:
            token = json.loads(text)["choices"][0]["delta"].get("content", "")
            print(token, end="", flush=True)
            full.append(token)
        except (json.JSONDecodeError, KeyError):
            pass
    print()
    return "".join(full)


def call_claude(system_prompt: str, user_message: str, label: str = "") -> str:
    """ลอง Anthropic API ก่อน — ถ้าถึง limit / ไม่มี key / credit หมด → fallback DeepSeek"""
    global claude_calls

    if claude_calls >= CLAUDE_LIMIT:
        print(f"\n{YL}  ⚠ Claude limit ถึง {claude_calls}/{CLAUDE_LIMIT} calls แล้ว → ใช้ DeepSeek แทน{RST}")
        return call_deepseek(system_prompt, user_message, label=f"{label} (via DeepSeek[limit])")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic as _ant
            claude_calls += 1
            remaining = CLAUDE_LIMIT - claude_calls
            print(f"\n{MG}{'━'*55}{RST}")
            print(f"{MG}  ✦ CLAUDE  {BLD}{label}{RST}  {DIM}[{claude_calls}/{CLAUDE_LIMIT} — เหลือ {remaining}]{RST}")
            print(f"{MG}{'━'*55}{RST}")
            client = _ant.Anthropic(api_key=api_key)
            with client.messages.stream(
                model=CLAUDE_MODEL, max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                full = []
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full.append(text)
            print()
            return "".join(full)
        except Exception as e:
            claude_calls -= 1  # ไม่นับ call ที่ fail
            msg = str(e)
            if "credit" in msg.lower():
                print(f"\n{RD}  ✗ CLAUDE credit หมด{RST} {YL}→ fallback DeepSeek{RST}")
            else:
                print(f"\n{YL}  ⚠ CLAUDE error ({e}) → fallback DeepSeek{RST}")
    else:
        print(f"\n{YL}  ⚠ ไม่พบ ANTHROPIC_API_KEY → ใช้ DeepSeek แทน{RST}")

    return call_deepseek(system_prompt, user_message, label=f"{label} (via DeepSeek)")


# ── Knowledge Base ────────────────────────────────────────────────────────────

def load_kb(agent_name: str) -> str:
    """โหลด KB ทุกไฟล์ของ agent นั้น (methods + decision_tree + อื่นๆ)"""
    files = sorted(KNOWLEDGE_DIR.glob(f"{agent_name}_*.md"))
    if not files:
        return ""
    parts = []
    for f in files:
        try:
            parts.append(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return "\n\n".join(parts)

def load_relevant_kb(agent_name: str, task: str, top_n: int = 4) -> str:
    """RAG-style: ดึงเฉพาะ KB sections ที่ตรงกับ task (TF-IDF cosine similarity)"""
    kb = load_kb(agent_name)
    if not kb:
        return ""
    sections = [s.strip() for s in re.split(r'\n(?=##)', kb.strip()) if s.strip()]
    if len(sections) <= top_n:
        return kb  # KB เล็กพอ — โหลดทั้งหมด
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        docs = [task] + sections
        mat  = TfidfVectorizer(min_df=1).fit_transform(docs)
        scores = cos_sim(mat[0:1], mat[1:])[0]
        top_idx = scores.argsort()[::-1][:top_n]
        return "\n\n".join(sections[i] for i in sorted(top_idx))
    except ImportError:
        words = set(task.lower().split())
        scored = [(len(words & set(s.lower().split())), s) for s in sections]
        return "\n\n".join(s for _, s in sorted(scored, reverse=True)[:top_n])

def save_kb(agent_name: str, content: str, entry_type: str = "discovery"):
    """
    entry_type:
      'discovery'  — วิธีใหม่ที่พบ ยังไม่ได้พิสูจน์ซ้ำ
      'feedback'   — แก้งาน อาจไม่ถูกเสมอ
      'proven'     — พิสูจน์แล้วซ้ำหลายครั้งว่าได้ผล → consolidate จะไม่ลบ
      'deprecated' — ล้าสมัย / ผิด → consolidate จะลบออกทันที
    """
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    f = KNOWLEDGE_DIR / f"{agent_name}_methods.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    tag_map = {"feedback": "FEEDBACK", "proven": "PROVEN", "deprecated": "DEPRECATED"}
    tag = tag_map.get(entry_type, "DISCOVERY")
    with open(f, "a", encoding="utf-8") as fp:
        fp.write(f"\n\n## [{ts}] [{tag}]\n{content.strip()}\n")
    log_raw("system", f"KB [{tag}] {agent_name}: {content[:80]}", task="kb_update")

def consolidate_kb(agent_name: str):
    """ลบ KB entries ที่ซ้ำกัน (cosine similarity > 0.85) เก็บตัวที่ใหม่กว่า
    กฎพิเศษ: [PROVEN] → ไม่ลบเลย | [DEPRECATED] → ลบทันทีโดยไม่ต้องเปรียบเทียบ
    """
    kb = load_kb(agent_name)
    if not kb:
        return
    sections = [s.strip() for s in re.split(r'\n(?=##)', kb.strip()) if s.strip()]
    if len(sections) < 10:
        return  # ยังไม่จำเป็น

    # แยก PROVEN, DEPRECATED, และ sections ทั่วไปก่อน
    proven     = [s for s in sections if "[PROVEN]"     in s]
    deprecated = [s for s in sections if "[DEPRECATED]" in s]
    normal     = [s for s in sections if "[PROVEN]" not in s and "[DEPRECATED]" not in s]
    removed_deprecated = len(deprecated)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        if len(normal) >= 2:
            mat    = TfidfVectorizer(min_df=1).fit_transform(normal)
            sim    = cos_sim(mat)
            remove = set()
            for i in range(len(normal)):
                if i in remove:
                    continue
                for j in range(i + 1, len(normal)):
                    if j not in remove and sim[i, j] > 0.85:
                        remove.add(i)  # เก็บตัวหลัง (ใหม่กว่า)
            normal = [normal[i] for i in range(len(normal)) if i not in remove]
    except ImportError:
        pass

    kept = proven + normal  # PROVEN ต้องอยู่ด้านบนเสมอ
    f    = KNOWLEDGE_DIR / f"{agent_name}_methods.md"
    f.write_text("\n\n".join(kept), encoding="utf-8")
    removed_total = len(sections) - len(kept)
    if removed_total:
        log_raw("system",
                f"KB consolidate {agent_name}: ลบ {removed_total} entries "
                f"(deprecated={removed_deprecated}, duplicate={removed_total-removed_deprecated})",
                task="kb_consolidate")


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
    PIPELINE_DIR.mkdir(exist_ok=True)
    (PIPELINE_DIR / f"{agent_name}_path.txt").write_text(str(file_path), encoding="utf-8")

def pipeline_read(agent_name: str) -> str:
    f = PIPELINE_DIR / f"{agent_name}_path.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""

def pipeline_clear():
    if PIPELINE_DIR.exists():
        for f in PIPELINE_DIR.glob("*_path.txt"):
            f.unlink()


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
    global _current_proc
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{CY}  ▶ SCRIPT{RST}  {BLD}{script_path.name}{RST}  {DIM}← {input_path or 'no input'}{RST}")
    print(f"{DIM}  กด ESC เพื่อหยุด script นี้{RST}")

    _stop_requested.clear()
    proc = subprocess.Popen(
        [sys.executable, str(script_path),
         "--input", input_path,
         "--output-dir", str(output_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    _current_proc = proc
    try:
        stdout, stderr = proc.communicate(timeout=300)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        proc.kill()
        stdout, stderr = proc.communicate()
        _current_proc = None
        raise KeyboardInterrupt
    _current_proc = None

    if _stop_requested.is_set():
        return str(output_dir), -999, "หยุดโดยผู้ใช้ (ESC)"

    if stdout:
        print(stdout[-2000:])
    if proc.returncode != 0:
        print(f"{RD}  ╔══ SCRIPT ERROR ══╗{RST}")
        print(f"{RD}{stderr[:500]}{RST}")
        print(f"{RD}  ╚{'═'*18}╝{RST}")

    csvs = sorted(output_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    out_path = str(csvs[0]) if csvs else str(output_dir)
    if proc.returncode == 0 and not csvs:
        fake_err = "Script ran successfully but produced no CSV in OUTPUT_DIR. Must add: df.to_csv(os.path.join(OUTPUT_DIR, 'output.csv'), index=False)"
        return out_path, -1, fake_err
    return out_path, proc.returncode, stderr


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


def run_agent(agent_name: str, task: str, prev_agent: str = "",
              project_dir: Path|None = None, discover: bool = False) -> str:
    # CRISP-DM iteration guard — ป้องกัน infinite loop
    agent_iter_count[agent_name] = agent_iter_count.get(agent_name, 0) + 1
    if agent_iter_count[agent_name] > MAX_AGENT_ITER:
        print(f"\n{RD}  ✗ {BLD}{agent_name.upper()}{RST}{RD} ถึง max iterations ({MAX_AGENT_ITER}) — ข้าม CRISP-DM loop{RST}")
        log_raw("system", f"CRISP-DM loop guard: {agent_name} ถึง max {MAX_AGENT_ITER} iterations", task="loop-guard")
        return pipeline_read(agent_name) or ""

    iter_label = f" [{agent_iter_count[agent_name]}/{MAX_AGENT_ITER}]" if agent_iter_count[agent_name] > 1 else ""
    bar = "─" * max(0, 48 - len(agent_name) - len(iter_label))
    print(f"\n{CY}┌─ {BLD}{agent_name.upper()}{RST}{CY}{YL}{iter_label}{RST}{CY} {bar}┐{RST}")

    raw_input_path = pipeline_read(prev_agent) if prev_agent else ""
    input_path     = resolve_input_path(prev_agent, raw_input_path, project_dir)
    if input_path != raw_input_path and input_path:
        print(f"{CY}  ⟳ input resolved:{RST} {DIM}{input_path}{RST}")
        log_raw("system", f"resolve input: {raw_input_path} → {input_path}", task=f"{agent_name}")

    # fallback: ถ้าไม่มี input path ให้หา CSV หรือ SQLite ใน project/input/ อัตโนมัติ
    if not input_path and project_dir:
        input_dir = project_dir / "input"
        if input_dir.exists():
            sqlites = sorted(input_dir.glob("*.sqlite"), key=lambda x: x.stat().st_mtime, reverse=True)
            if sqlites:
                input_path = str(sqlites[0])
                print(f"{CY}  ⟳ input sqlite fallback:{RST} {DIM}{input_path}{RST}")
                log_raw("system", f"input sqlite fallback → {input_path}", task=agent_name)
            else:
                csvs = sorted(input_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
                if csvs:
                    input_path = str(csvs[0])
                    print(f"{CY}  ⟳ input fallback:{RST} {DIM}{input_path}{RST}")
                    log_raw("system", f"input fallback → {input_path}", task=agent_name)

    # fallback: ถ้า input เป็น .md (agent ก่อนหน้าไม่ผลิต CSV) → หา CSV ที่ดีที่สุดใน project
    if input_path and input_path.endswith(".md") and project_dir:
        output_root = project_dir / "output"
        # เรียงตาม mtime ล่าสุด — ได้ CSV ที่ผลิตล่าสุด
        all_csvs = sorted(output_root.glob("**/*_output.csv"),
                          key=lambda x: x.stat().st_mtime, reverse=True)
        if all_csvs:
            old_input = input_path
            input_path = str(all_csvs[0])
            print(f"{YL}  ⟳ input .md → CSV fallback:{RST} {DIM}{input_path}{RST}")
            log_raw("system", f"input .md fallback: {old_input} → {input_path}", task=agent_name)

    output_dir = (project_dir / "output" / agent_name) if project_dir else None

    # ── Priority 1: script จริง (+ auto-fix via DeepSeek ถ้า error) ──────────
    script = find_agent_script(agent_name, project_dir)
    if script and output_dir and not discover:
        MAX_RETRIES = 15
        output_path = str(output_dir)
        returncode = -1
        stderr = ""
        for attempt in range(MAX_RETRIES):
            output_path, returncode, stderr = run_script(script, input_path, output_dir)
            if returncode == 0:
                break
            if attempt < MAX_RETRIES - 1:
                print(f"\n{YL}  ⟳ Script error (รอบ {attempt+1}/{MAX_RETRIES}) → DeepSeek กำลังแก้ไข...{RST}")
                script_content = script.read_text(encoding="utf-8")
                fix_prompt = (
                    f"Script นี้ error:\n\n```python\n{script_content[:3000]}\n```\n\n"
                    f"Error:\n```\n{stderr[:800]}\n```\n\n"
                    f"Input path: {input_path}\nOutput dir: {output_dir}\n\n"
                    f"แก้ script ให้รันได้ ตอบเป็น python code block เดียวเท่านั้น"
                )
                fixed = call_deepseek(get_system_prompt(agent_name, task=task), fix_prompt,
                                      label=f"{agent_name.upper()} auto-fix #{attempt+1}")
                blocks = re.findall(r'```python\n(.*?)```', fixed, re.DOTALL)
                if blocks:
                    script.write_text("\n\n".join(blocks), encoding="utf-8")
                    print(f"{GR}  ✓ Script แก้แล้ว — รันใหม่...{RST}")
                    log_raw("system", f"auto-fix {agent_name} script attempt {attempt+1}", task="auto-fix")
                else:
                    print(f"{RD}  ✗ DeepSeek ไม่ส่ง code กลับมา — หยุด retry{RST}")
                    break
            else:
                print(f"{RD}  ✗ Script ยังไม่สำเร็จหลัง {MAX_RETRIES} รอบ → Anna auto-fix (Claude){RST}")

        # ── Anna auto-fix (Claude) หลัง DeepSeek ทุกรอบล้มเหลว ──────────────
        if returncode != 0:
            output_path, success = anna_autofix_script(
                agent_name, task, script, input_path, output_dir, stderr)
            if not success:
                print(f"\n{RD}{'─'*55}{RST}")
                print(f"{RD}  ✗ Auto-fix ทั้งหมด {MAX_RETRIES} รอบ + Anna ล้มเหลว{RST}")
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

        pipeline_write(agent_name, output_path)
        report_summary = read_report_summary(output_dir, agent_name)
        action_msg = f"รัน script {script.name} สำเร็จ"
        if report_summary:
            action_msg += f"\n{report_summary}"
        log_raw(agent_name, action_msg, task=task, output=output_path)
        log_raw("system", f"pipeline handoff: {agent_name} → {output_path}", task="pipeline")
        print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} done{RST}  {DIM}→ {output_path}{RST}")
        return output_path

    # ── Priority 2: LLM ───────────────────────────────────────
    system = get_system_prompt(agent_name, task=task)

    path_lines = []
    if input_path:
        path_lines.append(f"Input file path : {input_path}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path_lines.append(f"Save CSV to     : {output_dir / f'{agent_name}_output.csv'}")
        path_lines.append(f"Save script to  : {output_dir / f'{agent_name}_script.py'}")
        path_lines.append(f"Save report to  : {output_dir / f'{agent_name}_report.md'}")
    if agent_name == "scout" and project_dir:
        path_lines.append(f"Save dataset to : {project_dir / 'input'}/ ← ไฟล์ข้อมูลจริงต้องอยู่ที่นี่เท่านั้น")

    message = "\n".join(path_lines) + f"\n\nTask: {task}" if path_lines else task

    if discover:
        result = call_claude(system, task, label=f"{agent_name.upper()} discover")
        # Extract key finding only — ห้าม dump ผล LLM ทั้งก้อนลง KB
        first_para = result.strip().split("\n\n")[0][:400]
        save_kb(agent_name, f"Task: {task[:100]}\nKey finding: {first_para}", entry_type="discovery")
    else:
        result = call_deepseek(system, message, label=f"{agent_name.upper()} execute")

    # Auto-extract Self-Improvement discovery ไปเก็บ KB
    auto_extract_kb_learning(agent_name, result)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"{agent_name}_report.md"
        report_path.write_text(result, encoding="utf-8")

        code_blocks = re.findall(r'```python\n(.*?)```', result, re.DOTALL)

        # ถ้า LLM ไม่ส่ง code เลย → retry สูงสุด 3 รอบด้วย prompt บังคับ
        if not code_blocks:
            for _nc in range(3):
                print(f"{YL}  ⟳ ไม่พบ Python code block — retry #{_nc+1} (บังคับ code){RST}")
                _force_msg = (
                    f"คำตอบของคุณต้องมี Python code block เท่านั้น ห้ามตอบเป็น text\n"
                    f"เขียน script ที่รันได้ทันที โดย:\n"
                    f"- อ่านข้อมูลจาก INPUT_PATH (args.input)\n"
                    f"- ประมวลผลตาม task\n"
                    f"- save CSV ไปที่ OUTPUT_DIR (args.output_dir)\n"
                    f"ตอบเป็น ```python ... ``` เท่านั้น ห้ามอธิบาย\n\n"
                    f"Task: {task}\nInput: {input_path}\nOutput dir: {output_dir}"
                )
                result = call_deepseek(system, _force_msg, label=f"{agent_name.upper()} force-code #{_nc+1}")
                code_blocks = re.findall(r'```python\n(.*?)```', result, re.DOTALL)
                if code_blocks:
                    break

        if code_blocks:
            py_path = output_dir / f"{agent_name}_script.py"
            py_path.write_text("\n\n".join(code_blocks), encoding="utf-8")
            print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} script saved — กำลังรัน...{RST}  {DIM}→ {py_path}{RST}")

            # รัน script จริงๆ + auto-fix 15 รอบถ้า error
            _MAX = 15
            _returncode = -1
            _stderr = ""
            _out = str(output_dir)
            for _attempt in range(_MAX):
                _out, _returncode, _stderr = run_script(py_path, input_path, output_dir)
                if _returncode == 0:
                    break
                if _attempt < _MAX - 1:
                    print(f"\n{YL}  ⟳ Script error (รอบ {_attempt+1}/{_MAX}) → DeepSeek แก้ไข...{RST}")
                    _fix = call_deepseek(
                        get_system_prompt(agent_name, task=task),
                        f"Script error:\n```python\n{py_path.read_text(encoding='utf-8')[:3000]}\n```\n"
                        f"Error:\n```\n{_stderr[:800]}\n```\n"
                        f"Input: {input_path}\nOutput dir: {output_dir}\n"
                        f"แก้ให้รันได้ ตอบ python code block เดียว",
                        label=f"{agent_name.upper()} auto-fix #{_attempt+1}",
                    )
                    _blocks = re.findall(r'```python\n(.*?)```', _fix, re.DOTALL)
                    if _blocks:
                        py_path.write_text("\n\n".join(_blocks), encoding="utf-8")
                        log_raw("system", f"auto-fix {agent_name} attempt {_attempt+1}", task="auto-fix")
                    else:
                        print(f"{RD}  ✗ DeepSeek ไม่ส่ง code — หยุด retry{RST}")
                        break
                else:
                    print(f"{RD}  ✗ Script ยังไม่สำเร็จหลัง {_MAX} รอบ → Anna auto-fix (Claude){RST}")

            if _returncode != 0:
                _out, _ok = anna_autofix_script(agent_name, task, py_path, input_path, output_dir, _stderr)
                if not _ok:
                    print(f"{RD}  ✗ Auto-fix ทั้งหมดล้มเหลว — ใช้ report แทน{RST}")
                    _out = str(report_path)
                    try:
                        _choice = input(f"  {BLD}ข้าม agent นี้ (skip) หรือหยุด (stop)? [skip/stop]:{RST} ").strip().lower()
                    except EOFError:
                        _choice = "skip"
                    if _choice == "stop":
                        raise RuntimeError(f"{agent_name} script failed after all retries")

            # Scout: ชี้ CSV ใน input/ แทน
            if agent_name == "scout" and project_dir:
                _input_dir = project_dir / "input"
                _csvs = sorted(_input_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True) if _input_dir.exists() else []
                _out = str(_csvs[0]) if _csvs else _out

            pipeline_write(agent_name, _out)
            log_raw(agent_name, f"รัน script (DeepSeek+run) → {_out}", task=task, output=_out)
            log_raw("system", f"pipeline handoff: {agent_name} → {_out}", task="pipeline")
            print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} done{RST}  {DIM}→ {_out}{RST}")
            return _out

        # Scout: ถ้าไม่มี script ให้ check input/ ก่อน
        if agent_name == "scout" and project_dir:
            input_dir = project_dir / "input"
            csvs = sorted(input_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True) if input_dir.exists() else []
            if csvs:
                pipeline_write(agent_name, str(csvs[0]))
                log_raw(agent_name, "พบ dataset ใน input/ — pipeline → CSV", task=task, output=str(csvs[0]))
                log_raw("system", f"pipeline handoff: scout → {csvs[0]}", task="pipeline")
                print(f"{GR}  ✓ {BLD}SCOUT{RST}{GR} dataset ready in input/{RST}  {DIM}→ {csvs[0]}{RST}")
                return str(csvs[0])

        pipeline_write(agent_name, str(report_path))
        log_raw(agent_name, "สร้าง report (DeepSeek)", task=task, output=str(report_path))
        log_raw("system", f"pipeline handoff: {agent_name} → {report_path}", task="pipeline")
        print(f"{GR}  ✓ {BLD}{agent_name.upper()}{RST}{GR} report saved{RST}  {DIM}→ {report_path}{RST}")
        return str(report_path)

    log_raw(agent_name, result[:200], task=task)
    return result


# ── Dispatch Parser ───────────────────────────────────────────────────────────

DISPATCH_RE = re.compile(r'<DISPATCH>(.*?)</DISPATCH>', re.DOTALL)
ASK_USER_RE = re.compile(r'<ASK_USER>(.*?)</ASK_USER>', re.DOTALL)

def parse_dispatches(text: str) -> list[dict]:
    results = []
    for match in DISPATCH_RE.finditer(text):
        raw = match.group(1).strip()
        # ลบ code fence ที่ DeepSeek บางครั้งใส่ไว้ข้างใน
        raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        d = None
        # attempt 1: JSON ปกติ
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            pass
        # attempt 2: DeepSeek ลืม {} → ครอบให้
        if d is None:
            try:
                d = json.loads("{" + raw + "}")
            except json.JSONDecodeError:
                pass
        # attempt 3: หา inline JSON ข้างใน (DeepSeek ใส่ ``` ครอบ DISPATCH อีกที)
        if d is None:
            m2 = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m2:
                try:
                    d = json.loads(m2.group())
                except json.JSONDecodeError:
                    pass
        if d is None:
            continue
        agent = d.get("agent", "").lower().strip()
        task  = d.get("task", "").strip()
        if agent in VALID_AGENTS and task and task not in ("...", ""):
            results.append(d)
        elif agent:
            print(f"{RD}  ✗ dispatch ถูกปฏิเสธ — agent='{agent}' ไม่ถูกต้อง{RST}")
    return results

def parse_ask_user(text: str) -> str|None:
    m = ASK_USER_RE.search(text)
    return m.group(1).strip() if m else None


# ── Project Detection ─────────────────────────────────────────────────────────

def detect_project(text: str) -> Path|None:
    """ตรวจจับ project จาก Anna response — คืน None ถ้าไม่พบชัดเจน
    ไม่ fallback เป็น project ล่าสุดอีกต่อไป เพื่อป้องกัน task ใหม่ใช้ project เก่าผิด
    """
    m = re.search(r'projects[/\\]([\w\-]+)', text)
    if m:
        p = PROJECTS_DIR / m.group(1)
        if p.exists():
            return p
    return None


# ── Anna Full-Power Action Executor ──────────────────────────────────────────

def execute_anna_actions(response: str) -> str:
    """
    Parse และ execute action tags จาก Anna's response จริง ๆ
    คืน string ของผลลัพธ์ทั้งหมด (ส่งกลับให้ Anna อ่านต่อ)
    """
    parts = []

    # READ_FILE
    for m in re.finditer(r'<READ_FILE\s+path="([^"]+)"\s*/?>', response):
        raw = m.group(1)
        # รองรับทั้ง absolute path (C:\... หรือ /...) และ relative path
        p = Path(raw)
        fpath = p if p.is_absolute() else BASE_DIR / raw
        print(f"\n{CY}  ▶ READ_FILE{RST}  {DIM}{raw}{RST}")
        log_raw("anna", f"READ_FILE: {raw}", task="full-power")
        try:
            content = fpath.read_text(encoding="utf-8")
            parts.append(f'[READ_FILE: {raw}]\n{content[:100000]}')
        except Exception as e:
            parts.append(f'[READ_FILE ERROR: {e}]')
            log_raw("anna", f"READ_FILE ERROR: {raw} — {e}", task="full-power")

    # RUN_SHELL
    for m in re.finditer(r'<RUN_SHELL>(.*?)</RUN_SHELL>', response, re.DOTALL):
        cmd = m.group(1).strip()
        print(f"\n{CY}  ▶ RUN_SHELL{RST}  {DIM}{cmd[:60]}{RST}")
        log_raw("anna", f"RUN_SHELL: {cmd[:100]}", task="full-power")
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               encoding="utf-8", timeout=60, cwd=str(BASE_DIR))
            out = (r.stdout + r.stderr)[:1500]
            parts.append(f'[RUN_SHELL: {cmd[:60]}]\n{out}')
            if r.returncode != 0:
                log_raw("anna", f"RUN_SHELL exit={r.returncode}: {cmd[:60]}", task="full-power")
        except Exception as e:
            parts.append(f'[RUN_SHELL ERROR: {e}]')
            log_raw("anna", f"RUN_SHELL ERROR: {cmd[:60]} — {e}", task="full-power")

    # WRITE_FILE
    for m in re.finditer(r'<WRITE_FILE\s+path="([^"]+)">(.*?)</WRITE_FILE>', response, re.DOTALL):
        _p = Path(m.group(1)); fpath = _p if _p.is_absolute() else BASE_DIR / m.group(1)
        print(f"\n{GR}  ▶ WRITE_FILE{RST}  {DIM}{m.group(1)}{RST}")
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(m.group(2), encoding="utf-8")
            parts.append(f'[WRITE_FILE: {m.group(1)}] เขียนสำเร็จ')
            log_raw("anna", f"WRITE_FILE: {m.group(1)}", task="full-power")
        except Exception as e:
            parts.append(f'[WRITE_FILE ERROR: {e}]')
            log_raw("anna", f"WRITE_FILE ERROR: {m.group(1)} — {e}", task="full-power")

    # APPEND_FILE
    for m in re.finditer(r'<APPEND_FILE\s+path="([^"]+)">(.*?)</APPEND_FILE>', response, re.DOTALL):
        _p = Path(m.group(1)); fpath = _p if _p.is_absolute() else BASE_DIR / m.group(1)
        print(f"\n{GR}  ▶ APPEND_FILE{RST}  {DIM}{m.group(1)}{RST}")
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "a", encoding="utf-8") as fp:
                fp.write(m.group(2))
            parts.append(f'[APPEND_FILE: {m.group(1)}] เพิ่มสำเร็จ')
            log_raw("anna", f"APPEND_FILE: {m.group(1)}", task="full-power")
        except Exception as e:
            parts.append(f'[APPEND_FILE ERROR: {e}]')
            log_raw("anna", f"APPEND_FILE ERROR: {m.group(1)} — {e}", task="full-power")

    # EDIT_FILE
    for m in re.finditer(r'<EDIT_FILE\s+path="([^"]+)"><old>(.*?)</old><new>(.*?)</new></EDIT_FILE>', response, re.DOTALL):
        _p = Path(m.group(1)); fpath = _p if _p.is_absolute() else BASE_DIR / m.group(1)
        print(f"\n{GR}  ▶ EDIT_FILE{RST}  {DIM}{m.group(1)}{RST}")
        try:
            original = fpath.read_text(encoding="utf-8")
            fpath.write_text(original.replace(m.group(2), m.group(3), 1), encoding="utf-8")
            parts.append(f'[EDIT_FILE: {m.group(1)}] แก้ไขสำเร็จ')
            log_raw("anna", f"EDIT_FILE: {m.group(1)}", task="full-power")
        except Exception as e:
            parts.append(f'[EDIT_FILE ERROR: {e}]')
            log_raw("anna", f"EDIT_FILE ERROR: {m.group(1)} — {e}", task="full-power")

    # CREATE_DIR
    for m in re.finditer(r'<CREATE_DIR\s+path="([^"]+)"\s*/?>', response):
        dpath = BASE_DIR / m.group(1)
        print(f"\n{GR}  ▶ CREATE_DIR{RST}  {DIM}{m.group(1)}{RST}")
        try:
            dpath.mkdir(parents=True, exist_ok=True)
            parts.append(f'[CREATE_DIR: {m.group(1)}] สร้างสำเร็จ')
            log_raw("anna", f"CREATE_DIR: {m.group(1)}", task="full-power")
            # ถ้าสร้าง folder ใน projects/ → set active_project ทันที
            try:
                global active_project
                rel_parts = dpath.relative_to(PROJECTS_DIR).parts
                if rel_parts:
                    active_project = PROJECTS_DIR / rel_parts[0]
            except ValueError:
                pass
        except Exception as e:
            parts.append(f'[CREATE_DIR ERROR: {e}]')

    # DELETE_FILE
    for m in re.finditer(r'<DELETE_FILE\s+path="([^"]+)"\s*/?>', response):
        _p = Path(m.group(1)); fpath = _p if _p.is_absolute() else BASE_DIR / m.group(1)
        print(f"\n{RD}  ▶ DELETE_FILE{RST}  {DIM}{m.group(1)}{RST}")
        try:
            fpath.unlink()
            parts.append(f'[DELETE_FILE: {m.group(1)}] ลบสำเร็จ')
            log_raw("anna", f"DELETE_FILE: {m.group(1)}", task="full-power")
        except Exception as e:
            parts.append(f'[DELETE_FILE ERROR: {e}]')

    # UPDATE_KB — tagged FEEDBACK เพราะมาจากการแก้งานของ Anna
    for m in re.finditer(r'<UPDATE_KB\s+agent="([^"]+)">(.*?)</UPDATE_KB>', response, re.DOTALL):
        save_kb(m.group(1), m.group(2).strip(), entry_type="feedback")
        print(f"\n{GR}  ▶ UPDATE_KB{RST}  agent={BLD}{m.group(1)}{RST}")
        parts.append(f'[UPDATE_KB: {m.group(1)}] อัปเดตสำเร็จ')

    # ASK_DEEPSEEK
    for m in re.finditer(r'<ASK_DEEPSEEK>(.*?)</ASK_DEEPSEEK>', response, re.DOTALL):
        q = m.group(1).strip()
        print(f"\n{BL}  ▶ ASK_DEEPSEEK{RST}")
        log_raw("anna", f"ASK_DEEPSEEK: {q[:80]}", task="full-power")
        ans = call_deepseek("You are a helpful AI assistant.", q, label="DEEPSEEK direct")
        parts.append(f'[ASK_DEEPSEEK]\nQ: {q[:200]}\nA: {ans[:1000]}')

    # ASK_CLAUDE
    for m in re.finditer(r'<ASK_CLAUDE>(.*?)</ASK_CLAUDE>', response, re.DOTALL):
        q = m.group(1).strip()
        print(f"\n{MG}  ▶ ASK_CLAUDE{RST}")
        log_raw("anna", f"ASK_CLAUDE: {q[:80]}", task="full-power")
        ans = call_claude("You are a helpful AI assistant.", q, label="CLAUDE direct")
        parts.append(f'[ASK_CLAUDE]\nQ: {q[:200]}\nA: {ans[:1000]}')

    # RESEARCH — save key findings only ไม่ dump ผลดิบลง KB
    for m in re.finditer(r'<RESEARCH>(.*?)</RESEARCH>', response, re.DOTALL):
        topic = m.group(1).strip()
        print(f"\n{BL}  ▶ RESEARCH{RST}  {DIM}{topic[:60]}{RST}")
        log_raw("anna", f"RESEARCH: {topic[:80]}", task="full-power")
        ans = call_deepseek("You are a research assistant. Be thorough.", topic, label="RESEARCH")
        first_finding = ans.strip().split("\n\n")[0][:400]
        save_kb("anna", f"Research: {topic[:100]}\nKey finding: {first_finding}", entry_type="discovery")
        parts.append(f'[RESEARCH: {topic[:60]}]\n{ans[:1000]}')

    return "\n\n".join(parts)


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


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def load_agent_specs() -> str:
    """โหลด MD ของทุก agent เข้า Anna's context (max 100,000 chars รวม)"""
    parts = []
    total = 0
    for name in sorted(VALID_AGENTS):
        f = AGENTS_DIR / f"{name}.md"
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8")
        chunk = f"\n\n=== AGENT SPEC: {name.upper()} ===\n{content}"
        if total + len(chunk) > 100_000:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)


def run_pipeline(user_input: str):
    global active_project, agent_iter_count
    agent_iter_count = {}  # reset iteration counter ทุก pipeline run
    active_project   = None  # reset project ทุก pipeline run — Anna เลือกใหม่ตาม task

    anna_kb = load_kb("anna")
    projects_list = "\n".join(
        p.name for p in sorted(PROJECTS_DIR.iterdir()) if p.is_dir()
    ) if PROJECTS_DIR.exists() else ""
    agent_specs = load_agent_specs()

    anna_system = (
        ANNA_SYSTEM
        + (f"\n\n---\n## Anna KB\n{anna_kb}" if anna_kb else "")
        + (f"\n\n---\n## Available Projects\n{projects_list}" if projects_list else "")
        + (f"\n\n---\n## Agent Specs (อ่านก่อน dispatch ทุกครั้ง)\n{agent_specs}" if agent_specs else "")
    )

    # Auto-consolidate KB ทุก 10 project
    project_count = len(list(PROJECTS_DIR.iterdir())) if PROJECTS_DIR.exists() else 0
    if project_count > 0 and project_count % 10 == 0:
        print(f"{DIM}  ⟳ KB consolidation (project #{project_count})...{RST}")
        for ag in VALID_AGENTS | {"anna"}:
            consolidate_kb(ag)

    print(f"\n{YL}{'═'*55}{RST}")
    anna_response = call_deepseek(anna_system, user_input, label="ANNA", history=anna_history)
    log_raw("User", user_input)
    log_raw("Anna", anna_response, task="รับคำสั่งจาก User และวางแผน dispatch")

    # เก็บ dispatch จาก response แรกก่อน — ป้องกันหายหลัง action execution
    first_dispatches = parse_dispatches(anna_response)

    # Execute full-power actions แล้ว feed ผลกลับให้ Anna
    action_results = execute_anna_actions(anna_response)
    if action_results:
        print(f"\n{CY}  ⟳ ส่งผลลัพธ์กลับให้ Anna...{RST}")
        followup = f"ผลลัพธ์จากการดำเนินการ:\n\n{action_results}\n\nโปรดสรุปและตอบผู้ใช้เป็นภาษาไทย"
        anna_history.append({"role": "user",      "content": user_input})
        anna_history.append({"role": "assistant", "content": anna_response})
        anna_response = call_deepseek(anna_system, followup, label="ANNA", history=anna_history)
        anna_history.append({"role": "user",      "content": followup})
        anna_history.append({"role": "assistant", "content": anna_response})
    else:
        anna_history.append({"role": "user",      "content": user_input})
        anna_history.append({"role": "assistant", "content": anna_response})

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
        anna_history[-1] = {"role": "assistant", "content": anna_response}
        dispatches = parse_dispatches(anna_response)

    if not dispatches:
        return

    # detect project จาก response เฉพาะกรณีที่ยังไม่มี active_project เท่านั้น
    # ไม่ override project ที่ user ตั้งไว้ผ่าน "project <name>" command
    if active_project is None:
        active_project = detect_project(anna_response)

    pipeline_clear()
    proj_name = active_project.name if active_project else "unknown"
    print(f"\n{CY}  ⟳ Pipeline:{RST} {BLD}{len(dispatches)} agent(s){RST}  {DIM}│ project: {proj_name}{RST}")

    prev_agent = ""
    completed  = []
    _stop_pipeline = False

    for i, d in enumerate(dispatches):
        agent    = d.get("agent", "").lower()
        task     = d.get("task", "")
        discover = d.get("discover", False)
        if not agent or not task:
            continue

        # Step confirmation ก่อนรัน agent แรกไม่ต้องถาม (เพิ่งรับคำสั่งมา)
        # ถามก่อนรัน agent ที่ 2 เป็นต้นไปเท่านั้น
        if STEP_MODE and completed:
            ans = confirm_next_step(prev_agent, agent, task, i, len(dispatches))
            if ans == "n":
                print(f"\n{YL}  หยุด pipeline ตามที่คุณสั่ง{RST}")
                _stop_pipeline = True
                break
            elif ans == "s":
                print(f"\n{YL}  ข้าม {BLD}{agent.upper()}{RST}")
                continue

        run_agent(agent, task, prev_agent=prev_agent,
                  project_dir=active_project, discover=discover)
        completed.append(agent)
        prev_agent = agent

    if completed:
        print(f"\n{YL}{'═'*55}{RST}")
        last_path = pipeline_read(completed[-1])

        report_sections = []
        for agent in completed:
            out = pipeline_read(agent)
            if not out:
                continue
            p = Path(out)
            if p.suffix == ".md" and p.exists():
                full    = p.read_text(encoding="utf-8")
                header  = full[:1200]
                blocks  = extract_key_blocks(full)
                content = header + ("\n\n--- Key Blocks ---\n" + blocks if blocks and blocks not in header else "")
                report_sections.append(f"=== {agent.upper()} REPORT ===\n{content}")
            else:
                search_dir = p.parent if p.suffix in (".csv", ".py") else p
                summary = read_report_summary(search_dir, agent)
                if summary:
                    report_sections.append(f"=== {agent.upper()} REPORT ===\n{summary}")

        reports_block = "\n\n".join(report_sections)
        # ตรวจสอบ CRISP-DM phase ที่เสร็จแล้ว
        completed_phases = list(dict.fromkeys(
            AGENT_TO_PHASE.get(a, "unknown") for a in completed
        ))
        iter_status = ", ".join(f"{a}×{n}" for a, n in agent_iter_count.items() if n > 1)

        summary_msg = (
            f"Team completed: {', '.join(completed)}\n"
            f"CRISP-DM phases done: {', '.join(completed_phases)}\n"
            + (f"CRISP-DM iterations: {iter_status}\n" if iter_status else "")
            + f"Final output: {last_path}\n\n"
            + (f"--- Agent Reports ---\n{reports_block}\n\n" if reports_block else "")
            + "วิเคราะห์ตาม CRISP-DM process:\n"
            + "1. สรุปผลลัพธ์ให้ผู้ใช้เป็นภาษาไทย โดยอ้างอิงตัวเลขจาก report\n"
            + "2. ถ้า Mo report มี 'Loop Back To Finn: YES' → dispatch finn แล้ว mo ใหม่ทันที\n"
            + "3. ถ้า Quinn พบปัญหา → dispatch agent ที่เกี่ยวข้องใหม่ตาม CRISP-DM\n"
            + "4. ถ้าทุก phase ผ่านแล้ว → บอก user ว่า CRISP-DM cycle เสร็จสมบูรณ์\n"
        )
        summary = call_deepseek(anna_system, summary_msg, label="ANNA summary", history=anna_history)
        anna_history.append({"role": "user",      "content": summary_msg})
        anna_history.append({"role": "assistant", "content": summary})

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
                          project_dir=active_project, discover=discover)
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
            summary = call_deepseek(anna_system, cont_msg, label="ANNA summary", history=anna_history)
            anna_history.append({"role": "user",      "content": cont_msg})
            anna_history.append({"role": "assistant", "content": summary})


# ── Logging ───────────────────────────────────────────────────────────────────

def log_raw(role: str, content: str, task: str = "", output: str = ""):
    """เขียน log ทั้ง global logs/ และ project logs/ พร้อมกัน"""
    ts   = datetime.now().strftime("%H:%M")
    date = datetime.now().strftime("%Y-%m-%d")

    if role.lower() == "user":
        line = f"[{ts}] User: {content[:300]}\n"
    elif role.lower() in ("anna", "anna summary"):
        line = f"[{ts}] Anna | Action: {content[:200]}\n"
    elif role.lower() == "system":
        task_part = f" | {task}" if task else ""
        line = f"[{ts}] [SYS{task_part}] {content[:200]}\n"
    else:
        parts = [f"[{ts}] {role.upper()}"]
        if task:
            parts.append(f"Task: {task[:100]}")
        parts.append(f"Action: {content[:200]}")
        if output:
            parts.append(f"→ {output}")
        line = " | ".join(parts) + "\n"

    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / f"{date}_raw.md", "a", encoding="utf-8") as fp:
        fp.write(line)

    if active_project:
        proj_log_dir = active_project / "logs"
        proj_log_dir.mkdir(exist_ok=True)
        with open(proj_log_dir / f"{date}_raw.md", "a", encoding="utf-8") as fp:
            fp.write(line)


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_help():
    print(f"""
{CY}┌─ คำสั่ง ──────────────────────────────────────────────┐{RST}
{CY}│{RST}  {BLD}{WH}<ข้อความ>{RST}            {YL}»{RST} Anna รับ แล้ว pipeline อัตโนมัติ
{CY}│{RST}  {BLD}{WH}!! <ข้อความ>{RST}          {YL}»{RST} {MG}Claude{RST} discover mode
{CY}│{RST}  {BLD}{WH}@<agent> <task>{RST}       {YL}»{RST} dispatch ตรงไป agent ({BL}DeepSeek{RST})
{CY}│{RST}  {BLD}{WH}@<agent>! <task>{RST}      {YL}»{RST} dispatch ตรงไป agent ({MG}Claude{RST})
{CY}│{RST}  {BLD}{WH}project <name>{RST}        {YL}»{RST} set active project
{CY}│{RST}  {BLD}{WH}kb <agent>{RST}            {YL}»{RST} ดู knowledge base ของ agent
{CY}│{RST}  {BLD}{WH}claude{RST}                {YL}»{RST} ดู {MG}Claude{RST} usage / calls เหลือ
{CY}│{RST}  {BLD}{WH}end session{RST}           {YL}»{RST} ล้าง history + reset Claude calls
{CY}│{RST}  {BLD}{WH}exit{RST}                  {YL}»{RST} ออกจากระบบ
{CY}│{RST}  {DIM}--claude-limit N{RST}      {YL}»{RST} {DIM}ตั้ง limit เมื่อเริ่มโปรแกรม (default 10){RST}
{CY}└──────────────────────────────────────────────────────┘{RST}""")


def anna_discover(user_input: str):
    anna_kb = load_kb("anna")
    system  = ANNA_SYSTEM + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
    result  = call_claude(system, user_input, label="ANNA discover")
    save_kb("anna", f"Task: {user_input}\nDiscovery:\n{result}")


def main():
    global active_project
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    ds_ok = bool(os.environ.get("DEEPSEEK_API_KEY"))
    cl_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # ── Header ────────────────────────────────────────────────
    plain = [
        "DataScienceOS  —  Anna (CEO)",
        f"DeepSeek: {DEEPSEEK_MODEL}  |  Claude: {CLAUDE_MODEL}",
        f"PATH-BASED pipeline v2  |  mode: {MODE}  |  Claude limit: {CLAUDE_LIMIT}",
        "Type  help  for commands",
    ]
    w = max(len(l) for l in plain) + 4   # inner padding 2+2

    def box_row(plain_text: str, colored_text: str) -> str:
        pad = " " * (w - 2 - len(plain_text))
        return f"{CY}│{RST}  {colored_text}{pad}  {CY}│{RST}"

    print(f"{CY}┌{'─'*w}┐{RST}")
    print(box_row(plain[0],
        f"{BLD}{WH}DataScienceOS{RST}  {DIM}—{RST}  {BLD}{YL}Anna (CEO){RST}"))
    print(box_row(plain[1],
        f"{BL}DeepSeek:{RST} {BLD}{DEEPSEEK_MODEL}{RST}  {DIM}|{RST}  {MG}Claude:{RST} {BLD}{CLAUDE_MODEL}{RST}"))
    print(box_row(plain[2],
        f"{DIM}PATH-BASED pipeline v2{RST}  {DIM}|{RST}  mode: {BLD}{MODE}{RST}  {DIM}|{RST}  Claude limit: {MG}{BLD}{CLAUDE_LIMIT}{RST}"))
    print(box_row(plain[3],
        f"Type  {BLD}{WH}help{RST}  for commands"))
    print(f"{CY}└{'─'*w}┘{RST}")
    print()

    ds_str = f"{GR}✓{RST}" if ds_ok else f"{RD}✗ ไม่พบ key{RST}"
    cl_str = f"{GR}✓{RST}" if cl_ok else f"{RD}✗ ไม่พบ key{RST}"
    print(f"  {BL}{BLD}DeepSeek:{RST} {ds_str}    {MG}{BLD}Claude:{RST} {cl_str}  {DIM}(limit: {CLAUDE_LIMIT} calls/session){RST}")
    print()

    # ── ESC monitor (daemon thread) ───────────────────────────
    _mon = threading.Thread(target=_esc_monitor, daemon=True)
    _mon.start()

    # ── Main loop ─────────────────────────────────────────────
    while True:
        try:
            proj = f" {DIM}[{active_project.name}]{RST}" if active_project else ""
            user_input = input(f"{BLD}{WH}คุณ{RST}{proj}{BLD}{WH}:{RST} ").strip()
        except EOFError:
            print(f"\n{YL}  ลาก่อนค่ะ{RST}")
            break
        except KeyboardInterrupt:
            print(f"\n{YL}  ลาก่อนค่ะ{RST}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"{YL}  ลาก่อนค่ะ{RST}")
            break
        if user_input.lower() == "end session":
            anna_history.clear()
            active_project = None
            claude_calls = 0
            agent_iter_count.clear()
            print(f"{YL}  ANNA:{RST} เริ่ม session ใหม่แล้วค่ะ  {DIM}(Claude calls reset → 0/{CLAUDE_LIMIT}){RST}")
            continue
        if user_input.lower() == "help":
            print_help()
            continue
        if user_input.lower().startswith("project "):
            name = user_input[8:].strip()
            p = PROJECTS_DIR / name
            active_project = p if p.exists() else None
            status = f"{GR}{active_project}{RST}" if active_project else f"{RD}ไม่พบ project นี้{RST}"
            print(f"{YL}  ANNA:{RST} Active project → {status}")
            continue
        if user_input.lower().startswith("kb "):
            name = user_input[3:].strip()
            kb = load_kb(name)
            if kb:
                bar = "─" * max(0, 44 - len(name))
                print(f"\n{CY}┌─ KB: {BLD}{name}{RST}{CY} {bar}┐{RST}")
                print(kb)
                print(f"{CY}└{'─'*50}┘{RST}")
            else:
                print(f"{YL}  [{name}]{RST} ยังไม่มี Knowledge Base")
            continue
        if user_input.lower() in ("claude", "claude status"):
            used  = claude_calls
            limit = CLAUDE_LIMIT
            pct   = int(used / limit * 100) if limit > 0 else 100
            bar_len = 20
            filled  = int(bar_len * used / limit) if limit > 0 else bar_len
            bar_color = GR if pct < 60 else (YL if pct < 90 else RD)
            bar = f"{bar_color}{'█' * filled}{DIM}{'░' * (bar_len - filled)}{RST}"
            print(f"\n  {MG}{BLD}Claude usage:{RST}  {bar}  {BLD}{used}/{limit}{RST}  ({pct}%)")
            if used >= limit:
                print(f"  {RD}  ✗ ถึง limit แล้ว — ทุก call จะใช้ DeepSeek แทน{RST}")
            else:
                print(f"  เหลืออีก {BLD}{limit - used}{RST} calls  →  reset ด้วย {BLD}end session{RST}  หรือ {BLD}--claude-limit N{RST}")
            print()
            continue
        if user_input.startswith("!!"):
            anna_discover(user_input[2:].strip())
            continue
        if user_input.startswith("@"):
            parts      = user_input[1:].split(" ", 1)
            agent_part = parts[0].lower()
            task       = parts[1] if len(parts) > 1 else ""
            if not task:
                print(f"{RD}  ใช้งาน:{RST} @{agent_part} <task>")
                continue
            discover   = agent_part.endswith("!")
            agent_name = agent_part.rstrip("!")
            run_agent(agent_name, task, project_dir=active_project, discover=discover)
            continue

        try:
            run_pipeline(user_input)
        except KeyboardInterrupt:
            if _current_proc:
                _current_proc.kill()
            print(f"\n{YL}  หยุด pipeline แล้ว — พร้อมรับคำสั่งใหม่{RST}")
            _stop_requested.clear()


if __name__ == "__main__":
    main()
