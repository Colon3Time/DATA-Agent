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
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import anthropic

load_dotenv(Path(__file__).parent / ".env")

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

ANNA_SYSTEM = (BASE_DIR / ("anna_short.md" if MODE == "light" else "CLAUDE.md")).read_text(encoding="utf-8")

anna_history:   list      = []
active_project: Path|None = None  # project ที่กำลังทำงานอยู่


# ── LLM Callers ───────────────────────────────────────────────────────────────

def call_deepseek(system_prompt: str, user_message: str, label: str = "", history: list|None = None) -> str:
    """DeepSeek API — streaming, OpenAI-compatible"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "[ERROR] DEEPSEEK_API_KEY not found in .env"
    if label:
        print(f"\n[{label}] ", end="", flush=True)
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
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return "[ERROR] DeepSeek connection failed"
    except requests.exceptions.Timeout:
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY not found"
    print(f"\n{'*'*55}\n  [CLAUDE] {label}\n{'*'*55}")
    client = anthropic.Anthropic(api_key=api_key)
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


# ── Knowledge Base ────────────────────────────────────────────────────────────

def load_kb(agent_name: str) -> str:
    f = KNOWLEDGE_DIR / f"{agent_name}_methods.md"
    return f.read_text(encoding="utf-8") if f.exists() else ""

def save_kb(agent_name: str, content: str):
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    f = KNOWLEDGE_DIR / f"{agent_name}_methods.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(f, "a", encoding="utf-8") as fp:
        fp.write(f"\n\n## [{ts}] Discovery\n{content}\n")


# ── Pipeline (PATH-BASED) ─────────────────────────────────────────────────────

def pipeline_write(agent_name: str, file_path: str):
    """บันทึก PATH ของ output file — ไม่ส่ง content เลย"""
    PIPELINE_DIR.mkdir(exist_ok=True)
    (PIPELINE_DIR / f"{agent_name}_path.txt").write_text(str(file_path), encoding="utf-8")

def pipeline_read(agent_name: str) -> str:
    """คืน PATH ของ output file — agent เปิดไฟล์เองตาม path"""
    f = PIPELINE_DIR / f"{agent_name}_path.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""

def pipeline_clear():
    if PIPELINE_DIR.exists():
        for f in PIPELINE_DIR.glob("*_path.txt"):
            f.unlink()


# ── Script Runner ─────────────────────────────────────────────────────────────

def find_agent_script(agent_name: str, project_dir: Path|None) -> Path|None:
    """หา .py script ของ agent จาก project output folder"""
    if not project_dir:
        return None
    agent_dir = project_dir / "output" / agent_name
    if agent_dir.exists():
        scripts = sorted(agent_dir.glob("*.py"), key=lambda x: x.stat().st_mtime)
        if scripts:
            return scripts[-1]  # ใช้ไฟล์ล่าสุด
    return None

def run_script(script_path: Path, input_path: str, output_dir: Path) -> str:
    """
    รัน Python script จริงด้วย subprocess
    ส่ง --input และ --output-dir เป็น argument
    คืน path ของ output ที่สร้าง (CSV ถ้ามี, ไม่งั้นใช้ output_dir)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[SCRIPT] {script_path.name}  input={input_path or 'none'}")

    result = subprocess.run(
        [sys.executable, str(script_path),
         "--input", input_path,
         "--output-dir", str(output_dir)],
        capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    if result.stdout:
        print(result.stdout[-2000:])  # แสดงแค่ 2000 ตัวอักษรล่าสุด
    if result.returncode != 0:
        print(f"[SCRIPT ERROR]\n{result.stderr[:500]}")

    # หา output CSV ล่าสุดที่ script สร้าง
    csvs = sorted(output_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if csvs:
        return str(csvs[0])
    return str(output_dir)


# ── Agent Runner ──────────────────────────────────────────────────────────────

def get_system_prompt(agent_name: str) -> str:
    f = AGENTS_DIR / f"{agent_name}.md"
    base = f.read_text(encoding="utf-8") if f.exists() else f"You are {agent_name}, a data science specialist."
    kb = load_kb(agent_name)
    if kb:
        base += f"\n\n---\n## Knowledge Base\n{kb[:1000]}"
    return base


def read_report_summary(output_dir: Path, agent_name: str, max_chars: int = 800) -> str:
    """อ่าน .md report ล่าสุดจาก output dir คืน content สำหรับใส่ใน log และ Anna context"""
    if not output_dir or not output_dir.exists():
        return ""
    reports = sorted(output_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not reports:
        return ""
    try:
        return reports[0].read_text(encoding="utf-8")[:max_chars]
    except Exception:
        return ""


def run_agent(agent_name: str, task: str, prev_agent: str = "",
              project_dir: Path|None = None, discover: bool = False) -> str:
    """
    Priority 1: ถ้ามี Python script → รัน script จริง (ผลถูกต้อง 100%)
    Priority 2: ถ้าไม่มี → DeepSeek LLM สร้าง report + บันทึก code เป็น .py
    """
    print(f"\n{'─'*55}")

    input_path = pipeline_read(prev_agent) if prev_agent else ""
    output_dir = (project_dir / "output" / agent_name) if project_dir else None

    # ── Priority 1: รัน script จริง ──────────────────────────────
    script = find_agent_script(agent_name, project_dir)
    if script and output_dir and not discover:
        output_path = run_script(script, input_path, output_dir)
        pipeline_write(agent_name, output_path)
        report_summary = read_report_summary(output_dir, agent_name)
        action_msg = f"รัน script {script.name} สำเร็จ"
        if report_summary:
            action_msg += f"\n{report_summary}"
        log_raw(agent_name, action_msg, task=task, output=output_path)
        print(f"[{agent_name.upper()}] Script done → {output_path}")
        return output_path

    # ── Priority 2: LLM ───────────────────────────────────────────
    system = get_system_prompt(agent_name)

    # บอก LLM ว่า input/output path คืออะไร
    path_lines = []
    if input_path:
        path_lines.append(f"Input file path : {input_path}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path_lines.append(f"Save CSV to     : {output_dir / f'{agent_name}_output.csv'}")
        path_lines.append(f"Save script to  : {output_dir / f'{agent_name}_script.py'}")
        path_lines.append(f"Save report to  : {output_dir / f'{agent_name}_report.md'}")

    message = "\n".join(path_lines) + f"\n\nTask: {task}" if path_lines else task

    if discover:
        result = call_claude(system, task, label=f"{agent_name.upper()} discover")
        save_kb(agent_name, f"Task: {task}\nDiscovery:\n{result}")
    else:
        result = call_deepseek(system, message, label=f"{agent_name.upper()} execute")

    # บันทึก output ของ LLM
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # บันทึก report
        report_path = output_dir / f"{agent_name}_report.md"
        report_path.write_text(result, encoding="utf-8")

        # Extract Python code blocks → บันทึกเป็น .py
        code_blocks = re.findall(r'```python\n(.*?)```', result, re.DOTALL)
        if code_blocks:
            py_path = output_dir / f"{agent_name}_script.py"
            py_path.write_text("\n\n".join(code_blocks), encoding="utf-8")
            print(f"[{agent_name.upper()}] Saved script → {py_path}")
            pipeline_write(agent_name, str(py_path))
            log_raw(agent_name, "สร้าง report และ script สำเร็จ (DeepSeek)", task=task, output=str(py_path))
            return str(py_path)

        pipeline_write(agent_name, str(report_path))
        log_raw(agent_name, "สร้าง report สำเร็จ (DeepSeek)", task=task, output=str(report_path))
        print(f"[{agent_name.upper()}] Saved report → {report_path}")
        return str(report_path)

    log_raw(agent_name, result[:200], task=task)
    return result


# ── Dispatch Parser ───────────────────────────────────────────────────────────

DISPATCH_RE = re.compile(r'<DISPATCH>(.*?)</DISPATCH>', re.DOTALL)
ASK_USER_RE = re.compile(r'<ASK_USER>(.*?)</ASK_USER>', re.DOTALL)

def parse_dispatches(text: str) -> list[dict]:
    results = []
    for match in DISPATCH_RE.finditer(text):
        try:
            results.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            pass
    return results

def parse_ask_user(text: str) -> str|None:
    m = ASK_USER_RE.search(text)
    return m.group(1).strip() if m else None


# ── Project Detection ─────────────────────────────────────────────────────────

def detect_project(text: str) -> Path|None:
    """หา project dir จาก text ของ Anna"""
    m = re.search(r'projects[/\\]([\w\-]+)', text)
    if m:
        p = PROJECTS_DIR / m.group(1)
        if p.exists():
            return p
    # ใช้ project ล่าสุดถ้าหาไม่เจอ
    if PROJECTS_DIR.exists():
        projects = sorted([p for p in PROJECTS_DIR.iterdir() if p.is_dir()])
        if projects:
            return projects[-1]
    return None


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(user_input: str):
    global active_project

    anna_kb = load_kb("anna")
    projects_list = "\n".join(
        p.name for p in sorted(PROJECTS_DIR.iterdir()) if p.is_dir()
    ) if PROJECTS_DIR.exists() else ""

    anna_system = (
        ANNA_SYSTEM
        + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
        + (f"\n\n---\n## Available Projects\n{projects_list}" if projects_list else "")
    )

    print(f"\n{'═'*55}")
    anna_response = call_deepseek(anna_system, user_input, label="ANNA", history=anna_history)
    anna_history.append({"role": "user",      "content": user_input})
    anna_history.append({"role": "assistant", "content": anna_response})
    log_raw("User", user_input)
    log_raw("Anna", anna_response, task="รับคำสั่งจาก User และวางแผน dispatch")

    ask = parse_ask_user(anna_response)
    if ask:
        print(f"\n[ANNA] {ask}")
        if input("You (y/n): ").strip().lower() != "y":
            print("[ANNA] Understood. Stopping.")
            return

    dispatches = parse_dispatches(anna_response)
    if not dispatches:
        return

    # Detect active project ถ้ายังไม่ได้ set
    if active_project is None:
        active_project = detect_project(anna_response)

    pipeline_clear()
    proj_name = active_project.name if active_project else "unknown"
    print(f"\n[ANNA] Pipeline: {len(dispatches)} agent(s) | Project: {proj_name}")

    prev_agent = ""
    completed  = []

    for i, d in enumerate(dispatches):
        agent    = d.get("agent", "").lower()
        task     = d.get("task", "")
        discover = d.get("discover", False)
        if not agent or not task:
            continue

        run_agent(agent, task, prev_agent=prev_agent,
                  project_dir=active_project, discover=discover)
        completed.append(agent)
        prev_agent = agent
        print(f"\n[ANNA] ✓ {agent} ({i+1}/{len(dispatches)})")

    # Anna summary — รวม report content จริงให้ Anna อ่านได้
    if completed:
        print(f"\n{'═'*55}")
        last_path = pipeline_read(completed[-1])

        # รวบรวม report content จากทุก agent ที่เสร็จ
        report_sections = []
        for agent in completed:
            out = pipeline_read(agent)
            if not out:
                continue
            p = Path(out)
            # ถ้า output เป็น .md อ่านตรง
            if p.suffix == ".md" and p.exists():
                content = p.read_text(encoding="utf-8")[:800]
                report_sections.append(f"=== {agent.upper()} REPORT ===\n{content}")
            else:
                # ถ้าเป็น .csv หรือ script ให้หา .md ใน output dir เดียวกัน
                search_dir = p.parent if p.suffix in (".csv", ".py") else p
                summary = read_report_summary(search_dir, agent)
                if summary:
                    report_sections.append(f"=== {agent.upper()} REPORT ===\n{summary}")

        reports_block = "\n\n".join(report_sections)
        summary_msg = (
            f"Team completed: {', '.join(completed)}\n"
            f"Final output: {last_path}\n\n"
            + (f"--- Agent Reports ---\n{reports_block}\n\n" if reports_block else "")
            + "สรุปผลลัพธ์ให้ผู้ใช้เป็นภาษาไทย โดยอ้างอิงตัวเลขและข้อมูลจาก report ด้านบนจริงๆ"
        )
        summary = call_deepseek(anna_system, summary_msg, label="ANNA summary", history=anna_history)
        anna_history.append({"role": "user",      "content": summary_msg})
        anna_history.append({"role": "assistant", "content": summary})


# ── Logging ───────────────────────────────────────────────────────────────────

def log_raw(role: str, content: str, task: str = "", output: str = ""):
    """เขียน log ทั้ง global logs/ และ project logs/ พร้อมกัน"""
    ts   = datetime.now().strftime("%H:%M")
    date = datetime.now().strftime("%Y-%m-%d")

    # สร้าง log line ตาม format ใน CLAUDE.md
    if role.lower() == "user":
        line = f"[{ts}] User: {content[:300]}\n"
    elif role.lower() in ("anna", "anna summary"):
        line = f"[{ts}] Agent: Anna | Action: {content[:200]}\n"
    else:
        # agent log: Agent: {ชื่อ} | Task: {งาน} | Action: {สิ่งที่ทำ} | Output: {ไฟล์}
        parts = [f"[{ts}] Agent: {role}"]
        if task:
            parts.append(f"Task: {task[:100]}")
        parts.append(f"Action: {content[:200]}")
        if output:
            parts.append(f"Output: {output}")
        line = " | ".join(parts) + "\n"

    # เขียน global log
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / f"{date}_raw.md", "a", encoding="utf-8") as fp:
        fp.write(line)

    # เขียน project log (ถ้ามี active project)
    if active_project:
        proj_log_dir = active_project / "logs"
        proj_log_dir.mkdir(exist_ok=True)
        with open(proj_log_dir / f"{date}_raw.md", "a", encoding="utf-8") as fp:
            fp.write(line)


# ── CLI ───────────────────────────────────────────────────────────────────────

HELP_TEXT = """
คำสั่ง:
  <ข้อความ>          → Anna รับ แล้ว pipeline อัตโนมัติ
  !! <ข้อความ>       → Anna ใช้ Claude discover
  @<agent> <task>    → ส่งตรงไป agent (Ollama)
  @<agent>! <task>   → ส่งตรงไป agent (Claude discover)
  project <name>     → set active project
  kb <agent>         → ดู knowledge_base
  exit               → ออก
"""


def anna_discover(user_input: str):
    anna_kb = load_kb("anna")
    system  = ANNA_SYSTEM + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
    result  = call_claude(system, user_input, label="ANNA discover")
    save_kb("anna", f"Task: {user_input}\nDiscovery:\n{result}")


def main():
    global active_project
    print("=" * 55)
    print("  DataScienceOS  |  PATH-BASED pipeline v2")
    print(f"  DeepSeek: {DEEPSEEK_MODEL}  |  Claude: {CLAUDE_MODEL}")
    print("  Agent ที่มี script → รันจริง | ไม่มี → LLM")
    print("=" * 55)

    print("  DeepSeek:", "✓" if os.environ.get("DEEPSEEK_API_KEY") else "✗ ไม่พบ DEEPSEEK_API_KEY")
    print("  Claude  :", "✓" if os.environ.get("ANTHROPIC_API_KEY") else "✗ ไม่พบ API key")
    print()

    while True:
        try:
            user_input = input("คุณ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nออกจากระบบ")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "end session":
            anna_history.clear()
            active_project = None
            print("[ANNA] Session ended.")
            continue
        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue
        if user_input.lower().startswith("project "):
            name = user_input[8:].strip()
            p = PROJECTS_DIR / name
            active_project = p if p.exists() else None
            print(f"[ANNA] Active project: {active_project or 'not found'}")
            continue
        if user_input.lower().startswith("kb "):
            name = user_input[3:].strip()
            kb = load_kb(name)
            print(kb if kb else f"[{name}] ยังไม่มี KB")
            continue
        if user_input.startswith("!!"):
            anna_discover(user_input[2:].strip())
            continue
        if user_input.startswith("@"):
            parts      = user_input[1:].split(" ", 1)
            agent_part = parts[0].lower()
            task       = parts[1] if len(parts) > 1 else ""
            if not task:
                print(f"ใช้งาน: @{agent_part} <task>")
                continue
            discover   = agent_part.endswith("!")
            agent_name = agent_part.rstrip("!")
            run_agent(agent_name, task, project_dir=active_project, discover=discover)
            continue

        run_pipeline(user_input)


if __name__ == "__main__":
    main()
