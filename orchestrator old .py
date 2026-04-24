"""
DataScienceOS Orchestrator — File-Based Pipeline
แต่ละ agent เริ่ม session ใหม่ 100% สื่อสารผ่านไฟล์เท่านั้น
ไม่มี context สะสม = ไม่โดน token limit
"""

import os
import re
import json
import requests
import sys
from pathlib import Path
from datetime import datetime

import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
CLAUDE_MODEL = "claude-sonnet-4-6"

BASE_DIR      = Path(__file__).parent
AGENTS_DIR    = BASE_DIR / "agents"
LOGS_DIR      = BASE_DIR / "logs"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
PIPELINE_DIR  = BASE_DIR / "pipeline"  # ไฟล์สื่อสารระหว่าง agent

MODE = "light"
if "--mode" in sys.argv:
    idx = sys.argv.index("--mode")
    if idx + 1 < len(sys.argv):
        MODE = sys.argv[idx + 1]

if MODE == "light":
    OLLAMA_MODEL = " llama3:latest"
    ANNA_SYSTEM  = (BASE_DIR / "anna_short.md").read_text(encoding="utf-8")
else:
    OLLAMA_MODEL = "llama3:latest"
    ANNA_SYSTEM  = (BASE_DIR / "CLAUDE.md").read_text(encoding="utf-8")


# ── LLM Callers ───────────────────────────────────────────────────────────────

def call_ollama(system_prompt: str, user_message: str, label: str = "") -> str:
    """Fresh Ollama call — ไม่มี history ทุกครั้ง"""
    if label:
        print(f"\n[{label}] ", end="", flush=True)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "stream": True,
            },
            stream=True,
            timeout=180,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama ไม่ได้รันอยู่"
    except requests.exceptions.Timeout:
        return "[ERROR] Ollama timeout"

    full = []
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        print(token, end="", flush=True)
        full.append(token)
        if chunk.get("done"):
            break
    print()
    return "".join(full)


def call_claude(system_prompt: str, user_message: str, label: str = "") -> str:
    """Fresh Claude call — discover only"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ไม่พบ ANTHROPIC_API_KEY"
    if label:
        print(f"\n[{label} → CLAUDE] ", end="", flush=True)
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=4096,
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
    print(f"\n[KB] บันทึกลง {f.name}")


# ── File-Based Pipeline ───────────────────────────────────────────────────────

def pipeline_write(agent_name: str, content: str):
    """Agent เขียนผลลงไฟล์ เพื่อให้ agent ถัดไปอ่าน"""
    PIPELINE_DIR.mkdir(exist_ok=True)
    f = PIPELINE_DIR / f"{agent_name}_output.md"
    f.write_text(content, encoding="utf-8")


def pipeline_read(agent_name: str) -> str:
    """Agent อ่านผลจาก agent ก่อนหน้า — แค่ 500 ตัวอักษรแรก ประหยัด token"""
    f = PIPELINE_DIR / f"{agent_name}_output.md"
    if not f.exists():
        return ""
    content = f.read_text(encoding="utf-8")
    return content[:500] + "\n...(ดูไฟล์เต็มได้ที่ pipeline/)" if len(content) > 500 else content


def pipeline_clear():
    """เคลียร์ pipeline files ก่อนเริ่ม project ใหม่"""
    if PIPELINE_DIR.exists():
        for f in PIPELINE_DIR.glob("*.md"):
            f.unlink()


# ── Agent Runner ──────────────────────────────────────────────────────────────

def get_system_prompt(agent_name: str) -> str:
    """โหลด system prompt + KB — fresh ทุกครั้ง"""
    f = AGENTS_DIR / f"{agent_name}.md"
    base = f.read_text(encoding="utf-8") if f.exists() else f"You are {agent_name}, a data science specialist."
    kb = load_kb(agent_name)
    if kb:
        base += f"\n\n---\n## Knowledge Base\n{kb[:1000]}"
    return base


def run_agent(agent_name: str, task: str, prev_agent: str = "", discover: bool = False) -> str:
    """
    รัน agent 1 ครั้ง — fresh session ทุกครั้ง
    อ่าน input จากไฟล์ เขียน output ลงไฟล์
    """
    system = get_system_prompt(agent_name)

    # อ่านผลจาก agent ก่อนหน้าผ่านไฟล์ (ไม่ผ่าน memory)
    context = pipeline_read(prev_agent) if prev_agent else ""
    message = f"ผลจาก {prev_agent}:\n{context}\n\nงานของคุณ:\n{task}" if context else task

    print(f"\n{'─'*55}")

    if discover:
        result = call_claude(system, task, label=f"{agent_name.upper()} discover")
        save_kb(agent_name, f"Task: {task}\nDiscovery:\n{result}")
    else:
        result = call_ollama(system, message, label=f"{agent_name.upper()} execute")

    # เขียนผลลงไฟล์เพื่อส่งต่อ
    pipeline_write(agent_name, result)
    log_raw(f"{agent_name}({'CLAUDE' if discover else 'OLLAMA'})", result)
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


def parse_ask_user(text: str) -> str | None:
    m = ASK_USER_RE.search(text)
    return m.group(1).strip() if m else None


def strip_tags(text: str) -> str:
    text = DISPATCH_RE.sub("", text)
    text = ASK_USER_RE.sub("", text)
    return text.strip()


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(user_input: str):
    """
    Flow:
    1. Anna รับงาน → ตัดสินใจ → จบ session Anna
    2. แต่ละ agent เริ่ม session ใหม่ → อ่านไฟล์ → ทำงาน → เขียนไฟล์ → จบ
    3. Anna สรุปผลสุดท้าย → จบ
    """

    # Step 1: Anna ตัดสินใจ (fresh session)
    anna_kb = load_kb("anna")
    anna_system = ANNA_SYSTEM + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
    print(f"\n{'═'*55}")
    anna_response = call_ollama(anna_system, user_input, label="ANNA")
    log_raw("User", user_input)
    log_raw("Anna", anna_response)

    # ตรวจสอบ ASK_USER
    ask = parse_ask_user(anna_response)
    if ask:
        print(f"\n[ANNA] {ask}")
        confirm = input("คุณ (y/n): ").strip().lower()
        if confirm != "y":
            print("[ANNA] รับทราบ หยุดการทำงาน")
            return

    dispatches = parse_dispatches(anna_response)
    clean_response = strip_tags(anna_response)
    if clean_response:
        print(f"\n[ANNA] {clean_response}")

    if not dispatches:
        return

    # Step 2: รัน agent ทีละตัว — แต่ละตัว fresh session
    print(f"\n[ANNA] pipeline เริ่ม {len(dispatches)} agent(s)...")
    pipeline_clear()  # เคลียร์ผลเก่า

    prev_agent = ""
    completed  = []

    for i, d in enumerate(dispatches):
        agent    = d.get("agent", "").lower()
        task     = d.get("task", "")
        discover = d.get("discover", False)

        if not agent or not task:
            continue

        # fresh session ทุกครั้ง — อ่านจากไฟล์เท่านั้น
        run_agent(agent, task, prev_agent=prev_agent, discover=discover)
        completed.append(agent)
        prev_agent = agent

        print(f"\n[ANNA] ✓ {agent} เสร็จ ({i+1}/{len(dispatches)})")

    # Step 3: Anna สรุปผลสุดท้าย (fresh session — อ่านจากไฟล์)
    if completed:
        print(f"\n{'═'*55}")
        # อ่านแค่ผลของ agent สุดท้าย ไม่อ่านทั้งหมด
        last_output = pipeline_read(completed[-1])
        summary_msg = f"ทีมทำงานเสร็จแล้ว {len(completed)} agent: {', '.join(completed)}\n\nผลสุดท้าย:\n{last_output}\n\nสรุปให้ผู้ใช้สั้นๆ"
        call_ollama(anna_system, summary_msg, label="ANNA สรุป")


# ── Logging ───────────────────────────────────────────────────────────────────

def log_raw(role: str, content: str):
    LOGS_DIR.mkdir(exist_ok=True)
    f = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_raw.md"
    ts = datetime.now().strftime("%H:%M")
    with open(f, "a", encoding="utf-8") as fp:
        fp.write(f"[{ts}] {role}: {content[:300]}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

HELP_TEXT = """
คำสั่ง:
  <ข้อความ>        → Anna รับ แล้ว pipeline อัตโนมัติ (แต่ละ agent fresh session)
  !! <ข้อความ>     → Anna ใช้ Claude discover
  @<agent> <task>  → ส่งตรงไป agent (Ollama)
  @<agent>! <task> → ส่งตรงไป agent (Claude discover)
  kb <agent>       → ดู knowledge_base
  exit             → ออก
"""


def anna_discover(user_input: str):
    anna_kb = load_kb("anna")
    system  = ANNA_SYSTEM + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
    print(f"\n{'═'*55}")
    result = call_claude(system, user_input, label="ANNA discover")
    save_kb("anna", f"Task: {user_input}\nDiscovery:\n{result}")


def main():
    print("=" * 55)
    print("  DataScienceOS  |  D:\\DataScienceOS")
    mode_label = "LIGHT (Surface)" if MODE == "light" else "FULL (คอม)"
    print(f"  Mode   : {mode_label}")
    print(f"  Ollama : {OLLAMA_MODEL}  |  Claude: {CLAUDE_MODEL}")
    print("  แต่ละ agent = fresh session ไม่มี context สะสม")
    print("=" * 55)

    try:
        requests.get("http://localhost:11434", timeout=3)
        print("  Ollama : ✓ พร้อมใช้งาน")
    except requests.exceptions.ConnectionError:
        print("  Ollama : ✗ ไม่ได้รัน  →  รัน `ollama serve` ก่อน")

    print("  Claude : ✓" if os.environ.get("ANTHROPIC_API_KEY") else "  Claude : ✗ ไม่พบ API key")
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
        if user_input.lower() == "help":
            print(HELP_TEXT)
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
            run_agent(agent_name, task, discover=discover)
            continue

        run_pipeline(user_input)


if __name__ == "__main__":
    main()
