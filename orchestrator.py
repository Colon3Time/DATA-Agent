"""
DataScienceOS Orchestrator — File-Based Pipeline
แต่ละ agent เริ่ม session ใหม่ 100% สื่อสารผ่านไฟล์เท่านั้น
ไม่มี context สะสม = ไม่โดน token limit
"""

import os
import re
import json
import sys
from pathlib import Path
from datetime import datetime

import anthropic
from google import genai
from google.genai import types

# โหลด .env ถ้ามี (ไม่บังคับ — ถ้าไม่มี python-dotenv ก็ยังทำงานได้)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# -- Config --------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.0-flash"
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_RETRY    = 2

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
    ANNA_SYSTEM = (BASE_DIR / "anna_short.md").read_text(encoding="utf-8")
else:
    ANNA_SYSTEM = (BASE_DIR / "CLAUDE.md").read_text(encoding="utf-8")


# -- LLM Callers ---------------------------------------------------------------

def call_gemini(system_prompt: str, user_message: str, label: str = "", silent: bool = False) -> str:
    """Execute call — ใช้ Claude แทน Gemini (Gemini quota หมด)"""
    return call_claude(system_prompt, user_message, label=label if not silent else "")


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


# -- Knowledge Base ------------------------------------------------------------

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


# -- File-Based Pipeline -------------------------------------------------------

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


# -- Agent Runner --------------------------------------------------------------

def get_system_prompt(agent_name: str) -> str:
    """โหลด system prompt + KB — fresh ทุกครั้ง"""
    f = AGENTS_DIR / f"{agent_name}.md"
    base = f.read_text(encoding="utf-8") if f.exists() else f"You are {agent_name}, a data science specialist."
    kb = load_kb(agent_name)
    if kb:
        base += f"\n\n---\n## Knowledge Base\n{kb[:1000]}"
    return base


def _extract_error_snippet(text: str) -> str:
    """ดึงแค่บรรทัด error + code block แรกที่เจอ — ไม่ส่ง full text"""
    lines = text.splitlines()
    error_lines, in_block, block = [], False, []
    for line in lines:
        if any(k in line for k in ("Error", "error", "Traceback", "Exception", "line ")):
            error_lines.append(line)
        if line.strip().startswith("```"):
            in_block = not in_block
            block.append(line)
        elif in_block:
            block.append(line)
    snippet = "\n".join(error_lines[:10])
    if block:
        snippet += "\n" + "\n".join(block[:30])
    return snippet[:800] or text[:800]


def run_agent(agent_name: str, task: str, prev_agent: str = "", discover: bool = False) -> str:
    """fresh session ทุกครั้ง — retry สูงสุด MAX_RETRY รอบ ถ้าเกินถาม user"""
    system = get_system_prompt(agent_name)
    context = pipeline_read(prev_agent) if prev_agent else ""
    base_message = f"ผลจาก {prev_agent}:\n{context}\n\nงานของคุณ:\n{task}" if context else task

    print(f"\n{'-'*55}")

    if discover:
        result = call_claude(system, task, label=f"{agent_name.upper()} discover")
        save_kb(agent_name, f"Task: {task}\nDiscovery:\n{result}")
        pipeline_write(agent_name, result)
        log_raw(f"{agent_name}(CLAUDE)", result)
        return result

    message = base_message
    result  = ""
    for attempt in range(MAX_RETRY + 1):
        label = f"{agent_name.upper()}" if attempt == 0 else f"{agent_name.upper()} retry{attempt}"
        result = call_gemini(system, message, label=label)

        need = parse_need_claude(result)
        if not need:
            break  # สำเร็จ — ออกจาก loop

        # มี error — ครั้งต่อไปส่งแค่ snippet ไม่ส่ง full message
        snippet = _extract_error_snippet(result)
        if attempt < MAX_RETRY:
            print(f"\n[{agent_name.upper()}] error รอบ {attempt+1}/{MAX_RETRY} — แก้ใหม่...")
            message = f"งานเดิม: {task}\nError ที่เจอ:\n{snippet}\nแก้เฉพาะจุดนี้"
            log_raw(f"{agent_name}_error{attempt+1}", snippet)
        else:
            # หมด retry — หยุดและบันทึก log ให้ paste ต่อเอง
            print(f"\n[{agent_name.upper()}] แก้ไม่สำเร็จหลัง {MAX_RETRY} รอบ — หยุดรอ user")
            save_resume_log(agent_name, task, snippet, result)
            user_action = input("จะทำอะไรต่อ? (s=skip, r=retry, q=quit): ").strip().lower()
            if user_action == "r":
                message = base_message
                continue
            elif user_action == "q":
                raise SystemExit("หยุดโดย user")
            # s หรืออื่นๆ = skip ไปต่อ

    pipeline_write(agent_name, result)
    log_raw(f"{agent_name}(CLAUDE)", result)
    return result


# -- Dispatch Parser -----------------------------------------------------------

DISPATCH_RE   = re.compile(r'<DISPATCH>(.*?)</DISPATCH>', re.DOTALL)
ASK_USER_RE   = re.compile(r'<ASK_USER>(.*?)</ASK_USER>', re.DOTALL)
ASK_CLAUDE_RE = re.compile(r'<ASK_CLAUDE>(.*?)</ASK_CLAUDE>', re.DOTALL)
NEED_CLAUDE_RE = re.compile(r'NEED_CLAUDE:\s*(.+)', re.IGNORECASE)


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


def parse_ask_claude(text: str) -> str | None:
    m = ASK_CLAUDE_RE.search(text)
    return m.group(1).strip() if m else None


def parse_need_claude(text: str) -> str | None:
    m = NEED_CLAUDE_RE.search(text)
    return m.group(1).strip() if m else None


def strip_tags(text: str) -> str:
    text = DISPATCH_RE.sub("", text)
    text = ASK_USER_RE.sub("", text)
    text = ASK_CLAUDE_RE.sub("", text)
    return text.strip()


# -- Main Pipeline -------------------------------------------------------------

def run_pipeline(user_input: str):
    """
    Flow:
    1. Anna รับงาน → ตัดสินใจ → จบ session Anna
    2. แต่ละ agent เริ่ม session ใหม่ → อ่านไฟล์ → ทำงาน → เขียนไฟล์ → จบ
    3. Anna สรุปผลสุดท้าย → จบ
    """

    # Step 1: Anna ตัดสินใจ (fresh session) — silent เพื่อป้องกัน double print
    anna_kb = load_kb("anna")
    anna_system = ANNA_SYSTEM + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
    print(f"\n{'='*55}")
    anna_response = call_gemini(anna_system, user_input, silent=True)
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

    # ตรวจสอบ ASK_CLAUDE — Anna ต้องการปรึกษา Claude ก่อนวางแผน
    claude_q = parse_ask_claude(anna_response)
    if claude_q:
        print(f"\n[ANNA] ต้องการปรึกษา Claude เรื่อง:\n  {claude_q}")
        perm = input("อนุญาตไหมคะ? (y/n): ").strip().lower()
        if perm == "y":
            print(f"\n{'-'*55}")
            claude_ans = call_claude(ANNA_SYSTEM, claude_q, label="ANNA → CLAUDE")
            save_kb("anna", f"คำถาม: {claude_q}\nClaude แนะนำ:\n{claude_ans}")
            print(f"\n[ANNA] ได้รับ guidance แล้ว กำลังวางแผนใหม่...")
            guided_system  = anna_system + f"\n\n---\n## Claude Guidance\n{claude_ans[:800]}"
            anna_response  = call_gemini(guided_system, user_input, silent=True)
            log_raw("Anna(guided)", anna_response)

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

        print(f"\n[ANNA] OK {agent} เสร็จ ({i+1}/{len(dispatches)})")

    # Step 3: แสดงผลสุดท้าย (โดยไม่ต้องเรียก LLM ซ้ำ)
    if completed:
        print(f"\n{'='*55}")
        last_output = pipeline_read(completed[-1])
        print(f"\n[สรุปผล]:\n{last_output}")

        # บันทึก handoff ทุกครั้งที่ pipeline จบ
        save_handoff(user_input, completed, dispatches)


# -- Logging -------------------------------------------------------------------

def log_raw(role: str, content: str):
    LOGS_DIR.mkdir(exist_ok=True)
    f = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_raw.md"
    ts = datetime.now().strftime("%H:%M")
    with open(f, "a", encoding="utf-8") as fp:
        fp.write(f"[{ts}] {role}: {content[:300]}\n")


def save_resume_log(failed_agent: str, task: str, error_snippet: str, last_output: str):
    """สร้าง resume.md — paste ใน Gemini/Claude browser เพื่อทำงานต่อได้ทันที"""
    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # อ่าน output ของ agent ก่อนหน้าที่สำเร็จแล้ว
    done_summary = ""
    if PIPELINE_DIR.exists():
        for f in sorted(PIPELINE_DIR.glob("*.md")):
            name = f.stem.replace("_output", "")
            preview = f.read_text(encoding="utf-8")[:200].replace("\n", " ")
            done_summary += f"- {name}: {preview}...\n"

    content = f"""# Resume Log — {ts}
paste ไฟล์นี้ใน Gemini/Claude browser เพื่อทำงานต่อ

## Agent ที่ติดปัญหา
{failed_agent} — แก้ไม่สำเร็จหลัง {MAX_RETRY} รอบ

## Task ที่ต้องทำ
{task}

## Error
{error_snippet}

## Output ล่าสุดก่อน error
{last_output[:500]}

## Agent ที่เสร็จแล้ว
{done_summary or "ยังไม่มี"}

## วิธีทำต่อ (paste ข้อความนี้ใน Gemini)
ช่วย {failed_agent} ทำงานต่อ:
งาน: {task}
error ที่เจอ: {error_snippet[:300]}
output files อยู่ที่: DATA-Agent/pipeline/
แก้ error แล้ว return แค่ code ที่แก้ ไม่ต้องอธิบาย
"""
    f = LOGS_DIR / "resume.md"
    f.write_text(content, encoding="utf-8")
    print(f"\n[LOG] resume.md บันทึกแล้ว → {f}")
    print(f"[LOG] paste ไฟล์นี้ใน Gemini browser เพื่อทำงานต่อได้เลย")


# -- Handoff -------------------------------------------------------------------

def save_handoff(user_input: str, completed: list[str], dispatches: list[dict]):
    """บันทึก handoff.md หลัง pipeline จบ — ใช้ resume session ครั้งต่อไป"""
    LOGS_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    date = datetime.now().strftime("%Y-%m-%d")

    # สรุป agent ที่ทำเสร็จพร้อม output สั้นๆ
    agent_summary = ""
    for name in completed:
        output = pipeline_read(name)
        preview = output[:300].replace("\n", " ")
        agent_summary += f"- **{name}**: {preview}...\n"

    # หา agent ที่ยังไม่ได้ทำ (ถ้า pipeline ไม่ครบ)
    all_agents = ["scout","dana","eddie","max","finn","mo","iris","vera","quinn","rex"]
    done_set   = set(completed)
    pending    = [a for a in all_agents if a not in done_set]
    pending_str = ", ".join(pending) if pending else "ครบทุกขั้นตอนแล้ว"

    content = f"""# Handoff — {ts}

## คำสั่งล่าสุดจาก User
{user_input}

## Agent ที่ทำเสร็จแล้ว
{agent_summary}
## ยังไม่ได้ทำ
{pending_str}

## ไฟล์ที่เกี่ยวข้อง
- Pipeline outputs: `DATA-Agent/pipeline/`
- Logs วันนี้: `DATA-Agent/logs/{date}_raw.md`
- Knowledge base: `DATA-Agent/knowledge_base/`

## วิธี Resume
เปิด orchestrator.py แล้วพิมพ์ path ของไฟล์นี้:
`{LOGS_DIR}/handoff.md`
Anna จะอ่านและทำงานต่อได้ทันที
"""
    f = LOGS_DIR / "handoff.md"
    f.write_text(content, encoding="utf-8")
    print(f"\n[HANDOFF] บันทึกแล้ว → {f}")


def load_handoff(path: str) -> str | None:
    """โหลด .md file เป็น context — ถ้า path ไม่มีให้ return None"""
    p = Path(path)
    if not p.exists() or p.suffix != ".md":
        return None
    return p.read_text(encoding="utf-8")


# -- CLI -----------------------------------------------------------------------

HELP_TEXT = """
คำสั่ง:
  <ข้อความ>        → Anna รับ แล้ว pipeline อัตโนมัติ (แต่ละ agent fresh session)
  !! <ข้อความ>     → Anna ใช้ Claude discover
  @<agent> <task>  → ส่งตรงไป agent (Ollama)
  @<agent>! <task> → ส่งตรงไป agent (Claude discover)
  kb <agent>       → ดู knowledge_base
  <path>.md        → โหลด handoff file เพื่อ resume งานจาก session ที่แล้ว
  exit             → ออก
"""


def anna_discover(user_input: str):
    anna_kb = load_kb("anna")
    system  = ANNA_SYSTEM + (f"\n\n---\n## Anna KB\n{anna_kb[:500]}" if anna_kb else "")
    print(f"\n{'='*55}")
    result = call_claude(system, user_input, label="ANNA discover")
    save_kb("anna", f"Task: {user_input}\nDiscovery:\n{result}")


def main():
    mode_label = "LIGHT" if MODE == "light" else "FULL"
    gemini_ok  = bool(os.environ.get("GEMINI_API_KEY"))
    claude_ok  = bool(os.environ.get("ANTHROPIC_API_KEY"))

    print("=" * 55)
    print(f"  DataScienceOS | Mode: {mode_label}")
    print(f"  Gemini : {GEMINI_MODEL}  {'OK' if gemini_ok else 'NO'}")
    print(f"  Claude : {'OK' if claude_ok else 'NO'}")
    print("  แต่ละ agent = fresh session ไม่มี context สะสม")
    print("=" * 55)

    if not gemini_ok:
        print("  !  ไม่พบ GEMINI_API_KEY — เพิ่มใน .env ก่อนนะคะ")
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

        # โหลด .md file เป็น context — resume session จาก handoff
        if user_input.strip().endswith(".md"):
            ctx = load_handoff(user_input.strip())
            if ctx:
                print(f"[RESUME] โหลดไฟล์สำเร็จ — Anna กำลังอ่าน context...")
                user_input = f"[Session Resume — อ่านไฟล์นี้แล้วบอกว่าจะทำอะไรต่อ]\n\n{ctx}"
            else:
                print(f"[ERROR] ไม่พบไฟล์ {user_input}")
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
