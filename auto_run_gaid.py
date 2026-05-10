"""
Auto-runner for GAID pipeline — Claude ตอบแทน user เพื่อ flow ไม่สะดุด
Fixed: prompt detection, encoding, idle timing
"""
import subprocess
import sys
import time
import threading
import queue
import re
import io
from pathlib import Path

# Force UTF-8 on this script's output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT    = "2026-05-05_gaid_master_v2"
INPUT_CSV  = f"projects/{PROJECT}/input/GAID_MASTER_V2_COMPILATION_FINAL.csv"
IDLE_SECS  = 8   # วินาทีที่ไม่มี output ถือว่า orchestrator รอ input

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mKHABCDJGsu]")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

# Prompt signatures (หลัง strip ANSI แล้ว)
PROMPTS = [
    "คุณ:",          # main prompt (no project)
    "คุณ [",         # main prompt with project name e.g. "คุณ [2026-05-05_gaid]:"
    "คุณ (y/n):",   # yes/no from ASK_USER
    "ไปต่อไหม?",    # step confirm
    "y=ใช่",         # step confirm (ส่วนหนึ่ง)
]

# Decision tree: ดู text สะสมแล้วเลือก response
def pick_response(clean: str, turn: int) -> str:
    if turn == 0:
        return (
            f"รัน pipeline ใหม่ project {PROJECT} "
            f"ใช้ input/{INPUT_CSV.split('/')[-1]} ที่มีอยู่แล้ว "
            "ไม่ต้อง Scout — เริ่มจาก Dana ทันที "
            "Business goal: วิเคราะห์แนวโน้ม AI development ของแต่ละประเทศ "
            "ผ่าน metrics เช่น publications, patents, investment, talent "
            "target column = Value (ค่า metric) — dataset เป็น long-format AI Index"
        )
    # step confirm
    if "ไปต่อไหม" in clean or "y=ใช่" in clean:
        return "y"
    # yes/no ASK_USER
    if "y/n" in clean or "(y/n)" in clean:
        return "y"
    # Eddie ถาม target
    if "target" in clean.lower() and ("eddie" in clean.lower() or "business goal" in clean.lower()):
        return (
            "Target column คือ Value (numeric metric value) "
            "problem_type = regression (ทำนาย Value จาก Country+Year+Metric) "
            "ดำเนินการต่อได้เลย"
        )
    # Anna ถาม business goal
    if "business goal" in clean.lower() or "business question" in clean.lower():
        return (
            "Business goal: วิเคราะห์และ forecast Value ของ AI metrics แต่ละประเทศตามปี "
            "target = Value, problem_type = regression — ดำเนินการต่อได้เลย"
        )
    # ถาม restart
    if "restart" in clean.lower():
        return "n"
    # default approve
    return "y"

def reader_thread(proc, q):
    """อ่าน stdout ต่อเนื่องเพื่อไม่ให้ pipe เต็ม"""
    buf = b""
    while True:
        # read1() คืนทันทีเท่าที่มีข้อมูล ไม่รอจนครบ n bytes
        # แก้ deadlock กรณี prompt ไม่มี newline เช่น "คุณ: "
        chunk = proc.stdout.read1(512)
        if not chunk:
            q.put(None)
            break
        buf += chunk
        # ส่งทีละบรรทัด (หรือ chunk ถ้าไม่มี newline)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            q.put(line + b"\n")
        # flush ถ้าสะสมนานเกิน (เช่น prompt ไม่มี newline)
        if len(buf) > 0:
            q.put(buf)
            buf = b""

def send(proc, msg, turn):
    print(f"\n{'='*55}")
    print(f"[AUTO #{turn+1}] {msg[:120]}")
    print(f"{'='*55}\n")
    sys.stdout.flush()
    proc.stdin.write((msg + "\n").encode("utf-8"))
    proc.stdin.flush()

def is_waiting(accumulated: str) -> bool:
    clean = strip_ansi(accumulated)
    return any(p in clean for p in PROMPTS)

def main():
    env = {**__import__("os").environ,
           "PYTHONIOENCODING": "utf-8",
           "PYTHONUTF8": "1",
           "PYTHONUNBUFFERED": "1"}

    proc = subprocess.Popen(
        [sys.executable, "-u", "orchestrator_v3.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent),
        env=env,
    )

    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=reader_thread, args=(proc, q), daemon=True)
    t.start()

    print(f"[AUTO] Pipeline {PROJECT} starting...\n{'='*55}")
    sys.stdout.flush()

    buf = ""
    last_data = time.time()
    turn = 0
    last_turn_time = 0.0
    COOLDOWN = 5.0  # วินาที min ระหว่าง 2 response

    while True:
        try:
            item = q.get(timeout=0.25)
        except queue.Empty:
            idle = time.time() - last_data
            since_last_turn = time.time() - last_turn_time
            if (idle >= IDLE_SECS
                    and buf.strip()
                    and is_waiting(buf)
                    and since_last_turn >= COOLDOWN):
                msg = pick_response(strip_ansi(buf), turn)
                send(proc, msg, turn)
                turn += 1
                last_turn_time = time.time()
                buf = ""
                last_data = time.time()
            continue

        if item is None:
            break

        text = item.decode("utf-8", errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        buf += text
        last_data = time.time()

        # ตรวจ prompt ทันทีเมื่อมีข้อมูลใหม่ (responsive heuristic)
        since_last_turn = time.time() - last_turn_time
        if is_waiting(buf) and since_last_turn >= COOLDOWN:
            time.sleep(0.5)   # รอให้ orchestrator flush output จบก่อน
            msg = pick_response(strip_ansi(buf), turn)
            send(proc, msg, turn)
            turn += 1
            last_turn_time = time.time()
            buf = ""
            last_data = time.time()

    proc.wait()
    print(f"\n[AUTO] Done — exit code {proc.returncode}")

if __name__ == "__main__":
    main()
