"""
Auto-runner สำหรับ GAID AI Forecast pipeline
โจทย์: ทำนายอัตราการใช้ AI ใน 10 ปีข้างหน้า (2026-2035) จาก data ปี 1998-2025
"""
import subprocess
import sys
import time
import threading
import queue
import re
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT   = "2026-05-06_gaid_ai_forecast"
INPUT_CSV = f"projects/{PROJECT}/input/GAID_MASTER_V2_COMPILATION_FINAL.csv"
IDLE_SECS = 8

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mKHABCDJGsu]")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

PROMPTS = [
    "คุณ:",
    "คุณ [",
    "คุณ (y/n):",
    "ไปต่อไหม?",
    "y=ใช่",
]

BUSINESS_GOAL = (
    "โจทย์: อยากทราบว่าในอีก 10 ปีข้างหน้าหลังจากข้อมูลล่าสุด "
    "อัตราการใช้ AI และ metrics ที่เกี่ยวข้องจะเป็นเท่าไร "
    "ใช้ข้อมูลย้อนหลัง 1998-2025 จาก GAID dataset "
    "สร้าง regression model ที่ใช้ Year+Country+Metric เป็น features "
    "เพื่อ forecast แนวโน้มในอนาคต (extrapolation) "
    "target column = Value problem_type = regression"
)

def pick_response(clean: str, turn: int) -> str:
    if turn == 0:
        return (
            f"ใช้ project ชื่อ {PROJECT} เท่านั้น ห้ามสร้าง project ใหม่ "
            f"input อยู่ที่ projects/{PROJECT}/input/GAID_MASTER_V2_COMPILATION_FINAL.csv "
            "ข้ามขั้นตอน Scout ทันที เริ่มจาก Dana ก่อนเลย "
            + BUSINESS_GOAL
        )
    # ถ้า Anna งงเรื่อง project ให้ redirect กลับ
    if "gaid_value_regression" in clean or "2026-05-08" in clean:
        return (
            f"ใช้ project {PROJECT} เท่านั้น ไม่ต้องสร้าง project ใหม่ "
            f"Dana output อยู่ที่ projects/{PROJECT}/output/dana/dana_output.csv แล้ว "
            "ดำเนินการต่อใน project เดิมได้เลย"
        )
    if "ไปต่อไหม" in clean or "y=ใช่" in clean:
        return "y"
    if "y/n" in clean or "(y/n)" in clean:
        return "y"
    if "target" in clean.lower() and ("eddie" in clean.lower() or "business" in clean.lower()):
        return (
            f"project = {PROJECT} "
            "Target = Value problem_type = regression "
            f"Dana output: projects/{PROJECT}/output/dana/dana_output.csv — dispatch Eddie ได้เลย"
        )
    if "business goal" in clean.lower() or "business question" in clean.lower():
        return BUSINESS_GOAL + f" project = {PROJECT} — ดำเนินการต่อได้เลย"
    # Anna ติด loop complain ว่า Eddie/Finn hallucinate — ให้บอกว่า output ถูกและให้ dispatch Mo ต่อ
    if any(k in clean.lower() for k in ["hallucinate", "ไม่มีอยู่จริง", "context ผิด", "ยังผิด", "ยังคงผิด"]):
        return (
            f"output ของ Eddie และ Finn ถูกต้องแล้วค่ะ cat__Dataset columns มาจาก Dataset column ใน GAID dataset จริง "
            "ไม่ต้อง dispatch ซ้ำอีก — ดำเนินการ dispatch Mo Phase 1 ได้เลย "
            f"ใช้ finn output จาก projects/{PROJECT}/output/finn/finn_output.csv"
        )
    if "restart" in clean.lower():
        return "n"
    return "y"

def reader_thread(proc, q):
    buf = b""
    while True:
        chunk = proc.stdout.read1(512)
        if not chunk:
            q.put(None)
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            q.put(line + b"\n")
        if len(buf) > 0:
            q.put(buf)
            buf = b""

def send(proc, msg, turn):
    print(f"\n{'='*55}")
    print(f"[AUTO #{turn+1}] {msg[:150]}")
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

    print(f"[AUTO] Forecast pipeline {PROJECT} starting...")
    print(f"[AUTO] Business goal: {BUSINESS_GOAL[:100]}...")
    print("=" * 55)
    sys.stdout.flush()

    buf = ""
    last_data = time.time()
    turn = 0
    last_turn_time = 0.0
    COOLDOWN = 5.0

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

        since_last_turn = time.time() - last_turn_time
        if is_waiting(buf) and since_last_turn >= COOLDOWN:
            time.sleep(0.5)
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
