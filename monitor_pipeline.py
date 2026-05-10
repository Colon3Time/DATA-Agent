"""Monitor auto_run_gaid.py output — reports every 30s, alerts on error immediately."""
import sys
import time
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTFILE = Path(r"C:\Users\Amorntep\AppData\Local\Temp\claude\C--Users-Amorntep-DATA-Agent\cb4bf5aa-4b00-4053-bb38-a0bbb782772e\tasks\bu74rub32.output")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

def clean(line):
    return ANSI.sub("", line).strip()

last_line = 0
last_report = time.time()

while True:
    time.sleep(2)
    if not OUTFILE.exists():
        continue
    lines = OUTFILE.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        continue

    new_lines = lines[last_line:]
    for line in new_lines:
        c = clean(line)
        if re.search(r"GATE FAIL|Traceback|SystemExit|exit code [1-9]|SCRIPT ERROR", c, re.IGNORECASE):
            print(f"[ERROR] {c}", flush=True)

    last_line = len(lines)

    now = time.time()
    if now - last_report >= 30:
        tail = [clean(l) for l in lines[-3:] if clean(l)]
        print("[30s] " + " | ".join(tail), flush=True)
        last_report = now

    last = clean(lines[-1])
    if re.search(r"\[AUTO\] Done|exit code 0|exit code \d", last):
        print(f"[DONE] Pipeline จบ: {last}", flush=True)
        break
