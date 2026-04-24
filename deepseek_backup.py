"""
DeepSeek Direct Chat — ไม่มี Anna ไม่มี pipeline
ใช้เมื่อ Claude token หมด หรือต้องการคุยกับ DeepSeek เปล่าๆ
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

load_dotenv(Path(__file__).parent / ".env")

DEEPSEEK_URL   = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

history     = []
input_history = InMemoryHistory()

prompt_style = Style.from_dict({"prompt": "ansicyan bold"})


def chat(user_message: str, system: str = "") -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "[ERROR] ไม่พบ DEEPSEEK_API_KEY ใน .env"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
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
        return "[ERROR] เชื่อมต่อ DeepSeek ไม่ได้"
    except requests.exceptions.Timeout:
        return "[ERROR] DeepSeek timeout"

    full = []
    print("\nDeepSeek: ", end="", flush=True)
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
    print("\n")
    return "".join(full)


def save_log(role: str, content: str):
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    f = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}_deepseek_direct.md"
    ts = datetime.now().strftime("%H:%M")
    with open(f, "a", encoding="utf-8") as fp:
        fp.write(f"[{ts}] {role}: {content[:300]}\n")


HELP = """
คำสั่ง:
  /clear     → ล้าง history เริ่มใหม่
  /history   → ดู history ปัจจุบัน
  /system    → set system prompt ใหม่
  /exit      → ออก
  ↑↓         → เลื่อนดู history ที่พิมพ์ไปแล้ว
"""


def main():
    system_prompt = ""

    print("=" * 50)
    print("  DeepSeek Direct Chat")
    print(f"  Model: {DEEPSEEK_MODEL}")
    print("  พิมพ์ /help สำหรับคำสั่ง")
    print("=" * 50)

    api_ok = bool(os.environ.get("DEEPSEEK_API_KEY"))
    print(f"  DeepSeek API: {'✓ พร้อมใช้' if api_ok else '✗ ไม่พบ DEEPSEEK_API_KEY'}\n")

    while True:
        try:
            user_input = prompt(
                "คุณ: ",
                history=input_history,
                style=prompt_style,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nออก")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break
        if user_input == "/help":
            print(HELP)
            continue
        if user_input == "/clear":
            history.clear()
            print("[ล้าง history แล้ว]\n")
            continue
        if user_input == "/history":
            if not history:
                print("[ยังไม่มี history]\n")
            else:
                for h in history:
                    print(f"  [{h['role']}] {h['content'][:100]}")
            print()
            continue
        if user_input.startswith("/system"):
            system_prompt = user_input[7:].strip()
            print(f"[System prompt: {system_prompt[:80] or '(ล้างแล้ว)'}]\n")
            continue

        save_log("User", user_input)
        result = chat(user_input, system=system_prompt)
        save_log("DeepSeek", result)

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()
